const std = @import("std");
const types = @import("types.zig");
const prng = @import("prng.zig");
const linkage = @import("linkage.zig");
const crossover = @import("crossover.zig");
const dedup = @import("dedup.zig");
const pool = @import("pool.zig");

pub const EngineOptions = struct {
    enable_dedup: bool = false,
    num_threads: usize = 1,
};

pub fn GeneticEngine(comptime L: usize, comptime Adapter: type) type {
    return GeneticEngineCustom(L, Adapter, .{ .enable_dedup = false, .num_threads = 1 });
}

pub fn GeneticEngineCustom(comptime L: usize, comptime Adapter: type, comptime options: EngineOptions) type {
    const Chrom = types.Chromosome(L);
    const Pop = types.Population(L);
    const LinkEngine = linkage.LinkageEngine(L);
    const CrossEngine = crossover.CrossoverEngine(L);
    const DedupTable = dedup.L1Dedup(8192);
    const words_len = (L + 63) / 64;
    const K = LinkEngine.elite_size;

    return struct {
        const Self = @This();
        pub const engine_options = options;
        pub const elite_size: usize = K;

        allocator: std.mem.Allocator,
        pop_current: Pop,
        pop_next: Pop,
        linkage_engine: LinkEngine,
        dedup_cache: DedupTable,
        rng: prng.Xoshiro256,
        adapter: Adapter,
        generation: usize,
        elite_buffer: [K]Chrom,
        crossover_mask: [words_len]u64,
        mutation_mask: [words_len]u64,
        entropy_floor: f64,
        thread_pool: ?*pool.StaticThreadPool,

        pub fn init(
            allocator: std.mem.Allocator,
            pop_size: usize,
            adapter: Adapter,
            seed: u64,
        ) !Self {
            std.debug.assert(pop_size >= K);

            var p_ptr: ?*pool.StaticThreadPool = null;
            if (options.num_threads > 1) {
                const p = try allocator.create(pool.StaticThreadPool);
                p.* = .{};
                try p.init(options.num_threads);
                p_ptr = p;
            }

            var engine = Self{
                .allocator = allocator,
                .pop_current = try Pop.init(allocator, pop_size),
                .pop_next = try Pop.init(allocator, pop_size),
                .linkage_engine = try LinkEngine.init(allocator),
                .dedup_cache = DedupTable.init(),
                .rng = prng.Xoshiro256.init(seed),
                .adapter = adapter,
                .generation = 0,
                .elite_buffer = undefined,
                .crossover_mask = undefined,
                .mutation_mask = undefined,
                .entropy_floor = 0.15,
                .thread_pool = p_ptr,
            };

            for (engine.pop_current.individuals) |*indiv| {
                engine.rng.fillWords(&indiv.words);
                indiv.cleanPadding();
                engine.evaluateIndividual(indiv);
            }
            engine.pop_current.updateMetrics();
            return engine;
        }

        pub fn deinit(self: *Self) void {
            if (self.thread_pool) |p| {
                p.deinit();
                self.allocator.destroy(p);
                self.thread_pool = null;
            }
            self.pop_current.deinit(self.allocator);
            self.pop_next.deinit(self.allocator);
            self.linkage_engine.deinit(self.allocator);
        }

        pub fn evaluateIndividual(self: *Self, chrom: *Chrom) void {
            if (options.enable_dedup) {
                const h = DedupTable.hashChromosome(&chrom.words);
                chrom.hash = h;

                if (self.dedup_cache.lookup(h)) |cached_fit| {
                    chrom.fitness = cached_fit;
                    return;
                }
                const phenotype = self.adapter.decode(chrom);
                const score = self.adapter.evaluate(phenotype);
                chrom.fitness = score;
                self.dedup_cache.insert(h, score);
            } else {
                const phenotype = self.adapter.decode(chrom);
                chrom.fitness = self.adapter.evaluate(phenotype);
            }
        }

        pub fn extractElites(self: *Self) void {
            const inds = self.pop_current.individuals;
            var rank_indices: [K]usize = undefined;

            for (0..K) |i| rank_indices[i] = i;

            var min_pos: usize = 0;
            var min_val: f64 = inds[rank_indices[0]].fitness;
            for (1..K) |j| {
                const candidate_f = inds[rank_indices[j]].fitness;
                if (candidate_f < min_val) {
                    min_val = candidate_f;
                    min_pos = j;
                }
            }

            for (K..inds.len) |i| {
                const f = inds[i].fitness;
                if (f > min_val) {
                    rank_indices[min_pos] = i;
                    min_pos = 0;
                    min_val = inds[rank_indices[0]].fitness;
                    for (1..K) |j| {
                        const candidate_f = inds[rank_indices[j]].fitness;
                        if (candidate_f < min_val) {
                            min_val = candidate_f;
                            min_pos = j;
                        }
                    }
                }
            }

            for (rank_indices, 0..) |src_idx, dst_idx| {
                self.elite_buffer[dst_idx] = inds[src_idx];
            }
        }

        pub fn step(self: *Self) !void {
            self.extractElites();
            self.linkage_engine.ingestElites(&self.elite_buffer);

            if (self.generation == 0 or self.generation % 4 == 0) {
                if (self.generation > 0 and self.generation % 8 == 0) {
                    self.linkage_engine.shufflePermutation(&self.rng);
                }
                try self.linkage_engine.computeFullLinkageScheduled(self.allocator, self.generation);
            }

            const mut_depth: u3 = if (self.pop_current.entropy < self.entropy_floor) 4 else 5;

            for (0..K) |i| {
                self.pop_next.individuals[i] = self.elite_buffer[i];
            }

            const pop_len = self.pop_current.individuals.len;

            if (options.num_threads <= 1) {
                var out_idx: usize = K;
                while (out_idx < pop_len) {
                    const p1 = self.tournamentSelect();
                    const p2 = self.tournamentSelect();

                    CrossEngine.generateMask(&self.linkage_engine, &self.rng, &self.crossover_mask);
                    var pair = CrossEngine.recombineAndMutateFused(&p1, &p2, &self.crossover_mask, &self.rng, mut_depth);

                    self.evaluateIndividual(&pair.c1);
                    self.pop_next.individuals[out_idx] = pair.c1;
                    out_idx += 1;

                    if (out_idx < pop_len) {
                        self.evaluateIndividual(&pair.c2);
                        self.pop_next.individuals[out_idx] = pair.c2;
                        out_idx += 1;
                    }
                }
            } else {
                const total_offspring = pop_len - K;
                const threads_count = @min(options.num_threads, 16);

                const Context = struct {
                    engine: *Self,
                    depth: u3,
                    total_offspring: usize,
                    threads_count: usize,
                    seeds: [16]u64,

                    fn run(ctx_ptr: *anyopaque, worker_id: usize) void {
                        const ctx: *@This() = @ptrCast(@alignCast(ctx_ptr));
                        const t = worker_id;
                        const start_idx = K + (t * ctx.total_offspring) / ctx.threads_count;
                        const end_idx = K + ((t + 1) * ctx.total_offspring) / ctx.threads_count;
                        const target_slice = ctx.engine.pop_next.individuals[start_idx..end_idx];

                        var local_rng = prng.Xoshiro256.init(ctx.seeds[t]);
                        var local_cross_mask: [words_len]u64 = undefined;

                        var out_idx: usize = 0;
                        while (out_idx < target_slice.len) {
                            const p1 = ctx.engine.tournamentSelectWithRng(&local_rng);
                            const p2 = ctx.engine.tournamentSelectWithRng(&local_rng);

                            CrossEngine.generateMask(&ctx.engine.linkage_engine, &local_rng, &local_cross_mask);
                            var pair = CrossEngine.recombineAndMutateFused(&p1, &p2, &local_cross_mask, &local_rng, ctx.depth);

                            const pheno1 = ctx.engine.adapter.decode(&pair.c1);
                            pair.c1.fitness = ctx.engine.adapter.evaluate(pheno1);
                            target_slice[out_idx] = pair.c1;
                            out_idx += 1;

                            if (out_idx < target_slice.len) {
                                const pheno2 = ctx.engine.adapter.decode(&pair.c2);
                                pair.c2.fitness = ctx.engine.adapter.evaluate(pheno2);
                                target_slice[out_idx] = pair.c2;
                                out_idx += 1;
                            }
                        }
                    }
                };

                var ctx = Context{
                    .engine = self,
                    .depth = mut_depth,
                    .total_offspring = total_offspring,
                    .threads_count = threads_count,
                    .seeds = undefined,
                };
                for (0..threads_count) |t| {
                    ctx.seeds[t] = self.rng.next();
                }

                if (self.thread_pool) |p| {
                    p.dispatch(&ctx, Context.run);
                } else {
                    Context.run(&ctx, 0);
                }
            }

            const tmp = self.pop_current.individuals;
            self.pop_current.individuals = self.pop_next.individuals;
            self.pop_next.individuals = tmp;

            self.pop_current.updateMetrics();
            self.generation += 1;
        }

        pub inline fn tournamentSelect(self: *Self) Chrom {
            return self.tournamentSelectWithRng(&self.rng);
        }

        pub inline fn tournamentSelectWithRng(self: *const Self, rng: *prng.Xoshiro256) Chrom {
            const n = self.pop_current.individuals.len;
            const idx1 = rng.uintLessThan(n);
            const idx2 = rng.uintLessThan(n);
            const indiv1 = self.pop_current.individuals[idx1];
            const indiv2 = self.pop_current.individuals[idx2];
            return if (indiv1.fitness >= indiv2.fitness) indiv1 else indiv2;
        }

        pub fn getBest(self: *const Self) Chrom {
            return self.pop_current.individuals[self.pop_current.best_idx];
        }
    };
}

pub fn DynamicGeneticEngine(comptime Adapter: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        pop_current: types.DynamicPopulation,
        pop_next: types.DynamicPopulation,
        linkage_engine: linkage.DynamicLinkageEngine,
        dedup_cache: dedup.L1Dedup(8192),
        rng: prng.Xoshiro256,
        adapter: Adapter,
        generation: usize,
        bit_length: usize,
        words_len: usize,
        elite_k: usize,
        elite_indices: []usize,
        elite_bitboard: []align(64) u64,
        crossover_mask: []align(64) u64,
        mutation_mask: []align(64) u64,
        entropy_floor: f64,
        enable_dedup: bool,

        pub fn init(
            allocator: std.mem.Allocator,
            bit_len: usize,
            pop_size: usize,
            adapter: Adapter,
            seed: u64,
            options: EngineOptions,
        ) !Self {
            std.debug.assert(bit_len > 0);
            const e_k: usize = if (bit_len <= 1024) 64 else if (bit_len <= 8192) 128 else 256;
            std.debug.assert(pop_size >= e_k);

            var pop_cur = try types.DynamicPopulation.init(allocator, pop_size, bit_len);
            errdefer pop_cur.deinit(allocator);

            var pop_nxt = try types.DynamicPopulation.init(allocator, pop_size, bit_len);
            errdefer pop_nxt.deinit(allocator);

            const w_len = pop_cur.pitch;

            var link_eng = try linkage.DynamicLinkageEngine.init(allocator, bit_len);
            errdefer link_eng.deinit(allocator);

            const e_indices = try allocator.alloc(usize, e_k);
            errdefer allocator.free(e_indices);
            for (e_indices, 0..) |*e, i| e.* = i;

            const e_board = try allocator.alignedAlloc(u64, std.mem.Alignment.fromByteUnits(64), e_k * w_len);
            errdefer allocator.free(e_board);
            @memset(e_board, 0);

            const x_mask = try allocator.alignedAlloc(u64, std.mem.Alignment.fromByteUnits(64), w_len);
            errdefer allocator.free(x_mask);
            @memset(x_mask, 0);

            const m_mask = try allocator.alignedAlloc(u64, std.mem.Alignment.fromByteUnits(64), w_len);
            errdefer allocator.free(m_mask);
            @memset(m_mask, 0);

            var engine_inst = Self{
                .allocator = allocator,
                .pop_current = pop_cur,
                .pop_next = pop_nxt,
                .linkage_engine = link_eng,
                .dedup_cache = dedup.L1Dedup(8192).init(),
                .rng = prng.Xoshiro256.init(seed),
                .adapter = adapter,
                .generation = 0,
                .bit_length = bit_len,
                .words_len = w_len,
                .elite_k = e_k,
                .elite_indices = e_indices,
                .elite_bitboard = e_board,
                .crossover_mask = x_mask,
                .mutation_mask = m_mask,
                .entropy_floor = 0.05,
                .enable_dedup = options.enable_dedup,
            };

            for (0..pop_size) |i| {
                const words = engine_inst.pop_current.getWordsMut(i);
                for (words) |*w| {
                    w.* = engine_inst.rng.next();
                }
                crossover.DynamicCrossoverEngine.cleanMaskPadding(words, bit_len);
                engine_inst.evaluateIndividual(i);
            }
            engine_inst.pop_current.updateMetrics();

            return engine_inst;
        }

        pub fn deinit(self: *Self) void {
            self.pop_current.deinit(self.allocator);
            self.pop_next.deinit(self.allocator);
            self.linkage_engine.deinit(self.allocator);
            self.allocator.free(self.elite_indices);
            self.allocator.free(self.elite_bitboard);
            self.allocator.free(self.crossover_mask);
            self.allocator.free(self.mutation_mask);
        }

        pub fn evaluateIndividual(self: *Self, pop_idx: usize) void {
            const words = self.pop_current.getWords(pop_idx);

            if (self.enable_dedup) {
                const h = dedup.L1Dedup(8192).hashChromosome(words);
                self.pop_current.hashes[pop_idx] = h;

                if (self.dedup_cache.lookup(h)) |cached_fit| {
                    self.pop_current.fitnesses[pop_idx] = cached_fit;
                    return;
                }

                const chrom_view = self.pop_current.get(pop_idx);
                const phenotype = self.adapter.decode(&chrom_view);
                const score = self.adapter.evaluate(phenotype);
                self.pop_current.fitnesses[pop_idx] = score;
                self.dedup_cache.insert(h, score);
            } else {
                const chrom_view = self.pop_current.get(pop_idx);
                const phenotype = self.adapter.decode(&chrom_view);
                self.pop_current.fitnesses[pop_idx] = self.adapter.evaluate(phenotype);
            }
        }

        pub fn evaluateOffspring(self: *Self, next_idx: usize) void {
            const words = self.pop_next.getWords(next_idx);

            if (self.enable_dedup) {
                const h = dedup.L1Dedup(8192).hashChromosome(words);
                self.pop_next.hashes[next_idx] = h;

                if (self.dedup_cache.lookup(h)) |cached_fit| {
                    self.pop_next.fitnesses[next_idx] = cached_fit;
                    return;
                }

                const chrom_view = self.pop_next.get(next_idx);
                const phenotype = self.adapter.decode(&chrom_view);
                const score = self.adapter.evaluate(phenotype);
                self.pop_next.fitnesses[next_idx] = score;
                self.dedup_cache.insert(h, score);
            } else {
                const chrom_view = self.pop_next.get(next_idx);
                const phenotype = self.adapter.decode(&chrom_view);
                self.pop_next.fitnesses[next_idx] = self.adapter.evaluate(phenotype);
            }
        }

        pub fn extractElites(self: *Self) void {
            const fits = self.pop_current.fitnesses;
            const k = self.elite_k;

            for (0..k) |i| self.elite_indices[i] = i;

            var min_pos: usize = 0;
            var min_val: f64 = fits[self.elite_indices[0]];
            for (1..k) |j| {
                const candidate_f = fits[self.elite_indices[j]];
                if (candidate_f < min_val) {
                    min_val = candidate_f;
                    min_pos = j;
                }
            }

            for (k..fits.len) |i| {
                const f = fits[i];
                if (f > min_val) {
                    self.elite_indices[min_pos] = i;
                    min_pos = 0;
                    min_val = fits[self.elite_indices[0]];
                    for (1..k) |j| {
                        const candidate_f = fits[self.elite_indices[j]];
                        if (candidate_f < min_val) {
                            min_val = candidate_f;
                            min_pos = j;
                        }
                    }
                }
            }

            for (self.elite_indices, 0..) |src_idx, dst_idx| {
                const src_words = self.pop_current.getWords(src_idx);
                const dst_offset = dst_idx * self.words_len;
                @memcpy(self.elite_bitboard[dst_offset .. dst_offset + self.words_len], src_words);
            }
        }

        pub fn step(self: *Self) !void {
            self.extractElites();
            self.linkage_engine.ingestElitesFromPopulation(&self.pop_current, self.elite_indices);

            if (self.generation == 0 or self.generation % 4 == 0) {
                if (self.generation > 0 and self.generation % 8 == 0) {
                    self.linkage_engine.shufflePermutation(&self.rng);
                }
                try self.linkage_engine.computeFullLinkageScheduled(self.allocator, self.generation);
            }

            const mut_depth: u3 = if (self.pop_current.entropy < self.entropy_floor) 4 else 5;
            const k = self.elite_k;
            const pop_len = self.pop_current.pop_size;

            for (0..k) |i| {
                const src_offset = i * self.words_len;
                const dst_words = self.pop_next.getWordsMut(i);
                @memcpy(dst_words, self.elite_bitboard[src_offset .. src_offset + self.words_len]);
                const src_pop_idx = self.elite_indices[i];
                self.pop_next.fitnesses[i] = self.pop_current.fitnesses[src_pop_idx];
                self.pop_next.hashes[i] = self.pop_current.hashes[src_pop_idx];
            }

            var out_idx: usize = k;
            while (out_idx < pop_len) {
                const p1_idx = self.tournamentSelectIdx();
                const p2_idx = self.tournamentSelectIdx();

                const p1_words = self.pop_current.getWords(p1_idx);
                const p2_words = self.pop_current.getWords(p2_idx);

                crossover.DynamicCrossoverEngine.generateMask(
                    self.linkage_engine.adjacency_heads,
                    self.linkage_engine.adjacency_targets.items,
                    self.linkage_engine.adjacency_next.items,
                    &self.rng,
                    self.bit_length,
                    self.crossover_mask,
                );

                const c1_words = self.pop_next.getWordsMut(out_idx);
                if (out_idx + 1 < pop_len) {
                    const c2_words = self.pop_next.getWordsMut(out_idx + 1);
                    crossover.DynamicCrossoverEngine.recombineAndMutateFused(
                        p1_words,
                        p2_words,
                        self.crossover_mask,
                        &self.rng,
                        mut_depth,
                        c1_words,
                        c2_words,
                        self.bit_length,
                    );

                    self.evaluateOffspring(out_idx);
                    self.evaluateOffspring(out_idx + 1);
                    out_idx += 2;
                } else {
                    for (p1_words, p2_words, self.crossover_mask, c1_words) |w1, w2, m, *out| {
                        out.* = ((w1 & m) | (w2 & ~m)) ^ self.rng.sparseMask(mut_depth);
                    }
                    crossover.DynamicCrossoverEngine.cleanMaskPadding(c1_words, self.bit_length);

                    self.evaluateOffspring(out_idx);
                    out_idx += 1;
                }
            }

            const tmp_bitboard = self.pop_current.bitboard;
            self.pop_current.bitboard = self.pop_next.bitboard;
            self.pop_next.bitboard = tmp_bitboard;

            const tmp_fits = self.pop_current.fitnesses;
            self.pop_current.fitnesses = self.pop_next.fitnesses;
            self.pop_next.fitnesses = tmp_fits;

            const tmp_hashes = self.pop_current.hashes;
            self.pop_current.hashes = self.pop_next.hashes;
            self.pop_next.hashes = tmp_hashes;

            self.pop_current.updateMetrics();
            self.generation += 1;
        }

        pub inline fn tournamentSelectIdx(self: *Self) usize {
            const n = self.pop_current.pop_size;
            const idx1 = self.rng.uintLessThan(n);
            const idx2 = self.rng.uintLessThan(n);
            const f1 = self.pop_current.fitnesses[idx1];
            const f2 = self.pop_current.fitnesses[idx2];
            return if (f1 >= f2) idx1 else idx2;
        }

        pub fn getBestFitness(self: *const Self) f64 {
            return self.pop_current.best_fitness;
        }

        pub fn getBestWords(self: *const Self) []const u64 {
            return self.pop_current.getWords(self.pop_current.best_idx);
        }
    };
}
