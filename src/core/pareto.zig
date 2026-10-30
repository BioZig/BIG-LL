const std = @import("std");
const types = @import("types.zig");
const prng = @import("prng.zig");

pub fn ParetoChromosome(comptime L: usize, comptime M: usize) type {
    comptime {
        std.debug.assert(L > 0);
        std.debug.assert(M >= 2);
    }
    const words_len = (L + 63) / 64;

    return struct {
        const Self = @This();
        pub const bit_length: usize = L;
        pub const num_objectives: usize = M;

        words: [words_len]u64 align(64),
        objectives: [M]f64,
        rank: usize,
        crowding_distance: f64,

        pub fn initZero() Self {
            return .{
                .words = [_]u64{0} ** words_len,
                .objectives = [_]f64{0.0} ** M,
                .rank = 0,
                .crowding_distance = 0.0,
            };
        }

        pub inline fn setBit(self: *Self, idx: usize, val: u1) void {
            const w = idx / 64;
            const b: u6 = @intCast(idx % 64);
            const mask = @as(u64, 1) << b;
            if (val == 1) {
                self.words[w] |= mask;
            } else {
                self.words[w] &= ~mask;
            }
        }

        pub inline fn getBit(self: *const Self, idx: usize) u1 {
            const w = idx / 64;
            const b: u6 = @intCast(idx % 64);
            return @truncate((self.words[w] >> b) & 1);
        }

        pub fn popCount(self: *const Self) usize {
            var sum: usize = 0;
            inline for (0..words_len) |w| {
                sum += @popCount(self.words[w]);
            }
            return sum;
        }

        pub fn cleanPadding(self: *Self) void {
            const rem = L % 64;
            if (rem > 0) {
                const mask = (@as(u64, 1) << @as(u6, @intCast(rem))) - 1;
                self.words[words_len - 1] &= mask;
            }
        }
    };
}

pub fn dominates(comptime M: usize, a: [M]f64, b: [M]f64) bool {
    var at_least_one_greater = false;
    for (0..M) |i| {
        if (a[i] < b[i]) return false;
        if (a[i] > b[i]) at_least_one_greater = true;
    }
    return at_least_one_greater;
}

pub fn computeNonDominatedRanks(
    comptime L: usize,
    comptime M: usize,
    population: []ParetoChromosome(L, M),
    allocator: std.mem.Allocator,
) !void {
    const n = population.len;
    if (n == 0) return;

    var domination_count = try allocator.alloc(usize, n);
    defer allocator.free(domination_count);
    @memset(domination_count, 0);

    var dominated_lists = try allocator.alloc(std.ArrayListUnmanaged(usize), n);
    defer {
        for (dominated_lists) |*list| list.deinit(allocator);
        allocator.free(dominated_lists);
    }
    for (dominated_lists) |*list| list.* = .empty;

    var current_front = try std.ArrayListUnmanaged(usize).initCapacity(allocator, n);
    defer current_front.deinit(allocator);

    for (0..n) |i| {
        for (i + 1..n) |j| {
            if (dominates(M, population[i].objectives, population[j].objectives)) {
                try dominated_lists[i].append(allocator, j);
                domination_count[j] += 1;
            } else if (dominates(M, population[j].objectives, population[i].objectives)) {
                try dominated_lists[j].append(allocator, i);
                domination_count[i] += 1;
            }
        }
        if (domination_count[i] == 0) {
            population[i].rank = 0;
            current_front.appendAssumeCapacity(i);
        }
    }

    var front_rank: usize = 0;
    var next_front = try std.ArrayListUnmanaged(usize).initCapacity(allocator, n);
    defer next_front.deinit(allocator);

    while (current_front.items.len > 0) {
        next_front.clearRetainingCapacity();

        for (current_front.items) |p| {
            for (dominated_lists[p].items) |q| {
                domination_count[q] -= 1;
                if (domination_count[q] == 0) {
                    population[q].rank = front_rank + 1;
                    try next_front.append(allocator, q);
                }
            }
        }

        front_rank += 1;
        current_front.clearRetainingCapacity();
        try current_front.appendSlice(allocator, next_front.items);
    }
}

pub fn computeCrowdingDistances(
    comptime L: usize,
    comptime M: usize,
    population: []ParetoChromosome(L, M),
    allocator: std.mem.Allocator,
) !void {
    const n = population.len;
    if (n == 0) return;

    for (population) |*ind| {
        ind.crowding_distance = 0.0;
    }

    var max_rank: usize = 0;
    for (population) |ind| {
        if (ind.rank > max_rank) max_rank = ind.rank;
    }

    var front_indices = try std.ArrayListUnmanaged(usize).initCapacity(allocator, n);
    defer front_indices.deinit(allocator);

    for (0..max_rank + 1) |r| {
        front_indices.clearRetainingCapacity();
        for (population, 0..) |ind, idx| {
            if (ind.rank == r) {
                try front_indices.append(allocator, idx);
            }
        }

        const f_len = front_indices.items.len;
        if (f_len <= 2) {
            for (front_indices.items) |idx| {
                population[idx].crowding_distance = std.math.inf(f64);
            }
            continue;
        }

        for (0..M) |m| {
            const SortCtx = struct {
                pop: []const ParetoChromosome(L, M),
                obj: usize,
                fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                    return ctx.pop[a].objectives[ctx.obj] < ctx.pop[b].objectives[ctx.obj];
                }
            };

            std.sort.block(usize, front_indices.items, SortCtx{ .pop = population, .obj = m }, SortCtx.lessThan);

            population[front_indices.items[0]].crowding_distance = std.math.inf(f64);
            population[front_indices.items[f_len - 1]].crowding_distance = std.math.inf(f64);

            const min_val = population[front_indices.items[0]].objectives[m];
            const max_val = population[front_indices.items[f_len - 1]].objectives[m];
            const denom = max_val - min_val;

            if (denom > 1e-9) {
                for (1..f_len - 1) |i| {
                    const prev_val = population[front_indices.items[i - 1]].objectives[m];
                    const next_val = population[front_indices.items[i + 1]].objectives[m];
                    population[front_indices.items[i]].crowding_distance += (next_val - prev_val) / denom;
                }
            }
        }
    }
}

pub fn ParetoEngine(comptime L: usize, comptime M: usize, comptime Adapter: type) type {
    const Chrom = ParetoChromosome(L, M);
    const words_len = (L + 63) / 64;

    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        pop_current: []Chrom,
        pop_next: []Chrom,
        adapter: Adapter,
        rng: prng.Xoshiro256,
        generation: usize,

        pub fn init(allocator: std.mem.Allocator, pop_size: usize, adapter: Adapter, seed: u64) !Self {
            const cur = try allocator.alloc(Chrom, pop_size);
            const nxt = try allocator.alloc(Chrom, pop_size);

            var rng = prng.Xoshiro256.init(seed);

            for (cur) |*indiv| {
                indiv.* = Chrom.initZero();
                rng.fillWords(&indiv.words);
                indiv.cleanPadding();
                indiv.objectives = adapter.evaluate(indiv);
            }

            try computeNonDominatedRanks(L, M, cur, allocator);
            try computeCrowdingDistances(L, M, cur, allocator);

            return Self{
                .allocator = allocator,
                .pop_current = cur,
                .pop_next = nxt,
                .adapter = adapter,
                .rng = rng,
                .generation = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.pop_current);
            self.allocator.free(self.pop_next);
        }

        pub fn tournamentSelect(self: *Self) Chrom {
            const n = self.pop_current.len;
            const idx_a = self.rng.uintLessThan(n);
            const idx_b = self.rng.uintLessThan(n);
            const a = self.pop_current[idx_a];
            const b = self.pop_current[idx_b];

            if (a.rank < b.rank) return a;
            if (b.rank < a.rank) return b;
            if (a.crowding_distance > b.crowding_distance) return a;
            return b;
        }

        pub fn step(self: *Self) !void {
            const pop_len = self.pop_current.len;

            var out_idx: usize = 0;
            var mut_mask: [words_len]u64 = undefined;

            while (out_idx < pop_len) {
                const p1 = self.tournamentSelect();
                const p2 = self.tournamentSelect();

                var c1 = Chrom.initZero();
                var c2 = Chrom.initZero();

                var cross_mask: [words_len]u64 = undefined;
                self.rng.fillWords(&cross_mask);

                for (0..words_len) |w| {
                    const m = cross_mask[w];
                    c1.words[w] = (p1.words[w] & m) | (p2.words[w] & ~m);
                    c2.words[w] = (p2.words[w] & m) | (p1.words[w] & ~m);
                }

                self.rng.fillSparseWords(&mut_mask, 5);
                for (0..words_len) |w| c1.words[w] ^= mut_mask[w];
                c1.cleanPadding();

                self.rng.fillSparseWords(&mut_mask, 5);
                for (0..words_len) |w| c2.words[w] ^= mut_mask[w];
                c2.cleanPadding();

                c1.objectives = self.adapter.evaluate(&c1);
                self.pop_next[out_idx] = c1;
                out_idx += 1;

                if (out_idx < pop_len) {
                    c2.objectives = self.adapter.evaluate(&c2);
                    self.pop_next[out_idx] = c2;
                    out_idx += 1;
                }
            }

            const tmp = self.pop_current;
            self.pop_current = self.pop_next;
            self.pop_next = tmp;

            try computeNonDominatedRanks(L, M, self.pop_current, self.allocator);
            try computeCrowdingDistances(L, M, self.pop_current, self.allocator);
            self.generation += 1;
        }

        pub fn countFrontZero(self: *const Self) usize {
            var count: usize = 0;
            for (self.pop_current) |ind| {
                if (ind.rank == 0) count += 1;
            }
            return count;
        }
    };
}
