const std = @import("std");
const types = @import("types.zig");
const prng = @import("prng.zig");

pub fn LinkageEngine(comptime L: usize) type {
    comptime {
        std.debug.assert(L > 0);
    }

    const BATCH_SIZE: usize = if (L <= 2048) L else 2048;
    const NUM_BATCHES: usize = (L + BATCH_SIZE - 1) / BATCH_SIZE;
    const ELITE_K: usize = if (L <= 1024) 64 else if (L <= 8192) 128 else 256;
    const ELITE_WORDS: usize = ELITE_K / 64;
    const Chrom = types.Chromosome(L);

    return struct {
        const Self = @This();

        pub const batch_size: usize = BATCH_SIZE;
        pub const num_batches: usize = NUM_BATCHES;
        pub const elite_size: usize = ELITE_K;

        pub const Edge = struct {
            target: u32,
        };

        transposed_elites: [L][ELITE_WORDS]u64,
        permutation: [L]u32,
        dense_tile: [BATCH_SIZE][(BATCH_SIZE + 63) / 64]u64,
        adjacency_heads: [L]u32,
        adjacency_next: std.ArrayListUnmanaged(u32),
        adjacency_targets: std.ArrayListUnmanaged(u32),
        edge_count: usize,
        threshold_count: u32,
        threshold_scaled: u64,

        pub fn init(allocator: std.mem.Allocator) !Self {
            const thresh = calculateAdaptiveThreshold(L, ELITE_K);
            var self = Self{
                .transposed_elites = [_][ELITE_WORDS]u64{[_]u64{0} ** ELITE_WORDS} ** L,
                .permutation = undefined,
                .dense_tile = [_][(BATCH_SIZE + 63) / 64]u64{[_]u64{0} ** ((BATCH_SIZE + 63) / 64)} ** BATCH_SIZE,
                .adjacency_heads = [_]u32{std.math.maxInt(u32)} ** L,
                .adjacency_next = try std.ArrayListUnmanaged(u32).initCapacity(allocator, 4096),
                .adjacency_targets = try std.ArrayListUnmanaged(u32).initCapacity(allocator, 4096),
                .edge_count = 0,
                .threshold_count = thresh,
                .threshold_scaled = @as(u64, thresh) * @as(u64, ELITE_K),
            };

            for (0..L) |i| {
                self.permutation[i] = @intCast(i);
            }
            return self;
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.adjacency_next.deinit(allocator);
            self.adjacency_targets.deinit(allocator);
        }

        pub fn calculateAdaptiveThreshold(gene_len: usize, k: usize) u32 {
            const l_f = @as(f64, @floatFromInt(gene_len));
            const k_f = @as(f64, @floatFromInt(k));
            const tau = @sqrt((2.0 * @log(l_f)) / k_f);
            const raw_thresh = tau * (k_f / 2.0);
            const thresh_int = @as(u32, @intFromFloat(@round(raw_thresh)));
            return @max(thresh_int, 4);
        }

        pub fn shufflePermutation(self: *Self, rng: *prng.Xoshiro256) void {
            var i: usize = L - 1;
            while (i > 0) : (i -= 1) {
                const j = rng.uintLessThan(i + 1);
                const tmp = self.permutation[i];
                self.permutation[i] = self.permutation[j];
                self.permutation[j] = tmp;
            }
        }

        pub fn ingestElites(self: *Self, elites: []const Chrom) void {
            std.debug.assert(elites.len >= ELITE_K);

            for (&self.transposed_elites) |*w_arr| {
                @memset(w_arr, 0);
            }

            for (0..ELITE_K) |indiv_idx| {
                const w_slot = indiv_idx / 64;
                const b_slot: u6 = @intCast(indiv_idx % 64);
                const mask = @as(u64, 1) << b_slot;

                const chrom = &elites[indiv_idx];
                for (0..L) |gene_idx| {
                    if (chrom.getBit(gene_idx) == 1) {
                        self.transposed_elites[gene_idx][w_slot] |= mask;
                    }
                }
            }
        }

        pub fn clearGraph(self: *Self) void {
            @memset(&self.adjacency_heads, std.math.maxInt(u32));
            self.adjacency_next.clearRetainingCapacity();
            self.adjacency_targets.clearRetainingCapacity();
            self.edge_count = 0;
        }

        pub fn addEdge(self: *Self, allocator: std.mem.Allocator, u: u32, v: u32) !void {
            const edge_idx: u32 = @intCast(self.adjacency_targets.items.len);
            try self.adjacency_targets.append(allocator, v);
            try self.adjacency_next.append(allocator, self.adjacency_heads[u]);
            self.adjacency_heads[u] = edge_idx;
            self.edge_count += 1;
        }

        pub fn computeBatchLinkage(self: *Self, allocator: std.mem.Allocator, batch_idx: usize) !void {
            std.debug.assert(batch_idx < NUM_BATCHES);
            const start = batch_idx * BATCH_SIZE;
            const end = @min(start + BATCH_SIZE, L);
            const count = end - start;

            const tile_words = (count + 63) / 64;
            for (0..count) |i| {
                @memset(self.dense_tile[i][0..tile_words], 0);
            }

            for (0..count) |i| {
                const g_i = self.permutation[start + i];
                const pop_i = self.getGenePopCount(g_i);
                const elite_words_i = self.transposed_elites[g_i];

                for (i + 1..count) |j| {
                    const g_j = self.permutation[start + j];
                    const pop_j = self.getGenePopCount(g_j);

                    var joint_pop: usize = 0;
                    inline for (0..ELITE_WORDS) |w| {
                        joint_pop += @popCount(elite_words_i[w] & self.transposed_elites[g_j][w]);
                    }

                    const scaled_joint = @as(u64, @intCast(joint_pop)) * @as(u64, ELITE_K);
                    const prod = @as(u64, @intCast(pop_i)) * @as(u64, @intCast(pop_j));
                    const diff = if (scaled_joint >= prod) scaled_joint - prod else prod - scaled_joint;

                    if (diff >= self.threshold_scaled) {
                        try self.addEdge(allocator, g_i, g_j);
                        try self.addEdge(allocator, g_j, g_i);
                    }
                }
            }
        }

        pub fn computeCrossBatchLinkage(self: *Self, allocator: std.mem.Allocator, batch_a: usize, batch_b: usize) !void {
            std.debug.assert(batch_a < NUM_BATCHES);
            std.debug.assert(batch_b < NUM_BATCHES);
            std.debug.assert(batch_a != batch_b);

            const start_a = batch_a * BATCH_SIZE;
            const end_a = @min(start_a + BATCH_SIZE, L);
            const count_a = end_a - start_a;

            const start_b = batch_b * BATCH_SIZE;
            const end_b = @min(start_b + BATCH_SIZE, L);
            const count_b = end_b - start_b;

            for (0..count_a) |i| {
                const g_i = self.permutation[start_a + i];
                const pop_i = self.getGenePopCount(g_i);
                const elite_words_i = self.transposed_elites[g_i];

                for (0..count_b) |j| {
                    const g_j = self.permutation[start_b + j];
                    const pop_j = self.getGenePopCount(g_j);

                    var joint_pop: usize = 0;
                    inline for (0..ELITE_WORDS) |w| {
                        joint_pop += @popCount(elite_words_i[w] & self.transposed_elites[g_j][w]);
                    }

                    const scaled_joint = @as(u64, @intCast(joint_pop)) * @as(u64, ELITE_K);
                    const prod = @as(u64, @intCast(pop_i)) * @as(u64, @intCast(pop_j));
                    const diff = if (scaled_joint >= prod) scaled_joint - prod else prod - scaled_joint;

                    if (diff >= self.threshold_scaled) {
                        try self.addEdge(allocator, g_i, g_j);
                        try self.addEdge(allocator, g_j, g_i);
                    }
                }
            }
        }

        pub fn computeFullLinkage(self: *Self, allocator: std.mem.Allocator) !void {
            try self.computeFullLinkageScheduled(allocator, 0);
        }

        pub fn computeFullLinkageScheduled(self: *Self, allocator: std.mem.Allocator, epoch: usize) !void {
            self.clearGraph();
            for (0..NUM_BATCHES) |b| {
                try self.computeBatchLinkage(allocator, b);
            }

            if (NUM_BATCHES > 1) {
                if (NUM_BATCHES <= 4) {
                    for (0..NUM_BATCHES) |a| {
                        for (a + 1..NUM_BATCHES) |b| {
                            try self.computeCrossBatchLinkage(allocator, a, b);
                        }
                    }
                } else {
                    const stride = 1 + (epoch % (NUM_BATCHES - 1));
                    for (0..NUM_BATCHES) |a| {
                        const b = (a + stride) % NUM_BATCHES;
                        if (a < b) {
                            try self.computeCrossBatchLinkage(allocator, a, b);
                        }
                    }
                }
            }
        }

        pub fn hasEdge(self: *const Self, u: u32, v: u32) bool {
            if (u >= L or v >= L) return false;
            var curr = self.adjacency_heads[u];
            while (curr != std.math.maxInt(u32)) {
                if (self.adjacency_targets.items[curr] == v) return true;
                curr = self.adjacency_next.items[curr];
            }
            return false;
        }

        inline fn getGenePopCount(self: *const Self, gene_idx: usize) usize {
            var sum: usize = 0;
            inline for (0..ELITE_WORDS) |w| {
                sum += @popCount(self.transposed_elites[gene_idx][w]);
            }
            return sum;
        }
    };
}

pub const DynamicLinkageEngine = struct {
    transposed_elites: []align(64) u64,
    permutation: []u32,
    adjacency_heads: []u32,
    adjacency_next: std.ArrayListUnmanaged(u32),
    adjacency_targets: std.ArrayListUnmanaged(u32),
    dense_tile: []align(64) u64,

    chrom_len: usize,
    elite_k: usize,
    elite_words: usize,
    batch_size: usize,
    num_batches: usize,
    edge_count: usize,
    threshold_count: u32,
    threshold_scaled: u64,

    pub fn init(allocator: std.mem.Allocator, l: usize) !DynamicLinkageEngine {
        std.debug.assert(l > 0);
        const b_size: usize = if (l <= 2048) l else 2048;
        const n_batches: usize = (l + b_size - 1) / b_size;
        const e_k: usize = if (l <= 1024) 64 else if (l <= 8192) 128 else 256;
        const e_words: usize = e_k / 64;
        const thresh = calculateAdaptiveThreshold(l, e_k);

        const trans_elites = try allocator.alignedAlloc(u64, std.mem.Alignment.fromByteUnits(64), l * e_words);
        errdefer allocator.free(trans_elites);
        @memset(trans_elites, 0);

        const perm = try allocator.alloc(u32, l);
        errdefer allocator.free(perm);
        for (perm, 0..) |*p, i| p.* = @intCast(i);

        const adj_heads = try allocator.alloc(u32, l);
        errdefer allocator.free(adj_heads);
        @memset(adj_heads, std.math.maxInt(u32));

        const tile_words = (b_size + 63) / 64;
        const dtile = try allocator.alignedAlloc(u64, std.mem.Alignment.fromByteUnits(64), b_size * tile_words);
        errdefer allocator.free(dtile);
        @memset(dtile, 0);

        var next_arr = try std.ArrayListUnmanaged(u32).initCapacity(allocator, 4096);
        errdefer next_arr.deinit(allocator);

        var target_arr = try std.ArrayListUnmanaged(u32).initCapacity(allocator, 4096);
        errdefer target_arr.deinit(allocator);

        return DynamicLinkageEngine{
            .transposed_elites = trans_elites,
            .permutation = perm,
            .adjacency_heads = adj_heads,
            .adjacency_next = next_arr,
            .adjacency_targets = target_arr,
            .dense_tile = dtile,
            .chrom_len = l,
            .elite_k = e_k,
            .elite_words = e_words,
            .batch_size = b_size,
            .num_batches = n_batches,
            .edge_count = 0,
            .threshold_count = thresh,
            .threshold_scaled = @as(u64, thresh) * @as(u64, e_k),
        };
    }

    pub fn deinit(self: *DynamicLinkageEngine, allocator: std.mem.Allocator) void {
        allocator.free(self.transposed_elites);
        allocator.free(self.permutation);
        allocator.free(self.adjacency_heads);
        allocator.free(self.dense_tile);
        self.adjacency_next.deinit(allocator);
        self.adjacency_targets.deinit(allocator);
    }

    pub fn calculateAdaptiveThreshold(gene_len: usize, k: usize) u32 {
        const l_f = @as(f64, @floatFromInt(gene_len));
        const k_f = @as(f64, @floatFromInt(k));
        const tau = @sqrt((2.0 * @log(l_f)) / k_f);
        const raw_thresh = tau * (k_f / 2.0);
        const thresh_int = @as(u32, @intFromFloat(@round(raw_thresh)));
        return @max(thresh_int, 4);
    }

    pub fn shufflePermutation(self: *DynamicLinkageEngine, rng: *prng.Xoshiro256) void {
        var i: usize = self.chrom_len - 1;
        while (i > 0) : (i -= 1) {
            const j = rng.uintLessThan(i + 1);
            const tmp = self.permutation[i];
            self.permutation[i] = self.permutation[j];
            self.permutation[j] = tmp;
        }
    }

    pub fn ingestElitesFromPopulation(self: *DynamicLinkageEngine, pop: *const types.DynamicPopulation, elite_indices: []const usize) void {
        std.debug.assert(elite_indices.len >= self.elite_k);
        @memset(self.transposed_elites, 0);

        for (0..self.elite_k) |indiv_rank| {
            const indiv_idx = elite_indices[indiv_rank];
            const w_slot = indiv_rank / 64;
            const b_slot: u6 = @intCast(indiv_rank % 64);
            const mask = @as(u64, 1) << b_slot;

            const words = pop.getWords(indiv_idx);
            for (0..self.chrom_len) |gene_idx| {
                const w_idx = gene_idx / 64;
                const b_idx: u6 = @intCast(gene_idx % 64);
                if ((words[w_idx] & (@as(u64, 1) << b_idx)) != 0) {
                    self.transposed_elites[gene_idx * self.elite_words + w_slot] |= mask;
                }
            }
        }
    }

    pub fn clearGraph(self: *DynamicLinkageEngine) void {
        @memset(self.adjacency_heads, std.math.maxInt(u32));
        self.adjacency_next.clearRetainingCapacity();
        self.adjacency_targets.clearRetainingCapacity();
        self.edge_count = 0;
    }

    pub fn addEdge(self: *DynamicLinkageEngine, allocator: std.mem.Allocator, u: u32, v: u32) !void {
        const edge_idx: u32 = @intCast(self.adjacency_targets.items.len);
        try self.adjacency_targets.append(allocator, v);
        try self.adjacency_next.append(allocator, self.adjacency_heads[u]);
        self.adjacency_heads[u] = edge_idx;
        self.edge_count += 1;
    }

    pub fn computeBatchLinkage(self: *DynamicLinkageEngine, allocator: std.mem.Allocator, batch_idx: usize) !void {
        std.debug.assert(batch_idx < self.num_batches);
        const start = batch_idx * self.batch_size;
        const end = @min(start + self.batch_size, self.chrom_len);
        const count = end - start;
        for (0..count) |i| {
            const g_i = self.permutation[start + i];
            const pop_i = self.getGenePopCount(g_i);
            const offset_i = g_i * self.elite_words;
            const words_i = self.transposed_elites[offset_i .. offset_i + self.elite_words];

            for (i + 1..count) |j| {
                const g_j = self.permutation[start + j];
                const pop_j = self.getGenePopCount(g_j);
                const offset_j = g_j * self.elite_words;

                var joint_pop: usize = 0;
                for (0..self.elite_words) |w| {
                    joint_pop += @popCount(words_i[w] & self.transposed_elites[offset_j + w]);
                }

                const scaled_joint = @as(u64, @intCast(joint_pop)) * @as(u64, @intCast(self.elite_k));
                const prod = @as(u64, @intCast(pop_i)) * @as(u64, @intCast(pop_j));
                const diff = if (scaled_joint >= prod) scaled_joint - prod else prod - scaled_joint;

                if (diff >= self.threshold_scaled) {
                    try self.addEdge(allocator, g_i, g_j);
                    try self.addEdge(allocator, g_j, g_i);
                }
            }
        }
    }

    pub fn computeCrossBatchLinkage(self: *DynamicLinkageEngine, allocator: std.mem.Allocator, batch_a: usize, batch_b: usize) !void {
        std.debug.assert(batch_a < self.num_batches and batch_b < self.num_batches and batch_a != batch_b);
        const start_a = batch_a * self.batch_size;
        const end_a = @min(start_a + self.batch_size, self.chrom_len);
        const count_a = end_a - start_a;

        const start_b = batch_b * self.batch_size;
        const end_b = @min(start_b + self.batch_size, self.chrom_len);
        const count_b = end_b - start_b;

        for (0..count_a) |i| {
            const g_i = self.permutation[start_a + i];
            const pop_i = self.getGenePopCount(g_i);
            const offset_i = g_i * self.elite_words;
            const words_i = self.transposed_elites[offset_i .. offset_i + self.elite_words];

            for (0..count_b) |j| {
                const g_j = self.permutation[start_b + j];
                const pop_j = self.getGenePopCount(g_j);
                const offset_j = g_j * self.elite_words;

                var joint_pop: usize = 0;
                for (0..self.elite_words) |w| {
                    joint_pop += @popCount(words_i[w] & self.transposed_elites[offset_j + w]);
                }

                const scaled_joint = @as(u64, @intCast(joint_pop)) * @as(u64, @intCast(self.elite_k));
                const prod = @as(u64, @intCast(pop_i)) * @as(u64, @intCast(pop_j));
                const diff = if (scaled_joint >= prod) scaled_joint - prod else prod - scaled_joint;

                if (diff >= self.threshold_scaled) {
                    try self.addEdge(allocator, g_i, g_j);
                    try self.addEdge(allocator, g_j, g_i);
                }
            }
        }
    }

    pub fn computeFullLinkage(self: *DynamicLinkageEngine, allocator: std.mem.Allocator) !void {
        try self.computeFullLinkageScheduled(allocator, 0);
    }

    pub fn computeFullLinkageScheduled(self: *DynamicLinkageEngine, allocator: std.mem.Allocator, epoch: usize) !void {
        self.clearGraph();
        for (0..self.num_batches) |b| {
            try self.computeBatchLinkage(allocator, b);
        }

        if (self.num_batches > 1) {
            if (self.num_batches <= 4) {
                for (0..self.num_batches) |a| {
                    for (a + 1..self.num_batches) |b| {
                        try self.computeCrossBatchLinkage(allocator, a, b);
                    }
                }
            } else {
                const stride = 1 + (epoch % (self.num_batches - 1));
                for (0..self.num_batches) |a| {
                    const b = (a + stride) % self.num_batches;
                    if (a < b) {
                        try self.computeCrossBatchLinkage(allocator, a, b);
                    }
                }
            }
        }
    }

    pub fn hasEdge(self: *const DynamicLinkageEngine, u: u32, v: u32) bool {
        if (u >= self.chrom_len or v >= self.chrom_len) return false;
        var curr = self.adjacency_heads[u];
        while (curr != std.math.maxInt(u32)) {
            if (self.adjacency_targets.items[curr] == v) return true;
            curr = self.adjacency_next.items[curr];
        }
        return false;
    }

    inline fn getGenePopCount(self: *const DynamicLinkageEngine, gene_idx: usize) usize {
        var sum: usize = 0;
        const offset = gene_idx * self.elite_words;
        for (0..self.elite_words) |w| {
            sum += @popCount(self.transposed_elites[offset + w]);
        }
        return sum;
    }
};
