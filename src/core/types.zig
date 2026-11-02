const std = @import("std");

pub fn Chromosome(comptime L: usize) type {
    comptime {
        std.debug.assert(L > 0);
    }
    const words_len = (L + 63) / 64;

    return struct {
        const Self = @This();
        pub const bit_length: usize = L;
        pub const word_count: usize = words_len;

        words: [words_len]u64 align(64),
        fitness: f64,
        hash: u64,

        pub fn initZero() Self {
            return .{
                .words = [_]u64{0} ** words_len,
                .fitness = -std.math.inf(f64),
                .hash = 0,
            };
        }

        pub fn getBit(self: *const Self, idx: usize) u1 {
            std.debug.assert(idx < L);
            const w_idx = idx / 64;
            const b_idx: u6 = @intCast(idx % 64);
            return @truncate((self.words[w_idx] >> b_idx) & 1);
        }

        pub fn setBit(self: *Self, idx: usize, val: u1) void {
            std.debug.assert(idx < L);
            const w_idx = idx / 64;
            const b_idx: u6 = @intCast(idx % 64);
            const mask = @as(u64, 1) << b_idx;
            if (val == 1) {
                self.words[w_idx] |= mask;
            } else {
                self.words[w_idx] &= ~mask;
            }
        }

        pub fn toggleBit(self: *Self, idx: usize) void {
            std.debug.assert(idx < L);
            const w_idx = idx / 64;
            const b_idx: u6 = @intCast(idx % 64);
            self.words[w_idx] ^= (@as(u64, 1) << b_idx);
        }

        pub fn popCount(self: *const Self) usize {
            var sum: usize = 0;
            inline for (0..words_len) |i| {
                sum += @popCount(self.words[i]);
            }
            return sum;
        }

        pub fn hammingDistance(self: *const Self, other: *const Self) usize {
            var dist: usize = 0;
            inline for (0..words_len) |i| {
                dist += @popCount(self.words[i] ^ other.words[i]);
            }
            return dist;
        }

        pub fn bitwiseEqual(self: *const Self, other: *const Self) bool {
            inline for (0..words_len) |i| {
                if (self.words[i] != other.words[i]) return false;
            }
            return true;
        }

        pub fn cleanPadding(self: *Self) void {
            const remainder = L % 64;
            if (remainder != 0) {
                const mask = (@as(u64, 1) << @intCast(remainder)) - 1;
                self.words[words_len - 1] &= mask;
            }
        }
    };
}

pub const BITS_PER_BLOCK: usize = 512;
pub const WORDS_PER_BLOCK: usize = 8; // 8 * 64 = 512 bits = 64 bytes = 1 L1D cacheline
pub const BYTES_PER_BLOCK: usize = 64;

/// Cacheline-tiled block chromosome designed for ultra-large dimensions (e.g. 20,480+ bits).
/// Each block is exactly 64 bytes (1 L1D cacheline) and 64-byte aligned.
/// Vectorized inner loops execute across 512-bit boundaries with 0 heap overhead.
pub fn BlockChromosome(comptime num_blocks: usize) type {
    comptime {
        std.debug.assert(num_blocks > 0);
    }
    const total_words = num_blocks * WORDS_PER_BLOCK;
    const total_bits = num_blocks * BITS_PER_BLOCK;

    return struct {
        const Self = @This();
        pub const block_count: usize = num_blocks;
        pub const word_count: usize = total_words;
        pub const bit_length: usize = total_bits;

        // Exactly num_blocks L1D cachelines (64 bytes each), aligned to 64 bytes
        blocks: [num_blocks][WORDS_PER_BLOCK]u64 align(64),
        fitness: f64,
        hash: u64,

        pub fn initZero() Self {
            return .{
                .blocks = [_][WORDS_PER_BLOCK]u64{[_]u64{0} ** WORDS_PER_BLOCK} ** num_blocks,
                .fitness = -std.math.inf(f64),
                .hash = 0,
            };
        }

        pub inline fn getBit(self: *const Self, idx: usize) u1 {
            std.debug.assert(idx < total_bits);
            const blk = idx / BITS_PER_BLOCK;
            const rem = idx % BITS_PER_BLOCK;
            const w_idx = rem / 64;
            const b_idx: u6 = @intCast(rem % 64);
            return @truncate((self.blocks[blk][w_idx] >> b_idx) & 1);
        }

        pub inline fn setBit(self: *Self, idx: usize, val: u1) void {
            std.debug.assert(idx < total_bits);
            const blk = idx / BITS_PER_BLOCK;
            const rem = idx % BITS_PER_BLOCK;
            const w_idx = rem / 64;
            const b_idx: u6 = @intCast(rem % 64);
            const mask = @as(u64, 1) << b_idx;
            if (val == 1) {
                self.blocks[blk][w_idx] |= mask;
            } else {
                self.blocks[blk][w_idx] &= ~mask;
            }
        }

        pub inline fn toggleBit(self: *Self, idx: usize) void {
            std.debug.assert(idx < total_bits);
            const blk = idx / BITS_PER_BLOCK;
            const rem = idx % BITS_PER_BLOCK;
            const w_idx = rem / 64;
            const b_idx: u6 = @intCast(rem % 64);
            self.blocks[blk][w_idx] ^= (@as(u64, 1) << b_idx);
        }

        pub fn popCount(self: *const Self) usize {
            var sum: usize = 0;
            inline for (0..num_blocks) |blk| {
                inline for (0..WORDS_PER_BLOCK) |w| {
                    sum += @popCount(self.blocks[blk][w]);
                }
            }
            return sum;
        }

        pub fn hammingDistance(self: *const Self, other: *const Self) usize {
            var dist: usize = 0;
            inline for (0..num_blocks) |blk| {
                inline for (0..WORDS_PER_BLOCK) |w| {
                    dist += @popCount(self.blocks[blk][w] ^ other.blocks[blk][w]);
                }
            }
            return dist;
        }

        pub fn bitwiseEqual(self: *const Self, other: *const Self) bool {
            inline for (0..num_blocks) |blk| {
                inline for (0..WORDS_PER_BLOCK) |w| {
                    if (self.blocks[blk][w] != other.blocks[blk][w]) return false;
                }
            }
            return true;
        }

        pub inline fn crossoverBlock(
            self: *Self,
            p1: *const Self,
            p2: *const Self,
            mask: [WORDS_PER_BLOCK]u64,
            blk_idx: usize,
        ) void {
            inline for (0..WORDS_PER_BLOCK) |w| {
                self.blocks[blk_idx][w] = (p1.blocks[blk_idx][w] & mask[w]) | (p2.blocks[blk_idx][w] & ~mask[w]);
            }
        }

        pub fn cleanPadding(self: *Self, active_bits: usize) void {
            if (active_bits >= total_bits) return;
            const active_blk = active_bits / BITS_PER_BLOCK;
            const rem_bits = active_bits % BITS_PER_BLOCK;

            inline for (0..num_blocks) |blk| {
                if (blk > active_blk) {
                    inline for (0..WORDS_PER_BLOCK) |w| {
                        self.blocks[blk][w] = 0;
                    }
                }
            }

            if (active_blk < num_blocks) {
                const active_w = rem_bits / 64;
                const rem_w_bits = rem_bits % 64;
                inline for (0..WORDS_PER_BLOCK) |w| {
                    if (w > active_w) {
                        self.blocks[active_blk][w] = 0;
                    } else if (w == active_w and rem_w_bits != 0) {
                        const mask = (@as(u64, 1) << @intCast(rem_w_bits)) - 1;
                        self.blocks[active_blk][w] &= mask;
                    }
                }
            }
        }
    };
}

pub fn Population(comptime L: usize) type {
    const Chrom = Chromosome(L);

    return struct {
        const Self = @This();

        individuals: []Chrom,
        best_idx: usize,
        best_fitness: f64,
        mean_fitness: f64,
        entropy: f64,

        pub fn init(allocator: std.mem.Allocator, size: usize) !Self {
            std.debug.assert(size > 0);
            const memory = try allocator.alloc(Chrom, size);
            for (memory) |*indiv| {
                indiv.* = Chrom.initZero();
            }
            return .{
                .individuals = memory,
                .best_idx = 0,
                .best_fitness = -std.math.inf(f64),
                .mean_fitness = 0.0,
                .entropy = 0.0,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.individuals);
        }

        pub fn updateMetrics(self: *Self) void {
            var sum: f64 = 0.0;
            var max_fit: f64 = -std.math.inf(f64);
            var best_i: usize = 0;

            for (self.individuals, 0..) |*indiv, i| {
                const f = indiv.fitness;
                sum += f;
                if (f > max_fit) {
                    max_fit = f;
                    best_i = i;
                }
            }

            self.best_idx = best_i;
            self.best_fitness = max_fit;
            self.mean_fitness = sum / @as(f64, @floatFromInt(self.individuals.len));
            self.entropy = self.computeShannonEntropy();
        }

        pub fn computeShannonEntropy(self: *const Self) f64 {
            const pop_len = self.individuals.len;
            if (pop_len == 0) return 0.0;
            const pop_f = @as(f64, @floatFromInt(pop_len));

            var total_entropy: f64 = 0.0;
            for (0..L) |bit_idx| {
                var ones: usize = 0;
                for (self.individuals) |*indiv| {
                    if (indiv.getBit(bit_idx) == 1) ones += 1;
                }
                const p1 = @as(f64, @floatFromInt(ones)) / pop_f;
                const p0 = 1.0 - p1;

                var h: f64 = 0.0;
                if (p1 > 1e-9) h -= p1 * @log2(p1);
                if (p0 > 1e-9) h -= p0 * @log2(p0);
                total_entropy += h;
            }
            return total_entropy / @as(f64, @floatFromInt(L));
        }
    };
}

pub const DynamicChromosome = struct {
    words: []u64,
    bit_length: usize,
    fitness: f64,
    hash: u64,

    pub fn initZero(words_slice: []u64, bit_len: usize) DynamicChromosome {
        @memset(words_slice, 0);
        return .{
            .words = words_slice,
            .bit_length = bit_len,
            .fitness = -std.math.inf(f64),
            .hash = 0,
        };
    }

    pub fn getBit(self: *const DynamicChromosome, idx: usize) u1 {
        std.debug.assert(idx < self.bit_length);
        const w_idx = idx / 64;
        const b_idx: u6 = @intCast(idx % 64);
        return @truncate((self.words[w_idx] >> b_idx) & 1);
    }

    pub fn setBit(self: *DynamicChromosome, idx: usize, val: u1) void {
        std.debug.assert(idx < self.bit_length);
        const w_idx = idx / 64;
        const b_idx: u6 = @intCast(idx % 64);
        const mask = @as(u64, 1) << b_idx;
        if (val == 1) {
            self.words[w_idx] |= mask;
        } else {
            self.words[w_idx] &= ~mask;
        }
    }

    pub fn toggleBit(self: *DynamicChromosome, idx: usize) void {
        std.debug.assert(idx < self.bit_length);
        const w_idx = idx / 64;
        const b_idx: u6 = @intCast(idx % 64);
        self.words[w_idx] ^= (@as(u64, 1) << b_idx);
    }

    pub fn popCount(self: *const DynamicChromosome) usize {
        var sum: usize = 0;
        for (self.words) |w| {
            sum += @popCount(w);
        }
        return sum;
    }

    pub fn hammingDistance(self: *const DynamicChromosome, other: *const DynamicChromosome) usize {
        std.debug.assert(self.words.len == other.words.len);
        var dist: usize = 0;
        for (self.words, other.words) |w1, w2| {
            dist += @popCount(w1 ^ w2);
        }
        return dist;
    }

    pub fn bitwiseEqual(self: *const DynamicChromosome, other: *const DynamicChromosome) bool {
        std.debug.assert(self.words.len == other.words.len);
        for (self.words, other.words) |w1, w2| {
            if (w1 != w2) return false;
        }
        return true;
    }

    pub fn cleanPadding(self: *DynamicChromosome) void {
        const active_words = (self.bit_length + 63) / 64;
        const remainder = self.bit_length % 64;
        if (remainder != 0 and active_words > 0 and active_words <= self.words.len) {
            const mask = (@as(u64, 1) << @intCast(remainder)) - 1;
            self.words[active_words - 1] &= mask;
        }
        for (active_words..self.words.len) |w| {
            self.words[w] = 0;
        }
    }
};

pub const DynamicPopulation = struct {
    bitboard: []align(64) u64,
    fitnesses: []f64,
    hashes: []u64,

    pop_size: usize,
    bit_length: usize,
    pitch: usize,
    best_idx: usize,
    best_fitness: f64,
    mean_fitness: f64,
    entropy: f64,

    pub fn init(allocator: std.mem.Allocator, size: usize, bit_len: usize) !DynamicPopulation {
        std.debug.assert(size > 0);
        std.debug.assert(bit_len > 0);

        const raw_words = (bit_len + 63) / 64;
        const words_len = std.mem.alignForward(usize, raw_words, WORDS_PER_BLOCK);
        const total_words = size * words_len;

        const memory = try allocator.alignedAlloc(u64, std.mem.Alignment.fromByteUnits(64), total_words);
        errdefer allocator.free(memory);
        @memset(memory, 0);

        const fits = try allocator.alloc(f64, size);
        errdefer allocator.free(fits);
        for (fits) |*f| f.* = -std.math.inf(f64);

        const h_arr = try allocator.alloc(u64, size);
        errdefer allocator.free(h_arr);
        @memset(h_arr, 0);

        return DynamicPopulation{
            .bitboard = memory,
            .fitnesses = fits,
            .hashes = h_arr,
            .pop_size = size,
            .bit_length = bit_len,
            .pitch = words_len,
            .best_idx = 0,
            .best_fitness = -std.math.inf(f64),
            .mean_fitness = 0.0,
            .entropy = 0.0,
        };
    }

    pub fn deinit(self: *DynamicPopulation, allocator: std.mem.Allocator) void {
        allocator.free(self.bitboard);
        allocator.free(self.fitnesses);
        allocator.free(self.hashes);
    }

    pub inline fn getWords(self: *const DynamicPopulation, idx: usize) []const u64 {
        std.debug.assert(idx < self.pop_size);
        const offset = idx * self.pitch;
        return self.bitboard[offset .. offset + self.pitch];
    }

    pub inline fn getWordsMut(self: *DynamicPopulation, idx: usize) []u64 {
        std.debug.assert(idx < self.pop_size);
        const offset = idx * self.pitch;
        return self.bitboard[offset .. offset + self.pitch];
    }

    pub fn get(self: *const DynamicPopulation, idx: usize) DynamicChromosome {
        std.debug.assert(idx < self.pop_size);
        const offset = idx * self.pitch;
        return .{
            .words = @constCast(self.bitboard[offset .. offset + self.pitch]),
            .bit_length = self.bit_length,
            .fitness = self.fitnesses[idx],
            .hash = self.hashes[idx],
        };
    }

    pub fn updateMetrics(self: *DynamicPopulation) void {
        var sum: f64 = 0.0;
        var max_fit: f64 = -std.math.inf(f64);
        var best_i: usize = 0;

        for (self.fitnesses, 0..) |f, i| {
            sum += f;
            if (f > max_fit) {
                max_fit = f;
                best_i = i;
            }
        }

        self.best_idx = best_i;
        self.best_fitness = max_fit;
        self.mean_fitness = sum / @as(f64, @floatFromInt(self.pop_size));
        self.entropy = self.computeShannonEntropy();
    }

    pub fn computeShannonEntropy(self: *const DynamicPopulation) f64 {
        if (self.pop_size == 0 or self.bit_length == 0) return 0.0;
        const pop_f = @as(f64, @floatFromInt(self.pop_size));

        var total_entropy: f64 = 0.0;
        for (0..self.bit_length) |bit_idx| {
            const w_idx = bit_idx / 64;
            const b_idx: u6 = @intCast(bit_idx % 64);
            const mask = @as(u64, 1) << b_idx;

            var ones: usize = 0;
            for (0..self.pop_size) |i| {
                const word = self.bitboard[i * self.pitch + w_idx];
                if ((word & mask) != 0) ones += 1;
            }
            const p1 = @as(f64, @floatFromInt(ones)) / pop_f;
            const p0 = 1.0 - p1;

            var h: f64 = 0.0;
            if (p1 > 1e-9) h -= p1 * @log2(p1);
            if (p0 > 1e-9) h -= p0 * @log2(p0);
            total_entropy += h;
        }
        return total_entropy / @as(f64, @floatFromInt(self.bit_length));
    }
};
