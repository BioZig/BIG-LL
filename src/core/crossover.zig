const std = @import("std");
const types = @import("types.zig");
const prng = @import("prng.zig");
const linkage = @import("linkage.zig");

pub fn CrossoverEngine(comptime L: usize) type {
    const Chrom = types.Chromosome(L);
    const LinkEngine = linkage.LinkageEngine(L);
    const words_len = (L + 63) / 64;
    const max_cluster_size: usize = @max(1, (L * 35) / 100);

    return struct {
        const Self = @This();
        pub const cluster_ceiling: usize = max_cluster_size;

        pub fn generateMask(
            engine: *const LinkEngine,
            rng: *prng.Xoshiro256,
            out_mask: *[words_len]u64,
        ) void {
            @memset(out_mask, 0);

            const seed_gene = rng.uintLessThan(L);
            setMaskBit(out_mask, seed_gene);

            var cluster_size: usize = 1;
            var edge_cursor = engine.adjacency_heads[seed_gene];

            while (edge_cursor != std.math.maxInt(u32) and cluster_size < max_cluster_size) {
                const neighbor = engine.adjacency_targets.items[edge_cursor];
                if (getMaskBit(out_mask, neighbor) == 0) {
                    setMaskBit(out_mask, neighbor);
                    cluster_size += 1;
                }
                edge_cursor = engine.adjacency_next.items[edge_cursor];
            }

            cleanMaskPadding(out_mask);
        }

        pub fn recombine(
            p1: *const Chrom,
            p2: *const Chrom,
            mask: *const [words_len]u64,
        ) struct { c1: Chrom, c2: Chrom } {
            var c1 = Chrom.initZero();
            var c2 = Chrom.initZero();

            inline for (0..words_len) |w| {
                const m = mask[w];
                c1.words[w] = (p1.words[w] & m) | (p2.words[w] & ~m);
                c2.words[w] = (p2.words[w] & m) | (p1.words[w] & ~m);
            }

            c1.cleanPadding();
            c2.cleanPadding();
            return .{ .c1 = c1, .c2 = c2 };
        }

        pub fn recombineAndMutateFused(
            p1: *const Chrom,
            p2: *const Chrom,
            mask: *const [words_len]u64,
            rng: *prng.Xoshiro256,
            depth: u3,
        ) struct { c1: Chrom, c2: Chrom } {
            var c1 = Chrom.initZero();
            var c2 = Chrom.initZero();

            inline for (0..words_len) |w| {
                const m = mask[w];
                const mut1 = rng.sparseMask(depth);
                const mut2 = rng.sparseMask(depth);
                c1.words[w] = ((p1.words[w] & m) | (p2.words[w] & ~m)) ^ mut1;
                c2.words[w] = ((p2.words[w] & m) | (p1.words[w] & ~m)) ^ mut2;
            }

            c1.cleanPadding();
            c2.cleanPadding();
            return .{ .c1 = c1, .c2 = c2 };
        }

        pub fn mutate(chrom: *Chrom, mut_mask: *const [words_len]u64) void {
            inline for (0..words_len) |w| {
                chrom.words[w] ^= mut_mask[w];
            }
            chrom.cleanPadding();
        }

        inline fn setMaskBit(mask: *[words_len]u64, idx: usize) void {
            const w_idx = idx / 64;
            const b_idx: u6 = @intCast(idx % 64);
            mask[w_idx] |= (@as(u64, 1) << b_idx);
        }

        inline fn getMaskBit(mask: *const [words_len]u64, idx: usize) u1 {
            const w_idx = idx / 64;
            const b_idx: u6 = @intCast(idx % 64);
            return @truncate((mask[w_idx] >> b_idx) & 1);
        }

        inline fn cleanMaskPadding(mask: *[words_len]u64) void {
            const remainder = L % 64;
            if (remainder != 0) {
                const pad_mask = (@as(u64, 1) << @intCast(remainder)) - 1;
                mask[words_len - 1] &= pad_mask;
            }
        }
    };
}

pub const DynamicCrossoverEngine = struct {
    pub fn generateMask(
        adjacency_heads: []const u32,
        adjacency_targets: []const u32,
        adjacency_next: []const u32,
        rng: *prng.Xoshiro256,
        bit_length: usize,
        out_mask: []u64,
    ) void {
        @memset(out_mask, 0);
        if (bit_length == 0) return;

        const max_cluster_size = @max(1, (bit_length * 35) / 100);
        const seed_gene = rng.uintLessThan(bit_length);
        setMaskBit(out_mask, seed_gene);

        var cluster_size: usize = 1;
        var edge_cursor = if (seed_gene < adjacency_heads.len) adjacency_heads[seed_gene] else std.math.maxInt(u32);

        while (edge_cursor != std.math.maxInt(u32) and cluster_size < max_cluster_size) {
            const neighbor = adjacency_targets[edge_cursor];
            if (getMaskBit(out_mask, neighbor) == 0) {
                setMaskBit(out_mask, neighbor);
                cluster_size += 1;
            }
            edge_cursor = adjacency_next[edge_cursor];
        }

        cleanMaskPadding(out_mask, bit_length);
    }

    pub fn recombine(
        p1: []const u64,
        p2: []const u64,
        mask: []const u64,
        c1: []u64,
        c2: []u64,
        bit_length: usize,
    ) void {
        std.debug.assert(p1.len == p2.len and p1.len == mask.len);
        std.debug.assert(p1.len == c1.len and p1.len == c2.len);

        const len = p1.len;
        var i: usize = 0;
        const Vec2 = @Vector(2, u64);

        while (i + 1 < len) : (i += 2) {
            const v_p1: Vec2 = p1[i..][0..2].*;
            const v_p2: Vec2 = p2[i..][0..2].*;
            const v_m: Vec2 = mask[i..][0..2].*;

            c1[i..][0..2].* = (v_p1 & v_m) | (v_p2 & ~v_m);
            c2[i..][0..2].* = (v_p2 & v_m) | (v_p1 & ~v_m);
        }

        while (i < len) : (i += 1) {
            const w1 = p1[i];
            const w2 = p2[i];
            const m = mask[i];
            c1[i] = (w1 & m) | (w2 & ~m);
            c2[i] = (w2 & m) | (w1 & ~m);
        }

        cleanMaskPadding(c1, bit_length);
        cleanMaskPadding(c2, bit_length);
    }

    pub fn recombineAndMutateFused(
        p1: []const u64,
        p2: []const u64,
        mask: []const u64,
        rng: *prng.Xoshiro256,
        depth: u3,
        c1: []u64,
        c2: []u64,
        bit_length: usize,
    ) void {
        std.debug.assert(p1.len == p2.len and p1.len == mask.len);
        std.debug.assert(p1.len == c1.len and p1.len == c2.len);

        const len = p1.len;
        var i: usize = 0;
        const Vec2 = @Vector(2, u64);

        while (i + 1 < len) : (i += 2) {
            const v_p1: Vec2 = p1[i..][0..2].*;
            const v_p2: Vec2 = p2[i..][0..2].*;
            const v_m: Vec2 = mask[i..][0..2].*;

            const mut1: Vec2 = .{ rng.sparseMask(depth), rng.sparseMask(depth) };
            const mut2: Vec2 = .{ rng.sparseMask(depth), rng.sparseMask(depth) };

            c1[i..][0..2].* = ((v_p1 & v_m) | (v_p2 & ~v_m)) ^ mut1;
            c2[i..][0..2].* = ((v_p2 & v_m) | (v_p1 & ~v_m)) ^ mut2;
        }

        while (i < len) : (i += 1) {
            const w1 = p1[i];
            const w2 = p2[i];
            const m = mask[i];
            const mut1 = rng.sparseMask(depth);
            const mut2 = rng.sparseMask(depth);
            c1[i] = ((w1 & m) | (w2 & ~m)) ^ mut1;
            c2[i] = ((w2 & m) | (w1 & ~m)) ^ mut2;
        }

        cleanMaskPadding(c1, bit_length);
        cleanMaskPadding(c2, bit_length);
    }

    pub fn mutate(words: []u64, mut_mask: []const u64, bit_length: usize) void {
        std.debug.assert(words.len == mut_mask.len);
        for (words, mut_mask) |*w, m| {
            w.* ^= m;
        }
        cleanMaskPadding(words, bit_length);
    }

    pub inline fn setMaskBit(mask: []u64, idx: usize) void {
        const w_idx = idx / 64;
        const b_idx: u6 = @intCast(idx % 64);
        mask[w_idx] |= (@as(u64, 1) << b_idx);
    }

    pub inline fn getMaskBit(mask: []const u64, idx: usize) u1 {
        const w_idx = idx / 64;
        const b_idx: u6 = @intCast(idx % 64);
        return @truncate((mask[w_idx] >> b_idx) & 1);
    }

    pub inline fn cleanMaskPadding(mask: []u64, bit_length: usize) void {
        const remainder = bit_length % 64;
        if (remainder != 0 and mask.len > 0) {
            const pad_mask = (@as(u64, 1) << @intCast(remainder)) - 1;
            mask[mask.len - 1] &= pad_mask;
        }
    }
};
