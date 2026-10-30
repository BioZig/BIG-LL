const std = @import("std");
const big_ll = @import("big_ll");
const GeneticEngine = big_ll.GeneticEngine;
const Chromosome = big_ll.Chromosome;

pub const DeceptiveTrapAdapter = struct {
    pub const BLOCK_SIZE: usize = 5;
    pub const NUM_BLOCKS: usize = 8;
    pub const CHROM_LEN: usize = 64;
    pub const Chrom = Chromosome(CHROM_LEN);

    pub fn decode(self: *const DeceptiveTrapAdapter, chrom: *const Chrom) *const Chrom {
        _ = self;
        return chrom;
    }

    pub fn evaluateTrapBlock(u: usize) f64 {
        if (u == BLOCK_SIZE) {
            return 5.0;
        } else {
            return 4.0 - @as(f64, @floatFromInt(u));
        }
    }

    pub fn evaluate(self: *const DeceptiveTrapAdapter, chrom: *const Chrom) f64 {
        _ = self;
        var total: f64 = 0.0;

        for (0..NUM_BLOCKS) |b| {
            var ones: usize = 0;
            const start = b * BLOCK_SIZE;
            for (0..BLOCK_SIZE) |offset| {
                if (chrom.getBit(start + offset) == 1) {
                    ones += 1;
                }
            }
            total += evaluateTrapBlock(ones);
        }

        for (40..64) |i| {
            if (chrom.getBit(i) == 1) {
                total += 0.1;
            }
        }

        return total;
    }
};

test "Deceptive trap function properties" {
    try std.testing.expectEqual(@as(f64, 4.0), DeceptiveTrapAdapter.evaluateTrapBlock(0));
    try std.testing.expectEqual(@as(f64, 3.0), DeceptiveTrapAdapter.evaluateTrapBlock(1));
    try std.testing.expectEqual(@as(f64, 2.0), DeceptiveTrapAdapter.evaluateTrapBlock(2));
    try std.testing.expectEqual(@as(f64, 1.0), DeceptiveTrapAdapter.evaluateTrapBlock(3));
    try std.testing.expectEqual(@as(f64, 0.0), DeceptiveTrapAdapter.evaluateTrapBlock(4));
    try std.testing.expectEqual(@as(f64, 5.0), DeceptiveTrapAdapter.evaluateTrapBlock(5));
}

test "BIG-LL escape from deceptive attractor" {
    const allocator = std.testing.allocator;
    const adapter = DeceptiveTrapAdapter{};
    var engine = try GeneticEngine(64, DeceptiveTrapAdapter).init(allocator, 128, adapter, 777);
    defer engine.deinit();

    const start_fit = engine.getBest().fitness;

    for (0..40) |_| {
        try engine.step();
    }

    const final_fit = engine.getBest().fitness;
    try std.testing.expect(final_fit >= start_fit);
    try std.testing.expect(final_fit > 32.0);
}
