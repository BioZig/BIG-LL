const std = @import("std");
const big_ll = @import("big_ll");
const types = big_ll.types;

test "Chromosome bit manipulation and invariants" {
    const Chrom64 = types.Chromosome(64);
    var c = Chrom64.initZero();

    try std.testing.expectEqual(@as(usize, 0), c.popCount());
    try std.testing.expectEqual(@as(u1, 0), c.getBit(0));
    try std.testing.expectEqual(@as(u1, 0), c.getBit(63));

    c.setBit(0, 1);
    c.setBit(63, 1);
    try std.testing.expectEqual(@as(u1, 1), c.getBit(0));
    try std.testing.expectEqual(@as(u1, 1), c.getBit(63));
    try std.testing.expectEqual(@as(usize, 2), c.popCount());

    c.toggleBit(0);
    try std.testing.expectEqual(@as(u1, 0), c.getBit(0));
    try std.testing.expectEqual(@as(usize, 1), c.popCount());
}

test "Chromosome multi-word and hamming distance" {
    const Chrom130 = types.Chromosome(130);
    var c1 = Chrom130.initZero();
    var c2 = Chrom130.initZero();

    try std.testing.expect(c1.bitwiseEqual(&c2));
    try std.testing.expectEqual(@as(usize, 0), c1.hammingDistance(&c2));

    c1.setBit(0, 1);
    c1.setBit(64, 1);
    c1.setBit(129, 1);

    c2.setBit(64, 1);

    try std.testing.expectEqual(@as(usize, 3), c1.popCount());
    try std.testing.expectEqual(@as(usize, 1), c2.popCount());
    try std.testing.expectEqual(@as(usize, 2), c1.hammingDistance(&c2));
    try std.testing.expect(!c1.bitwiseEqual(&c2));
}

test "Chromosome padding cleanup" {
    const Chrom70 = types.Chromosome(70);
    var c = Chrom70.initZero();
    c.words[1] = 0xFFFFFFFFFFFFFFFF;
    c.cleanPadding();

    const expected_mask = (@as(u64, 1) << 6) - 1;
    try std.testing.expectEqual(expected_mask, c.words[1]);
}

test "Population initialization and metrics" {
    const allocator = std.testing.allocator;
    const Pop = types.Population(64);
    var pop = try Pop.init(allocator, 10);
    defer pop.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 10), pop.individuals.len);

    for (pop.individuals, 0..) |*indiv, i| {
        indiv.fitness = @as(f64, @floatFromInt(i * 10));
        if (i % 2 == 0) {
            indiv.setBit(0, 1);
        }
    }

    pop.updateMetrics();
    try std.testing.expectEqual(@as(usize, 9), pop.best_idx);
    try std.testing.expectEqual(@as(f64, 90.0), pop.best_fitness);
    try std.testing.expectEqual(@as(f64, 45.0), pop.mean_fitness);
    try std.testing.expect(pop.entropy >= 0.0);
}
