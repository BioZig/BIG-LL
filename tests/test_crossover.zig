const std = @import("std");
const big_ll = @import("big_ll");
const types = big_ll.types;
const linkage = big_ll.linkage;
const crossover = big_ll.crossover;
const prng = big_ll.prng;

test "Recombination multiplexing logic" {
    const L = 64;
    const Chrom = types.Chromosome(L);
    const Cross = crossover.CrossoverEngine(L);

    var p1 = Chrom.initZero();
    var p2 = Chrom.initZero();
    p1.words[0] = 0xAAAAAAAAAAAAAAAA;
    p2.words[0] = 0x5555555555555555;

    var mask = [_]u64{0xFF00FF00FF00FF00};
    const offspring = Cross.recombine(&p1, &p2, &mask);

    for (0..64) |i| {
        const m_bit = (mask[0] >> @intCast(i)) & 1;
        const c1_bit = offspring.c1.getBit(i);
        const c2_bit = offspring.c2.getBit(i);

        if (m_bit == 1) {
            try std.testing.expectEqual(p1.getBit(i), c1_bit);
            try std.testing.expectEqual(p2.getBit(i), c2_bit);
        } else {
            try std.testing.expectEqual(p2.getBit(i), c1_bit);
            try std.testing.expectEqual(p1.getBit(i), c2_bit);
        }
    }
}

test "Cluster ceiling adherence" {
    const allocator = std.testing.allocator;
    const L = 100;
    const Engine = linkage.LinkageEngine(L);
    var eng = try Engine.init(allocator);
    defer eng.deinit(allocator);

    for (1..L) |target| {
        try eng.addEdge(allocator, 0, @intCast(target));
    }

    const Cross = crossover.CrossoverEngine(L);
    var rng = prng.Xoshiro256.init(99);
    var mask = [_]u64{0} ** ((L + 63) / 64);

    Cross.generateMask(&eng, &rng, &mask);

    var count: usize = 0;
    for (mask) |w| {
        count += @popCount(w);
    }

    try std.testing.expect(count <= Cross.cluster_ceiling);
    try std.testing.expect(count >= 1);
}

test "Bitwise mutation" {
    const L = 64;
    const Chrom = types.Chromosome(L);
    const Cross = crossover.CrossoverEngine(L);

    var c = Chrom.initZero();
    const mut_mask = [_]u64{0x0000000000000001};

    Cross.mutate(&c, &mut_mask);
    try std.testing.expectEqual(@as(u1, 1), c.getBit(0));
    try std.testing.expectEqual(@as(usize, 1), c.popCount());

    Cross.mutate(&c, &mut_mask);
    try std.testing.expectEqual(@as(u1, 0), c.getBit(0));
    try std.testing.expectEqual(@as(usize, 0), c.popCount());
}
