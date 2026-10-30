const std = @import("std");
const big_ll = @import("big_ll");
const types = big_ll.types;
const linkage = big_ll.linkage;
const prng = big_ll.prng;

test "Adaptive threshold scaling" {
    const thresh64 = linkage.LinkageEngine(64).calculateAdaptiveThreshold(64, 64);
    const thresh1000 = linkage.LinkageEngine(1000).calculateAdaptiveThreshold(1000, 64);
    const thresh20k = linkage.LinkageEngine(20000).calculateAdaptiveThreshold(20000, 128);

    try std.testing.expect(thresh64 >= 4);
    try std.testing.expect(thresh1000 > thresh64);
    try std.testing.expect(thresh20k >= thresh64);
}

test "In-register elite transposition correctness" {
    const allocator = std.testing.allocator;
    const L = 128;
    const Engine = linkage.LinkageEngine(L);
    var eng = try Engine.init(allocator);
    defer eng.deinit(allocator);

    const Chrom = types.Chromosome(L);
    var elites: [Engine.elite_size]Chrom = undefined;
    var rng = prng.Xoshiro256.init(1337);

    for (&elites) |*c| {
        c.* = Chrom.initZero();
        rng.fillWords(&c.words);
        c.cleanPadding();
    }

    eng.ingestElites(&elites);

    for (0..Engine.elite_size) |indiv_idx| {
        const w_slot = indiv_idx / 64;
        const b_slot: u6 = @intCast(indiv_idx % 64);

        for (0..L) |gene_idx| {
            const expected_bit = elites[indiv_idx].getBit(gene_idx);
            const transposed_word = eng.transposed_elites[gene_idx][w_slot];
            const actual_bit: u1 = @truncate((transposed_word >> b_slot) & 1);
            try std.testing.expectEqual(expected_bit, actual_bit);
        }
    }
}

test "Epistatic linkage discovery" {
    const allocator = std.testing.allocator;
    const L = 64;
    const Engine = linkage.LinkageEngine(L);
    var eng = try Engine.init(allocator);
    defer eng.deinit(allocator);

    const Chrom = types.Chromosome(L);
    var elites: [Engine.elite_size]Chrom = undefined;
    var rng = prng.Xoshiro256.init(42);

    for (&elites) |*c| {
        c.* = Chrom.initZero();
        rng.fillWords(&c.words);

        const coupled = rng.uintLessThan(2);
        c.setBit(5, @intCast(coupled));
        c.setBit(12, @intCast(coupled));
    }

    eng.ingestElites(&elites);
    try eng.computeFullLinkage(allocator);

    var found_coupling = false;
    var cursor = eng.adjacency_heads[5];
    while (cursor != std.math.maxInt(u32)) {
        if (eng.adjacency_targets.items[cursor] == 12) {
            found_coupling = true;
            break;
        }
        cursor = eng.adjacency_next.items[cursor];
    }

    try std.testing.expect(found_coupling);
}
