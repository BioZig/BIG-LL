const std = @import("std");
const big_ll = @import("big_ll");

const DynamicChromosome = big_ll.DynamicChromosome;
const DynamicPopulation = big_ll.DynamicPopulation;
const DynamicLinkageEngine = big_ll.DynamicLinkageEngine;
const DynamicCrossoverEngine = big_ll.DynamicCrossoverEngine;
const DynamicGeneticEngine = big_ll.DynamicGeneticEngine;

test "DynamicPopulation pitch and 64-byte alignment" {
    const allocator = std.testing.allocator;

    // Test small L = 64 (pitch = 1 word = 8 bytes)
    var pop64 = try DynamicPopulation.init(allocator, 1024, 64);
    defer pop64.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), pop64.pitch);
    try std.testing.expectEqual(@as(usize, 1024), pop64.pop_size);
    try std.testing.expectEqual(@as(usize, 64), pop64.bit_length);
    try std.testing.expectEqual(@as(usize, 1024), pop64.bitboard.len);

    // Verify 64-byte alignment
    const addr = @intFromPtr(pop64.bitboard.ptr);
    try std.testing.expectEqual(@as(usize, 0), addr % 64);

    // Test irregular L = 350 (pitch = (350+63)/64 = 6 words)
    var pop350 = try DynamicPopulation.init(allocator, 128, 350);
    defer pop350.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 6), pop350.pitch);
    try std.testing.expectEqual(@as(usize, 128 * 6), pop350.bitboard.len);
}

test "DynamicChromosome bit manipulation and operations" {
    const allocator = std.testing.allocator;
    var pop = try DynamicPopulation.init(allocator, 10, 150);
    defer pop.deinit(allocator);

    var chrom = pop.get(0);
    try std.testing.expectEqual(@as(usize, 150), chrom.bit_length);
    try std.testing.expectEqual(@as(usize, 0), chrom.popCount());

    chrom.setBit(0, 1);
    chrom.setBit(63, 1);
    chrom.setBit(64, 1);
    chrom.setBit(149, 1);

    try std.testing.expectEqual(@as(u1, 1), chrom.getBit(0));
    try std.testing.expectEqual(@as(u1, 1), chrom.getBit(63));
    try std.testing.expectEqual(@as(u1, 1), chrom.getBit(64));
    try std.testing.expectEqual(@as(u1, 1), chrom.getBit(149));
    try std.testing.expectEqual(@as(u1, 0), chrom.getBit(1));
    try std.testing.expectEqual(@as(usize, 4), chrom.popCount());

    chrom.toggleBit(0);
    try std.testing.expectEqual(@as(u1, 0), chrom.getBit(0));
    try std.testing.expectEqual(@as(usize, 3), chrom.popCount());

    // Clean padding test
    chrom.words[2] = 0xFFFFFFFFFFFFFFFF;
    chrom.cleanPadding();
    // 150 % 64 = 22 bits active in word 2. Mask is (1 << 22) - 1 = 0x3FFFFF
    try std.testing.expectEqual(@as(u64, 0x00000000003FFFFF), chrom.words[2]);
}

test "DynamicPopulation Shannon entropy" {
    const allocator = std.testing.allocator;
    var pop = try DynamicPopulation.init(allocator, 100, 64);
    defer pop.deinit(allocator);

    // Initial population all zeros -> entropy 0.0
    pop.updateMetrics();
    try std.testing.expectEqual(@as(f64, 0.0), pop.entropy);

    // Set exactly 50 individuals to 1 for all bits -> max entropy 1.0
    for (0..50) |i| {
        pop.getWordsMut(i)[0] = 0xFFFFFFFFFFFFFFFF;
    }
    pop.updateMetrics();
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), pop.entropy, 1e-6);
}

test "DynamicGeneticEngine OneMax dynamic-length optimization" {
    const allocator = std.testing.allocator;

    const OneMaxAdapter = struct {
        bit_len: usize,

        const Decoded = struct {
            count: usize,
        };

        pub fn decode(self: @This(), chrom: *const DynamicChromosome) Decoded {
            _ = self;
            return .{ .count = chrom.popCount() };
        }

        pub fn evaluate(self: @This(), dec: Decoded) f64 {
            _ = self;
            return @as(f64, @floatFromInt(dec.count));
        }
    };

    // Test with arbitrary user dimension L = 77 (not a power of 2, spans 2 words)
    const test_l: usize = 77;
    const adapter = OneMaxAdapter{ .bit_len = test_l };

    var engine = try DynamicGeneticEngine(OneMaxAdapter).init(
        allocator,
        test_l,
        128, // pop size
        adapter,
        42,
        .{},
    );
    defer engine.deinit();

    const initial_best = engine.getBestFitness();

    // Step 25 generations
    for (0..25) |_| {
        try engine.step();
    }

    const final_best = engine.getBestFitness();

    // Genetic linkage learning must improve fitness over generations
    try std.testing.expect(final_best >= initial_best);
    try std.testing.expect(final_best > 50.0); // Solves over 65% of ones in 25 gens
}
