const std = @import("std");
const big_ll = @import("big_ll");
const types = big_ll.types;
const BlockChromosome = types.BlockChromosome;

test "BlockChromosome 512-bit cacheline block alignment and size invariants" {
    const Chrom1 = BlockChromosome(1);
    try std.testing.expectEqual(@as(usize, 1), Chrom1.block_count);
    try std.testing.expectEqual(@as(usize, 8), Chrom1.word_count);
    try std.testing.expectEqual(@as(usize, 512), Chrom1.bit_length);

    // Verify 64-byte cacheline alignment
    try std.testing.expectEqual(@as(usize, 64), @alignOf(Chrom1));

    var chrom = Chrom1.initZero();
    try std.testing.expectEqual(@as(usize, 0), chrom.popCount());

    chrom.setBit(0, 1);
    chrom.setBit(511, 1);
    chrom.setBit(255, 1);
    try std.testing.expectEqual(@as(u1, 1), chrom.getBit(0));
    try std.testing.expectEqual(@as(u1, 1), chrom.getBit(511));
    try std.testing.expectEqual(@as(u1, 1), chrom.getBit(255));
    try std.testing.expectEqual(@as(u1, 0), chrom.getBit(1));
    try std.testing.expectEqual(@as(usize, 3), chrom.popCount());

    chrom.toggleBit(0);
    try std.testing.expectEqual(@as(u1, 0), chrom.getBit(0));
    try std.testing.expectEqual(@as(usize, 2), chrom.popCount());
}

test "BlockChromosome 20,480-bit scaling (40 x 512-bit cacheline blocks)" {
    const num_blocks: usize = 40;
    const Chrom20k = BlockChromosome(num_blocks);

    try std.testing.expectEqual(@as(usize, 40), Chrom20k.block_count);
    try std.testing.expectEqual(@as(usize, 320), Chrom20k.word_count);
    try std.testing.expectEqual(@as(usize, 20480), Chrom20k.bit_length);

    // Verify exact memory footprint: 40 blocks * 64 bytes = 2560 bytes (+ 16 bytes metadata = 2576 -> align 64 = 2624 bytes)
    const chrom_size = @sizeOf(Chrom20k);
    try std.testing.expect(chrom_size >= 2560 and chrom_size <= 2624);

    // Verify Cache-Resident Invariant: 1024 individuals * ~2.6 KB ≈ 2.6 MB (fits 100% in L2/L3 cache)
    const pop_bytes = 1024 * chrom_size;
    try std.testing.expect(pop_bytes < 3 * 1024 * 1024); // strictly under 3 MB

    var chrom1 = Chrom20k.initZero();
    var chrom2 = Chrom20k.initZero();

    // Set bits across multiple 512-bit blocks
    chrom1.setBit(0, 1);
    chrom1.setBit(512 * 5 + 10, 1); // block 5
    chrom1.setBit(512 * 39 + 511, 1); // last bit in block 39 (bit 20,479)
    try std.testing.expectEqual(@as(usize, 3), chrom1.popCount());

    chrom2.setBit(0, 1);
    chrom2.setBit(512 * 10 + 20, 1); // block 10
    try std.testing.expectEqual(@as(usize, 2), chrom2.popCount());

    const dist = chrom1.hammingDistance(&chrom2);
    try std.testing.expectEqual(@as(usize, 3), dist);

    // Verify block crossover
    var child = Chrom20k.initZero();
    const mask = [_]u64{0xFFFFFFFFFFFFFFFF} ** 8; // select p1
    child.crossoverBlock(&chrom1, &chrom2, mask, 5);
    try std.testing.expectEqual(@as(u1, 1), child.getBit(512 * 5 + 10));

    // Verify active-bit mask for 20,248 bits
    child.setBit(20479, 1); // beyond 20248
    try std.testing.expectEqual(@as(u1, 1), child.getBit(20479));
    child.cleanPadding(20248);
    try std.testing.expectEqual(@as(u1, 0), child.getBit(20479)); // properly masked to 0
}
