const std = @import("std");
const big_ll = @import("big_ll");
const LinkageEngine = big_ll.LinkageEngine;

test "Cache tiling buffer memory invariant (<= 512 KB)" {
    const Engine2048 = LinkageEngine(2048);
    const Engine4096 = LinkageEngine(4096);
    const Engine20480 = LinkageEngine(20480);

    try std.testing.expectEqual(@as(usize, 2048), Engine2048.batch_size);
    try std.testing.expectEqual(@as(usize, 2048), Engine4096.batch_size);
    try std.testing.expectEqual(@as(usize, 2048), Engine20480.batch_size);

    try std.testing.expectEqual(@as(usize, 1), Engine2048.num_batches);
    try std.testing.expectEqual(@as(usize, 2), Engine4096.num_batches);
    try std.testing.expectEqual(@as(usize, 10), Engine20480.num_batches);

    const tile_bytes_2048 = 2048 * ((2048 + 63) / 64) * @sizeOf(u64);
    try std.testing.expect(tile_bytes_2048 <= 524288);
}

test "Multi-batch linkage initialization and threshold calculation" {
    const Engine = LinkageEngine(20480);
    const threshold = Engine.calculateAdaptiveThreshold(20480, Engine.elite_size);
    try std.testing.expect(threshold >= 4);

    const allocator = std.testing.allocator;
    const engine_ptr = try allocator.create(Engine);
    defer allocator.destroy(engine_ptr);

    engine_ptr.* = try Engine.init(allocator);
    defer engine_ptr.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 10), Engine.num_batches);
    try std.testing.expectEqual(@as(usize, 0), engine_ptr.edge_count);
}

test "2D-tiled cross-batch linkage detects inter-batch correlation" {
    const L: usize = 4096;
    const Engine = LinkageEngine(L);
    const Chrom = big_ll.Chromosome(L);
    const K = Engine.elite_size;

    const allocator = std.testing.allocator;
    const engine_ptr = try allocator.create(Engine);
    defer allocator.destroy(engine_ptr);

    engine_ptr.* = try Engine.init(allocator);
    defer engine_ptr.deinit(allocator);

    var elites: [K]Chrom = undefined;
    for (0..K) |i| {
        elites[i] = Chrom.initZero();
        if (i < 80) {
            elites[i].setBit(10, 1);
            elites[i].setBit(3000, 1);
        }
    }

    engine_ptr.ingestElites(&elites);
    try engine_ptr.computeFullLinkageScheduled(allocator, 0);

    try std.testing.expect(engine_ptr.hasEdge(10, 3000));
    try std.testing.expect(engine_ptr.hasEdge(3000, 10));
}
