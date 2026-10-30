const std = @import("std");
const big_ll = @import("big_ll");
const IslandEngine = big_ll.IslandEngine;
const Chromosome = big_ll.Chromosome;

const MockAdapter = struct {
    pub fn decode(self: *const MockAdapter, chrom: *const Chromosome(64)) usize {
        _ = self;
        return chrom.popCount();
    }

    pub fn evaluate(self: *const MockAdapter, ones: usize) f64 {
        _ = self;
        return @as(f64, @floatFromInt(ones));
    }
};

test "IslandEngine initialization and sub-population allocation" {
    const allocator = std.testing.allocator;
    const adapter = MockAdapter{};
    var engine = try IslandEngine(64, MockAdapter, 4, 10).init(allocator, 256, adapter, 12345);
    defer engine.deinit();

    try std.testing.expectEqual(@as(usize, 0), engine.generation);
    try std.testing.expectEqual(@as(usize, 64), engine.sub_pop_size);
    try std.testing.expectEqual(@as(usize, 256), engine.total_pop_size);
    try std.testing.expect(engine.getGlobalBest().fitness >= 0.0);
    try std.testing.expect(engine.getGlobalBest().fitness <= 64.0);
}

test "IslandEngine stepEpoch parallel execution and ring migration" {
    const allocator = std.testing.allocator;
    const adapter = MockAdapter{};
    var engine = try IslandEngine(64, MockAdapter, 4, 5).init(allocator, 256, adapter, 54321);
    defer engine.deinit();

    const start_fit = engine.getGlobalBest().fitness;
    try engine.stepEpoch();

    try std.testing.expectEqual(@as(usize, 5), engine.generation);
    const end_fit = engine.getGlobalBest().fitness;
    try std.testing.expect(end_fit >= start_fit);
}

test "IslandEngine multi-epoch convergence" {
    const allocator = std.testing.allocator;
    const adapter = MockAdapter{};
    var engine = try IslandEngine(64, MockAdapter, 4, 10).init(allocator, 256, adapter, 99999);
    defer engine.deinit();

    const initial_fit = engine.getGlobalBest().fitness;
    for (0..3) |_| {
        try engine.stepEpoch();
    }

    try std.testing.expectEqual(@as(usize, 30), engine.generation);
    const final_fit = engine.getGlobalBest().fitness;
    try std.testing.expect(final_fit >= initial_fit);
    try std.testing.expect(engine.getGlobalEntropy() > 0.0);
}
