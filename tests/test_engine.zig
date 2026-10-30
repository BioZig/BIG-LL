const std = @import("std");
const big_ll = @import("big_ll");
const GeneticEngine = big_ll.GeneticEngine;
const Chromosome = big_ll.Chromosome;

const MockOneMaxAdapter = struct {
    const Self = @This();

    pub fn decode(self: *const Self, chrom: *const Chromosome(64)) usize {
        _ = self;
        return chrom.popCount();
    }

    pub fn evaluate(self: *const Self, ones: usize) f64 {
        _ = self;
        return @as(f64, @floatFromInt(ones));
    }
};

test "GeneticEngine initialization and initial evaluation" {
    const allocator = std.testing.allocator;
    const adapter = MockOneMaxAdapter{};
    var engine = try GeneticEngine(64, MockOneMaxAdapter).init(allocator, 64, adapter, 12345);
    defer engine.deinit();

    try std.testing.expectEqual(@as(usize, 0), engine.generation);
    try std.testing.expectEqual(@as(usize, 64), engine.pop_current.individuals.len);
    try std.testing.expect(engine.pop_current.best_fitness >= 0.0);
    try std.testing.expect(engine.pop_current.best_fitness <= 64.0);
    try std.testing.expect(engine.pop_current.entropy >= 0.0);
}

test "GeneticEngine generational step and elitism preservation" {
    const allocator = std.testing.allocator;
    const adapter = MockOneMaxAdapter{};
    var engine = try GeneticEngine(64, MockOneMaxAdapter).init(allocator, 64, adapter, 54321);
    defer engine.deinit();

    const initial_best = engine.getBest().fitness;
    try engine.step();

    try std.testing.expectEqual(@as(usize, 1), engine.generation);
    const step1_best = engine.getBest().fitness;
    try std.testing.expect(step1_best >= initial_best);
}

test "GeneticEngine multi-step optimization and deduplication" {
    const allocator = std.testing.allocator;
    const adapter = MockOneMaxAdapter{};
    var engine = try big_ll.GeneticEngineCustom(64, MockOneMaxAdapter, .{ .enable_dedup = true }).init(allocator, 64, adapter, 99999);
    defer engine.deinit();

    const start_best = engine.getBest().fitness;
    var gen: usize = 0;
    while (gen < 25) : (gen += 1) {
        try engine.step();
    }

    const end_best = engine.getBest().fitness;
    try std.testing.expect(end_best >= start_best);
    try std.testing.expect(engine.dedup_cache.hits > 0 or engine.dedup_cache.misses > 0);
}

test "GeneticEngine zero-allocation multi-threaded parallelization" {
    const allocator = std.testing.allocator;
    const adapter = MockOneMaxAdapter{};
    var engine = try big_ll.GeneticEngineCustom(64, MockOneMaxAdapter, .{ .enable_dedup = false, .num_threads = 4 }).init(allocator, 64, adapter, 4444);
    defer engine.deinit();

    const start_best = engine.getBest().fitness;
    var gen: usize = 0;
    while (gen < 20) : (gen += 1) {
        try engine.step();
    }

    const end_best = engine.getBest().fitness;
    try std.testing.expect(end_best >= start_best);
    try std.testing.expectEqual(@as(usize, 20), engine.generation);
}
