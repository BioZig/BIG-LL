const std = @import("std");
const big_ll = @import("big_ll");
const ParetoChromosome = big_ll.ParetoChromosome;
const ParetoEngine = big_ll.ParetoEngine;
const dominates = big_ll.dominates;

test "Pareto dominance operator" {
    const a = [_]f64{ 5.0, 10.0 };
    const b = [_]f64{ 4.0, 10.0 };
    const c = [_]f64{ 5.0, 10.0 };
    const d = [_]f64{ 4.0, 12.0 };

    try std.testing.expect(dominates(2, a, b));
    try std.testing.expect(!dominates(2, b, a));
    try std.testing.expect(!dominates(2, a, c));
    try std.testing.expect(!dominates(2, a, d));
    try std.testing.expect(!dominates(2, d, a));
}

test "Non-dominated ranking partitioning" {
    const allocator = std.testing.allocator;
    const Chrom = ParetoChromosome(64, 2);

    var pop = try allocator.alloc(Chrom, 4);
    defer allocator.free(pop);

    pop[0] = Chrom.initZero();
    pop[0].objectives = [_]f64{ 10.0, 1.0 }; // Front 0

    pop[1] = Chrom.initZero();
    pop[1].objectives = [_]f64{ 1.0, 10.0 }; // Front 0

    pop[2] = Chrom.initZero();
    pop[2].objectives = [_]f64{ 5.0, 5.0 }; // Front 0

    pop[3] = Chrom.initZero();
    pop[3].objectives = [_]f64{ 2.0, 2.0 }; // Front 1 (dominated by pop[2])

    try big_ll.pareto.computeNonDominatedRanks(64, 2, pop, allocator);

    try std.testing.expectEqual(@as(usize, 0), pop[0].rank);
    try std.testing.expectEqual(@as(usize, 0), pop[1].rank);
    try std.testing.expectEqual(@as(usize, 0), pop[2].rank);
    try std.testing.expectEqual(@as(usize, 1), pop[3].rank);
}

test "Crowding distance assignment" {
    const allocator = std.testing.allocator;
    const Chrom = ParetoChromosome(64, 2);

    var pop = try allocator.alloc(Chrom, 3);
    defer allocator.free(pop);

    pop[0] = Chrom.initZero();
    pop[0].objectives = [_]f64{ 1.0, 10.0 };
    pop[0].rank = 0;

    pop[1] = Chrom.initZero();
    pop[1].objectives = [_]f64{ 5.0, 5.0 };
    pop[1].rank = 0;

    pop[2] = Chrom.initZero();
    pop[2].objectives = [_]f64{ 10.0, 1.0 };
    pop[2].rank = 0;

    try big_ll.pareto.computeCrowdingDistances(64, 2, pop, allocator);

    try std.testing.expect(std.math.isInf(pop[0].crowding_distance));
    try std.testing.expect(std.math.isInf(pop[2].crowding_distance));
    try std.testing.expect(!std.math.isInf(pop[1].crowding_distance));
    try std.testing.expect(pop[1].crowding_distance > 0.0);
}

const BiObjectiveMockAdapter = struct {
    pub fn evaluate(self: *const @This(), chrom: *const ParetoChromosome(64, 2)) [2]f64 {
        _ = self;
        const ones = @as(f64, @floatFromInt(chrom.popCount()));
        const zeros = 64.0 - ones;
        return [_]f64{ ones, zeros };
    }
};

test "ParetoEngine generational optimization" {
    const allocator = std.testing.allocator;
    const adapter = BiObjectiveMockAdapter{};

    var engine = try ParetoEngine(64, 2, BiObjectiveMockAdapter).init(allocator, 32, adapter, 12345);
    defer engine.deinit();

    for (0..15) |_| {
        try engine.step();
    }

    try std.testing.expectEqual(@as(usize, 15), engine.generation);
    try std.testing.expect(engine.countFrontZero() > 0);
}
