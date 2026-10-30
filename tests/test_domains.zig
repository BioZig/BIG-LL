const std = @import("std");
const big_ll = @import("big_ll");
const GeneticEngine = big_ll.GeneticEngine;

test "AirfoilDomain integration with GeneticEngine" {
    const allocator = std.testing.allocator;
    const domain = big_ll.AirfoilDomain{};
    var engine = try GeneticEngine(big_ll.AirfoilDomain.CHROM_LEN, big_ll.AirfoilDomain).init(allocator, 64, domain, 42);
    defer engine.deinit();

    const initial_best = engine.getBest().fitness;
    try engine.step();
    try engine.step();
    try engine.step();

    try std.testing.expect(engine.getBest().fitness >= initial_best);
}

test "LogisticsDomain CVRP integration with GeneticEngine" {
    const allocator = std.testing.allocator;
    const domain = big_ll.LogisticsDomain.initDefault();
    var engine = try GeneticEngine(big_ll.LogisticsDomain.CHROM_LEN, big_ll.LogisticsDomain).init(allocator, 64, domain, 101);
    defer engine.deinit();

    const initial_best = engine.getBest().fitness;
    try engine.step();
    try engine.step();
    try engine.step();

    try std.testing.expect(engine.getBest().fitness >= initial_best);
}

test "LogisticsDomain Prins Split optimality vs Greedy" {
    const domain = big_ll.LogisticsDomain.initDefault();
    var route: [big_ll.LogisticsDomain.NUM_CUSTOMERS]u32 = undefined;
    for (0..big_ll.LogisticsDomain.NUM_CUSTOMERS) |i| {
        route[i] = @intCast(i);
    }

    const split_dist = domain.splitTour(route);
    const greedy_fit = domain.evaluateGreedy(route);
    const greedy_dist = (10000.0 / greedy_fit) - 1.0;

    // Mathematical invariant: Prins' optimal DAG split must be <= greedy partition distance
    try std.testing.expect(split_dist <= greedy_dist + 1e-6);
}

test "LogisticsDomain 2-opt tour polish permutation invariant" {
    const domain = big_ll.LogisticsDomain.initDefault();
    var route: [big_ll.LogisticsDomain.NUM_CUSTOMERS]u32 = undefined;
    for (0..big_ll.LogisticsDomain.NUM_CUSTOMERS) |i| {
        route[i] = @intCast(i);
    }

    domain.polishTour(&route);

    // Permutation invariant test: every customer 0..31 visited exactly once
    var visited = [_]bool{false} ** big_ll.LogisticsDomain.NUM_CUSTOMERS;
    for (route) |c| {
        try std.testing.expect(c < big_ll.LogisticsDomain.NUM_CUSTOMERS);
        try std.testing.expect(!visited[c]);
        visited[c] = true;
    }
}

test "PowerGridDomain Unit Commitment integration with GeneticEngine" {
    const allocator = std.testing.allocator;
    const domain = big_ll.PowerGridDomain.initDefault();
    var engine = try GeneticEngine(big_ll.PowerGridDomain.CHROM_LEN, big_ll.PowerGridDomain).init(allocator, 64, domain, 202);
    defer engine.deinit();

    const initial_best = engine.getBest().fitness;
    try engine.step();
    try engine.step();
    try engine.step();

    try std.testing.expect(engine.getBest().fitness >= initial_best);
}

test "PowerGridDomain projectPowerBalance reduces mismatch" {
    const domain = big_ll.PowerGridDomain.initDefault();
    var state = big_ll.PowerGridDomain.DecodedState{
        .switches = [_]u1{ 1, 1, 1, 1, 0, 0, 0, 0 },
        .scalars = [_]f64{ 50.0, 50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 20.0 }, // Total = 200 MW, demand = 550 MW
    };

    domain.projectPowerBalance(&state);

    var new_gen: f64 = 0.0;
    for (0..big_ll.PowerGridDomain.NUM_GENERATORS) |i| {
        if (state.switches[i] == 1) {
            new_gen += state.scalars[i];
            try std.testing.expect(state.scalars[i] >= domain.specs[i].p_min);
            try std.testing.expect(state.scalars[i] <= domain.specs[i].p_max);
        }
    }
    // Power balance must have shifted generation towards 550 MW target
    try std.testing.expect(new_gen > 200.0);
}

test "BioinformaticsDomain biomarker selection integration with GeneticEngine" {
    const allocator = std.testing.allocator;
    const domain = big_ll.BioinformaticsDomain{};
    var engine = try GeneticEngine(big_ll.BioinformaticsDomain.CHROM_LEN, big_ll.BioinformaticsDomain).init(allocator, 64, domain, 303);
    defer engine.deinit();

    const initial_best = engine.getBest().fitness;
    try engine.step();
    try engine.step();
    try engine.step();

    try std.testing.expect(engine.getBest().fitness >= initial_best);
}

test "FinanceDomain portfolio optimization integration with GeneticEngine" {
    const allocator = std.testing.allocator;
    const domain = big_ll.FinanceDomain.initDefault();
    var engine = try GeneticEngine(big_ll.FinanceDomain.CHROM_LEN, big_ll.FinanceDomain).init(allocator, 64, domain, 404);
    defer engine.deinit();

    const initial_best = engine.getBest().fitness;
    try engine.step();
    try engine.step();
    try engine.step();

    try std.testing.expect(engine.getBest().fitness >= initial_best);
}

test "TelecomDomain MIMO channel assignment integration with GeneticEngine" {
    const allocator = std.testing.allocator;
    const domain = big_ll.TelecomDomain.initDefault();
    var engine = try GeneticEngine(big_ll.TelecomDomain.CHROM_LEN, big_ll.TelecomDomain).init(allocator, 64, domain, 505);
    defer engine.deinit();

    const initial_best = engine.getBest().fitness;
    try engine.step();
    try engine.step();
    try engine.step();

    try std.testing.expect(engine.getBest().fitness >= initial_best);
}
