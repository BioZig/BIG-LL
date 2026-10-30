const std = @import("std");
const big_ll = @import("big_ll");
const GeneticEngine = big_ll.GeneticEngine;
const generator = big_ll.data_generator;
const loader = big_ll.data_loader;

test "End-to-End Synthetic Logistics Domain Evolution" {
    const allocator = std.testing.allocator;

    const buf = try generator.generateLogistics(allocator, 32, 35.0, 50.0, 50.0);
    defer allocator.free(buf);

    var inst = try loader.StaticBufferInstance.init(buf);
    const domain = try big_ll.LogisticsDomain.initFromInstance(&inst);

    var engine = try GeneticEngine(big_ll.LogisticsDomain.CHROM_LEN, big_ll.LogisticsDomain).init(allocator, 64, domain, 1111);
    defer engine.deinit();

    const initial_best = engine.getBest().fitness;
    for (0..25) |_| {
        try engine.step();
    }

    const final_best = engine.getBest();
    try std.testing.expect(final_best.fitness >= initial_best);

    // Verify domain invariant: route decoded must be a valid permutation
    const route = domain.decode(&final_best);
    var visited = [_]bool{false} ** big_ll.LogisticsDomain.NUM_CUSTOMERS;
    for (route) |cust| {
        try std.testing.expect(cust < big_ll.LogisticsDomain.NUM_CUSTOMERS);
        try std.testing.expect(!visited[cust]);
        visited[cust] = true;
    }
}

test "End-to-End Synthetic Power Grid Domain Evolution" {
    const allocator = std.testing.allocator;

    const buf = try generator.generatePowerGrid(allocator, 550.0, 630.0);
    defer allocator.free(buf);

    var inst = try loader.StaticBufferInstance.init(buf);
    const domain = try big_ll.PowerGridDomain.initFromInstance(&inst);

    var engine = try GeneticEngine(big_ll.PowerGridDomain.CHROM_LEN, big_ll.PowerGridDomain).init(allocator, 64, domain, 2222);
    defer engine.deinit();

    const initial_best = engine.getBest().fitness;
    for (0..25) |_| {
        try engine.step();
    }

    const final_best = engine.getBest();
    try std.testing.expect(final_best.fitness >= initial_best);

    // Verify domain invariant: generation state decoded correctly
    const state = domain.decode(&final_best);
    var total_active: usize = 0;
    for (0..big_ll.PowerGridDomain.NUM_GENERATORS) |i| {
        if (state.switches[i] == 1) {
            total_active += 1;
            try std.testing.expect(state.scalars[i] >= 10.0 and state.scalars[i] <= 150.0);
        }
    }
}

test "End-to-End Synthetic Finance Portfolio Evolution" {
    const allocator = std.testing.allocator;

    const buf = try generator.generateFinance(allocator, 6, 8, 0.03);
    defer allocator.free(buf);

    var inst = try loader.StaticBufferInstance.init(buf);
    const domain = try big_ll.FinanceDomain.initFromInstance(&inst);

    var engine = try GeneticEngine(big_ll.FinanceDomain.CHROM_LEN, big_ll.FinanceDomain).init(allocator, 64, domain, 3333);
    defer engine.deinit();

    const initial_best = engine.getBest().fitness;
    for (0..25) |_| {
        try engine.step();
    }

    const final_best = engine.getBest();
    try std.testing.expect(final_best.fitness >= initial_best);

    // Verify domain invariant: weights positive
    const port = domain.decode(&final_best);
    var sum_w: f64 = 0.0;
    for (port.weights) |w| {
        try std.testing.expect(w >= 0.0 and w <= 1.0);
        sum_w += w;
    }
    try std.testing.expect(sum_w > 0.0);
}

test "End-to-End Synthetic Bioinformatics Biomarker Selection Evolution" {
    const allocator = std.testing.allocator;

    const buf = try generator.generateBioinformatics(allocator, 128, 8, 12);
    defer allocator.free(buf);

    var inst = try loader.StaticBufferInstance.init(buf);
    const domain = try big_ll.BioinformaticsDomain.initFromInstance(&inst);

    var engine = try GeneticEngine(big_ll.BioinformaticsDomain.CHROM_LEN, big_ll.BioinformaticsDomain).init(allocator, 64, domain, 4444);
    defer engine.deinit();

    const initial_best = engine.getBest().fitness;
    for (0..25) |_| {
        try engine.step();
    }

    const final_best = engine.getBest();
    try std.testing.expect(final_best.fitness >= initial_best);

    // Verify domain invariant: panel indices within [0, NUM_SNPS)
    const panel = domain.decode(&final_best);
    for (panel.indices[0..panel.count]) |idx| {
        try std.testing.expect(idx < big_ll.BioinformaticsDomain.NUM_SNPS);
    }
}

test "End-to-End Synthetic Airfoil Geometry Evolution" {
    const allocator = std.testing.allocator;

    const buf = try generator.generateAirfoil(allocator, 16, 2.5, 0.01);
    defer allocator.free(buf);

    var inst = try loader.StaticBufferInstance.init(buf);
    const domain = try big_ll.AirfoilDomain.initFromInstance(&inst);

    var engine = try GeneticEngine(big_ll.AirfoilDomain.CHROM_LEN, big_ll.AirfoilDomain).init(allocator, 64, domain, 5555);
    defer engine.deinit();

    const initial_best = engine.getBest().fitness;
    for (0..25) |_| {
        try engine.step();
    }

    const final_best = engine.getBest();
    try std.testing.expect(final_best.fitness >= initial_best);

    // Verify domain invariant: geometry decoded within expected ranges
    const geom = domain.decode(&final_best);
    for (0..8) |i| {
        try std.testing.expect(geom.upper_y[i] >= 0.0 and geom.upper_y[i] <= 0.25);
        try std.testing.expect(geom.lower_y[i] >= -0.20 and geom.lower_y[i] <= 0.0);
    }
}

test "End-to-End Synthetic Telecom Channel Allocation Evolution" {
    const allocator = std.testing.allocator;

    const buf = try generator.generateTelecom(allocator, 16, 4, 0.05);
    defer allocator.free(buf);

    var inst = try loader.StaticBufferInstance.init(buf);
    const domain = try big_ll.TelecomDomain.initFromInstance(&inst);

    var engine = try GeneticEngine(big_ll.TelecomDomain.CHROM_LEN, big_ll.TelecomDomain).init(allocator, 64, domain, 6666);
    defer engine.deinit();

    const initial_best = engine.getBest().fitness;
    for (0..25) |_| {
        try engine.step();
    }

    const final_best = engine.getBest();
    try std.testing.expect(final_best.fitness >= initial_best);

    // Verify domain invariant: channels in [0, NUM_CHANNELS) and powers in [0.1, 2.0]
    const alloc_state = domain.decode(&final_best);
    for (0..big_ll.TelecomDomain.NUM_USERS) |i| {
        try std.testing.expect(alloc_state.channels[i] < big_ll.TelecomDomain.NUM_CHANNELS);
        try std.testing.expect(alloc_state.powers[i] >= 0.1 and alloc_state.powers[i] <= 2.0);
    }
}
