// Apple Silicon Dedicated Metal GPU Batch Acceleration Test Suite
const std = @import("std");
const big_ll = @import("big_ll");
const MetalEngine = big_ll.MetalEngine;
const Chromosome = big_ll.Chromosome;
const Xoshiro256 = big_ll.Xoshiro256;

test "MetalEngine availability and initialization lifecycle" {
    if (!MetalEngine.isSupported()) return;

    var engine = MetalEngine.init() orelse return error.MetalInitFailed;
    defer engine.deinit();

    try std.testing.expect(engine.handle != null);
}

test "MetalEngine OneMax GPU ALU bit-exact evaluation against CPU reference" {
    if (!MetalEngine.isSupported()) return;

    var engine = MetalEngine.init() orelse return error.MetalInitFailed;
    defer engine.deinit();

    const L = 256;
    const pop_size = 2048;

    var prng = Xoshiro256.init(42);
    var chroms: [pop_size]Chromosome(L) = undefined;
    var cpu_fitness: [pop_size]f32 = undefined;
    var gpu_fitness: [pop_size]f32 = undefined;

    for (0..pop_size) |i| {
        for (&chroms[i].words) |*w| {
            w.* = prng.next();
        }
        cpu_fitness[i] = @as(f32, @floatFromInt(chroms[i].popCount()));
    }

    const ok = engine.evalOneMax(L, &chroms, &gpu_fitness);
    try std.testing.expect(ok);

    for (0..pop_size) |i| {
        try std.testing.expectEqual(cpu_fitness[i], gpu_fitness[i]);
    }
}

test "MetalEngine Finance portfolio optimization GPU ALU against reference" {
    if (!MetalEngine.isSupported()) return;

    var engine = MetalEngine.init() orelse return error.MetalInitFailed;
    defer engine.deinit();

    const pop_size = 1024;
    const num_assets: u32 = 16;
    const risk_aversion: f32 = 1.0;

    var returns: [16]f32 = undefined;
    var cov: [256]f32 = undefined;
    for (0..16) |i| {
        returns[i] = 0.05 + 0.01 * @as(f32, @floatFromInt(i));
        for (0..16) |j| {
            cov[i * 16 + j] = if (i == j) 0.04 else 0.005;
        }
    }

    var prng = Xoshiro256.init(99);
    var chroms: [pop_size]u64 = undefined;
    var gpu_fitness: [pop_size]f32 = undefined;

    for (0..pop_size) |i| {
        chroms[i] = prng.next();
    }

    const ok = engine.evalFinance(
        &chroms,
        &gpu_fitness,
        &returns,
        &cov,
        num_assets,
        risk_aversion,
    );
    try std.testing.expect(ok);

    // Verify CPU reference on sample individuals
    for (0..10) |idx| {
        const c_val = chroms[idx];
        var w: [16]f32 = undefined;
        var sum_w: f32 = 0.0;
        for (0..16) |i| {
            const weight = @as(f32, @floatFromInt((c_val >> @intCast(i * 4)) & 0xF));
            w[i] = weight;
            sum_w += weight;
        }

        if (sum_w > 0.0001) {
            const inv_s = 1.0 / sum_w;
            for (&w) |*val| val.* *= inv_s;

            var r: f32 = 0.0;
            for (0..16) |i| r += w[i] * returns[i];

            var v: f32 = 0.0;
            for (0..16) |i| {
                var row: f32 = 0.0;
                for (0..16) |j| row += cov[i * 16 + j] * w[j];
                v += w[i] * row;
            }
            const expected_fit = r - (0.5 * risk_aversion * v);
            try std.testing.expectApproxEqAbs(expected_fit, gpu_fitness[idx], 1e-4);
        }
    }
}
