// Apple Silicon Dedicated Metal GPU Batch Acceleration Benchmark
const std = @import("std");
const builtin = @import("builtin");
const big_ll = @import("big_ll");
const MetalEngine = big_ll.MetalEngine;
const Chromosome = big_ll.Chromosome;
const Xoshiro256 = big_ll.Xoshiro256;

var timebase_factor: f64 = 1.0;
var timebase_initialized: bool = false;

fn initTimer() void {
    if (builtin.os.tag.isDarwin()) {
        var info: std.c.mach_timebase_info_data = undefined;
        _ = std.c.mach_timebase_info(&info);
        timebase_factor = @as(f64, @floatFromInt(info.numer)) / @as(f64, @floatFromInt(info.denom));
    }
    timebase_initialized = true;
}

fn getTimestampNs() u64 {
    if (!timebase_initialized) initTimer();
    if (builtin.os.tag.isDarwin()) {
        const ticks = std.c.mach_absolute_time();
        return @intFromFloat(@as(f64, @floatFromInt(ticks)) * timebase_factor);
    }
    return 0;
}

fn benchOneMaxGpu(
    allocator: std.mem.Allocator,
    engine: *MetalEngine,
    comptime L: usize,
    pop_size: usize,
    batches: usize,
) !void {
    const chroms = try allocator.alloc(Chromosome(L), pop_size);
    defer allocator.free(chroms);
    const fitnesses = try allocator.alloc(f32, pop_size);
    defer allocator.free(fitnesses);

    var prng = Xoshiro256.init(12345);
    for (chroms) |*c| {
        for (&c.words) |*w| {
            w.* = prng.next();
        }
    }

    // Warmup
    for (0..10) |_| {
        _ = engine.evalOneMax(L, chroms, fitnesses);
    }

    const t_start = getTimestampNs();
    for (0..batches) |_| {
        _ = engine.evalOneMax(L, chroms, fitnesses);
    }
    const t_end = getTimestampNs();

    const total_ns = t_end - t_start;
    const total_evals = pop_size * batches;
    const total_sec = @as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0;
    const evals_per_sec = @as(f64, @floatFromInt(total_evals)) / total_sec;
    const us_per_batch = (@as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(batches))) / 1_000.0;

    std.debug.print("benchmark=metal domain=onemax l={d} pop_size={d} batches={d} total_sec={d:.4} us_per_batch={d:.2} evals_per_sec={d:.2} million_evals_sec={d:.2} sample_fit={d:.0}\n", .{
        L,
        pop_size,
        batches,
        total_sec,
        us_per_batch,
        evals_per_sec,
        evals_per_sec / 1_000_000.0,
        fitnesses[0],
    });
}

fn benchFinanceGpu(
    allocator: std.mem.Allocator,
    engine: *MetalEngine,
    pop_size: usize,
    batches: usize,
) !void {
    const chroms = try allocator.alloc(u64, pop_size);
    defer allocator.free(chroms);
    const fitnesses = try allocator.alloc(f32, pop_size);
    defer allocator.free(fitnesses);

    var prng = Xoshiro256.init(54321);
    for (chroms) |*c| {
        c.* = prng.next();
    }

    var returns: [16]f32 = undefined;
    var cov: [256]f32 = undefined;
    for (0..16) |i| {
        returns[i] = 0.05 + 0.01 * @as(f32, @floatFromInt(i));
        for (0..16) |j| {
            cov[i * 16 + j] = if (i == j) 0.04 else 0.005;
        }
    }

    // Warmup
    for (0..10) |_| {
        _ = engine.evalFinance(chroms, fitnesses, &returns, &cov, 16, 1.0);
    }

    const t_start = getTimestampNs();
    for (0..batches) |_| {
        _ = engine.evalFinance(chroms, fitnesses, &returns, &cov, 16, 1.0);
    }
    const t_end = getTimestampNs();

    const total_ns = t_end - t_start;
    const total_evals = pop_size * batches;
    const total_sec = @as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0;
    const evals_per_sec = @as(f64, @floatFromInt(total_evals)) / total_sec;
    const us_per_batch = (@as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(batches))) / 1_000.0;

    std.debug.print("benchmark=metal domain=finance assets=16 pop_size={d} batches={d} total_sec={d:.4} us_per_batch={d:.2} evals_per_sec={d:.2} million_evals_sec={d:.2} sample_fit={d:.4}\n", .{
        pop_size,
        batches,
        total_sec,
        us_per_batch,
        evals_per_sec,
        evals_per_sec / 1_000_000.0,
        fitnesses[0],
    });
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("Apple Silicon Metal GPU Dedicated Acceleration Benchmark\n", .{});

    if (!MetalEngine.isSupported()) {
        std.debug.print("Metal compute is not supported or not Apple Silicon UMA on this hardware.\n", .{});
        return;
    }

    var engine = MetalEngine.init() orelse {
        std.debug.print("Failed to initialize MetalEngine context.\n", .{});
        return;
    };
    defer engine.deinit();

    std.debug.print("Hardware confirmed: Apple Silicon Unified Memory Architecture (UMA)\n\n", .{});

    // Suite 1: OneMax L=256 Scaling across Pop Sizes (4096 -> 131072)
    const pops = [_]usize{ 4096, 8192, 16384, 32768, 65536, 131072 };
    inline for (pops) |pop| {
        try benchOneMaxGpu(allocator, &engine, 256, pop, 200);
    }

    std.debug.print("\n", .{});

    // Suite 2: Finance Markowitz QP Portfolio Optimization across Pop Sizes
    inline for (pops) |pop| {
        try benchFinanceGpu(allocator, &engine, pop, 200);
    }
}
