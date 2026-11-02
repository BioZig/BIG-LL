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

fn benchOneMaxGpuZeroCopy(
    engine: *MetalEngine,
    comptime L: usize,
    pop_size: usize,
    batches: usize,
) !void {
    const chroms = engine.getSharedChromBuffer(Chromosome(L), pop_size) orelse return error.OutOfMemory;
    const fitnesses = engine.getSharedFitnessBuffer(pop_size) orelse return error.OutOfMemory;

    var prng = Xoshiro256.init(12345);
    for (chroms) |*c| {
        for (&c.words) |*w| {
            w.* = prng.next();
        }
    }

    // Warmup
    for (0..5) |_| {
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
    const mb_per_batch = @as(f64, @floatFromInt(pop_size * @sizeOf(Chromosome(L)))) / (1024.0 * 1024.0);

    std.debug.print("benchmark=metal_zerocopy domain=onemax l={d:<5} pop_size={d:<8} mb={d:>6.2} batches={d:<3} total_sec={d:.4} us_per_batch={d:>9.2} evals_per_sec={d:>12.2} million_evals_sec={d:>8.2} sample_fit={d:.0}\n", .{
        L,
        pop_size,
        mb_per_batch,
        batches,
        total_sec,
        us_per_batch,
        evals_per_sec,
        evals_per_sec / 1_000_000.0,
        fitnesses[0],
    });
}

fn benchFinanceGpuZeroCopy(
    engine: *MetalEngine,
    pop_size: usize,
    batches: usize,
) !void {
    const chroms = engine.getSharedChromBuffer(u64, pop_size) orelse return error.OutOfMemory;
    const fitnesses = engine.getSharedFitnessBuffer(pop_size) orelse return error.OutOfMemory;

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
    for (0..5) |_| {
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
    const mb_per_batch = @as(f64, @floatFromInt(pop_size * @sizeOf(u64))) / (1024.0 * 1024.0);

    std.debug.print("benchmark=metal_zerocopy domain=finance assets=16 pop_size={d:<8} mb={d:>6.2} batches={d:<3} total_sec={d:.4} us_per_batch={d:>9.2} evals_per_sec={d:>12.2} million_evals_sec={d:>8.2} sample_fit={d:.4}\n", .{
        pop_size,
        mb_per_batch,
        batches,
        total_sec,
        us_per_batch,
        evals_per_sec,
        evals_per_sec / 1_000_000.0,
        fitnesses[0],
    });
}

pub fn main() !void {
    std.debug.print("================================================================================\n", .{});
    std.debug.print("   BIG-LL Dedicated Apple Silicon Metal GPU Acceleration Suite\n", .{});
    std.debug.print("   Zero-Copy Unified Memory Architecture (UMA) Scaling Benchmark\n", .{});
    std.debug.print("================================================================================\n\n", .{});

    if (!MetalEngine.isSupported()) {
        std.debug.print("Metal compute is not supported or not Apple Silicon UMA on this hardware.\n", .{});
        return;
    }

    var engine = MetalEngine.init() orelse {
        std.debug.print("Failed to initialize MetalEngine context.\n", .{});
        return;
    };
    defer engine.deinit();

    std.debug.print("Hardware confirmed: Apple Silicon Unified Memory Architecture (UMA) Active\n\n", .{});

    // -------------------------------------------------------------------------
    // Suite 1: Finance Markowitz QP Ultra-High Population Scaling (4K -> 4.19M)
    // -------------------------------------------------------------------------
    std.debug.print("--- Suite 1: Finance Markowitz QP Population Scaling (4,096 -> 4,194,304) ---\n", .{});
    const finance_pops = [_]usize{ 4096, 16384, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304 };
    inline for (finance_pops) |pop| {
        const batches: usize = if (pop >= 2097152) 25 else if (pop >= 524288) 50 else if (pop >= 262144) 100 else 200;
        try benchFinanceGpuZeroCopy(&engine, pop, batches);
    }
    std.debug.print("\n", .{});

    // -------------------------------------------------------------------------
    // Suite 2: OneMax L=256 Ultra-High Population Scaling (4K -> 4.19M)
    // -------------------------------------------------------------------------
    std.debug.print("--- Suite 2: OneMax L=256 Population Scaling (4,096 -> 4,194,304) ---\n", .{});
    const onemax_pops = [_]usize{ 4096, 16384, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304 };
    inline for (onemax_pops) |pop| {
        const batches: usize = if (pop >= 2097152) 25 else if (pop >= 524288) 50 else if (pop >= 262144) 100 else 200;
        try benchOneMaxGpuZeroCopy(&engine, 256, pop, batches);
    }
    std.debug.print("\n", .{});

    // -------------------------------------------------------------------------
    // Suite 3: High-Dimensional Input Sets Scaling (L = 256 .. 20,480 bits)
    // -------------------------------------------------------------------------
    std.debug.print("--- Suite 3: OneMax High-Dimensional Input Sets Scaling (L=256 -> 20,480 bits) ---\n", .{});
    const test_pop: usize = 65536;
    const test_batches: usize = 50;

    try benchOneMaxGpuZeroCopy(&engine, 256, test_pop, test_batches);
    try benchOneMaxGpuZeroCopy(&engine, 512, test_pop, test_batches);
    try benchOneMaxGpuZeroCopy(&engine, 1024, test_pop, test_batches);
    try benchOneMaxGpuZeroCopy(&engine, 2048, test_pop, test_batches);
    try benchOneMaxGpuZeroCopy(&engine, 4096, test_pop, test_batches);
    try benchOneMaxGpuZeroCopy(&engine, 8192, test_pop, test_batches);
    try benchOneMaxGpuZeroCopy(&engine, 20480, test_pop, test_batches);

    std.debug.print("\n================================================================================\n", .{});
    std.debug.print("   Metal GPU Scaled Benchmark Complete\n", .{});
    std.debug.print("================================================================================\n", .{});
}
