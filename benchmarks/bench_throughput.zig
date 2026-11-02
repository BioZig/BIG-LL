const std = @import("std");
const builtin = @import("builtin");
const big_ll = @import("big_ll");
const GeneticEngineCustom = big_ll.GeneticEngineCustom;
const IslandEngine = big_ll.IslandEngine;
const Chromosome = big_ll.Chromosome;

var timebase_factor: f64 = 1.0;
var timebase_initialized: bool = false;

fn initTimer() void {
    if (builtin.os.tag == .macos or builtin.os.tag == .ios or builtin.os.tag == .watchos or builtin.os.tag == .tvos) {
        var info: std.c.mach_timebase_info_data = undefined;
        _ = std.c.mach_timebase_info(&info);
        timebase_factor = @as(f64, @floatFromInt(info.numer)) / @as(f64, @floatFromInt(info.denom));
    }
    timebase_initialized = true;
}

fn getTimestampNs() u64 {
    if (!timebase_initialized) initTimer();
    if (builtin.os.tag == .macos or builtin.os.tag == .ios or builtin.os.tag == .watchos or builtin.os.tag == .tvos) {
        const ticks = std.c.mach_absolute_time();
        return @intFromFloat(@as(f64, @floatFromInt(ticks)) * timebase_factor);
    } else if (builtin.os.tag == .linux) {
        var ts: std.os.linux.timespec = undefined;
        _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
    } else {
        return 0;
    }
}

fn FastEvalAdapter(comptime L: usize) type {
    return struct {
        pub fn decode(self: *const @This(), chrom: *const Chromosome(L)) usize {
            _ = self;
            return chrom.popCount();
        }

        pub fn evaluate(self: *const @This(), ones: usize) f64 {
            _ = self;
            return @as(f64, @floatFromInt(ones));
        }
    };
}

fn runBenchmark(
    allocator: std.mem.Allocator,
    comptime L: usize,
    pop_size: usize,
    num_gens: usize,
    seed: u64,
    comptime num_threads: usize,
) !void {
    const Adapter = FastEvalAdapter(L);
    const adapter = Adapter{};

    const t0 = getTimestampNs();
    var engine = try GeneticEngineCustom(L, Adapter, .{ .enable_dedup = false, .num_threads = num_threads }).init(allocator, pop_size, adapter, seed);
    defer engine.deinit();
    const t1 = getTimestampNs();

    const init_ns = t1 - t0;

    const t_start = getTimestampNs();
    for (0..num_gens) |_| {
        try engine.step();
    }
    const t_end = getTimestampNs();

    const total_ns = t_end - t_start;
    const total_evals = pop_size * num_gens;
    const total_sec = @as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0;
    const evals_per_sec = @as(f64, @floatFromInt(total_evals)) / total_sec;
    const ns_per_gen = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(num_gens));

    std.debug.print("benchmark=engine threads={d} pop_size={d} l={d} init_ms={d:.3} gens={d} total_evals={d} total_sec={d:.4} evals_per_sec={d:.2} latency_us={d:.2} best_fitness={d:.2} entropy={d:.4}\n", .{
        num_threads,
        pop_size,
        L,
        @as(f64, @floatFromInt(init_ns)) / 1_000_000.0,
        num_gens,
        total_evals,
        total_sec,
        evals_per_sec,
        ns_per_gen / 1_000.0,
        engine.getBest().fitness,
        engine.pop_current.entropy,
    });
}

fn runIslandBenchmark(
    allocator: std.mem.Allocator,
    comptime L: usize,
    pop_size: usize,
    num_gens: usize,
    seed: u64,
    comptime num_islands: usize,
    comptime migration_interval: usize,
) !void {
    const Adapter = FastEvalAdapter(L);
    const adapter = Adapter{};

    const t0 = getTimestampNs();
    var engine = try IslandEngine(L, Adapter, num_islands, migration_interval).init(allocator, pop_size, adapter, seed);
    defer engine.deinit();
    const t1 = getTimestampNs();

    const init_ns = t1 - t0;

    const t_start = getTimestampNs();
    var completed: usize = 0;
    while (completed < num_gens) {
        const step = @min(migration_interval, num_gens - completed);
        try engine.stepCount(step);
        completed += step;
    }
    const t_end = getTimestampNs();

    const total_ns = t_end - t_start;
    const total_evals = pop_size * num_gens;
    const total_sec = @as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0;
    const evals_per_sec = @as(f64, @floatFromInt(total_evals)) / total_sec;
    const ns_per_gen = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(num_gens));

    std.debug.print("benchmark=island islands={d} interval={d} pop_size={d} l={d} init_ms={d:.3} gens={d} total_evals={d} total_sec={d:.4} evals_per_sec={d:.2} latency_us={d:.2} best_fitness={d:.2} entropy={d:.4}\n", .{
        num_islands,
        migration_interval,
        pop_size,
        L,
        @as(f64, @floatFromInt(init_ns)) / 1_000_000.0,
        num_gens,
        total_evals,
        total_sec,
        evals_per_sec,
        ns_per_gen / 1_000.0,
        engine.getGlobalBest().fitness,
        engine.getGlobalEntropy(),
    });
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const gen_targets = [_]usize{ 50, 100, 500, 1000, 2000, 5000 };

    // Section 1: Island Topology (4 threads, thread-local cache, ring migration every 100 gens)
    inline for (gen_targets, 0..) |gens, i| {
        try runIslandBenchmark(allocator, 256, 1024, gens, 42 + i, 4, 100);
    }

    // Section 2: Workload Granularity (Pop size scaling up to 4096)
    try runBenchmark(allocator, 256, 2048, 500, 101, 1);
    try runBenchmark(allocator, 256, 2048, 500, 102, 4);
    try runIslandBenchmark(allocator, 256, 2048, 500, 103, 4, 100);

    try runBenchmark(allocator, 256, 4096, 500, 201, 1);
    try runBenchmark(allocator, 256, 4096, 500, 202, 4);
    try runIslandBenchmark(allocator, 256, 4096, 500, 203, 4, 100);
    try runIslandBenchmark(allocator, 256, 4096, 500, 204, 8, 100);

    // Section 3: High-Dimensional Scaling (L = 1024 and L = 4096)
    try runBenchmark(allocator, 1024, 2048, 200, 301, 1);
    try runBenchmark(allocator, 1024, 2048, 200, 302, 4);
    try runIslandBenchmark(allocator, 1024, 2048, 200, 303, 4, 100);
}
