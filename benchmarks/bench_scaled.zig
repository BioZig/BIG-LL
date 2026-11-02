// Scaled Dimension & Multi-Batch L2 Cache Tiling Benchmark (L = 256 .. 8192)
const std = @import("std");
const builtin = @import("builtin");
const big_ll = @import("big_ll");

const LinkageEngine = big_ll.LinkageEngine;
const GeneticEngineCustom = big_ll.GeneticEngineCustom;
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

fn ScaledDeceptiveAdapter(comptime L: usize) type {
    return struct {
        const Self = @This();
        const Chrom = Chromosome(L);

        pub fn decode(self: *const Self, chrom: *const Chrom) usize {
            _ = self;
            return chrom.popCount();
        }

        pub fn evaluate(self: *const Self, ones: usize) f64 {
            _ = self;
            return @as(f64, @floatFromInt(ones));
        }
    };
}

fn benchmarkDimension(
    allocator: std.mem.Allocator,
    comptime L: usize,
    pop_size: usize,
    gens: usize,
) !void {
    const Engine = GeneticEngineCustom(L, ScaledDeceptiveAdapter(L), .{ .enable_dedup = false, .num_threads = 1 });
    const Link = LinkageEngine(L);

    const eng_ptr = try allocator.create(Engine);
    defer allocator.destroy(eng_ptr);

    const t0 = getTimestampNs();
    eng_ptr.* = try Engine.init(allocator, pop_size, .{}, 42);
    defer eng_ptr.deinit();
    const t1 = getTimestampNs();
    const init_ms = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;

    const t_start = getTimestampNs();
    for (0..gens) |_| {
        try eng_ptr.step();
    }
    const t_end = getTimestampNs();

    const total_ns = t_end - t_start;
    const total_evals = pop_size * gens;
    const total_sec = @as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0;
    const evals_per_sec = @as(f64, @floatFromInt(total_evals)) / total_sec;
    const us_per_gen = (@as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(gens))) / 1_000.0;

    const footprint_kb = (pop_size * @sizeOf(Chromosome(L))) / 1024;
    const tile_kb = (Link.batch_size * ((Link.batch_size + 63) / 64) * @sizeOf(u64)) / 1024;

    std.debug.print("scale=L{d:<5} batches={d:<2} elite_k={d:<3} pop={d:<5} pop_mem={d:>4} KB tile={d:>3} KB gens={d:<4} evals_per_sec={d:>10.2} us_per_gen={d:>8.2} edges={d:<5} best_fit={d:>6.1} init_ms={d:.2}\n", .{
        L,
        Link.num_batches,
        Link.elite_size,
        pop_size,
        footprint_kb,
        tile_kb,
        gens,
        evals_per_sec,
        us_per_gen,
        eng_ptr.linkage_engine.edge_count,
        eng_ptr.getBest().fitness,
        init_ms,
    });
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("BIG-LL Scaled Dimension & Multi-Batch L2 Cache Tiling Benchmark\n", .{});
    std.debug.print("Testing L=256 up to L=8192 across multi-batch epistasis scheduling (tile size <= 512 KB)\n\n", .{});

    // Tier 1: L=256 (Small, L1D resident: 64 KB for 1024 pop)
    try benchmarkDimension(allocator, 256, 1024, 100);

    // Tier 2: L=512 (Medium, single tile: 32 KB tile)
    try benchmarkDimension(allocator, 512, 1024, 100);

    // Tier 3: L=1024 (Single tile: 128 KB tile)
    try benchmarkDimension(allocator, 1024, 1024, 100);

    // Tier 4: L=2048 (Single full tile: 512 KB L2 tile boundary)
    try benchmarkDimension(allocator, 2048, 1024, 50);

    // Tier 5: L=4096 (2 batches: cross-tile interleaved epistasis scheduling)
    try benchmarkDimension(allocator, 4096, 1024, 40);

    // Tier 6: L=8192 (4 batches: multi-stride rotation scheduling)
    try benchmarkDimension(allocator, 8192, 1024, 25);

    // Tier 7: L=20480 (10 batches: multi-stride rotation scheduling)
    try benchmarkDimension(allocator, 20480, 1024, 15);

    std.debug.print("\n--- CPU Population Scaling Suite (L=256, pop=1024 -> 16384) ---\n", .{});
    try benchmarkDimension(allocator, 256, 1024, 100);
    try benchmarkDimension(allocator, 256, 4096, 50);
    try benchmarkDimension(allocator, 256, 16384, 25);
}
