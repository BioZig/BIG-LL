const std = @import("std");
const builtin = @import("builtin");
const big_ll = @import("big_ll");

const GeneticEngineCustom = big_ll.GeneticEngineCustom;
const IslandEngine = big_ll.IslandEngine;
const generator = big_ll.data_generator;
const loader = big_ll.data_loader;

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
        return @as(u64, @intCast(ts.tv_sec)) * 1_000_000_000 + @as(u64, @intCast(ts.tv_nsec));
    } else {
        return 0;
    }
}

fn runDomainSingle(
    allocator: std.mem.Allocator,
    comptime L: usize,
    comptime Domain: type,
    domain: Domain,
    domain_name: []const u8,
    tier_name: []const u8,
    pop_size: usize,
    gens: usize,
    seed: u64,
) !void {
    const Engine = GeneticEngineCustom(L, Domain, .{ .enable_dedup = false, .num_threads = 1 });

    const t0 = getTimestampNs();
    var eng = try Engine.init(allocator, pop_size, domain, seed);
    defer eng.deinit();
    const t1 = getTimestampNs();
    const init_ms = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;

    const init_fitness = eng.getBest().fitness;

    const t_start = getTimestampNs();
    for (0..gens) |_| {
        try eng.step();
    }
    const t_end = getTimestampNs();

    const total_ns = t_end - t_start;
    const total_evals = pop_size * gens;
    const total_sec = @as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0;
    const evals_per_sec = @as(f64, @floatFromInt(total_evals)) / total_sec;
    const us_per_gen = (@as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(gens))) / 1_000.0;
    const final_best = eng.getBest().fitness;
    const entropy = eng.pop_current.entropy;

    std.debug.print("domain={s:<14} tier={s:<8} arch=single   pop={d:<5} gens={d:<4} evals_per_sec={d:>12.2} us_per_gen={d:>8.2} init_fit={d:>10.2} best_fit={d:>10.2} entropy={d:>6.4} init_ms={d:.2}\n", .{
        domain_name,
        tier_name,
        pop_size,
        gens,
        evals_per_sec,
        us_per_gen,
        init_fitness,
        final_best,
        entropy,
        init_ms,
    });
}

fn runDomainIsland(
    allocator: std.mem.Allocator,
    comptime L: usize,
    comptime Domain: type,
    domain: Domain,
    domain_name: []const u8,
    tier_name: []const u8,
    pop_size: usize,
    gens: usize,
    seed: u64,
    comptime num_islands: usize,
    comptime migration_interval: usize,
) !void {
    const Engine = IslandEngine(L, Domain, num_islands, migration_interval);

    const t0 = getTimestampNs();
    var eng = try Engine.init(allocator, pop_size, domain, seed);
    defer eng.deinit();
    const t1 = getTimestampNs();
    const init_ms = @as(f64, @floatFromInt(t1 - t0)) / 1_000_000.0;

    const init_fitness = eng.getGlobalBest().fitness;

    const t_start = getTimestampNs();
    var completed: usize = 0;
    while (completed < gens) {
        const step_len = @min(migration_interval, gens - completed);
        try eng.stepCount(step_len);
        completed += step_len;
    }
    const t_end = getTimestampNs();

    const total_ns = t_end - t_start;
    const total_evals = pop_size * gens;
    const total_sec = @as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0;
    const evals_per_sec = @as(f64, @floatFromInt(total_evals)) / total_sec;
    const us_per_gen = (@as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(gens))) / 1_000.0;
    const final_best = eng.getGlobalBest().fitness;
    const entropy = eng.getGlobalEntropy();

    std.debug.print("domain={s:<14} tier={s:<8} arch=island-4 pop={d:<5} gens={d:<4} evals_per_sec={d:>12.2} us_per_gen={d:>8.2} init_fit={d:>10.2} best_fit={d:>10.2} entropy={d:>6.4} init_ms={d:.2}\n", .{
        domain_name,
        tier_name,
        pop_size,
        gens,
        evals_per_sec,
        us_per_gen,
        init_fitness,
        final_best,
        entropy,
        init_ms,
    });
}

const TierDef = struct {
    name: []const u8,
    file_path: [:0]const u8,
};

fn openOrGenerate(allocator: std.mem.Allocator, file_path: [:0]const u8) !loader.MemoryMappedInstance {
    return loader.MemoryMappedInstance.openFile(file_path) catch {
        try generator.generateBenchmarkSuite(allocator, "data/synthetic");
        return try loader.MemoryMappedInstance.openFile(file_path);
    };
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const pop_size: usize = 1024;
    const gens: usize = 300;

    std.debug.print("BIG-LL Multi-Tier Synthetic Domain Benchmark Suite\n", .{});
    std.debug.print("Configuration: 6 domains x 3 hardness tiers (slack, standard, stress) x 2 architectures (single, island-4)\n", .{});
    std.debug.print("Workload: pop_size={d}, gens={d}, total_evals_per_run={d}, zero-copy mmap binary containers\n\n", .{ pop_size, gens, pop_size * gens });

    // 1. Logistics (CVRP)
    {
        const tiers = [_]TierDef{
            .{ .name = "slack", .file_path = "data/synthetic/logistics/c32_slack.bigd" },
            .{ .name = "standard", .file_path = "data/synthetic/logistics/c32_standard.bigd" },
            .{ .name = "stress", .file_path = "data/synthetic/logistics/c32_stress.bigd" },
        };
        for (tiers) |t| {
            var inst = try openOrGenerate(allocator, t.file_path);
            defer inst.deinit();
            const dom = try big_ll.LogisticsDomain.initFromInstance(&inst);

            try runDomainSingle(allocator, big_ll.LogisticsDomain.CHROM_LEN, big_ll.LogisticsDomain, dom, "logistics", t.name, pop_size, gens, 101);
            try runDomainIsland(allocator, big_ll.LogisticsDomain.CHROM_LEN, big_ll.LogisticsDomain, dom, "logistics", t.name, pop_size, gens, 102, 4, 50);
        }
    }

    // 2. Power Grid (MINLP Unit Commitment)
    {
        const tiers = [_]TierDef{
            .{ .name = "slack", .file_path = "data/synthetic/power_grid/g8_slack.bigd" },
            .{ .name = "standard", .file_path = "data/synthetic/power_grid/g8_standard.bigd" },
            .{ .name = "stress", .file_path = "data/synthetic/power_grid/g8_stress.bigd" },
        };
        for (tiers) |t| {
            var inst = try openOrGenerate(allocator, t.file_path);
            defer inst.deinit();
            const dom = try big_ll.PowerGridDomain.initFromInstance(&inst);

            try runDomainSingle(allocator, big_ll.PowerGridDomain.CHROM_LEN, big_ll.PowerGridDomain, dom, "power_grid", t.name, pop_size, gens, 201);
            try runDomainIsland(allocator, big_ll.PowerGridDomain.CHROM_LEN, big_ll.PowerGridDomain, dom, "power_grid", t.name, pop_size, gens, 202, 4, 50);
        }
    }

    // 3. Finance (Portfolio Allocation & Rules)
    {
        const tiers = [_]TierDef{
            .{ .name = "slack", .file_path = "data/synthetic/finance/a6_slack.bigd" },
            .{ .name = "standard", .file_path = "data/synthetic/finance/a6_standard.bigd" },
            .{ .name = "stress", .file_path = "data/synthetic/finance/a6_stress.bigd" },
        };
        for (tiers) |t| {
            var inst = try openOrGenerate(allocator, t.file_path);
            defer inst.deinit();
            const dom = try big_ll.FinanceDomain.initFromInstance(&inst);

            try runDomainSingle(allocator, big_ll.FinanceDomain.CHROM_LEN, big_ll.FinanceDomain, dom, "finance", t.name, pop_size, gens, 301);
            try runDomainIsland(allocator, big_ll.FinanceDomain.CHROM_LEN, big_ll.FinanceDomain, dom, "finance", t.name, pop_size, gens, 302, 4, 50);
        }
    }

    // 4. Bioinformatics (Epistasis & Biomarkers)
    {
        const tiers = [_]TierDef{
            .{ .name = "sparse", .file_path = "data/synthetic/bioinformatics/s128_sparse.bigd" },
            .{ .name = "standard", .file_path = "data/synthetic/bioinformatics/s128_standard.bigd" },
            .{ .name = "dense", .file_path = "data/synthetic/bioinformatics/s128_dense.bigd" },
        };
        for (tiers) |t| {
            var inst = try openOrGenerate(allocator, t.file_path);
            defer inst.deinit();
            const dom = try big_ll.BioinformaticsDomain.initFromInstance(&inst);

            try runDomainSingle(allocator, big_ll.BioinformaticsDomain.CHROM_LEN, big_ll.BioinformaticsDomain, dom, "bioinformatics", t.name, pop_size, gens, 401);
            try runDomainIsland(allocator, big_ll.BioinformaticsDomain.CHROM_LEN, big_ll.BioinformaticsDomain, dom, "bioinformatics", t.name, pop_size, gens, 402, 4, 50);
        }
    }

    // 5. Airfoil (Subsonic Aerodynamics)
    {
        const tiers = [_]TierDef{
            .{ .name = "slack", .file_path = "data/synthetic/airfoil/p16_slack.bigd" },
            .{ .name = "standard", .file_path = "data/synthetic/airfoil/p16_standard.bigd" },
            .{ .name = "stress", .file_path = "data/synthetic/airfoil/p16_stress.bigd" },
        };
        for (tiers) |t| {
            var inst = try openOrGenerate(allocator, t.file_path);
            defer inst.deinit();
            const dom = try big_ll.AirfoilDomain.initFromInstance(&inst);

            try runDomainSingle(allocator, big_ll.AirfoilDomain.CHROM_LEN, big_ll.AirfoilDomain, dom, "airfoil", t.name, pop_size, gens, 501);
            try runDomainIsland(allocator, big_ll.AirfoilDomain.CHROM_LEN, big_ll.AirfoilDomain, dom, "airfoil", t.name, pop_size, gens, 502, 4, 50);
        }
    }

    // 6. Telecom (MIMO Channel & Power Allocation)
    {
        const tiers = [_]TierDef{
            .{ .name = "clean", .file_path = "data/synthetic/telecom/u16_clean.bigd" },
            .{ .name = "standard", .file_path = "data/synthetic/telecom/u16_standard.bigd" },
            .{ .name = "noisy", .file_path = "data/synthetic/telecom/u16_noisy.bigd" },
        };
        for (tiers) |t| {
            var inst = try openOrGenerate(allocator, t.file_path);
            defer inst.deinit();
            const dom = try big_ll.TelecomDomain.initFromInstance(&inst);

            try runDomainSingle(allocator, big_ll.TelecomDomain.CHROM_LEN, big_ll.TelecomDomain, dom, "telecom", t.name, pop_size, gens, 601);
            try runDomainIsland(allocator, big_ll.TelecomDomain.CHROM_LEN, big_ll.TelecomDomain, dom, "telecom", t.name, pop_size, gens, 602, 4, 50);
        }
    }
}
