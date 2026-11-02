const std = @import("std");
const builtin = @import("builtin");
const big_ll = @import("big_ll");
const GeneticEngine = big_ll.GeneticEngine;

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

fn printUsage() void {
    std.debug.print(
        \\Usage: big-ll <command> [options]
        \\
        \\Commands:
        \\  solve <file> [options]         Execute zero-allocation evolutionary solve on .bigd or raw file
        \\                                 (Supported formats: .bigd, .npy, .bed, .vrp, .csv, .tsv, .dat)
        \\                                 Options:
        \\                                   --pop <int>       Population size (default: 1024)
        \\                                   --gens <int>      Generations budget (default: 200)
        \\                                   --islands <1|4|8> Core island count (default: 4)
        \\                                   --seed <int>      PRNG seed (default: 42)
        \\                                   --dedup <bool>    Toggle clone filter (default: true)
        \\                                   --save-ir <path>  Optional: persist compiled in-memory .bigd IR to disk
        \\                                   --json            Output results as JSON
        \\                                   --quiet, -q       Suppress interactive progress
        \\  inspect <file>                 Inspect a dataset (supports .bigd, .npy, .bed, .vrp, .csv, etc.)
        \\  generate <domain> <out.bigd>   Generate a synthetic .bigd binary file
        \\                                 Domains: logistics, power_grid, finance,
        \\                                          bioinformatics, airfoil, telecom
        \\  generate-suite [dir]           Generate full 18-dataset benchmark suite
        \\                                 (defaults to data/synthetic)
        \\  ingest-real [raw] [out]        Ingest all standard real-world benchmarks
        \\                                 (defaults: data/raw -> data/real)
        \\  demo                           Run 6-domain evolutionary baseline demo
        \\  help                           Print this command-line manual
        \\
        \\Examples:
        \\  big-ll solve data/raw/logistics/A-n33-k5.vrp --pop 2048 --gens 300
        \\  big-ll solve data/raw/power_grid/ieee_rts96_thermal.csv --islands 4
        \\  big-ll solve data/raw/logistics/A-n33-k5.vrp --save-ir /tmp/cached_An33k5.bigd
        \\  big-ll solve /tmp/cached_An33k5.bigd
        \\  big-ll solve data/real/finance/sp500_portfolio.bigd --json
        \\  big-ll inspect data/real/power_grid/ieee_rts96.bigd
        \\  big-ll demo
        \\
    , .{});
}

pub const SolveOptions = struct {
    pop_size: usize = 1024,
    gens: usize = 200,
    islands: usize = 4,
    seed: u64 = 42,
    enable_dedup: bool = true,
    json: bool = false,
    quiet: bool = false,
    save_ir: ?[]const u8 = null,
};

fn runSingle(
    comptime L: usize,
    comptime Domain: type,
    comptime enable_dedup: bool,
    domain: Domain,
    options: SolveOptions,
    allocator: std.mem.Allocator,
    best_fit: *f64,
    mean_fit: *f64,
    final_entropy: *f64,
) !void {
    const Engine = big_ll.GeneticEngineCustom(L, Domain, .{ .enable_dedup = enable_dedup, .num_threads = 1 });
    const eng = try allocator.create(Engine);
    defer allocator.destroy(eng);
    eng.* = try Engine.init(allocator, options.pop_size, domain, options.seed);
    defer eng.deinit();

    for (0..options.gens) |g| {
        try eng.step();
        if (!options.quiet and !options.json and (g % 20 == 0 or g + 1 == options.gens)) {
            const pct = (@as(f64, @floatFromInt(g + 1)) / @as(f64, @floatFromInt(options.gens))) * 100.0;
            std.debug.print("\r[Gen {d:>4}/{d}] {d:>5.1}% | Best: {d:>10.4} | Mean: {d:>10.4} | Entropy: {d:.4} | Dedup: {d:>4.1}%", .{
                g + 1,
                options.gens,
                pct,
                eng.getBest().fitness,
                eng.pop_current.mean_fitness,
                eng.pop_current.entropy,
                eng.dedup_cache.hitRatio() * 100.0,
            });
        }
    }
    best_fit.* = eng.getBest().fitness;
    final_entropy.* = eng.pop_current.entropy;
    mean_fit.* = eng.pop_current.mean_fitness;
}

fn executeSolve(
    comptime L: usize,
    comptime Domain: type,
    domain: Domain,
    domain_name: []const u8,
    file_path: []const u8,
    options: SolveOptions,
    allocator: std.mem.Allocator,
) !void {
    if (!options.json and !options.quiet) {
        std.debug.print("BIG-LL Industrial Genetic Solver\n", .{});
        std.debug.print("Instance:     {s}\n", .{file_path});
        std.debug.print("Domain:       {s} (L={d})\n", .{ domain_name, L });
        std.debug.print("Topology:     {d}-island(s) | Pop: {d} | Gens: {d} | Seed: {d}\n\n", .{
            options.islands,
            options.pop_size,
            options.gens,
            options.seed,
        });
    }

    const t_start = getTimestampNs();
    var best_fit: f64 = 0.0;
    var final_entropy: f64 = 0.0;
    var mean_fit: f64 = 0.0;

    if (options.islands == 1) {
        if (options.enable_dedup) {
            try runSingle(L, Domain, true, domain, options, allocator, &best_fit, &mean_fit, &final_entropy);
        } else {
            try runSingle(L, Domain, false, domain, options, allocator, &best_fit, &mean_fit, &final_entropy);
        }
    } else if (options.islands == 4) {
        const Islands = big_ll.IslandEngine(L, Domain, 4, 25);
        const eng = try allocator.create(Islands);
        defer allocator.destroy(eng);
        eng.* = try Islands.init(allocator, options.pop_size, domain, options.seed);
        defer eng.deinit();

        var completed: usize = 0;
        while (completed < options.gens) {
            const step_len = @min(25, options.gens - completed);
            try eng.stepCount(step_len);
            completed += step_len;
            if (!options.quiet and !options.json and (completed % 25 == 0 or completed == options.gens)) {
                const pct = (@as(f64, @floatFromInt(completed)) / @as(f64, @floatFromInt(options.gens))) * 100.0;
                std.debug.print("\r[Gen {d:>4}/{d}] {d:>5.1}% | Best: {d:>10.4} | Mean: {d:>10.4} | Entropy: {d:.4}", .{
                    completed,
                    options.gens,
                    pct,
                    eng.getGlobalBest().fitness,
                    eng.getMeanFitness(),
                    eng.getGlobalEntropy(),
                });
            }
        }
        best_fit = eng.getGlobalBest().fitness;
        final_entropy = eng.getGlobalEntropy();
        mean_fit = eng.getMeanFitness();
    } else if (options.islands == 8) {
        const Islands = big_ll.IslandEngine(L, Domain, 8, 25);
        const eng = try allocator.create(Islands);
        defer allocator.destroy(eng);
        eng.* = try Islands.init(allocator, options.pop_size, domain, options.seed);
        defer eng.deinit();

        var completed: usize = 0;
        while (completed < options.gens) {
            const step_len = @min(25, options.gens - completed);
            try eng.stepCount(step_len);
            completed += step_len;
            if (!options.quiet and !options.json and (completed % 25 == 0 or completed == options.gens)) {
                const pct = (@as(f64, @floatFromInt(completed)) / @as(f64, @floatFromInt(options.gens))) * 100.0;
                std.debug.print("\r[Gen {d:>4}/{d}] {d:>5.1}% | Best: {d:>10.4} | Mean: {d:>10.4} | Entropy: {d:.4}", .{
                    completed,
                    options.gens,
                    pct,
                    eng.getGlobalBest().fitness,
                    eng.getMeanFitness(),
                    eng.getGlobalEntropy(),
                });
            }
        }
        best_fit = eng.getGlobalBest().fitness;
        final_entropy = eng.getGlobalEntropy();
        mean_fit = eng.getMeanFitness();
    } else {
        std.debug.print("error: unsupported island count {d}. Use 1, 4, or 8.\n", .{options.islands});
        return error.UnsupportedIslandCount;
    }

    const t_end = getTimestampNs();
    const total_ns = t_end - t_start;
    const total_sec = @as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0;
    const total_evals = options.pop_size * options.gens;
    const evals_per_sec = @as(f64, @floatFromInt(total_evals)) / total_sec;
    const us_per_gen = (@as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(options.gens))) / 1_000.0;

    if (options.json) {
        std.debug.print(
            \\{{
            \\  "file": "{s}",
            \\  "domain": "{s}",
            \\  "chrom_len": {d},
            \\  "pop_size": {d},
            \\  "generations": {d},
            \\  "islands": {d},
            \\  "elapsed_sec": {d:.6},
            \\  "evals_per_sec": {d:.2},
            \\  "us_per_gen": {d:.2},
            \\  "best_fitness": {d:.6},
            \\  "mean_fitness": {d:.6},
            \\  "final_entropy": {d:.6}
            \\}}
            \\
        , .{
            file_path,
            domain_name,
            L,
            options.pop_size,
            options.gens,
            options.islands,
            total_sec,
            evals_per_sec,
            us_per_gen,
            best_fit,
            mean_fit,
            final_entropy,
        });
    } else {
        std.debug.print("\n\nSolution Summary:\n", .{});
        std.debug.print("  Best Fitness:   {d:.6}\n", .{best_fit});
        std.debug.print("  Mean Fitness:   {d:.6}\n", .{mean_fit});
        std.debug.print("  Elapsed Time:   {d:.2} ms\n", .{total_sec * 1000.0});
        std.debug.print("  Throughput:     {d:.2} evals/sec ({d:.2} us/gen)\n", .{ evals_per_sec, us_per_gen });
        std.debug.print("  Final Entropy:  {d:.4}\n", .{final_entropy});
    }
}

fn cmdSolve(file_path: []const u8, options: SolveOptions, allocator: std.mem.Allocator) !void {
    const file_path_z = try allocator.dupeZ(u8, file_path);
    defer allocator.free(file_path_z);

    var inst = big_ll.BigDInstance.openFileAuto(file_path_z, allocator) catch |err| {
        std.debug.print("error: failed to open '{s}': {s}\n", .{ file_path, @errorName(err) });
        return;
    };
    defer inst.deinit();

    if (options.save_ir) |save_path| {
        const save_path_z = try allocator.dupeZ(u8, save_path);
        defer allocator.free(save_path_z);
        big_ll.data_generator.saveToFile(save_path_z, inst.raw_slice) catch |err| {
            std.debug.print("error: failed to save compiled IR to '{s}': {s}\n", .{ save_path, @errorName(err) });
            return;
        };
        if (!options.quiet and !options.json) {
            std.debug.print("Compiled IR saved to: {s} ({d} bytes)\n", .{ save_path, inst.raw_slice.len });
        }
    }

    const hdr = inst.header();
    const domain_tag: big_ll.DomainTag = @enumFromInt(hdr.domain_tag);

    switch (domain_tag) {
        .logistics => {
            switch (hdr.param_u32_0) {
                31 => {
                    const Dom = big_ll.ParameterizedLogisticsDomain(31, 16);
                    const dom = try Dom.initFromInstance(&inst);
                    try executeSolve(Dom.CHROM_LEN, Dom, dom, "logistics", file_path, options, allocator);
                },
                32 => {
                    const Dom = big_ll.ParameterizedLogisticsDomain(32, 16);
                    const dom = try Dom.initFromInstance(&inst);
                    try executeSolve(Dom.CHROM_LEN, Dom, dom, "logistics", file_path, options, allocator);
                },
                64 => {
                    const Dom = big_ll.ParameterizedLogisticsDomain(64, 16);
                    const dom = try Dom.initFromInstance(&inst);
                    try executeSolve(Dom.CHROM_LEN, Dom, dom, "logistics", file_path, options, allocator);
                },
                else => {
                    std.debug.print("error: unsupported customer count {d} for logistics domain. Supported: 31, 32, 64.\n", .{hdr.param_u32_0});
                    return error.UnsupportedCustomerCount;
                },
            }
        },
        .power_grid => {
            switch (hdr.param_u32_0) {
                8 => {
                    const Dom = big_ll.ParameterizedPowerGridDomain(8, 16, 150.0);
                    const dom = try Dom.initFromInstance(&inst);
                    try executeSolve(Dom.CHROM_LEN, Dom, dom, "power_grid", file_path, options, allocator);
                },
                32 => {
                    const Dom = big_ll.ParameterizedPowerGridDomain(32, 16, 450.0);
                    const dom = try Dom.initFromInstance(&inst);
                    try executeSolve(Dom.CHROM_LEN, Dom, dom, "power_grid", file_path, options, allocator);
                },
                else => {
                    std.debug.print("error: unsupported generator count {d} for power grid domain. Supported: 8, 32.\n", .{hdr.param_u32_0});
                    return error.UnsupportedGeneratorCount;
                },
            }
        },
        .finance => {
            switch (hdr.param_u32_0) {
                6 => {
                    const Dom = big_ll.ParameterizedFinanceDomain(6, 8, 16);
                    const dom = try Dom.initFromInstance(&inst);
                    try executeSolve(Dom.CHROM_LEN, Dom, dom, "finance", file_path, options, allocator);
                },
                16 => {
                    const Dom = big_ll.ParameterizedFinanceDomain(16, 8, 16);
                    const dom = try Dom.initFromInstance(&inst);
                    try executeSolve(Dom.CHROM_LEN, Dom, dom, "finance", file_path, options, allocator);
                },
                else => {
                    const Dom = big_ll.FinanceDomain;
                    const dom = try Dom.initFromInstance(&inst);
                    try executeSolve(Dom.CHROM_LEN, Dom, dom, "finance", file_path, options, allocator);
                },
            }
        },
        .bioinformatics => {
            switch (hdr.param_u32_0) {
                64 => {
                    const Dom = big_ll.ParameterizedBioinformaticsDomain(64);
                    const dom = try Dom.initFromInstance(&inst);
                    try executeSolve(Dom.CHROM_LEN, Dom, dom, "bioinformatics", file_path, options, allocator);
                },
                128 => {
                    const Dom = big_ll.ParameterizedBioinformaticsDomain(128);
                    const dom = try Dom.initFromInstance(&inst);
                    try executeSolve(Dom.CHROM_LEN, Dom, dom, "bioinformatics", file_path, options, allocator);
                },
                256 => {
                    const Dom = big_ll.ParameterizedBioinformaticsDomain(256);
                    const dom = try Dom.initFromInstance(&inst);
                    try executeSolve(Dom.CHROM_LEN, Dom, dom, "bioinformatics", file_path, options, allocator);
                },
                else => {
                    const Dom = big_ll.BioinformaticsDomain;
                    const dom = try Dom.initFromInstance(&inst);
                    try executeSolve(Dom.CHROM_LEN, Dom, dom, "bioinformatics", file_path, options, allocator);
                },
            }
        },
        .airfoil => {
            const dom = try big_ll.AirfoilDomain.initFromInstance(&inst);
            try executeSolve(big_ll.AirfoilDomain.CHROM_LEN, big_ll.AirfoilDomain, dom, "airfoil", file_path, options, allocator);
        },
        .telecom => {
            const dom = try big_ll.TelecomDomain.initFromInstance(&inst);
            try executeSolve(big_ll.TelecomDomain.CHROM_LEN, big_ll.TelecomDomain, dom, "telecom", file_path, options, allocator);
        },
    }
}

fn cmdInspect(path: []const u8, allocator: std.mem.Allocator) !void {
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    var inst = big_ll.BigDInstance.openFileAuto(path_z, allocator) catch |err| {
        std.debug.print("error: failed to open '{s}': {s}\n", .{ path, @errorName(err) });
        return;
    };
    defer inst.deinit();

    const hdr = inst.header();
    const domain_tag: big_ll.DomainTag = @enumFromInt(hdr.domain_tag);

    std.debug.print("file:         {s}\n", .{path});
    std.debug.print("magic:        0x{X:0>8} ({s})\n", .{ hdr.magic, if (hdr.magic == big_ll.data_format.Magic) "valid" else "invalid" });
    std.debug.print("version:      {d}\n", .{hdr.version});
    std.debug.print("domain:       {s} (tag {d})\n", .{ @tagName(domain_tag), hdr.domain_tag });
    std.debug.print("total_size:   {d} bytes\n", .{hdr.total_bytes});
    std.debug.print("checksum:     0x{X:0>8}\n", .{hdr.checksum});
    std.debug.print("header_size:  128 bytes (align 64)\n", .{});
    std.debug.print("offsets:      section_a={d} section_b={d} section_c={d}\n", .{
        hdr.offset_section_a,
        hdr.offset_section_b,
        hdr.offset_section_c,
    });

    switch (domain_tag) {
        .logistics => {
            const num_cust = hdr.param_u32_0;
            const capacity = hdr.param_f64_0;
            const depot_x = hdr.param_f64_1;
            const depot_y = hdr.param_f64_2;
            std.debug.print("customers:    {d}\n", .{num_cust});
            std.debug.print("capacity:     {d:.2}\n", .{capacity});
            std.debug.print("depot:        ({d:.2}, {d:.2})\n", .{ depot_x, depot_y });

            const points = try inst.getSlice(big_ll.data_generator.Point, hdr.offset_section_a, num_cust);
            const demands = try inst.getSlice(f64, hdr.offset_section_b, num_cust);
            const preview_count = @min(3, num_cust);
            std.debug.print("preview (first {d}):\n", .{preview_count});
            for (0..preview_count) |i| {
                std.debug.print("  [{d}] loc=({d:.2}, {d:.2}) demand={d:.2}\n", .{ i, points[i].x, points[i].y, demands[i] });
            }

            const stride = num_cust + 1;
            const dist_mat = try inst.getSlice(f64, hdr.offset_section_c, stride * stride);
            std.debug.print("dist_matrix:  {d}x{d} flat ({d} elements)\n", .{ stride, stride, dist_mat.len });
        },
        .power_grid => {
            const num_gen = hdr.param_u32_0;
            const demand = hdr.param_f64_0;
            const reserve = hdr.param_f64_1;
            std.debug.print("generators:   {d}\n", .{num_gen});
            std.debug.print("demand:       {d:.2} MW\n", .{demand});
            std.debug.print("reserve:      {d:.2} MW\n", .{reserve});

            const specs = try inst.getSlice(big_ll.data_generator.GeneratorSpec, hdr.offset_section_a, num_gen);
            const preview_count = @min(4, num_gen);
            std.debug.print("preview (first {d}):\n", .{preview_count});
            for (0..preview_count) |i| {
                std.debug.print("  G{d}: p_min={d:.1} p_max={d:.1} cost_a={d:.4} cost_b={d:.2} no_load={d:.1}\n", .{
                    i, specs[i].p_min, specs[i].p_max, specs[i].cost_a, specs[i].cost_b, specs[i].cost_no_load,
                });
            }
        },
        .finance => {
            const num_assets = hdr.param_u32_0;
            const num_rules = hdr.param_u32_1;
            const rf_rate = hdr.param_f64_0;
            std.debug.print("assets:       {d}\n", .{num_assets});
            std.debug.print("rules:        {d}\n", .{num_rules});
            std.debug.print("rf_rate:      {d:.4}\n", .{rf_rate});

            const returns = try inst.getSlice(f64, hdr.offset_section_a, num_assets);
            std.debug.print("returns:      [", .{});
            for (returns, 0..) |r, i| {
                std.debug.print("{d:.4}{s}", .{ r, if (i + 1 < returns.len) ", " else "]\n" });
            }

            const cov = try inst.getSlice(f64, hdr.offset_section_b, num_assets * num_assets);
            std.debug.print("covariance:   {d}x{d} flat ({d} elements)\n", .{ num_assets, num_assets, cov.len });
        },
        .bioinformatics => {
            const num_snps = hdr.param_u32_0;
            const target_card = hdr.param_u32_1;
            const max_card = hdr.param_u32_2;
            const num_pairs = hdr.param_u32_3;
            const num_triplets = hdr.param_u32_4;
            const penalty_w = hdr.param_f64_0;
            std.debug.print("snps:         {d}\n", .{num_snps});
            std.debug.print("cardinality:  target={d} max={d}\n", .{ target_card, max_card });
            std.debug.print("epistasis:    pairs={d} triplets={d} penalty_w={d:.2}\n", .{ num_pairs, num_triplets, penalty_w });

            const pairs = try inst.getSlice(big_ll.data_format.EpistasisPair, hdr.offset_section_b, num_pairs);
            for (pairs, 0..) |p, i| {
                std.debug.print("  pair {d}: ({d}, {d}) weight={d:.1}\n", .{ i, p.idx1, p.idx2, p.weight });
            }
        },
        .airfoil => {
            const num_points = hdr.param_u32_0;
            const cl_factor = hdr.param_f64_0;
            const base_cd = hdr.param_f64_1;
            std.debug.print("points:       {d}\n", .{num_points});
            std.debug.print("cl_factor:    {d:.4}\n", .{cl_factor});
            std.debug.print("base_cd:      {d:.4}\n", .{base_cd});
        },
        .telecom => {
            const num_users = hdr.param_u32_0;
            const num_channels = hdr.param_u32_1;
            const noise = hdr.param_f64_0;
            std.debug.print("users:        {d}\n", .{num_users});
            std.debug.print("channels:     {d}\n", .{num_channels});
            std.debug.print("noise:        {d:.4}\n", .{noise});

            const mat = try inst.getSlice(f64, hdr.offset_section_a, num_users * num_users);
            std.debug.print("channel_mat:  {d}x{d} flat ({d} elements)\n", .{ num_users, num_users, mat.len });
        },
    }
}

fn cmdGenerate(domain_str: []const u8, out_path: []const u8, allocator: std.mem.Allocator) !void {
    var buffer: []align(64) u8 = undefined;

    if (std.mem.eql(u8, domain_str, "logistics")) {
        buffer = try big_ll.data_generator.generateLogistics(allocator, 32, 35.0, 50.0, 50.0);
    } else if (std.mem.eql(u8, domain_str, "power_grid")) {
        buffer = try big_ll.data_generator.generatePowerGrid(allocator, 550.0, 630.0);
    } else if (std.mem.eql(u8, domain_str, "finance")) {
        buffer = try big_ll.data_generator.generateFinance(allocator, 6, 8, 0.03);
    } else if (std.mem.eql(u8, domain_str, "bioinformatics")) {
        buffer = try big_ll.data_generator.generateBioinformatics(allocator, 128, 8, 12);
    } else if (std.mem.eql(u8, domain_str, "airfoil")) {
        buffer = try big_ll.data_generator.generateAirfoil(allocator, 16, 2.5, 0.01);
    } else if (std.mem.eql(u8, domain_str, "telecom")) {
        buffer = try big_ll.data_generator.generateTelecom(allocator, 16, 4, 0.05);
    } else {
        std.debug.print("error: unknown domain '{s}'. supported: logistics, power_grid, finance, bioinformatics, airfoil, telecom\n", .{domain_str});
        return;
    }
    defer allocator.free(buffer);

    const out_path_z = try allocator.dupeZ(u8, out_path);
    defer allocator.free(out_path_z);

    try big_ll.data_generator.saveToFile(out_path_z, buffer);
    const hdr: *const big_ll.BigDHeader = @ptrCast(buffer.ptr);

    std.debug.print("generated: domain={s} path={s} size={d} crc=0x{X:0>8}\n", .{
        domain_str,
        out_path,
        buffer.len,
        hdr.checksum,
    });
}

fn cmdGenerateSuite(target_dir: []const u8, allocator: std.mem.Allocator) !void {
    std.debug.print("generating 18-file synthetic benchmark suite into '{s}'...\n", .{target_dir});
    try big_ll.data_generator.generateBenchmarkSuite(allocator, target_dir);
    std.debug.print("suite generation complete: 18 datasets written across 6 domains (slack, standard, stress).\n", .{});
}

fn cmdIngestReal(raw_dir: []const u8, out_dir: []const u8, allocator: std.mem.Allocator) !void {
    std.debug.print("ingesting real-world benchmarks: '{s}' -> '{s}'...\n", .{ raw_dir, out_dir });
    try big_ll.data_importer.ingestAllRealDatasets(allocator, raw_dir, out_dir);
    std.debug.print("ingestion complete: real-world benchmark datasets written to '{s}'.\n", .{out_dir});
}

fn cmdDemo(allocator: std.mem.Allocator) !void {
    {
        const domain = big_ll.AirfoilDomain{};
        var engine = try GeneticEngine(big_ll.AirfoilDomain.CHROM_LEN, big_ll.AirfoilDomain).init(allocator, 128, domain, 1);
        defer engine.deinit();
        for (0..15) |_| try engine.step();
        std.debug.print("domain=airfoil l={d} gen={d} best_fitness={d:.4} dedup_hits={d} entropy={d:.4}\n", .{
            big_ll.AirfoilDomain.CHROM_LEN,
            engine.generation,
            engine.getBest().fitness,
            engine.dedup_cache.hits,
            engine.pop_current.entropy,
        });
    }

    {
        const domain = big_ll.LogisticsDomain.initDefault();
        var engine = try GeneticEngine(big_ll.LogisticsDomain.CHROM_LEN, big_ll.LogisticsDomain).init(allocator, 128, domain, 2);
        defer engine.deinit();
        for (0..15) |_| try engine.step();
        std.debug.print("domain=logistics l={d} gen={d} best_fitness={d:.4} dedup_hits={d} entropy={d:.4}\n", .{
            big_ll.LogisticsDomain.CHROM_LEN,
            engine.generation,
            engine.getBest().fitness,
            engine.dedup_cache.hits,
            engine.pop_current.entropy,
        });
    }

    {
        const domain = big_ll.PowerGridDomain.initDefault();
        var engine = try GeneticEngine(big_ll.PowerGridDomain.CHROM_LEN, big_ll.PowerGridDomain).init(allocator, 128, domain, 3);
        defer engine.deinit();
        for (0..15) |_| try engine.step();
        std.debug.print("domain=power_grid l={d} gen={d} best_fitness={d:.4} dedup_hits={d} entropy={d:.4}\n", .{
            big_ll.PowerGridDomain.CHROM_LEN,
            engine.generation,
            engine.getBest().fitness,
            engine.dedup_cache.hits,
            engine.pop_current.entropy,
        });
    }

    {
        const domain = big_ll.BioinformaticsDomain{};
        var engine = try GeneticEngine(big_ll.BioinformaticsDomain.CHROM_LEN, big_ll.BioinformaticsDomain).init(allocator, 128, domain, 4);
        defer engine.deinit();
        for (0..15) |_| try engine.step();
        std.debug.print("domain=bioinformatics l={d} gen={d} best_fitness={d:.4} dedup_hits={d} entropy={d:.4}\n", .{
            big_ll.BioinformaticsDomain.CHROM_LEN,
            engine.generation,
            engine.getBest().fitness,
            engine.dedup_cache.hits,
            engine.pop_current.entropy,
        });
    }

    {
        const domain = big_ll.FinanceDomain.initDefault();
        var engine = try GeneticEngine(big_ll.FinanceDomain.CHROM_LEN, big_ll.FinanceDomain).init(allocator, 128, domain, 5);
        defer engine.deinit();
        for (0..15) |_| try engine.step();
        std.debug.print("domain=finance l={d} gen={d} best_fitness={d:.4} dedup_hits={d} entropy={d:.4}\n", .{
            big_ll.FinanceDomain.CHROM_LEN,
            engine.generation,
            engine.getBest().fitness,
            engine.dedup_cache.hits,
            engine.pop_current.entropy,
        });
    }

    {
        const domain = big_ll.TelecomDomain.initDefault();
        var engine = try GeneticEngine(big_ll.TelecomDomain.CHROM_LEN, big_ll.TelecomDomain).init(allocator, 128, domain, 6);
        defer engine.deinit();
        for (0..15) |_| try engine.step();
        std.debug.print("domain=telecom l={d} gen={d} best_fitness={d:.4} dedup_hits={d} entropy={d:.4}\n", .{
            big_ll.TelecomDomain.CHROM_LEN,
            engine.generation,
            engine.getBest().fitness,
            engine.dedup_cache.hits,
            engine.pop_current.entropy,
        });
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    var it = init.minimal.args.iterate();

    _ = it.next(); // Skip binary path

    const cmd = it.next() orelse {
        try cmdDemo(allocator);
        return;
    };

    if (std.mem.eql(u8, cmd, "solve")) {
        const file_path = it.next() orelse {
            std.debug.print("error: missing file path.\nusage: big-ll solve <file.bigd> [--pop N] [--gens G] [--islands 1|4|8] [--seed S] [--dedup true|false] [--json] [--quiet]\n", .{});
            return;
        };

        var options = SolveOptions{};
        while (it.next()) |arg| {
            if (std.mem.eql(u8, arg, "--pop")) {
                const val_str = it.next() orelse {
                    std.debug.print("error: --pop requires an integer argument\n", .{});
                    return;
                };
                options.pop_size = std.fmt.parseInt(usize, val_str, 10) catch {
                    std.debug.print("error: invalid integer for --pop: '{s}'\n", .{val_str});
                    return;
                };
            } else if (std.mem.eql(u8, arg, "--gens")) {
                const val_str = it.next() orelse {
                    std.debug.print("error: --gens requires an integer argument\n", .{});
                    return;
                };
                options.gens = std.fmt.parseInt(usize, val_str, 10) catch {
                    std.debug.print("error: invalid integer for --gens: '{s}'\n", .{val_str});
                    return;
                };
            } else if (std.mem.eql(u8, arg, "--islands")) {
                const val_str = it.next() orelse {
                    std.debug.print("error: --islands requires 1, 4, or 8\n", .{});
                    return;
                };
                options.islands = std.fmt.parseInt(usize, val_str, 10) catch {
                    std.debug.print("error: invalid integer for --islands: '{s}'\n", .{val_str});
                    return;
                };
            } else if (std.mem.eql(u8, arg, "--seed")) {
                const val_str = it.next() orelse {
                    std.debug.print("error: --seed requires an integer argument\n", .{});
                    return;
                };
                options.seed = std.fmt.parseInt(u64, val_str, 10) catch {
                    std.debug.print("error: invalid integer for --seed: '{s}'\n", .{val_str});
                    return;
                };
            } else if (std.mem.eql(u8, arg, "--dedup")) {
                const val_str = it.next() orelse {
                    std.debug.print("error: --dedup requires true or false\n", .{});
                    return;
                };
                options.enable_dedup = std.mem.eql(u8, val_str, "true") or std.mem.eql(u8, val_str, "1");
            } else if (std.mem.eql(u8, arg, "--save-ir")) {
                const val_str = it.next() orelse {
                    std.debug.print("error: --save-ir requires an output file path\n", .{});
                    return;
                };
                options.save_ir = val_str;
            } else if (std.mem.eql(u8, arg, "--json")) {
                options.json = true;
            } else if (std.mem.eql(u8, arg, "--quiet") or std.mem.eql(u8, arg, "-q")) {
                options.quiet = true;
            } else {
                std.debug.print("warning: unrecognized option '{s}'\n", .{arg});
            }
        }

        try cmdSolve(file_path, options, allocator);
    } else if (std.mem.eql(u8, cmd, "inspect")) {
        const file_path = it.next() orelse {
            std.debug.print("error: missing file path. usage: big-ll inspect <file.bigd>\n", .{});
            return;
        };
        try cmdInspect(file_path, allocator);
    } else if (std.mem.eql(u8, cmd, "generate")) {
        const domain = it.next() orelse {
            std.debug.print("error: missing domain. usage: big-ll generate <domain> <out.bigd>\n", .{});
            return;
        };
        const out_path = it.next() orelse {
            std.debug.print("error: missing output path. usage: big-ll generate <domain> <out.bigd>\n", .{});
            return;
        };
        try cmdGenerate(domain, out_path, allocator);
    } else if (std.mem.eql(u8, cmd, "generate-suite")) {
        const target_dir = it.next() orelse "data/synthetic";
        try cmdGenerateSuite(target_dir, allocator);
    } else if (std.mem.eql(u8, cmd, "ingest-real")) {
        const raw_dir = it.next() orelse "data/raw";
        const out_dir = it.next() orelse "data/real";
        try cmdIngestReal(raw_dir, out_dir, allocator);
    } else if (std.mem.eql(u8, cmd, "demo")) {
        try cmdDemo(allocator);
    } else if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        printUsage();
    } else {
        std.debug.print("error: unknown command '{s}'\n\n", .{cmd});
        printUsage();
    }
}
