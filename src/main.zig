const std = @import("std");
const big_ll = @import("big_ll");
const GeneticEngine = big_ll.GeneticEngine;

fn printUsage() void {
    std.debug.print(
        \\Usage: big-ll <command> [arguments]
        \\
        \\Commands:
        \\  inspect <file.bigd>            Memory-map and inspect a .bigd dataset
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
        \\  big-ll generate logistics sample_logistics.bigd
        \\  big-ll generate-suite data/synthetic
        \\  big-ll inspect data/synthetic/logistics/c32_standard.bigd
        \\  big-ll demo
        \\
    , .{});
}

fn cmdInspect(path: []const u8, allocator: std.mem.Allocator) !void {
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    var inst = big_ll.MemoryMappedInstance.openFile(path_z) catch |err| {
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
        // No args: run default demo
        try cmdDemo(allocator);
        return;
    };

    if (std.mem.eql(u8, cmd, "inspect")) {
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
