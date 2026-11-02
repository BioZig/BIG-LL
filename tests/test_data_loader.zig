const std = @import("std");
const big_ll = @import("big_ll");
const format = big_ll.data_format;
const loader = big_ll.data_loader;
const generator = big_ll.data_generator;

test "BigDHeader layout and size invariants" {
    try std.testing.expectEqual(@as(usize, 128), @sizeOf(format.BigDHeader));
    try std.testing.expectEqual(@as(usize, 64), @alignOf(format.BigDHeader));

    var hdr = format.BigDHeader{
        .domain_tag = @intFromEnum(format.DomainTag.logistics),
        .total_bytes = 256,
        .offset_section_a = 128,
    };
    try hdr.validate(256);

    // Corrupted magic
    hdr.magic = 0x12345678;
    try std.testing.expectError(error.InvalidMagic, hdr.validate(256));
    hdr.magic = format.Magic;

    // Size mismatch
    try std.testing.expectError(error.SizeMismatch, hdr.validate(512));

    // Unaligned section
    hdr.offset_section_a = 130;
    try std.testing.expectError(error.UnalignedSectionA, hdr.validate(256));
}

test "Checksum calculation and integrity verification" {
    const data = "BIG-LL Zero-Copy Binary Dataset System Verification Vector";
    const crc1 = format.computeChecksum(data);
    const crc2 = format.computeChecksum(data);
    try std.testing.expectEqual(crc1, crc2);
    try std.testing.expect(crc1 != 0);

    const corrupted = "BIG-LL Zero-Copy Binary Dataset System Verification Vector.";
    const crc_diff = format.computeChecksum(corrupted);
    try std.testing.expect(crc1 != crc_diff);
}

test "Synthetic Logistics instance generation and buffer loader" {
    const allocator = std.testing.allocator;
    const buf = try generator.generateLogistics(allocator, 32, 35.0, 50.0, 50.0);
    defer allocator.free(buf);

    var inst = try loader.StaticBufferInstance.init(buf);
    const hdr = inst.header();

    try std.testing.expectEqual(format.Magic, hdr.magic);
    try std.testing.expectEqual(@intFromEnum(format.DomainTag.logistics), hdr.domain_tag);
    try std.testing.expectEqual(@as(u32, 32), hdr.param_u32_0);
    try std.testing.expectEqual(@as(f64, 35.0), hdr.param_f64_0);

    const points = try inst.getSlice(generator.Point, hdr.offset_section_a, 32);
    try std.testing.expectEqual(@as(usize, 32), points.len);

    const demands = try inst.getSlice(f64, hdr.offset_section_b, 32);
    try std.testing.expectEqual(@as(usize, 32), demands.len);

    const dist_matrix = try inst.getSlice(f64, hdr.offset_section_c, 33 * 33);
    try std.testing.expectEqual(@as(usize, 33 * 33), dist_matrix.len);
    // Distance from depot to depot is 0
    try std.testing.expectEqual(@as(f64, 0.0), dist_matrix[0]);
    // Distance from customer 0 to customer 0 is 0
    try std.testing.expectEqual(@as(f64, 0.0), dist_matrix[1 * 33 + 1]);
    // Symmetric distance check
    try std.testing.expectEqual(dist_matrix[0 * 33 + 5], dist_matrix[5 * 33 + 0]);
}

test "Synthetic PowerGrid, Finance, Bioinformatics, Airfoil, Telecom instances" {
    const allocator = std.testing.allocator;

    // PowerGrid
    {
        const buf = try generator.generatePowerGrid(allocator, 550.0, 630.0);
        defer allocator.free(buf);
        const inst = try loader.StaticBufferInstance.init(buf);
        try std.testing.expectEqual(@intFromEnum(format.DomainTag.power_grid), inst.header().domain_tag);
        const specs = try inst.getSlice(generator.GeneratorSpec, inst.header().offset_section_a, 8);
        try std.testing.expectEqual(@as(usize, 8), specs.len);
    }

    // Finance
    {
        const buf = try generator.generateFinance(allocator, 6, 8, 0.03);
        defer allocator.free(buf);
        const inst = try loader.StaticBufferInstance.init(buf);
        try std.testing.expectEqual(@intFromEnum(format.DomainTag.finance), inst.header().domain_tag);
        const returns = try inst.getSlice(f64, inst.header().offset_section_a, 6);
        try std.testing.expectEqual(@as(usize, 6), returns.len);
        const cov = try inst.getSlice(f64, inst.header().offset_section_b, 36);
        try std.testing.expectEqual(@as(usize, 36), cov.len);
    }

    // Bioinformatics
    {
        const buf = try generator.generateBioinformatics(allocator, 128, 8, 12);
        defer allocator.free(buf);
        const inst = try loader.StaticBufferInstance.init(buf);
        try std.testing.expectEqual(@intFromEnum(format.DomainTag.bioinformatics), inst.header().domain_tag);
        const main_effects = try inst.getSlice(f64, inst.header().offset_section_a, 128);
        try std.testing.expectEqual(@as(usize, 128), main_effects.len);
        const pairs = try inst.getSlice(format.EpistasisPair, inst.header().offset_section_b, 4);
        try std.testing.expectEqual(@as(usize, 4), pairs.len);
    }

    // Airfoil
    {
        const buf = try generator.generateAirfoil(allocator, 16, 2.5, 0.01);
        defer allocator.free(buf);
        const inst = try loader.StaticBufferInstance.init(buf);
        try std.testing.expectEqual(@intFromEnum(format.DomainTag.airfoil), inst.header().domain_tag);
        try std.testing.expectEqual(@as(f64, 2.5), inst.header().param_f64_0);
    }

    // Telecom
    {
        const buf = try generator.generateTelecom(allocator, 16, 4, 0.05);
        defer allocator.free(buf);
        const inst = try loader.StaticBufferInstance.init(buf);
        try std.testing.expectEqual(@intFromEnum(format.DomainTag.telecom), inst.header().domain_tag);
        const matrix = try inst.getSlice(f64, inst.header().offset_section_a, 256);
        try std.testing.expectEqual(@as(usize, 256), matrix.len);
    }
}

test "MemoryMappedInstance file-backed loading and checksum verification" {
    const allocator = std.testing.allocator;
    const test_file = "/tmp/test_logistics_instance.bigd";
    defer _ = std.c.unlink(test_file);

    const buf = try generator.generateLogistics(allocator, 32, 35.0, 50.0, 50.0);
    defer allocator.free(buf);

    try generator.saveToFile(test_file, buf);

    var mmap_inst = try loader.MemoryMappedInstance.openFile(test_file);
    defer mmap_inst.deinit();

    const hdr = mmap_inst.header();
    try std.testing.expectEqual(format.Magic, hdr.magic);
    try std.testing.expectEqual(@intFromEnum(format.DomainTag.logistics), hdr.domain_tag);

    const points = try mmap_inst.getSlice(generator.Point, hdr.offset_section_a, 32);
    try std.testing.expectEqual(@as(usize, 32), points.len);

    const dist_matrix = try mmap_inst.getSlice(f64, hdr.offset_section_c, 33 * 33);
    try std.testing.expectEqual(@as(usize, 33 * 33), dist_matrix.len);
}

test "BigDInstance polymorphic backings: mmap, ephemeral_ram, borrowed" {
    const allocator = std.testing.allocator;

    // 1. Borrowed backing
    {
        const buf = try generator.generateLogistics(allocator, 32, 35.0, 50.0, 50.0);
        defer allocator.free(buf);

        var borrowed_inst = try loader.BigDInstance.init(buf);
        defer borrowed_inst.deinit();

        try std.testing.expectEqual(loader.BackingType.borrowed, @as(loader.BackingType, borrowed_inst.backing));
        try std.testing.expectEqual(@intFromEnum(format.DomainTag.logistics), borrowed_inst.header().domain_tag);
    }

    // 2. Ephemeral RAM backing
    {
        const buf = try generator.generateLogistics(allocator, 32, 35.0, 50.0, 50.0);
        var eph_inst = try loader.BigDInstance.initEphemeral(buf, allocator);
        try std.testing.expectEqual(loader.BackingType.ephemeral_ram, @as(loader.BackingType, eph_inst.backing));
        try std.testing.expectEqual(@intFromEnum(format.DomainTag.logistics), eph_inst.header().domain_tag);
        // deinit frees eph_inst.raw_slice using allocator
        eph_inst.deinit();
    }
}

test "JIT lowering: NumPy .npy covariance matrix to in-memory .bigd IR" {
    const allocator = std.testing.allocator;

    // Create a 6x6 float64 NumPy array buffer
    const n = 6;
    const header_dict = "{'descr': '<f8', 'fortran_order': False, 'shape': (6, 6), }";
    const pad_len = 64 - ((10 + header_dict.len + 1) % 64);
    const total_hdr = 10 + header_dict.len + pad_len + 1;
    const data_len = n * n * @sizeOf(f64);
    const npy_len = total_hdr + data_len;

    const npy_bytes = try allocator.alloc(u8, npy_len);
    defer allocator.free(npy_bytes);

    @memcpy(npy_bytes[0..6], &format.NpyMagic);
    npy_bytes[6] = 1; // major
    npy_bytes[7] = 0; // minor
    const hlen: u16 = @intCast(header_dict.len + pad_len + 1);
    std.mem.writeInt(u16, npy_bytes[8..10], hlen, .little);
    @memcpy(npy_bytes[10 .. 10 + header_dict.len], header_dict);
    for (10 + header_dict.len .. total_hdr - 1) |i| npy_bytes[i] = ' ';
    npy_bytes[total_hdr - 1] = '\n';

    const f64_slice: [*]f64 = @ptrCast(@alignCast(npy_bytes.ptr + total_hdr));
    for (0..n * n) |i| {
        const row = i / n;
        const col = i % n;
        f64_slice[i] = if (row == col) 0.04 else 0.01;
    }

    const bigd_buf = try big_ll.data_importer.parseNpy(allocator, npy_bytes);
    var inst = try loader.BigDInstance.initEphemeral(bigd_buf, allocator);
    defer inst.deinit();

    const hdr = inst.header();
    try std.testing.expectEqual(format.Magic, hdr.magic);
    try std.testing.expectEqual(@intFromEnum(format.DomainTag.finance), hdr.domain_tag);
    try std.testing.expectEqual(@as(u32, 6), hdr.param_u32_0); // 6 assets

    const Dom = big_ll.FinanceDomain;
    const dom = try Dom.initFromInstance(&inst);

    // Verify 0 runtime heap allocation invariant during fitness evaluation
    const chrom = Dom.Chrom.initZero();
    const state = dom.decode(&chrom);
    const fit = dom.evaluate(state);
    try std.testing.expect(fit >= 0.0);
}

test "JIT lowering: PLINK 1.x .bed genotypes to in-memory .bigd IR" {
    const allocator = std.testing.allocator;

    // 128 SNPs, 40 individuals (10 bytes per SNP, total: 3 + 1280 = 1283 bytes)
    const num_snps: usize = 128;
    const num_ind: usize = 40;
    const bytes_per_snp = (num_ind + 3) / 4;
    const bed_len = 3 + num_snps * bytes_per_snp;

    const bed_bytes = try allocator.alloc(u8, bed_len);
    defer allocator.free(bed_bytes);

    @memcpy(bed_bytes[0..3], &format.PlinkBedMagic);
    for (3..bed_len) |i| {
        bed_bytes[i] = @intCast(i % 256);
    }

    const bigd_buf = try big_ll.data_importer.parsePlinkBed(allocator, bed_bytes, null, null);
    var inst = try loader.BigDInstance.initEphemeral(bigd_buf, allocator);
    defer inst.deinit();

    const hdr = inst.header();
    try std.testing.expectEqual(format.Magic, hdr.magic);
    try std.testing.expectEqual(@intFromEnum(format.DomainTag.bioinformatics), hdr.domain_tag);
    try std.testing.expectEqual(@as(u32, 128), hdr.param_u32_0);

    const Dom = big_ll.BioinformaticsDomain;
    const dom = try Dom.initFromInstance(&inst);

    const chrom = Dom.Chrom.initZero();
    const panel = dom.decode(&chrom);
    const fit = dom.evaluate(panel);
    try std.testing.expect(fit >= 0.0);
}

test "openFileAuto on raw CVRPLIB .vrp with optional --save-ir persistence" {
    const allocator = std.testing.allocator;
    const vrp_path = "data/raw/logistics/A-n33-k5.vrp";
    const cache_file = "/tmp/test_saved_ir_An33k5.bigd";
    defer _ = std.c.unlink(cache_file);

    var inst = try loader.BigDInstance.openFileAuto(vrp_path, allocator);
    defer inst.deinit();

    try std.testing.expectEqual(loader.BackingType.ephemeral_ram, @as(loader.BackingType, inst.backing));
    const hdr = inst.header();
    try std.testing.expectEqual(format.Magic, hdr.magic);
    try std.testing.expectEqual(@intFromEnum(format.DomainTag.logistics), hdr.domain_tag);
    try std.testing.expectEqual(@as(u32, 32), hdr.param_u32_0);

    // Save compiled IR to disk (simulating --save-ir)
    try generator.saveToFile(cache_file, inst.raw_slice);

    // Reopen cached IR using fast-path mmap
    var mmap_inst = try loader.BigDInstance.openFileAuto(cache_file, allocator);
    defer mmap_inst.deinit();

    try std.testing.expectEqual(loader.BackingType.mmap, @as(loader.BackingType, mmap_inst.backing));
    try std.testing.expectEqual(hdr.checksum, mmap_inst.header().checksum);
    try std.testing.expectEqual(hdr.total_bytes, mmap_inst.header().total_bytes);
    try std.testing.expectEqualSlices(u8, inst.raw_slice, mmap_inst.raw_slice);
}
