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
