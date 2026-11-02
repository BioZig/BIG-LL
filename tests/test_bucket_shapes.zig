const std = @import("std");
const big_ll = @import("big_ll");
const format = big_ll.data_format;

fn alignOffset(offset: usize) usize {
    return (offset + 63) & ~@as(usize, 63);
}

test "Bucket Shapes: Logistics N=47 in bucket 64" {
    const allocator = std.testing.allocator;
    const Dom64 = big_ll.ParameterizedLogisticsDomain(64, 16);

    // Create a synthetic .bigd buffer for 47 customers
    const n: u32 = 47;
    const offset_a = 128;
    const bytes_a = n * @sizeOf(Dom64.Point);
    const offset_b = alignOffset(offset_a + bytes_a);
    const bytes_b = n * @sizeOf(f64);
    const offset_c = alignOffset(offset_b + bytes_b);
    const stride = n + 1;
    const bytes_c = stride * stride * @sizeOf(f64);
    const total_bytes = alignOffset(offset_c + bytes_c);

    const mem = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), total_bytes);
    defer allocator.free(mem);
    @memset(mem, 0);

    const pts: [*]Dom64.Point = @ptrCast(@alignCast(mem.ptr + offset_a));
    const demands: [*]f64 = @ptrCast(@alignCast(mem.ptr + offset_b));
    const dists: [*]f64 = @ptrCast(@alignCast(mem.ptr + offset_c));

    for (0..n) |i| {
        const fi: f64 = @floatFromInt(i);
        pts[i] = .{ .x = 10.0 + fi, .y = 20.0 + fi * 0.5 };
        demands[i] = 2.0 + @mod(fi, 5.0);
    }
    for (0..stride) |i| {
        for (0..stride) |j| {
            const di: f64 = @floatFromInt(i);
            const dj: f64 = @floatFromInt(j);
            dists[i * stride + j] = @abs(di - dj) * 1.5;
        }
    }

    const hdr: *format.BigDHeader = @ptrCast(@alignCast(mem.ptr));
    hdr.* = .{
        .magic = format.Magic,
        .version = format.CurrentVersion,
        .domain_tag = @intFromEnum(format.DomainTag.logistics),
        .flags = 0,
        .total_bytes = @intCast(total_bytes),
        .checksum = 0,
        .param_u32_0 = n,
        .param_u32_1 = 16,
        .param_u32_2 = 0,
        .param_u32_3 = 0,
        .param_u32_4 = 0,
        .param_f64_0 = 100.0, // capacity
        .param_f64_1 = 0.0, // depot x
        .param_f64_2 = 0.0, // depot y
        .offset_section_a = @intCast(offset_a),
        .offset_section_b = @intCast(offset_b),
        .offset_section_c = @intCast(offset_c),
    };

    var inst = try big_ll.BigDInstance.init(mem);
    defer inst.deinit();

    const dom = try Dom64.initFromInstance(&inst);
    try std.testing.expectEqual(@as(usize, 47), dom.actual_customers);

    const chrom = Dom64.Chrom.initZero();
    const route = dom.decode(&chrom);
    const fit = dom.evaluate(route);
    try std.testing.expect(!std.math.isNan(fit));
    try std.testing.expect(fit > 0.0);
}

test "Bucket Shapes: Power Grid N=14 in bucket 16" {
    const allocator = std.testing.allocator;
    const Dom16 = big_ll.ParameterizedPowerGridDomain(16, 16, 300.0);

    const n: u32 = 14;
    const offset_a = 128;
    const bytes_a = n * @sizeOf(Dom16.GeneratorSpec);
    const total_bytes = alignOffset(offset_a + bytes_a);

    const mem = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), total_bytes);
    defer allocator.free(mem);
    @memset(mem, 0);

    const specs: [*]Dom16.GeneratorSpec = @ptrCast(@alignCast(mem.ptr + offset_a));
    for (0..n) |i| {
        specs[i] = .{
            .p_min = 10.0,
            .p_max = 100.0,
            .cost_a = 0.003,
            .cost_b = 8.0,
            .cost_no_load = 50.0,
        };
    }

    const hdr: *format.BigDHeader = @ptrCast(@alignCast(mem.ptr));
    hdr.* = .{
        .magic = format.Magic,
        .version = format.CurrentVersion,
        .domain_tag = @intFromEnum(format.DomainTag.power_grid),
        .flags = 0,
        .total_bytes = @intCast(total_bytes),
        .checksum = 0,
        .param_u32_0 = n,
        .param_u32_1 = 16,
        .param_u32_2 = 0,
        .param_u32_3 = 0,
        .param_u32_4 = 0,
        .param_f64_0 = 500.0, // demand
        .param_f64_1 = 600.0, // reserve
        .param_f64_2 = 0.0,
        .offset_section_a = @intCast(offset_a),
        .offset_section_b = 0,
        .offset_section_c = 0,
    };

    var inst = try big_ll.BigDInstance.init(mem);
    defer inst.deinit();

    const dom = try Dom16.initFromInstance(&inst);
    try std.testing.expectEqual(@as(usize, 14), dom.actual_generators);

    const chrom = Dom16.Chrom.initZero();
    var state = dom.decode(&chrom);
    dom.projectPowerBalance(&state);
    const fit = dom.evaluate(state);
    try std.testing.expect(!std.math.isNan(fit));
    try std.testing.expect(fit > 0.0);
}

test "Bucket Shapes: Finance N=10 in bucket 16" {
    const allocator = std.testing.allocator;
    const Dom16 = big_ll.ParameterizedFinanceDomain(16, 8, 16);

    const n: u32 = 10;
    const offset_a = 128;
    const bytes_a = n * @sizeOf(f64);
    const offset_b = alignOffset(offset_a + bytes_a);
    const bytes_b = n * n * @sizeOf(f64);
    const total_bytes = alignOffset(offset_b + bytes_b);

    const mem = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), total_bytes);
    defer allocator.free(mem);
    @memset(mem, 0);

    const ret: [*]f64 = @ptrCast(@alignCast(mem.ptr + offset_a));
    const cov: [*]f64 = @ptrCast(@alignCast(mem.ptr + offset_b));
    for (0..n) |i| {
        ret[i] = 0.08 + 0.01 * @as(f64, @floatFromInt(i));
        for (0..n) |j| {
            cov[i * n + j] = if (i == j) 0.04 else 0.01;
        }
    }

    const hdr: *format.BigDHeader = @ptrCast(@alignCast(mem.ptr));
    hdr.* = .{
        .magic = format.Magic,
        .version = format.CurrentVersion,
        .domain_tag = @intFromEnum(format.DomainTag.finance),
        .flags = 0,
        .total_bytes = @intCast(total_bytes),
        .checksum = 0,
        .param_u32_0 = n,
        .param_u32_1 = 8,
        .param_u32_2 = 16,
        .param_u32_3 = 0,
        .param_u32_4 = 0,
        .param_f64_0 = 0.02,
        .param_f64_1 = 0.0,
        .param_f64_2 = 0.0,
        .offset_section_a = @intCast(offset_a),
        .offset_section_b = @intCast(offset_b),
        .offset_section_c = 0,
    };

    var inst = try big_ll.BigDInstance.init(mem);
    defer inst.deinit();

    const dom = try Dom16.initFromInstance(&inst);
    try std.testing.expectEqual(@as(usize, 10), dom.actual_assets);

    const chrom = Dom16.Chrom.initZero();
    const state = dom.decode(&chrom);
    const fit = dom.evaluate(state);
    try std.testing.expect(!std.math.isNan(fit));
}

test "Bucket Shapes: Bioinformatics N=200 in bucket 256" {
    const allocator = std.testing.allocator;
    const Dom256 = big_ll.ParameterizedBioinformaticsDomain(256);

    const n: u32 = 200;
    const offset_a = 128;
    const bytes_a = n * @sizeOf(f64);
    const total_bytes = alignOffset(offset_a + bytes_a);

    const mem = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), total_bytes);
    defer allocator.free(mem);
    @memset(mem, 0);

    const me: [*]f64 = @ptrCast(@alignCast(mem.ptr + offset_a));
    for (0..n) |i| {
        me[i] = 0.5 + 0.1 * @as(f64, @floatFromInt(i % 10));
    }

    const hdr: *format.BigDHeader = @ptrCast(@alignCast(mem.ptr));
    hdr.* = .{
        .magic = format.Magic,
        .version = format.CurrentVersion,
        .domain_tag = @intFromEnum(format.DomainTag.bioinformatics),
        .flags = 0,
        .total_bytes = @intCast(total_bytes),
        .checksum = 0,
        .param_u32_0 = n,
        .param_u32_1 = 0,
        .param_u32_2 = 15, // max card
        .param_u32_3 = 0,
        .param_u32_4 = 0,
        .param_f64_0 = 5.0, // penalty weight
        .param_f64_1 = 0.0,
        .param_f64_2 = 0.0,
        .offset_section_a = @intCast(offset_a),
        .offset_section_b = 0,
        .offset_section_c = 0,
    };

    var inst = try big_ll.BigDInstance.init(mem);
    defer inst.deinit();

    const dom = try Dom256.initFromInstance(&inst);
    try std.testing.expectEqual(@as(usize, 200), dom.actual_snps);

    var chrom = Dom256.Chrom.initZero();
    chrom.setBit(5, 1);
    chrom.setBit(50, 1);
    chrom.setBit(150, 1);

    const panel = dom.decode(&chrom);
    const fit = dom.evaluate(panel);
    try std.testing.expect(!std.math.isNan(fit));
    try std.testing.expect(fit > 0.0);

    const fit_direct = dom.evaluateDirect(&chrom);
    try std.testing.expect(!std.math.isNan(fit_direct));
    try std.testing.expect(fit_direct > 0.0);
}

test "Bucket Shapes: Telecom N=24 in bucket 32" {
    const allocator = std.testing.allocator;
    const Dom32 = big_ll.ParameterizedTelecomDomain(32, 4, 2, 4);

    const n: u32 = 24;
    const offset_a = 128;
    const bytes_a = n * n * @sizeOf(f64);
    const total_bytes = alignOffset(offset_a + bytes_a);

    const mem = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), total_bytes);
    defer allocator.free(mem);
    @memset(mem, 0);

    const mat: [*]f64 = @ptrCast(@alignCast(mem.ptr + offset_a));
    for (0..n) |i| {
        for (0..n) |j| {
            mat[i * n + j] = if (i == j) 1.0 else 0.1;
        }
    }

    const hdr: *format.BigDHeader = @ptrCast(@alignCast(mem.ptr));
    hdr.* = .{
        .magic = format.Magic,
        .version = format.CurrentVersion,
        .domain_tag = @intFromEnum(format.DomainTag.telecom),
        .flags = 0,
        .total_bytes = @intCast(total_bytes),
        .checksum = 0,
        .param_u32_0 = n,
        .param_u32_1 = 4, // 4 channels
        .param_u32_2 = 2,
        .param_u32_3 = 4,
        .param_u32_4 = 0,
        .param_f64_0 = 0.05, // noise
        .param_f64_1 = 0.0,
        .param_f64_2 = 0.0,
        .offset_section_a = @intCast(offset_a),
        .offset_section_b = 0,
        .offset_section_c = 0,
    };

    var inst = try big_ll.BigDInstance.init(mem);
    defer inst.deinit();

    const dom = try Dom32.initFromInstance(&inst);
    try std.testing.expectEqual(@as(usize, 24), dom.actual_users);

    const chrom = Dom32.Chrom.initZero();
    const alloc_state = dom.decode(&chrom);
    const fit = dom.evaluate(alloc_state);
    try std.testing.expect(!std.math.isNan(fit));
}
