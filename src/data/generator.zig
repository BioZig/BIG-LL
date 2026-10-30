const std = @import("std");
const format = @import("format.zig");
const BigDHeader = format.BigDHeader;
const DomainTag = format.DomainTag;

pub const GeneratorError = error{
    OutOfMemory,
    InvalidParameter,
    WriteFailed,
};

pub const Point = extern struct {
    x: f64,
    y: f64,
};

pub const GeneratorSpec = extern struct {
    p_min: f64,
    p_max: f64,
    cost_a: f64,
    cost_b: f64,
    cost_no_load: f64,
};

fn alignOffset(offset: usize) usize {
    return std.mem.alignForward(usize, offset, 64);
}

pub fn generateLogistics(
    allocator: std.mem.Allocator,
    num_customers: usize,
    capacity: f64,
    depot_x: f64,
    depot_y: f64,
) ![]align(64) u8 {
    if (num_customers == 0) return GeneratorError.InvalidParameter;

    const offset_points: usize = 128;
    const points_bytes = num_customers * @sizeOf(Point);
    const offset_demands = alignOffset(offset_points + points_bytes);
    const demands_bytes = num_customers * @sizeOf(f64);

    // Matrix size is (num_customers + 1) * (num_customers + 1) where index 0 is depot
    const total_nodes = num_customers + 1;
    const offset_dist_matrix = alignOffset(offset_demands + demands_bytes);
    const dist_bytes = total_nodes * total_nodes * @sizeOf(f64);
    const total_bytes = alignOffset(offset_dist_matrix + dist_bytes);

    const buffer = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), total_bytes);
    @memset(buffer, 0);

    const points: [*]Point = @ptrCast(@alignCast(buffer.ptr + offset_points));
    const demands: [*]f64 = @ptrCast(@alignCast(buffer.ptr + offset_demands));
    const dist_matrix: [*]f64 = @ptrCast(@alignCast(buffer.ptr + offset_dist_matrix));

    // Generate customers
    const depot = Point{ .x = depot_x, .y = depot_y };
    for (0..num_customers) |i| {
        const fi: f64 = @floatFromInt(i);
        const angle = fi * (2.0 * std.math.pi / @as(f64, @floatFromInt(num_customers)));
        const radius = 20.0 + 15.0 * @sin(fi * 0.5);
        points[i] = .{
            .x = depot_x + radius * @cos(angle),
            .y = depot_y + radius * @sin(angle),
        };
        demands[i] = 3.0 + @mod(fi * 7.0, 8.0);
    }

    // Build (total_nodes x total_nodes) distance matrix
    // Index 0: depot, Index 1..num_customers: customer i-1
    for (0..total_nodes) |i| {
        const p1 = if (i == 0) depot else points[i - 1];
        for (0..total_nodes) |j| {
            const p2 = if (j == 0) depot else points[j - 1];
            const dx = p1.x - p2.x;
            const dy = p1.y - p2.y;
            dist_matrix[i * total_nodes + j] = @sqrt(dx * dx + dy * dy);
        }
    }

    const checksum = format.computeChecksum(buffer[128..total_bytes]);

    const hdr: *BigDHeader = @ptrCast(buffer.ptr);
    hdr.* = .{
        .magic = format.Magic,
        .version = format.CurrentVersion,
        .domain_tag = @intFromEnum(DomainTag.logistics),
        .flags = 0,
        .total_bytes = @intCast(total_bytes),
        .checksum = checksum,
        .param_u32_4 = 0,
        .param_u32_0 = @intCast(num_customers),
        .param_u32_1 = @intCast(total_nodes),
        .param_u32_2 = 0,
        .param_u32_3 = 0,
        .param_f64_0 = capacity,
        .param_f64_1 = depot_x,
        .param_f64_2 = depot_y,
        .param_f64_3 = 0.0,
        .offset_section_a = offset_points,
        .offset_section_b = offset_demands,
        .offset_section_c = offset_dist_matrix,
        .reserved = [_]u64{0} ** 3,
    };

    return buffer;
}

pub fn generatePowerGrid(
    allocator: std.mem.Allocator,
    demand: f64,
    reserve_target: f64,
) ![]align(64) u8 {
    const base_specs = [_]GeneratorSpec{
        .{ .p_min = 20.0, .p_max = 120.0, .cost_a = 0.003, .cost_b = 8.5, .cost_no_load = 150.0 },
        .{ .p_min = 15.0, .p_max = 100.0, .cost_a = 0.004, .cost_b = 9.0, .cost_no_load = 120.0 },
        .{ .p_min = 25.0, .p_max = 140.0, .cost_a = 0.002, .cost_b = 7.8, .cost_no_load = 200.0 },
        .{ .p_min = 10.0, .p_max = 80.0, .cost_a = 0.006, .cost_b = 10.2, .cost_no_load = 80.0 },
        .{ .p_min = 30.0, .p_max = 150.0, .cost_a = 0.0025, .cost_b = 8.0, .cost_no_load = 220.0 },
        .{ .p_min = 20.0, .p_max = 110.0, .cost_a = 0.0035, .cost_b = 8.8, .cost_no_load = 140.0 },
        .{ .p_min = 10.0, .p_max = 90.0, .cost_a = 0.005, .cost_b = 9.8, .cost_no_load = 90.0 },
        .{ .p_min = 15.0, .p_max = 70.0, .cost_a = 0.007, .cost_b = 11.0, .cost_no_load = 60.0 },
    };
    const num_generators = base_specs.len;

    const offset_specs: usize = 128;
    const specs_bytes = num_generators * @sizeOf(GeneratorSpec);
    const total_bytes = alignOffset(offset_specs + specs_bytes);

    const buffer = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), total_bytes);
    @memset(buffer, 0);

    const specs: [*]GeneratorSpec = @ptrCast(@alignCast(buffer.ptr + offset_specs));
    @memcpy(specs[0..num_generators], &base_specs);

    const checksum = format.computeChecksum(buffer[128..total_bytes]);

    const hdr: *BigDHeader = @ptrCast(buffer.ptr);
    hdr.* = .{
        .magic = format.Magic,
        .version = format.CurrentVersion,
        .domain_tag = @intFromEnum(DomainTag.power_grid),
        .flags = 0,
        .total_bytes = @intCast(total_bytes),
        .checksum = checksum,
        .param_u32_4 = 0,
        .param_u32_0 = @intCast(num_generators),
        .param_u32_1 = 0,
        .param_u32_2 = 0,
        .param_u32_3 = 0,
        .param_f64_0 = demand,
        .param_f64_1 = reserve_target,
        .param_f64_2 = 0.0,
        .param_f64_3 = 0.0,
        .offset_section_a = offset_specs,
        .offset_section_b = 0,
        .offset_section_c = 0,
        .reserved = [_]u64{0} ** 3,
    };

    return buffer;
}

pub fn generateFinance(
    allocator: std.mem.Allocator,
    num_assets: usize,
    num_rules: usize,
    risk_free_rate: f64,
) ![]align(64) u8 {
    return generateFinanceEx(allocator, num_assets, num_rules, risk_free_rate, 0.35);
}

pub fn generateFinanceEx(
    allocator: std.mem.Allocator,
    num_assets: usize,
    num_rules: usize,
    risk_free_rate: f64,
    base_correlation: f64,
) ![]align(64) u8 {
    if (num_assets == 0) return GeneratorError.InvalidParameter;

    const offset_returns: usize = 128;
    const returns_bytes = num_assets * @sizeOf(f64);
    const offset_cov = alignOffset(offset_returns + returns_bytes);
    const cov_bytes = num_assets * num_assets * @sizeOf(f64);
    const total_bytes = alignOffset(offset_cov + cov_bytes);

    const buffer = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), total_bytes);
    @memset(buffer, 0);

    const returns: [*]f64 = @ptrCast(@alignCast(buffer.ptr + offset_returns));
    const cov: [*]f64 = @ptrCast(@alignCast(buffer.ptr + offset_cov));

    // Base expected returns and standard deviations
    for (0..num_assets) |i| {
        const fi: f64 = @floatFromInt(i);
        returns[i] = 0.05 + 0.02 * @mod(fi * 3.7, 7.0);
    }

    for (0..num_assets) |i| {
        const fi: f64 = @floatFromInt(i);
        const s_i = 0.10 + 0.04 * @mod(fi * 2.3, 5.0);
        for (0..num_assets) |j| {
            const fj: f64 = @floatFromInt(j);
            const s_j = 0.10 + 0.04 * @mod(fj * 2.3, 5.0);
            const corr: f64 = if (i == j) 1.0 else base_correlation + 0.05 * @sin(fi + fj);
            cov[i * num_assets + j] = corr * s_i * s_j;
        }
    }

    const checksum = format.computeChecksum(buffer[128..total_bytes]);

    const hdr: *BigDHeader = @ptrCast(buffer.ptr);
    hdr.* = .{
        .magic = format.Magic,
        .version = format.CurrentVersion,
        .domain_tag = @intFromEnum(DomainTag.finance),
        .flags = 0,
        .total_bytes = @intCast(total_bytes),
        .checksum = checksum,
        .param_u32_4 = 0,
        .param_u32_0 = @intCast(num_assets),
        .param_u32_1 = @intCast(num_rules),
        .param_u32_2 = 0,
        .param_u32_3 = 0,
        .param_f64_0 = risk_free_rate,
        .param_f64_1 = 0.0,
        .param_f64_2 = 0.0,
        .param_f64_3 = 0.0,
        .offset_section_a = offset_returns,
        .offset_section_b = offset_cov,
        .offset_section_c = 0,
        .reserved = [_]u64{0} ** 3,
    };

    return buffer;
}

pub fn generateBioinformatics(
    allocator: std.mem.Allocator,
    num_snps: usize,
    target_card: usize,
    max_card: usize,
) ![]align(64) u8 {
    return generateBioinformaticsEx(allocator, num_snps, target_card, max_card, 1);
}

pub fn generateBioinformaticsEx(
    allocator: std.mem.Allocator,
    num_snps: usize,
    target_card: usize,
    max_card: usize,
    pair_mode: u8,
) ![]align(64) u8 {
    if (num_snps < 128) return GeneratorError.InvalidParameter;

    const base_pairs_std = [_]format.EpistasisPair{
        .{ .idx1 = 5, .idx2 = 23, .weight = 15.0 },
        .{ .idx1 = 12, .idx2 = 67, .weight = 15.0 },
        .{ .idx1 = 34, .idx2 = 89, .weight = 15.0 },
        .{ .idx1 = 45, .idx2 = 110, .weight = 15.0 },
    };

    const base_pairs_dense = [_]format.EpistasisPair{
        .{ .idx1 = 5, .idx2 = 23, .weight = 15.0 },
        .{ .idx1 = 12, .idx2 = 67, .weight = 15.0 },
        .{ .idx1 = 34, .idx2 = 89, .weight = 15.0 },
        .{ .idx1 = 45, .idx2 = 110, .weight = 15.0 },
        .{ .idx1 = 18, .idx2 = 54, .weight = 20.0 },
        .{ .idx1 = 29, .idx2 = 93, .weight = 20.0 },
        .{ .idx1 = 61, .idx2 = 102, .weight = 20.0 },
        .{ .idx1 = 75, .idx2 = 120, .weight = 20.0 },
    };

    const base_triplets_std = [_]format.EpistasisTriplet{
        .{ .idx1 = 7, .idx2 = 42, .idx3 = 99, .pad = 0, .weight = 35.0 },
    };

    const base_triplets_dense = [_]format.EpistasisTriplet{
        .{ .idx1 = 7, .idx2 = 42, .idx3 = 99, .pad = 0, .weight = 35.0 },
        .{ .idx1 = 14, .idx2 = 38, .idx3 = 85, .pad = 0, .weight = 40.0 },
        .{ .idx1 = 21, .idx2 = 63, .idx3 = 105, .pad = 0, .weight = 45.0 },
    };

    const pairs_slice: []const format.EpistasisPair = switch (pair_mode) {
        0 => &[_]format.EpistasisPair{},
        1 => &base_pairs_std,
        else => &base_pairs_dense,
    };

    const triplets_slice: []const format.EpistasisTriplet = switch (pair_mode) {
        0 => &[_]format.EpistasisTriplet{},
        1 => &base_triplets_std,
        else => &base_triplets_dense,
    };

    const num_pairs = pairs_slice.len;
    const num_triplets = triplets_slice.len;

    const offset_main_effects: usize = 128;
    const main_effects_bytes = num_snps * @sizeOf(f64);
    const offset_pairs = alignOffset(offset_main_effects + main_effects_bytes);
    const pairs_bytes = num_pairs * @sizeOf(format.EpistasisPair);
    const offset_triplets = alignOffset(offset_pairs + pairs_bytes);
    const triplets_bytes = num_triplets * @sizeOf(format.EpistasisTriplet);
    const total_bytes = alignOffset(offset_triplets + triplets_bytes);

    const buffer = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), total_bytes);
    @memset(buffer, 0);

    const main_effects: [*]f64 = @ptrCast(@alignCast(buffer.ptr + offset_main_effects));
    for (0..num_snps) |i| {
        const fi: f64 = @floatFromInt(i);
        main_effects[i] = 0.2 + 0.1 * @sin(fi * 0.3);
    }

    if (num_pairs > 0) {
        const pairs: [*]format.EpistasisPair = @ptrCast(@alignCast(buffer.ptr + offset_pairs));
        @memcpy(pairs[0..num_pairs], pairs_slice);
    }

    if (num_triplets > 0) {
        const triplets: [*]format.EpistasisTriplet = @ptrCast(@alignCast(buffer.ptr + offset_triplets));
        @memcpy(triplets[0..num_triplets], triplets_slice);
    }

    const checksum = format.computeChecksum(buffer[128..total_bytes]);

    const hdr: *BigDHeader = @ptrCast(buffer.ptr);
    hdr.* = .{
        .magic = format.Magic,
        .version = format.CurrentVersion,
        .domain_tag = @intFromEnum(DomainTag.bioinformatics),
        .flags = 0,
        .total_bytes = @intCast(total_bytes),
        .checksum = checksum,
        .param_u32_4 = @intCast(num_triplets),
        .param_u32_0 = @intCast(num_snps),
        .param_u32_1 = @intCast(target_card),
        .param_u32_2 = @intCast(max_card),
        .param_u32_3 = @intCast(num_pairs),
        .param_f64_0 = 8.0, // cardinality penalty weight
        .param_f64_1 = 0.0,
        .param_f64_2 = 0.0,
        .param_f64_3 = 0.0,
        .offset_section_a = offset_main_effects,
        .offset_section_b = if (num_pairs > 0) offset_pairs else 0,
        .offset_section_c = if (num_triplets > 0) offset_triplets else 0,
        .reserved = [_]u64{0} ** 3,
    };

    return buffer;
}

pub fn generateAirfoil(
    allocator: std.mem.Allocator,
    num_points: usize,
    cl_mult: f64,
    cd_base: f64,
) ![]align(64) u8 {
    if (num_points != 16) return GeneratorError.InvalidParameter;

    const total_bytes: usize = 128; // Airfoil instance parameters fit entirely within the 128-byte header
    const buffer = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), total_bytes);
    @memset(buffer, 0);

    const hdr: *BigDHeader = @ptrCast(buffer.ptr);
    hdr.* = .{
        .magic = format.Magic,
        .version = format.CurrentVersion,
        .domain_tag = @intFromEnum(DomainTag.airfoil),
        .flags = 0,
        .total_bytes = @intCast(total_bytes),
        .checksum = 0,
        .param_u32_4 = 0,
        .param_u32_0 = @intCast(num_points),
        .param_u32_1 = 0,
        .param_u32_2 = 0,
        .param_u32_3 = 0,
        .param_f64_0 = cl_mult,
        .param_f64_1 = cd_base,
        .param_f64_2 = 50.0, // smoothness weight
        .param_f64_3 = 0.05, // thickness weight
        .offset_section_a = 128,
        .offset_section_b = 0,
        .offset_section_c = 0,
        .reserved = [_]u64{0} ** 3,
    };

    return buffer;
}

pub fn generateTelecom(
    allocator: std.mem.Allocator,
    num_users: usize,
    num_channels: usize,
    noise_power: f64,
) ![]align(64) u8 {
    if (num_users == 0 or num_channels == 0) return GeneratorError.InvalidParameter;

    const offset_matrix: usize = 128;
    const matrix_bytes = num_users * num_users * @sizeOf(f64);
    const total_bytes = alignOffset(offset_matrix + matrix_bytes);

    const buffer = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), total_bytes);
    @memset(buffer, 0);

    const matrix: [*]f64 = @ptrCast(@alignCast(buffer.ptr + offset_matrix));
    for (0..num_users) |i| {
        for (0..num_users) |j| {
            if (i == j) {
                matrix[i * num_users + j] = 1.0;
            } else {
                const diff: f64 = @floatFromInt(if (i > j) i - j else j - i);
                matrix[i * num_users + j] = 0.4 / (1.0 + diff * diff);
            }
        }
    }

    const checksum = format.computeChecksum(buffer[128..total_bytes]);

    const hdr: *BigDHeader = @ptrCast(buffer.ptr);
    hdr.* = .{
        .magic = format.Magic,
        .version = format.CurrentVersion,
        .domain_tag = @intFromEnum(DomainTag.telecom),
        .flags = 0,
        .total_bytes = @intCast(total_bytes),
        .checksum = checksum,
        .param_u32_4 = 0,
        .param_u32_0 = @intCast(num_users),
        .param_u32_1 = @intCast(num_channels),
        .param_u32_2 = 0,
        .param_u32_3 = 0,
        .param_f64_0 = noise_power,
        .param_f64_1 = 0.25, // sinr threshold
        .param_f64_2 = 0.0,
        .param_f64_3 = 0.0,
        .offset_section_a = offset_matrix,
        .offset_section_b = 0,
        .offset_section_c = 0,
        .reserved = [_]u64{0} ** 3,
    };

    return buffer;
}

pub fn saveToFile(file_path: [:0]const u8, data: []align(64) const u8) !void {
    const fd = try std.posix.openat(
        std.posix.AT.FDCWD,
        file_path,
        .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true },
        0o666,
    );
    defer _ = std.c.close(fd);

    var total_written: usize = 0;
    while (total_written < data.len) {
        const rc = std.c.write(fd, data.ptr + total_written, data.len - total_written);
        if (rc < 0) return GeneratorError.WriteFailed;
        total_written += @intCast(rc);
    }
}

pub fn makePath(allocator: std.mem.Allocator, path: []const u8) !void {
    if (path.len == 0) return;
    var i: usize = 0;
    while (i < path.len) : (i += 1) {
        if (path[i] == '/' and i > 0) {
            const sub = try allocator.dupeZ(u8, path[0..i]);
            defer allocator.free(sub);
            const rc = std.c.mkdir(sub.ptr, 0o755);
            if (rc != 0) {
                const err = std.c._errno().*;
                if (err != 17) return GeneratorError.WriteFailed; // EEXIST = 17
            }
        }
    }
    const full = try allocator.dupeZ(u8, path);
    defer allocator.free(full);
    const rc = std.c.mkdir(full.ptr, 0o755);
    if (rc != 0) {
        const err = std.c._errno().*;
        if (err != 17) return GeneratorError.WriteFailed;
    }
}

fn saveHelper(allocator: std.mem.Allocator, dir_path: []const u8, filename: []const u8, data: []align(64) const u8) !void {
    try makePath(allocator, dir_path);
    const full_path = try std.fs.path.join(allocator, &[_][]const u8{ dir_path, filename });
    defer allocator.free(full_path);
    const full_path_z = try allocator.dupeZ(u8, full_path);
    defer allocator.free(full_path_z);
    try saveToFile(full_path_z, data);
}

pub fn generateBenchmarkSuite(allocator: std.mem.Allocator, base_dir: []const u8) !void {
    // 1. Logistics (CVRP)
    {
        const dir = try std.fs.path.join(allocator, &[_][]const u8{ base_dir, "logistics" });
        defer allocator.free(dir);
        const slack = try generateLogistics(allocator, 32, 60.0, 50.0, 50.0);
        defer allocator.free(slack);
        try saveHelper(allocator, dir, "c32_slack.bigd", slack);

        const standard = try generateLogistics(allocator, 32, 35.0, 50.0, 50.0);
        defer allocator.free(standard);
        try saveHelper(allocator, dir, "c32_standard.bigd", standard);

        const stress = try generateLogistics(allocator, 32, 20.0, 50.0, 50.0);
        defer allocator.free(stress);
        try saveHelper(allocator, dir, "c32_stress.bigd", stress);
    }

    // 2. Power Grid (MINLP Unit Commitment)
    {
        const dir = try std.fs.path.join(allocator, &[_][]const u8{ base_dir, "power_grid" });
        defer allocator.free(dir);
        const slack = try generatePowerGrid(allocator, 350.0, 400.0);
        defer allocator.free(slack);
        try saveHelper(allocator, dir, "g8_slack.bigd", slack);

        const standard = try generatePowerGrid(allocator, 550.0, 630.0);
        defer allocator.free(standard);
        try saveHelper(allocator, dir, "g8_standard.bigd", standard);

        const stress = try generatePowerGrid(allocator, 720.0, 800.0);
        defer allocator.free(stress);
        try saveHelper(allocator, dir, "g8_stress.bigd", stress);
    }

    // 3. Finance (Portfolio Allocation & Rules)
    {
        const dir = try std.fs.path.join(allocator, &[_][]const u8{ base_dir, "finance" });
        defer allocator.free(dir);
        const slack = try generateFinanceEx(allocator, 6, 8, 0.01, 0.10);
        defer allocator.free(slack);
        try saveHelper(allocator, dir, "a6_slack.bigd", slack);

        const standard = try generateFinanceEx(allocator, 6, 8, 0.03, 0.35);
        defer allocator.free(standard);
        try saveHelper(allocator, dir, "a6_standard.bigd", standard);

        const stress = try generateFinanceEx(allocator, 6, 8, 0.06, 0.75);
        defer allocator.free(stress);
        try saveHelper(allocator, dir, "a6_stress.bigd", stress);
    }

    // 4. Bioinformatics (Epistasis & Biomarkers)
    {
        const dir = try std.fs.path.join(allocator, &[_][]const u8{ base_dir, "bioinformatics" });
        defer allocator.free(dir);
        const slack = try generateBioinformaticsEx(allocator, 128, 8, 12, 0);
        defer allocator.free(slack);
        try saveHelper(allocator, dir, "s128_sparse.bigd", slack);

        const standard = try generateBioinformaticsEx(allocator, 128, 8, 12, 1);
        defer allocator.free(standard);
        try saveHelper(allocator, dir, "s128_standard.bigd", standard);

        const stress = try generateBioinformaticsEx(allocator, 128, 8, 12, 2);
        defer allocator.free(stress);
        try saveHelper(allocator, dir, "s128_dense.bigd", stress);
    }

    // 5. Airfoil (Subsonic Aerodynamics)
    {
        const dir = try std.fs.path.join(allocator, &[_][]const u8{ base_dir, "airfoil" });
        defer allocator.free(dir);
        const slack = try generateAirfoil(allocator, 16, 3.5, 0.008);
        defer allocator.free(slack);
        try saveHelper(allocator, dir, "p16_slack.bigd", slack);

        const standard = try generateAirfoil(allocator, 16, 2.5, 0.010);
        defer allocator.free(standard);
        try saveHelper(allocator, dir, "p16_standard.bigd", standard);

        const stress = try generateAirfoil(allocator, 16, 1.2, 0.025);
        defer allocator.free(stress);
        try saveHelper(allocator, dir, "p16_stress.bigd", stress);
    }

    // 6. Telecom (MIMO Channel & Power Allocation)
    {
        const dir = try std.fs.path.join(allocator, &[_][]const u8{ base_dir, "telecom" });
        defer allocator.free(dir);
        const slack = try generateTelecom(allocator, 16, 4, 0.01);
        defer allocator.free(slack);
        try saveHelper(allocator, dir, "u16_clean.bigd", slack);

        const standard = try generateTelecom(allocator, 16, 4, 0.05);
        defer allocator.free(standard);
        try saveHelper(allocator, dir, "u16_standard.bigd", standard);

        const stress = try generateTelecom(allocator, 16, 4, 0.25);
        defer allocator.free(stress);
        try saveHelper(allocator, dir, "u16_noisy.bigd", stress);
    }
}
