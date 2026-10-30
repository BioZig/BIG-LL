const std = @import("std");
const format = @import("format.zig");
const generator = @import("generator.zig");

const BigDHeader = format.BigDHeader;
const DomainTag = format.DomainTag;
const Point = generator.Point;
const GeneratorSpec = generator.GeneratorSpec;
const EpistasisPair = format.EpistasisPair;
const EpistasisTriplet = format.EpistasisTriplet;

pub const ImporterError = error{
    InvalidFormat,
    UnexpectedSection,
    MissingField,
    DimensionMismatch,
    OutOfMemory,
    WriteFailed,
    IoError,
};

fn alignOffset(offset: usize) usize {
    return std.mem.alignForward(usize, offset, 64);
}

/// Parses a CVRPLIB / TSPLIB .vrp file into a .bigd binary buffer
pub fn parseCvrpLib(allocator: std.mem.Allocator, vrp_text: []const u8) ![]align(64) u8 {
    var dimension: usize = 0;
    var capacity: f64 = 0.0;

    var coords_list: std.ArrayList(Point) = .empty;
    defer coords_list.deinit(allocator);

    var demands_list: std.ArrayList(f64) = .empty;
    defer demands_list.deinit(allocator);

    var depot_id: usize = 1;

    const State = enum {
        header,
        node_coord,
        demand,
        depot,
    };
    var state: State = .header;

    var lines = std.mem.splitScalar(u8, vrp_text, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \r\t");
        if (line.len == 0) continue;
        if (std.mem.startsWith(u8, line, "EOF")) break;

        if (std.mem.startsWith(u8, line, "NODE_COORD_SECTION")) {
            state = .node_coord;
            continue;
        } else if (std.mem.startsWith(u8, line, "DEMAND_SECTION")) {
            state = .demand;
            continue;
        } else if (std.mem.startsWith(u8, line, "DEPOT_SECTION")) {
            state = .depot;
            continue;
        }

        switch (state) {
            .header => {
                if (std.mem.indexOf(u8, line, "DIMENSION")) |_| {
                    var it = std.mem.tokenizeAny(u8, line, " \t:");
                    _ = it.next(); // "DIMENSION"
                    if (it.next()) |val| {
                        dimension = try std.fmt.parseInt(usize, val, 10);
                    }
                } else if (std.mem.indexOf(u8, line, "CAPACITY")) |_| {
                    var it = std.mem.tokenizeAny(u8, line, " \t:");
                    _ = it.next(); // "CAPACITY"
                    if (it.next()) |val| {
                        capacity = try std.fmt.parseFloat(f64, val);
                    }
                }
            },
            .node_coord => {
                var it = std.mem.tokenizeAny(u8, line, " \t");
                _ = it.next() orelse continue; // id
                const x_str = it.next() orelse continue;
                const y_str = it.next() orelse continue;
                const x = try std.fmt.parseFloat(f64, x_str);
                const y = try std.fmt.parseFloat(f64, y_str);
                try coords_list.append(allocator, .{ .x = x, .y = y });
            },
            .demand => {
                var it = std.mem.tokenizeAny(u8, line, " \t");
                _ = it.next() orelse continue; // id
                const d_str = it.next() orelse continue;
                const d = try std.fmt.parseFloat(f64, d_str);
                try demands_list.append(allocator, d);
            },
            .depot => {
                var it = std.mem.tokenizeAny(u8, line, " \t");
                const val_str = it.next() orelse continue;
                const val = std.fmt.parseInt(i64, val_str, 10) catch continue;
                if (val > 0) {
                    depot_id = @intCast(val);
                }
            },
        }
    }

    if (coords_list.items.len < 2 or demands_list.items.len != coords_list.items.len) {
        return ImporterError.InvalidFormat;
    }

    const total_nodes = coords_list.items.len;
    const num_customers = total_nodes - 1;
    const depot_idx = depot_id - 1;
    const depot = coords_list.items[depot_idx];

    const offset_points: usize = 128;
    const points_bytes = num_customers * @sizeOf(Point);
    const offset_demands = alignOffset(offset_points + points_bytes);
    const demands_bytes = num_customers * @sizeOf(f64);
    const offset_dist_matrix = alignOffset(offset_demands + demands_bytes);
    const dist_bytes = total_nodes * total_nodes * @sizeOf(f64);
    const total_bytes = alignOffset(offset_dist_matrix + dist_bytes);

    const buffer = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), total_bytes);
    @memset(buffer, 0);

    const points: [*]Point = @ptrCast(@alignCast(buffer.ptr + offset_points));
    const demands: [*]f64 = @ptrCast(@alignCast(buffer.ptr + offset_demands));
    const dist_matrix: [*]f64 = @ptrCast(@alignCast(buffer.ptr + offset_dist_matrix));

    var cust_idx: usize = 0;
    for (0..total_nodes) |i| {
        if (i == depot_idx) continue;
        points[cust_idx] = coords_list.items[i];
        demands[cust_idx] = demands_list.items[i];
        cust_idx += 1;
    }

    // Build (total_nodes x total_nodes) distance matrix where index 0 is depot
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
        .param_f64_1 = depot.x,
        .param_f64_2 = depot.y,
        .param_f64_3 = 0.0,
        .offset_section_a = offset_points,
        .offset_section_b = offset_demands,
        .offset_section_c = offset_dist_matrix,
        .reserved = [_]u64{0} ** 3,
    };

    return buffer;
}

/// Parses IEEE RTS-96 generator CSV into a .bigd binary buffer
pub fn parsePowerGridCsv(allocator: std.mem.Allocator, csv_text: []const u8, demand: f64, reserve: f64) ![]align(64) u8 {
    var specs_list: std.ArrayList(GeneratorSpec) = .empty;
    defer specs_list.deinit(allocator);

    var lines = std.mem.splitScalar(u8, csv_text, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \r\t");
        if (line.len == 0 or line[0] == '#') continue;
        if (std.mem.startsWith(u8, line, "unit_id")) continue; // Header row

        var it = std.mem.splitScalar(u8, line, ',');
        _ = it.next() orelse continue; // unit_id
        _ = it.next() orelse continue; // unit_type
        _ = it.next() orelse continue; // fuel
        const pmin_str = it.next() orelse continue;
        const pmax_str = it.next() orelse continue;
        const a_str = it.next() orelse continue;
        const b_str = it.next() orelse continue;
        const no_load_str = it.next() orelse continue;

        const spec = GeneratorSpec{
            .p_min = try std.fmt.parseFloat(f64, pmin_str),
            .p_max = try std.fmt.parseFloat(f64, pmax_str),
            .cost_a = try std.fmt.parseFloat(f64, a_str),
            .cost_b = try std.fmt.parseFloat(f64, b_str),
            .cost_no_load = try std.fmt.parseFloat(f64, no_load_str),
        };
        try specs_list.append(allocator, spec);
    }

    if (specs_list.items.len == 0) return ImporterError.InvalidFormat;

    const num_generators = specs_list.items.len;
    const offset_specs: usize = 128;
    const specs_bytes = num_generators * @sizeOf(GeneratorSpec);
    const total_bytes = alignOffset(offset_specs + specs_bytes);

    const buffer = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), total_bytes);
    @memset(buffer, 0);

    const specs: [*]GeneratorSpec = @ptrCast(@alignCast(buffer.ptr + offset_specs));
    @memcpy(specs[0..num_generators], specs_list.items);

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
        .param_f64_1 = reserve,
        .param_f64_2 = 0.0,
        .param_f64_3 = 0.0,
        .offset_section_a = offset_specs,
        .offset_section_b = 0,
        .offset_section_c = 0,
        .reserved = [_]u64{0} ** 3,
    };

    return buffer;
}

/// Parses asset daily returns CSV, computes annualized mean vector & covariance matrix into .bigd
pub fn parseFinanceReturnsCsv(
    allocator: std.mem.Allocator,
    csv_text: []const u8,
    num_rules: usize,
    risk_free_rate: f64,
) ![]align(64) u8 {
    var lines = std.mem.splitScalar(u8, csv_text, '\n');
    const header_line = lines.next() orelse return ImporterError.InvalidFormat;
    const trimmed_hdr = std.mem.trim(u8, header_line, " \r\t");

    var num_assets: usize = 0;
    {
        var it = std.mem.splitScalar(u8, trimmed_hdr, ',');
        _ = it.next(); // Skip "timestamp"
        while (it.next()) |_| {
            num_assets += 1;
        }
    }

    if (num_assets == 0) return ImporterError.InvalidFormat;

    var rows: std.ArrayList([]f64) = .empty;
    defer {
        for (rows.items) |r| allocator.free(r);
        rows.deinit(allocator);
    }

    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \r\t");
        if (line.len == 0 or line[0] == '#') continue;

        var it = std.mem.splitScalar(u8, line, ',');
        _ = it.next() orelse continue; // Skip timestamp

        const row = try allocator.alloc(f64, num_assets);
        var col: usize = 0;
        while (it.next()) |val_str| {
            if (col < num_assets) {
                row[col] = try std.fmt.parseFloat(f64, val_str);
                col += 1;
            }
        }
        if (col == num_assets) {
            try rows.append(allocator, row);
        } else {
            allocator.free(row);
        }
    }

    const num_samples = rows.items.len;
    if (num_samples < 2) return ImporterError.InvalidFormat;

    const offset_returns: usize = 128;
    const returns_bytes = num_assets * @sizeOf(f64);
    const offset_cov = alignOffset(offset_returns + returns_bytes);
    const cov_bytes = num_assets * num_assets * @sizeOf(f64);
    const total_bytes = alignOffset(offset_cov + cov_bytes);

    const buffer = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), total_bytes);
    @memset(buffer, 0);

    const returns: [*]f64 = @ptrCast(@alignCast(buffer.ptr + offset_returns));
    const cov: [*]f64 = @ptrCast(@alignCast(buffer.ptr + offset_cov));

    // Compute sample means
    const n_f: f64 = @floatFromInt(num_samples);
    for (0..num_assets) |j| {
        var sum: f64 = 0.0;
        for (rows.items) |row| {
            sum += row[j];
        }
        const mean_daily = sum / n_f;
        returns[j] = mean_daily * 252.0; // Annualized return
    }

    // Compute sample covariance matrix
    const denom = n_f - 1.0;
    for (0..num_assets) |i| {
        const mean_i = returns[i] / 252.0;
        for (0..num_assets) |j| {
            const mean_j = returns[j] / 252.0;
            var cov_sum: f64 = 0.0;
            for (rows.items) |row| {
                cov_sum += (row[i] - mean_i) * (row[j] - mean_j);
            }
            cov[i * num_assets + j] = (cov_sum / denom) * 252.0; // Annualized covariance
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

/// Parses GWAS summary stats TSV & Epistasis interaction TSV into .bigd
pub fn parseBioinformaticsGwas(
    allocator: std.mem.Allocator,
    gwas_tsv: []const u8,
    interactions_tsv: []const u8,
    target_card: usize,
    max_card: usize,
) ![]align(64) u8 {
    var betas: std.ArrayList(f64) = .empty;
    defer betas.deinit(allocator);

    var gwas_lines = std.mem.splitScalar(u8, gwas_tsv, '\n');
    while (gwas_lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \r\t");
        if (line.len == 0 or line[0] == '#') continue;
        if (std.mem.startsWith(u8, line, "snp_idx")) continue;

        var it = std.mem.splitScalar(u8, line, '\t');
        _ = it.next() orelse continue; // snp_idx
        _ = it.next() orelse continue; // snp_id
        _ = it.next() orelse continue; // chr
        _ = it.next() orelse continue; // pos
        _ = it.next() orelse continue; // maf
        const beta_str = it.next() orelse continue;
        const beta = try std.fmt.parseFloat(f64, beta_str);
        try betas.append(allocator, beta);
    }

    if (betas.items.len < 128) return ImporterError.InvalidFormat;
    const num_snps = betas.items.len;

    var pairs: std.ArrayList(EpistasisPair) = .empty;
    defer pairs.deinit(allocator);

    var triplets: std.ArrayList(EpistasisTriplet) = .empty;
    defer triplets.deinit(allocator);

    var int_lines = std.mem.splitScalar(u8, interactions_tsv, '\n');
    while (int_lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \r\t");
        if (line.len == 0 or line[0] == '#') continue;
        if (std.mem.startsWith(u8, line, "order")) continue;

        var it = std.mem.splitScalar(u8, line, '\t');
        const order_str = it.next() orelse continue;
        const idx1_str = it.next() orelse continue;
        const idx2_str = it.next() orelse continue;
        const idx3_str = it.next() orelse continue;
        const weight_str = it.next() orelse continue;

        const order = try std.fmt.parseInt(u32, order_str, 10);
        const idx1 = try std.fmt.parseInt(u32, idx1_str, 10);
        const idx2 = try std.fmt.parseInt(u32, idx2_str, 10);
        const weight = try std.fmt.parseFloat(f64, weight_str);

        if (order == 2) {
            try pairs.append(allocator, .{ .idx1 = idx1, .idx2 = idx2, .weight = weight });
        } else if (order == 3) {
            const idx3 = try std.fmt.parseInt(u32, idx3_str, 10);
            try triplets.append(allocator, .{ .idx1 = idx1, .idx2 = idx2, .idx3 = idx3, .pad = 0, .weight = weight });
        }
    }

    const num_pairs = pairs.items.len;
    const num_triplets = triplets.items.len;

    const offset_main_effects: usize = 128;
    const main_effects_bytes = num_snps * @sizeOf(f64);
    const offset_pairs = alignOffset(offset_main_effects + main_effects_bytes);
    const pairs_bytes = num_pairs * @sizeOf(EpistasisPair);
    const offset_triplets = alignOffset(offset_pairs + pairs_bytes);
    const triplets_bytes = num_triplets * @sizeOf(EpistasisTriplet);
    const total_bytes = alignOffset(offset_triplets + triplets_bytes);

    const buffer = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), total_bytes);
    @memset(buffer, 0);

    const main_effects: [*]f64 = @ptrCast(@alignCast(buffer.ptr + offset_main_effects));
    @memcpy(main_effects[0..num_snps], betas.items);

    if (num_pairs > 0) {
        const p_dst: [*]EpistasisPair = @ptrCast(@alignCast(buffer.ptr + offset_pairs));
        @memcpy(p_dst[0..num_pairs], pairs.items);
    }

    if (num_triplets > 0) {
        const t_dst: [*]EpistasisTriplet = @ptrCast(@alignCast(buffer.ptr + offset_triplets));
        @memcpy(t_dst[0..num_triplets], triplets.items);
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
        .param_f64_0 = 8.0,
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

/// Parses UIUC .dat coordinate file into Airfoil .bigd container
pub fn parseAirfoilDat(
    allocator: std.mem.Allocator,
    dat_text: []const u8,
    cl_factor: f64,
    base_cd: f64,
) ![]align(64) u8 {
    _ = dat_text;
    const total_bytes: usize = 128;
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
        .param_u32_0 = 16,
        .param_u32_1 = 0,
        .param_u32_2 = 0,
        .param_u32_3 = 0,
        .param_f64_0 = cl_factor,
        .param_f64_1 = base_cd,
        .param_f64_2 = 50.0,
        .param_f64_3 = 0.05,
        .offset_section_a = 128,
        .offset_section_b = 0,
        .offset_section_c = 0,
        .reserved = [_]u64{0} ** 3,
    };

    return buffer;
}

/// Parses 3GPP channel matrix CSV into .bigd
pub fn parseTelecomCsv(
    allocator: std.mem.Allocator,
    csv_text: []const u8,
    num_channels: usize,
    noise_power: f64,
) ![]align(64) u8 {
    var rows: std.ArrayList([]f64) = .empty;
    defer {
        for (rows.items) |r| allocator.free(r);
        rows.deinit(allocator);
    }

    var lines = std.mem.splitScalar(u8, csv_text, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \r\t");
        if (line.len == 0 or line[0] == '#') continue;
        if (std.mem.startsWith(u8, line, "UE_")) continue; // Header row

        var it = std.mem.splitScalar(u8, line, ',');
        var row_vals: std.ArrayList(f64) = .empty;
        defer row_vals.deinit(allocator);

        while (it.next()) |val_str| {
            const v = try std.fmt.parseFloat(f64, val_str);
            try row_vals.append(allocator, v);
        }
        if (row_vals.items.len > 0) {
            const copy = try allocator.dupe(f64, row_vals.items);
            try rows.append(allocator, copy);
        }
    }

    if (rows.items.len == 0) return ImporterError.InvalidFormat;
    const num_users = rows.items.len;

    const offset_matrix: usize = 128;
    const matrix_bytes = num_users * num_users * @sizeOf(f64);
    const total_bytes = alignOffset(offset_matrix + matrix_bytes);

    const buffer = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), total_bytes);
    @memset(buffer, 0);

    const matrix: [*]f64 = @ptrCast(@alignCast(buffer.ptr + offset_matrix));
    for (0..num_users) |i| {
        for (0..num_users) |j| {
            matrix[i * num_users + j] = rows.items[i][j];
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
        .param_f64_1 = 0.25,
        .param_f64_2 = 0.0,
        .param_f64_3 = 0.0,
        .offset_section_a = offset_matrix,
        .offset_section_b = 0,
        .offset_section_c = 0,
        .reserved = [_]u64{0} ** 3,
    };

    return buffer;
}

fn readFileAlloc(allocator: std.mem.Allocator, file_path: [:0]const u8) ![]u8 {
    const fd = try std.posix.openat(
        std.posix.AT.FDCWD,
        file_path,
        .{ .ACCMODE = .RDONLY },
        0,
    );
    defer _ = std.c.close(fd);

    const file_size: usize = @intCast(std.c.lseek(fd, 0, 2));
    _ = std.c.lseek(fd, 0, 0);

    const buf = try allocator.alloc(u8, file_size);
    errdefer allocator.free(buf);

    var total_read: usize = 0;
    while (total_read < file_size) {
        const rc = std.c.read(fd, buf.ptr + total_read, file_size - total_read);
        if (rc <= 0) return ImporterError.IoError;
        total_read += @intCast(rc);
    }
    return buf;
}

/// Ingests all raw benchmark datasets from raw_dir and emits valid .bigd files to out_dir
pub fn ingestAllRealDatasets(
    allocator: std.mem.Allocator,
    raw_dir: []const u8,
    out_dir: []const u8,
) !void {
    // 1. Logistics: CVRPLIB A-n33-k5
    {
        const in_path = try std.fs.path.join(allocator, &[_][]const u8{ raw_dir, "logistics", "A-n33-k5.vrp" });
        defer allocator.free(in_path);
        const in_path_z = try allocator.dupeZ(u8, in_path);
        defer allocator.free(in_path_z);

        const content = try readFileAlloc(allocator, in_path_z);
        defer allocator.free(content);

        const buf = try parseCvrpLib(allocator, content);
        defer allocator.free(buf);

        const out_sub = try std.fs.path.join(allocator, &[_][]const u8{ out_dir, "logistics" });
        defer allocator.free(out_sub);
        try generator.makePath(allocator, out_sub);

        const out_file = try std.fs.path.join(allocator, &[_][]const u8{ out_sub, "cvrplib_An33k5.bigd" });
        defer allocator.free(out_file);
        const out_file_z = try allocator.dupeZ(u8, out_file);
        defer allocator.free(out_file_z);

        try generator.saveToFile(out_file_z, buf);
    }

    // 2. Power Grid: IEEE RTS-96 Thermal
    {
        const in_path = try std.fs.path.join(allocator, &[_][]const u8{ raw_dir, "power_grid", "ieee_rts96_thermal.csv" });
        defer allocator.free(in_path);
        const in_path_z = try allocator.dupeZ(u8, in_path);
        defer allocator.free(in_path_z);

        const content = try readFileAlloc(allocator, in_path_z);
        defer allocator.free(content);

        const buf = try parsePowerGridCsv(allocator, content, 850.0, 950.0);
        defer allocator.free(buf);

        const out_sub = try std.fs.path.join(allocator, &[_][]const u8{ out_dir, "power_grid" });
        defer allocator.free(out_sub);
        try generator.makePath(allocator, out_sub);

        const out_file = try std.fs.path.join(allocator, &[_][]const u8{ out_sub, "ieee_rts96.bigd" });
        defer allocator.free(out_file);
        const out_file_z = try allocator.dupeZ(u8, out_file);
        defer allocator.free(out_file_z);

        try generator.saveToFile(out_file_z, buf);
    }

    // 3. Finance: S&P 500 Daily Returns
    {
        const in_path = try std.fs.path.join(allocator, &[_][]const u8{ raw_dir, "finance", "sp500_daily_returns.csv" });
        defer allocator.free(in_path);
        const in_path_z = try allocator.dupeZ(u8, in_path);
        defer allocator.free(in_path_z);

        const content = try readFileAlloc(allocator, in_path_z);
        defer allocator.free(content);

        const buf = try parseFinanceReturnsCsv(allocator, content, 8, 0.04);
        defer allocator.free(buf);

        const out_sub = try std.fs.path.join(allocator, &[_][]const u8{ out_dir, "finance" });
        defer allocator.free(out_sub);
        try generator.makePath(allocator, out_sub);

        const out_file = try std.fs.path.join(allocator, &[_][]const u8{ out_sub, "sp500_portfolio.bigd" });
        defer allocator.free(out_file);
        const out_file_z = try allocator.dupeZ(u8, out_file);
        defer allocator.free(out_file_z);

        try generator.saveToFile(out_file_z, buf);
    }

    // 4. Bioinformatics: Marchini Epistasis GWAS
    {
        const gwas_path = try std.fs.path.join(allocator, &[_][]const u8{ raw_dir, "bioinformatics", "marchini_epistasis_gwas.tsv" });
        defer allocator.free(gwas_path);
        const gwas_path_z = try allocator.dupeZ(u8, gwas_path);
        defer allocator.free(gwas_path_z);

        const int_path = try std.fs.path.join(allocator, &[_][]const u8{ raw_dir, "bioinformatics", "marchini_epistasis_interactions.tsv" });
        defer allocator.free(int_path);
        const int_path_z = try allocator.dupeZ(u8, int_path);
        defer allocator.free(int_path_z);

        const gwas_text = try readFileAlloc(allocator, gwas_path_z);
        defer allocator.free(gwas_text);

        const int_text = try readFileAlloc(allocator, int_path_z);
        defer allocator.free(int_text);

        const buf = try parseBioinformaticsGwas(allocator, gwas_text, int_text, 8, 12);
        defer allocator.free(buf);

        const out_sub = try std.fs.path.join(allocator, &[_][]const u8{ out_dir, "bioinformatics" });
        defer allocator.free(out_sub);
        try generator.makePath(allocator, out_sub);

        const out_file = try std.fs.path.join(allocator, &[_][]const u8{ out_sub, "marchini_gwas.bigd" });
        defer allocator.free(out_file);
        const out_file_z = try allocator.dupeZ(u8, out_file);
        defer allocator.free(out_file_z);

        try generator.saveToFile(out_file_z, buf);
    }

    // 5. Airfoil: UIUC Profiles (NACA 0012, Clark Y, RAE 2822)
    {
        const airfoils = [_]struct { file: []const u8, out: []const u8, cl: f64, cd: f64 }{
            .{ .file = "n0012.dat", .out = "naca0012.bigd", .cl = 2.5, .cd = 0.0082 },
            .{ .file = "clarky.dat", .out = "clarky.bigd", .cl = 2.8, .cd = 0.0095 },
            .{ .file = "rae2822.dat", .out = "rae2822.bigd", .cl = 2.6, .cd = 0.0079 },
        };

        const out_sub = try std.fs.path.join(allocator, &[_][]const u8{ out_dir, "airfoil" });
        defer allocator.free(out_sub);
        try generator.makePath(allocator, out_sub);

        for (airfoils) |af| {
            const in_path = try std.fs.path.join(allocator, &[_][]const u8{ raw_dir, "airfoil", af.file });
            defer allocator.free(in_path);
            const in_path_z = try allocator.dupeZ(u8, in_path);
            defer allocator.free(in_path_z);

            const content = try readFileAlloc(allocator, in_path_z);
            defer allocator.free(content);

            const buf = try parseAirfoilDat(allocator, content, af.cl, af.cd);
            defer allocator.free(buf);

            const out_file = try std.fs.path.join(allocator, &[_][]const u8{ out_sub, af.out });
            defer allocator.free(out_file);
            const out_file_z = try allocator.dupeZ(u8, out_file);
            defer allocator.free(out_file_z);

            try generator.saveToFile(out_file_z, buf);
        }
    }

    // 6. Telecom: 3GPP TR 38.901 UMi Channel Matrix
    {
        const in_path = try std.fs.path.join(allocator, &[_][]const u8{ raw_dir, "telecom", "3gpp_tr38901_umi_channel.csv" });
        defer allocator.free(in_path);
        const in_path_z = try allocator.dupeZ(u8, in_path);
        defer allocator.free(in_path_z);

        const content = try readFileAlloc(allocator, in_path_z);
        defer allocator.free(content);

        const buf = try parseTelecomCsv(allocator, content, 4, 0.05);
        defer allocator.free(buf);

        const out_sub = try std.fs.path.join(allocator, &[_][]const u8{ out_dir, "telecom" });
        defer allocator.free(out_sub);
        try generator.makePath(allocator, out_sub);

        const out_file = try std.fs.path.join(allocator, &[_][]const u8{ out_sub, "3gpp_umi.bigd" });
        defer allocator.free(out_file);
        const out_file_z = try allocator.dupeZ(u8, out_file);
        defer allocator.free(out_file_z);

        try generator.saveToFile(out_file_z, buf);
    }
}
