const std = @import("std");
const big_ll = @import("big_ll");
const importer = big_ll.data_importer;
const loader = big_ll.data_loader;
const format = big_ll.data_format;

test "Importer: parse CVRPLIB instance and verify distance matrix" {
    const allocator = std.testing.allocator;
    const vrp_sample =
        \\NAME : sample-n4
        \\TYPE : CVRP
        \\DIMENSION : 4
        \\CAPACITY : 50
        \\NODE_COORD_SECTION
        \\1 0.0 0.0
        \\2 3.0 4.0
        \\3 0.0 10.0
        \\4 6.0 8.0
        \\DEMAND_SECTION
        \\1 0
        \\2 10
        \\3 20
        \\4 15
        \\DEPOT_SECTION
        \\1
        \\-1
        \\EOF
    ;

    const buf = try importer.parseCvrpLib(allocator, vrp_sample);
    defer allocator.free(buf);

    var inst = try loader.StaticBufferInstance.init(buf);
    const hdr = inst.header();

    try std.testing.expectEqual(@as(u32, format.Magic), hdr.magic);
    try std.testing.expectEqual(@as(u32, 1), hdr.domain_tag); // logistics
    try std.testing.expectEqual(@as(u32, 3), hdr.param_u32_0); // 3 customers
    try std.testing.expectEqual(@as(u32, 4), hdr.param_u32_1); // 4 total nodes
    try std.testing.expectEqual(@as(f64, 50.0), hdr.param_f64_0); // capacity
    try std.testing.expectEqual(@as(f64, 0.0), hdr.param_f64_1); // depot x
    try std.testing.expectEqual(@as(f64, 0.0), hdr.param_f64_2); // depot y

    const dist_mat = try inst.getSlice(f64, hdr.offset_section_c, 4 * 4);
    // Node 0 is depot (0,0), Node 1 is customer 1 at (3,4) -> distance = sqrt(3^2 + 4^2) = 5.0
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), dist_mat[0 * 4 + 1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), dist_mat[1 * 4 + 0], 1e-6);
    // Distance to self is 0
    try std.testing.expectEqual(@as(f64, 0.0), dist_mat[0 * 4 + 0]);
    try std.testing.expectEqual(@as(f64, 0.0), dist_mat[1 * 4 + 1]);
}

test "Importer: parse IEEE RTS-96 generator specs" {
    const allocator = std.testing.allocator;
    const csv_sample =
        \\# Test Power Grid Specs
        \\unit_id,unit_type,fuel,p_min,p_max,cost_a,cost_b,cost_no_load
        \\1,U12,Oil,2.4,12.0,0.025,25.5,86.4
        \\2,U20,Gas,5.0,20.0,0.018,37.5,119.8
    ;

    const buf = try importer.parsePowerGridCsv(allocator, csv_sample, 30.0, 32.0);
    defer allocator.free(buf);

    var inst = try loader.StaticBufferInstance.init(buf);
    const hdr = inst.header();

    try std.testing.expectEqual(@as(u32, format.Magic), hdr.magic);
    try std.testing.expectEqual(@as(u32, 2), hdr.domain_tag); // power_grid
    try std.testing.expectEqual(@as(u32, 2), hdr.param_u32_0); // 2 generators
    try std.testing.expectEqual(@as(f64, 30.0), hdr.param_f64_0);
    try std.testing.expectEqual(@as(f64, 32.0), hdr.param_f64_1);

    const specs = try inst.getSlice(big_ll.data_generator.GeneratorSpec, hdr.offset_section_a, 2);
    try std.testing.expectEqual(@as(f64, 2.4), specs[0].p_min);
    try std.testing.expectEqual(@as(f64, 12.0), specs[0].p_max);
    try std.testing.expectEqual(@as(f64, 0.025), specs[0].cost_a);
    try std.testing.expectEqual(@as(f64, 25.5), specs[0].cost_b);
    try std.testing.expectEqual(@as(f64, 86.4), specs[0].cost_no_load);
}

test "Importer: parse Asset Returns CSV and verify statistical moments" {
    const allocator = std.testing.allocator;
    const returns_csv =
        \\timestamp,AssetA,AssetB
        \\1,0.01,0.02
        \\2,-0.01,-0.02
        \\3,0.01,0.02
        \\4,-0.01,-0.02
    ;

    const buf = try importer.parseFinanceReturnsCsv(allocator, returns_csv, 4, 0.03);
    defer allocator.free(buf);

    var inst = try loader.StaticBufferInstance.init(buf);
    const hdr = inst.header();

    try std.testing.expectEqual(@as(u32, format.Magic), hdr.magic);
    try std.testing.expectEqual(@as(u32, 3), hdr.domain_tag); // finance
    try std.testing.expectEqual(@as(u32, 2), hdr.param_u32_0); // 2 assets
    try std.testing.expectEqual(@as(u32, 4), hdr.param_u32_1); // 4 rules

    const returns = try inst.getSlice(f64, hdr.offset_section_a, 2);
    // Mean return is 0.0
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), returns[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), returns[1], 1e-6);

    const cov = try inst.getSlice(f64, hdr.offset_section_b, 4);
    // Var(AssetB) should be strictly greater than Var(AssetA)
    try std.testing.expect(cov[1 * 2 + 1] > cov[0 * 2 + 0]);
}
