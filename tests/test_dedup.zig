const std = @import("std");
const big_ll = @import("big_ll");
const L1Dedup = big_ll.L1Dedup;

test "L1Dedup initialization and empty lookup" {
    var table = L1Dedup(256).init();
    try std.testing.expectEqual(@as(usize, 0), table.hits);
    try std.testing.expectEqual(@as(usize, 0), table.misses);
    try std.testing.expectEqual(@as(f64, 0.0), table.hitRatio());

    const res = table.lookup(0x12345678abcdef01);
    try std.testing.expect(res == null);
    try std.testing.expectEqual(@as(usize, 0), table.hits);
    try std.testing.expectEqual(@as(usize, 1), table.misses);
}

test "L1Dedup insertion, lookup, and hit ratio" {
    var table = L1Dedup(1024).init();
    const words = [_]u64{ 0x1111222233334444, 0x5555666677778888 };
    const h = L1Dedup(1024).hashChromosome(&words);

    try std.testing.expect(table.lookup(h) == null);

    table.insert(h, 42.5);

    const hit_val = table.lookup(h);
    try std.testing.expect(hit_val != null);
    try std.testing.expectEqual(@as(f64, 42.5), hit_val.?);
    try std.testing.expectEqual(@as(usize, 1), table.hits);
    try std.testing.expectEqual(@as(usize, 1), table.misses);
    try std.testing.expectEqual(@as(f64, 0.5), table.hitRatio());
}

test "L1Dedup hash avalanche and distinctness" {
    const w1 = [_]u64{ 0x0000000000000001 };
    const w2 = [_]u64{ 0x0000000000000002 };
    const h1 = L1Dedup(256).hashChromosome(&w1);
    const h2 = L1Dedup(256).hashChromosome(&w2);

    try std.testing.expect(h1 != h2);
    const diff_bits = @popCount(h1 ^ h2);
    try std.testing.expect(diff_bits >= 16);
}

test "L1Dedup slot collision overwrite and clear" {
    var table = L1Dedup(64).init();
    const h1: u64 = 5;
    const h2: u64 = 69;

    table.insert(h1, 100.0);
    try std.testing.expectEqual(@as(f64, 100.0), table.lookup(h1).?);

    table.insert(h2, 200.0);
    try std.testing.expect(table.lookup(h1) == null);
    try std.testing.expectEqual(@as(f64, 200.0), table.lookup(h2).?);

    table.clear();
    try std.testing.expect(table.lookup(h2) == null);
    try std.testing.expectEqual(@as(usize, 0), table.hits);
    try std.testing.expectEqual(@as(usize, 1), table.misses);
}

test "AdaptiveDedup entropy-gated bypass" {
    var table = big_ll.AdaptiveDedup(64, 2).init();
    const h: u64 = 0xabcdef1234567890;

    // High entropy: bypasses lookup and insert
    table.insert(h, 99.0, 0.55);
    try std.testing.expect(table.lookup(h, 0.55) == null);
    try std.testing.expectEqual(@as(usize, 1), table.bypasses);

    // Low entropy: activates caching
    table.insert(h, 99.0, 0.20);
    const hit = table.lookup(h, 0.20);
    try std.testing.expect(hit != null);
    try std.testing.expectEqual(@as(f64, 99.0), hit.?);
    try std.testing.expectEqual(@as(usize, 1), table.hits);
}

test "AdaptiveDedup 2-way associativity prevents collision eviction" {
    var table = big_ll.AdaptiveDedup(64, 2).init();
    const h1: u64 = 7;
    const h2: u64 = 71; // 71 % 64 == 7

    table.insert(h1, 10.0, 0.10);
    table.insert(h2, 20.0, 0.10);

    // Both coexist without evicting each other
    try std.testing.expectEqual(@as(f64, 10.0), table.lookup(h1, 0.10).?);
    try std.testing.expectEqual(@as(f64, 20.0), table.lookup(h2, 0.10).?);
}
