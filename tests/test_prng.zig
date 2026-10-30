const std = @import("std");
const big_ll = @import("big_ll");
const prng = big_ll.prng;

test "Xoshiro256 deterministic sequence" {
    var rng1 = prng.Xoshiro256.init(12345);
    var rng2 = prng.Xoshiro256.init(12345);

    for (0..100) |_| {
        try std.testing.expectEqual(rng1.next(), rng2.next());
    }
}

test "Xoshiro256 float bounds" {
    var rng = prng.Xoshiro256.init(9999);
    for (0..1000) |_| {
        const f = rng.nextFloat();
        try std.testing.expect(f >= 0.0 and f < 1.0);
    }
}

test "Xoshiro256 bounded int" {
    var rng = prng.Xoshiro256.init(42);
    for (0..500) |_| {
        const v = rng.uintLessThan(17);
        try std.testing.expect(v < 17);
    }
}

test "Xoshiro256 sparse mask density" {
    var rng = prng.Xoshiro256.init(777);
    var total_set: usize = 0;
    const trials = 1000;

    for (0..trials) |_| {
        const mask = rng.sparseMask(4);
        total_set += @popCount(mask);
    }

    const mean_set = @as(f64, @floatFromInt(total_set)) / @as(f64, @floatFromInt(trials));
    try std.testing.expect(mean_set > 2.0 and mean_set < 6.0);
}
