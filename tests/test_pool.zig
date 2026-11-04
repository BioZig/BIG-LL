const std = @import("std");
const big_ll = @import("big_ll");
const StaticThreadPool = big_ll.StaticThreadPool;

const IncrementContext = struct {
    slots: [16]usize = [_]usize{0} ** 16,

    pub fn task(ctx_ptr: *anyopaque, worker_id: usize) void {
        const self: *@This() = @ptrCast(@alignCast(ctx_ptr));
        self.slots[worker_id] += 1;
    }
};

test "StaticThreadPool single-worker execution" {
    var pool: StaticThreadPool = .{};
    try pool.init(1);
    defer pool.deinit();

    var ctx = IncrementContext{};
    pool.dispatch(&ctx, IncrementContext.task);

    try std.testing.expectEqual(@as(usize, 1), ctx.slots[0]);
    try std.testing.expectEqual(@as(usize, 0), ctx.slots[1]);
}

test "StaticThreadPool multi-worker parallel dispatches" {
    var pool: StaticThreadPool = .{};
    try pool.init(4);
    defer pool.deinit();

    var ctx = IncrementContext{};
    for (0..100) |_| {
        pool.dispatch(&ctx, IncrementContext.task);
    }

    for (0..4) |i| {
        try std.testing.expectEqual(@as(usize, 100), ctx.slots[i]);
    }
}

test "CpuTopology dynamic detection" {
    const topo = big_ll.pool.CpuTopology.get();
    try std.testing.expect(topo.total_physical >= 1);
    try std.testing.expect(topo.p_cores >= 1);
    if (topo.is_heterogeneous) {
        try std.testing.expect(topo.e_cores >= 1);
        try std.testing.expectEqual(topo.total_physical, topo.p_cores + topo.e_cores);
    }
}
