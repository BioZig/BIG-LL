const std = @import("std");
const big_ll = @import("big_ll");

const CountingAllocator = struct {
    parent: std.mem.Allocator,
    total_allocations: usize = 0,
    total_bytes: usize = 0,
    active_bytes: usize = 0,

    pub fn init(parent: std.mem.Allocator) CountingAllocator {
        return .{ .parent = parent };
    }

    pub fn allocator(self: *CountingAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = allocFn,
                .resize = resizeFn,
                .remap = remapFn,
                .free = freeFn,
            },
        };
    }

    fn allocFn(ctx: *anyopaque, len: usize, ptr_align: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *@This() = @ptrCast(@alignCast(ctx));
        const res = self.parent.rawAlloc(len, ptr_align, ret_addr);
        if (res != null) {
            self.total_allocations += 1;
            self.total_bytes += len;
            self.active_bytes += len;
        }
        return res;
    }

    fn resizeFn(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *@This() = @ptrCast(@alignCast(ctx));
        if (self.parent.rawResize(buf, buf_align, new_len, ret_addr)) {
            if (new_len > buf.len) {
                self.total_bytes += (new_len - buf.len);
                self.active_bytes += (new_len - buf.len);
            } else {
                self.active_bytes -= (buf.len - new_len);
            }
            return true;
        }
        return false;
    }

    fn remapFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *@This() = @ptrCast(@alignCast(ctx));
        const res = self.parent.rawRemap(memory, alignment, new_len, ret_addr);
        if (res != null) {
            if (new_len > memory.len) {
                self.total_bytes += (new_len - memory.len);
                self.active_bytes += (new_len - memory.len);
            } else {
                self.active_bytes -= (memory.len - new_len);
            }
        }
        return res;
    }

    fn freeFn(ctx: *anyopaque, buf: []u8, buf_align: std.mem.Alignment, ret_addr: usize) void {
        const self: *@This() = @ptrCast(@alignCast(ctx));
        self.active_bytes -= buf.len;
        self.parent.rawFree(buf, buf_align, ret_addr);
    }
};

const MockPopCountAdapter = struct {
    pub fn decode(self: *const MockPopCountAdapter, chrom: *const big_ll.Chromosome(256)) usize {
        _ = self;
        return chrom.popCount();
    }
    pub fn evaluate(self: *const MockPopCountAdapter, ones: usize) f64 {
        _ = self;
        return @as(f64, @floatFromInt(ones));
    }
};

test "Zero runtime heap allocation invariant in evolutionary loop" {
    var counting = CountingAllocator.init(std.testing.allocator);
    const tracking_alloc = counting.allocator();

    const L = 256;
    const pop_size = 128;
    const adapter = MockPopCountAdapter{};

    // Initialization phase: allocates slabs, deduplication table, linkage buffers
    var engine = try big_ll.GeneticEngine(L, MockPopCountAdapter).init(tracking_alloc, pop_size, adapter, 42);
    defer engine.deinit();

    const init_allocs = counting.total_allocations;
    const init_bytes = counting.total_bytes;
    try std.testing.expect(init_allocs > 0);
    try std.testing.expect(init_bytes > 0);

    // Evolutionary loop: execute 50 generations with linkage learning and deduplication active
    for (0..50) |_| {
        try engine.step();
    }

    // Invariant verification: Exactly ZERO allocations must occur during generational loops
    try std.testing.expectEqual(init_allocs, counting.total_allocations);
    try std.testing.expectEqual(init_bytes, counting.total_bytes);
    try std.testing.expectEqual(@as(usize, 50), engine.generation);
}
