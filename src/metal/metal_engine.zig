// Apple Silicon Dedicated Metal GPU Batch Acceleration Zig Wrapper
const std = @import("std");
const builtin = @import("builtin");
const types = @import("../core/types.zig");
const Chromosome = types.Chromosome;

const is_darwin = builtin.os.tag.isDarwin();

const c = if (is_darwin) @cImport({
    @cInclude("metal_bridge.h");
}) else struct {};

pub const MetalEngine = struct {
    const Self = @This();

    handle: ?*anyopaque = null,

    pub fn isSupported() bool {
        if (!is_darwin) return false;
        return c.bigll_metal_is_supported();
    }

    pub fn init() ?Self {
        if (!isSupported()) return null;
        if (!is_darwin) return null;
        const ctx = c.bigll_metal_init() orelse return null;
        return Self{ .handle = @ptrCast(ctx) };
    }

    pub fn deinit(self: *Self) void {
        if (!is_darwin) return;
        if (self.handle) |ctx| {
            c.bigll_metal_deinit(@ptrCast(@alignCast(ctx)));
            self.handle = null;
        }
    }

    pub fn allocShared(self: *Self, bytes: usize) ?[*]u8 {
        if (!is_darwin) return null;
        if (self.handle) |ctx| {
            const ptr = c.bigll_metal_alloc_shared(@ptrCast(@alignCast(ctx)), bytes) orelse return null;
            return @ptrCast(ptr);
        }
        return null;
    }

    pub fn freeShared(self: *Self, ptr: [*]u8, bytes: usize) void {
        if (!is_darwin) return;
        if (self.handle) |ctx| {
            c.bigll_metal_free_shared(@ptrCast(@alignCast(ctx)), @ptrCast(ptr), bytes);
        }
    }

    pub fn getSharedChromBuffer(self: *Self, comptime T: type, count: usize) ?[]T {
        if (!is_darwin) return null;
        const raw_ctx = self.handle orelse return null;
        const total_bytes = count * @sizeOf(T);
        const ptr = c.bigll_metal_get_shared_chrom_buffer(@ptrCast(@alignCast(raw_ctx)), total_bytes) orelse return null;
        const typed_ptr: [*]T = @ptrCast(@alignCast(ptr));
        return typed_ptr[0..count];
    }

    pub fn getSharedFitnessBuffer(self: *Self, count: usize) ?[]f32 {
        if (!is_darwin) return null;
        const raw_ctx = self.handle orelse return null;
        const total_bytes = count * @sizeOf(f32);
        const ptr = c.bigll_metal_get_shared_fit_buffer(@ptrCast(@alignCast(raw_ctx)), total_bytes) orelse return null;
        const typed_ptr: [*]f32 = @ptrCast(@alignCast(ptr));
        return typed_ptr[0..count];
    }

    pub fn evalOneMax(
        self: *Self,
        comptime L: usize,
        chroms: []const Chromosome(L),
        fitness_out: []f32,
    ) bool {
        if (!is_darwin) return false;
        const raw_ctx = self.handle orelse return false;
        if (chroms.len == 0 or fitness_out.len < chroms.len) return false;

        const num_words: u32 = @intCast(Chromosome(L).word_count);
        const stride_words: u32 = @intCast(@sizeOf(Chromosome(L)) / @sizeOf(u64));
        const pop_size: u32 = @intCast(chroms.len);
        const chrom_ptr: [*]const u64 = @ptrCast(chroms.ptr);

        return c.bigll_metal_eval_onemax(
            @ptrCast(@alignCast(raw_ctx)),
            chrom_ptr,
            fitness_out.ptr,
            num_words,
            stride_words,
            pop_size,
        );
    }

    pub fn evalFinance(
        self: *Self,
        chroms_u64: []const u64,
        fitness_out: []f32,
        expected_returns: []const f32,
        cov_matrix: []const f32,
        num_assets: u32,
        risk_aversion: f32,
    ) bool {
        if (!is_darwin) return false;
        const raw_ctx = self.handle orelse return false;
        if (chroms_u64.len == 0 or fitness_out.len < chroms_u64.len) return false;
        if (expected_returns.len < num_assets or cov_matrix.len < num_assets * num_assets) return false;

        const pop_size: u32 = @intCast(chroms_u64.len);

        return c.bigll_metal_eval_finance(
            @ptrCast(@alignCast(raw_ctx)),
            chroms_u64.ptr,
            fitness_out.ptr,
            expected_returns.ptr,
            cov_matrix.ptr,
            num_assets,
            risk_aversion,
            pop_size,
        );
    }
};
