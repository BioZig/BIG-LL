const std = @import("std");
const types = @import("../core/types.zig");

pub const Magic: u32 = 0x4249474C; // "BIGL"
pub const CurrentVersion: u32 = 1;

pub const StatusCode = enum(u32) {
    idle = 0,
    eval_requested = 1,
    eval_completed = 2,
    terminated = 3,
};

pub const SharedPopulationHeader = extern struct {
    magic: u32 align(64) = Magic,
    version: u32 = CurrentVersion,
    status: u32 = @intFromEnum(StatusCode.idle),
    pop_size: u32,
    chromosome_bits: u32,
    word_count: u32,
    generation: u32 = 0,
    best_index: u32 = 0,
    best_fitness: f64 = -std.math.inf(f64),
    genes_offset: u64 = 128,
    fitness_offset: u64,
    flags: u32 = 0,
    reserved: [6]u64 = [_]u64{0} ** 6,
};

pub const MemoryLayout = struct {
    word_count: usize,
    genes_offset: usize,
    fitness_offset: usize,
    total_size: usize,
};

pub fn calculateLayout(pop_size: usize, chromosome_bits: usize) MemoryLayout {
    const word_count = (chromosome_bits + 63) / 64;
    const genes_offset: usize = 128;
    const genes_bytes = pop_size * word_count * @sizeOf(u64);
    const fitness_offset = std.mem.alignForward(usize, genes_offset + genes_bytes, 64);
    const fitness_bytes = pop_size * @sizeOf(f64);
    const total_bytes = fitness_offset + fitness_bytes;
    const aligned_total = std.mem.alignForward(usize, total_bytes, std.heap.page_size_min);

    return .{
        .word_count = word_count,
        .genes_offset = genes_offset,
        .fitness_offset = fitness_offset,
        .total_size = aligned_total,
    };
}

pub const SharedMemoryRegion = struct {
    backing_slice: []u8,
    is_mmap: bool,
    allocator: ?std.mem.Allocator,
    layout: MemoryLayout,

    pub fn initAnonymous(allocator: std.mem.Allocator, pop_size: usize, chromosome_bits: usize) !SharedMemoryRegion {
        const layout = calculateLayout(pop_size, chromosome_bits);
        const mem = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), layout.total_size);
        @memset(mem, 0);

        var region = SharedMemoryRegion{
            .backing_slice = mem,
            .is_mmap = false,
            .allocator = allocator,
            .layout = layout,
        };

        const hdr = region.header();
        hdr.* = .{
            .magic = Magic,
            .version = CurrentVersion,
            .status = @intFromEnum(StatusCode.idle),
            .pop_size = @intCast(pop_size),
            .chromosome_bits = @intCast(chromosome_bits),
            .word_count = @intCast(layout.word_count),
            .generation = 0,
            .best_index = 0,
            .best_fitness = -std.math.inf(f64),
            .genes_offset = layout.genes_offset,
            .fitness_offset = layout.fitness_offset,
            .flags = 0,
            .reserved = [_]u64{0} ** 6,
        };

        return region;
    }

    pub fn createFileBacked(file_path: [:0]const u8, pop_size: usize, chromosome_bits: usize) !SharedMemoryRegion {
        const layout = calculateLayout(pop_size, chromosome_bits);
        const fd = try std.posix.openat(
            std.posix.AT.FDCWD,
            file_path,
            .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true },
            0o666,
        );
        defer _ = std.c.close(fd);

        _ = std.c.ftruncate(fd, @intCast(layout.total_size));

        const mem = try std.posix.mmap(
            null,
            layout.total_size,
            .{ .READ = true, .WRITE = true },
            .{ .TYPE = .SHARED },
            fd,
            0,
        );

        var region = SharedMemoryRegion{
            .backing_slice = mem,
            .is_mmap = true,
            .allocator = null,
            .layout = layout,
        };

        const hdr = region.header();
        hdr.* = .{
            .magic = Magic,
            .version = CurrentVersion,
            .status = @intFromEnum(StatusCode.idle),
            .pop_size = @intCast(pop_size),
            .chromosome_bits = @intCast(chromosome_bits),
            .word_count = @intCast(layout.word_count),
            .generation = 0,
            .best_index = 0,
            .best_fitness = -std.math.inf(f64),
            .genes_offset = layout.genes_offset,
            .fitness_offset = layout.fitness_offset,
            .flags = 0,
            .reserved = [_]u64{0} ** 6,
        };

        return region;
    }

    pub fn openFileBacked(file_path: [:0]const u8) !SharedMemoryRegion {
        const fd = try std.posix.openat(
            std.posix.AT.FDCWD,
            file_path,
            .{ .ACCMODE = .RDWR },
            0,
        );
        defer _ = std.c.close(fd);

        const file_size: usize = @intCast(std.c.lseek(fd, 0, 2));
        _ = std.c.lseek(fd, 0, 0);

        if (file_size < @sizeOf(SharedPopulationHeader)) {
            return error.InvalidShmFileSize;
        }

        const mem = try std.posix.mmap(
            null,
            file_size,
            .{ .READ = true, .WRITE = true },
            .{ .TYPE = .SHARED },
            fd,
            0,
        );

        const hdr_ptr: *const SharedPopulationHeader = @ptrCast(@alignCast(mem.ptr));
        if (hdr_ptr.magic != Magic or hdr_ptr.version != CurrentVersion) {
            const aligned_slice: []align(std.heap.page_size_min) u8 = @alignCast(mem);
            std.posix.munmap(aligned_slice);
            return error.InvalidShmHeader;
        }

        const layout = calculateLayout(hdr_ptr.pop_size, hdr_ptr.chromosome_bits);

        return SharedMemoryRegion{
            .backing_slice = mem,
            .is_mmap = true,
            .allocator = null,
            .layout = layout,
        };
    }

    pub fn deinit(self: *SharedMemoryRegion) void {
        if (self.is_mmap) {
            const aligned_slice: []align(std.heap.page_size_min) u8 = @alignCast(self.backing_slice);
            std.posix.munmap(aligned_slice);
        } else if (self.allocator) |alloc| {
            const aligned_slice: []align(64) u8 = @alignCast(self.backing_slice);
            alloc.free(aligned_slice);
        }
    }

    pub fn header(self: *SharedMemoryRegion) *SharedPopulationHeader {
        return @ptrCast(@alignCast(self.backing_slice.ptr));
    }

    pub fn headerConst(self: *const SharedMemoryRegion) *const SharedPopulationHeader {
        return @ptrCast(@alignCast(self.backing_slice.ptr));
    }

    pub fn getGenesSlice(self: *SharedMemoryRegion) []u64 {
        const hdr = self.headerConst();
        const start = hdr.genes_offset;
        const count = hdr.pop_size * hdr.word_count;
        const ptr: [*]u64 = @ptrCast(@alignCast(self.backing_slice.ptr + start));
        return ptr[0..count];
    }

    pub fn getFitnessSlice(self: *SharedMemoryRegion) []f64 {
        const hdr = self.headerConst();
        const start = hdr.fitness_offset;
        const count = hdr.pop_size;
        const ptr: [*]f64 = @ptrCast(@alignCast(self.backing_slice.ptr + start));
        return ptr[0..count];
    }

    pub fn getChromosomeGenes(self: *SharedMemoryRegion, idx: usize) []u64 {
        const hdr = self.headerConst();
        std.debug.assert(idx < hdr.pop_size);
        const start = hdr.genes_offset + (idx * hdr.word_count * @sizeOf(u64));
        const ptr: [*]u64 = @ptrCast(@alignCast(self.backing_slice.ptr + start));
        return ptr[0..hdr.word_count];
    }

    pub fn getChromosomeGenesConst(self: *const SharedMemoryRegion, idx: usize) []const u64 {
        const hdr = self.headerConst();
        std.debug.assert(idx < hdr.pop_size);
        const start = hdr.genes_offset + (idx * hdr.word_count * @sizeOf(u64));
        const ptr: [*]const u64 = @ptrCast(@alignCast(self.backing_slice.ptr + start));
        return ptr[0..hdr.word_count];
    }

    pub fn setFitness(self: *SharedMemoryRegion, idx: usize, val: f64) void {
        const hdr = self.header();
        std.debug.assert(idx < hdr.pop_size);
        const start = hdr.fitness_offset + (idx * @sizeOf(f64));
        const ptr: *f64 = @ptrCast(@alignCast(self.backing_slice.ptr + start));
        ptr.* = val;
    }

    pub fn getFitness(self: *const SharedMemoryRegion, idx: usize) f64 {
        const hdr = self.headerConst();
        std.debug.assert(idx < hdr.pop_size);
        const start = hdr.fitness_offset + (idx * @sizeOf(f64));
        const ptr: *const f64 = @ptrCast(@alignCast(self.backing_slice.ptr + start));
        return ptr.*;
    }

    pub fn requestEvaluation(self: *SharedMemoryRegion, generation: usize) void {
        const hdr = self.header();
        hdr.generation = @intCast(generation);
        @atomicStore(u32, &hdr.status, @intFromEnum(StatusCode.eval_requested), .release);
    }

    pub fn markEvaluationCompleted(self: *SharedMemoryRegion) void {
        const hdr = self.header();
        @atomicStore(u32, &hdr.status, @intFromEnum(StatusCode.eval_completed), .release);
    }

    pub fn terminate(self: *SharedMemoryRegion) void {
        const hdr = self.header();
        @atomicStore(u32, &hdr.status, @intFromEnum(StatusCode.terminated), .release);
    }

    pub fn waitForEvaluation(self: *const SharedMemoryRegion, max_spins: usize) bool {
        const hdr = self.headerConst();
        var spins: usize = 0;
        while (spins < max_spins) : (spins += 1) {
            const st = @atomicLoad(u32, &hdr.status, .acquire);
            if (st == @intFromEnum(StatusCode.eval_completed)) {
                return true;
            }
            if (st == @intFromEnum(StatusCode.terminated)) {
                return false;
            }
            std.Thread.yield() catch {};
        }
        return false;
    }

    pub fn exportFromPopulation(self: *SharedMemoryRegion, comptime L: usize, pop: *const types.Population(L)) void {
        const hdr = self.header();
        const n = @min(hdr.pop_size, pop.individuals.len);
        const words_len = (L + 63) / 64;
        std.debug.assert(words_len == hdr.word_count);

        for (0..n) |i| {
            const dest = self.getChromosomeGenes(i);
            @memcpy(dest, pop.individuals[i].words[0..words_len]);
        }
    }

    pub fn importToPopulation(self: *SharedMemoryRegion, comptime L: usize, pop: *types.Population(L)) void {
        const hdr = self.header();
        const n = @min(hdr.pop_size, pop.individuals.len);
        var best_idx: usize = 0;
        var best_fit: f64 = -std.math.inf(f64);

        for (0..n) |i| {
            const fit = self.getFitness(i);
            pop.individuals[i].fitness = fit;
            if (fit > best_fit) {
                best_fit = fit;
                best_idx = i;
            }
        }

        hdr.best_index = @intCast(best_idx);
        hdr.best_fitness = best_fit;
        pop.best_idx = best_idx;
        pop.best_fitness = best_fit;
    }
};
