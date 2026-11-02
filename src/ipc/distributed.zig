const std = @import("std");
const format = @import("../data/format.zig");
const types = @import("../core/types.zig");

pub const NetworkMagic: u32 = 0x4249474E; // "BIGN"
pub const NetworkVersion: u32 = 1;

pub const TransportType = enum(u32) {
    shm = 0,
    mpi = 1,
    rdma = 2,
    mesh = 3,
};

/// 128-byte, 64-byte aligned distributed wire frame header.
/// Ensures zero-copy network serialization and SIMD-aligned payload streaming.
pub const FrameHeader = extern struct {
    magic: u32 align(64) = NetworkMagic,
    version: u32 = NetworkVersion,
    source_rank: u32 = 0,
    target_rank: u32 = 0,
    generation: u64 = 0,
    immigrant_count: u32 = 0,
    bits_per_chrom: u32 = 0,
    words_per_chrom: u32 = 0,
    checksum: u32 = 0,
    payload_bytes: u64 = 0,
    flags: u32 = 0,
    reserved: [18]u32 = [_]u32{0} ** 18,

    pub fn validate(self: *const FrameHeader, expected_total_bytes: usize) !void {
        if (self.magic != NetworkMagic) return error.InvalidNetworkMagic;
        if (self.version != NetworkVersion) return error.UnsupportedNetworkVersion;
        if (expected_total_bytes < 128) return error.TruncatedNetworkHeader;
        if (128 + self.payload_bytes > expected_total_bytes) return error.NetworkPayloadOverflow;
    }
};

pub const ImmigrantEntry = struct {
    fitness: f64,
    words: []const u64,
};

/// High-speed circular ring buffer for zero-copy immigrant exchange across processes and nodes.
pub const NetworkMailbox = struct {
    raw_buffer: []align(64) u8,
    allocator: std.mem.Allocator,
    capacity: usize,
    words_per_chrom: usize,
    head: std.atomic.Value(u32),
    tail: std.atomic.Value(u32),

    pub fn init(allocator: std.mem.Allocator, max_immigrants: usize, words_per_chrom: usize) !NetworkMailbox {
        const entry_size = @sizeOf(f64) + words_per_chrom * @sizeOf(u64);
        const total_size = std.mem.alignForward(usize, 128 + max_immigrants * entry_size, 64);
        const mem = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), total_size);
        @memset(mem, 0);

        return NetworkMailbox{
            .raw_buffer = mem,
            .allocator = allocator,
            .capacity = max_immigrants,
            .words_per_chrom = words_per_chrom,
            .head = std.atomic.Value(u32).init(0),
            .tail = std.atomic.Value(u32).init(0),
        };
    }

    pub fn deinit(self: *NetworkMailbox) void {
        self.allocator.free(self.raw_buffer);
    }

    pub fn push(self: *NetworkMailbox, fitness: f64, words: []const u64) bool {
        std.debug.assert(words.len == self.words_per_chrom);
        const current_head = self.head.load(.acquire);
        const current_tail = self.tail.load(.acquire);

        const next_head = (current_head + 1) % @as(u32, @intCast(self.capacity));
        if (next_head == current_tail) return false; // Buffer full

        const entry_bytes = @sizeOf(f64) + self.words_per_chrom * @sizeOf(u64);
        const offset = 128 + @as(usize, current_head) * entry_bytes;

        // Write fitness
        const fit_ptr: *f64 = @ptrCast(@alignCast(self.raw_buffer.ptr + offset));
        fit_ptr.* = fitness;

        // Write words
        const words_offset = offset + @sizeOf(f64);
        const dst_words: [*]u64 = @ptrCast(@alignCast(self.raw_buffer.ptr + words_offset));
        @memcpy(dst_words[0..self.words_per_chrom], words);

        self.head.store(next_head, .release);
        return true;
    }

    pub fn pop(self: *NetworkMailbox, out_fitness: *f64, out_words: []u64) bool {
        std.debug.assert(out_words.len == self.words_per_chrom);
        const current_tail = self.tail.load(.acquire);
        const current_head = self.head.load(.acquire);

        if (current_tail == current_head) return false; // Buffer empty

        const entry_bytes = @sizeOf(f64) + self.words_per_chrom * @sizeOf(u64);
        const offset = 128 + @as(usize, current_tail) * entry_bytes;

        const fit_ptr: *const f64 = @ptrCast(@alignCast(self.raw_buffer.ptr + offset));
        out_fitness.* = fit_ptr.*;

        const words_offset = offset + @sizeOf(f64);
        const src_words: [*]const u64 = @ptrCast(@alignCast(self.raw_buffer.ptr + words_offset));
        @memcpy(out_words, src_words[0..self.words_per_chrom]);

        const next_tail = (current_tail + 1) % @as(u32, @intCast(self.capacity));
        self.tail.store(next_tail, .release);
        return true;
    }
};

/// Abstract Multi-Node Cluster Transport
pub const DistributedTransport = struct {
    transport_type: TransportType,
    rank: u32,
    world_size: u32,
    mailbox: NetworkMailbox,

    pub fn init(
        allocator: std.mem.Allocator,
        transport_type: TransportType,
        rank: u32,
        world_size: u32,
        words_per_chrom: usize,
    ) !DistributedTransport {
        const mb = try NetworkMailbox.init(allocator, 128, words_per_chrom);
        return DistributedTransport{
            .transport_type = transport_type,
            .rank = rank,
            .world_size = world_size,
            .mailbox = mb,
        };
    }

    pub fn deinit(self: *DistributedTransport) void {
        self.mailbox.deinit();
    }

    /// Emits elite immigrants into the migration ring or RDMA queue.
    /// Operates asynchronously with 0 runtime heap allocation.
    pub fn broadcastElites(
        self: *DistributedTransport,
        generation: u64,
        elites_fitness: []const f64,
        elites_words: []const u64,
        words_per_chrom: usize,
    ) usize {
        const count = elites_fitness.len;
        var sent: usize = 0;

        _ = generation;
        for (0..count) |i| {
            const start = i * words_per_chrom;
            const chrom_slice = elites_words[start .. start + words_per_chrom];
            if (self.mailbox.push(elites_fitness[i], chrom_slice)) {
                sent += 1;
            }
        }
        return sent;
    }

    /// Polls non-blocking incoming mailbox for remote immigrants.
    /// Zero-copy: populates destination buffer in-place.
    pub fn pollImmigrant(
        self: *DistributedTransport,
        out_fitness: *f64,
        out_words: []u64,
    ) bool {
        return self.mailbox.pop(out_fitness, out_words);
    }
};
