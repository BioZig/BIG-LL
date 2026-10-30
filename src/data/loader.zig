const std = @import("std");
const format = @import("format.zig");

pub const DataLoaderError = error{
    InvalidMagic,
    UnsupportedVersion,
    SizeMismatch,
    TruncatedHeader,
    UnalignedSectionA,
    UnalignedSectionB,
    UnalignedSectionC,
    ChecksumMismatch,
    SliceOutOfBounds,
    UnalignedSlice,
    ReadFailed,
};

pub const MemoryMappedInstance = struct {
    raw_slice: []align(std.heap.page_size_min) const u8,
    hdr: *const format.BigDHeader,

    pub fn openFile(file_path: [:0]const u8) !MemoryMappedInstance {
        const fd = try std.posix.openat(
            std.posix.AT.FDCWD,
            file_path,
            .{ .ACCMODE = .RDONLY },
            0,
        );
        defer _ = std.c.close(fd);

        const file_size: usize = @intCast(std.c.lseek(fd, 0, 2));
        _ = std.c.lseek(fd, 0, 0);

        if (file_size < 128) return DataLoaderError.TruncatedHeader;

        const mem = try std.posix.mmap(
            null,
            file_size,
            .{ .READ = true },
            .{ .TYPE = .PRIVATE },
            fd,
            0,
        );

        const hdr_ptr: *const format.BigDHeader = @ptrCast(@alignCast(mem.ptr));
        try hdr_ptr.validate(file_size);

        if (hdr_ptr.checksum != 0) {
            const actual_crc = format.computeChecksum(mem[128..file_size]);
            if (actual_crc != hdr_ptr.checksum) {
                const aligned_slice: []align(std.heap.page_size_min) const u8 = @alignCast(mem);
                std.posix.munmap(aligned_slice);
                return DataLoaderError.ChecksumMismatch;
            }
        }

        return MemoryMappedInstance{
            .raw_slice = @alignCast(mem),
            .hdr = hdr_ptr,
        };
    }

    pub fn deinit(self: *MemoryMappedInstance) void {
        std.posix.munmap(self.raw_slice);
    }

    pub fn header(self: *const MemoryMappedInstance) *const format.BigDHeader {
        return self.hdr;
    }

    pub fn getSlice(self: *const MemoryMappedInstance, comptime T: type, offset: u64, count: usize) DataLoaderError![]const T {
        if (count == 0) return &[_]T{};
        const required_bytes = count * @sizeOf(T);
        const end_offset = offset + required_bytes;
        if (end_offset > self.raw_slice.len) return DataLoaderError.SliceOutOfBounds;
        if (offset % @alignOf(T) != 0) return DataLoaderError.UnalignedSlice;

        const ptr: [*]const T = @ptrCast(@alignCast(self.raw_slice.ptr + offset));
        return ptr[0..count];
    }
};

pub const StaticBufferInstance = struct {
    raw_slice: []align(64) const u8,
    hdr: *const format.BigDHeader,

    pub fn init(buffer: []align(64) const u8) !StaticBufferInstance {
        if (buffer.len < 128) return DataLoaderError.TruncatedHeader;
        const hdr_ptr: *const format.BigDHeader = @ptrCast(buffer.ptr);
        try hdr_ptr.validate(buffer.len);

        if (hdr_ptr.checksum != 0) {
            const actual_crc = format.computeChecksum(buffer[128..]);
            if (actual_crc != hdr_ptr.checksum) return DataLoaderError.ChecksumMismatch;
        }

        return StaticBufferInstance{
            .raw_slice = buffer,
            .hdr = hdr_ptr,
        };
    }

    pub fn header(self: *const StaticBufferInstance) *const format.BigDHeader {
        return self.hdr;
    }

    pub fn getSlice(self: *const StaticBufferInstance, comptime T: type, offset: u64, count: usize) DataLoaderError![]const T {
        if (count == 0) return &[_]T{};
        const required_bytes = count * @sizeOf(T);
        const end_offset = offset + required_bytes;
        if (end_offset > self.raw_slice.len) return DataLoaderError.SliceOutOfBounds;
        if (offset % @alignOf(T) != 0) return DataLoaderError.UnalignedSlice;

        const ptr: [*]const T = @ptrCast(@alignCast(self.raw_slice.ptr + offset));
        return ptr[0..count];
    }
};
