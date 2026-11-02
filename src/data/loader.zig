const std = @import("std");
const format = @import("format.zig");
const importer = @import("importer.zig");

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
    IoError,
    UnsupportedFormat,
    OutOfMemory,
    InvalidCharacter,
    Overflow,
    UnexpectedSection,
    MissingField,
    DimensionMismatch,
    WriteFailed,
};

pub const BackingType = enum {
    mmap,
    ephemeral_ram,
    borrowed,
};

/// Unified polymorphic container for .bigd instances.
/// Guarantees that raw_slice is always 64-byte aligned for L1D cacheline residency,
/// SIMD vectorization, and Metal GPU buffer bridge compatibility.
pub const BigDInstance = struct {
    raw_slice: []align(64) const u8,
    hdr: *const format.BigDHeader,
    allocator: ?std.mem.Allocator = null,
    backing: union(BackingType) {
        mmap: []align(std.heap.page_size_min) const u8,
        ephemeral_ram: []align(64) const u8,
        borrowed: void,
    },

    /// Deinitializes the instance according to its backing store:
    /// - mmap: unmaps the virtual memory pages via std.posix.munmap.
    /// - ephemeral_ram: releases the allocated 64-byte aligned buffer via allocator.
    /// - borrowed: zero cleanup overhead (no-op).
    pub fn deinit(self: *BigDInstance) void {
        switch (self.backing) {
            .mmap => |slice| std.posix.munmap(slice),
            .ephemeral_ram => |slice| {
                if (self.allocator) |alloc| {
                    alloc.free(slice);
                }
            },
            .borrowed => {},
        }
    }

    pub fn header(self: *const BigDInstance) *const format.BigDHeader {
        return self.hdr;
    }

    /// Zero-copy, zero-allocation slice accessor into the 64-byte aligned payload buffer.
    pub fn getSlice(self: *const BigDInstance, comptime T: type, offset: u64, count: usize) DataLoaderError![]const T {
        if (count == 0) return &[_]T{};
        const required_bytes = count * @sizeOf(T);
        const end_offset = offset + required_bytes;
        if (end_offset > self.raw_slice.len) return DataLoaderError.SliceOutOfBounds;
        if (offset % @alignOf(T) != 0) return DataLoaderError.UnalignedSlice;

        const ptr: [*]const T = @ptrCast(@alignCast(self.raw_slice.ptr + offset));
        return ptr[0..count];
    }

    /// Initializes a borrowed BigDInstance over an existing in-memory 64-byte aligned buffer.
    /// Preserves the 0 B runtime heap invariant with zero ownership transfer.
    pub fn init(buffer: []align(64) const u8) !BigDInstance {
        if (buffer.len < 128) return DataLoaderError.TruncatedHeader;
        const hdr_ptr: *const format.BigDHeader = @ptrCast(buffer.ptr);
        try hdr_ptr.validate(buffer.len);

        if (hdr_ptr.checksum != 0) {
            const actual_crc = format.computeChecksum(buffer[128..]);
            if (actual_crc != hdr_ptr.checksum) return DataLoaderError.ChecksumMismatch;
        }

        return BigDInstance{
            .raw_slice = buffer,
            .hdr = hdr_ptr,
            .allocator = null,
            .backing = .{ .borrowed = {} },
        };
    }

    /// Initializes an ephemeral BigDInstance taking ownership of a 64-byte aligned heap buffer.
    /// Strictly zero disk footprint: buffer lives in RAM and is freed on deinit().
    pub fn initEphemeral(buffer: []align(64) const u8, allocator: std.mem.Allocator) !BigDInstance {
        if (buffer.len < 128) return DataLoaderError.TruncatedHeader;
        const hdr_ptr: *const format.BigDHeader = @ptrCast(buffer.ptr);
        try hdr_ptr.validate(buffer.len);

        if (hdr_ptr.checksum != 0) {
            const actual_crc = format.computeChecksum(buffer[128..]);
            if (actual_crc != hdr_ptr.checksum) return DataLoaderError.ChecksumMismatch;
        }

        return BigDInstance{
            .raw_slice = buffer,
            .hdr = hdr_ptr,
            .allocator = allocator,
            .backing = .{ .ephemeral_ram = buffer },
        };
    }

    /// Fast-path memory-mapping for native .bigd files (~1.8 µs, 0 bytes heap allocation).
    pub fn openBigDFile(file_path: [:0]const u8) !BigDInstance {
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
        if (hdr_ptr.validate(file_size)) |_| {} else |err| {
            const aligned_slice: []align(std.heap.page_size_min) const u8 = @alignCast(mem);
            std.posix.munmap(aligned_slice);
            return err;
        }

        if (hdr_ptr.checksum != 0) {
            const actual_crc = format.computeChecksum(mem[128..file_size]);
            if (actual_crc != hdr_ptr.checksum) {
                const aligned_slice: []align(std.heap.page_size_min) const u8 = @alignCast(mem);
                std.posix.munmap(aligned_slice);
                return DataLoaderError.ChecksumMismatch;
            }
        }

        const aligned_page: []align(std.heap.page_size_min) const u8 = @alignCast(mem);
        const aligned_64: []align(64) const u8 = @alignCast(mem);

        return BigDInstance{
            .raw_slice = aligned_64,
            .hdr = hdr_ptr,
            .allocator = null,
            .backing = .{ .mmap = aligned_page },
        };
    }

    pub const openFile = openBigDFile;

    /// Universal auto-dispatch loader:
    /// - Inspects first 16 bytes for magic signatures.
    /// - Magic == 0x42494744 ("BIGD"): Fast-path POSIX mmap (~1.8 µs, zero heap).
    /// - Magic == "\x93NUMPY": Pure in-memory JIT lowering via parseNpy.
    /// - Magic == 0x6C 0x1B 0x01: Pure in-memory JIT lowering via parsePlinkBed.
    /// - Suffix .vrp: Pure in-memory JIT lowering via parseCvrpLib.
    /// - Suffix .dat: Pure in-memory JIT lowering via parseAirfoilDat.
    /// - Suffix .tsv: Pure in-memory JIT lowering via parseBioinformaticsGwas.
    /// - Suffix .csv: Pure in-memory JIT lowering via power grid, telecom, or finance parser.
    /// Strictly zero disk footprint: no intermediate files written to disk.
    pub fn openFileAuto(file_path: [:0]const u8, allocator: std.mem.Allocator) !BigDInstance {
        const fd = try std.posix.openat(
            std.posix.AT.FDCWD,
            file_path,
            .{ .ACCMODE = .RDONLY },
            0,
        );
        defer _ = std.c.close(fd);

        const file_size: usize = @intCast(std.c.lseek(fd, 0, 2));
        _ = std.c.lseek(fd, 0, 0);

        if (file_size < 3) return DataLoaderError.TruncatedHeader;

        var header_bytes: [16]u8 = undefined;
        const to_read = @min(file_size, header_bytes.len);
        var total_read: usize = 0;
        while (total_read < to_read) {
            const rc = std.c.read(fd, header_bytes[total_read..to_read].ptr, to_read - total_read);
            if (rc <= 0) return DataLoaderError.ReadFailed;
            total_read += @intCast(rc);
        }

        // 1. Native .bigd Magic
        if (to_read >= 4 and std.mem.readInt(u32, header_bytes[0..4], .little) == format.Magic) {
            return openBigDFile(file_path);
        }

        // 2. NumPy .npy Magic
        if (to_read >= 6 and std.mem.eql(u8, header_bytes[0..6], &format.NpyMagic)) {
            const npy_bytes = try importer.readFileAlloc(allocator, file_path);
            defer allocator.free(npy_bytes);
            const bigd_buf = try importer.parseNpy(allocator, npy_bytes);
            return initEphemeral(bigd_buf, allocator);
        }

        // 3. PLINK 1.x .bed Magic (SNP-major mode)
        if (to_read >= 3 and std.mem.eql(u8, header_bytes[0..3], &format.PlinkBedMagic)) {
            const bed_bytes = try importer.readFileAlloc(allocator, file_path);
            defer allocator.free(bed_bytes);

            var opt_bim: ?[]const u8 = null;
            var opt_fam: ?[]const u8 = null;
            defer if (opt_bim) |b| allocator.free(b);
            defer if (opt_fam) |f| allocator.free(f);

            if (std.mem.endsWith(u8, file_path, ".bed")) {
                const base = file_path[0 .. file_path.len - 4];
                const bim_str = try std.fmt.allocPrint(allocator, "{s}.bim", .{base});
                defer allocator.free(bim_str);
                const bim_path = try allocator.dupeZ(u8, bim_str);
                defer allocator.free(bim_path);
                opt_bim = importer.readFileAlloc(allocator, bim_path) catch null;

                const fam_str = try std.fmt.allocPrint(allocator, "{s}.fam", .{base});
                defer allocator.free(fam_str);
                const fam_path = try allocator.dupeZ(u8, fam_str);
                defer allocator.free(fam_path);
                opt_fam = importer.readFileAlloc(allocator, fam_path) catch null;
            }

            const bigd_buf = try importer.parsePlinkBed(allocator, bed_bytes, opt_bim, opt_fam);
            return initEphemeral(bigd_buf, allocator);
        }

        // 4. Raw text formats: check extension and content headers
        const ext = std.fs.path.extension(file_path);
        if (std.mem.eql(u8, ext, ".vrp")) {
            const text = try importer.readFileAlloc(allocator, file_path);
            defer allocator.free(text);
            const bigd_buf = try importer.parseCvrpLib(allocator, text);
            return initEphemeral(bigd_buf, allocator);
        } else if (std.mem.eql(u8, ext, ".dat")) {
            const text = try importer.readFileAlloc(allocator, file_path);
            defer allocator.free(text);
            const bigd_buf = try importer.parseAirfoilDat(allocator, text, 2.5, 0.01);
            return initEphemeral(bigd_buf, allocator);
        } else if (std.mem.eql(u8, ext, ".tsv")) {
            const text = try importer.readFileAlloc(allocator, file_path);
            defer allocator.free(text);

            var opt_int: ?[]const u8 = null;
            defer if (opt_int) |i| allocator.free(i);

            if (std.mem.indexOf(u8, file_path, "_gwas.tsv")) |idx| {
                const int_str = try std.fmt.allocPrint(allocator, "{s}_interactions.tsv", .{file_path[0..idx]});
                defer allocator.free(int_str);
                const int_path = try allocator.dupeZ(u8, int_str);
                defer allocator.free(int_path);
                opt_int = importer.readFileAlloc(allocator, int_path) catch null;
            }

            const bigd_buf = try importer.parseBioinformaticsGwas(allocator, text, opt_int orelse "", 8, 12);
            return initEphemeral(bigd_buf, allocator);
        } else if (std.mem.eql(u8, ext, ".csv")) {
            const text = try importer.readFileAlloc(allocator, file_path);
            defer allocator.free(text);

            if (std.mem.indexOf(u8, text, "unit_id") != null or std.mem.indexOf(u8, text, "cost_a") != null or std.mem.indexOf(u8, text, "IEEE RTS") != null) {
                const bigd_buf = try importer.parsePowerGridCsv(allocator, text, 0, 0);
                return initEphemeral(bigd_buf, allocator);
            } else if (std.mem.indexOf(u8, text, "UE_") != null or std.mem.indexOf(u8, text, "channel") != null) {
                const bigd_buf = try importer.parseTelecomCsv(allocator, text, 4, 0.05);
                return initEphemeral(bigd_buf, allocator);
            } else {
                const bigd_buf = try importer.parseFinanceReturnsCsv(allocator, text, 8, 0.04);
                return initEphemeral(bigd_buf, allocator);
            }
        }

        // Fallback for native .bigd or report accurate validation error
        return openBigDFile(file_path);
    }
};

pub const MemoryMappedInstance = BigDInstance;
pub const StaticBufferInstance = BigDInstance;
