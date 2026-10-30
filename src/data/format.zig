const std = @import("std");

pub const Magic: u32 = 0x42494744; // "BIGD"
pub const CurrentVersion: u32 = 1;

pub const DomainTag = enum(u32) {
    logistics = 1,
    power_grid = 2,
    finance = 3,
    bioinformatics = 4,
    airfoil = 5,
    telecom = 6,
};

pub const BigDHeader = extern struct {
    magic: u32 align(64) = Magic,
    version: u32 = CurrentVersion,
    domain_tag: u32,
    flags: u32 = 0,
    total_bytes: u64,
    checksum: u32 = 0,
    param_u32_4: u32 = 0,
    param_u32_0: u32 = 0,
    param_u32_1: u32 = 0,
    param_u32_2: u32 = 0,
    param_u32_3: u32 = 0,
    param_f64_0: f64 = 0.0,
    param_f64_1: f64 = 0.0,
    param_f64_2: f64 = 0.0,
    param_f64_3: f64 = 0.0,
    offset_section_a: u64 = 128,
    offset_section_b: u64 = 0,
    offset_section_c: u64 = 0,
    reserved: [3]u64 = [_]u64{0} ** 3,

    pub fn validate(self: *const BigDHeader, actual_size: usize) !void {
        if (self.magic != Magic) return error.InvalidMagic;
        if (self.version != CurrentVersion) return error.UnsupportedVersion;
        if (self.total_bytes != actual_size) return error.SizeMismatch;
        if (self.total_bytes < 128) return error.TruncatedHeader;
        if (self.offset_section_a < 128 or self.offset_section_a % 64 != 0) return error.UnalignedSectionA;
        if (self.offset_section_b != 0 and (self.offset_section_b < self.offset_section_a or self.offset_section_b % 64 != 0)) return error.UnalignedSectionB;
        if (self.offset_section_c != 0 and (self.offset_section_c < self.offset_section_b or self.offset_section_c % 64 != 0)) return error.UnalignedSectionC;
    }
};

comptime {
    std.debug.assert(@sizeOf(BigDHeader) == 128);
    std.debug.assert(@alignOf(BigDHeader) == 64);
}

pub fn computeChecksum(data: []const u8) u32 {
    var c0: u32 = 0xFFFF;
    var c1: u32 = 0xFFFF;
    var idx: usize = 0;
    while (idx < data.len) {
        const block = @min(data.len - idx, 359);
        const end = idx + block;
        while (idx < end) : (idx += 1) {
            c0 = c0 +% @as(u32, data[idx]);
            c1 = c1 +% c0;
        }
        c0 = (c0 & 0xFFFF) +% (c0 >> 16);
        c1 = (c1 & 0xFFFF) +% (c1 >> 16);
    }
    c0 = (c0 & 0xFFFF) +% (c0 >> 16);
    c1 = (c1 & 0xFFFF) +% (c1 >> 16);
    return (c1 << 16) | c0;
}

// Domain-specific metadata structures

pub const EpistasisPair = extern struct {
    idx1: u32,
    idx2: u32,
    weight: f64,
};

pub const EpistasisTriplet = extern struct {
    idx1: u32,
    idx2: u32,
    idx3: u32,
    pad: u32 = 0,
    weight: f64,
};
