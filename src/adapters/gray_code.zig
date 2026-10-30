const std = @import("std");

pub inline fn binaryToGray(b: u64) u64 {
    return b ^ (b >> 1);
}

pub inline fn grayToBinary(g: u64) u64 {
    var b = g;
    b ^= b >> 1;
    b ^= b >> 2;
    b ^= b >> 4;
    b ^= b >> 8;
    b ^= b >> 16;
    b ^= b >> 32;
    return b;
}

pub fn extractBits(words: []const u64, bit_offset: usize, bit_len: usize) u64 {
    std.debug.assert(bit_len > 0 and bit_len <= 64);
    const w_idx = bit_offset / 64;
    const b_idx: u6 = @intCast(bit_offset % 64);
    const mask = if (bit_len == 64) ~@as(u64, 0) else (@as(u64, 1) << @as(u6, @intCast(bit_len))) - 1;

    if (@as(usize, b_idx) + bit_len <= 64) {
        return (words[w_idx] >> b_idx) & mask;
    } else {
        const part1 = words[w_idx] >> b_idx;
        const remaining: u6 = @intCast(64 - @as(usize, b_idx));
        const part2 = words[w_idx + 1] << remaining;
        return (part1 | part2) & mask;
    }
}

pub fn insertBits(words: []u64, bit_offset: usize, bit_len: usize, val: u64) void {
    std.debug.assert(bit_len > 0 and bit_len <= 64);
    const w_idx = bit_offset / 64;
    const b_idx: u6 = @intCast(bit_offset % 64);
    const mask = if (bit_len == 64) ~@as(u64, 0) else (@as(u64, 1) << @as(u6, @intCast(bit_len))) - 1;
    const clean_val = val & mask;

    if (@as(usize, b_idx) + bit_len <= 64) {
        const shifted_mask = mask << b_idx;
        words[w_idx] = (words[w_idx] & ~shifted_mask) | (clean_val << b_idx);
    } else {
        const part1_bits: u6 = @intCast(64 - @as(usize, b_idx));
        const part1_mask = (@as(u64, 1) << part1_bits) - 1;
        words[w_idx] = (words[w_idx] & ~(part1_mask << b_idx)) | ((clean_val & part1_mask) << b_idx);

        const part2_bits: u6 = @intCast(bit_len - part1_bits);
        const part2_mask = (@as(u64, 1) << part2_bits) - 1;
        words[w_idx + 1] = (words[w_idx + 1] & ~part2_mask) | ((clean_val >> part1_bits) & part2_mask);
    }
}

pub fn decodeFloat(words: []const u64, bit_offset: usize, bit_len: usize, min_val: f64, max_val: f64) f64 {
    const raw_gray = extractBits(words, bit_offset, bit_len);
    const bin_val = grayToBinary(raw_gray);
    const max_int: f64 = if (bit_len == 64) 18446744073709551615.0 else @floatFromInt((@as(u64, 1) << @as(u6, @intCast(bit_len))) - 1);
    const norm = @as(f64, @floatFromInt(bin_val)) / max_int;
    return min_val + norm * (max_val - min_val);
}

pub fn encodeFloat(words: []u64, bit_offset: usize, bit_len: usize, min_val: f64, max_val: f64, value: f64) void {
    const clamped = std.math.clamp(value, min_val, max_val);
    const max_int: f64 = if (bit_len == 64) 18446744073709551615.0 else @floatFromInt((@as(u64, 1) << @as(u6, @intCast(bit_len))) - 1);
    const norm = (clamped - min_val) / (max_val - min_val);
    const bin_val: u64 = @intFromFloat(@round(norm * max_int));
    const gray_val = binaryToGray(bin_val);
    insertBits(words, bit_offset, bit_len, gray_val);
}

pub const FloatParam = struct {
    min: f64,
    max: f64,
    bit_len: usize,
};

pub fn GrayDecoder(comptime Params: []const FloatParam) type {
    comptime {
        var total_bits: usize = 0;
        for (Params) |p| {
            std.debug.assert(p.bit_len > 0 and p.bit_len <= 64);
            std.debug.assert(p.max > p.min);
            total_bits += p.bit_len;
        }
    }

    return struct {
        const Self = @This();
        pub const dimension: usize = Params.len;
        pub const total_bit_len: usize = blk: {
            var sum: usize = 0;
            for (Params) |p| sum += p.bit_len;
            break :blk sum;
        };

        pub fn decode(words: []const u64, out: *[dimension]f64) void {
            var offset: usize = 0;
            inline for (Params, 0..) |p, i| {
                out[i] = decodeFloat(words, offset, p.bit_len, p.min, p.max);
                offset += p.bit_len;
            }
        }
    };
}
