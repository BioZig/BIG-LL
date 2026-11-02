const std = @import("std");
const gray = @import("gray_code.zig");

pub fn RandomKeyDecoder(comptime M: usize, comptime KeyBits: usize) type {
    comptime {
        std.debug.assert(M > 0);
        std.debug.assert(KeyBits >= 8 and KeyBits <= 32);
    }

    const total_bits: usize = M * KeyBits;

    return struct {
        const Self = @This();
        pub const size: usize = M;
        pub const key_bits: usize = KeyBits;
        pub const total_bit_len: usize = total_bits;

        const Item = struct {
            key: u32,
            idx: u32,
        };

        fn itemLessThan(_: void, a: Item, b: Item) bool {
            if (a.key == b.key) return a.idx < b.idx;
            return a.key < b.key;
        }

        pub fn decodeCount(words: []const u64, out_perm: *[M]u32, actual_m: usize) void {
            const count = @min(actual_m, M);
            var items: [M]Item = undefined;

            for (0..count) |i| {
                const bit_offset = i * KeyBits;
                const raw = gray.extractBits(words, bit_offset, KeyBits);
                items[i] = .{
                    .key = @intCast(raw),
                    .idx = @intCast(i),
                };
            }

            std.sort.block(Item, items[0..count], {}, itemLessThan);

            for (0..count) |i| {
                out_perm[i] = items[i].idx;
            }
            for (count..M) |i| {
                out_perm[i] = @intCast(i);
            }
        }

        pub fn decodeFastCount(words: []const u64, out_perm: *[M]u32, actual_m: usize) void {
            const count = @min(actual_m, M);
            if (count <= 64) {
                var keys: [M]u32 = undefined;
                for (0..count) |i| {
                    const bit_offset = i * KeyBits;
                    keys[i] = @intCast(gray.extractBits(words, bit_offset, KeyBits));
                }

                for (0..count) |i| {
                    var rank: u32 = 0;
                    for (0..count) |j| {
                        if (keys[j] < keys[i] or (keys[j] == keys[i] and j < i)) {
                            rank += 1;
                        }
                    }
                    out_perm[rank] = @intCast(i);
                }
                for (count..M) |i| {
                    out_perm[i] = @intCast(i);
                }
            } else {
                decodeCount(words, out_perm, count);
            }
        }

        pub fn decode(words: []const u64, out_perm: *[M]u32) void {
            var items: [M]Item = undefined;

            inline for (0..M) |i| {
                const bit_offset = i * KeyBits;
                const raw = gray.extractBits(words, bit_offset, KeyBits);
                items[i] = .{
                    .key = @intCast(raw),
                    .idx = @intCast(i),
                };
            }

            std.sort.block(Item, &items, {}, itemLessThan);

            inline for (0..M) |i| {
                out_perm[i] = items[i].idx;
            }
        }

        pub fn decodeFast(words: []const u64, out_perm: *[M]u32) void {
            if (M <= 64) {
                var keys: [M]u32 = undefined;
                inline for (0..M) |i| {
                    const bit_offset = i * KeyBits;
                    keys[i] = @intCast(gray.extractBits(words, bit_offset, KeyBits));
                }

                for (0..M) |i| {
                    var rank: u32 = 0;
                    for (0..M) |j| {
                        if (keys[j] < keys[i] or (keys[j] == keys[i] and j < i)) {
                            rank += 1;
                        }
                    }
                    out_perm[rank] = @intCast(i);
                }
            } else {
                decode(words, out_perm);
            }
        }
    };
}
