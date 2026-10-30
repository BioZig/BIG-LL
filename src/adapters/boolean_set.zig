const std = @import("std");

pub fn BooleanSetDecoder(comptime N: usize) type {
    comptime {
        std.debug.assert(N > 0);
    }
    const words_len = (N + 63) / 64;

    return struct {
        const Self = @This();
        pub const universe_size: usize = N;
        pub const word_count: usize = words_len;

        pub fn countSelected(words: []const u64) usize {
            var sum: usize = 0;
            inline for (0..words_len) |i| {
                const w = if (i == words_len - 1 and (N % 64) != 0)
                    words[i] & ((@as(u64, 1) << @as(u6, @intCast(N % 64))) - 1)
                else
                    words[i];
                sum += @popCount(w);
            }
            return sum;
        }

        pub fn decodeIndices(words: []const u64, out_indices: []usize) usize {
            var count: usize = 0;
            inline for (0..words_len) |i| {
                var w = if (i == words_len - 1 and (N % 64) != 0)
                    words[i] & ((@as(u64, 1) << @as(u6, @intCast(N % 64))) - 1)
                else
                    words[i];

                const base_idx = i * 64;
                while (w != 0) {
                    const bit_idx: usize = @ctz(w);
                    if (count < out_indices.len) {
                        out_indices[count] = base_idx + bit_idx;
                        count += 1;
                    }
                    w &= w - 1;
                }
            }
            return count;
        }

        pub fn evaluateCardinalityPenalty(selected_count: usize, target_cardinality: usize, penalty_weight: f64) f64 {
            if (selected_count <= target_cardinality) return 0.0;
            const excess = selected_count - target_cardinality;
            return @as(f64, @floatFromInt(excess)) * penalty_weight;
        }
    };
}
