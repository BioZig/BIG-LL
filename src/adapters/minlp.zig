const std = @import("std");
const gray = @import("gray_code.zig");

pub fn MinlpDecoder(comptime NumSwitches: usize, comptime ScalarParams: []const gray.FloatParam) type {
    const ScalarDecoder = gray.GrayDecoder(ScalarParams);
    const total_bits: usize = NumSwitches + ScalarDecoder.total_bit_len;

    return struct {
        const Self = @This();
        pub const num_switches: usize = NumSwitches;
        pub const num_scalars: usize = ScalarParams.len;
        pub const total_bit_len: usize = total_bits;

        pub const DecodedState = struct {
            switches: [NumSwitches]u1,
            scalars: [ScalarParams.len]f64,
        };

        pub fn decode(words: []const u64, out: *DecodedState) void {
            inline for (0..NumSwitches) |i| {
                const w_idx = i / 64;
                const b_idx: u6 = @intCast(i % 64);
                out.switches[i] = @truncate((words[w_idx] >> b_idx) & 1);
            }

            var offset: usize = NumSwitches;
            inline for (ScalarParams, 0..) |p, i| {
                out.scalars[i] = gray.decodeFloat(words, offset, p.bit_len, p.min, p.max);
                offset += p.bit_len;
            }
        }

        pub fn encode(words: []u64, state: *const DecodedState) void {
            inline for (0..NumSwitches) |i| {
                const w_idx = i / 64;
                const b_idx: u6 = @intCast(i % 64);
                const mask = @as(u64, 1) << b_idx;
                if (state.switches[i] == 1) {
                    words[w_idx] |= mask;
                } else {
                    words[w_idx] &= ~mask;
                }
            }

            var offset: usize = NumSwitches;
            inline for (ScalarParams, 0..) |p, i| {
                gray.encodeFloat(words, offset, p.bit_len, p.min, p.max, state.scalars[i]);
                offset += p.bit_len;
            }
        }
    };
}

pub const AugmentedLagrangianConstraint = struct {
    mu: f64 = 100.0,
    min_mu: f64 = 1.0,
    max_mu: f64 = 1_000_000.0,
    increase_factor: f64 = 1.2,
    decrease_factor: f64 = 0.9,
    violation_count: usize = 0,
    total_evals: usize = 0,

    pub fn computePenalty(self: *const @This(), violations: []const f64) f64 {
        var sum_sq: f64 = 0.0;
        for (violations) |v| {
            if (v > 0.0) {
                sum_sq += v * v;
            }
        }
        return self.mu * sum_sq;
    }

    pub fn recordEvaluation(self: *@This(), is_violating: bool) void {
        self.total_evals += 1;
        if (is_violating) {
            self.violation_count += 1;
        }
    }

    pub fn updatePenalty(self: *@This()) void {
        if (self.total_evals == 0) return;
        const ratio = @as(f64, @floatFromInt(self.violation_count)) / @as(f64, @floatFromInt(self.total_evals));
        if (ratio > 0.05) {
            self.mu = @min(self.mu * self.increase_factor, self.max_mu);
        } else if (ratio <= 0.01) {
            self.mu = @max(self.mu * self.decrease_factor, self.min_mu);
        }
        self.violation_count = 0;
        self.total_evals = 0;
    }
};
