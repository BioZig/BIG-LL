const std = @import("std");
const types = @import("../core/types.zig");
const gray = @import("../adapters/gray_code.zig");
const minlp = @import("../adapters/minlp.zig");
const format = @import("../data/format.zig");

pub fn ParameterizedFinanceDomain(comptime num_assets: usize, comptime num_rules: usize, comptime bits_per_weight: usize) type {
    return struct {
        pub const NUM_RULES: usize = num_rules;
        pub const NUM_ASSETS: usize = num_assets;
        pub const BITS_PER_WEIGHT: usize = bits_per_weight;
        pub const CHROM_LEN: usize = NUM_RULES + (NUM_ASSETS * BITS_PER_WEIGHT);
        pub const Chrom = types.Chromosome(CHROM_LEN);

        pub const PortfolioState = struct {
            rules: [NUM_RULES]u1,
            weights: [NUM_ASSETS]f64,
        };

        expected_returns: [NUM_ASSETS]f64,
        cov_matrix: [NUM_ASSETS][NUM_ASSETS]f64,
        risk_free_rate: f64,
        actual_assets: usize = num_assets,

        const weight_params = [_]gray.FloatParam{.{ .min = 0.0, .max = 1.0, .bit_len = bits_per_weight }} ** NUM_ASSETS;
        const Decoder = minlp.MinlpDecoder(NUM_RULES, &weight_params);

        pub fn initDefault() @This() {
            var dom = @This(){
                .expected_returns = undefined,
                .cov_matrix = undefined,
                .risk_free_rate = 0.03,
                .actual_assets = NUM_ASSETS,
            };

            const base_returns = [_]f64{ 0.12, 0.09, 0.15, 0.07, 0.18, 0.05 };
            const base_stds = [_]f64{ 0.18, 0.12, 0.25, 0.08, 0.30, 0.04 };

            for (0..NUM_ASSETS) |i| {
                dom.expected_returns[i] = base_returns[i % base_returns.len] + 0.01 * @sin(@as(f64, @floatFromInt(i)));
            }

            for (0..NUM_ASSETS) |i| {
                const s_i = base_stds[i % base_stds.len];
                for (0..NUM_ASSETS) |j| {
                    const s_j = base_stds[j % base_stds.len];
                    const corr: f64 = if (i == j) 1.0 else 0.30 + 0.15 * @sin(@as(f64, @floatFromInt(i + j)));
                    dom.cov_matrix[i][j] = corr * s_i * s_j;
                }
            }
            return dom;
        }

        pub fn initFromInstance(inst: anytype) !@This() {
            const hdr = inst.header();
            if (hdr.domain_tag != @intFromEnum(format.DomainTag.finance)) return error.DomainMismatch;
            const n = hdr.param_u32_0;
            if (n > NUM_ASSETS) return error.AssetCountMismatch;
            if (hdr.param_u32_1 != NUM_RULES) return error.RuleCountMismatch;

            const returns = try inst.getSlice(f64, hdr.offset_section_a, n);
            const cov_flat = try inst.getSlice(f64, hdr.offset_section_b, n * n);

            var dom = @This(){
                .actual_assets = n,
                .expected_returns = [_]f64{0.0} ** NUM_ASSETS,
                .cov_matrix = [_][NUM_ASSETS]f64{[_]f64{0.0} ** NUM_ASSETS} ** NUM_ASSETS,
                .risk_free_rate = hdr.param_f64_0,
            };
            @memcpy(dom.expected_returns[0..n], returns);
            for (0..n) |i| {
                for (0..n) |j| {
                    dom.cov_matrix[i][j] = cov_flat[i * n + j];
                }
            }
            return dom;
        }

        pub fn decode(self: *const @This(), chrom: *const Chrom) PortfolioState {
            _ = self;
            var decoded: Decoder.DecodedState = undefined;
            Decoder.decode(&chrom.words, &decoded);
            return .{
                .rules = decoded.switches,
                .weights = decoded.scalars,
            };
        }

        pub fn evaluate(self: *const @This(), state: PortfolioState) f64 {
            const n = self.actual_assets;
            var sum_w: f64 = 0.0;
            for (state.weights[0..n]) |w| sum_w += w;
            if (sum_w <= 1e-6) return 0.0;

            const inv_sum_w = 1.0 / sum_w;
            var norm_w: [NUM_ASSETS]f64 = [_]f64{0.0} ** NUM_ASSETS;
            for (0..n) |i| norm_w[i] = state.weights[i] * inv_sum_w;

            var port_return: f64 = 0.0;
            for (0..n) |i| {
                port_return += norm_w[i] * self.expected_returns[i];
            }

            var rule_multiplier: f64 = 1.0;
            if (NUM_RULES >= 3 and state.rules[0] == 1 and state.rules[2] == 1) rule_multiplier += 0.20;
            if (NUM_RULES >= 6 and state.rules[3] == 1 and state.rules[5] == 0) rule_multiplier += 0.15;
            if (NUM_RULES >= 8 and state.rules[6] == 1 and state.rules[7] == 1) rule_multiplier -= 0.30;

            port_return *= rule_multiplier;

            var port_var: f64 = 0.0;
            for (0..n) |i| {
                const w_i = norm_w[i];
                if (w_i <= 1e-6) continue;
                var row_sum: f64 = 0.0;
                for (0..n) |j| {
                    if (norm_w[j] > 1e-6) {
                        row_sum += norm_w[j] * self.cov_matrix[i][j];
                    }
                }
                port_var += w_i * row_sum;
            }
            const port_vol = @sqrt(@max(1e-6, port_var));

            var entropy_bonus: f64 = 0.0;
            for (norm_w[0..n]) |w| {
                if (w > 1e-6) {
                    entropy_bonus -= w * @log(w);
                }
            }

            const sharpe = (port_return - self.risk_free_rate) / port_vol;
            return @max(0.0, sharpe + (0.1 * entropy_bonus));
        }
    };
}

pub const FinanceDomain = ParameterizedFinanceDomain(6, 8, 16);
