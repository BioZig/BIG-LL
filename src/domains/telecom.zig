const std = @import("std");
const types = @import("../core/types.zig");
const gray = @import("../adapters/gray_code.zig");
const format = @import("../data/format.zig");

pub fn ParameterizedTelecomDomain(
    comptime num_users: usize,
    comptime num_channels: usize,
    comptime bits_per_channel: usize,
    comptime bits_per_power: usize,
) type {
    return struct {
        pub const NUM_USERS: usize = num_users;
        pub const NUM_CHANNELS: usize = num_channels;
        pub const BITS_PER_CHANNEL: usize = bits_per_channel;
        pub const BITS_PER_POWER: usize = bits_per_power;
        pub const CHROM_LEN: usize = (NUM_USERS * BITS_PER_CHANNEL) + (NUM_USERS * BITS_PER_POWER);
        pub const Chrom = types.Chromosome(CHROM_LEN);

        pub const Allocation = struct {
            channels: [NUM_USERS]u32,
            powers: [NUM_USERS]f64,
        };

        channel_matrix: [NUM_USERS][NUM_USERS]f64,
        noise_power: f64,

        const power_params = [_]gray.FloatParam{.{ .min = 0.1, .max = 2.0, .bit_len = bits_per_power }} ** NUM_USERS;
        const PowerDecoder = gray.GrayDecoder(&power_params);

        pub fn initDefault() @This() {
            var dom = @This(){
                .channel_matrix = undefined,
                .noise_power = 0.05,
            };

            for (0..NUM_USERS) |i| {
                for (0..NUM_USERS) |j| {
                    if (i == j) {
                        dom.channel_matrix[i][j] = 1.0;
                    } else {
                        const diff: f64 = @floatFromInt(if (i > j) i - j else j - i);
                        dom.channel_matrix[i][j] = 0.4 / (1.0 + diff * diff);
                    }
                }
            }
            return dom;
        }

        pub fn initFromInstance(inst: anytype) !@This() {
            const hdr = inst.header();
            if (hdr.domain_tag != @intFromEnum(format.DomainTag.telecom)) return error.DomainMismatch;
            if (hdr.param_u32_0 != NUM_USERS) return error.UserCountMismatch;
            if (hdr.param_u32_1 != NUM_CHANNELS) return error.ChannelCountMismatch;

            const matrix_flat = try inst.getSlice(f64, hdr.offset_section_a, NUM_USERS * NUM_USERS);
            var dom = @This(){
                .channel_matrix = undefined,
                .noise_power = hdr.param_f64_0,
            };
            for (0..NUM_USERS) |i| {
                for (0..NUM_USERS) |j| {
                    dom.channel_matrix[i][j] = matrix_flat[i * NUM_USERS + j];
                }
            }
            return dom;
        }

        pub fn decode(self: *const @This(), chrom: *const Chrom) Allocation {
            _ = self;
            var alloc: Allocation = undefined;

            for (0..NUM_USERS) |i| {
                const raw = gray.extractBits(&chrom.words, i * BITS_PER_CHANNEL, BITS_PER_CHANNEL);
                alloc.channels[i] = @intCast(raw % NUM_CHANNELS);
            }

            const power_offset = NUM_USERS * BITS_PER_CHANNEL;
            var power_words = [_]u64{0} ** ((NUM_USERS * BITS_PER_POWER + 63) / 64);
            for (0..NUM_USERS) |i| {
                const p_raw = gray.extractBits(&chrom.words, power_offset + i * BITS_PER_POWER, BITS_PER_POWER);
                gray.insertBits(&power_words, i * BITS_PER_POWER, BITS_PER_POWER, p_raw);
            }
            PowerDecoder.decode(&power_words, &alloc.powers);

            return alloc;
        }

        pub fn evaluate(self: *const @This(), alloc: Allocation) f64 {
            var sum_rate: f64 = 0.0;
            var fairness_penalty: f64 = 0.0;

            for (0..NUM_USERS) |i| {
                const c_i = alloc.channels[i];
                const s_i = alloc.powers[i] * self.channel_matrix[i][i];
                var interference: f64 = 0.0;

                for (0..NUM_USERS) |j| {
                    if (i != j and alloc.channels[j] == c_i) {
                        interference += alloc.powers[j] * self.channel_matrix[j][i];
                    }
                }

                const sinr = s_i / (self.noise_power + interference);
                const rate = @log2(1.0 + sinr);
                sum_rate += rate;

                if (sinr < 0.25) {
                    fairness_penalty += 2.0;
                }
            }

            return @max(0.0, sum_rate - fairness_penalty);
        }
    };
}

pub const TelecomDomain = ParameterizedTelecomDomain(16, 4, 2, 8);
