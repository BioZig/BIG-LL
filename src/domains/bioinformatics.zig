const std = @import("std");
const types = @import("../core/types.zig");
const bool_set = @import("../adapters/boolean_set.zig");
const format = @import("../data/format.zig");

pub fn ParameterizedBioinformaticsDomain(comptime num_snps: usize) type {
    return struct {
        pub const NUM_SNPS: usize = num_snps;
        pub const CHROM_LEN: usize = NUM_SNPS;
        pub const Chrom = types.Chromosome(CHROM_LEN);

        pub const TargetCard: usize = @max(8, num_snps / 16);
        pub const MaxCard: usize = @max(12, num_snps / 10);

        main_effects: ?[]const f64 = null,
        pairs: ?[]const format.EpistasisPair = null,
        triplets: ?[]const format.EpistasisTriplet = null,
        max_card: usize = MaxCard,
        card_penalty_weight: f64 = 8.0,

        const Decoder = bool_set.BooleanSetDecoder(NUM_SNPS);

        pub const EpistaticPanel = struct {
            indices: [NUM_SNPS]usize,
            count: usize,
        };

        pub fn initDefault() @This() {
            return @This(){};
        }

        pub fn initFromInstance(inst: anytype) !@This() {
            const hdr = inst.header();
            if (hdr.domain_tag != @intFromEnum(format.DomainTag.bioinformatics)) return error.DomainMismatch;
            if (hdr.param_u32_0 != NUM_SNPS) return error.SnpCountMismatch;

            const main_effects = try inst.getSlice(f64, hdr.offset_section_a, NUM_SNPS);
            const pairs = try inst.getSlice(format.EpistasisPair, hdr.offset_section_b, hdr.param_u32_3);
            const triplets = try inst.getSlice(format.EpistasisTriplet, hdr.offset_section_c, hdr.param_u32_4);

            return @This(){
                .main_effects = main_effects,
                .pairs = pairs,
                .triplets = triplets,
                .max_card = hdr.param_u32_2,
                .card_penalty_weight = hdr.param_f64_0,
            };
        }

        pub fn decode(self: *const @This(), chrom: *const Chrom) EpistaticPanel {
            _ = self;
            var panel: EpistaticPanel = undefined;
            panel.count = Decoder.decodeIndices(&chrom.words, &panel.indices);
            return panel;
        }

        pub fn evaluate(self: *const @This(), panel: EpistaticPanel) f64 {
            if (panel.count == 0) return 0.0;

            var score: f64 = 0.0;
            var has_snp = [_]bool{false} ** NUM_SNPS;

            for (panel.indices[0..panel.count]) |idx| {
                if (idx < NUM_SNPS) {
                    has_snp[idx] = true;
                    if (self.main_effects) |me| {
                        score += me[idx];
                    } else {
                        const f_idx: f64 = @floatFromInt(idx);
                        score += 0.2 + 0.1 * @sin(f_idx * 0.3);
                    }
                }
            }

            if (self.pairs) |p_slice| {
                for (p_slice) |p| {
                    if (p.idx1 < NUM_SNPS and p.idx2 < NUM_SNPS) {
                        if (has_snp[p.idx1] and has_snp[p.idx2]) {
                            score += p.weight;
                        }
                    }
                }
            } else {
                const epistasis_pairs = [_][2]usize{
                    .{ 5 % NUM_SNPS, 23 % NUM_SNPS },
                    .{ 12 % NUM_SNPS, 67 % NUM_SNPS },
                    .{ 34 % NUM_SNPS, 89 % NUM_SNPS },
                    .{ 45 % NUM_SNPS, 110 % NUM_SNPS },
                };
                for (epistasis_pairs) |pair| {
                    if (has_snp[pair[0]] and has_snp[pair[1]]) {
                        score += 15.0;
                    }
                }
            }

            if (self.triplets) |t_slice| {
                for (t_slice) |t| {
                    if (t.idx1 < NUM_SNPS and t.idx2 < NUM_SNPS and t.idx3 < NUM_SNPS) {
                        if (has_snp[t.idx1] and has_snp[t.idx2] and has_snp[t.idx3]) {
                            score += t.weight;
                        }
                    }
                }
            } else {
                if (NUM_SNPS >= 100) {
                    if (has_snp[7] and has_snp[42] and has_snp[99]) {
                        score += 35.0;
                    }
                }
            }

            const penalty = Decoder.evaluateCardinalityPenalty(panel.count, self.max_card, self.card_penalty_weight);
            return (100.0 + score) / (1.0 + penalty);
        }

        /// Direct bitboard evaluation bypassing intermediate index array decoding.
        /// Evaluates active SNP main effects and pairwise epistasis directly on the bitboard words.
        pub fn evaluateDirect(self: *const @This(), chrom: *const Chrom) f64 {
            const active_count = chrom.popCount();
            if (active_count == 0) return 0.0;

            var score: f64 = 0.0;
            const words = &chrom.words;

            if (self.main_effects) |me| {
                for (0..NUM_SNPS) |idx| {
                    const w_idx = idx / 64;
                    const b_idx: u6 = @intCast(idx % 64);
                    if ((words[w_idx] & (@as(u64, 1) << b_idx)) != 0) {
                        score += me[idx];
                    }
                }
            } else {
                for (0..NUM_SNPS) |idx| {
                    const w_idx = idx / 64;
                    const b_idx: u6 = @intCast(idx % 64);
                    if ((words[w_idx] & (@as(u64, 1) << b_idx)) != 0) {
                        const f_idx: f64 = @floatFromInt(idx);
                        score += 0.2 + 0.1 * @sin(f_idx * 0.3);
                    }
                }
            }

            if (self.pairs) |p_slice| {
                for (p_slice) |p| {
                    if (p.idx1 < NUM_SNPS and p.idx2 < NUM_SNPS) {
                        const w1 = p.idx1 / 64;
                        const b1: u6 = @intCast(p.idx1 % 64);
                        const w2 = p.idx2 / 64;
                        const b2: u6 = @intCast(p.idx2 % 64);
                        if ((words[w1] & (@as(u64, 1) << b1)) != 0 and (words[w2] & (@as(u64, 1) << b2)) != 0) {
                            score += p.weight;
                        }
                    }
                }
            }

            const penalty = Decoder.evaluateCardinalityPenalty(active_count, self.max_card, self.card_penalty_weight);
            return (100.0 + score) / (1.0 + penalty);
        }
    };
}

pub const BioinformaticsDomain = ParameterizedBioinformaticsDomain(128);
