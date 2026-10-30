const std = @import("std");
const types = @import("../core/types.zig");
const gray = @import("../adapters/gray_code.zig");
const minlp = @import("../adapters/minlp.zig");
const format = @import("../data/format.zig");

pub const PowerGridDomain = struct {
    pub const NUM_GENERATORS: usize = 8;
    pub const BITS_PER_SCALAR: usize = 16;
    pub const CHROM_LEN: usize = NUM_GENERATORS + (NUM_GENERATORS * BITS_PER_SCALAR);
    pub const Chrom = types.Chromosome(CHROM_LEN);

    pub const GeneratorSpec = struct {
        p_min: f64,
        p_max: f64,
        cost_a: f64,
        cost_b: f64,
        cost_no_load: f64,
    };

    specs: [NUM_GENERATORS]GeneratorSpec,
    demand: f64,
    reserve_target: f64,

    const scalar_params = [_]gray.FloatParam{.{ .min = 10.0, .max = 150.0, .bit_len = 16 }} ** NUM_GENERATORS;
    pub const Decoder = minlp.MinlpDecoder(NUM_GENERATORS, &scalar_params);
    pub const DecodedState = Decoder.DecodedState;

    pub fn initDefault() PowerGridDomain {
        var dom = PowerGridDomain{
            .specs = undefined,
            .demand = 550.0,
            .reserve_target = 630.0,
        };

        const base_specs = [_]GeneratorSpec{
            .{ .p_min = 20.0, .p_max = 120.0, .cost_a = 0.003, .cost_b = 8.5, .cost_no_load = 150.0 },
            .{ .p_min = 15.0, .p_max = 100.0, .cost_a = 0.004, .cost_b = 9.0, .cost_no_load = 120.0 },
            .{ .p_min = 25.0, .p_max = 140.0, .cost_a = 0.002, .cost_b = 7.8, .cost_no_load = 200.0 },
            .{ .p_min = 10.0, .p_max = 80.0, .cost_a = 0.006, .cost_b = 10.2, .cost_no_load = 80.0 },
            .{ .p_min = 30.0, .p_max = 150.0, .cost_a = 0.0025, .cost_b = 8.0, .cost_no_load = 220.0 },
            .{ .p_min = 20.0, .p_max = 110.0, .cost_a = 0.0035, .cost_b = 8.8, .cost_no_load = 140.0 },
            .{ .p_min = 10.0, .p_max = 90.0, .cost_a = 0.005, .cost_b = 9.8, .cost_no_load = 90.0 },
            .{ .p_min = 15.0, .p_max = 70.0, .cost_a = 0.007, .cost_b = 11.0, .cost_no_load = 60.0 },
        };
        dom.specs = base_specs;
        return dom;
    }

    pub fn initFromInstance(inst: anytype) !PowerGridDomain {
        const hdr = inst.header();
        if (hdr.domain_tag != @intFromEnum(format.DomainTag.power_grid)) return error.DomainMismatch;
        if (hdr.param_u32_0 != NUM_GENERATORS) return error.GeneratorCountMismatch;

        const specs_slice = try inst.getSlice(GeneratorSpec, hdr.offset_section_a, NUM_GENERATORS);
        var dom = PowerGridDomain{
            .specs = undefined,
            .demand = hdr.param_f64_0,
            .reserve_target = hdr.param_f64_1,
        };
        @memcpy(&dom.specs, specs_slice);
        return dom;
    }

    pub fn decode(self: *const PowerGridDomain, chrom: *const Chrom) Decoder.DecodedState {
        _ = self;
        var state: Decoder.DecodedState = undefined;
        Decoder.decode(&chrom.words, &state);
        return state;
    }

    pub fn projectPowerBalance(self: *const PowerGridDomain, state: *Decoder.DecodedState) void {
        var total_gen: f64 = 0.0;
        var total_cap: f64 = 0.0;
        var active_count: usize = 0;

        for (0..NUM_GENERATORS) |i| {
            if (state.switches[i] == 1) {
                total_gen += state.scalars[i];
                total_cap += self.specs[i].p_max;
                active_count += 1;
            }
        }

        if (active_count == 0 or total_cap < 1e-6) return;

        const delta = total_gen - self.demand;
        if (@abs(delta) < 1e-3) return;

        const delta_inv_cap = delta / total_cap;

        for (0..NUM_GENERATORS) |i| {
            if (state.switches[i] == 1) {
                const adjusted = state.scalars[i] - (delta_inv_cap * self.specs[i].p_max);
                state.scalars[i] = std.math.clamp(adjusted, self.specs[i].p_min, self.specs[i].p_max);
            }
        }
    }

    pub fn evaluate(self: *const PowerGridDomain, state: Decoder.DecodedState) f64 {
        var total_gen: f64 = 0.0;
        var total_cost: f64 = 0.0;
        var total_capacity: f64 = 0.0;

        for (0..NUM_GENERATORS) |i| {
            if (state.switches[i] == 1) {
                const spec = self.specs[i];
                const p = state.scalars[i];
                total_gen += p;
                total_capacity += spec.p_max;
                total_cost += (spec.cost_a * p * p) + (spec.cost_b * p) + spec.cost_no_load;
            }
        }

        const imbalance = @abs(total_gen - self.demand);
        const power_balance_penalty = imbalance * imbalance * 25.0;

        const reserve_shortfall = if (total_capacity < self.reserve_target) self.reserve_target - total_capacity else 0.0;
        const reserve_penalty = reserve_shortfall * 150.0;

        return 100000.0 / (1.0 + total_cost + power_balance_penalty + reserve_penalty);
    }
};
