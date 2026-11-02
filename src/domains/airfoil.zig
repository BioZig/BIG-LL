const std = @import("std");
const types = @import("../core/types.zig");
const gray = @import("../adapters/gray_code.zig");
const format = @import("../data/format.zig");

pub const AirfoilDomain = struct {
    pub const NUM_POINTS: usize = 16;
    pub const BITS_PER_POINT: usize = 16;
    pub const CHROM_LEN: usize = NUM_POINTS * BITS_PER_POINT;
    pub const Chrom = types.Chromosome(CHROM_LEN);

    pub const AirfoilGeometry = struct {
        upper_y: [8]f64,
        lower_y: [8]f64,
    };

    pub const Point = struct {
        x: f64,
        y: f64,
    };

    cl_factor: f64 = 2.5,
    base_cd: f64 = 0.01,
    raw_spline_points: ?[]const Point = null,

    comptime {
        std.debug.assert(CHROM_LEN == 256);
    }

    const upper_params = [_]gray.FloatParam{.{ .min = 0.0, .max = 0.25, .bit_len = 16 }} ** 8;
    const lower_params = [_]gray.FloatParam{.{ .min = -0.20, .max = 0.0, .bit_len = 16 }} ** 8;

    const UpperDecoder = gray.GrayDecoder(&upper_params);
    const LowerDecoder = gray.GrayDecoder(&lower_params);

    pub fn initDefault() AirfoilDomain {
        return AirfoilDomain{};
    }

    pub fn initFromInstance(inst: anytype) !AirfoilDomain {
        const hdr = inst.header();
        if (hdr.domain_tag != @intFromEnum(format.DomainTag.airfoil)) return error.DomainMismatch;
        if (hdr.param_u32_0 != NUM_POINTS) return error.PointCountMismatch;

        const raw_pts = if (hdr.param_u32_1 > 0)
            try inst.getSlice(Point, hdr.offset_section_a, hdr.param_u32_1)
        else
            null;

        return AirfoilDomain{
            .cl_factor = hdr.param_f64_0,
            .base_cd = hdr.param_f64_1,
            .raw_spline_points = raw_pts,
        };
    }

    pub fn decode(self: *const AirfoilDomain, chrom: *const Chrom) AirfoilGeometry {
        _ = self;
        var geom: AirfoilGeometry = undefined;
        UpperDecoder.decode(chrom.words[0..2], &geom.upper_y);
        LowerDecoder.decode(chrom.words[2..4], &geom.lower_y);
        return geom;
    }

    pub fn evaluate(self: *const AirfoilDomain, geom: AirfoilGeometry) f64 {
        var camber_integral: f64 = 0.0;
        var thickness_integral: f64 = 0.0;
        var smoothness_penalty: f64 = 0.0;

        for (0..8) |i| {
            const t = geom.upper_y[i] - geom.lower_y[i];
            if (t < 0.0) return -1000.0;
            thickness_integral += t;
            camber_integral += 0.5 * (geom.upper_y[i] + geom.lower_y[i]);
        }

        for (1..7) |i| {
            const d2_upper = geom.upper_y[i + 1] - 2.0 * geom.upper_y[i] + geom.upper_y[i - 1];
            const d2_lower = geom.lower_y[i + 1] - 2.0 * geom.lower_y[i] + geom.lower_y[i - 1];
            smoothness_penalty += (d2_upper * d2_upper) + (d2_lower * d2_lower);
        }

        const te_gap = @abs(geom.upper_y[7] - geom.lower_y[7]);
        const cl = camber_integral * self.cl_factor;
        const cd = self.base_cd + (thickness_integral * 0.05) + (smoothness_penalty * 50.0) + (te_gap * 10.0);

        if (cd <= 1e-6) return 0.0;
        return cl / cd;
    }
};
