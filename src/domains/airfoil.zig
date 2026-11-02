const std = @import("std");
const types = @import("../core/types.zig");
const gray = @import("../adapters/gray_code.zig");
const format = @import("../data/format.zig");

pub const AirfoilDomain = struct {
    pub const NUM_POINTS: usize = 16;
    pub const BITS_PER_POINT: usize = 16;
    pub const CHROM_LEN: usize = NUM_POINTS * BITS_PER_POINT;
    pub const Chrom = types.Chromosome(CHROM_LEN);

    pub const STATIONS_X = [8]f64{ 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95 };

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
    baseline_upper_y: [8]f64 = [_]f64{0.0} ** 8,
    baseline_lower_y: [8]f64 = [_]f64{0.0} ** 8,
    has_baseline: bool = false,

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

        const total_pts = hdr.param_u32_1;
        const raw_pts = if (total_pts > 0)
            try inst.getSlice(Point, hdr.offset_section_a, total_pts)
        else
            null;

        var dom = AirfoilDomain{
            .cl_factor = hdr.param_f64_0,
            .base_cd = hdr.param_f64_1,
            .raw_spline_points = raw_pts,
            .has_baseline = false,
        };

        if (raw_pts) |pts| {
            if (pts.len >= 4) {
                const num_upper: usize = if (hdr.param_u32_2 > 0 and hdr.param_u32_2 <= pts.len)
                    hdr.param_u32_2
                else
                    pts.len / 2;
                const num_lower: usize = if (hdr.param_u32_3 > 0 and num_upper + hdr.param_u32_3 <= pts.len)
                    hdr.param_u32_3
                else
                    pts.len - num_upper;

                const upper_pts = pts[0..num_upper];
                const lower_pts = pts[num_upper .. num_upper + num_lower];

                for (STATIONS_X, 0..) |st_x, i| {
                    dom.baseline_upper_y[i] = interpolateAt(upper_pts, st_x);
                    dom.baseline_lower_y[i] = interpolateAt(lower_pts, st_x);
                }
                dom.has_baseline = true;
            }
        }

        return dom;
    }

    pub fn interpolateAt(points: []const Point, x: f64) f64 {
        if (points.len == 0) return 0.0;
        if (points.len == 1) return points[0].y;

        const is_descending = points[0].x > points[points.len - 1].x;
        if (is_descending) {
            if (x >= points[0].x) return points[0].y;
            if (x <= points[points.len - 1].x) return points[points.len - 1].y;

            for (0..points.len - 1) |i| {
                const p1 = points[i];
                const p2 = points[i + 1];
                if (x <= p1.x and x >= p2.x) {
                    const dx = p1.x - p2.x;
                    if (@abs(dx) < 1e-12) return p1.y;
                    const t = (p1.x - x) / dx;
                    return (1.0 - t) * p1.y + t * p2.y;
                }
            }
            return points[points.len - 1].y;
        } else {
            if (x <= points[0].x) return points[0].y;
            if (x >= points[points.len - 1].x) return points[points.len - 1].y;

            for (0..points.len - 1) |i| {
                const p1 = points[i];
                const p2 = points[i + 1];
                if (x >= p1.x and x <= p2.x) {
                    const dx = p2.x - p1.x;
                    if (@abs(dx) < 1e-12) return p1.y;
                    const t = (x - p1.x) / dx;
                    return (1.0 - t) * p1.y + t * p2.y;
                }
            }
            return points[points.len - 1].y;
        }
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
        var structural_penalty: f64 = 0.0;

        for (0..8) |i| {
            const t = geom.upper_y[i] - geom.lower_y[i];
            if (t < 0.0) return -1000.0;
            thickness_integral += t;
            camber_integral += 0.5 * (geom.upper_y[i] + geom.lower_y[i]);

            if (self.has_baseline) {
                const t_base = self.baseline_upper_y[i] - self.baseline_lower_y[i];
                if (t_base > 0.0) {
                    const min_req_t = t_base * 0.70; // 70% spar thickness compliance
                    if (t < min_req_t) {
                        const def = min_req_t - t;
                        structural_penalty += def * def * 500.0;
                    }
                }
            }
        }

        for (1..7) |i| {
            const d2_upper = geom.upper_y[i + 1] - 2.0 * geom.upper_y[i] + geom.upper_y[i - 1];
            const d2_lower = geom.lower_y[i + 1] - 2.0 * geom.lower_y[i] + geom.lower_y[i - 1];
            smoothness_penalty += (d2_upper * d2_upper) + (d2_lower * d2_lower);
        }

        const te_gap = @abs(geom.upper_y[7] - geom.lower_y[7]);
        const cl = camber_integral * self.cl_factor;
        const cd = self.base_cd + (thickness_integral * 0.05) + (smoothness_penalty * 50.0) + (te_gap * 10.0) + structural_penalty;

        if (cd <= 1e-6) return 0.0;
        return cl / cd;
    }
};
