const std = @import("std");
const types = @import("../core/types.zig");
const rk = @import("../adapters/random_key.zig");
const format = @import("../data/format.zig");

pub fn ParameterizedLogisticsDomain(comptime num_customers: usize, comptime key_bits: usize) type {
    return struct {
        pub const NUM_CUSTOMERS: usize = num_customers;
        pub const KEY_BITS: usize = key_bits;
        pub const CHROM_LEN: usize = NUM_CUSTOMERS * KEY_BITS;
        pub const Chrom = types.Chromosome(CHROM_LEN);

        pub const Point = struct {
            x: f64,
            y: f64,
        };

        pub const Customer = struct {
            loc: Point,
            demand: f64,
        };

        depot: Point,
        customers: [NUM_CUSTOMERS]Customer,
        vehicle_capacity: f64,
        distance_matrix: ?[]const f64 = null,

        const PermDecoder = rk.RandomKeyDecoder(NUM_CUSTOMERS, KEY_BITS);

        pub fn initDefault() @This() {
            var dom = @This(){
                .depot = .{ .x = 50.0, .y = 50.0 },
                .customers = undefined,
                .vehicle_capacity = 35.0,
                .distance_matrix = null,
            };

            for (0..NUM_CUSTOMERS) |i| {
                const fi: f64 = @floatFromInt(i);
                const angle = fi * (2.0 * std.math.pi / @as(f64, @floatFromInt(NUM_CUSTOMERS)));
                const radius = 20.0 + 15.0 * @sin(fi * 0.5);
                dom.customers[i] = .{
                    .loc = .{
                        .x = 50.0 + radius * @cos(angle),
                        .y = 50.0 + radius * @sin(angle),
                    },
                    .demand = 3.0 + @mod(fi * 7.0, 8.0),
                };
            }
            return dom;
        }

        pub fn initFromInstance(inst: anytype) !@This() {
            const hdr = inst.header();
            if (hdr.domain_tag != @intFromEnum(format.DomainTag.logistics)) return error.DomainMismatch;
            if (hdr.param_u32_0 != NUM_CUSTOMERS) return error.CustomerCountMismatch;

            const points = try inst.getSlice(Point, hdr.offset_section_a, NUM_CUSTOMERS);
            const demands = try inst.getSlice(f64, hdr.offset_section_b, NUM_CUSTOMERS);
            const dist_matrix = try inst.getSlice(f64, hdr.offset_section_c, (NUM_CUSTOMERS + 1) * (NUM_CUSTOMERS + 1));

            var dom = @This(){
                .depot = .{ .x = hdr.param_f64_1, .y = hdr.param_f64_2 },
                .customers = undefined,
                .vehicle_capacity = hdr.param_f64_0,
                .distance_matrix = dist_matrix,
            };
            for (0..NUM_CUSTOMERS) |i| {
                dom.customers[i] = .{
                    .loc = points[i],
                    .demand = demands[i],
                };
            }
            return dom;
        }

        pub fn decode(self: *const @This(), chrom: *const Chrom) [NUM_CUSTOMERS]u32 {
            _ = self;
            var route: [NUM_CUSTOMERS]u32 = undefined;
            PermDecoder.decode(&chrom.words, &route);
            return route;
        }

        fn dist(p1: Point, p2: Point) f64 {
            const dx = p1.x - p2.x;
            const dy = p1.y - p2.y;
            return @sqrt(dx * dx + dy * dy);
        }

        pub fn getDistance(self: *const @This(), u_node: usize, v_node: usize) f64 {
            if (self.distance_matrix) |dm| {
                const stride = NUM_CUSTOMERS + 1;
                return dm[u_node * stride + v_node];
            } else {
                const p1 = if (u_node == 0) self.depot else self.customers[u_node - 1].loc;
                const p2 = if (v_node == 0) self.depot else self.customers[v_node - 1].loc;
                return dist(p1, p2);
            }
        }

        pub fn splitTour(self: *const @This(), route: [NUM_CUSTOMERS]u32) f64 {
            var V = [_]f64{std.math.inf(f64)} ** (NUM_CUSTOMERS + 1);
            V[0] = 0.0;

            for (0..NUM_CUSTOMERS) |i| {
                if (V[i] == std.math.inf(f64)) continue;

                var load: f64 = 0.0;
                var cost: f64 = 0.0;

                for (i..NUM_CUSTOMERS) |j| {
                    const cust_idx = route[j];
                    const cust_node = cust_idx + 1;
                    load += self.customers[cust_idx].demand;
                    if (load > self.vehicle_capacity) break;

                    if (j == i) {
                        cost = self.getDistance(0, cust_node) + self.getDistance(cust_node, 0);
                    } else {
                        const prev_cust_node = route[j - 1] + 1;
                        cost = cost - self.getDistance(prev_cust_node, 0) +
                            self.getDistance(prev_cust_node, cust_node) +
                            self.getDistance(cust_node, 0);
                    }

                    if (V[i] + cost < V[j + 1]) {
                        V[j + 1] = V[i] + cost;
                    }
                }
            }

            return V[NUM_CUSTOMERS];
        }

        pub fn polishTour(self: *const @This(), route: *[NUM_CUSTOMERS]u32) void {
            var improved = true;
            var iterations: usize = 0;
            const max_iterations: usize = 30;

            while (improved and iterations < max_iterations) : (iterations += 1) {
                improved = false;

                for (0..NUM_CUSTOMERS - 1) |i| {
                    const u_node = route[i] + 1;
                    const u_next_node = route[i + 1] + 1;
                    const d_u_unext = self.getDistance(u_node, u_next_node);

                    for (i + 2..NUM_CUSTOMERS) |j| {
                        const v_node = route[j] + 1;
                        const v_next_node = if (j + 1 < NUM_CUSTOMERS) route[j + 1] + 1 else 0;
                        const d_v_vnext = if (j + 1 < NUM_CUSTOMERS) self.getDistance(v_node, v_next_node) else self.getDistance(v_node, 0);

                        const d_new1 = self.getDistance(u_node, v_node);
                        const d_new2 = if (j + 1 < NUM_CUSTOMERS) self.getDistance(u_next_node, v_next_node) else self.getDistance(u_next_node, 0);

                        const delta = (d_new1 + d_new2) - (d_u_unext + d_v_vnext);
                        if (delta < -1e-6) {
                            var left = i + 1;
                            var right = j;
                            while (left < right) {
                                const tmp = route[left];
                                route[left] = route[right];
                                route[right] = tmp;
                                left += 1;
                                right -= 1;
                            }
                            improved = true;
                            break;
                        }
                    }
                    if (improved) break;
                }
            }
        }

        pub fn evaluateGreedy(self: *const @This(), route: [NUM_CUSTOMERS]u32) f64 {
            var total_dist: f64 = 0.0;
            var current_load: f64 = 0.0;

            if (self.distance_matrix) |dm| {
                const stride = NUM_CUSTOMERS + 1;
                var current_node: usize = 0; // 0 is depot

                for (route) |cust_idx| {
                    const cust = self.customers[cust_idx];
                    const cust_node = cust_idx + 1;
                    if (current_load + cust.demand > self.vehicle_capacity) {
                        total_dist += dm[current_node * stride + 0];
                        current_node = 0;
                        current_load = 0.0;
                    }
                    total_dist += dm[current_node * stride + cust_node];
                    current_node = cust_node;
                    current_load += cust.demand;
                }
                total_dist += dm[current_node * stride + 0];
            } else {
                var current_loc = self.depot;
                for (route) |cust_idx| {
                    const cust = self.customers[cust_idx];
                    if (current_load + cust.demand > self.vehicle_capacity) {
                        total_dist += dist(current_loc, self.depot);
                        current_loc = self.depot;
                        current_load = 0.0;
                    }
                    total_dist += dist(current_loc, cust.loc);
                    current_loc = cust.loc;
                    current_load += cust.demand;
                }
                total_dist += dist(current_loc, self.depot);
            }

            return 10000.0 / (1.0 + total_dist);
        }

        pub fn evaluate(self: *const @This(), route: [NUM_CUSTOMERS]u32) f64 {
            const split_cost = self.splitTour(route);
            if (split_cost < std.math.inf(f64)) {
                return 10000.0 / (1.0 + split_cost);
            }
            return self.evaluateGreedy(route);
        }
    };
}

pub const LogisticsDomain = ParameterizedLogisticsDomain(32, 16);
