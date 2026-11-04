const std = @import("std");
const types = @import("types.zig");
const engine = @import("engine.zig");
const pool = @import("pool.zig");

pub fn IslandEngine(
    comptime L: usize,
    comptime Adapter: type,
    comptime num_islands: usize,
    comptime migration_interval: usize,
) type {
    comptime {
        std.debug.assert(num_islands >= 2);
        std.debug.assert(migration_interval >= 1);
    }

    const Chrom = types.Chromosome(L);
    const SubEngine = engine.GeneticEngineCustom(L, Adapter, .{ .enable_dedup = false, .num_threads = 1 });
    const migration_count: usize = 2;

    return struct {
        const Self = @This();
        pub const islands_count = num_islands;
        pub const interval = migration_interval;

        allocator: std.mem.Allocator,
        islands: [num_islands]SubEngine,
        migration_buffer: [num_islands][migration_count]Chrom,
        total_pop_size: usize,
        sub_pop_size: usize,
        generation: usize,
        thread_pool: *pool.StaticThreadPool,

        pub fn init(
            allocator: std.mem.Allocator,
            total_pop_size: usize,
            adapter: Adapter,
            seed: u64,
        ) !Self {
            const sub_pop = total_pop_size / num_islands;
            std.debug.assert(sub_pop >= SubEngine.elite_size);

            const p = try allocator.create(pool.StaticThreadPool);
            p.* = .{};
            try p.init(num_islands);

            var self: Self = undefined;
            self.allocator = allocator;
            self.total_pop_size = total_pop_size;
            self.sub_pop_size = sub_pop;
            self.generation = 0;
            self.thread_pool = p;

            // Heterogeneous asymmetric allocation:
            // Proportional weighting between P-cores and E-cores based on measured IPC and clock frequency ratio (~2.57x)
            const topo = pool.CpuTopology.get();
            const p_islands = @min(num_islands, topo.p_cores);
            const e_islands = num_islands - p_islands;
            const use_heterogeneous = topo.is_heterogeneous and e_islands > 0;

            // Scaled weights: P-core = 257, E-core = 100
            const total_weight: usize = if (use_heterogeneous)
                (@as(usize, p_islands) * 257 + @as(usize, e_islands) * 100)
            else
                num_islands;

            var actual_total: usize = 0;
            for (0..num_islands) |i| {
                var isl_pop = sub_pop;
                if (use_heterogeneous) {
                    const weight: usize = if (i < p_islands) 257 else 100;
                    isl_pop = (total_pop_size * weight) / total_weight;
                    isl_pop = @max(isl_pop, SubEngine.elite_size);
                }
                actual_total += isl_pop;
                const island_seed = seed +% (@as(u64, i) *% 0x9E3779B97F4A7C15) +% 1;
                self.islands[i] = try SubEngine.init(allocator, isl_pop, adapter, island_seed);
            }
            self.total_pop_size = actual_total;

            return self;
        }

        pub fn deinit(self: *Self) void {
            self.thread_pool.deinit();
            self.allocator.destroy(self.thread_pool);
            for (&self.islands) |*isl| {
                isl.deinit();
            }
        }

        pub fn stepEpoch(self: *Self) !void {
            try self.stepCount(migration_interval);
        }

        pub fn stepCount(self: *Self, count: usize) !void {
            if (count == 0) return;

            const Context = struct {
                islands: []SubEngine,
                steps: usize,

                fn run(ctx_ptr: *anyopaque, worker_id: usize) void {
                    const ctx: *@This() = @ptrCast(@alignCast(ctx_ptr));
                    var isl = &ctx.islands[worker_id];
                    for (0..ctx.steps) |_| {
                        isl.step() catch return;
                    }
                }
            };

            var ctx = Context{
                .islands = &self.islands,
                .steps = count,
            };

            self.thread_pool.dispatch(&ctx, Context.run);

            self.generation += count;

            if (self.generation % migration_interval == 0) {
                self.migrate();
            }
        }

        fn migrate(self: *Self) void {
            for (0..num_islands) |i| {
                const inds = self.islands[i].pop_current.individuals;
                var top0: usize = 0;
                var top1: usize = 0;
                var max0: f64 = -std.math.inf(f64);
                var max1: f64 = -std.math.inf(f64);

                for (inds, 0..) |ind, idx| {
                    if (ind.fitness > max0) {
                        max1 = max0;
                        top1 = top0;
                        max0 = ind.fitness;
                        top0 = idx;
                    } else if (ind.fitness > max1) {
                        max1 = ind.fitness;
                        top1 = idx;
                    }
                }
                self.migration_buffer[i][0] = inds[top0];
                self.migration_buffer[i][1] = inds[top1];
            }

            for (0..num_islands) |i| {
                const src_island = (i + num_islands - 1) % num_islands;
                const incoming0 = self.migration_buffer[src_island][0];
                const incoming1 = self.migration_buffer[src_island][1];

                injectElite(&self.islands[i], incoming0);
                injectElite(&self.islands[i], incoming1);
                self.islands[i].pop_current.updateMetrics();
            }
        }

        fn injectElite(isl: *SubEngine, elite: Chrom) void {
            const inds = isl.pop_current.individuals;
            var min_idx: usize = 0;
            var min_fit: f64 = inds[0].fitness;

            for (inds[1..], 1..) |ind, idx| {
                if (ind.fitness < min_fit) {
                    min_fit = ind.fitness;
                    min_idx = idx;
                }
            }

            if (elite.fitness > min_fit) {
                isl.pop_current.individuals[min_idx] = elite;
            }
        }

        pub fn getGlobalBest(self: *const Self) Chrom {
            var best = self.islands[0].getBest();
            for (self.islands[1..]) |*isl| {
                const candidate = isl.getBest();
                if (candidate.fitness > best.fitness) {
                    best = candidate;
                }
            }
            return best;
        }

        pub fn getMeanFitness(self: *const Self) f64 {
            var sum: f64 = 0.0;
            for (&self.islands) |*isl| {
                sum += isl.pop_current.mean_fitness;
            }
            return sum / @as(f64, @floatFromInt(num_islands));
        }

        pub fn getGlobalEntropy(self: *const Self) f64 {
            var sum: f64 = 0.0;
            for (&self.islands) |*isl| {
                sum += isl.pop_current.entropy;
            }
            return sum / @as(f64, @floatFromInt(num_islands));
        }
    };
}
