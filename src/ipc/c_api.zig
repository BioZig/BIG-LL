const std = @import("std");
const types = @import("../core/types.zig");
const engine = @import("../core/engine.zig");

pub const BigLlEvalFn = ?*const fn (
    genes: [*]const u64,
    word_count: usize,
    user_data: ?*anyopaque,
) callconv(.c) f64;

pub const BigLlOptions = extern struct {
    enable_dedup: bool = false,
    num_threads: usize = 1,
};

pub fn CFunctionAdapter(comptime L: usize) type {
    const words_len = (L + 63) / 64;
    return struct {
        eval_fn: BigLlEvalFn,
        user_data: ?*anyopaque,

        pub fn decode(self: *const @This(), chrom: *const types.Chromosome(L)) [*]const u64 {
            _ = self;
            return &chrom.words;
        }

        pub fn evaluate(self: *const @This(), words_ptr: [*]const u64) f64 {
            if (self.eval_fn) |f| {
                return f(words_ptr, words_len, self.user_data);
            }
            return 0.0;
        }
    };
}

const EngineVTable = struct {
    step: *const fn (ctx: *anyopaque) bool,
    run: *const fn (ctx: *anyopaque, generations: usize) bool,
    get_best_fitness: *const fn (ctx: *anyopaque) f64,
    get_best_genes: *const fn (ctx: *anyopaque, out_buf: [*]u64, buf_words: usize) usize,
    get_best_genes_ptr: *const fn (ctx: *anyopaque) [*]const u64,
    get_generation: *const fn (ctx: *anyopaque) usize,
    get_pop_size: *const fn (ctx: *anyopaque) usize,
    get_chromosome_bits: *const fn (ctx: *anyopaque) usize,
    get_word_count: *const fn (ctx: *anyopaque) usize,
    get_chromosome_words: *const fn (ctx: *anyopaque, index: usize) ?[*]const u64,
    get_chromosome_fitness: *const fn (ctx: *anyopaque, index: usize) f64,
    destroy: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) void,
};

pub const EngineWrapper = struct {
    ctx: *anyopaque,
    vtable: *const EngineVTable,
    allocator: std.mem.Allocator,
    requested_bits: usize,
};

fn ConcreteEngineImplementation(comptime L: usize, comptime options: engine.EngineOptions) type {
    const Adapter = CFunctionAdapter(L);
    const Eng = engine.GeneticEngineCustom(L, Adapter, options);
    const words_len = (L + 63) / 64;

    return struct {
        fn step(ctx: *anyopaque) bool {
            const self: *Eng = @ptrCast(@alignCast(ctx));
            self.step() catch return false;
            return true;
        }

        fn run(ctx: *anyopaque, generations: usize) bool {
            const self: *Eng = @ptrCast(@alignCast(ctx));
            for (0..generations) |_| {
                self.step() catch return false;
            }
            return true;
        }

        fn getBestFitness(ctx: *anyopaque) f64 {
            const self: *const Eng = @ptrCast(@alignCast(ctx));
            return self.pop_current.best_fitness;
        }

        fn getBestGenes(ctx: *anyopaque, out_buf: [*]u64, buf_words: usize) usize {
            const self: *const Eng = @ptrCast(@alignCast(ctx));
            const best = self.getBest();
            const copy_count = @min(buf_words, words_len);
            @memcpy(out_buf[0..copy_count], best.words[0..copy_count]);
            return copy_count;
        }

        fn getBestGenesPtr(ctx: *anyopaque) [*]const u64 {
            const self: *const Eng = @ptrCast(@alignCast(ctx));
            return &self.pop_current.individuals[self.pop_current.best_idx].words;
        }

        fn getGeneration(ctx: *anyopaque) usize {
            const self: *const Eng = @ptrCast(@alignCast(ctx));
            return self.generation;
        }

        fn getPopSize(ctx: *anyopaque) usize {
            const self: *const Eng = @ptrCast(@alignCast(ctx));
            return self.pop_current.individuals.len;
        }

        fn getChromosomeBits(ctx: *anyopaque) usize {
            _ = ctx;
            return L;
        }

        fn getWordCount(ctx: *anyopaque) usize {
            _ = ctx;
            return words_len;
        }

        fn getChromosomeWords(ctx: *anyopaque, index: usize) ?[*]const u64 {
            const self: *const Eng = @ptrCast(@alignCast(ctx));
            if (index >= self.pop_current.individuals.len) return null;
            return &self.pop_current.individuals[index].words;
        }

        fn getChromosomeFitness(ctx: *anyopaque, index: usize) f64 {
            const self: *const Eng = @ptrCast(@alignCast(ctx));
            if (index >= self.pop_current.individuals.len) return -std.math.inf(f64);
            return self.pop_current.individuals[index].fitness;
        }

        fn destroy(ctx: *anyopaque, allocator: std.mem.Allocator) void {
            const self: *Eng = @ptrCast(@alignCast(ctx));
            self.deinit();
            allocator.destroy(self);
        }

        const vtable: EngineVTable = .{
            .step = step,
            .run = run,
            .get_best_fitness = getBestFitness,
            .get_best_genes = getBestGenes,
            .get_best_genes_ptr = getBestGenesPtr,
            .get_generation = getGeneration,
            .get_pop_size = getPopSize,
            .get_chromosome_bits = getChromosomeBits,
            .get_word_count = getWordCount,
            .get_chromosome_words = getChromosomeWords,
            .get_chromosome_fitness = getChromosomeFitness,
            .destroy = destroy,
        };

        fn create(
            allocator: std.mem.Allocator,
            pop_size: usize,
            seed: u64,
            eval_fn: BigLlEvalFn,
            user_data: ?*anyopaque,
        ) !*anyopaque {
            const self = try allocator.create(Eng);
            errdefer allocator.destroy(self);

            const adapter = Adapter{
                .eval_fn = eval_fn,
                .user_data = user_data,
            };

            self.* = try Eng.init(allocator, pop_size, adapter, seed);
            return self;
        }
    };
}

pub fn normalizeLength(requested: usize) ?usize {
    if (requested == 0 or requested > 8192) return null;
    const standard_lengths = [_]usize{ 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
    for (standard_lengths) |len| {
        if (requested <= len) return len;
    }
    return null;
}

fn createEngineForComptime(
    comptime L: usize,
    comptime eng_options: engine.EngineOptions,
    allocator: std.mem.Allocator,
    pop_size: usize,
    seed: u64,
    eval_fn: BigLlEvalFn,
    user_data: ?*anyopaque,
    requested_bits: usize,
) !*EngineWrapper {
    const Impl = ConcreteEngineImplementation(L, eng_options);
    const ctx = try Impl.create(allocator, pop_size, seed, eval_fn, user_data);
    errdefer Impl.destroy(ctx, allocator);

    const wrapper = try allocator.create(EngineWrapper);
    wrapper.* = .{
        .ctx = ctx,
        .vtable = &Impl.vtable,
        .allocator = allocator,
        .requested_bits = requested_bits,
    };
    return wrapper;
}

fn dispatchOptionsForL(
    comptime L: usize,
    allocator: std.mem.Allocator,
    pop_size: usize,
    seed: u64,
    eval_fn: BigLlEvalFn,
    user_data: ?*anyopaque,
    options: BigLlOptions,
    requested_bits: usize,
) !*EngineWrapper {
    if (options.enable_dedup) {
        if (options.num_threads <= 1) {
            return createEngineForComptime(L, .{ .enable_dedup = true, .num_threads = 1 }, allocator, pop_size, seed, eval_fn, user_data, requested_bits);
        } else if (options.num_threads <= 2) {
            return createEngineForComptime(L, .{ .enable_dedup = true, .num_threads = 2 }, allocator, pop_size, seed, eval_fn, user_data, requested_bits);
        } else if (options.num_threads <= 4) {
            return createEngineForComptime(L, .{ .enable_dedup = true, .num_threads = 4 }, allocator, pop_size, seed, eval_fn, user_data, requested_bits);
        } else {
            return createEngineForComptime(L, .{ .enable_dedup = true, .num_threads = 8 }, allocator, pop_size, seed, eval_fn, user_data, requested_bits);
        }
    } else {
        if (options.num_threads <= 1) {
            return createEngineForComptime(L, .{ .enable_dedup = false, .num_threads = 1 }, allocator, pop_size, seed, eval_fn, user_data, requested_bits);
        } else if (options.num_threads <= 2) {
            return createEngineForComptime(L, .{ .enable_dedup = false, .num_threads = 2 }, allocator, pop_size, seed, eval_fn, user_data, requested_bits);
        } else if (options.num_threads <= 4) {
            return createEngineForComptime(L, .{ .enable_dedup = false, .num_threads = 4 }, allocator, pop_size, seed, eval_fn, user_data, requested_bits);
        } else {
            return createEngineForComptime(L, .{ .enable_dedup = false, .num_threads = 8 }, allocator, pop_size, seed, eval_fn, user_data, requested_bits);
        }
    }
}

pub fn createEngine(
    allocator: std.mem.Allocator,
    chromosome_bits: usize,
    pop_size: usize,
    seed: u64,
    eval_fn: BigLlEvalFn,
    user_data: ?*anyopaque,
    options: BigLlOptions,
) ?*EngineWrapper {
    const l = normalizeLength(chromosome_bits) orelse return null;
    if (pop_size < 64) return null;

    inline for ([_]usize{ 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 }) |target_l| {
        if (l == target_l) {
            return dispatchOptionsForL(
                target_l,
                allocator,
                pop_size,
                seed,
                eval_fn,
                user_data,
                options,
                chromosome_bits,
            ) catch return null;
        }
    }
    return null;
}

// C-ABI Exported API

pub export fn big_ll_create(
    chromosome_bits: usize,
    pop_size: usize,
    seed: u64,
    eval_fn: BigLlEvalFn,
    user_data: ?*anyopaque,
) ?*anyopaque {
    return big_ll_create_custom(chromosome_bits, pop_size, seed, eval_fn, user_data, .{});
}

pub export fn big_ll_create_custom(
    chromosome_bits: usize,
    pop_size: usize,
    seed: u64,
    eval_fn: BigLlEvalFn,
    user_data: ?*anyopaque,
    options: BigLlOptions,
) ?*anyopaque {
    const allocator = std.heap.page_allocator;
    const wrapper = createEngine(
        allocator,
        chromosome_bits,
        pop_size,
        seed,
        eval_fn,
        user_data,
        options,
    ) orelse return null;
    return wrapper;
}

pub export fn big_ll_step(handle: ?*anyopaque) c_int {
    const wrapper: *EngineWrapper = @ptrCast(@alignCast(handle orelse return -1));
    return if (wrapper.vtable.step(wrapper.ctx)) 0 else -1;
}

pub export fn big_ll_run(handle: ?*anyopaque, generations: usize) c_int {
    const wrapper: *EngineWrapper = @ptrCast(@alignCast(handle orelse return -1));
    return if (wrapper.vtable.run(wrapper.ctx, generations)) 0 else -1;
}

pub export fn big_ll_get_best_fitness(handle: ?*anyopaque) f64 {
    const wrapper: *EngineWrapper = @ptrCast(@alignCast(handle orelse return -std.math.inf(f64)));
    return wrapper.vtable.get_best_fitness(wrapper.ctx);
}

pub export fn big_ll_get_best_genes(handle: ?*anyopaque, out_buf: [*]u64, buf_words: usize) usize {
    const wrapper: *EngineWrapper = @ptrCast(@alignCast(handle orelse return 0));
    return wrapper.vtable.get_best_genes(wrapper.ctx, out_buf, buf_words);
}

pub export fn big_ll_get_best_genes_ptr(handle: ?*anyopaque) ?[*]const u64 {
    const wrapper: *EngineWrapper = @ptrCast(@alignCast(handle orelse return null));
    return wrapper.vtable.get_best_genes_ptr(wrapper.ctx);
}

pub export fn big_ll_get_generation(handle: ?*anyopaque) usize {
    const wrapper: *EngineWrapper = @ptrCast(@alignCast(handle orelse return 0));
    return wrapper.vtable.get_generation(wrapper.ctx);
}

pub export fn big_ll_get_pop_size(handle: ?*anyopaque) usize {
    const wrapper: *EngineWrapper = @ptrCast(@alignCast(handle orelse return 0));
    return wrapper.vtable.get_pop_size(wrapper.ctx);
}

pub export fn big_ll_get_chromosome_bits(handle: ?*anyopaque) usize {
    const wrapper: *EngineWrapper = @ptrCast(@alignCast(handle orelse return 0));
    return wrapper.requested_bits;
}

pub export fn big_ll_get_word_count(handle: ?*anyopaque) usize {
    const wrapper: *EngineWrapper = @ptrCast(@alignCast(handle orelse return 0));
    return wrapper.vtable.get_word_count(wrapper.ctx);
}

pub export fn big_ll_get_chromosome_words(handle: ?*anyopaque, index: usize) ?[*]const u64 {
    const wrapper: *EngineWrapper = @ptrCast(@alignCast(handle orelse return null));
    return wrapper.vtable.get_chromosome_words(wrapper.ctx, index);
}

pub export fn big_ll_get_chromosome_fitness(handle: ?*anyopaque, index: usize) f64 {
    const wrapper: *EngineWrapper = @ptrCast(@alignCast(handle orelse return -std.math.inf(f64)));
    return wrapper.vtable.get_chromosome_fitness(wrapper.ctx, index);
}

pub export fn big_ll_destroy(handle: ?*anyopaque) void {
    const wrapper: *EngineWrapper = @ptrCast(@alignCast(handle orelse return));
    wrapper.vtable.destroy(wrapper.ctx, wrapper.allocator);
    wrapper.allocator.destroy(wrapper);
}
