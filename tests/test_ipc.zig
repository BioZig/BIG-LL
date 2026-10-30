const std = @import("std");
const big_ll = @import("big_ll");
const SharedMemoryRegion = big_ll.SharedMemoryRegion;
const SharedPopulationHeader = big_ll.SharedPopulationHeader;
const types = big_ll.types;
const c_api = big_ll.c_api;

test "Shared memory layout invariants" {
    const allocator = std.testing.allocator;
    const pop_size = 64;
    const chrom_bits = 256;

    var shm = try SharedMemoryRegion.initAnonymous(allocator, pop_size, chrom_bits);
    defer shm.deinit();

    const hdr = shm.header();
    try std.testing.expectEqual(big_ll.shm.Magic, hdr.magic);
    try std.testing.expectEqual(big_ll.shm.CurrentVersion, hdr.version);
    try std.testing.expectEqual(@as(u32, pop_size), hdr.pop_size);
    try std.testing.expectEqual(@as(u32, chrom_bits), hdr.chromosome_bits);
    try std.testing.expectEqual(@as(u32, 4), hdr.word_count);

    // Verify 64-byte alignment
    try std.testing.expectEqual(@as(u64, 0), hdr.genes_offset % 64);
    try std.testing.expectEqual(@as(u64, 0), hdr.fitness_offset % 64);

    const genes = shm.getGenesSlice();
    try std.testing.expectEqual(pop_size * 4, genes.len);

    const fitness = shm.getFitnessSlice();
    try std.testing.expectEqual(pop_size, fitness.len);
}

test "Population synchronization with shared memory" {
    const allocator = std.testing.allocator;
    const L = 64;
    const pop_size = 32;

    var pop = try types.Population(L).init(allocator, pop_size);
    defer pop.deinit(allocator);

    for (pop.individuals, 0..) |*indiv, i| {
        indiv.words[0] = @as(u64, @intCast(i * 100 + 1));
        indiv.fitness = 0.0;
    }

    var shm = try SharedMemoryRegion.initAnonymous(allocator, pop_size, L);
    defer shm.deinit();

    shm.exportFromPopulation(L, &pop);

    for (0..pop_size) |i| {
        const genes = shm.getChromosomeGenesConst(i);
        try std.testing.expectEqual(@as(u64, @intCast(i * 100 + 1)), genes[0]);
        // Simulate external simulator scoring
        shm.setFitness(i, @as(f64, @floatFromInt(i)) * 2.5);
    }

    shm.importToPopulation(L, &pop);

    try std.testing.expectEqual(@as(f64, 31.0 * 2.5), pop.best_fitness);
    try std.testing.expectEqual(@as(usize, 31), pop.best_idx);
    try std.testing.expectEqual(@as(f64, 31.0 * 2.5), pop.individuals[31].fitness);
}

test "Simulation IPC evaluation signaling" {
    const allocator = std.testing.allocator;
    const pop_size = 16;
    const chrom_bits = 64;

    var shm = try SharedMemoryRegion.initAnonymous(allocator, pop_size, chrom_bits);
    defer shm.deinit();

    const Worker = struct {
        fn run(shm_ptr: *SharedMemoryRegion) void {
            var spins: usize = 0;
            while (spins < 1000000) : (spins += 1) {
                const st = @atomicLoad(u32, &shm_ptr.header().status, .acquire);
                if (st == @intFromEnum(big_ll.shm.StatusCode.eval_requested)) {
                    for (0..shm_ptr.header().pop_size) |i| {
                        const genes = shm_ptr.getChromosomeGenesConst(i);
                        const score = @as(f64, @floatFromInt(@popCount(genes[0])));
                        shm_ptr.setFitness(i, score);
                    }
                    shm_ptr.markEvaluationCompleted();
                    return;
                }
                std.Thread.yield() catch {};
            }
        }
    };

    // Populate genes
    for (0..pop_size) |i| {
        shm.getChromosomeGenes(i)[0] = @as(u64, @intCast(i));
    }

    const thread = try std.Thread.spawn(.{}, Worker.run, .{&shm});
    shm.requestEvaluation(1);

    const completed = shm.waitForEvaluation(1000000);
    thread.join();

    try std.testing.expect(completed);
    try std.testing.expectEqual(@as(f64, 0.0), shm.getFitness(0));
    try std.testing.expectEqual(@as(f64, @floatFromInt(@popCount(@as(u64, 15)))), shm.getFitness(15));
}

test "File-backed shared memory roundtrip" {
    const file_path = "/tmp/test_shm_roundtrip.bin";
    defer _ = std.c.unlink(file_path);

    const pop_size = 32;
    const chrom_bits = 128;

    {
        var shm_writer = try SharedMemoryRegion.createFileBacked(file_path, pop_size, chrom_bits);
        defer shm_writer.deinit();

        shm_writer.getChromosomeGenes(0)[0] = 0xDEADBEEFCAFEBABE;
        shm_writer.getChromosomeGenes(0)[1] = 0x0123456789ABCDEF;
        shm_writer.setFitness(0, 999.5);
    }

    {
        var shm_reader = try SharedMemoryRegion.openFileBacked(file_path);
        defer shm_reader.deinit();

        const genes = shm_reader.getChromosomeGenesConst(0);
        try std.testing.expectEqual(@as(u64, 0xDEADBEEFCAFEBABE), genes[0]);
        try std.testing.expectEqual(@as(u64, 0x0123456789ABCDEF), genes[1]);
        try std.testing.expectEqual(@as(f64, 999.5), shm_reader.getFitness(0));
    }
}

fn cTestEvalFn(genes: [*]const u64, word_count: usize, user_data: ?*anyopaque) callconv(.c) f64 {
    _ = user_data;
    var count: usize = 0;
    for (0..word_count) |i| {
        count += @popCount(genes[i]);
    }
    return @floatFromInt(count);
}

test "C-ABI lifecycle and optimization" {
    const handle = c_api.big_ll_create(64, 64, 42, cTestEvalFn, null);
    try std.testing.expect(handle != null);
    defer c_api.big_ll_destroy(handle);

    try std.testing.expectEqual(@as(usize, 64), c_api.big_ll_get_pop_size(handle));
    try std.testing.expectEqual(@as(usize, 64), c_api.big_ll_get_chromosome_bits(handle));
    try std.testing.expectEqual(@as(usize, 1), c_api.big_ll_get_word_count(handle));
    try std.testing.expectEqual(@as(usize, 0), c_api.big_ll_get_generation(handle));

    const initial_best = c_api.big_ll_get_best_fitness(handle);
    try std.testing.expect(initial_best >= 0.0);

    const step_res = c_api.big_ll_step(handle);
    try std.testing.expectEqual(@as(c_int, 0), step_res);
    try std.testing.expectEqual(@as(usize, 1), c_api.big_ll_get_generation(handle));

    const run_res = c_api.big_ll_run(handle, 20);
    try std.testing.expectEqual(@as(c_int, 0), run_res);
    try std.testing.expectEqual(@as(usize, 21), c_api.big_ll_get_generation(handle));

    const final_best = c_api.big_ll_get_best_fitness(handle);
    try std.testing.expect(final_best >= initial_best);

    var best_words = [_]u64{0};
    const words_copied = c_api.big_ll_get_best_genes(handle, &best_words, 1);
    try std.testing.expectEqual(@as(usize, 1), words_copied);
    try std.testing.expectEqual(final_best, @as(f64, @floatFromInt(@popCount(best_words[0]))));
}

test "C-ABI multithreaded engine execution" {
    const options = c_api.BigLlOptions{
        .enable_dedup = true,
        .num_threads = 2,
    };
    const handle = c_api.big_ll_create_custom(128, 64, 9999, cTestEvalFn, null, options);
    try std.testing.expect(handle != null);
    defer c_api.big_ll_destroy(handle);

    try std.testing.expectEqual(@as(usize, 64), c_api.big_ll_get_pop_size(handle));
    try std.testing.expectEqual(@as(usize, 2), c_api.big_ll_get_word_count(handle));

    const run_res = c_api.big_ll_run(handle, 15);
    try std.testing.expectEqual(@as(c_int, 0), run_res);
    try std.testing.expectEqual(@as(usize, 15), c_api.big_ll_get_generation(handle));
}
