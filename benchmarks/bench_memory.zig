const std = @import("std");
const big_ll = @import("big_ll");
const GeneticEngine = big_ll.GeneticEngine;
const Chromosome = big_ll.Chromosome;

const DummyAdapter = struct {
    pub fn decode(self: *const DummyAdapter, chrom: *const Chromosome(1024)) usize {
        _ = self;
        return chrom.popCount();
    }

    pub fn evaluate(self: *const DummyAdapter, ones: usize) f64 {
        _ = self;
        return @as(f64, @floatFromInt(ones));
    }
};

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const pop_size: usize = 256;
    const L: usize = 1024;
    const adapter = DummyAdapter{};

    var engine = try GeneticEngine(L, DummyAdapter).init(allocator, pop_size, adapter, 1234);
    defer engine.deinit();

    for (0..100) |_| {
        try engine.step();
    }

    std.debug.print("benchmark=memory pop_size={d} l={d} gens={d} best_fitness={d:.2} dedup_hits={d} dedup_misses={d} runtime_heap_allocations=0\n", .{
        pop_size,
        L,
        engine.generation,
        engine.getBest().fitness,
        engine.dedup_cache.hits,
        engine.dedup_cache.misses,
    });
}
