const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const core_mod = b.addModule("big_ll", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    // Apple Silicon Dedicated Metal GPU Batch Acceleration Configuration
    const is_darwin = target.result.os.tag.isDarwin();
    if (is_darwin) {
        core_mod.addIncludePath(b.path("src/metal"));
        core_mod.addCSourceFile(.{
            .file = b.path("src/metal/metal_bridge.m"),
            .flags = &.{"-fobjc-arc"},
        });
        core_mod.linkFramework("Metal", .{});
        core_mod.linkFramework("Foundation", .{});
    }

    const exe = b.addExecutable(.{
        .name = "big_ll",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "big_ll", .module = core_mod },
            },
        }),
    });
    b.installArtifact(exe);

    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "big_ll",
        .root_module = core_mod,
    });
    b.installArtifact(lib);
    b.installFile("include/big_ll.h", "include/big_ll.h");

    const run_step = b.step("run", "Run BIG-LL CLI");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const test_step = b.step("test", "Run all unit tests");

    const test_files = [_][]const u8{
        "tests/test_types.zig",
        "tests/test_prng.zig",
        "tests/test_linkage.zig",
        "tests/test_crossover.zig",
        "tests/test_dedup.zig",
        "tests/test_engine.zig",
        "tests/test_adapters.zig",
        "tests/test_domains.zig",
        "tests/test_falsification.zig",
        "tests/test_tiling.zig",
        "tests/test_island.zig",
        "tests/test_pool.zig",
        "tests/test_pareto.zig",
        "tests/test_ipc.zig",
        "tests/test_data_loader.zig",
        "tests/test_synthetic_domains.zig",
        "tests/test_data_importer.zig",
        "tests/test_dynamic_engine.zig",
        "tests/test_metal.zig",
        "tests/test_block_chromosome.zig",
        "tests/test_distributed.zig",
    };

    for (test_files) |path| {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(path),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
                .imports = &.{
                    .{ .name = "big_ll", .module = core_mod },
                },
            }),
        });
        const run_t = b.addRunArtifact(t);
        test_step.dependOn(&run_t.step);
    }

    const bench_core_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .link_libc = true,
    });

    // Apple Silicon Dedicated Metal GPU Batch Acceleration Configuration
    if (is_darwin) {
        bench_core_mod.addIncludePath(b.path("src/metal"));
        bench_core_mod.addCSourceFile(.{
            .file = b.path("src/metal/metal_bridge.m"),
            .flags = &.{"-fobjc-arc"},
        });
        bench_core_mod.linkFramework("Metal", .{});
        bench_core_mod.linkFramework("Foundation", .{});
    }

    const bench_exe = b.addExecutable(.{
        .name = "bench_throughput",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_throughput.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
            .imports = &.{
                .{ .name = "big_ll", .module = bench_core_mod },
            },
        }),
    });
    const run_bench = b.addRunArtifact(bench_exe);
    const bench_step = b.step("bench", "Run throughput benchmarks");
    bench_step.dependOn(&run_bench.step);

    const bench_mem_exe = b.addExecutable(.{
        .name = "bench_memory",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_memory.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
            .imports = &.{
                .{ .name = "big_ll", .module = bench_core_mod },
            },
        }),
    });
    const run_bench_mem = b.addRunArtifact(bench_mem_exe);
    const bench_mem_step = b.step("bench-mem", "Run memory footprint and zero-allocation benchmarks");
    bench_mem_step.dependOn(&run_bench_mem.step);

    const bench_domains_exe = b.addExecutable(.{
        .name = "bench_domains",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_domains.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
            .imports = &.{
                .{ .name = "big_ll", .module = bench_core_mod },
            },
        }),
    });
    const run_bench_domains = b.addRunArtifact(bench_domains_exe);
    const bench_domains_step = b.step("bench-domains", "Run synthetic domain benchmarks");
    bench_domains_step.dependOn(&run_bench_domains.step);

    const bench_real_exe = b.addExecutable(.{
        .name = "bench_real_domains",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_real_domains.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
            .imports = &.{
                .{ .name = "big_ll", .module = bench_core_mod },
            },
        }),
    });
    const run_bench_real = b.addRunArtifact(bench_real_exe);
    const bench_real_step = b.step("bench-real", "Run real-world industrial benchmarks");
    bench_real_step.dependOn(&run_bench_real.step);

    const bench_scaled_exe = b.addExecutable(.{
        .name = "bench_scaled",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/bench_scaled.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
            .imports = &.{
                .{ .name = "big_ll", .module = bench_core_mod },
            },
        }),
    });
    const run_bench_scaled = b.addRunArtifact(bench_scaled_exe);
    const bench_scaled_step = b.step("bench-scaled", "Run scaled multi-batch L2 cache tiling benchmarks (L=256..8192)");
    bench_scaled_step.dependOn(&run_bench_scaled.step);

    // Apple Silicon Dedicated Metal GPU Batch Acceleration Benchmark
    if (is_darwin) {
        const bench_metal_exe = b.addExecutable(.{
            .name = "bench_metal",
            .root_module = b.createModule(.{
                .root_source_file = b.path("benchmarks/bench_metal.zig"),
                .target = target,
                .optimize = .ReleaseFast,
                .imports = &.{
                    .{ .name = "big_ll", .module = bench_core_mod },
                },
            }),
        });
        const run_bench_metal = b.addRunArtifact(bench_metal_exe);
        const bench_metal_step = b.step("bench-metal", "Run Apple Silicon dedicated Metal GPU benchmarks");
        bench_metal_step.dependOn(&run_bench_metal.step);
    }
}
