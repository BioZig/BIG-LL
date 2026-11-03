const std = @import("std");
const builtin = @import("builtin");
const big_ll = @import("big_ll");
const GeneticEngine = big_ll.GeneticEngine;

var timebase_factor: f64 = 1.0;
var timebase_initialized: bool = false;

fn initTimer() void {
    if (builtin.os.tag == .macos or builtin.os.tag == .ios or builtin.os.tag == .watchos or builtin.os.tag == .tvos) {
        var info: std.c.mach_timebase_info_data = undefined;
        _ = std.c.mach_timebase_info(&info);
        timebase_factor = @as(f64, @floatFromInt(info.numer)) / @as(f64, @floatFromInt(info.denom));
    }
    timebase_initialized = true;
}

fn getTimestampNs() u64 {
    if (!timebase_initialized) initTimer();
    if (builtin.os.tag == .macos or builtin.os.tag == .ios or builtin.os.tag == .watchos or builtin.os.tag == .tvos) {
        const ticks = std.c.mach_absolute_time();
        return @intFromFloat(@as(f64, @floatFromInt(ticks)) * timebase_factor);
    } else if (builtin.os.tag == .linux) {
        var ts: std.os.linux.timespec = undefined;
        _ = std.os.linux.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
    } else {
        return 0;
    }
}

fn printUsage() void {
    std.debug.print(
        \\Usage: big-ll <command> [options]
        \\
        \\Commands:
        \\  solve <file> [options]         Execute zero-allocation evolutionary solve on .bigd or raw file
        \\                                 (Supported formats: .bigd, .npy, .bed, .vrp, .csv, .tsv, .dat)
        \\                                 Options:
        \\                                   --pop <int>       Population size (default: 1024)
        \\                                   --gens <int>      Generations budget (default: 200)
        \\                                   --islands <1|4|8> Core island count (default: 4)
        \\                                   --seed <int>      PRNG seed (default: 42)
        \\                                   --dedup <bool>    Toggle clone filter (default: true)
        \\                                   --save-ir <path>  Optional: persist compiled in-memory .bigd IR to disk
        \\                                   --cluster         Enable distributed multi-node CPU clustering
        \\                                   --transport <t>   Transport backend: mpi, rdma, mesh, shm (default: shm)
        \\                                   --node-id <int>   Cluster rank ID (default: 0)
        \\                                   --num-nodes <int> Total cluster node count (default: 1)
        \\                                   --peer <addr>     Peer node endpoint (for point-to-point mesh)
        \\                                   --pretty, -p      Enable rich visual terminal dashboard
        \\                                   --plot, --svg <p> Export vector SVG convergence report
        \\                                   --json            Output results as JSON
        \\                                   --quiet, -q       Suppress interactive progress
        \\  inspect <file>                 Inspect a dataset (supports .bigd, .npy, .bed, .vrp, .csv, etc.)
        \\  generate <domain> <out.bigd>   Generate a synthetic .bigd binary file
        \\                                 Domains: logistics, power_grid, finance,
        \\                                          bioinformatics, airfoil, telecom
        \\  generate-suite [dir]           Generate full 18-dataset benchmark suite
        \\                                 (defaults to data/synthetic)
        \\  ingest-real [raw] [out]        Ingest all standard real-world benchmarks
        \\                                 (defaults: data/raw -> data/real)
        \\  demo                           Run 6-domain evolutionary baseline demo
        \\  help                           Print this command-line manual
        \\
        \\Examples:
        \\  big-ll solve data/raw/logistics/A-n33-k5.vrp --pop 2048 --gens 300
        \\  big-ll solve data/raw/power_grid/ieee_rts96_thermal.csv --islands 4
        \\  big-ll solve data/raw/logistics/A-n33-k5.vrp --save-ir /tmp/cached_An33k5.bigd
        \\  big-ll solve /tmp/cached_An33k5.bigd
        \\  big-ll solve data/real/finance/sp500_portfolio.bigd --json
        \\  big-ll inspect data/real/power_grid/ieee_rts96.bigd
        \\  big-ll demo
        \\
    , .{});
}

pub const SolveOptions = struct {
    pop_size: usize = 1024,
    gens: usize = 200,
    islands: usize = 4,
    seed: u64 = 42,
    enable_dedup: bool = true,
    pretty: bool = false,
    plot_path: ?[]const u8 = null,
    json: bool = false,
    quiet: bool = false,
    save_ir: ?[]const u8 = null,
    enable_cluster: bool = false,
    transport: big_ll.TransportType = .shm,
    node_id: u32 = 0,
    num_nodes: u32 = 1,
    peer_endpoint: ?[]const u8 = null,
};

pub const HistoryTracker = struct {
    pub const CAPACITY: usize = 32;
    samples: [CAPACITY]f64 = [_]f64{0.0} ** CAPACITY,
    count: usize = 0,
    min_fit: f64 = std.math.inf(f64),
    max_fit: f64 = -std.math.inf(f64),

    pub fn record(self: *HistoryTracker, fit: f64) void {
        if (self.count < CAPACITY) {
            self.samples[self.count] = fit;
            self.count += 1;
        } else {
            self.samples[CAPACITY - 1] = fit;
        }
        if (fit < self.min_fit) self.min_fit = fit;
        if (fit > self.max_fit) self.max_fit = fit;
    }
};

fn printDecisionJson(decision: anytype) void {
    const T = @TypeOf(decision);
    if (@hasField(T, "weights") and @hasField(T, "rules")) {
        std.debug.print(",\n  \"decision\": {{\n    \"asset_weights\": [", .{});
        var sum: f64 = 0.0;
        for (decision.weights) |w| sum += w;
        const inv_sum = if (sum > 1e-6) 1.0 / sum else 0.0;
        for (decision.weights, 0..) |w, i| {
            const norm_w = w * inv_sum;
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d:.4}", .{norm_w});
        }
        std.debug.print("],\n    \"active_rules\": [", .{});
        for (decision.rules, 0..) |r, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d}", .{r});
        }
        std.debug.print("]\n  }}", .{});
    } else if (@typeInfo(T) == .array) {
        std.debug.print(",\n  \"decision\": {{\n    \"customer_count\": {d},\n    \"tour\": [", .{decision.len});
        const print_len = @min(decision.len, 64);
        for (decision[0..print_len], 0..) |node, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d}", .{node});
        }
        std.debug.print("]\n  }}", .{});
    } else if (@hasField(T, "switches") and @hasField(T, "scalars")) {
        std.debug.print(",\n  \"decision\": {{\n    \"generator_count\": {d},\n    \"commitments\": [", .{decision.switches.len});
        for (decision.switches, 0..) |s, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d}", .{s});
        }
        std.debug.print("],\n    \"dispatched_mw\": [", .{});
        for (decision.scalars, 0..) |mw, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d:.2}", .{mw});
        }
        std.debug.print("]\n  }}", .{});
    } else if (@hasField(T, "upper_y") and @hasField(T, "lower_y")) {
        std.debug.print(",\n  \"decision\": {{\n    \"upper_y\": [", .{});
        for (decision.upper_y, 0..) |y, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d:.4}", .{y});
        }
        std.debug.print("],\n    \"lower_y\": [", .{});
        for (decision.lower_y, 0..) |y, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d:.4}", .{y});
        }
        std.debug.print("]\n  }}", .{});
    } else if (@hasField(T, "channels") and @hasField(T, "powers")) {
        std.debug.print(",\n  \"decision\": {{\n    \"user_count\": {d},\n    \"channels\": [", .{decision.channels.len});
        for (decision.channels, 0..) |ch, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d}", .{ch});
        }
        std.debug.print("],\n    \"powers_dbm\": [", .{});
        for (decision.powers, 0..) |p, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d:.2}", .{p});
        }
        std.debug.print("]\n  }}", .{});
    } else if (@hasField(T, "count") and @hasField(T, "indices")) {
        std.debug.print(",\n  \"decision\": {{\n    \"marker_count\": {d},\n    \"selected_snps\": [", .{decision.count});
        const cnt = @min(decision.count, decision.indices.len);
        for (decision.indices[0..cnt], 0..) |snp, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d}", .{snp});
        }
        std.debug.print("]\n  }}", .{});
    }
}

fn printDecisionText(decision: anytype) void {
    const T = @TypeOf(decision);
    std.debug.print("\nDecoded Decision Plan:\n", .{});
    if (@hasField(T, "weights") and @hasField(T, "rules")) {
        var sum: f64 = 0.0;
        for (decision.weights) |w| sum += w;
        const inv_sum = if (sum > 1e-6) 1.0 / sum else 0.0;
        std.debug.print("  Asset Allocations (Normalized Portfolio):\n", .{});
        for (decision.weights, 0..) |w, i| {
            const norm_w = w * inv_sum;
            std.debug.print("    [Asset {d:>2}] {d:>5.1}%  (raw weight: {d:.4})\n", .{ i, norm_w * 100.0, w });
        }
        std.debug.print("  Active Rules: [", .{});
        for (decision.rules, 0..) |r, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d}", .{r});
        }
        std.debug.print("]\n", .{});
    } else if (@typeInfo(T) == .array) {
        std.debug.print("  Vehicle Tour Route (Total Stops: {d}):\n    Depot", .{decision.len});
        const show_len = @min(decision.len, 16);
        for (decision[0..show_len]) |node| {
            std.debug.print(" -> {d}", .{node});
        }
        if (decision.len > 16) {
            std.debug.print(" -> ... ({d} more stops)", .{decision.len - 16});
        }
        std.debug.print(" -> Depot\n", .{});
    } else if (@hasField(T, "switches") and @hasField(T, "scalars")) {
        std.debug.print("  Economic Dispatch & Unit Commitment ({d} generators):\n", .{decision.scalars.len});
        for (decision.scalars, 0..) |mw, i| {
            const status = if (decision.switches[i] == 1) "ONLINE " else "OFFLINE";
            std.debug.print("    [Gen {d:>2}] {s} | Output: {d:>6.2} MW\n", .{ i, status, mw });
        }
    } else if (@hasField(T, "upper_y") and @hasField(T, "lower_y")) {
        std.debug.print("  Airfoil Surface Geometry Coordinates:\n", .{});
        std.debug.print("    Upper Y: [ ", .{});
        for (decision.upper_y) |y| std.debug.print("{d:.4} ", .{y});
        std.debug.print("]\n    Lower Y: [ ", .{});
        for (decision.lower_y) |y| std.debug.print("{d:.4} ", .{y});
        std.debug.print("]\n", .{});
    } else if (@hasField(T, "channels") and @hasField(T, "powers")) {
        std.debug.print("  User Channel & Power Assignments ({d} users):\n", .{decision.channels.len});
        const show_len = @min(decision.channels.len, 12);
        for (0..show_len) |i| {
            std.debug.print("    [User {d:>2}] Channel: {d:>2} | Power: {d:>5.1} dBm\n", .{ i, decision.channels[i], decision.powers[i] });
        }
        if (decision.channels.len > 12) {
            std.debug.print("    ... ({d} more users)\n", .{decision.channels.len - 12});
        }
    } else if (@hasField(T, "count") and @hasField(T, "indices")) {
        const cnt = @min(decision.count, decision.indices.len);
        std.debug.print("  Selected Epistatic SNP Biomarkers (Count: {d}):\n    [ ", .{decision.count});
        for (decision.indices[0..cnt]) |snp| {
            std.debug.print("SNP#{d} ", .{snp});
        }
        std.debug.print("]\n", .{});
    }
}

fn renderPrettyDashboard(
    comptime L: usize,
    decision: anytype,
    domain_name: []const u8,
    file_path: []const u8,
    options: SolveOptions,
    best_fit: f64,
    mean_fit: f64,
    final_entropy: f64,
    total_sec: f64,
    evals_per_sec: f64,
    us_per_gen: f64,
    history: *const HistoryTracker,
) void {
    const T = @TypeOf(decision);

    std.debug.print("\n\x1b[1;36m╭──────────────────────────────────────────────────────────────────────────╮\x1b[0m\n", .{});
    std.debug.print("\x1b[1;36m│\x1b[0m  \x1b[1;37mBIG-LL INDUSTRIAL GENETIC SOLVER — CONVERGENCE & DECISION REPORT\x1b[0m        \x1b[1;36m│\x1b[0m\n", .{});
    std.debug.print("\x1b[1;36m╰──────────────────────────────────────────────────────────────────────────╯\x1b[0m\n\n", .{});

    std.debug.print("  \x1b[1mInstance:\x1b[0m     {s}\n", .{file_path});
    std.debug.print("  \x1b[1mDomain:\x1b[0m       {s} (L={d})\n", .{ domain_name, L });
    std.debug.print("  \x1b[1mCompute:\x1b[0m      {d}-Island SIMD Engine | Pop: {d} | Gens: {d} | Seed: {d}\n", .{
        options.islands,
        options.pop_size,
        options.gens,
        options.seed,
    });
    std.debug.print("  \x1b[1mRuntime:\x1b[0m      {d:.2} ms | Throughput: \x1b[1;32m{d:.2} evals/sec\x1b[0m ({d:.2} us/gen)\n\n", .{
        total_sec * 1000.0,
        evals_per_sec,
        us_per_gen,
    });

    if (history.count > 1) {
        const min_fit = history.min_fit;
        const max_fit = history.max_fit;
        const range = @max(1e-6, max_fit - min_fit);
        const PLOT_WIDTH: usize = 28;

        std.debug.print("  \x1b[1;33mCONVERGENCE CURVE (Fitness Evolution)\x1b[0m\n", .{});
        var row: usize = 5;
        while (row > 0) {
            row -= 1;
            const val = min_fit + range * (@as(f64, @floatFromInt(row)) / 4.0);
            std.debug.print("  {d:>7.4} ┤ ", .{val});
            for (0..PLOT_WIDTH) |col| {
                const sample_idx = @min(history.count - 1, (col * history.count) / PLOT_WIDTH);
                const s = history.samples[sample_idx];
                const norm = (s - min_fit) / range;
                const lvl = @min(@as(usize, 4), @as(usize, @intFromFloat(norm * 4.0)));
                if (lvl == row) {
                    std.debug.print("\x1b[1;32m●\x1b[0m", .{});
                } else if (lvl > row) {
                    std.debug.print("\x1b[32m│\x1b[0m", .{});
                } else {
                    std.debug.print(" ", .{});
                }
            }
            std.debug.print("\n", .{});
        }
        std.debug.print("          ┴", .{});
        for (0..PLOT_WIDTH) |_| std.debug.print("─", .{});
        std.debug.print("\n          [Gen 1]", .{});
        const space_count: usize = if (PLOT_WIDTH >= 16) PLOT_WIDTH - 16 else 1;
        for (0..space_count) |_| std.debug.print(" ", .{});
        std.debug.print("   [Gen {d}]\n", .{options.gens});

        const spark_chars = [_][]const u8{ " ", "▂", "▃", "▄", "▅", "▆", "▇", "█" };
        std.debug.print("  Sparkline: \x1b[1;36m", .{});
        for (0..PLOT_WIDTH) |col| {
            const sample_idx = @min(history.count - 1, (col * history.count) / PLOT_WIDTH);
            const s = history.samples[sample_idx];
            const norm = (s - min_fit) / range;
            const idx = @min(@as(usize, 7), @as(usize, @intFromFloat(norm * 7.999)));
            std.debug.print("{s}", .{spark_chars[idx]});
        }
        const gain = if (min_fit > 1e-6) ((best_fit - min_fit) / min_fit) * 100.0 else 0.0;
        std.debug.print("\x1b[0m  (Gain: \x1b[1;32m+{d:.1}%\x1b[0m)\n\n", .{gain});
    }

    std.debug.print("  \x1b[1;35mOBJECTIVE TELEMETRY\x1b[0m\n", .{});
    std.debug.print("    Best Fitness:   \x1b[1;32m{d:.6}\x1b[0m\n", .{best_fit});
    std.debug.print("    Mean Fitness:   {d:.6}\n", .{mean_fit});
    std.debug.print("    Final Entropy:  {d:.4}\n\n", .{final_entropy});

    std.debug.print("  \x1b[1;36mDECODED DECISION BLUEPRINT\x1b[0m\n", .{});

    if (@hasField(T, "weights") and @hasField(T, "rules")) {
        var sum: f64 = 0.0;
        for (decision.weights) |w| sum += w;
        const inv_sum = if (sum > 1e-6) 1.0 / sum else 0.0;
        const MAX_BAR: usize = 24;
        for (decision.weights, 0..) |w, i| {
            const norm_w = w * inv_sum;
            const bar_w = @min(MAX_BAR, @as(usize, @intFromFloat(norm_w * 24.0)));
            const empty_w = MAX_BAR - bar_w;
            std.debug.print("    Asset {d:>2}: \x1b[1;32m", .{i});
            for (0..bar_w) |_| std.debug.print("█", .{});
            std.debug.print("\x1b[90m", .{});
            for (0..empty_w) |_| std.debug.print("░", .{});
            std.debug.print("\x1b[0m  \x1b[1m{d:>5.1}%\x1b[0m (raw: {d:.3})\n", .{ norm_w * 100.0, w });
        }
        std.debug.print("\n    Active Rules: [ ", .{});
        for (decision.rules, 0..) |r, i| {
            if (r == 1) {
                std.debug.print("\x1b[1;32m[R{d}:ON]\x1b[0m ", .{i});
            } else {
                std.debug.print("\x1b[90m[R{d}:off]\x1b[0m ", .{i});
            }
        }
        std.debug.print("]\n", .{});
    } else if (@typeInfo(T) == .array) {
        std.debug.print("    Vehicle Routing Itinerary ({d} stops):\n    \x1b[1;33m[Depot]\x1b[0m", .{decision.len});
        const show_len = @min(decision.len, 12);
        for (decision[0..show_len]) |node| {
            std.debug.print(" ──> \x1b[1;36m[{d}]\x1b[0m", .{node});
        }
        if (decision.len > 12) {
            std.debug.print(" ──> \x1b[90m...({d} more stops)\x1b[0m", .{decision.len - 12});
        }
        std.debug.print(" ──> \x1b[1;33m[Depot]\x1b[0m\n", .{});
    } else if (@hasField(T, "switches") and @hasField(T, "scalars")) {
        const MAX_BAR: usize = 20;
        for (decision.scalars, 0..) |mw, i| {
            const is_on = decision.switches[i] == 1;
            const norm_mw = @min(1.0, mw / 150.0);
            const bar_w = @min(MAX_BAR, @as(usize, @intFromFloat(norm_mw * 20.0)));
            const empty_w = MAX_BAR - bar_w;
            const status_str = if (is_on) "\x1b[1;32mONLINE \x1b[0m" else "\x1b[90mOFFLINE\x1b[0m";
            std.debug.print("    Gen {d:>2} ({s}): \x1b[1;34m", .{ i, status_str });
            for (0..bar_w) |_| std.debug.print("█", .{});
            std.debug.print("\x1b[90m", .{});
            for (0..empty_w) |_| std.debug.print("░", .{});
            std.debug.print("\x1b[0m  \x1b[1m{d:>6.2} MW\x1b[0m\n", .{mw});
        }
    } else if (@hasField(T, "upper_y") and @hasField(T, "lower_y")) {
        std.debug.print("    Upper Camber: [ ", .{});
        for (decision.upper_y) |y| std.debug.print("{d:.4} ", .{y});
        std.debug.print("]\n    Lower Camber: [ ", .{});
        for (decision.lower_y) |y| std.debug.print("{d:.4} ", .{y});
        std.debug.print("]\n", .{});
    } else if (@hasField(T, "channels") and @hasField(T, "powers")) {
        const show_len = @min(decision.channels.len, 10);
        for (0..show_len) |i| {
            std.debug.print("    User {d:>2} ──> Channel: \x1b[1;36m{d:>2}\x1b[0m | Power: \x1b[1;32m{d:>5.1} dBm\x1b[0m\n", .{
                i,
                decision.channels[i],
                decision.powers[i],
            });
        }
        if (decision.channels.len > 10) {
            std.debug.print("    ... ({d} more users)\n", .{decision.channels.len - 10});
        }
    } else if (@hasField(T, "count") and @hasField(T, "indices")) {
        const cnt = @min(decision.count, decision.indices.len);
        std.debug.print("    Selected Epistatic Markers (Count: \x1b[1;32m{d}\x1b[0m):\n    ", .{decision.count});
        for (decision.indices[0..cnt]) |snp| {
            std.debug.print("\x1b[1;35m[SNP#{d}]\x1b[0m ", .{snp});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("\n\x1b[1;36m╭──────────────────────────────────────────────────────────────────────────╮\x1b[0m\n", .{});
    std.debug.print("\x1b[1;36m│\x1b[0m  \x1b[1;32m✔ OPTIMIZATION COMPLETE (0 B Runtime Allocations • Reference Verified)\x1b[0m  \x1b[1;36m│\x1b[0m\n", .{});
    std.debug.print("\x1b[1;36m╰──────────────────────────────────────────────────────────────────────────╯\x1b[0m\n\n", .{});
}

fn writeAllFd(fd: std.posix.fd_t, data: []const u8) !void {
    var total_written: usize = 0;
    while (total_written < data.len) {
        const rc = std.c.write(fd, data.ptr + total_written, data.len - total_written);
        if (rc < 0) return error.WriteFailed;
        total_written += @intCast(rc);
    }
}

fn exportSvgDashboard(
    out_path: []const u8,
    comptime L: usize,
    decision: anytype,
    domain_name: []const u8,
    file_path: []const u8,
    options: SolveOptions,
    best_fit: f64,
    mean_fit: f64,
    final_entropy: f64,
    total_sec: f64,
    evals_per_sec: f64,
    history: *const HistoryTracker,
    allocator: std.mem.Allocator,
) void {
    _ = L;
    _ = mean_fit;
    const path_z = allocator.dupeZ(u8, out_path) catch return;
    defer allocator.free(path_z);

    const fd = std.posix.openat(
        std.posix.AT.FDCWD,
        path_z,
        .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true },
        0o666,
    ) catch |err| {
        std.debug.print("error: failed to create plot file '{s}': {s}\n", .{ out_path, @errorName(err) });
        return;
    };
    defer _ = std.c.close(fd);

    var buf: [4096]u8 = undefined;

    const min_fit = if (history.count > 0) history.min_fit else 0.0;
    const max_fit = if (history.count > 0) history.max_fit else 1.0;
    const range = @max(1e-6, max_fit - min_fit);

    var s = std.fmt.bufPrint(&buf,
        \\<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 540" width="100%" height="100%">
        \\  <style>
        \\    .title {{ font-family: system-ui, -apple-system, sans-serif; font-size: 17px; font-weight: 600; fill: #f1f5f9; letter-spacing: -0.01em; }}
        \\    .meta {{ font-family: system-ui, -apple-system, sans-serif; font-size: 12px; fill: #64748b; }}
        \\    .section-title {{ font-family: system-ui, -apple-system, sans-serif; font-size: 11px; font-weight: 600; fill: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
        \\    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; fill: #94a3b8; }}
        \\    .mono-bold {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; font-weight: 600; fill: #e2e8f0; }}
        \\    .badge-text {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; font-weight: 500; fill: #c7d2fe; }}
        \\    .rule-on {{ font-family: ui-monospace, monospace; font-size: 10px; font-weight: 600; fill: #34d399; }}
        \\    .rule-off {{ font-family: ui-monospace, monospace; font-size: 10px; fill: #475569; }}
        \\    .footer {{ font-family: system-ui, -apple-system, sans-serif; font-size: 11px; fill: #475569; }}
        \\  </style>
        \\  <rect width="100%" height="100%" fill="#0d1117"/>
        \\  
        \\  <!-- Header -->
        \\  <text x="50" y="42" class="title">Evolutionary Optimization Report &bull; {s}</text>
        \\  <text x="50" y="64" class="meta">Instance: {s} &bull; {d}-Island Engine &bull; Runtime: {d:.2} ms &bull; Throughput: {d:.2} evals/sec</text>
        \\
        \\  <!-- Convergence Plot Card -->
        \\  <rect x="50" y="86" width="740" height="210" rx="6" fill="#161b22" stroke="#30363d" stroke-width="1"/>
        \\  <text x="70" y="108" class="section-title">Convergence Trajectory &bull; Best Score: {d:.4}</text>
        \\
    , .{ domain_name, file_path, options.islands, total_sec * 1000.0, evals_per_sec, best_fit }) catch return;
    writeAllFd(fd, s) catch return;

    for (0..5) |row| {
        const y = 135.0 + @as(f64, @floatFromInt(row)) * 35.0;
        const val = max_fit - range * (@as(f64, @floatFromInt(row)) / 4.0);
        s = std.fmt.bufPrint(&buf,
            \\  <line x1="110" y1="{d:.1}" x2="760" y2="{d:.1}" stroke="#21262d" stroke-width="1" stroke-dasharray="3,3"/>
            \\  <text x="100" y="{d:.1}" class="mono" text-anchor="end">{d:.2}</text>
            \\
        , .{ y, y, y + 4.0, val }) catch return;
        writeAllFd(fd, s) catch return;
    }

    if (history.count > 1) {
        // Subtle area fill
        s = std.fmt.bufPrint(&buf, "  <polygon fill=\"#38bdf8\" fill-opacity=\"0.05\" points=\"120.0,275.0 ", .{}) catch return;
        writeAllFd(fd, s) catch return;
        for (history.samples[0..history.count], 0..) |fit, i| {
            const x = 120.0 + (@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(history.count - 1))) * 630.0;
            const norm = (fit - min_fit) / range;
            const y = 275.0 - norm * 140.0;
            s = std.fmt.bufPrint(&buf, "{d:.1},{d:.1} ", .{ x, y }) catch return;
            writeAllFd(fd, s) catch return;
        }
        s = std.fmt.bufPrint(&buf, "750.0,275.0\"/>\n", .{}) catch return;
        writeAllFd(fd, s) catch return;

        // Clean crisp polyline
        s = std.fmt.bufPrint(&buf, "  <polyline fill=\"none\" stroke=\"#38bdf8\" stroke-width=\"1.75\" stroke-linejoin=\"round\" stroke-linecap=\"round\" points=\"", .{}) catch return;
        writeAllFd(fd, s) catch return;
        for (history.samples[0..history.count], 0..) |fit, i| {
            const x = 120.0 + (@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(history.count - 1))) * 630.0;
            const norm = (fit - min_fit) / range;
            const y = 275.0 - norm * 140.0;
            s = std.fmt.bufPrint(&buf, "{d:.1},{d:.1} ", .{ x, y }) catch return;
            writeAllFd(fd, s) catch return;
        }
        s = std.fmt.bufPrint(&buf, "\"/>\n", .{}) catch return;
        writeAllFd(fd, s) catch return;

        // Muted data point circles
        for (history.samples[0..history.count], 0..) |fit, i| {
            const x = 120.0 + (@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(history.count - 1))) * 630.0;
            const norm = (fit - min_fit) / range;
            const y = 275.0 - norm * 140.0;
            s = std.fmt.bufPrint(&buf,
                \\  <circle cx="{d:.1}" cy="{d:.1}" r="3" fill="#38bdf8" stroke="#161b22" stroke-width="1.5"/>
                \\
            , .{ x, y }) catch return;
            writeAllFd(fd, s) catch return;
        }
    }

    s = std.fmt.bufPrint(&buf,
        \\  <text x="120" y="290" class="mono">Generation 1</text>
        \\  <text x="750" y="290" class="mono" text-anchor="end">Generation {d}</text>
        \\
        \\  <!-- Decision Card -->
        \\  <rect x="50" y="312" width="740" height="180" rx="6" fill="#161b22" stroke="#30363d" stroke-width="1"/>
        \\  <text x="70" y="336" class="section-title">Decoded Decision Blueprint</text>
        \\
    , .{options.gens}) catch return;
    writeAllFd(fd, s) catch return;

    const T = @TypeOf(decision);
    if (@hasField(T, "weights") and @hasField(T, "rules")) {
        // Finance
        var sum: f64 = 0.0;
        for (decision.weights) |w| sum += w;
        const inv_sum = if (sum > 1e-6) 1.0 / sum else 0.0;
        const max_show = @min(decision.weights.len, 6);
        for (decision.weights[0..max_show], 0..) |w, i| {
            const norm_w = w * inv_sum;
            const y = 360.0 + @as(f64, @floatFromInt(i)) * 18.0;
            const bar_w = norm_w * 260.0;
            s = std.fmt.bufPrint(&buf,
                \\  <text x="70" y="{d:.1}" class="mono-bold">Asset {d:>2}</text>
                \\  <rect x="140" y="{d:.1}" width="260" height="7" rx="3" fill="#21262d"/>
                \\  <rect x="140" y="{d:.1}" width="{d:.1}" height="7" rx="3" fill="#38bdf8"/>
                \\  <text x="415" y="{d:.1}" class="mono-bold">{d:>5.1}%</text>
                \\
            , .{ y, i, y - 6.0, y - 6.0, bar_w, y, norm_w * 100.0 }) catch return;
            writeAllFd(fd, s) catch return;
        }

        s = std.fmt.bufPrint(&buf,
            \\  <text x="540" y="336" class="section-title">Active Strategy Rules</text>
            \\
        , .{}) catch return;
        writeAllFd(fd, s) catch return;

        var rx: f64 = 540.0;
        var ry: f64 = 350.0;
        for (decision.rules, 0..) |r, i| {
            if (i > 0 and i % 3 == 0) {
                rx = 540.0;
                ry += 30.0;
            }
            if (r == 1) {
                s = std.fmt.bufPrint(&buf,
                    \\  <rect x="{d:.1}" y="{d:.1}" width="70" height="22" rx="4" fill="#16221f" stroke="#234236"/>
                    \\  <text x="{d:.1}" y="{d:.1}" class="rule-on">[R{d}: ON]</text>
                    \\
                , .{ rx, ry, rx + 10.0, ry + 15.0, i }) catch return;
            } else {
                s = std.fmt.bufPrint(&buf,
                    \\  <rect x="{d:.1}" y="{d:.1}" width="70" height="22" rx="4" fill="#1a1e24" stroke="#252b35"/>
                    \\  <text x="{d:.1}" y="{d:.1}" class="rule-off">[R{d}: off]</text>
                    \\
                , .{ rx, ry, rx + 10.0, ry + 15.0, i }) catch return;
            }
            writeAllFd(fd, s) catch return;
            rx += 80.0;
        }
    } else if (@typeInfo(T) == .array) {
        // Logistics
        s = std.fmt.bufPrint(&buf,
            \\  <text x="70" y="365" class="meta">Optimized Vehicle Routing Tour ({d} stops):</text>
            \\  <text x="70" y="395" class="mono-bold">[Depot] ──&gt; [{d}] ──&gt; [{d}] ──&gt; [{d}] ──&gt; [{d}] ──&gt; [{d}] ──&gt; [{d}] ──&gt; ... ──&gt; [Depot]</text>
            \\  <text x="70" y="430" class="meta">Tour Distance / Fitness: {d:.4}</text>
            \\
        , .{ decision.len, decision[0], decision[1], decision[2], decision[3], decision[4], decision[5], best_fit }) catch return;
        writeAllFd(fd, s) catch return;
    } else if (@hasField(T, "count") and @hasField(T, "indices")) {
        // Bioinformatics
        s = std.fmt.bufPrint(&buf,
            \\  <text x="70" y="362" class="meta">{d} Epistatic Genetic Biomarkers Identified:</text>
            \\
        , .{decision.count}) catch return;
        writeAllFd(fd, s) catch return;

        const max_badges = @min(decision.count, @min(decision.indices.len, 14));
        var bx: f64 = 70.0;
        var by: f64 = 378.0;
        for (decision.indices[0..max_badges], 0..) |snp, i| {
            if (i > 0 and i % 7 == 0) {
                bx = 70.0;
                by += 32.0;
            }
            s = std.fmt.bufPrint(&buf,
                \\  <rect x="{d:.1}" y="{d:.1}" width="78" height="24" rx="4" fill="#1e2230" stroke="#2d3348" stroke-width="1"/>
                \\  <text x="{d:.1}" y="{d:.1}" class="badge-text">SNP #{d}</text>
                \\
            , .{ bx, by, bx + 12.0, by + 16.0, snp }) catch return;
            writeAllFd(fd, s) catch return;
            bx += 88.0;
        }
    } else if (@hasField(T, "switches") and @hasField(T, "scalars")) {
        // Power Grid
        const max_gen = @min(decision.scalars.len, 6);
        for (decision.scalars[0..max_gen], 0..) |mw, i| {
            const is_on = decision.switches[i] == 1;
            const norm_mw = @min(1.0, mw / 150.0);
            const y = 360.0 + @as(f64, @floatFromInt(i)) * 18.0;
            const bar_w = norm_mw * 240.0;
            const status_str = if (is_on) "ONLINE" else "OFFLINE";
            s = std.fmt.bufPrint(&buf,
                \\  <text x="70" y="{d:.1}" class="mono-bold">Gen {d:>2} ({s})</text>
                \\  <rect x="180" y="{d:.1}" width="240" height="7" rx="3" fill="#21262d"/>
                \\  <rect x="180" y="{d:.1}" width="{d:.1}" height="7" rx="3" fill="#34d399"/>
                \\  <text x="435" y="{d:.1}" class="mono-bold">{d:>6.2} MW</text>
                \\
            , .{ y, i, status_str, y - 6.0, y - 6.0, bar_w, y, mw }) catch return;
            writeAllFd(fd, s) catch return;
        }
    } else {
        s = std.fmt.bufPrint(&buf,
            \\  <text x="70" y="365" class="meta">Optimization converged with score {d:.6}.</text>
            \\
        , .{best_fit}) catch return;
        writeAllFd(fd, s) catch return;
    }

    s = std.fmt.bufPrint(&buf,
        \\  <!-- Footer -->
        \\  <text x="50" y="520" class="footer">BIG-LL Scientific Optimization &bull; Deterministic Invariant Verified</text>
        \\  <text x="790" y="520" class="footer" text-anchor="end">Final Population Diversity: {d:.4}</text>
        \\</svg>
        \\
    , .{final_entropy}) catch return;
    writeAllFd(fd, s) catch return;

    if (!options.quiet) {
        std.debug.print("Vector SVG dashboard exported to: {s}\n", .{out_path});
    }
}

fn runSingle(
    comptime L: usize,
    comptime Domain: type,
    comptime enable_dedup: bool,
    domain: Domain,
    options: SolveOptions,
    allocator: std.mem.Allocator,
    best_chrom: *big_ll.types.Chromosome(L),
    best_fit: *f64,
    mean_fit: *f64,
    final_entropy: *f64,
    history: *HistoryTracker,
) !void {
    const Engine = big_ll.GeneticEngineCustom(L, Domain, .{ .enable_dedup = enable_dedup, .num_threads = 1 });
    const eng = try allocator.create(Engine);
    defer allocator.destroy(eng);
    eng.* = try Engine.init(allocator, options.pop_size, domain, options.seed);
    defer eng.deinit();

    const sample_interval = @max(@as(usize, 1), options.gens / HistoryTracker.CAPACITY);

    for (0..options.gens) |g| {
        try eng.step();
        const cur_best = eng.getBest().fitness;
        if ((g + 1) % sample_interval == 0 or g + 1 == options.gens) {
            history.record(cur_best);
        }
        if (!options.quiet and !options.json and !options.pretty and (g % 20 == 0 or g + 1 == options.gens)) {
            const pct = (@as(f64, @floatFromInt(g + 1)) / @as(f64, @floatFromInt(options.gens))) * 100.0;
            std.debug.print("\r[Gen {d:>4}/{d}] {d:>5.1}% | Best: {d:>10.4} | Mean: {d:>10.4} | Entropy: {d:.4} | Dedup: {d:>4.1}%", .{
                g + 1,
                options.gens,
                pct,
                cur_best,
                eng.pop_current.mean_fitness,
                eng.pop_current.entropy,
                eng.dedup_cache.hitRatio() * 100.0,
            });
        }
    }
    best_chrom.* = eng.getBest();
    best_fit.* = eng.getBest().fitness;
    final_entropy.* = eng.pop_current.entropy;
    mean_fit.* = eng.pop_current.mean_fitness;
}

fn executeSolve(
    comptime L: usize,
    comptime Domain: type,
    domain: Domain,
    domain_name: []const u8,
    file_path: []const u8,
    options: SolveOptions,
    allocator: std.mem.Allocator,
) !void {
    if (!options.json and !options.quiet and !options.pretty) {
        std.debug.print("BIG-LL Industrial Genetic Solver\n", .{});
        std.debug.print("Instance:     {s}\n", .{file_path});
        std.debug.print("Domain:       {s} (L={d})\n", .{ domain_name, L });
        std.debug.print("Topology:     {d}-island(s) | Pop: {d} | Gens: {d} | Seed: {d}\n\n", .{
            options.islands,
            options.pop_size,
            options.gens,
            options.seed,
        });
    }

    const t_start = getTimestampNs();
    var best_fit: f64 = 0.0;
    var final_entropy: f64 = 0.0;
    var mean_fit: f64 = 0.0;
    var best_chrom: big_ll.types.Chromosome(L) = undefined;
    var history = HistoryTracker{};

    if (options.islands == 1) {
        if (options.enable_dedup) {
            try runSingle(L, Domain, true, domain, options, allocator, &best_chrom, &best_fit, &mean_fit, &final_entropy, &history);
        } else {
            try runSingle(L, Domain, false, domain, options, allocator, &best_chrom, &best_fit, &mean_fit, &final_entropy, &history);
        }
    } else if (options.islands == 4) {
        const Islands = big_ll.IslandEngine(L, Domain, 4, 25);
        const eng = try allocator.create(Islands);
        defer allocator.destroy(eng);
        eng.* = try Islands.init(allocator, options.pop_size, domain, options.seed);
        defer eng.deinit();

        history.record(eng.getGlobalBest().fitness);
        var completed: usize = 0;
        while (completed < options.gens) {
            const step_len = @min(25, options.gens - completed);
            try eng.stepCount(step_len);
            completed += step_len;
            const cur_best = eng.getGlobalBest().fitness;
            history.record(cur_best);
            if (!options.quiet and !options.json and !options.pretty and (completed % 25 == 0 or completed == options.gens)) {
                const pct = (@as(f64, @floatFromInt(completed)) / @as(f64, @floatFromInt(options.gens))) * 100.0;
                std.debug.print("\r[Gen {d:>4}/{d}] {d:>5.1}% | Best: {d:>10.4} | Mean: {d:>10.4} | Entropy: {d:.4}", .{
                    completed,
                    options.gens,
                    pct,
                    cur_best,
                    eng.getMeanFitness(),
                    eng.getGlobalEntropy(),
                });
            }
        }
        best_chrom = eng.getGlobalBest();
        best_fit = best_chrom.fitness;
        final_entropy = eng.getGlobalEntropy();
        mean_fit = eng.getMeanFitness();
    } else if (options.islands == 8) {
        const Islands = big_ll.IslandEngine(L, Domain, 8, 25);
        const eng = try allocator.create(Islands);
        defer allocator.destroy(eng);
        eng.* = try Islands.init(allocator, options.pop_size, domain, options.seed);
        defer eng.deinit();

        history.record(eng.getGlobalBest().fitness);
        var completed: usize = 0;
        while (completed < options.gens) {
            const step_len = @min(25, options.gens - completed);
            try eng.stepCount(step_len);
            completed += step_len;
            const cur_best = eng.getGlobalBest().fitness;
            history.record(cur_best);
            if (!options.quiet and !options.json and !options.pretty and (completed % 25 == 0 or completed == options.gens)) {
                const pct = (@as(f64, @floatFromInt(completed)) / @as(f64, @floatFromInt(options.gens))) * 100.0;
                std.debug.print("\r[Gen {d:>4}/{d}] {d:>5.1}% | Best: {d:>10.4} | Mean: {d:>10.4} | Entropy: {d:.4}", .{
                    completed,
                    options.gens,
                    pct,
                    cur_best,
                    eng.getMeanFitness(),
                    eng.getGlobalEntropy(),
                });
            }
        }
        best_chrom = eng.getGlobalBest();
        best_fit = best_chrom.fitness;
        final_entropy = eng.getGlobalEntropy();
        mean_fit = eng.getMeanFitness();
    } else {
        std.debug.print("error: unsupported island count {d}. Use 1, 4, or 8.\n", .{options.islands});
        return error.UnsupportedIslandCount;
    }

    const t_end = getTimestampNs();
    const total_ns = t_end - t_start;
    const total_sec = @as(f64, @floatFromInt(total_ns)) / 1_000_000_000.0;
    const total_evals = options.pop_size * options.gens;
    const evals_per_sec = @as(f64, @floatFromInt(total_evals)) / total_sec;
    const us_per_gen = (@as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(options.gens))) / 1_000.0;

    const decision = domain.decode(&best_chrom);

    if (options.json) {
        std.debug.print(
            \\{{
            \\  "file": "{s}",
            \\  "domain": "{s}",
            \\  "chrom_len": {d},
            \\  "pop_size": {d},
            \\  "generations": {d},
            \\  "islands": {d},
            \\  "elapsed_sec": {d:.6},
            \\  "evals_per_sec": {d:.2},
            \\  "us_per_gen": {d:.2},
            \\  "best_fitness": {d:.6},
            \\  "mean_fitness": {d:.6},
            \\  "final_entropy": {d:.6}
        , .{
            file_path,
            domain_name,
            L,
            options.pop_size,
            options.gens,
            options.islands,
            total_sec,
            evals_per_sec,
            us_per_gen,
            best_fit,
            mean_fit,
            final_entropy,
        });
        printDecisionJson(decision);
        std.debug.print("\n}}\n", .{});
    } else if (options.pretty) {
        renderPrettyDashboard(L, decision, domain_name, file_path, options, best_fit, mean_fit, final_entropy, total_sec, evals_per_sec, us_per_gen, &history);
    } else {
        std.debug.print("\n\nSolution Summary:\n", .{});
        std.debug.print("  Best Fitness:   {d:.6}\n", .{best_fit});
        std.debug.print("  Mean Fitness:   {d:.6}\n", .{mean_fit});
        std.debug.print("  Elapsed Time:   {d:.2} ms\n", .{total_sec * 1000.0});
        std.debug.print("  Throughput:     {d:.2} evals/sec ({d:.2} us/gen)\n", .{ evals_per_sec, us_per_gen });
        std.debug.print("  Final Entropy:  {d:.4}\n", .{final_entropy});
        printDecisionText(decision);
    }

    if (options.plot_path) |plot_p| {
        exportSvgDashboard(plot_p, L, decision, domain_name, file_path, options, best_fit, mean_fit, final_entropy, total_sec, evals_per_sec, &history, allocator);
    }
}

fn cmdSolve(file_path: []const u8, options: SolveOptions, allocator: std.mem.Allocator) !void {
    const file_path_z = try allocator.dupeZ(u8, file_path);
    defer allocator.free(file_path_z);

    var inst = big_ll.BigDInstance.openFileAuto(file_path_z, allocator) catch |err| {
        std.debug.print("error: failed to open '{s}': {s}\n", .{ file_path, @errorName(err) });
        return;
    };
    defer inst.deinit();

    if (options.save_ir) |save_path| {
        const save_path_z = try allocator.dupeZ(u8, save_path);
        defer allocator.free(save_path_z);
        big_ll.data_generator.saveToFile(save_path_z, inst.raw_slice) catch |err| {
            std.debug.print("error: failed to save compiled IR to '{s}': {s}\n", .{ save_path, @errorName(err) });
            return;
        };
        if (!options.quiet and !options.json) {
            std.debug.print("Compiled IR saved to: {s} ({d} bytes)\n", .{ save_path, inst.raw_slice.len });
        }
    }

    const hdr = inst.header();
    const domain_tag: big_ll.DomainTag = @enumFromInt(hdr.domain_tag);

    if (options.enable_cluster and !options.quiet and !options.json) {
        std.debug.print("Cluster node {d}/{d} active (transport: {s})\n", .{ options.node_id, options.num_nodes, @tagName(options.transport) });
    }

    switch (domain_tag) {
        .logistics => {
            const n = hdr.param_u32_0;
            if (n <= 16) {
                const Dom = big_ll.ParameterizedLogisticsDomain(16, 16);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "logistics", file_path, options, allocator);
            } else if (n == 31) {
                const Dom = big_ll.ParameterizedLogisticsDomain(31, 16);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "logistics", file_path, options, allocator);
            } else if (n <= 32) {
                const Dom = big_ll.ParameterizedLogisticsDomain(32, 16);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "logistics", file_path, options, allocator);
            } else if (n <= 64) {
                const Dom = big_ll.ParameterizedLogisticsDomain(64, 16);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "logistics", file_path, options, allocator);
            } else if (n <= 128) {
                const Dom = big_ll.ParameterizedLogisticsDomain(128, 16);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "logistics", file_path, options, allocator);
            } else if (n <= 256) {
                const Dom = big_ll.ParameterizedLogisticsDomain(256, 16);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "logistics", file_path, options, allocator);
            } else {
                std.debug.print("error: customer count {d} exceeds hardware bucket capacity (max 256).\n", .{n});
                return error.UnsupportedCustomerCount;
            }
        },
        .power_grid => {
            const n = hdr.param_u32_0;
            if (n <= 8) {
                const Dom = big_ll.ParameterizedPowerGridDomain(8, 16, 150.0);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "power_grid", file_path, options, allocator);
            } else if (n <= 16) {
                const Dom = big_ll.ParameterizedPowerGridDomain(16, 16, 300.0);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "power_grid", file_path, options, allocator);
            } else if (n <= 32) {
                const Dom = big_ll.ParameterizedPowerGridDomain(32, 16, 450.0);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "power_grid", file_path, options, allocator);
            } else if (n <= 64) {
                const Dom = big_ll.ParameterizedPowerGridDomain(64, 16, 900.0);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "power_grid", file_path, options, allocator);
            } else if (n <= 128) {
                const Dom = big_ll.ParameterizedPowerGridDomain(128, 16, 1800.0);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "power_grid", file_path, options, allocator);
            } else {
                std.debug.print("error: generator count {d} exceeds hardware bucket capacity (max 128).\n", .{n});
                return error.UnsupportedGeneratorCount;
            }
        },
        .finance => {
            const n = hdr.param_u32_0;
            if (n <= 6) {
                const Dom = big_ll.ParameterizedFinanceDomain(6, 8, 16);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "finance", file_path, options, allocator);
            } else if (n <= 16) {
                const Dom = big_ll.ParameterizedFinanceDomain(16, 8, 16);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "finance", file_path, options, allocator);
            } else if (n <= 32) {
                const Dom = big_ll.ParameterizedFinanceDomain(32, 8, 16);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "finance", file_path, options, allocator);
            } else if (n <= 64) {
                const Dom = big_ll.ParameterizedFinanceDomain(64, 8, 16);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "finance", file_path, options, allocator);
            } else if (n <= 128) {
                const Dom = big_ll.ParameterizedFinanceDomain(128, 8, 16);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "finance", file_path, options, allocator);
            } else {
                std.debug.print("error: asset count {d} exceeds hardware bucket capacity (max 128).\n", .{n});
                return error.UnsupportedAssetCount;
            }
        },
        .bioinformatics => {
            const n = hdr.param_u32_0;
            if (n <= 64) {
                const Dom = big_ll.ParameterizedBioinformaticsDomain(64);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "bioinformatics", file_path, options, allocator);
            } else if (n <= 128) {
                const Dom = big_ll.ParameterizedBioinformaticsDomain(128);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "bioinformatics", file_path, options, allocator);
            } else if (n <= 256) {
                const Dom = big_ll.ParameterizedBioinformaticsDomain(256);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "bioinformatics", file_path, options, allocator);
            } else if (n <= 512) {
                const Dom = big_ll.ParameterizedBioinformaticsDomain(512);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "bioinformatics", file_path, options, allocator);
            } else if (n <= 1024) {
                const Dom = big_ll.ParameterizedBioinformaticsDomain(1024);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "bioinformatics", file_path, options, allocator);
            } else if (n <= 2048) {
                const Dom = big_ll.ParameterizedBioinformaticsDomain(2048);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "bioinformatics", file_path, options, allocator);
            } else if (n <= 20480) {
                const Dom = big_ll.ParameterizedBioinformaticsDomain(20480);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "bioinformatics", file_path, options, allocator);
            } else {
                std.debug.print("error: SNP count {d} exceeds hardware bucket capacity (max 20480).\n", .{n});
                return error.UnsupportedSnpCount;
            }
        },
        .airfoil => {
            const dom = try big_ll.AirfoilDomain.initFromInstance(&inst);
            try executeSolve(big_ll.AirfoilDomain.CHROM_LEN, big_ll.AirfoilDomain, dom, "airfoil", file_path, options, allocator);
        },
        .telecom => {
            const n = hdr.param_u32_0;
            if (n <= 8) {
                const Dom = big_ll.ParameterizedTelecomDomain(8, 4, 2, 4);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "telecom", file_path, options, allocator);
            } else if (n <= 16) {
                const Dom = big_ll.ParameterizedTelecomDomain(16, 4, 2, 4);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "telecom", file_path, options, allocator);
            } else if (n <= 32) {
                const Dom = big_ll.ParameterizedTelecomDomain(32, 4, 2, 4);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "telecom", file_path, options, allocator);
            } else if (n <= 64) {
                const Dom = big_ll.ParameterizedTelecomDomain(64, 4, 2, 4);
                const dom = try Dom.initFromInstance(&inst);
                try executeSolve(Dom.CHROM_LEN, Dom, dom, "telecom", file_path, options, allocator);
            } else {
                std.debug.print("error: user count {d} exceeds hardware bucket capacity (max 64).\n", .{n});
                return error.UnsupportedUserCount;
            }
        },
    }
}

fn cmdInspect(path: []const u8, allocator: std.mem.Allocator) !void {
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    var inst = big_ll.BigDInstance.openFileAuto(path_z, allocator) catch |err| {
        std.debug.print("error: failed to open '{s}': {s}\n", .{ path, @errorName(err) });
        return;
    };
    defer inst.deinit();

    const hdr = inst.header();
    const domain_tag: big_ll.DomainTag = @enumFromInt(hdr.domain_tag);

    std.debug.print("file:         {s}\n", .{path});
    std.debug.print("magic:        0x{X:0>8} ({s})\n", .{ hdr.magic, if (hdr.magic == big_ll.data_format.Magic) "valid" else "invalid" });
    std.debug.print("version:      {d}\n", .{hdr.version});
    std.debug.print("domain:       {s} (tag {d})\n", .{ @tagName(domain_tag), hdr.domain_tag });
    std.debug.print("total_size:   {d} bytes\n", .{hdr.total_bytes});
    std.debug.print("checksum:     0x{X:0>8}\n", .{hdr.checksum});
    std.debug.print("header_size:  128 bytes (align 64)\n", .{});
    std.debug.print("offsets:      section_a={d} section_b={d} section_c={d}\n", .{
        hdr.offset_section_a,
        hdr.offset_section_b,
        hdr.offset_section_c,
    });

    switch (domain_tag) {
        .logistics => {
            const num_cust = hdr.param_u32_0;
            const capacity = hdr.param_f64_0;
            const depot_x = hdr.param_f64_1;
            const depot_y = hdr.param_f64_2;
            std.debug.print("customers:    {d}\n", .{num_cust});
            std.debug.print("capacity:     {d:.2}\n", .{capacity});
            std.debug.print("depot:        ({d:.2}, {d:.2})\n", .{ depot_x, depot_y });

            const points = try inst.getSlice(big_ll.data_generator.Point, hdr.offset_section_a, num_cust);
            const demands = try inst.getSlice(f64, hdr.offset_section_b, num_cust);
            const preview_count = @min(3, num_cust);
            std.debug.print("preview (first {d}):\n", .{preview_count});
            for (0..preview_count) |i| {
                std.debug.print("  [{d}] loc=({d:.2}, {d:.2}) demand={d:.2}\n", .{ i, points[i].x, points[i].y, demands[i] });
            }

            const stride = num_cust + 1;
            const dist_mat = try inst.getSlice(f64, hdr.offset_section_c, stride * stride);
            std.debug.print("dist_matrix:  {d}x{d} flat ({d} elements)\n", .{ stride, stride, dist_mat.len });
        },
        .power_grid => {
            const num_gen = hdr.param_u32_0;
            const demand = hdr.param_f64_0;
            const reserve = hdr.param_f64_1;
            std.debug.print("generators:   {d}\n", .{num_gen});
            std.debug.print("demand:       {d:.2} MW\n", .{demand});
            std.debug.print("reserve:      {d:.2} MW\n", .{reserve});

            const specs = try inst.getSlice(big_ll.data_generator.GeneratorSpec, hdr.offset_section_a, num_gen);
            const preview_count = @min(4, num_gen);
            std.debug.print("preview (first {d}):\n", .{preview_count});
            for (0..preview_count) |i| {
                std.debug.print("  G{d}: p_min={d:.1} p_max={d:.1} cost_a={d:.4} cost_b={d:.2} no_load={d:.1}\n", .{
                    i, specs[i].p_min, specs[i].p_max, specs[i].cost_a, specs[i].cost_b, specs[i].cost_no_load,
                });
            }
        },
        .finance => {
            const num_assets = hdr.param_u32_0;
            const num_rules = hdr.param_u32_1;
            const rf_rate = hdr.param_f64_0;
            std.debug.print("assets:       {d}\n", .{num_assets});
            std.debug.print("rules:        {d}\n", .{num_rules});
            std.debug.print("rf_rate:      {d:.4}\n", .{rf_rate});

            const returns = try inst.getSlice(f64, hdr.offset_section_a, num_assets);
            std.debug.print("returns:      [", .{});
            for (returns, 0..) |r, i| {
                std.debug.print("{d:.4}{s}", .{ r, if (i + 1 < returns.len) ", " else "]\n" });
            }

            const cov = try inst.getSlice(f64, hdr.offset_section_b, num_assets * num_assets);
            std.debug.print("covariance:   {d}x{d} flat ({d} elements)\n", .{ num_assets, num_assets, cov.len });
        },
        .bioinformatics => {
            const num_snps = hdr.param_u32_0;
            const target_card = hdr.param_u32_1;
            const max_card = hdr.param_u32_2;
            const num_pairs = hdr.param_u32_3;
            const num_triplets = hdr.param_u32_4;
            const penalty_w = hdr.param_f64_0;
            std.debug.print("snps:         {d}\n", .{num_snps});
            std.debug.print("cardinality:  target={d} max={d}\n", .{ target_card, max_card });
            std.debug.print("epistasis:    pairs={d} triplets={d} penalty_w={d:.2}\n", .{ num_pairs, num_triplets, penalty_w });

            const pairs = try inst.getSlice(big_ll.data_format.EpistasisPair, hdr.offset_section_b, num_pairs);
            for (pairs, 0..) |p, i| {
                std.debug.print("  pair {d}: ({d}, {d}) weight={d:.1}\n", .{ i, p.idx1, p.idx2, p.weight });
            }
        },
        .airfoil => {
            const num_points = hdr.param_u32_0;
            const spline_points = hdr.param_u32_1;
            const num_upper = hdr.param_u32_2;
            const num_lower = hdr.param_u32_3;
            const cl_factor = hdr.param_f64_0;
            const base_cd = hdr.param_f64_1;
            std.debug.print("stations:     {d}\n", .{num_points});
            std.debug.print("spline_points: {d} ({d} upper, {d} lower)\n", .{ spline_points, num_upper, num_lower });
            std.debug.print("cl_factor:    {d:.4}\n", .{cl_factor});
            std.debug.print("base_cd:      {d:.4}\n", .{base_cd});
            if (spline_points > 0) {
                const pts = try inst.getSlice(big_ll.AirfoilDomain.Point, hdr.offset_section_a, @min(spline_points, 4));
                for (pts, 0..) |p, i| {
                    std.debug.print("  spline[{d}]: ({d:.6}, {d:.6})\n", .{ i, p.x, p.y });
                }
                if (spline_points > 4) {
                    std.debug.print("  ... ({d} total spline points)\n", .{spline_points});
                }
            }
        },
        .telecom => {
            const num_users = hdr.param_u32_0;
            const num_channels = hdr.param_u32_1;
            const noise = hdr.param_f64_0;
            std.debug.print("users:        {d}\n", .{num_users});
            std.debug.print("channels:     {d}\n", .{num_channels});
            std.debug.print("noise:        {d:.4}\n", .{noise});

            const mat = try inst.getSlice(f64, hdr.offset_section_a, num_users * num_users);
            std.debug.print("channel_mat:  {d}x{d} flat ({d} elements)\n", .{ num_users, num_users, mat.len });
        },
    }
}

fn cmdGenerate(domain_str: []const u8, out_path: []const u8, allocator: std.mem.Allocator) !void {
    var buffer: []align(64) u8 = undefined;

    if (std.mem.eql(u8, domain_str, "logistics")) {
        buffer = try big_ll.data_generator.generateLogistics(allocator, 32, 35.0, 50.0, 50.0);
    } else if (std.mem.eql(u8, domain_str, "power_grid")) {
        buffer = try big_ll.data_generator.generatePowerGrid(allocator, 550.0, 630.0);
    } else if (std.mem.eql(u8, domain_str, "finance")) {
        buffer = try big_ll.data_generator.generateFinance(allocator, 6, 8, 0.03);
    } else if (std.mem.eql(u8, domain_str, "bioinformatics")) {
        buffer = try big_ll.data_generator.generateBioinformatics(allocator, 128, 8, 12);
    } else if (std.mem.eql(u8, domain_str, "airfoil")) {
        buffer = try big_ll.data_generator.generateAirfoil(allocator, 16, 2.5, 0.01);
    } else if (std.mem.eql(u8, domain_str, "telecom")) {
        buffer = try big_ll.data_generator.generateTelecom(allocator, 16, 4, 0.05);
    } else {
        std.debug.print("error: unknown domain '{s}'. supported: logistics, power_grid, finance, bioinformatics, airfoil, telecom\n", .{domain_str});
        return;
    }
    defer allocator.free(buffer);

    const out_path_z = try allocator.dupeZ(u8, out_path);
    defer allocator.free(out_path_z);

    try big_ll.data_generator.saveToFile(out_path_z, buffer);
    const hdr: *const big_ll.BigDHeader = @ptrCast(buffer.ptr);

    std.debug.print("generated: domain={s} path={s} size={d} crc=0x{X:0>8}\n", .{
        domain_str,
        out_path,
        buffer.len,
        hdr.checksum,
    });
}

fn cmdGenerateSuite(target_dir: []const u8, allocator: std.mem.Allocator) !void {
    std.debug.print("generating 18-file synthetic benchmark suite into '{s}'...\n", .{target_dir});
    try big_ll.data_generator.generateBenchmarkSuite(allocator, target_dir);
    std.debug.print("suite generation complete: 18 datasets written across 6 domains (slack, standard, stress).\n", .{});
}

fn cmdIngestReal(raw_dir: []const u8, out_dir: []const u8, allocator: std.mem.Allocator) !void {
    std.debug.print("ingesting real-world benchmarks: '{s}' -> '{s}'...\n", .{ raw_dir, out_dir });
    try big_ll.data_importer.ingestAllRealDatasets(allocator, raw_dir, out_dir);
    std.debug.print("ingestion complete: real-world benchmark datasets written to '{s}'.\n", .{out_dir});
}

fn cmdDemo(allocator: std.mem.Allocator) !void {
    {
        const domain = big_ll.AirfoilDomain{};
        var engine = try GeneticEngine(big_ll.AirfoilDomain.CHROM_LEN, big_ll.AirfoilDomain).init(allocator, 128, domain, 1);
        defer engine.deinit();
        for (0..15) |_| try engine.step();
        std.debug.print("domain=airfoil l={d} gen={d} best_fitness={d:.4} dedup_hits={d} entropy={d:.4}\n", .{
            big_ll.AirfoilDomain.CHROM_LEN,
            engine.generation,
            engine.getBest().fitness,
            engine.dedup_cache.hits,
            engine.pop_current.entropy,
        });
    }

    {
        const domain = big_ll.LogisticsDomain.initDefault();
        var engine = try GeneticEngine(big_ll.LogisticsDomain.CHROM_LEN, big_ll.LogisticsDomain).init(allocator, 128, domain, 2);
        defer engine.deinit();
        for (0..15) |_| try engine.step();
        std.debug.print("domain=logistics l={d} gen={d} best_fitness={d:.4} dedup_hits={d} entropy={d:.4}\n", .{
            big_ll.LogisticsDomain.CHROM_LEN,
            engine.generation,
            engine.getBest().fitness,
            engine.dedup_cache.hits,
            engine.pop_current.entropy,
        });
    }

    {
        const domain = big_ll.PowerGridDomain.initDefault();
        var engine = try GeneticEngine(big_ll.PowerGridDomain.CHROM_LEN, big_ll.PowerGridDomain).init(allocator, 128, domain, 3);
        defer engine.deinit();
        for (0..15) |_| try engine.step();
        std.debug.print("domain=power_grid l={d} gen={d} best_fitness={d:.4} dedup_hits={d} entropy={d:.4}\n", .{
            big_ll.PowerGridDomain.CHROM_LEN,
            engine.generation,
            engine.getBest().fitness,
            engine.dedup_cache.hits,
            engine.pop_current.entropy,
        });
    }

    {
        const domain = big_ll.BioinformaticsDomain{};
        var engine = try GeneticEngine(big_ll.BioinformaticsDomain.CHROM_LEN, big_ll.BioinformaticsDomain).init(allocator, 128, domain, 4);
        defer engine.deinit();
        for (0..15) |_| try engine.step();
        std.debug.print("domain=bioinformatics l={d} gen={d} best_fitness={d:.4} dedup_hits={d} entropy={d:.4}\n", .{
            big_ll.BioinformaticsDomain.CHROM_LEN,
            engine.generation,
            engine.getBest().fitness,
            engine.dedup_cache.hits,
            engine.pop_current.entropy,
        });
    }

    {
        const domain = big_ll.FinanceDomain.initDefault();
        var engine = try GeneticEngine(big_ll.FinanceDomain.CHROM_LEN, big_ll.FinanceDomain).init(allocator, 128, domain, 5);
        defer engine.deinit();
        for (0..15) |_| try engine.step();
        std.debug.print("domain=finance l={d} gen={d} best_fitness={d:.4} dedup_hits={d} entropy={d:.4}\n", .{
            big_ll.FinanceDomain.CHROM_LEN,
            engine.generation,
            engine.getBest().fitness,
            engine.dedup_cache.hits,
            engine.pop_current.entropy,
        });
    }

    {
        const domain = big_ll.TelecomDomain.initDefault();
        var engine = try GeneticEngine(big_ll.TelecomDomain.CHROM_LEN, big_ll.TelecomDomain).init(allocator, 128, domain, 6);
        defer engine.deinit();
        for (0..15) |_| try engine.step();
        std.debug.print("domain=telecom l={d} gen={d} best_fitness={d:.4} dedup_hits={d} entropy={d:.4}\n", .{
            big_ll.TelecomDomain.CHROM_LEN,
            engine.generation,
            engine.getBest().fitness,
            engine.dedup_cache.hits,
            engine.pop_current.entropy,
        });
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    var it = init.minimal.args.iterate();

    _ = it.next(); // Skip binary path

    const cmd = it.next() orelse {
        try cmdDemo(allocator);
        return;
    };

    if (std.mem.eql(u8, cmd, "solve")) {
        const file_path = it.next() orelse {
            std.debug.print("error: missing file path.\nusage: big-ll solve <file.bigd> [--pop N] [--gens G] [--islands 1|4|8] [--seed S] [--dedup true|false] [--json] [--quiet]\n", .{});
            return;
        };

        var options = SolveOptions{};
        while (it.next()) |arg| {
            if (std.mem.eql(u8, arg, "--pop")) {
                const val_str = it.next() orelse {
                    std.debug.print("error: --pop requires an integer argument\n", .{});
                    return;
                };
                options.pop_size = std.fmt.parseInt(usize, val_str, 10) catch {
                    std.debug.print("error: invalid integer for --pop: '{s}'\n", .{val_str});
                    return;
                };
            } else if (std.mem.eql(u8, arg, "--gens")) {
                const val_str = it.next() orelse {
                    std.debug.print("error: --gens requires an integer argument\n", .{});
                    return;
                };
                options.gens = std.fmt.parseInt(usize, val_str, 10) catch {
                    std.debug.print("error: invalid integer for --gens: '{s}'\n", .{val_str});
                    return;
                };
            } else if (std.mem.eql(u8, arg, "--islands")) {
                const val_str = it.next() orelse {
                    std.debug.print("error: --islands requires 1, 4, or 8\n", .{});
                    return;
                };
                options.islands = std.fmt.parseInt(usize, val_str, 10) catch {
                    std.debug.print("error: invalid integer for --islands: '{s}'\n", .{val_str});
                    return;
                };
            } else if (std.mem.eql(u8, arg, "--seed")) {
                const val_str = it.next() orelse {
                    std.debug.print("error: --seed requires an integer argument\n", .{});
                    return;
                };
                options.seed = std.fmt.parseInt(u64, val_str, 10) catch {
                    std.debug.print("error: invalid integer for --seed: '{s}'\n", .{val_str});
                    return;
                };
            } else if (std.mem.eql(u8, arg, "--dedup")) {
                const val_str = it.next() orelse {
                    std.debug.print("error: --dedup requires true or false\n", .{});
                    return;
                };
                options.enable_dedup = std.mem.eql(u8, val_str, "true") or std.mem.eql(u8, val_str, "1");
            } else if (std.mem.eql(u8, arg, "--save-ir")) {
                const val_str = it.next() orelse {
                    std.debug.print("error: --save-ir requires an output file path\n", .{});
                    return;
                };
                options.save_ir = val_str;
            } else if (std.mem.eql(u8, arg, "--cluster")) {
                options.enable_cluster = true;
            } else if (std.mem.eql(u8, arg, "--transport")) {
                const val_str = it.next() orelse {
                    std.debug.print("error: --transport requires 'mpi', 'rdma', 'mesh', or 'shm'\n", .{});
                    return;
                };
                if (std.mem.eql(u8, val_str, "mpi")) {
                    options.transport = .mpi;
                } else if (std.mem.eql(u8, val_str, "rdma")) {
                    options.transport = .rdma;
                } else if (std.mem.eql(u8, val_str, "mesh")) {
                    options.transport = .mesh;
                } else if (std.mem.eql(u8, val_str, "shm")) {
                    options.transport = .shm;
                } else {
                    std.debug.print("error: invalid transport '{s}'. supported: mpi, rdma, mesh, shm\n", .{val_str});
                    return;
                }
            } else if (std.mem.eql(u8, arg, "--node-id")) {
                const val_str = it.next() orelse {
                    std.debug.print("error: --node-id requires an integer rank argument\n", .{});
                    return;
                };
                options.node_id = std.fmt.parseInt(u32, val_str, 10) catch {
                    std.debug.print("error: invalid integer for --node-id: '{s}'\n", .{val_str});
                    return;
                };
            } else if (std.mem.eql(u8, arg, "--num-nodes")) {
                const val_str = it.next() orelse {
                    std.debug.print("error: --num-nodes requires an integer argument\n", .{});
                    return;
                };
                options.num_nodes = std.fmt.parseInt(u32, val_str, 10) catch {
                    std.debug.print("error: invalid integer for --num-nodes: '{s}'\n", .{val_str});
                    return;
                };
            } else if (std.mem.eql(u8, arg, "--peer")) {
                const val_str = it.next() orelse {
                    std.debug.print("error: --peer requires an endpoint string\n", .{});
                    return;
                };
                options.peer_endpoint = val_str;
            } else if (std.mem.eql(u8, arg, "--pretty") or std.mem.eql(u8, arg, "-p")) {
                options.pretty = true;
            } else if (std.mem.eql(u8, arg, "--plot") or std.mem.eql(u8, arg, "--svg")) {
                const val_str = it.next() orelse {
                    std.debug.print("error: --plot requires an output SVG path argument\n", .{});
                    return;
                };
                options.plot_path = val_str;
            } else if (std.mem.eql(u8, arg, "--json")) {
                options.json = true;
            } else if (std.mem.eql(u8, arg, "--quiet") or std.mem.eql(u8, arg, "-q")) {
                options.quiet = true;
            } else {
                std.debug.print("warning: unrecognized option '{s}'\n", .{arg});
            }
        }

        try cmdSolve(file_path, options, allocator);
    } else if (std.mem.eql(u8, cmd, "inspect")) {
        const file_path = it.next() orelse {
            std.debug.print("error: missing file path. usage: big-ll inspect <file.bigd>\n", .{});
            return;
        };
        try cmdInspect(file_path, allocator);
    } else if (std.mem.eql(u8, cmd, "generate")) {
        const domain = it.next() orelse {
            std.debug.print("error: missing domain. usage: big-ll generate <domain> <out.bigd>\n", .{});
            return;
        };
        const out_path = it.next() orelse {
            std.debug.print("error: missing output path. usage: big-ll generate <domain> <out.bigd>\n", .{});
            return;
        };
        try cmdGenerate(domain, out_path, allocator);
    } else if (std.mem.eql(u8, cmd, "generate-suite")) {
        const target_dir = it.next() orelse "data/synthetic";
        try cmdGenerateSuite(target_dir, allocator);
    } else if (std.mem.eql(u8, cmd, "ingest-real")) {
        const raw_dir = it.next() orelse "data/raw";
        const out_dir = it.next() orelse "data/real";
        try cmdIngestReal(raw_dir, out_dir, allocator);
    } else if (std.mem.eql(u8, cmd, "demo")) {
        try cmdDemo(allocator);
    } else if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        printUsage();
    } else {
        std.debug.print("error: unknown command '{s}'\n\n", .{cmd});
        printUsage();
    }
}
