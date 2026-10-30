const std = @import("std");
const builtin = @import("builtin");

extern "c" fn pthread_set_qos_class_self_np(qos_class: c_uint, relative_priority: c_int) c_int;

// Apple Silicon Heterogeneous QoS Affinity (Firestorm P-Cores vs Icestorm E-Cores)
pub fn setWorkerQoSAffinity(worker_id: usize) void {
    if (builtin.os.tag == .macos or builtin.os.tag == .ios or builtin.os.tag == .watchos or builtin.os.tag == .tvos) {
        if (worker_id < 4) {
            _ = pthread_set_qos_class_self_np(0x21, 0); // QOS_CLASS_USER_INTERACTIVE (Firestorm P-Cores)
        } else {
            _ = pthread_set_qos_class_self_np(0x11, 0); // QOS_CLASS_UTILITY (Icestorm E-Cores)
        }
    }
}

pub fn setHighPerformanceAffinity() void {
    if (builtin.os.tag == .macos or builtin.os.tag == .ios or builtin.os.tag == .watchos or builtin.os.tag == .tvos) {
        _ = pthread_set_qos_class_self_np(0x21, 0); // QOS_CLASS_USER_INTERACTIVE
    }
}

pub const StaticThreadPool = struct {
    const Self = @This();
    pub const max_workers: usize = 16;

    pub const TaskFn = *const fn (ctx: *anyopaque, worker_id: usize) void;

    pub const Worker = struct {
        command: u32 align(64) = 0,
        done: u32 = 1,
        id: usize = 0,
        task_ctx: ?*anyopaque = null,
        task_fn: ?TaskFn = null,
        thread: ?std.Thread = null,

        fn loop(worker: *Worker) void {
            setWorkerQoSAffinity(worker.id);
            while (true) {
                // Low-latency user-space spin-wait before OS kernel context switch
                var spin: u32 = 0;
                while (@atomicLoad(u32, &worker.command, .acquire) == 0) {
                    if (spin < 128) {
                        spin += 1;
                        std.atomic.spinLoopHint();
                    } else {
                        std.Thread.yield() catch {};
                    }
                }
                const cmd = worker.command;
                if (cmd == 2) break;

                if (worker.task_fn) |run| {
                    if (worker.task_ctx) |ctx| {
                        run(ctx, worker.id);
                    }
                }
                @atomicStore(u32, &worker.command, 0, .release);
                @atomicStore(u32, &worker.done, 1, .release);
            }
        }
    };

    workers: [max_workers]Worker = [_]Worker{.{}} ** max_workers,
    num_workers: usize = 1,

    pub fn init(self: *Self, count: usize) !void {
        setHighPerformanceAffinity();
        self.num_workers = @min(@max(count, 1), max_workers);

        for (0..self.num_workers) |i| {
            self.workers[i].command = 0;
            self.workers[i].done = 1;
            self.workers[i].id = i;
            self.workers[i].task_ctx = null;
            self.workers[i].task_fn = null;
            self.workers[i].thread = null;
        }

        for (1..self.num_workers) |i| {
            self.workers[i].thread = try std.Thread.spawn(.{}, Worker.loop, .{&self.workers[i]});
        }
    }

    pub fn deinit(self: *Self) void {
        for (1..self.num_workers) |i| {
            @atomicStore(u32, &self.workers[i].command, 2, .release);
        }

        for (1..self.num_workers) |i| {
            if (self.workers[i].thread) |h| {
                h.join();
                self.workers[i].thread = null;
            }
        }
    }

    pub fn dispatch(self: *Self, ctx: *anyopaque, task_fn: TaskFn) void {
        if (self.num_workers <= 1) {
            task_fn(ctx, 0);
            return;
        }

        for (1..self.num_workers) |i| {
            self.workers[i].task_ctx = ctx;
            self.workers[i].task_fn = task_fn;
            @atomicStore(u32, &self.workers[i].done, 0, .release);
            @atomicStore(u32, &self.workers[i].command, 1, .release);
        }

        task_fn(ctx, 0);

        // Low-latency user-space spin-wait before OS kernel context switch
        for (1..self.num_workers) |i| {
            var spin: u32 = 0;
            while (@atomicLoad(u32, &self.workers[i].done, .acquire) == 0) {
                if (spin < 128) {
                    spin += 1;
                    std.atomic.spinLoopHint();
                } else {
                    std.Thread.yield() catch {};
                }
            }
        }
    }
};
