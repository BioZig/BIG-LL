const std = @import("std");

pub fn L1Dedup(comptime Capacity: usize) type {
    comptime {
        std.debug.assert(Capacity > 0 and (Capacity & (Capacity - 1)) == 0);
    }
    const mask: usize = Capacity - 1;

    return struct {
        const Self = @This();

        pub const Slot = struct {
            hash: u64,
            fitness: f64,
            valid: bool,
        };

        slots: [Capacity]Slot,
        hits: usize,
        misses: usize,

        pub fn init() Self {
            return .{
                .slots = [_]Slot{.{ .hash = 0, .fitness = 0.0, .valid = false }} ** Capacity,
                .hits = 0,
                .misses = 0,
            };
        }

        pub fn hashChromosome(words: []const u64) u64 {
            var h: u64 = 0x517cc1b727220a95;
            for (words) |w| {
                h ^= w;
                h = std.math.rotl(u64, h, 31) *% 0x9e3779b97f4a7c15;
            }
            h ^= h >> 33;
            h *%= 0xff51afd7ed558ccd;
            h ^= h >> 33;
            h *%= 0xc4ceb9fe1a85ec53;
            h ^= h >> 33;
            return h;
        }

        pub fn lookup(self: *Self, hash: u64) ?f64 {
            const idx = @as(usize, @intCast(hash)) & mask;
            const slot = &self.slots[idx];
            if (slot.valid and slot.hash == hash) {
                self.hits += 1;
                return slot.fitness;
            }
            self.misses += 1;
            return null;
        }

        pub fn insert(self: *Self, hash: u64, fitness: f64) void {
            const idx = @as(usize, @intCast(hash)) & mask;
            self.slots[idx] = .{
                .hash = hash,
                .fitness = fitness,
                .valid = true,
            };
        }

        pub fn hitRatio(self: *const Self) f64 {
            const total = self.hits + self.misses;
            if (total == 0) return 0.0;
            return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(total));
        }

        pub fn clear(self: *Self) void {
            for (&self.slots) |*s| {
                s.valid = false;
            }
            self.hits = 0;
            self.misses = 0;
        }
    };
}

pub fn AdaptiveDedup(comptime NumSets: usize, comptime Assoc: usize) type {
    comptime {
        std.debug.assert(NumSets > 0 and (NumSets & (NumSets - 1)) == 0);
        std.debug.assert(Assoc >= 2 and Assoc <= 4);
    }
    const set_mask: usize = NumSets - 1;

    return struct {
        const Self = @This();

        pub const Slot = struct {
            hash: u64 = 0,
            fitness: f64 = 0.0,
            age: u16 = 0,
            valid: bool = false,
        };

        pub const Set = struct {
            slots: [Assoc]Slot = [_]Slot{.{}} ** Assoc,
        };

        sets: [NumSets]Set,
        bloom_filter: [16]u64,
        epoch_counter: u16,
        hits: usize,
        misses: usize,
        bypasses: usize,
        entropy_threshold: f64,

        pub fn init() Self {
            return .{
                .sets = [_]Set{.{}} ** NumSets,
                .bloom_filter = [_]u64{0} ** 16,
                .epoch_counter = 0,
                .hits = 0,
                .misses = 0,
                .bypasses = 0,
                .entropy_threshold = 0.35,
            };
        }

        pub fn hashChromosome(words: []const u64) u64 {
            return L1Dedup(64).hashChromosome(words);
        }

        pub inline fn shouldBypass(self: *const Self, entropy: f64) bool {
            return entropy > self.entropy_threshold;
        }

        pub fn checkBloom(self: *const Self, hash: u64) bool {
            const b1 = @as(usize, @intCast(hash & 1023));
            const b2 = @as(usize, @intCast((hash >> 32) & 1023));
            const w1 = b1 / 64;
            const bit1: u6 = @intCast(b1 % 64);
            const w2 = b2 / 64;
            const bit2: u6 = @intCast(b2 % 64);

            return ((self.bloom_filter[w1] & (@as(u64, 1) << bit1)) != 0) and
                ((self.bloom_filter[w2] & (@as(u64, 1) << bit2)) != 0);
        }

        pub fn markBloom(self: *Self, hash: u64) void {
            const b1 = @as(usize, @intCast(hash & 1023));
            const b2 = @as(usize, @intCast((hash >> 32) & 1023));
            const w1 = b1 / 64;
            const bit1: u6 = @intCast(b1 % 64);
            const w2 = b2 / 64;
            const bit2: u6 = @intCast(b2 % 64);

            self.bloom_filter[w1] |= (@as(u64, 1) << bit1);
            self.bloom_filter[w2] |= (@as(u64, 1) << bit2);
        }

        pub fn lookup(self: *Self, hash: u64, entropy: f64) ?f64 {
            if (self.shouldBypass(entropy)) {
                self.bypasses += 1;
                return null;
            }

            if (!self.checkBloom(hash)) {
                self.misses += 1;
                return null;
            }

            const set_idx = @as(usize, @intCast(hash)) & set_mask;
            const set = &self.sets[set_idx];

            self.epoch_counter +%= 1;
            for (&set.slots) |*slot| {
                if (slot.valid and slot.hash == hash) {
                    slot.age = self.epoch_counter;
                    self.hits += 1;
                    return slot.fitness;
                }
            }

            self.misses += 1;
            return null;
        }

        pub fn insert(self: *Self, hash: u64, fitness: f64, entropy: f64) void {
            if (self.shouldBypass(entropy)) return;

            self.markBloom(hash);

            const set_idx = @as(usize, @intCast(hash)) & set_mask;
            const set = &self.sets[set_idx];

            self.epoch_counter +%= 1;

            for (&set.slots) |*slot| {
                if (slot.valid and slot.hash == hash) {
                    slot.fitness = fitness;
                    slot.age = self.epoch_counter;
                    return;
                }
            }

            for (&set.slots) |*slot| {
                if (!slot.valid) {
                    slot.* = .{
                        .hash = hash,
                        .fitness = fitness,
                        .age = self.epoch_counter,
                        .valid = true,
                    };
                    return;
                }
            }

            var min_idx: usize = 0;
            var min_age: u16 = set.slots[0].age;
            for (set.slots[1..], 1..) |slot, i| {
                if (slot.age < min_age) {
                    min_age = slot.age;
                    min_idx = i;
                }
            }

            set.slots[min_idx] = .{
                .hash = hash,
                .fitness = fitness,
                .age = self.epoch_counter,
                .valid = true,
            };
        }

        pub fn clear(self: *Self) void {
            for (&self.sets) |*s| {
                for (&s.slots) |*slot| {
                    slot.valid = false;
                }
            }
            @memset(&self.bloom_filter, 0);
            self.hits = 0;
            self.misses = 0;
            self.bypasses = 0;
            self.epoch_counter = 0;
        }

        pub fn hitRatio(self: *const Self) f64 {
            const total = self.hits + self.misses;
            if (total == 0) return 0.0;
            return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(total));
        }
    };
}
