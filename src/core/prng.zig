const std = @import("std");

pub const Xoshiro256 = struct {
    const Self = @This();

    s: [4]u64,

    pub fn init(seed: u64) Self {
        var sm = seed;
        var s: [4]u64 = undefined;
        for (&s) |*word| {
            sm +%= 0x9E3779B97F4A7C15;
            var z = sm;
            z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
            z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
            word.* = z ^ (z >> 31);
        }
        return .{ .s = s };
    }

    pub fn next(self: *Self) u64 {
        const res = std.math.rotl(u64, self.s[0] +% self.s[3], 23) +% self.s[0];
        const t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = std.math.rotl(u64, self.s[3], 45);
        return res;
    }

    pub fn nextFloat(self: *Self) f64 {
        const val = self.next() >> 11;
        return @as(f64, @floatFromInt(val)) * (1.0 / 9007199254740992.0);
    }

    pub fn uintLessThan(self: *Self, bound: usize) usize {
        std.debug.assert(bound > 0);
        return @intCast(self.next() % @as(u64, @intCast(bound)));
    }

    pub fn fillWords(self: *Self, words: []u64) void {
        for (words) |*w| {
            w.* = self.next();
        }
    }

    pub fn sparseMask(self: *Self, depth: u3) u64 {
        var mask: u64 = self.next();
        var i: u3 = 1;
        while (i < depth) : (i += 1) {
            mask &= self.next();
        }
        return mask;
    }

    pub fn fillSparseWords(self: *Self, words: []u64, depth: u3) void {
        for (words) |*w| {
            w.* = self.sparseMask(depth);
        }
    }
};
