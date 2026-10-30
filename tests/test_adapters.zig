const std = @import("std");
const big_ll = @import("big_ll");
const gray = big_ll.gray_code;
const RandomKeyDecoder = big_ll.RandomKeyDecoder;
const MinlpDecoder = big_ll.MinlpDecoder;
const BooleanSetDecoder = big_ll.BooleanSetDecoder;

test "GrayCode bijection and single-bit transition property" {
    var i: u64 = 0;
    while (i < 1000) : (i += 1) {
        const g = gray.binaryToGray(i);
        const b = gray.grayToBinary(g);
        try std.testing.expectEqual(i, b);

        if (i > 0) {
            const prev_g = gray.binaryToGray(i - 1);
            try std.testing.expectEqual(@as(u32, 1), @popCount(g ^ prev_g));
        }
    }
}

test "GrayCode bitfield insertion and extraction across 64-bit word boundary" {
    var words = [_]u64{ 0, 0 };
    const offset: usize = 60;
    const len: usize = 8;
    const val: u64 = 0b10110101;

    gray.insertBits(&words, offset, len, val);
    const extracted = gray.extractBits(&words, offset, len);
    try std.testing.expectEqual(val, extracted);
}

test "GrayCode float encode and decode roundtrip" {
    var words = [_]u64{0};
    const min = -10.0;
    const max = 10.0;
    const bits = 16;
    const target = 3.1415;

    gray.encodeFloat(&words, 0, bits, min, max, target);
    const decoded = gray.decodeFloat(&words, 0, bits, min, max);

    const tolerance = (max - min) / 65535.0;
    try std.testing.expect(std.math.approxEqAbs(f64, target, decoded, tolerance));
}

test "GrayDecoder multi-variable decoding" {
    const params = [_]gray.FloatParam{
        .{ .min = 0.0, .max = 1.0, .bit_len = 8 },
        .{ .min = -50.0, .max = 50.0, .bit_len = 16 },
    };
    const Decoder = gray.GrayDecoder(&params);
    var words = [_]u64{0};

    gray.encodeFloat(&words, 0, 8, 0.0, 1.0, 0.75);
    gray.encodeFloat(&words, 8, 16, -50.0, 50.0, 25.0);

    var res: [2]f64 = undefined;
    Decoder.decode(&words, &res);

    try std.testing.expect(std.math.approxEqAbs(f64, 0.75, res[0], 0.01));
    try std.testing.expect(std.math.approxEqAbs(f64, 25.0, res[1], 0.01));
}

test "RandomKeyDecoder permutation validity and bijection" {
    const Decoder = RandomKeyDecoder(8, 16);
    var words = [_]u64{ 0x0004000100030002, 0x0008000500070006 };
    var perm: [8]u32 = undefined;
    Decoder.decode(&words, &perm);

    var seen = [_]bool{false} ** 8;
    for (perm) |idx| {
        try std.testing.expect(idx < 8);
        try std.testing.expect(!seen[idx]);
        seen[idx] = true;
    }
}

test "MinlpDecoder switch and scalar decoding" {
    const params = [_]gray.FloatParam{
        .{ .min = 10.0, .max = 100.0, .bit_len = 16 },
    };
    const Decoder = MinlpDecoder(4, &params);
    var words = [_]u64{0};

    const state = Decoder.DecodedState{
        .switches = [_]u1{ 1, 0, 1, 1 },
        .scalars = [_]f64{55.0},
    };

    Decoder.encode(&words, &state);

    var decoded_state: Decoder.DecodedState = undefined;
    Decoder.decode(&words, &decoded_state);

    try std.testing.expectEqualSlices(u1, &state.switches, &decoded_state.switches);
    try std.testing.expect(std.math.approxEqAbs(f64, 55.0, decoded_state.scalars[0], 0.01));
}

test "BooleanSetDecoder index decoding and cardinality" {
    const Decoder = BooleanSetDecoder(128);
    var words = [_]u64{ 0b100101, 0b1000 };
    var indices: [128]usize = undefined;

    const count = Decoder.decodeIndices(&words, &indices);
    try std.testing.expectEqual(@as(usize, 4), count);
    try std.testing.expectEqual(@as(usize, 0), indices[0]);
    try std.testing.expectEqual(@as(usize, 2), indices[1]);
    try std.testing.expectEqual(@as(usize, 5), indices[2]);
    try std.testing.expectEqual(@as(usize, 67), indices[3]);

    const penalty = Decoder.evaluateCardinalityPenalty(count, 3, 10.0);
    try std.testing.expectEqual(@as(f64, 10.0), penalty);
}

test "AugmentedLagrangianConstraint dynamic penalty scaling" {
    var alc = big_ll.AugmentedLagrangianConstraint{
        .mu = 100.0,
    };

    // 10 evaluations, all violating -> infeasible ratio = 1.0 > 0.05
    for (0..10) |_| {
        alc.recordEvaluation(true);
    }
    alc.updatePenalty();
    try std.testing.expect(std.math.approxEqAbs(f64, 120.0, alc.mu, 0.001));

    // 100 evaluations, all feasible -> infeasible ratio = 0.0 <= 0.01
    for (0..100) |_| {
        alc.recordEvaluation(false);
    }
    alc.updatePenalty();
    try std.testing.expect(std.math.approxEqAbs(f64, 108.0, alc.mu, 0.001));

    const violations = [_]f64{ 0.0, 2.0, 3.0, -1.0 };
    const penalty = alc.computePenalty(&violations);
    // penalty = 108.0 * (2^2 + 3^2) = 108.0 * 13 = 1404.0
    try std.testing.expect(std.math.approxEqAbs(f64, 1404.0, penalty, 0.001));
}

test "RandomKeyDecoder decodeFast ranking property and bijection" {
    const Decoder = RandomKeyDecoder(8, 16);
    var words = [_]u64{ 0x0004000100030002, 0x0008000500070006 };
    var perm_fast: [8]u32 = undefined;
    var perm_std: [8]u32 = undefined;

    Decoder.decodeFast(&words, &perm_fast);
    Decoder.decode(&words, &perm_std);

    var seen = [_]bool{false} ** 8;
    for (perm_fast) |idx| {
        try std.testing.expect(idx < 8);
        try std.testing.expect(!seen[idx]);
        seen[idx] = true;
    }

    try std.testing.expectEqualSlices(u32, &perm_std, &perm_fast);
}
