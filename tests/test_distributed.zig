const std = @import("std");
const big_ll = @import("big_ll");
const distributed = big_ll.distributed;

test "FrameHeader size, alignment, and validation invariants" {
    try std.testing.expectEqual(@as(usize, 128), @sizeOf(distributed.FrameHeader));
    try std.testing.expectEqual(@as(usize, 64), @alignOf(distributed.FrameHeader));

    var hdr = distributed.FrameHeader{
        .source_rank = 0,
        .target_rank = 1,
        .generation = 50,
        .immigrant_count = 4,
        .bits_per_chrom = 512,
        .words_per_chrom = 8,
        .payload_bytes = 4 * (8 + 8 * 8),
    };

    try hdr.validate(128 + hdr.payload_bytes);

    // Invalid magic rejection
    hdr.magic = 0x12345678;
    try std.testing.expectError(error.InvalidNetworkMagic, hdr.validate(256));
    hdr.magic = distributed.NetworkMagic;

    // Truncated network frame rejection
    try std.testing.expectError(error.TruncatedNetworkHeader, hdr.validate(64));
}

test "NetworkMailbox lock-free circular ring immigrant exchange" {
    const allocator = std.testing.allocator;
    const words_per_chrom = 8; // 512 bits
    var mb = try distributed.NetworkMailbox.init(allocator, 16, words_per_chrom);
    defer mb.deinit();

    const sample_words = [_]u64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const push_ok = mb.push(42.5, &sample_words);
    try std.testing.expect(push_ok);

    var out_fit: f64 = 0.0;
    var out_words = [_]u64{0} ** words_per_chrom;
    const pop_ok = mb.pop(&out_fit, &out_words);
    try std.testing.expect(pop_ok);
    try std.testing.expectEqual(@as(f64, 42.5), out_fit);
    try std.testing.expectEqualSlices(u64, &sample_words, &out_words);

    // Empty pop returns false
    try std.testing.expect(!mb.pop(&out_fit, &out_words));
}

test "DistributedTransport multi-node elite broadcast and immigrant polling" {
    const allocator = std.testing.allocator;
    const words_per_chrom = 8;

    var transport = try distributed.DistributedTransport.init(
        allocator,
        .mesh,
        0, // rank 0
        2, // 2 nodes
        words_per_chrom,
    );
    defer transport.deinit();

    try std.testing.expectEqual(distributed.TransportType.mesh, transport.transport_type);
    try std.testing.expectEqual(@as(u32, 0), transport.rank);
    try std.testing.expectEqual(@as(u32, 2), transport.world_size);

    const elites_fit = [_]f64{ 100.5, 95.2, 90.1 };
    const elites_words = [_]u64{
        1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3,
    };

    const sent = transport.broadcastElites(10, &elites_fit, &elites_words, words_per_chrom);
    try std.testing.expectEqual(@as(usize, 3), sent);

    var rec_fit: f64 = 0.0;
    var rec_words = [_]u64{0} ** words_per_chrom;
    try std.testing.expect(transport.pollImmigrant(&rec_fit, &rec_words));
    try std.testing.expectEqual(@as(f64, 100.5), rec_fit);
    try std.testing.expectEqual(@as(u64, 1), rec_words[0]);

    try std.testing.expect(transport.pollImmigrant(&rec_fit, &rec_words));
    try std.testing.expectEqual(@as(f64, 95.2), rec_fit);
    try std.testing.expectEqual(@as(u64, 2), rec_words[0]);
}
