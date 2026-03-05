const std = @import("std");

pub const SseEvent = struct {
    event_type: ?[]const u8,
    data: []const u8,
    /// Position in the source buffer after this event.
    end_pos: usize,
};

/// Parse the next SSE event from `data` starting at `position`.
/// Returns null if no complete event is found.
/// All returned slices point into `data` (no allocations).
pub fn nextEvent(data: []const u8, position: usize) ?SseEvent {
    var pos = position;
    var event_type: ?[]const u8 = null;
    var data_start: ?usize = null;
    var data_end: ?usize = null;
    var found_data = false;

    while (pos < data.len) {
        // Find end of current line
        const line_end = std.mem.indexOfScalar(u8, data[pos..], '\n') orelse {
            // No newline found — incomplete event
            break;
        };
        const line = blk: {
            const raw = data[pos .. pos + line_end];
            // Strip trailing \r
            break :blk if (raw.len > 0 and raw[raw.len - 1] == '\r') raw[0 .. raw.len - 1] else raw;
        };
        pos += line_end + 1;

        // Empty line = event delimiter
        if (line.len == 0) {
            if (found_data) {
                return .{
                    .event_type = event_type,
                    .data = data[data_start.? .. data_end.?],
                    .end_pos = pos,
                };
            }
            // Reset for next potential event
            event_type = null;
            data_start = null;
            data_end = null;
            continue;
        }

        // Comment line
        if (line[0] == ':') continue;

        // Parse field
        if (std.mem.startsWith(u8, line, "event:")) {
            event_type = std.mem.trimLeft(u8, line["event:".len..], " ");
        } else if (std.mem.startsWith(u8, line, "data:")) {
            const value_start = "data:".len + @as(usize, if (line.len > "data:".len and line["data:".len] == ' ') @as(usize, 1) else 0);
            const abs_start = pos - line_end - 1 + value_start;
            const abs_end = pos - 1 - @as(usize, if (data[pos - 2] == '\r') @as(usize, 1) else 0); // handles \r\n
            if (!found_data) {
                data_start = abs_start;
            }
            data_end = abs_end;
            found_data = true;
        }
    }

    return null;
}

// --- Tests ---

test "parse single SSE event" {
    const input = "event: message_start\ndata: {\"type\":\"message\"}\n\n";
    const event = nextEvent(input, 0) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("message_start", event.event_type.?);
    try std.testing.expectEqualStrings("{\"type\":\"message\"}", event.data);
    try std.testing.expectEqual(input.len, event.end_pos);
}

test "parse multiple SSE events" {
    const input = "event: a\ndata: first\n\nevent: b\ndata: second\n\n";
    const e1 = nextEvent(input, 0) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("a", e1.event_type.?);
    try std.testing.expectEqualStrings("first", e1.data);

    const e2 = nextEvent(input, e1.end_pos) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("b", e2.event_type.?);
    try std.testing.expectEqualStrings("second", e2.data);
}

test "skip comment lines" {
    const input = ": this is a comment\nevent: ping\ndata: hello\n\n";
    const event = nextEvent(input, 0) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("ping", event.event_type.?);
    try std.testing.expectEqualStrings("hello", event.data);
}

test "event without type" {
    const input = "data: no-type\n\n";
    const event = nextEvent(input, 0) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(?[]const u8, null), event.event_type);
    try std.testing.expectEqualStrings("no-type", event.data);
}

test "incomplete event returns null" {
    const input = "event: partial\ndata: incomplete";
    try std.testing.expectEqual(@as(?SseEvent, null), nextEvent(input, 0));
}
