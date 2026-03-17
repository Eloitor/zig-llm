const std = @import("std");
const Allocator = std.mem.Allocator;
const ProviderError = @import("errors.zig").ProviderError;

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

/// Incremental SSE parser that reads from a live stream (*std.Io.Reader).
/// Reads one line at a time and assembles complete SSE events.
/// The returned SseEvent data is valid until the next call to nextEvent().
pub const SseLineReader = struct {
    underlying: *std.Io.Reader,
    allocator: Allocator,
    line_buf: std.ArrayList(u8),
    event_type_buf: [128]u8,
    data_buf: std.ArrayList(u8),
    event_type_len: usize,
    has_event_type: bool,
    has_data: bool,
    eof: bool,

    pub fn init(reader: *std.Io.Reader, allocator: Allocator) SseLineReader {
        return .{
            .underlying = reader,
            .allocator = allocator,
            .line_buf = .{},
            .event_type_buf = undefined,
            .data_buf = .{},
            .event_type_len = 0,
            .has_event_type = false,
            .has_data = false,
            .eof = false,
        };
    }

    pub fn deinit(self: *SseLineReader) void {
        self.data_buf.deinit(self.allocator);
        self.line_buf.deinit(self.allocator);
    }

    /// Read a line from the underlying reader (up to and consuming '\n').
    /// Returns the line content without the trailing '\n' (and strips '\r' if present).
    /// Returns null on EOF.
    fn readLine(self: *SseLineReader) ProviderError!?[]const u8 {
        if (self.eof) return null;

        self.line_buf.clearRetainingCapacity();

        while (true) {
            // Try to get buffered data
            const buffered = self.underlying.buffered();
            if (buffered.len > 0) {
                // Scan for newline in buffered data
                if (std.mem.indexOfScalar(u8, buffered, '\n')) |nl_pos| {
                    // Found newline — append everything up to it
                    self.line_buf.appendSlice(self.allocator, buffered[0..nl_pos]) catch return error.OutOfMemory;
                    self.underlying.toss(nl_pos + 1); // consume including '\n'

                    // Strip trailing \r
                    const line = self.line_buf.items;
                    const len = if (line.len > 0 and line[line.len - 1] == '\r') line.len - 1 else line.len;
                    return line[0..len];
                } else {
                    // No newline yet — consume all buffered data and refill
                    self.line_buf.appendSlice(self.allocator, buffered) catch return error.OutOfMemory;
                    self.underlying.toss(buffered.len);
                }
            }

            // Try to fill the buffer with more data from the stream
            self.underlying.fillMore() catch {
                // EndOfStream or read error — treat as EOF
                self.eof = true;
                if (self.line_buf.items.len > 0) {
                    return self.line_buf.items;
                }
                return null;
            };
        }
    }

    /// Read and return the next complete SSE event from the stream.
    /// Returns null on EOF (connection closed). The returned SseEvent's
    /// data slice is valid until the next call to nextEvent().
    pub fn nextEvent(self: *SseLineReader) ProviderError!?SseEvent {
        while (true) {
            const line = (try self.readLine()) orelse {
                // EOF — if we have accumulated data, return it as a final event
                if (self.has_data) {
                    return self.emitEvent();
                }
                return null;
            };

            // Empty line = event boundary
            if (line.len == 0) {
                if (self.has_data) {
                    return self.emitEvent();
                }
                // Reset state and continue
                self.resetEventState();
                continue;
            }

            // Comment line — skip
            if (line[0] == ':') continue;

            // Parse field
            if (std.mem.startsWith(u8, line, "event:")) {
                const value = std.mem.trimLeft(u8, line["event:".len..], " ");
                const copy_len = @min(value.len, self.event_type_buf.len);
                @memcpy(self.event_type_buf[0..copy_len], value[0..copy_len]);
                self.event_type_len = copy_len;
                self.has_event_type = true;
            } else if (std.mem.startsWith(u8, line, "data:")) {
                const value_start = "data:".len + @as(usize, if (line.len > "data:".len and line["data:".len] == ' ') 1 else 0);
                const value = line[value_start..];
                // If we already have data, join with newline (SSE spec)
                if (self.data_buf.items.len > 0) {
                    self.data_buf.append(self.allocator, '\n') catch return error.OutOfMemory;
                }
                self.data_buf.appendSlice(self.allocator, value) catch return error.OutOfMemory;
                self.has_data = true;
            }
        }
    }

    fn emitEvent(self: *SseLineReader) SseEvent {
        const event = SseEvent{
            .event_type = if (self.has_event_type) self.event_type_buf[0..self.event_type_len] else null,
            .data = self.data_buf.items,
            .end_pos = 0,
        };
        // Don't clear data_buf here — the caller needs the slice to remain valid.
        // Instead, we'll clear it at the start of the next accumulation.
        self.resetEventState();
        return event;
    }

    fn resetEventState(self: *SseLineReader) void {
        self.has_event_type = false;
        self.has_data = false;
        self.event_type_len = 0;
        self.data_buf.clearRetainingCapacity();
    }
};

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

// --- SseLineReader tests ---

test "SseLineReader: single event" {
    const input = "event: message_start\ndata: {\"type\":\"message\"}\n\n";
    var fixed_reader = std.Io.Reader.fixed(input);
    var reader = SseLineReader.init(&fixed_reader, std.testing.allocator);
    defer reader.deinit();

    const event = (try reader.nextEvent()) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("message_start", event.event_type.?);
    try std.testing.expectEqualStrings("{\"type\":\"message\"}", event.data);

    // Next call should return null (EOF)
    try std.testing.expect((try reader.nextEvent()) == null);
}

test "SseLineReader: multiple events" {
    const input = "event: a\ndata: first\n\nevent: b\ndata: second\n\n";
    var fixed_reader = std.Io.Reader.fixed(input);
    var reader = SseLineReader.init(&fixed_reader, std.testing.allocator);
    defer reader.deinit();

    const e1 = (try reader.nextEvent()) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("a", e1.event_type.?);
    try std.testing.expectEqualStrings("first", e1.data);

    const e2 = (try reader.nextEvent()) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("b", e2.event_type.?);
    try std.testing.expectEqualStrings("second", e2.data);

    try std.testing.expect((try reader.nextEvent()) == null);
}

test "SseLineReader: skip comments" {
    const input = ": comment\nevent: ping\ndata: hello\n\n";
    var fixed_reader = std.Io.Reader.fixed(input);
    var reader = SseLineReader.init(&fixed_reader, std.testing.allocator);
    defer reader.deinit();

    const event = (try reader.nextEvent()) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("ping", event.event_type.?);
    try std.testing.expectEqualStrings("hello", event.data);
}

test "SseLineReader: event without type" {
    const input = "data: no-type\n\n";
    var fixed_reader = std.Io.Reader.fixed(input);
    var reader = SseLineReader.init(&fixed_reader, std.testing.allocator);
    defer reader.deinit();

    const event = (try reader.nextEvent()) orelse return error.TestUnexpectedResult;
    try std.testing.expect(event.event_type == null);
    try std.testing.expectEqualStrings("no-type", event.data);
}

test "SseLineReader: multiple data lines joined with newline" {
    const input = "event: content\ndata: line1\ndata: line2\ndata: line3\n\n";
    var fixed_reader = std.Io.Reader.fixed(input);
    var reader = SseLineReader.init(&fixed_reader, std.testing.allocator);
    defer reader.deinit();

    const event = (try reader.nextEvent()) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("content", event.event_type.?);
    try std.testing.expectEqualStrings("line1\nline2\nline3", event.data);
}

test "SseLineReader: handles \\r\\n line endings" {
    const input = "event: test\r\ndata: hello\r\n\r\n";
    var fixed_reader = std.Io.Reader.fixed(input);
    var reader = SseLineReader.init(&fixed_reader, std.testing.allocator);
    defer reader.deinit();

    const event = (try reader.nextEvent()) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("test", event.event_type.?);
    try std.testing.expectEqualStrings("hello", event.data);
}
