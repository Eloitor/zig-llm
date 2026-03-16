const std = @import("std");
const Allocator = std.mem.Allocator;

/// A streaming JSON writer that handles commas, nesting, and escaping.
pub fn JsonWriter(comptime Writer: type) type {
    return struct {
        writer: Writer,
        depth: usize,
        needs_comma: [max_depth]bool,

        const Self = @This();
        const max_depth = 256;

        pub fn init(writer: Writer) Self {
            return .{
                .writer = writer,
                .depth = 0,
                .needs_comma = [_]bool{false} ** max_depth,
            };
        }

        fn writeCommaIfNeeded(self: *Self) !void {
            if (self.depth > 0 and self.needs_comma[self.depth]) {
                try self.writer.writeByte(',');
            }
            if (self.depth > 0) {
                self.needs_comma[self.depth] = true;
            }
        }

        pub fn beginObject(self: *Self) !void {
            if (self.depth + 1 >= max_depth) return error.Overflow;
            try self.writeCommaIfNeeded();
            try self.writer.writeByte('{');
            self.depth += 1;
            self.needs_comma[self.depth] = false;
        }

        pub fn endObject(self: *Self) !void {
            self.depth -= 1;
            try self.writer.writeByte('}');
        }

        pub fn beginArray(self: *Self) !void {
            if (self.depth + 1 >= max_depth) return error.Overflow;
            try self.writeCommaIfNeeded();
            try self.writer.writeByte('[');
            self.depth += 1;
            self.needs_comma[self.depth] = false;
        }

        pub fn endArray(self: *Self) !void {
            self.depth -= 1;
            try self.writer.writeByte(']');
        }

        pub fn field(self: *Self, name: []const u8) !void {
            try self.writeCommaIfNeeded();
            try self.writeQuotedString(name);
            try self.writer.writeByte(':');
            // Reset comma flag since field value is next
            self.needs_comma[self.depth] = false;
        }

        pub fn valueString(self: *Self, s: []const u8) !void {
            try self.writeCommaIfNeeded();
            try self.writeQuotedString(s);
        }

        pub fn valueInt(self: *Self, v: i64) !void {
            try self.writeCommaIfNeeded();
            try self.writer.print("{d}", .{v});
        }

        pub fn valueFloat(self: *Self, v: f64) !void {
            try self.writeCommaIfNeeded();
            try self.writer.print("{d}", .{v});
        }

        pub fn valueBool(self: *Self, v: bool) !void {
            try self.writeCommaIfNeeded();
            try self.writer.writeAll(if (v) "true" else "false");
        }

        pub fn valueNull(self: *Self) !void {
            try self.writeCommaIfNeeded();
            try self.writer.writeAll("null");
        }

        pub fn valueRaw(self: *Self, raw_json: []const u8) !void {
            try self.writeCommaIfNeeded();
            try self.writer.writeAll(raw_json);
        }

        fn writeQuotedString(self: *Self, s: []const u8) !void {
            try self.writer.writeByte('"');
            for (s) |c| {
                switch (c) {
                    '"' => try self.writer.writeAll("\\\""),
                    '\\' => try self.writer.writeAll("\\\\"),
                    '\n' => try self.writer.writeAll("\\n"),
                    '\r' => try self.writer.writeAll("\\r"),
                    '\t' => try self.writer.writeAll("\\t"),
                    else => {
                        if (c < 0x20) {
                            try self.writer.print("\\u{x:0>4}", .{c});
                        } else {
                            try self.writer.writeByte(c);
                        }
                    },
                }
            }
            try self.writer.writeByte('"');
        }
    };
}

pub fn jsonWriter(writer: anytype) JsonWriter(@TypeOf(writer)) {
    return JsonWriter(@TypeOf(writer)).init(writer);
}

// --- JSON parsing helpers ---

pub fn getJsonString(value: std.json.Value) ?[]const u8 {
    return switch (value) {
        .string => |s| s,
        else => null,
    };
}

pub fn getJsonInt(value: std.json.Value) ?i64 {
    return switch (value) {
        .integer => |i| i,
        else => null,
    };
}

pub fn getJsonBool(value: std.json.Value) ?bool {
    return switch (value) {
        .bool => |b| b,
        else => null,
    };
}

pub fn getJsonObject(value: std.json.Value) ?std.json.ObjectMap {
    return switch (value) {
        .object => |o| o,
        else => null,
    };
}

pub fn getJsonArray(value: std.json.Value) ?std.json.Array {
    return switch (value) {
        .array => |a| a,
        else => null,
    };
}

/// Navigate a JSON value by dot-separated path (e.g., "content.0.text").
pub fn getPath(value: std.json.Value, path: []const u8) ?std.json.Value {
    var current = value;
    var remaining = path;

    while (remaining.len > 0) {
        const dot = std.mem.indexOfScalar(u8, remaining, '.') orelse remaining.len;
        const key = remaining[0..dot];
        remaining = if (dot < remaining.len) remaining[dot + 1 ..] else "";

        switch (current) {
            .object => |obj| {
                current = obj.get(key) orelse return null;
            },
            .array => |arr| {
                const idx = std.fmt.parseInt(usize, key, 10) catch return null;
                if (idx >= arr.items.len) return null;
                current = arr.items[idx];
            },
            else => return null,
        }
    }
    return current;
}

/// Stringify a std.json.Value to an owned JSON string.
pub fn stringifyValue(allocator: Allocator, value: std.json.Value) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();
    try std.json.stringify(value, .{}, buf.writer());
    return buf.toOwnedSlice();
}

// --- Tests ---

test "JsonWriter basic object" {
    var buf = std.ArrayList(u8).init(std.testing.allocator);
    defer buf.deinit();

    var jw = jsonWriter(buf.writer());
    try jw.beginObject();
    try jw.field("name");
    try jw.valueString("Claude");
    try jw.field("version");
    try jw.valueInt(3);
    try jw.endObject();

    try std.testing.expectEqualStrings("{\"name\":\"Claude\",\"version\":3}", buf.items);
}

test "JsonWriter nested" {
    var buf = std.ArrayList(u8).init(std.testing.allocator);
    defer buf.deinit();

    var jw = jsonWriter(buf.writer());
    try jw.beginObject();
    try jw.field("items");
    try jw.beginArray();
    try jw.valueString("a");
    try jw.valueString("b");
    try jw.endArray();
    try jw.endObject();

    try std.testing.expectEqualStrings("{\"items\":[\"a\",\"b\"]}", buf.items);
}

test "JsonWriter escaping" {
    var buf = std.ArrayList(u8).init(std.testing.allocator);
    defer buf.deinit();

    var jw = jsonWriter(buf.writer());
    try jw.valueString("hello \"world\"\nnewline");

    try std.testing.expectEqualStrings("\"hello \\\"world\\\"\\nnewline\"", buf.items);
}

test "getPath navigation" {
    const input = "{\"a\":{\"b\":[1,2,3]}}";
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, input, .{});
    defer parsed.deinit();

    const val = getPath(parsed.value, "a.b.1") orelse unreachable;
    try std.testing.expectEqual(@as(i64, 2), val.integer);
}

test "JsonWriter handles deep nesting" {
    var buf = std.ArrayList(u8).init(std.testing.allocator);
    defer buf.deinit();

    var jw = jsonWriter(buf.writer());

    // Nest to a reasonable depth
    for (0..100) |_| {
        try jw.beginObject();
        try jw.field("nested");
    }
    try jw.valueNull();
    for (0..100) |_| {
        try jw.endObject();
    }
}

test "JsonWriter rejects excessive depth" {
    var buf = std.ArrayList(u8).init(std.testing.allocator);
    defer buf.deinit();

    var jw = jsonWriter(buf.writer());

    // Try to exceed max depth
    var i: usize = 0;
    while (i < 300) : (i += 1) {
        jw.beginObject() catch |err| {
            try std.testing.expectEqual(error.Overflow, err);
            return; // Test passes
        };
        jw.field("x") catch return;
    }
    // Should not reach here
    return error.TestUnexpectedResult;
}
