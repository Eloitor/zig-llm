const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Role = enum {
    system,
    user,
    assistant,
    tool,

    pub fn toString(self: Role) []const u8 {
        return switch (self) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
            .tool => "tool",
        };
    }
};

pub const ContentBlock = union(enum) {
    text: TextBlock,
    image: ImageBlock,
    tool_use: ToolUseBlock,
    tool_result: ToolResultBlock,
    thinking: ThinkingBlock,

    pub fn deinit(self: ContentBlock, allocator: Allocator) void {
        switch (self) {
            .text => |b| allocator.free(b.text),
            .image => |b| {
                allocator.free(b.data);
                if (b.media_type) |mt| allocator.free(mt);
            },
            .tool_use => |b| {
                allocator.free(b.id);
                allocator.free(b.name);
                allocator.free(b.input_json);
            },
            .tool_result => |b| {
                allocator.free(b.tool_use_id);
                allocator.free(b.content);
            },
            .thinking => |b| allocator.free(b.text),
        }
    }
};

pub const TextBlock = struct {
    text: []const u8,
};

pub const ImageBlock = struct {
    source_type: ImageSourceType,
    data: []const u8,
    media_type: ?[]const u8,

    pub const ImageSourceType = enum { base64, url };
};

pub const ToolUseBlock = struct {
    id: []const u8,
    name: []const u8,
    input_json: []const u8,
};

pub const ToolResultBlock = struct {
    tool_use_id: []const u8,
    content: []const u8,
    is_error: bool,
};

pub const ThinkingBlock = struct {
    text: []const u8,
};

pub const Message = struct {
    role: Role,
    content: []ContentBlock,
    allocator: Allocator,

    pub fn text(self: Message) ?[]const u8 {
        for (self.content) |block| {
            switch (block) {
                .text => |t| return t.text,
                else => {},
            }
        }
        return null;
    }

    pub fn hasToolUse(self: Message) bool {
        for (self.content) |block| {
            switch (block) {
                .tool_use => return true,
                else => {},
            }
        }
        return false;
    }

    pub fn toolUseBlocks(self: Message) ToolUseIterator {
        return .{ .content = self.content, .index = 0 };
    }

    pub const ToolUseIterator = struct {
        content: []const ContentBlock,
        index: usize,

        pub fn next(self: *ToolUseIterator) ?ToolUseBlock {
            while (self.index < self.content.len) {
                const block = self.content[self.index];
                self.index += 1;
                switch (block) {
                    .tool_use => |tu| return tu,
                    else => {},
                }
            }
            return null;
        }
    };

    pub fn deinit(self: Message) void {
        for (self.content) |block| {
            block.deinit(self.allocator);
        }
        self.allocator.free(self.content);
    }
};

pub const StopReason = enum {
    end_turn,
    tool_use,
    max_tokens,
    stop_sequence,
    unknown,

    pub fn fromString(s: []const u8) StopReason {
        if (std.mem.eql(u8, s, "end_turn")) return .end_turn;
        if (std.mem.eql(u8, s, "tool_use")) return .tool_use;
        if (std.mem.eql(u8, s, "max_tokens")) return .max_tokens;
        if (std.mem.eql(u8, s, "stop_sequence")) return .stop_sequence;
        return .unknown;
    }

    pub fn toString(self: StopReason) []const u8 {
        return switch (self) {
            .end_turn => "end_turn",
            .tool_use => "tool_use",
            .max_tokens => "max_tokens",
            .stop_sequence => "stop_sequence",
            .unknown => "unknown",
        };
    }
};

pub const TokenUsage = struct {
    input_tokens: u32 = 0,
    output_tokens: u32 = 0,
    cache_read_tokens: u32 = 0,
    cache_creation_tokens: u32 = 0,

    pub fn total(self: TokenUsage) u32 {
        return self.input_tokens + self.output_tokens;
    }

    pub fn add(self: TokenUsage, other: TokenUsage) TokenUsage {
        return .{
            .input_tokens = self.input_tokens + other.input_tokens,
            .output_tokens = self.output_tokens + other.output_tokens,
            .cache_read_tokens = self.cache_read_tokens + other.cache_read_tokens,
            .cache_creation_tokens = self.cache_creation_tokens + other.cache_creation_tokens,
        };
    }
};

pub const ToolDefinition = struct {
    name: []const u8,
    description: []const u8,
    input_schema: []const u8, // raw JSON string
};

pub const StreamEvent = union(enum) {
    message_start: MessageStartEvent,
    text_delta: TextDeltaEvent,
    thinking_delta: ThinkingDeltaEvent,
    tool_use_start: ToolUseStartEvent,
    tool_input_delta: ToolInputDeltaEvent,
    content_block_stop: void,
    message_delta: MessageDeltaEvent,
    message_stop: void,

    pub fn deinit(self: StreamEvent, allocator: Allocator) void {
        switch (self) {
            .message_start => |e| {
                if (e.message_id) |id| allocator.free(id);
                if (e.model) |m| allocator.free(m);
            },
            .text_delta => |e| allocator.free(e.text),
            .thinking_delta => |e| allocator.free(e.text),
            .tool_use_start => |e| {
                allocator.free(e.id);
                allocator.free(e.name);
            },
            .tool_input_delta => |e| allocator.free(e.json),
            .message_delta => |e| {
                if (e.stop_reason) |sr| allocator.free(sr);
            },
            .content_block_stop, .message_stop => {},
        }
    }
};

pub const MessageStartEvent = struct {
    message_id: ?[]const u8,
    model: ?[]const u8,
};

pub const TextDeltaEvent = struct {
    text: []const u8,
};

pub const ThinkingDeltaEvent = struct {
    text: []const u8,
};

pub const ToolUseStartEvent = struct {
    id: []const u8,
    name: []const u8,
};

pub const ToolInputDeltaEvent = struct {
    json: []const u8,
};

pub const MessageDeltaEvent = struct {
    stop_reason: ?[]const u8,
    usage: ?TokenUsage,
};

// --- Tests ---

test "Role.toString" {
    try std.testing.expectEqualStrings("user", Role.user.toString());
    try std.testing.expectEqualStrings("assistant", Role.assistant.toString());
}

test "StopReason.fromString" {
    try std.testing.expectEqual(StopReason.end_turn, StopReason.fromString("end_turn"));
    try std.testing.expectEqual(StopReason.tool_use, StopReason.fromString("tool_use"));
    try std.testing.expectEqual(StopReason.unknown, StopReason.fromString("something_else"));
}

test "TokenUsage.add and total" {
    const a = TokenUsage{ .input_tokens = 10, .output_tokens = 20 };
    const b = TokenUsage{ .input_tokens = 5, .output_tokens = 15 };
    const c = a.add(b);
    try std.testing.expectEqual(@as(u32, 15), c.input_tokens);
    try std.testing.expectEqual(@as(u32, 35), c.output_tokens);
    try std.testing.expectEqual(@as(u32, 50), c.total());
}

test "Message.text returns first text block" {
    const allocator = std.testing.allocator;
    var content = try allocator.alloc(ContentBlock, 2);
    content[0] = .{ .thinking = .{ .text = "hmm" } };
    content[1] = .{ .text = .{ .text = "hello" } };

    const msg = Message{
        .role = .assistant,
        .content = content,
        .allocator = allocator,
    };
    // Don't call deinit since we used stack strings
    defer allocator.free(content);

    try std.testing.expectEqualStrings("hello", msg.text().?);
}

test "Message.hasToolUse" {
    const allocator = std.testing.allocator;
    var content = try allocator.alloc(ContentBlock, 1);
    content[0] = .{ .tool_use = .{ .id = "id", .name = "bash", .input_json = "{}" } };

    const msg = Message{
        .role = .assistant,
        .content = content,
        .allocator = allocator,
    };
    defer allocator.free(content);

    try std.testing.expect(msg.hasToolUse());
}

test "StreamEvent.deinit frees allocations" {
    const allocator = std.testing.allocator;
    const text = try allocator.dupe(u8, "hello");
    var event = StreamEvent{ .text_delta = .{ .text = text } };
    event.deinit(allocator);
}
