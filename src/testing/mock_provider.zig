const std = @import("std");
const Allocator = std.mem.Allocator;
const Provider = @import("../Provider.zig");
const types = @import("../types.zig");
const ProviderError = @import("../errors.zig").ProviderError;

pub const MockProvider = struct {
    allocator: Allocator,
    response_text: []const u8,
    call_count: usize,
    /// If set, the first complete() call returns tool_use response(s) instead of text.
    tool_call_on_first: ?struct {
        tool_id: []const u8,
        tool_name: []const u8,
        input_json: []const u8,
    } = null,
    /// Additional tool calls returned alongside tool_call_on_first (for parallel tool testing).
    extra_tool_calls: []const struct {
        tool_id: []const u8,
        tool_name: []const u8,
        input_json: []const u8,
    } = &.{},

    pub fn init(allocator: Allocator) ProviderError!*MockProvider {
        const self = allocator.create(MockProvider) catch return error.OutOfMemory;
        self.* = .{
            .allocator = allocator,
            .response_text = "This is a mock response.",
            .call_count = 0,
        };
        return self;
    }

    pub fn deinit(self: *MockProvider) void {
        self.allocator.destroy(self);
    }

    pub fn provider(self: *MockProvider) Provider {
        return Provider.init(self);
    }

    pub fn complete(self: *MockProvider, _: Provider.CompletionRequest, allocator: Allocator) ProviderError!Provider.CompletionResponse {
        self.call_count += 1;

        // On first call, return a tool_use response if configured
        if (self.call_count == 1) {
            if (self.tool_call_on_first) |tc| {
                const total = 1 + self.extra_tool_calls.len;
                var content = allocator.alloc(types.ContentBlock, total) catch return error.OutOfMemory;
                content[0] = .{ .tool_use = .{
                    .id = allocator.dupe(u8, tc.tool_id) catch return error.OutOfMemory,
                    .name = allocator.dupe(u8, tc.tool_name) catch return error.OutOfMemory,
                    .input_json = allocator.dupe(u8, tc.input_json) catch return error.OutOfMemory,
                } };
                for (self.extra_tool_calls, 1..) |extra, i| {
                    content[i] = .{ .tool_use = .{
                        .id = allocator.dupe(u8, extra.tool_id) catch return error.OutOfMemory,
                        .name = allocator.dupe(u8, extra.tool_name) catch return error.OutOfMemory,
                        .input_json = allocator.dupe(u8, extra.input_json) catch return error.OutOfMemory,
                    } };
                }
                return .{
                    .message = .{ .role = .assistant, .content = content, .allocator = allocator },
                    .usage = .{ .input_tokens = 10, .output_tokens = 5 },
                    .model = allocator.dupe(u8, "mock-model") catch return error.OutOfMemory,
                    .stop_reason = .tool_use,
                    .allocator = allocator,
                };
            }
        }

        var content = allocator.alloc(types.ContentBlock, 1) catch return error.OutOfMemory;
        content[0] = .{ .text = .{ .text = allocator.dupe(u8, self.response_text) catch return error.OutOfMemory } };

        return .{
            .message = .{
                .role = .assistant,
                .content = content,
                .allocator = allocator,
            },
            .usage = .{ .input_tokens = 10, .output_tokens = 5 },
            .model = allocator.dupe(u8, "mock-model") catch return error.OutOfMemory,
            .stop_reason = .end_turn,
            .allocator = allocator,
        };
    }

    pub fn stream(self: *MockProvider, _: Provider.CompletionRequest, allocator: Allocator) ProviderError!Provider.StreamIterator {
        self.call_count += 1;

        const ctx = allocator.create(MockStreamContext) catch return error.OutOfMemory;
        ctx.* = .{
            .allocator = allocator,
            .response_text = self.response_text,
            .position = 0,
            .sent_start = false,
            .sent_delta = false,
            .sent_stop = false,
        };

        return Provider.StreamIterator.initFrom(ctx);
    }

    pub fn listModels(_: *MockProvider, allocator: Allocator) ProviderError![]const Provider.ModelInfo {
        const models = allocator.alloc(Provider.ModelInfo, 1) catch return error.OutOfMemory;
        models[0] = .{
            .id = "mock-model",
            .display_name = "Mock Model",
            .context_window = 4096,
            .max_output_tokens = 1024,
            .supports_tools = true,
            .supports_vision = false,
            .supports_streaming = true,
        };
        return models;
    }
};

const MockStreamContext = struct {
    allocator: Allocator,
    response_text: []const u8,
    position: usize,
    sent_start: bool,
    sent_delta: bool,
    sent_stop: bool,

    pub fn next(self: *MockStreamContext) ProviderError!?types.StreamEvent {
        if (!self.sent_start) {
            self.sent_start = true;
            return .{ .message_start = .{ .message_id = null, .model = null } };
        }

        // Send text in chunks of 5 chars
        if (self.position < self.response_text.len) {
            const end = @min(self.position + 5, self.response_text.len);
            const chunk = self.response_text[self.position..end];
            self.position = end;
            return .{ .text_delta = .{
                .text = self.allocator.dupe(u8, chunk) catch return error.OutOfMemory,
            } };
        }

        // Send message_delta with stop_reason and usage before message_stop
        if (!self.sent_delta) {
            self.sent_delta = true;
            return .{ .message_delta = .{
                .stop_reason = self.allocator.dupe(u8, "end_turn") catch return error.OutOfMemory,
                .usage = .{ .input_tokens = 10, .output_tokens = 5 },
            } };
        }

        if (!self.sent_stop) {
            self.sent_stop = true;
            return .message_stop;
        }

        return null;
    }

    pub fn deinit(self: *MockStreamContext) void {
        self.allocator.destroy(self);
    }
};

// --- Tests ---

test "MockProvider complete" {
    const allocator = std.testing.allocator;

    var mp = try MockProvider.init(allocator);
    defer mp.deinit();

    var prov = mp.provider();
    var response = try prov.complete(.{
        .model = "test",
        .messages = &.{},
    }, allocator);
    defer response.deinit();

    try std.testing.expectEqualStrings("This is a mock response.", response.text().?);
    try std.testing.expectEqual(@as(usize, 1), mp.call_count);
}

test "MockProvider stream" {
    const allocator = std.testing.allocator;

    var mp = try MockProvider.init(allocator);
    defer mp.deinit();

    var prov = mp.provider();
    var iter = try prov.stream(.{
        .model = "test",
        .messages = &.{},
    }, allocator);
    defer iter.deinit();

    // Should get message_start
    const e1 = (try iter.next()).?;
    try std.testing.expectEqual(std.meta.activeTag(e1), .message_start);
    e1.deinit(allocator);

    // Should get text deltas
    var full_text = std.ArrayList(u8).init(allocator);
    defer full_text.deinit();

    while (try iter.next()) |event| {
        switch (event) {
            .text_delta => |td| {
                try full_text.appendSlice(td.text);
                event.deinit(allocator);
            },
            .message_stop => break,
            else => {
                event.deinit(allocator);
            },
        }
    }

    try std.testing.expectEqualStrings("This is a mock response.", full_text.items);
}
