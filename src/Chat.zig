const std = @import("std");
const Allocator = std.mem.Allocator;
const Provider = @import("Provider.zig");
const types = @import("types.zig");
const ProviderError = @import("errors.zig").ProviderError;
const retry = @import("retry.zig");

const Chat = @This();

allocator: Allocator,
provider: Provider,
model: []const u8,
system_prompt: ?[]const u8 = null,
tools: []const types.ToolDefinition = &.{},
max_tokens: u32 = 4096,
temperature: ?f32 = null,
top_p: ?f32 = null,
top_k: ?u32 = null,
frequency_penalty: ?f32 = null,
presence_penalty: ?f32 = null,
retry_config: ?retry.RetryConfig = null,
history: std.ArrayList(types.Message),
total_usage: types.TokenUsage = .{},

pub fn init(allocator: Allocator, prov: Provider, model: []const u8) Chat {
    return .{
        .allocator = allocator,
        .provider = prov,
        .model = model,
        .history = .{},
    };
}

pub fn deinit(self: *Chat) void {
    for (self.history.items) |msg| {
        msg.deinit();
    }
    self.history.deinit(self.allocator);
}

/// Send a text message and get a completion response.
/// The user message and assistant response are added to history.
pub fn send(self: *Chat, text: []const u8) ProviderError!Provider.CompletionResponse {
    // Add user message to history
    try self.appendUserMessage(text);
    errdefer _ = self.history.pop();

    // Build request
    const request = self.buildRequest();

    // Get completion (with retry if configured)
    const response = try self.completeWithRetry(request);

    // Track usage
    self.total_usage = self.total_usage.add(response.usage);

    // Clone assistant message for history
    const history_msg = cloneMessage(response.message, self.allocator) catch return error.OutOfMemory;
    self.history.append(self.allocator, history_msg) catch return error.OutOfMemory;

    return response;
}

/// Send a text message and get a managed streaming response.
/// The user message is added to history. The assistant response is automatically
/// appended to history when the stream is fully consumed (on `message_stop`),
/// or when the `StreamResponse` is deinitialized.
pub fn sendStreaming(self: *Chat, text: []const u8) ProviderError!StreamResponse {
    try self.appendUserMessage(text);
    errdefer _ = self.history.pop();

    const request = self.buildRequest();
    const iterator = try self.streamWithRetry(request);

    return StreamResponse.init(self, iterator, self.allocator);
}

/// A managed streaming response that wraps a `StreamIterator` and automatically
/// accumulates content, then appends the assistant message to chat history when done.
pub const StreamResponse = struct {
    chat: *Chat,
    iterator: Provider.StreamIterator,
    allocator: Allocator,
    // Accumulation state
    text_buf: std.ArrayList(u8),
    tool_calls: std.ArrayList(AccumulatedToolCall),
    finished: bool,
    usage: types.TokenUsage,
    stop_reason: types.StopReason,

    pub const AccumulatedToolCall = struct {
        id: []const u8,
        name: []const u8,
        input_buf: std.ArrayList(u8),

        pub fn deinit(self: *AccumulatedToolCall, allocator: Allocator) void {
            allocator.free(self.id);
            allocator.free(self.name);
            self.input_buf.deinit(allocator);
        }
    };

    pub fn init(chat: *Chat, iterator: Provider.StreamIterator, allocator: Allocator) StreamResponse {
        return .{
            .chat = chat,
            .iterator = iterator,
            .allocator = allocator,
            .text_buf = .{},
            .tool_calls = .{},
            .finished = false,
            .usage = .{},
            .stop_reason = .unknown,
        };
    }

    /// Get the next stream event. Also accumulates content internally.
    /// The caller should still call `event.deinit(allocator)` on each returned event.
    pub fn next(self: *StreamResponse) ProviderError!?types.StreamEvent {
        const event = (try self.iterator.next()) orelse return null;

        // Accumulate based on event type
        switch (event) {
            .text_delta => |td| {
                self.text_buf.appendSlice(self.allocator, td.text) catch {};
            },
            .tool_use_start => |tu| {
                self.tool_calls.append(self.allocator, .{
                    .id = self.allocator.dupe(u8, tu.id) catch return error.OutOfMemory,
                    .name = self.allocator.dupe(u8, tu.name) catch return error.OutOfMemory,
                    .input_buf = .{},
                }) catch return error.OutOfMemory;
            },
            .tool_input_delta => |tid| {
                if (self.tool_calls.items.len > 0) {
                    var last = &self.tool_calls.items[self.tool_calls.items.len - 1];
                    last.input_buf.appendSlice(self.allocator, tid.json) catch {};
                }
            },
            .message_delta => |md| {
                if (md.usage) |u| self.usage = u;
                if (md.stop_reason) |sr| self.stop_reason = types.StopReason.fromString(sr);
            },
            .message_stop => {
                if (!self.finished) {
                    self.finished = true;
                    self.finalizeHistory() catch {};
                }
            },
            else => {},
        }

        return event;
    }

    fn finalizeHistory(self: *StreamResponse) !void {
        // Count how many content blocks we need
        const text_count: usize = if (self.text_buf.items.len > 0) 1 else 0;
        const total_blocks = text_count + self.tool_calls.items.len;

        if (total_blocks == 0) return;

        var content = self.allocator.alloc(types.ContentBlock, total_blocks) catch return;
        var idx: usize = 0;
        errdefer {
            // Free any blocks we already populated before the failure
            for (content[0..idx]) |block| block.deinit(self.allocator);
            self.allocator.free(content);
        }

        // Add text block if we accumulated any text
        if (self.text_buf.items.len > 0) {
            content[idx] = .{ .text = .{
                .text = self.allocator.dupe(u8, self.text_buf.items) catch return,
            } };
            idx += 1;
        }

        // Add tool_use blocks
        for (self.tool_calls.items) |tc| {
            const id = self.allocator.dupe(u8, tc.id) catch return;
            errdefer self.allocator.free(id);
            const name = self.allocator.dupe(u8, tc.name) catch return;
            errdefer self.allocator.free(name);
            const input_json = self.allocator.dupe(u8, tc.input_buf.items) catch return;

            content[idx] = .{ .tool_use = .{
                .id = id,
                .name = name,
                .input_json = input_json,
            } };
            idx += 1;
        }

        // Update usage on the chat
        self.chat.total_usage = self.chat.total_usage.add(self.usage);

        // Append assistant message to history
        self.chat.history.append(self.allocator, .{
            .role = .assistant,
            .content = content,
            .allocator = self.allocator,
        }) catch return;
    }

    /// Thread-safe: forwards to the underlying iterator's transport-level abort.
    /// A concurrent `next()` call will return shortly with an error or null.
    pub fn abort(self: *StreamResponse) void {
        self.iterator.abort();
    }

    pub fn deinit(self: *StreamResponse) void {
        // If stream wasn't fully consumed, still try to finalize
        if (!self.finished) {
            // Drain remaining events to finalize
            while (self.next() catch null) |event| {
                const is_stop = std.meta.activeTag(event) == .message_stop;
                event.deinit(self.allocator);
                if (is_stop) break;
            }
        }

        // Clean up accumulated state
        self.text_buf.deinit(self.allocator);
        for (self.tool_calls.items) |*tc| {
            tc.deinit(self.allocator);
        }
        self.tool_calls.deinit(self.allocator);

        // Clean up the underlying iterator
        self.iterator.deinit();
    }
};

/// The result of a tool invocation, returned by the handler passed to sendWithTools.
pub const ToolResult = struct {
    content: []const u8,
    is_error: bool = false,
};

/// Send a message and automatically handle any tool calls.
/// When the model requests multiple tool calls, they run in parallel on
/// separate threads. Uses the same context + comptime fn pattern as std.sort.
///
/// For environments without thread support (freestanding, WASI) or when
/// deterministic sequential execution is preferred, use `sendWithToolsSeq`.
pub fn sendWithTools(
    self: *Chat,
    text: []const u8,
    context: anytype,
    comptime handleToolCall: fn (@TypeOf(context), []const u8, []const u8) ToolResult,
) ProviderError!Provider.CompletionResponse {
    return self.sendWithToolsImpl(true, text, context, handleToolCall);
}

/// Like `sendWithTools`, but executes tool calls sequentially.
/// Use this on targets without thread support or when you need deterministic
/// single-threaded execution.
pub fn sendWithToolsSeq(
    self: *Chat,
    text: []const u8,
    context: anytype,
    comptime handleToolCall: fn (@TypeOf(context), []const u8, []const u8) ToolResult,
) ProviderError!Provider.CompletionResponse {
    return self.sendWithToolsImpl(false, text, context, handleToolCall);
}

fn sendWithToolsImpl(
    self: *Chat,
    comptime parallel: bool,
    text: []const u8,
    context: anytype,
    comptime handleToolCall: fn (@TypeOf(context), []const u8, []const u8) ToolResult,
) ProviderError!Provider.CompletionResponse {
    var response = try self.send(text);

    var rounds: usize = 0;
    while (response.stop_reason == .tool_use) : (rounds += 1) {
        if (rounds >= max_tool_rounds) {
            response.deinit();
            return error.InvalidRequest;
        }

        // Collect all tool calls. We dupe the ids (needed after response.deinit()),
        // but name and input_json borrow from the response and stay valid until
        // all handlers have returned.
        var calls: std.ArrayList(PendingToolCall) = .{};
        defer {
            for (calls.items) |c| self.allocator.free(c.id);
            calls.deinit(self.allocator);
        }

        var tool_iter = response.message.toolUseBlocks();
        while (tool_iter.next()) |tc| {
            calls.append(self.allocator, .{
                .id = self.allocator.dupe(u8, tc.id) catch return error.OutOfMemory,
                .name = tc.name,
                .input_json = tc.input_json,
                .result = .{ .content = "" },
            }) catch return error.OutOfMemory;
        }

        if (parallel) {
            // Spawn a thread per call (except the last, which runs on the current thread).
            const Dispatch = struct {
                fn run(ctx: @TypeOf(context), call: *PendingToolCall) void {
                    call.result = handleToolCall(ctx, call.name, call.input_json);
                }
            };

            var threads: std.ArrayList(std.Thread) = .{};
            defer threads.deinit(self.allocator);

            for (calls.items[0 .. calls.items.len -| 1]) |*c| {
                const t = std.Thread.spawn(.{}, Dispatch.run, .{ context, c }) catch {
                    c.result = handleToolCall(context, c.name, c.input_json);
                    continue;
                };
                threads.append(self.allocator, t) catch {
                    t.join();
                    continue;
                };
            }

            if (calls.items.len > 0) {
                const last = &calls.items[calls.items.len - 1];
                last.result = handleToolCall(context, last.name, last.input_json);
            }

            for (threads.items) |t| t.join();
        } else {
            for (calls.items) |*c| {
                c.result = handleToolCall(context, c.name, c.input_json);
            }
        }

        response.deinit();

        try self.appendToolCallResults(calls.items);
        response = try self.completeAndRecord();
    }

    return response;
}

const max_tool_rounds = 20;

const PendingToolCall = struct {
    id: []const u8, // duped, freed by the ArrayList cleanup in sendWithTools
    name: []const u8, // borrowed from response (valid until response.deinit)
    input_json: []const u8, // borrowed from response
    result: ToolResult, // filled in by handler
};

/// Send a tool result and get a completion response.
pub fn sendToolResult(self: *Chat, tool_use_id: []const u8, content: []const u8, is_error: bool) ProviderError!Provider.CompletionResponse {
    try self.appendToolResult(tool_use_id, content, is_error);
    errdefer _ = self.history.pop();

    const request = self.buildRequest();

    const response = try self.completeWithRetry(request);
    self.total_usage = self.total_usage.add(response.usage);

    const history_msg = cloneMessage(response.message, self.allocator) catch return error.OutOfMemory;
    self.history.append(self.allocator, history_msg) catch return error.OutOfMemory;

    return response;
}

/// Add an assembled assistant message to history (e.g., after consuming a stream).
pub fn appendAssistantMessage(self: *Chat, message: types.Message) ProviderError!void {
    const cloned = cloneMessage(message, self.allocator) catch return error.OutOfMemory;
    self.history.append(self.allocator, cloned) catch return error.OutOfMemory;
}

/// Clear conversation history.
pub fn clearHistory(self: *Chat) void {
    for (self.history.items) |msg| {
        msg.deinit();
    }
    self.history.clearRetainingCapacity();
}

// --- Internal ---

fn buildRequest(self: *Chat) Provider.CompletionRequest {
    return .{
        .model = self.model,
        .messages = self.history.items,
        .system_prompt = self.system_prompt,
        .tools = self.tools,
        .max_tokens = self.max_tokens,
        .temperature = self.temperature,
        .top_p = self.top_p,
        .top_k = self.top_k,
        .frequency_penalty = self.frequency_penalty,
        .presence_penalty = self.presence_penalty,
    };
}

fn appendUserMessage(self: *Chat, text: []const u8) ProviderError!void {
    var content = self.allocator.alloc(types.ContentBlock, 1) catch return error.OutOfMemory;
    content[0] = .{ .text = .{ .text = self.allocator.dupe(u8, text) catch return error.OutOfMemory } };
    self.history.append(self.allocator, .{
        .role = .user,
        .content = content,
        .allocator = self.allocator,
    }) catch return error.OutOfMemory;
}

fn appendToolCallResults(self: *Chat, calls: []const PendingToolCall) ProviderError!void {
    var blocks = self.allocator.alloc(types.ContentBlock, calls.len) catch return error.OutOfMemory;
    for (calls, 0..) |c, i| {
        blocks[i] = .{ .tool_result = .{
            .tool_use_id = self.allocator.dupe(u8, c.id) catch return error.OutOfMemory,
            .content = self.allocator.dupe(u8, c.result.content) catch return error.OutOfMemory,
            .is_error = c.result.is_error,
        } };
    }
    self.history.append(self.allocator, .{
        .role = .tool,
        .content = blocks,
        .allocator = self.allocator,
    }) catch return error.OutOfMemory;
}

/// Complete against current history and record the assistant response.
fn completeAndRecord(self: *Chat) ProviderError!Provider.CompletionResponse {
    const request = self.buildRequest();
    const response = try self.completeWithRetry(request);
    self.total_usage = self.total_usage.add(response.usage);
    const history_msg = cloneMessage(response.message, self.allocator) catch return error.OutOfMemory;
    self.history.append(self.allocator, history_msg) catch return error.OutOfMemory;
    return response;
}

/// Execute a completion request with retry logic for transient errors.
fn completeWithRetry(self: *Chat, request: Provider.CompletionRequest) ProviderError!Provider.CompletionResponse {
    const retry_cfg = self.retry_config orelse return self.provider.complete(request, self.allocator);

    var attempt: u32 = 0;
    while (true) {
        const result = self.provider.complete(request, self.allocator);
        if (result) |response| {
            return response;
        } else |err| {
            if (!retry.isRetryable(err) or attempt >= retry_cfg.max_retries) {
                return err;
            }
            retry.sleepMs(retry.computeDelay(retry_cfg, attempt));
            attempt += 1;
        }
    }
}

/// Execute a streaming request with retry logic for transient errors.
/// Note: only the initial connection is retried. Once streaming begins,
/// errors during iteration are not retried at this level.
fn streamWithRetry(self: *Chat, request: Provider.CompletionRequest) ProviderError!Provider.StreamIterator {
    const retry_cfg = self.retry_config orelse return self.provider.stream(request, self.allocator);

    var attempt: u32 = 0;
    while (true) {
        const result = self.provider.stream(request, self.allocator);
        if (result) |iter| {
            return iter;
        } else |err| {
            if (!retry.isRetryable(err) or attempt >= retry_cfg.max_retries) {
                return err;
            }
            retry.sleepMs(retry.computeDelay(retry_cfg, attempt));
            attempt += 1;
        }
    }
}

fn appendToolResult(self: *Chat, tool_use_id: []const u8, content: []const u8, is_error: bool) ProviderError!void {
    var blocks = self.allocator.alloc(types.ContentBlock, 1) catch return error.OutOfMemory;
    blocks[0] = .{ .tool_result = .{
        .tool_use_id = self.allocator.dupe(u8, tool_use_id) catch return error.OutOfMemory,
        .content = self.allocator.dupe(u8, content) catch return error.OutOfMemory,
        .is_error = is_error,
    } };
    self.history.append(self.allocator, .{
        .role = .tool,
        .content = blocks,
        .allocator = self.allocator,
    }) catch return error.OutOfMemory;
}

fn cloneMessage(msg: types.Message, allocator: Allocator) Allocator.Error!types.Message {
    var content = try allocator.alloc(types.ContentBlock, msg.content.len);
    errdefer allocator.free(content);

    for (msg.content, 0..) |block, i| {
        content[i] = try cloneContentBlock(block, allocator);
    }

    return .{
        .role = msg.role,
        .content = content,
        .allocator = allocator,
    };
}

fn cloneContentBlock(block: types.ContentBlock, allocator: Allocator) Allocator.Error!types.ContentBlock {
    return switch (block) {
        .text => |t| .{ .text = .{ .text = try allocator.dupe(u8, t.text) } },
        .image => |img| .{ .image = .{
            .source_type = img.source_type,
            .data = try allocator.dupe(u8, img.data),
            .media_type = if (img.media_type) |mt| try allocator.dupe(u8, mt) else null,
        } },
        .tool_use => |tu| .{ .tool_use = .{
            .id = try allocator.dupe(u8, tu.id),
            .name = try allocator.dupe(u8, tu.name),
            .input_json = try allocator.dupe(u8, tu.input_json),
        } },
        .tool_result => |tr| .{ .tool_result = .{
            .tool_use_id = try allocator.dupe(u8, tr.tool_use_id),
            .content = try allocator.dupe(u8, tr.content),
            .is_error = tr.is_error,
        } },
        .thinking => |th| .{ .thinking = .{ .text = try allocator.dupe(u8, th.text) } },
    };
}

// --- Tests ---

const mock = @import("testing/mock_provider.zig");

test "Chat send and receive" {
    const allocator = std.testing.allocator;

    var mp = try mock.MockProvider.init(allocator);
    defer mp.deinit();

    var chat = Chat.init(allocator, mp.provider(), "test-model");
    defer chat.deinit();
    chat.system_prompt = "You are helpful.";

    var response = try chat.send("Hello!");
    defer response.deinit();

    try std.testing.expectEqualStrings("This is a mock response.", response.text().?);
    try std.testing.expectEqual(@as(usize, 2), chat.history.items.len); // user + assistant
    try std.testing.expectEqual(types.Role.user, chat.history.items[0].role);
    try std.testing.expectEqual(types.Role.assistant, chat.history.items[1].role);
}

test "Chat clearHistory" {
    const allocator = std.testing.allocator;

    var mp = try mock.MockProvider.init(allocator);
    defer mp.deinit();

    var chat = Chat.init(allocator, mp.provider(), "test-model");
    defer chat.deinit();

    var resp = try chat.send("Hello!");
    resp.deinit();
    try std.testing.expectEqual(@as(usize, 2), chat.history.items.len);

    chat.clearHistory();
    try std.testing.expectEqual(@as(usize, 0), chat.history.items.len);
}

test "Chat sendWithTools handles tool loop" {
    const allocator = std.testing.allocator;

    var mp = try mock.MockProvider.init(allocator);
    defer mp.deinit();
    mp.tool_call_on_first = .{
        .tool_id = "call_1",
        .tool_name = "get_weather",
        .input_json = "{\"city\":\"London\"}",
    };

    var chat = Chat.init(allocator, mp.provider(), "test-model");
    defer chat.deinit();
    chat.tools = &.{.{
        .name = "get_weather",
        .description = "Get weather",
        .input_schema = "{}",
    }};

    var response = try chat.sendWithTools("What's the weather?", {}, struct {
        fn call(_: void, name: []const u8, _: []const u8) ToolResult {
            if (std.mem.eql(u8, name, "get_weather")) {
                return .{ .content = "{\"temp\": 15}" };
            }
            return .{ .content = "unknown tool", .is_error = true };
        }
    }.call);
    defer response.deinit();

    // Should have completed the tool loop and returned text
    try std.testing.expectEqual(types.StopReason.end_turn, response.stop_reason);
    try std.testing.expectEqualStrings("This is a mock response.", response.text().?);
    // Provider was called twice: first returned tool_use, second returned text
    try std.testing.expectEqual(@as(usize, 2), mp.call_count);
    // History: user msg, assistant tool_use, user tool_result, assistant text
    try std.testing.expectEqual(@as(usize, 4), chat.history.items.len);
}

test "Chat sendWithTools runs parallel tool calls" {
    const allocator = std.testing.allocator;

    var mp = try mock.MockProvider.init(allocator);
    defer mp.deinit();
    mp.tool_call_on_first = .{
        .tool_id = "call_1",
        .tool_name = "get_weather",
        .input_json = "{\"city\":\"London\"}",
    };
    mp.extra_tool_calls = &.{
        .{ .tool_id = "call_2", .tool_name = "get_weather", .input_json = "{\"city\":\"Paris\"}" },
        .{ .tool_id = "call_3", .tool_name = "get_weather", .input_json = "{\"city\":\"Tokyo\"}" },
    };

    var chat = Chat.init(allocator, mp.provider(), "test-model");
    defer chat.deinit();
    chat.tools = &.{.{
        .name = "get_weather",
        .description = "Get weather",
        .input_schema = "{}",
    }};

    // Track which tools were called using a shared counter
    var call_counter = std.atomic.Value(u32).init(0);

    var response = try chat.sendWithTools("Weather in London, Paris, and Tokyo?", &call_counter, struct {
        fn call(counter: *std.atomic.Value(u32), name: []const u8, _: []const u8) ToolResult {
            if (std.mem.eql(u8, name, "get_weather")) {
                _ = counter.fetchAdd(1, .monotonic);
                return .{ .content = "{\"temp\": 15}" };
            }
            return .{ .content = "unknown tool", .is_error = true };
        }
    }.call);
    defer response.deinit();

    // All 3 tool calls should have been dispatched
    try std.testing.expectEqual(@as(u32, 3), call_counter.load(.monotonic));
    // Final response should be text
    try std.testing.expectEqual(types.StopReason.end_turn, response.stop_reason);
    // History: user, assistant(3 tool_use), user(3 tool_result), assistant(text)
    try std.testing.expectEqual(@as(usize, 4), chat.history.items.len);
    // The tool_result message should have 3 content blocks
    try std.testing.expectEqual(@as(usize, 3), chat.history.items[2].content.len);
}

test "Chat retry config" {
    const allocator = std.testing.allocator;

    var mp = try mock.MockProvider.init(allocator);
    defer mp.deinit();

    var chat = Chat.init(allocator, mp.provider(), "test-model");
    defer chat.deinit();

    // Retry is off by default
    try std.testing.expect(chat.retry_config == null);

    // Can be enabled
    chat.retry_config = .{ .max_retries = 5 };
    try std.testing.expectEqual(@as(u32, 5), chat.retry_config.?.max_retries);

    // Other fields have defaults
    try std.testing.expectEqual(@as(u64, 1000), chat.retry_config.?.initial_delay_ms);
    try std.testing.expectEqual(@as(u64, 30000), chat.retry_config.?.max_delay_ms);
    try std.testing.expectEqual(@as(u32, 2), chat.retry_config.?.backoff_multiplier);
}

test "Chat tracks cumulative usage" {
    const allocator = std.testing.allocator;

    var mp = try mock.MockProvider.init(allocator);
    defer mp.deinit();

    var chat = Chat.init(allocator, mp.provider(), "test-model");
    defer chat.deinit();

    var r1 = try chat.send("First");
    r1.deinit();
    var r2 = try chat.send("Second");
    r2.deinit();

    try std.testing.expectEqual(@as(u32, 20), chat.total_usage.input_tokens);
    try std.testing.expectEqual(@as(u32, 10), chat.total_usage.output_tokens);
}

test "StreamResponse automatically appends history after stream completes" {
    const allocator = std.testing.allocator;

    var mp = try mock.MockProvider.init(allocator);
    defer mp.deinit();

    var chat = Chat.init(allocator, mp.provider(), "test-model");
    defer chat.deinit();

    // sendStreaming returns a StreamResponse
    var stream = try chat.sendStreaming("Hello streaming!");
    defer stream.deinit();

    // At this point, only the user message is in history
    try std.testing.expectEqual(@as(usize, 1), chat.history.items.len);
    try std.testing.expectEqual(types.Role.user, chat.history.items[0].role);

    // Consume the stream fully
    var full_text: std.ArrayList(u8) = .{};
    defer full_text.deinit(allocator);

    while (try stream.next()) |event| {
        switch (event) {
            .text_delta => |td| {
                try full_text.appendSlice(allocator, td.text);
                event.deinit(allocator);
            },
            .message_stop => {
                // message_stop doesn't need deinit, but check history was updated
                break;
            },
            else => event.deinit(allocator),
        }
    }

    // The streamed text should match
    try std.testing.expectEqualStrings("This is a mock response.", full_text.items);

    // History should now have both user and assistant messages
    try std.testing.expectEqual(@as(usize, 2), chat.history.items.len);
    try std.testing.expectEqual(types.Role.user, chat.history.items[0].role);
    try std.testing.expectEqual(types.Role.assistant, chat.history.items[1].role);

    // The assistant message text in history should match
    try std.testing.expectEqualStrings(
        "This is a mock response.",
        chat.history.items[1].text().?,
    );

    // Usage should be tracked
    try std.testing.expectEqual(@as(u32, 10), chat.total_usage.input_tokens);
    try std.testing.expectEqual(@as(u32, 5), chat.total_usage.output_tokens);

    // Stop reason should be captured
    try std.testing.expectEqual(types.StopReason.end_turn, stream.stop_reason);
}

test "StreamResponse deinit finalizes if stream not fully consumed" {
    const allocator = std.testing.allocator;

    var mp = try mock.MockProvider.init(allocator);
    defer mp.deinit();

    var chat = Chat.init(allocator, mp.provider(), "test-model");
    defer chat.deinit();

    {
        var stream = try chat.sendStreaming("Hello!");
        // Read only the first event (message_start), then let deinit handle the rest
        const event = (try stream.next()).?;
        event.deinit(allocator);
        // deinit drains remaining events and finalizes history
        stream.deinit();
    }

    // History should have both user and assistant messages
    try std.testing.expectEqual(@as(usize, 2), chat.history.items.len);
    try std.testing.expectEqual(types.Role.user, chat.history.items[0].role);
    try std.testing.expectEqual(types.Role.assistant, chat.history.items[1].role);
    try std.testing.expectEqualStrings(
        "This is a mock response.",
        chat.history.items[1].text().?,
    );
}
