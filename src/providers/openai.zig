const std = @import("std");
const Allocator = std.mem.Allocator;
const Provider = @import("../Provider.zig");
const types = @import("../types.zig");
const ProviderError = @import("../errors.zig").ProviderError;
const http = @import("../http.zig");
const sse = @import("../sse.zig");
const jh = @import("../json_helpers.zig");

const OpenAI = @This();

allocator: Allocator,
api_key: []const u8,
api_base: []const u8,

pub const Config = struct {
    api_key: []const u8,
    api_base: ?[]const u8 = null,
};

const default_api_base = "https://api.openai.com";
const completions_path = "/v1/chat/completions";

pub fn init(allocator: Allocator, config: Config) ProviderError!*OpenAI {
    const self = allocator.create(OpenAI) catch return error.OutOfMemory;
    self.* = .{
        .allocator = allocator,
        .api_key = config.api_key,
        .api_base = config.api_base orelse default_api_base,
    };
    return self;
}

pub fn deinit(self: *OpenAI) void {
    self.allocator.destroy(self);
}

pub fn provider(self: *OpenAI) Provider {
    return Provider.init(self);
}

// --- Provider interface implementation ---

pub fn complete(self: *OpenAI, request: Provider.CompletionRequest, allocator: Allocator) ProviderError!Provider.CompletionResponse {
    const body = buildRequestBody(request, false, allocator) catch return error.OutOfMemory;
    defer allocator.free(body);

    var resp = try doPost(self, body, allocator);
    defer resp.deinit();

    if (http.mapStatusError(resp.status)) |err| {
        return mapApiError(resp.body) orelse err;
    }

    return parseCompletionResponse(resp.body, allocator);
}

pub fn stream(self: *OpenAI, request: Provider.CompletionRequest, allocator: Allocator) ProviderError!Provider.StreamIterator {
    const body = buildRequestBody(request, true, allocator) catch return error.OutOfMemory;
    defer allocator.free(body);

    const auth_value = std.fmt.allocPrint(allocator, "Bearer {s}", .{self.api_key}) catch return error.OutOfMemory;
    defer allocator.free(auth_value);

    const url = std.fmt.allocPrint(allocator, "{s}{s}", .{ self.api_base, completions_path }) catch return error.OutOfMemory;
    defer allocator.free(url);

    const http_stream = try http.openStream(allocator, url, &.{
        .{ .name = "Authorization", .value = auth_value },
    }, body);
    errdefer http_stream.deinit();

    const ctx = allocator.create(StreamContext) catch return error.OutOfMemory;
    ctx.* = .{
        .allocator = allocator,
        .http_stream = http_stream,
        .sse_reader = sse.SseLineReader.init(http_stream.reader(), allocator),
        .done = false,
        .sent_start = false,
        .finish_reason = null,
        .pending_usage = null,
        .pending_stop = false,
        .current_tool_calls = std.ArrayList(PendingToolCall).init(allocator),
    };

    return Provider.StreamIterator.initFrom(ctx);
}

pub fn listModels(_: *OpenAI, allocator: Allocator) ProviderError![]const Provider.ModelInfo {
    const models = allocator.alloc(Provider.ModelInfo, known_models.len) catch return error.OutOfMemory;
    @memcpy(models, &known_models);
    return models;
}

// --- HTTP ---

fn doPost(self: *OpenAI, body: []const u8, allocator: Allocator) ProviderError!http.HttpResponse {
    const auth_value = std.fmt.allocPrint(allocator, "Bearer {s}", .{self.api_key}) catch return error.OutOfMemory;
    defer allocator.free(auth_value);

    const url = std.fmt.allocPrint(allocator, "{s}{s}", .{ self.api_base, completions_path }) catch return error.OutOfMemory;
    defer allocator.free(url);

    return http.post(allocator, url, &.{
        .{ .name = "Authorization", .value = auth_value },
    }, body);
}

// --- Request building ---

fn buildRequestBody(request: Provider.CompletionRequest, do_stream: bool, allocator: Allocator) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    errdefer buf.deinit();
    var jw = jh.jsonWriter(buf.writer());

    try jw.beginObject();

    try jw.field("model");
    try jw.valueString(request.model);

    try jw.field("max_tokens");
    try jw.valueInt(@intCast(request.max_tokens));

    if (do_stream) {
        try jw.field("stream");
        try jw.valueBool(true);
        try jw.field("stream_options");
        try jw.beginObject();
        try jw.field("include_usage");
        try jw.valueBool(true);
        try jw.endObject();
    }

    if (request.temperature) |temp| {
        try jw.field("temperature");
        try jw.valueFloat(@floatCast(temp));
    }

    if (request.stop_sequences.len > 0) {
        try jw.field("stop");
        try jw.beginArray();
        for (request.stop_sequences) |seq| {
            try jw.valueString(seq);
        }
        try jw.endArray();
    }

    if (request.tools.len > 0) {
        try jw.field("tools");
        try jw.beginArray();
        for (request.tools) |tool| {
            try jw.beginObject();
            try jw.field("type");
            try jw.valueString("function");
            try jw.field("function");
            try jw.beginObject();
            try jw.field("name");
            try jw.valueString(tool.name);
            try jw.field("description");
            try jw.valueString(tool.description);
            try jw.field("parameters");
            try jw.valueRaw(tool.input_schema);
            try jw.endObject();
            try jw.endObject();
        }
        try jw.endArray();
    }

    // Messages — system prompt goes as first message
    try jw.field("messages");
    try jw.beginArray();

    if (request.system_prompt) |sys| {
        try jw.beginObject();
        try jw.field("role");
        try jw.valueString("system");
        try jw.field("content");
        try jw.valueString(sys);
        try jw.endObject();
    }

    for (request.messages) |msg| {
        if (msg.role == .system) continue;

        // OpenAI requires each tool result as a separate message
        if (msg.role == .tool) {
            for (msg.content) |block| {
                switch (block) {
                    .tool_result => |tr| {
                        try jw.beginObject();
                        try jw.field("role");
                        try jw.valueString("tool");
                        try jw.field("tool_call_id");
                        try jw.valueString(tr.tool_use_id);
                        try jw.field("content");
                        try jw.valueString(tr.content);
                        try jw.endObject();
                    },
                    else => {},
                }
            }
            continue;
        }

        try writeMessage(&jw, msg, allocator);
    }
    try jw.endArray();

    try jw.endObject();

    return buf.toOwnedSlice();
}

fn writeMessage(jw: anytype, msg: types.Message, allocator: Allocator) !void {
    try jw.beginObject();

    try jw.field("role");
    try jw.valueString(msg.role.toString());

    switch (msg.role) {
        .user => {
            // Check if there are any image blocks
            var has_images = false;
            for (msg.content) |block| {
                if (block == .image) {
                    has_images = true;
                    break;
                }
            }

            if (has_images) {
                try jw.field("content");
                try jw.beginArray();
                for (msg.content) |block| {
                    switch (block) {
                        .text => |t| {
                            try jw.beginObject();
                            try jw.field("type");
                            try jw.valueString("text");
                            try jw.field("text");
                            try jw.valueString(t.text);
                            try jw.endObject();
                        },
                        .image => |img| {
                            const data_url = std.fmt.allocPrint(allocator, "data:{s};base64,{s}", .{
                                img.media_type orelse "image/png",
                                img.data,
                            }) catch return error.OutOfMemory;
                            defer allocator.free(data_url);

                            try jw.beginObject();
                            try jw.field("type");
                            try jw.valueString("image_url");
                            try jw.field("image_url");
                            try jw.beginObject();
                            try jw.field("url");
                            try jw.valueString(data_url);
                            try jw.endObject();
                            try jw.endObject();
                        },
                        else => {},
                    }
                }
                try jw.endArray();
            } else {
                // Simple string content — concatenate all text blocks
                try jw.field("content");
                if (msg.content.len == 1) {
                    switch (msg.content[0]) {
                        .text => |t| try jw.valueString(t.text),
                        else => try jw.valueString(""),
                    }
                } else {
                    // Concatenate all text blocks
                    var combined = std.ArrayList(u8).init(allocator);
                    defer combined.deinit();
                    for (msg.content) |block| {
                        switch (block) {
                            .text => |t| {
                                try combined.appendSlice(t.text);
                            },
                            else => {},
                        }
                    }
                    if (combined.items.len > 0) {
                        try jw.valueString(combined.items);
                    } else {
                        try jw.valueString("");
                    }
                }
            }
        },
        .assistant => {
            // Text goes in content, tool_use blocks go in tool_calls array
            var has_tool_use = false;
            for (msg.content) |block| {
                if (block == .tool_use) {
                    has_tool_use = true;
                    break;
                }
            }

            // Write content (text)
            try jw.field("content");
            var found_text = false;
            for (msg.content) |block| {
                switch (block) {
                    .text => |t| {
                        if (!found_text) {
                            try jw.valueString(t.text);
                            found_text = true;
                        }
                    },
                    else => {},
                }
            }
            if (!found_text) try jw.valueNull();

            // Write tool_calls array if present
            if (has_tool_use) {
                try jw.field("tool_calls");
                try jw.beginArray();
                for (msg.content) |block| {
                    switch (block) {
                        .tool_use => |tu| {
                            try jw.beginObject();
                            try jw.field("id");
                            try jw.valueString(tu.id);
                            try jw.field("type");
                            try jw.valueString("function");
                            try jw.field("function");
                            try jw.beginObject();
                            try jw.field("name");
                            try jw.valueString(tu.name);
                            try jw.field("arguments");
                            try jw.valueString(tu.input_json);
                            try jw.endObject();
                            try jw.endObject();
                        },
                        else => {},
                    }
                }
                try jw.endArray();
            }
        },
        else => {
            // Fallback for other roles
            try jw.field("content");
            if (msg.content.len > 0) {
                switch (msg.content[0]) {
                    .text => |t| try jw.valueString(t.text),
                    else => try jw.valueString(""),
                }
            } else {
                try jw.valueString("");
            }
        },
    }

    try jw.endObject();
}

// --- Response parsing ---

fn parseCompletionResponse(body: []const u8, allocator: Allocator) ProviderError!Provider.CompletionResponse {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return error.InvalidResponse;
    defer parsed.deinit();
    const root = parsed.value;

    // Check for API error in response body
    if (jh.getPath(root, "error.type")) |_| {
        return mapApiErrorFromJson(root) orelse error.ProviderError;
    }
    if (jh.getPath(root, "error.code")) |_| {
        return mapApiErrorFromJson(root) orelse error.ProviderError;
    }

    const model_str = jh.getJsonString(jh.getPath(root, "model") orelse return error.InvalidResponse) orelse return error.InvalidResponse;
    const model = allocator.dupe(u8, model_str) catch return error.OutOfMemory;
    errdefer allocator.free(model);

    const finish_str = jh.getJsonString(jh.getPath(root, "choices.0.finish_reason") orelse return error.InvalidResponse) orelse "unknown";
    const stop_reason = mapFinishReason(finish_str);

    // Parse usage
    const usage = parseUsage(jh.getPath(root, "usage"));

    // Parse content — text from choices.0.message.content, tool_calls from choices.0.message.tool_calls
    var content_blocks = std.ArrayList(types.ContentBlock).init(allocator);
    defer content_blocks.deinit();

    if (jh.getPath(root, "choices.0.message.content")) |content_val| {
        if (jh.getJsonString(content_val)) |text| {
            if (text.len > 0) {
                content_blocks.append(.{ .text = .{
                    .text = allocator.dupe(u8, text) catch return error.OutOfMemory,
                } }) catch return error.OutOfMemory;
            }
        }
    }

    // Parse tool_calls
    if (jh.getPath(root, "choices.0.message.tool_calls")) |tc_val| {
        if (jh.getJsonArray(tc_val)) |tc_arr| {
            for (tc_arr.items) |tc_item| {
                const id = jh.getJsonString(jh.getPath(tc_item, "id") orelse continue) orelse continue;
                const name = jh.getJsonString(jh.getPath(tc_item, "function.name") orelse continue) orelse continue;
                const arguments = jh.getJsonString(jh.getPath(tc_item, "function.arguments") orelse continue) orelse continue;

                content_blocks.append(.{ .tool_use = .{
                    .id = allocator.dupe(u8, id) catch return error.OutOfMemory,
                    .name = allocator.dupe(u8, name) catch return error.OutOfMemory,
                    .input_json = allocator.dupe(u8, arguments) catch return error.OutOfMemory,
                } }) catch return error.OutOfMemory;
            }
        }
    }

    const content = content_blocks.toOwnedSlice() catch return error.OutOfMemory;

    return .{
        .message = .{
            .role = .assistant,
            .content = content,
            .allocator = allocator,
        },
        .usage = usage,
        .model = model,
        .stop_reason = stop_reason,
        .allocator = allocator,
    };
}

fn mapFinishReason(reason: []const u8) types.StopReason {
    if (std.mem.eql(u8, reason, "stop")) return .end_turn;
    if (std.mem.eql(u8, reason, "length")) return .max_tokens;
    if (std.mem.eql(u8, reason, "tool_calls")) return .tool_use;
    return .unknown;
}

fn parseUsage(value: ?std.json.Value) types.TokenUsage {
    const v = value orelse return .{};
    return .{
        .input_tokens = getOptionalInt(v, "prompt_tokens"),
        .output_tokens = getOptionalInt(v, "completion_tokens"),
    };
}

fn getOptionalInt(value: std.json.Value, path: []const u8) u32 {
    const v = jh.getPath(value, path) orelse return 0;
    return intToU32(jh.getJsonInt(v));
}

fn intToU32(v: ?i64) u32 {
    const i = v orelse return 0;
    if (i < 0) return 0;
    return @intCast(@min(i, std.math.maxInt(u32)));
}

// --- Error mapping ---

fn mapApiError(body: []const u8) ?ProviderError {
    const parsed = std.json.parseFromSlice(std.json.Value, std.heap.page_allocator, body, .{}) catch return null;
    defer parsed.deinit();
    return mapApiErrorFromJson(parsed.value);
}

fn mapApiErrorFromJson(root: std.json.Value) ?ProviderError {
    // OpenAI errors can have error.type or error.code
    const err_type = jh.getJsonString(jh.getPath(root, "error.type") orelse
        jh.getPath(root, "error.code") orelse return null) orelse return null;

    if (std.mem.eql(u8, err_type, "authentication_error")) return error.AuthenticationFailed;
    if (std.mem.eql(u8, err_type, "invalid_api_key")) return error.AuthenticationFailed;
    if (std.mem.eql(u8, err_type, "rate_limit_error")) return error.RateLimited;
    if (std.mem.eql(u8, err_type, "rate_limit_exceeded")) return error.RateLimited;
    if (std.mem.eql(u8, err_type, "invalid_request_error")) return error.InvalidRequest;
    if (std.mem.eql(u8, err_type, "model_not_found")) return error.ModelNotFound;
    if (std.mem.eql(u8, err_type, "not_found_error")) return error.ModelNotFound;
    if (std.mem.eql(u8, err_type, "server_error")) return error.ProviderError;
    if (std.mem.eql(u8, err_type, "overloaded_error")) return error.Overloaded;
    if (std.mem.eql(u8, err_type, "context_length_exceeded")) return error.ContextOverflow;

    return error.ProviderError;
}

// --- Stream context ---

const PendingToolCall = struct {
    id: ?[]const u8,
    name: ?[]const u8,
    started: bool,
};

const StreamContext = struct {
    allocator: Allocator,
    http_stream: *http.HttpStream,
    sse_reader: sse.SseLineReader,
    done: bool,
    sent_start: bool,
    finish_reason: ?[]const u8,
    pending_usage: ?types.TokenUsage,
    pending_stop: bool,
    current_tool_calls: std.ArrayList(PendingToolCall),

    pub fn next(self: *StreamContext) ProviderError!?types.StreamEvent {
        // If we have a pending stop event to emit after [DONE]
        if (self.pending_stop) {
            self.pending_stop = false;
            self.done = true;
            return .message_stop;
        }

        while (!self.done) {
            const event = (try self.sse_reader.nextEvent()) orelse return null;

            // OpenAI sends all data without event: field, so event_type is usually null
            const data = event.data;

            // Check for [DONE] sentinel
            if (std.mem.eql(u8, std.mem.trim(u8, data, " \t\r\n"), "[DONE]")) {
                // Emit message_delta with stop_reason and usage, then message_stop
                const stop_reason = self.finish_reason;
                self.finish_reason = null; // Transfer ownership

                self.pending_stop = true;
                return .{ .message_delta = .{
                    .stop_reason = stop_reason,
                    .usage = self.pending_usage,
                } };
            }

            // Parse JSON
            const parsed = std.json.parseFromSlice(std.json.Value, self.allocator, data, .{}) catch continue;
            defer parsed.deinit();
            const root = parsed.value;

            // Emit message_start on first chunk
            if (!self.sent_start) {
                self.sent_start = true;

                const msg_id = if (jh.getPath(root, "id")) |v|
                    (if (jh.getJsonString(v)) |s| self.allocator.dupe(u8, s) catch return error.OutOfMemory else null)
                else
                    null;

                const model_val = if (jh.getPath(root, "model")) |v|
                    (if (jh.getJsonString(v)) |s| self.allocator.dupe(u8, s) catch return error.OutOfMemory else null)
                else
                    null;

                return .{ .message_start = .{ .message_id = msg_id, .model = model_val } };
            }

            // Check for finish_reason
            if (jh.getPath(root, "choices.0.finish_reason")) |fr_val| {
                if (jh.getJsonString(fr_val)) |fr| {
                    if (self.finish_reason) |old| self.allocator.free(old);
                    const mapped = mapFinishReason(fr);
                    self.finish_reason = self.allocator.dupe(u8, mapped.toString()) catch return error.OutOfMemory;
                }
            }

            // Check for usage in final chunk
            if (jh.getPath(root, "usage")) |usage_val| {
                self.pending_usage = parseUsage(usage_val);
            }

            // Check for text delta
            if (jh.getPath(root, "choices.0.delta.content")) |content_val| {
                if (jh.getJsonString(content_val)) |text| {
                    if (text.len > 0) {
                        return .{ .text_delta = .{
                            .text = self.allocator.dupe(u8, text) catch return error.OutOfMemory,
                        } };
                    }
                }
            }

            // Check for tool call deltas
            if (jh.getPath(root, "choices.0.delta.tool_calls")) |tc_val| {
                if (jh.getJsonArray(tc_val)) |tc_arr| {
                    for (tc_arr.items) |tc_item| {
                        const maybe_event = try self.processToolCallDelta(tc_item);
                        if (maybe_event) |evt| return evt;
                    }
                }
            }
        }
        return null;
    }

    fn processToolCallDelta(self: *StreamContext, tc_item: std.json.Value) ProviderError!?types.StreamEvent {
        const index_val = jh.getPath(tc_item, "index") orelse return null;
        const index = intToU32(jh.getJsonInt(index_val));

        // Ensure we have enough slots
        while (self.current_tool_calls.items.len <= index) {
            self.current_tool_calls.append(.{
                .id = null,
                .name = null,
                .started = false,
            }) catch return error.OutOfMemory;
        }

        var tc = &self.current_tool_calls.items[index];

        // First chunk for this index has id and function.name
        if (jh.getPath(tc_item, "id")) |id_val| {
            if (jh.getJsonString(id_val)) |id| {
                if (tc.id) |old| self.allocator.free(old);
                tc.id = self.allocator.dupe(u8, id) catch return error.OutOfMemory;
            }
        }

        if (jh.getPath(tc_item, "function.name")) |name_val| {
            if (jh.getJsonString(name_val)) |name| {
                if (tc.name) |old| self.allocator.free(old);
                tc.name = self.allocator.dupe(u8, name) catch return error.OutOfMemory;
            }
        }

        // Emit tool_use_start when we first see id+name
        if (!tc.started and tc.id != null and tc.name != null) {
            tc.started = true;
            return .{ .tool_use_start = .{
                .id = self.allocator.dupe(u8, tc.id.?) catch return error.OutOfMemory,
                .name = self.allocator.dupe(u8, tc.name.?) catch return error.OutOfMemory,
            } };
        }

        // Subsequent chunks have function.arguments fragments
        if (jh.getPath(tc_item, "function.arguments")) |args_val| {
            if (jh.getJsonString(args_val)) |args| {
                if (args.len > 0) {
                    return .{ .tool_input_delta = .{
                        .json = self.allocator.dupe(u8, args) catch return error.OutOfMemory,
                    } };
                }
            }
        }

        return null;
    }

    pub fn deinit(self: *StreamContext) void {
        if (self.finish_reason) |fr| self.allocator.free(fr);
        for (self.current_tool_calls.items) |tc| {
            if (tc.id) |id| self.allocator.free(id);
            if (tc.name) |name| self.allocator.free(name);
        }
        self.current_tool_calls.deinit();
        self.sse_reader.deinit();
        self.http_stream.deinit();
        self.allocator.destroy(self);
    }
};

// --- Known models ---

const known_models = [_]Provider.ModelInfo{
    .{
        .id = "gpt-4o",
        .display_name = "GPT-4o",
        .context_window = 128000,
        .max_output_tokens = 16384,
        .supports_tools = true,
        .supports_vision = true,
        .supports_streaming = true,
    },
    .{
        .id = "gpt-4o-mini",
        .display_name = "GPT-4o Mini",
        .context_window = 128000,
        .max_output_tokens = 16384,
        .supports_tools = true,
        .supports_vision = true,
        .supports_streaming = true,
    },
    .{
        .id = "gpt-4-turbo",
        .display_name = "GPT-4 Turbo",
        .context_window = 128000,
        .max_output_tokens = 4096,
        .supports_tools = true,
        .supports_vision = true,
        .supports_streaming = true,
    },
    .{
        .id = "gpt-3.5-turbo",
        .display_name = "GPT-3.5 Turbo",
        .context_window = 16385,
        .max_output_tokens = 4096,
        .supports_tools = true,
        .supports_vision = false,
        .supports_streaming = true,
    },
};

// --- Tests ---

test "buildRequestBody basic" {
    const allocator = std.testing.allocator;

    var content = try allocator.alloc(types.ContentBlock, 1);
    defer allocator.free(content);
    content[0] = .{ .text = .{ .text = "Hello" } };

    const msgs = try allocator.alloc(types.Message, 1);
    defer allocator.free(msgs);
    msgs[0] = .{
        .role = .user,
        .content = content,
        .allocator = allocator,
    };

    const body = try buildRequestBody(.{
        .model = "gpt-4o-mini",
        .messages = msgs,
        .system_prompt = "You are helpful.",
    }, false, allocator);
    defer allocator.free(body);

    // Verify it's valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    const model = jh.getJsonString(jh.getPath(parsed.value, "model").?).?;
    try std.testing.expectEqualStrings("gpt-4o-mini", model);

    // System prompt should be first message, not a top-level field
    try std.testing.expect(jh.getPath(parsed.value, "system") == null);

    // First message should be system
    const first_role = jh.getJsonString(jh.getPath(parsed.value, "messages.0.role").?).?;
    try std.testing.expectEqualStrings("system", first_role);

    const sys_content = jh.getJsonString(jh.getPath(parsed.value, "messages.0.content").?).?;
    try std.testing.expectEqualStrings("You are helpful.", sys_content);

    // Second message should be user
    const second_role = jh.getJsonString(jh.getPath(parsed.value, "messages.1.role").?).?;
    try std.testing.expectEqualStrings("user", second_role);
}

test "buildRequestBody with tools" {
    const allocator = std.testing.allocator;

    const msgs = try allocator.alloc(types.Message, 0);
    defer allocator.free(msgs);

    const tools = [_]types.ToolDefinition{
        .{
            .name = "get_weather",
            .description = "Get weather for a city",
            .input_schema = "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}",
        },
    };

    const body = try buildRequestBody(.{
        .model = "gpt-4o",
        .messages = msgs,
        .tools = &tools,
    }, false, allocator);
    defer allocator.free(body);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    // Tools should be wrapped in function format
    const tool_type = jh.getJsonString(jh.getPath(parsed.value, "tools.0.type").?).?;
    try std.testing.expectEqualStrings("function", tool_type);

    const func_name = jh.getJsonString(jh.getPath(parsed.value, "tools.0.function.name").?).?;
    try std.testing.expectEqualStrings("get_weather", func_name);

    // Should use "parameters" not "input_schema"
    try std.testing.expect(jh.getPath(parsed.value, "tools.0.function.parameters") != null);
    try std.testing.expect(jh.getPath(parsed.value, "tools.0.function.input_schema") == null);
}

test "buildRequestBody with streaming includes stream_options" {
    const allocator = std.testing.allocator;

    const msgs = try allocator.alloc(types.Message, 0);
    defer allocator.free(msgs);

    const body = try buildRequestBody(.{
        .model = "gpt-4o",
        .messages = msgs,
    }, true, allocator);
    defer allocator.free(body);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    const stream_val = jh.getJsonBool(jh.getPath(parsed.value, "stream").?).?;
    try std.testing.expect(stream_val);

    const include_usage = jh.getJsonBool(jh.getPath(parsed.value, "stream_options.include_usage").?).?;
    try std.testing.expect(include_usage);
}

test "buildRequestBody with tool results emits separate messages" {
    const allocator = std.testing.allocator;

    // Create an assistant message with tool_use, then a tool result message
    var assistant_content = try allocator.alloc(types.ContentBlock, 1);
    defer allocator.free(assistant_content);
    assistant_content[0] = .{ .tool_use = .{ .id = "call_123", .name = "get_weather", .input_json = "{\"city\":\"London\"}" } };

    var tool_content = try allocator.alloc(types.ContentBlock, 1);
    defer allocator.free(tool_content);
    tool_content[0] = .{ .tool_result = .{ .tool_use_id = "call_123", .content = "{\"temp\": 15}", .is_error = false } };

    const msgs = try allocator.alloc(types.Message, 2);
    defer allocator.free(msgs);
    msgs[0] = .{ .role = .assistant, .content = assistant_content, .allocator = allocator };
    msgs[1] = .{ .role = .tool, .content = tool_content, .allocator = allocator };

    const body = try buildRequestBody(.{
        .model = "gpt-4o",
        .messages = msgs,
    }, false, allocator);
    defer allocator.free(body);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    // Tool result message should have role "tool" and tool_call_id
    const tool_msg_role = jh.getJsonString(jh.getPath(parsed.value, "messages.1.role").?).?;
    try std.testing.expectEqualStrings("tool", tool_msg_role);

    const tool_call_id = jh.getJsonString(jh.getPath(parsed.value, "messages.1.tool_call_id").?).?;
    try std.testing.expectEqualStrings("call_123", tool_call_id);
}

test "parseCompletionResponse text" {
    const allocator = std.testing.allocator;

    const response_json =
        \\{"id":"chatcmpl-123","object":"chat.completion","model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}
    ;

    var resp = try parseCompletionResponse(response_json, allocator);
    defer resp.deinit();

    try std.testing.expectEqualStrings("Hello!", resp.text().?);
    try std.testing.expectEqualStrings("gpt-4o-mini", resp.model);
    try std.testing.expectEqual(types.StopReason.end_turn, resp.stop_reason);
    try std.testing.expectEqual(@as(u32, 10), resp.usage.input_tokens);
    try std.testing.expectEqual(@as(u32, 5), resp.usage.output_tokens);
}

test "parseCompletionResponse with tool calls" {
    const allocator = std.testing.allocator;

    const response_json =
        \\{"id":"chatcmpl-456","object":"chat.completion","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"London\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":20,"completion_tokens":15,"total_tokens":35}}
    ;

    var resp = try parseCompletionResponse(response_json, allocator);
    defer resp.deinit();

    try std.testing.expectEqual(types.StopReason.tool_use, resp.stop_reason);
    try std.testing.expect(resp.message.hasToolUse());

    var iter = resp.message.toolUseBlocks();
    const tu = iter.next().?;
    try std.testing.expectEqualStrings("get_weather", tu.name);
    try std.testing.expectEqualStrings("call_abc", tu.id);
}

test "mapFinishReason" {
    try std.testing.expectEqual(types.StopReason.end_turn, mapFinishReason("stop"));
    try std.testing.expectEqual(types.StopReason.max_tokens, mapFinishReason("length"));
    try std.testing.expectEqual(types.StopReason.tool_use, mapFinishReason("tool_calls"));
    try std.testing.expectEqual(types.StopReason.unknown, mapFinishReason("something_else"));
}

test "mapApiErrorFromJson" {
    const allocator = std.testing.allocator;

    // Test with error.type
    {
        const error_json = "{\"error\":{\"type\":\"invalid_api_key\",\"message\":\"invalid key\"}}";
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, error_json, .{});
        defer parsed.deinit();

        const err = mapApiErrorFromJson(parsed.value);
        try std.testing.expectEqual(@as(?ProviderError, error.AuthenticationFailed), err);
    }

    // Test with error.code
    {
        const error_json = "{\"error\":{\"code\":\"rate_limit_exceeded\",\"message\":\"too many requests\"}}";
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, error_json, .{});
        defer parsed.deinit();

        const err = mapApiErrorFromJson(parsed.value);
        try std.testing.expectEqual(@as(?ProviderError, error.RateLimited), err);
    }
}

test "known models list" {
    const allocator = std.testing.allocator;
    var openai = try OpenAI.init(allocator, .{ .api_key = "test" });
    defer openai.deinit();

    const models = try openai.listModels(allocator);
    defer allocator.free(models);

    try std.testing.expect(models.len > 0);
    try std.testing.expectEqualStrings("gpt-4o", models[0].id);
}

test "stop sequences use stop field" {
    const allocator = std.testing.allocator;

    const msgs = try allocator.alloc(types.Message, 0);
    defer allocator.free(msgs);

    const stop_seqs = [_][]const u8{ "END", "STOP" };
    const body = try buildRequestBody(.{
        .model = "gpt-4o",
        .messages = msgs,
        .stop_sequences = &stop_seqs,
    }, false, allocator);
    defer allocator.free(body);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    // Should use "stop" not "stop_sequences"
    try std.testing.expect(jh.getPath(parsed.value, "stop") != null);
    try std.testing.expect(jh.getPath(parsed.value, "stop_sequences") == null);

    const first_stop = jh.getJsonString(jh.getPath(parsed.value, "stop.0").?).?;
    try std.testing.expectEqualStrings("END", first_stop);
}

test "buildRequestBody concatenates multiple text blocks" {
    const allocator = std.testing.allocator;

    var content = try allocator.alloc(types.ContentBlock, 2);
    defer allocator.free(content);
    content[0] = .{ .text = .{ .text = "Hello" } };
    content[1] = .{ .text = .{ .text = " World" } };

    const msgs = try allocator.alloc(types.Message, 1);
    defer allocator.free(msgs);
    msgs[0] = .{ .role = .user, .content = content, .allocator = allocator };

    const body = try buildRequestBody(.{
        .model = "gpt-4o",
        .messages = msgs,
    }, false, allocator);
    defer allocator.free(body);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();

    const msg_content = jh.getJsonString(jh.getPath(parsed.value, "messages.0.content").?).?;
    // Should contain both texts concatenated
    try std.testing.expect(std.mem.indexOf(u8, msg_content, "Hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, msg_content, "World") != null);
}
