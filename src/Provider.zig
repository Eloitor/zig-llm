const std = @import("std");
const Allocator = std.mem.Allocator;
const types = @import("types.zig");
const ProviderError = @import("errors.zig").ProviderError;

const Provider = @This();

ptr: *anyopaque,
vtable: *const VTable,

pub const VTable = struct {
    complete: *const fn (ptr: *anyopaque, request: CompletionRequest, allocator: Allocator) ProviderError!CompletionResponse,
    stream: *const fn (ptr: *anyopaque, request: CompletionRequest, allocator: Allocator) ProviderError!StreamIterator,
    listModels: *const fn (ptr: *anyopaque, allocator: Allocator) ProviderError![]const ModelInfo,
};

/// Create a Provider from any pointer type that implements the required methods.
/// The concrete type must have:
///   - `complete(self, CompletionRequest, Allocator) ProviderError!CompletionResponse`
///   - `stream(self, CompletionRequest, Allocator) ProviderError!StreamIterator`
///   - `listModels(self, Allocator) ProviderError![]const ModelInfo`
pub fn init(pointer: anytype) Provider {
    const Ptr = @TypeOf(pointer);
    const ptr_info = @typeInfo(Ptr);

    if (ptr_info != .pointer or ptr_info.pointer.size != .one) {
        @compileError("expected single-item pointer, got " ++ @typeName(Ptr));
    }

    const Impl = ptr_info.pointer.child;

    const gen = struct {
        fn completeImpl(p: *anyopaque, request: CompletionRequest, allocator: Allocator) ProviderError!CompletionResponse {
            const self: *Impl = @ptrCast(@alignCast(p));
            return self.complete(request, allocator);
        }
        fn streamImpl(p: *anyopaque, request: CompletionRequest, allocator: Allocator) ProviderError!StreamIterator {
            const self: *Impl = @ptrCast(@alignCast(p));
            return self.stream(request, allocator);
        }
        fn listModelsImpl(p: *anyopaque, allocator: Allocator) ProviderError![]const ModelInfo {
            const self: *Impl = @ptrCast(@alignCast(p));
            return self.listModels(allocator);
        }
    };

    return .{
        .ptr = pointer,
        .vtable = &.{
            .complete = gen.completeImpl,
            .stream = gen.streamImpl,
            .listModels = gen.listModelsImpl,
        },
    };
}

pub fn complete(self: Provider, request: CompletionRequest, allocator: Allocator) ProviderError!CompletionResponse {
    return self.vtable.complete(self.ptr, request, allocator);
}

pub fn stream(self: Provider, request: CompletionRequest, allocator: Allocator) ProviderError!StreamIterator {
    return self.vtable.stream(self.ptr, request, allocator);
}

pub fn listModels(self: Provider, allocator: Allocator) ProviderError![]const ModelInfo {
    return self.vtable.listModels(self.ptr, allocator);
}

// --- Request/Response types ---

pub const CompletionRequest = struct {
    model: []const u8,
    messages: []const types.Message,
    system_prompt: ?[]const u8 = null,
    tools: []const types.ToolDefinition = &.{},
    max_tokens: u32 = 4096,
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    top_k: ?u32 = null,
    frequency_penalty: ?f32 = null,
    presence_penalty: ?f32 = null,
    stop_sequences: []const []const u8 = &.{},
};

pub const CompletionResponse = struct {
    message: types.Message,
    usage: types.TokenUsage,
    model: []const u8,
    stop_reason: types.StopReason,
    allocator: Allocator,

    pub fn text(self: CompletionResponse) ?[]const u8 {
        return self.message.text();
    }

    pub fn deinit(self: *CompletionResponse) void {
        self.message.deinit();
        self.allocator.free(self.model);
    }
};

// --- Stream Iterator ---

pub const StreamIterator = struct {
    ptr: *anyopaque,
    vtable: *const StreamVTable,

    pub const StreamVTable = struct {
        next: *const fn (ptr: *anyopaque) ProviderError!?types.StreamEvent,
        deinit: *const fn (ptr: *anyopaque) void,
    };

    pub fn initFrom(pointer: anytype) StreamIterator {
        const Ptr = @TypeOf(pointer);
        const ptr_info = @typeInfo(Ptr);
        const Impl = ptr_info.pointer.child;

        const gen = struct {
            fn nextImpl(p: *anyopaque) ProviderError!?types.StreamEvent {
                const self: *Impl = @ptrCast(@alignCast(p));
                return self.next();
            }
            fn deinitImpl(p: *anyopaque) void {
                const self: *Impl = @ptrCast(@alignCast(p));
                self.deinit();
            }
        };

        return .{
            .ptr = pointer,
            .vtable = &.{
                .next = gen.nextImpl,
                .deinit = gen.deinitImpl,
            },
        };
    }

    pub fn next(self: *StreamIterator) ProviderError!?types.StreamEvent {
        return self.vtable.next(self.ptr);
    }

    pub fn deinit(self: *StreamIterator) void {
        self.vtable.deinit(self.ptr);
    }
};

pub const ModelInfo = struct {
    id: []const u8,
    display_name: []const u8,
    context_window: u32,
    max_output_tokens: u32,
    supports_tools: bool,
    supports_vision: bool,
    supports_streaming: bool,
};

// --- Tests ---

test "Provider vtable comptime generation" {
    const TestImpl = struct {
        value: u32,

        fn complete(_: *@This(), _: CompletionRequest, _: Allocator) ProviderError!CompletionResponse {
            return error.InvalidRequest;
        }
        fn stream(_: *@This(), _: CompletionRequest, _: Allocator) ProviderError!StreamIterator {
            return error.InvalidRequest;
        }
        fn listModels(_: *@This(), _: Allocator) ProviderError![]const ModelInfo {
            return error.InvalidRequest;
        }
    };

    var impl = TestImpl{ .value = 42 };
    const provider = Provider.init(&impl);

    // Calling through vtable should dispatch to our impl
    const result = provider.complete(.{
        .model = "test",
        .messages = &.{},
    }, std.testing.allocator);
    try std.testing.expectError(ProviderError.InvalidRequest, result);
}
