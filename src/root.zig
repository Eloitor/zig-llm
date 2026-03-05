pub const Provider = @import("Provider.zig");
pub const Chat = @import("Chat.zig");
pub const types = @import("types.zig");
pub const errors = @import("errors.zig");
pub const sse = @import("sse.zig");
pub const http = @import("http.zig");
pub const json_helpers = @import("json_helpers.zig");

pub const providers = struct {
    pub const Anthropic = @import("providers/anthropic.zig");
};

// Re-export commonly used types at top level
pub const Message = types.Message;
pub const Role = types.Role;
pub const ContentBlock = types.ContentBlock;
pub const StreamEvent = types.StreamEvent;
pub const TokenUsage = types.TokenUsage;
pub const ToolDefinition = types.ToolDefinition;
pub const StopReason = types.StopReason;
pub const ProviderError = errors.ProviderError;

test {
    // Pull in all tests from submodules
    @import("std").testing.refAllDecls(@This());
}
