const std = @import("std");
const Allocator = std.mem.Allocator;

pub const ProviderError = error{
    AuthenticationFailed,
    RateLimited,
    InvalidRequest,
    ModelNotFound,
    ContextOverflow,
    Overloaded,
    ConnectionFailed,
    Timeout,
    InvalidResponse,
    ProviderError,
    OutOfMemory,
    StreamInterrupted,
    ContentFiltered,
};

pub const ErrorDetails = struct {
    message: []const u8,
    error_type: []const u8,
    allocator: Allocator,

    pub fn deinit(self: *ErrorDetails) void {
        self.allocator.free(self.message);
        self.allocator.free(self.error_type);
    }
};

test "ErrorDetails deinit frees memory" {
    const allocator = std.testing.allocator;
    var details = ErrorDetails{
        .message = try allocator.dupe(u8, "Your API key has been revoked"),
        .error_type = try allocator.dupe(u8, "authentication_error"),
        .allocator = allocator,
    };
    details.deinit();
}
