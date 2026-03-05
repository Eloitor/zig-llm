const std = @import("std");
const Allocator = std.mem.Allocator;
const ProviderError = @import("errors.zig").ProviderError;

pub const Header = struct {
    name: []const u8,
    value: []const u8,
};

pub const HttpResponse = struct {
    status: std.http.Status,
    body: []const u8,
    allocator: Allocator,

    pub fn deinit(self: *HttpResponse) void {
        self.allocator.free(self.body);
    }
};

const max_response_size = 10 * 1024 * 1024; // 10 MB

/// Perform an HTTP POST request and return the response body.
pub fn post(
    allocator: Allocator,
    url: []const u8,
    headers: []const Header,
    body: []const u8,
) ProviderError!HttpResponse {
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    const uri = std.Uri.parse(url) catch return error.InvalidRequest;

    var server_header_buf: [16 * 1024]u8 = undefined;

    var extra_headers = std.ArrayList(std.http.Header).init(allocator);
    defer extra_headers.deinit();
    for (headers) |h| {
        extra_headers.append(.{ .name = h.name, .value = h.value }) catch return error.OutOfMemory;
    }
    extra_headers.append(.{ .name = "content-type", .value = "application/json" }) catch return error.OutOfMemory;

    var req = client.open(.POST, uri, .{
        .server_header_buffer = &server_header_buf,
        .extra_headers = extra_headers.items,
    }) catch return error.ConnectionFailed;
    defer req.deinit();

    req.transfer_encoding = .{ .content_length = body.len };
    req.send() catch return error.ConnectionFailed;
    req.writeAll(body) catch return error.ConnectionFailed;
    req.finish() catch return error.ConnectionFailed;
    req.wait() catch return error.ConnectionFailed;

    const status = req.response.status;
    const response_body = req.reader().readAllAlloc(allocator, max_response_size) catch return error.InvalidResponse;

    return .{
        .status = status,
        .body = response_body,
        .allocator = allocator,
    };
}

/// Map an HTTP status code to a ProviderError. Returns null if status is success.
pub fn mapStatusError(status: std.http.Status) ?ProviderError {
    const code = @intFromEnum(status);
    if (code >= 200 and code < 300) return null;

    return switch (status) {
        .unauthorized, .forbidden => error.AuthenticationFailed,
        .not_found => error.ModelNotFound,
        .too_many_requests => error.RateLimited,
        .bad_request => error.InvalidRequest,
        .payload_too_large => error.ContextOverflow,
        .service_unavailable => error.Overloaded,
        .gateway_timeout, .request_timeout => error.Timeout,
        else => error.ProviderError,
    };
}
