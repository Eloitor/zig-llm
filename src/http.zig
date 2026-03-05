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
const server_header_buf_size = 16 * 1024;

/// A streaming HTTP connection that allows incremental reading of the response body.
/// Client, request, and header buffer are heap-allocated so their addresses remain
/// stable for the lifetime of the stream.
pub const HttpStream = struct {
    allocator: Allocator,
    client: *std.http.Client,
    request: *std.http.Client.Request,
    server_header_buf: []u8,

    /// Returns a reader for incrementally reading the response body.
    pub fn reader(self: *HttpStream) std.io.AnyReader {
        return self.request.reader();
    }

    pub fn deinit(self: *HttpStream) void {
        self.request.deinit();
        self.allocator.destroy(self.request);
        self.client.deinit();
        self.allocator.destroy(self.client);
        self.allocator.free(self.server_header_buf);
        self.allocator.destroy(self);
    }
};

/// Open an HTTP POST connection for streaming. Sends the request body and waits
/// for response headers, but does NOT read the response body. Returns an HttpStream
/// whose reader() can be used to incrementally read the response.
pub fn openStream(
    allocator: Allocator,
    url: []const u8,
    headers: []const Header,
    body: []const u8,
) ProviderError!*HttpStream {
    const client = allocator.create(std.http.Client) catch return error.OutOfMemory;
    client.* = .{ .allocator = allocator };
    errdefer {
        client.deinit();
        allocator.destroy(client);
    }

    const uri = std.Uri.parse(url) catch return error.InvalidRequest;

    const header_buf = allocator.alloc(u8, server_header_buf_size) catch return error.OutOfMemory;
    errdefer allocator.free(header_buf);

    var extra_headers = std.ArrayList(std.http.Header).init(allocator);
    defer extra_headers.deinit();
    for (headers) |h| {
        extra_headers.append(.{ .name = h.name, .value = h.value }) catch return error.OutOfMemory;
    }
    extra_headers.append(.{ .name = "content-type", .value = "application/json" }) catch return error.OutOfMemory;

    const request = allocator.create(std.http.Client.Request) catch return error.OutOfMemory;
    errdefer allocator.destroy(request);

    request.* = client.open(.POST, uri, .{
        .server_header_buffer = header_buf,
        .extra_headers = extra_headers.items,
    }) catch return error.ConnectionFailed;
    errdefer request.deinit();

    request.transfer_encoding = .{ .content_length = body.len };
    request.send() catch return error.ConnectionFailed;
    request.writeAll(body) catch return error.ConnectionFailed;
    request.finish() catch return error.ConnectionFailed;
    request.wait() catch return error.ConnectionFailed;

    const status = request.response.status;
    if (mapStatusError(status)) |status_err| {
        // Try to read the error body for better error mapping
        const err_body = request.reader().readAllAlloc(allocator, max_response_size) catch {
            return status_err;
        };
        defer allocator.free(err_body);

        // Try to parse API-specific error from body
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, err_body, .{}) catch return status_err;
        defer parsed.deinit();

        const jh = @import("json_helpers.zig");
        const err_type = jh.getJsonString(jh.getPath(parsed.value, "error.type") orelse return status_err) orelse return status_err;

        if (std.mem.eql(u8, err_type, "authentication_error")) return error.AuthenticationFailed;
        if (std.mem.eql(u8, err_type, "rate_limit_error")) return error.RateLimited;
        if (std.mem.eql(u8, err_type, "invalid_request_error")) return error.InvalidRequest;
        if (std.mem.eql(u8, err_type, "overloaded_error")) return error.Overloaded;
        if (std.mem.eql(u8, err_type, "not_found_error")) return error.ModelNotFound;
        if (std.mem.eql(u8, err_type, "permission_error")) return error.AuthenticationFailed;

        return status_err;
    }

    const stream = allocator.create(HttpStream) catch return error.OutOfMemory;
    stream.* = .{
        .allocator = allocator,
        .client = client,
        .request = request,
        .server_header_buf = header_buf,
    };

    return stream;
}

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
