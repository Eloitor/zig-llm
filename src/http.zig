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

/// A streaming HTTP connection that allows incremental reading of the response body.
/// Client, request, and buffers are heap-allocated so their addresses remain
/// stable for the lifetime of the stream.
pub const HttpStream = struct {
    allocator: Allocator,
    client: *std.http.Client,
    request: *std.http.Client.Request,
    redirect_buf: []u8,
    transfer_buf: []u8,
    cached_response: std.http.Client.Response,

    /// Returns a reader for incrementally reading the response body.
    pub fn reader(self: *HttpStream) *std.Io.Reader {
        // bodyReader modifies request.reader.interface and returns a pointer
        // into request.reader, which is heap-allocated and stable.
        const head = &self.cached_response.head;
        return self.request.reader.bodyReader(
            self.transfer_buf,
            head.transfer_encoding,
            head.content_length,
        );
    }

    pub fn deinit(self: *HttpStream) void {
        self.request.deinit();
        self.allocator.destroy(self.request);
        self.client.deinit();
        self.allocator.destroy(self.client);
        self.allocator.free(self.redirect_buf);
        self.allocator.free(self.transfer_buf);
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

    var extra_headers: std.ArrayList(std.http.Header) = .{};
    defer extra_headers.deinit(allocator);
    for (headers) |h| {
        extra_headers.append(allocator, .{ .name = h.name, .value = h.value }) catch return error.OutOfMemory;
    }
    extra_headers.append(allocator, .{ .name = "content-type", .value = "application/json" }) catch return error.OutOfMemory;

    const request = allocator.create(std.http.Client.Request) catch return error.OutOfMemory;
    errdefer allocator.destroy(request);

    request.* = client.request(.POST, uri, .{
        .extra_headers = extra_headers.items,
    }) catch return error.ConnectionFailed;
    errdefer request.deinit();

    // Send the request body
    request.transfer_encoding = .{ .content_length = body.len };
    var body_writer = request.sendBodyUnflushed(&.{}) catch return error.ConnectionFailed;
    body_writer.writer.writeAll(body) catch return error.ConnectionFailed;
    body_writer.end() catch return error.ConnectionFailed;
    if (request.connection) |conn| conn.flush() catch return error.ConnectionFailed;

    // Receive response headers
    const redirect_buf = allocator.alloc(u8, 8 * 1024) catch return error.OutOfMemory;
    errdefer allocator.free(redirect_buf);

    const resp = request.receiveHead(redirect_buf) catch return error.ConnectionFailed;

    if (mapStatusError(resp.head.status)) |status_err| {
        // Try to read the error body for better error mapping
        const transfer_buf = allocator.alloc(u8, 4096) catch return status_err;
        defer allocator.free(transfer_buf);

        var resp_mut = resp;
        var body_reader = resp_mut.reader(transfer_buf);
        const err_body = body_reader.allocRemaining(allocator, std.Io.Limit.limited(max_response_size)) catch return status_err;
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

    const transfer_buf = allocator.alloc(u8, 4096) catch return error.OutOfMemory;
    errdefer allocator.free(transfer_buf);

    const stream = allocator.create(HttpStream) catch return error.OutOfMemory;
    stream.* = .{
        .allocator = allocator,
        .client = client,
        .request = request,
        .redirect_buf = redirect_buf,
        .transfer_buf = transfer_buf,
        .cached_response = resp,
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

    var extra_headers: std.ArrayList(std.http.Header) = .{};
    defer extra_headers.deinit(allocator);
    for (headers) |h| {
        extra_headers.append(allocator, .{ .name = h.name, .value = h.value }) catch return error.OutOfMemory;
    }
    extra_headers.append(allocator, .{ .name = "content-type", .value = "application/json" }) catch return error.OutOfMemory;

    var req = client.request(.POST, uri, .{
        .extra_headers = extra_headers.items,
    }) catch return error.ConnectionFailed;
    defer req.deinit();

    // Send the request body
    req.transfer_encoding = .{ .content_length = body.len };
    var body_writer = req.sendBodyUnflushed(&.{}) catch return error.ConnectionFailed;
    body_writer.writer.writeAll(body) catch return error.ConnectionFailed;
    body_writer.end() catch return error.ConnectionFailed;
    if (req.connection) |conn| conn.flush() catch return error.ConnectionFailed;

    // Receive response headers
    var redirect_buf: [8 * 1024]u8 = undefined;
    var resp = req.receiveHead(&redirect_buf) catch return error.ConnectionFailed;
    const status = resp.head.status;

    // Read the full response body
    var transfer_buf: [4096]u8 = undefined;
    var body_reader = resp.reader(&transfer_buf);
    const response_body = body_reader.allocRemaining(allocator, std.Io.Limit.limited(max_response_size)) catch return error.InvalidResponse;

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
