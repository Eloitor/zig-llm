const std = @import("std");
const ProviderError = @import("errors.zig").ProviderError;

pub const RetryConfig = struct {
    max_retries: u32 = 3,
    initial_delay_ms: u64 = 1000,
    max_delay_ms: u64 = 30000,
    backoff_multiplier: u32 = 2,
};

pub const default_config = RetryConfig{};

/// Returns true if the error is transient and worth retrying.
pub fn isRetryable(err: ProviderError) bool {
    return switch (err) {
        error.RateLimited,
        error.Overloaded,
        error.Timeout,
        error.ConnectionFailed,
        error.StreamInterrupted,
        => true,
        else => false,
    };
}

/// Compute the delay for a given attempt, capping at max_delay_ms.
pub fn computeDelay(config: RetryConfig, attempt: u32) u64 {
    var delay = config.initial_delay_ms;
    var i: u32 = 0;
    while (i < attempt) : (i += 1) {
        delay = @min(delay * config.backoff_multiplier, config.max_delay_ms);
    }
    return delay;
}

/// Sleep for the given number of milliseconds.
pub fn sleepMs(ms: u64) void {
    std.Thread.sleep(ms * std.time.ns_per_ms);
}

// --- Tests ---

test "isRetryable returns true for transient errors" {
    try std.testing.expect(isRetryable(error.RateLimited));
    try std.testing.expect(isRetryable(error.Overloaded));
    try std.testing.expect(isRetryable(error.Timeout));
    try std.testing.expect(isRetryable(error.ConnectionFailed));
    try std.testing.expect(isRetryable(error.StreamInterrupted));
}

test "isRetryable returns false for permanent errors" {
    try std.testing.expect(!isRetryable(error.AuthenticationFailed));
    try std.testing.expect(!isRetryable(error.InvalidRequest));
    try std.testing.expect(!isRetryable(error.ModelNotFound));
    try std.testing.expect(!isRetryable(error.ContextOverflow));
    try std.testing.expect(!isRetryable(error.InvalidResponse));
    try std.testing.expect(!isRetryable(error.ProviderError));
    try std.testing.expect(!isRetryable(error.OutOfMemory));
    try std.testing.expect(!isRetryable(error.ContentFiltered));
}

test "computeDelay applies exponential backoff" {
    const config = RetryConfig{
        .initial_delay_ms = 100,
        .max_delay_ms = 10000,
        .backoff_multiplier = 2,
    };
    try std.testing.expectEqual(@as(u64, 100), computeDelay(config, 0));
    try std.testing.expectEqual(@as(u64, 200), computeDelay(config, 1));
    try std.testing.expectEqual(@as(u64, 400), computeDelay(config, 2));
    try std.testing.expectEqual(@as(u64, 800), computeDelay(config, 3));
}

test "computeDelay caps at max_delay_ms" {
    const config = RetryConfig{
        .initial_delay_ms = 5000,
        .max_delay_ms = 8000,
        .backoff_multiplier = 2,
    };
    // 5000 * 2 = 10000, capped to 8000
    try std.testing.expectEqual(@as(u64, 8000), computeDelay(config, 1));
    try std.testing.expectEqual(@as(u64, 8000), computeDelay(config, 5));
}

test "default_config has expected values" {
    try std.testing.expectEqual(@as(u32, 3), default_config.max_retries);
    try std.testing.expectEqual(@as(u64, 1000), default_config.initial_delay_ms);
    try std.testing.expectEqual(@as(u64, 30000), default_config.max_delay_ms);
    try std.testing.expectEqual(@as(u32, 2), default_config.backoff_multiplier);
}
