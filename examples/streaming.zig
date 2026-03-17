const std = @import("std");
const llm = @import("zig-llm");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const api_key = std.process.getEnvVarOwned(allocator, "ANTHROPIC_API_KEY") catch {
        std.debug.print("Error: ANTHROPIC_API_KEY environment variable not set\n", .{});
        return;
    };
    defer allocator.free(api_key);

    var anthropic = try llm.providers.Anthropic.init(allocator, .{
        .api_key = api_key,
    });
    defer anthropic.deinit();

    var chat = llm.Chat.init(allocator, anthropic.provider(), "claude-sonnet-4-20250514");
    defer chat.deinit();
    chat.system_prompt = "You are a helpful assistant. Keep responses brief.";

    std.debug.print("Sending: Tell me a short joke about programming.\n\nAssistant: ", .{});

    var iter = try chat.sendStreaming("Tell me a short joke about programming.");
    defer iter.deinit();

    while (try iter.next()) |event| {
        switch (event) {
            .text_delta => |td| {
                std.debug.print("{s}", .{td.text});
                event.deinit(allocator);
            },
            .message_stop => break,
            else => event.deinit(allocator),
        }
    }

    std.debug.print("\n", .{});
}
