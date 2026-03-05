const std = @import("std");
const llm = @import("zig-llm");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Get API key from environment
    const api_key = std.posix.getenv("ANTHROPIC_API_KEY") orelse {
        std.debug.print("Error: ANTHROPIC_API_KEY environment variable not set\n", .{});
        return;
    };

    // Create provider
    var anthropic = try llm.providers.Anthropic.init(allocator, .{
        .api_key = api_key,
    });
    defer anthropic.deinit();

    // Create chat
    var chat = llm.Chat.init(allocator, anthropic.provider(), "claude-sonnet-4-20250514");
    defer chat.deinit();
    chat.system_prompt = "You are a helpful assistant. Keep responses brief.";

    // Send a message
    std.debug.print("Sending: Hello! What is Zig?\n", .{});
    var response = try chat.send("Hello! What is Zig?");
    defer response.deinit();

    if (response.text()) |text| {
        std.debug.print("\nAssistant: {s}\n", .{text});
    }

    std.debug.print("\nUsage: {d} input, {d} output tokens\n", .{
        response.usage.input_tokens,
        response.usage.output_tokens,
    });

    // Send a follow-up (conversation history is maintained)
    std.debug.print("\nSending: What are its main advantages?\n", .{});
    var response2 = try chat.send("What are its main advantages?");
    defer response2.deinit();

    if (response2.text()) |text| {
        std.debug.print("\nAssistant: {s}\n", .{text});
    }

    std.debug.print("\nTotal usage: {d} input, {d} output tokens\n", .{
        chat.total_usage.input_tokens,
        chat.total_usage.output_tokens,
    });
}
