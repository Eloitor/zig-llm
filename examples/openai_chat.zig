const std = @import("std");
const llm = @import("zig-llm");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Get API key from environment
    const api_key = std.process.getEnvVarOwned(allocator, "OPENAI_API_KEY") catch {
        std.debug.print("Error: OPENAI_API_KEY environment variable not set\n", .{});
        return;
    };
    defer allocator.free(api_key);

    // Create provider — use api_base to point at Groq, Ollama, Together, etc.
    var openai = try llm.providers.OpenAI.init(allocator, .{
        .api_key = api_key,
        // .api_base = "https://api.groq.com/openai",  // Groq
        // .api_base = "http://localhost:11434",         // Ollama
    });
    defer openai.deinit();

    var chat = llm.Chat.init(allocator, openai.provider(), "gpt-4o-mini");
    defer chat.deinit();
    chat.system_prompt = "You are a helpful assistant. Keep responses brief.";

    // Non-streaming
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

    // Streaming
    std.debug.print("\nSending (streaming): Tell me a short joke about programming.\n\nAssistant: ", .{});

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
