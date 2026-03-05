const std = @import("std");
const llm = @import("zig-llm");

const tools = [_]llm.ToolDefinition{
    .{
        .name = "get_weather",
        .description = "Get the current weather for a given city.",
        .input_schema =
        \\{"type":"object","properties":{"city":{"type":"string","description":"The city name"}},"required":["city"]}
        ,
    },
};

fn handleToolCall(_: void, name: []const u8, input_json: []const u8) llm.Chat.ToolResult {
    std.debug.print("Tool call: {s}({s})\n", .{ name, input_json });

    if (std.mem.eql(u8, name, "get_weather")) {
        return .{ .content = "{\"temperature\": 15, \"condition\": \"partly cloudy\", \"humidity\": 72}" };
    }

    return .{ .content = "Unknown tool", .is_error = true };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const api_key = std.posix.getenv("ANTHROPIC_API_KEY") orelse {
        std.debug.print("Error: ANTHROPIC_API_KEY environment variable not set\n", .{});
        return;
    };

    var anthropic = try llm.providers.Anthropic.init(allocator, .{
        .api_key = api_key,
    });
    defer anthropic.deinit();

    var chat = llm.Chat.init(allocator, anthropic.provider(), "claude-sonnet-4-20250514");
    defer chat.deinit();
    chat.system_prompt = "You are a helpful assistant with access to weather tools.";
    chat.tools = &tools;

    std.debug.print("Sending: What's the weather like in London?\n\n", .{});

    var response = try chat.sendWithTools("What's the weather like in London?", {}, handleToolCall);
    defer response.deinit();

    if (response.text()) |text| {
        std.debug.print("\nAssistant: {s}\n", .{text});
    }

    std.debug.print("\nTotal usage: {d} input, {d} output tokens\n", .{
        chat.total_usage.input_tokens,
        chat.total_usage.output_tokens,
    });
}
