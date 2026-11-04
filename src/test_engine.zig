const std = @import("std");
const engine = @import("engine.zig");
const testing = std.testing;
const eq = testing.expectEqual;

test "initialize leaf Value" {
    const value = engine.Value.init(6.7);

    try eq(@as(f64, 6.7), value.data);
    try eq(@as(f64, 0), value.grad);
}
