const std = @import("std");
const engine = @import("engine.zig");
const testing = std.testing;
const eq = testing.expectEqual;
const approx = testing.expectApproxEqAbs;
const allocator = std.testing.allocator;
const Value = engine.Value;


test "initialize leaf Value" {
    const value = Value.init(6.7);

    try eq(@as(f64, 6.7), value.data);
    try eq(@as(f64, 0), value.grad);
    try eq(@as(?*Value, null), value.parents[0]);
    try eq(@as(?*Value, null), value.parents[1]);
    try eq(@as(Value.Op, Value.Op.leaf), value.op);
}

test " Value formatting" {
    const node = Value.init(6.7);

    try testing.expectFmt("Value(data=6.7)" ,"{f}", .{node});
}

test "addition" {
    var node1 = Value.init(1);
    var node2 = Value.init(2);

    const result = try node1.add(allocator, &node2);
    defer allocator.destroy(result);

    try eq(@as(f64, 1+2), result.data);
    try eq(@as(f64, 0), result.grad);
    try eq(@as(?*Value, &node1), result.parents[0]);
    try eq(@as(?*Value, &node2), result.parents[1]);
    try eq(@as(Value.Op, Value.Op.add), result.op);
}

test "subtraction" {
    var node1 = Value.init(67);
    var node2 = Value.init(6.7);

    const result = try node1.sub(allocator, &node2);
    defer allocator.destroy(result);

    try eq(@as(f64, 67-6.7), result.data);
    try eq(@as(f64, 0), result.grad);
    try eq(@as(?*Value, &node1), result.parents[0]);
    try eq(@as(?*Value, &node2), result.parents[1]);
    try eq(@as(Value.Op, Value.Op.sub), result.op);

} 

test "multiplication" {
    var node1 = Value.init(6);
    var node2 = Value.init(7);

    const result = try node1.mul(allocator, &node2);
    defer allocator.destroy(result);

    try eq(@as(f64, 6*7), result.data);
    try eq(@as(f64, 0), result.grad);
    try eq(@as(?*Value, &node1), result.parents[0]);
    try eq(@as(?*Value, &node2), result.parents[1]);
    try eq(@as(Value.Op, Value.Op.mul), result.op);

}

test "division" {
    var node1 = Value.init(22.78);
    var node2 = Value.init(3.4);

    const result = try node1.div(allocator, &node2);
    defer allocator.destroy(result);

    try eq(@as(f64, 22.78/3.4), result.data);
    try eq(@as(f64, 0), result.grad);
    try eq(@as(?*Value, &node1), result.parents[0]);
    try eq(@as(?*Value, &node2), result.parents[1]);
    try eq(@as(Value.Op, Value.Op.div), result.op);

}

test "power" {
    var node1 = Value.init(10);
    const exponent = 2;

    const result = try node1.pow(allocator, exponent);
    defer allocator.destroy(result);

    try eq(@as(f64, std.math.pow(f64, 10, 2)), result.data);
    try eq(@as(f64, 0), result.grad);
    try eq(@as(?*Value, &node1), result.parents[0]);
    try eq(@as(?*Value, null), result.parents[1]);
    try eq(@as(Value.Op, Value.Op.pow), result.op);

}

test "ReLU with positive input" {
    var node = Value.init(6.767);

    const result = try node.relu(allocator);
    defer allocator.destroy(result);

    try eq(@as(f64, 6.767), result.data); 
}

test "ReLU with negative input" {
    var node = Value.init(-67.67);

    const result = try node.relu(allocator);
    defer allocator.destroy(result);

    try eq(@as(f64, 0), result.data);
}

test "tanh" {
    var node = Value.init(-2);

    const result = try node.tanh(allocator);
    defer allocator.destroy(result);

    try approx(@as(f64, -0.9640), result.data, 0.0001);
}

test "Leaky ReLU with positive input" {
    var node = Value.init(6.767);

    const result = try node.leaky_relu(allocator, 0.01);
    defer allocator.destroy(result);

    try eq(@as(f64, 6.767), result.data); 
}

test "Leaky ReLU with negative input" {
    var node = Value.init(-67);

    const result = try node.leaky_relu(allocator, 0.01);
    defer allocator.destroy(result);

    try eq(@as(f64, -0.67), result.data);
}
