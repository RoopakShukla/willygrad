const std = @import("std");
const engine = @import("engine.zig");
const nn = @import("nn.zig");
const testing = std.testing;
const eq = testing.expectEqual;
const approx = testing.expectApproxEqAbs;
const allocator = std.testing.allocator;
const Value = engine.Value;
const MLP = nn.MLP;


test "initialize leaf Value" {
    const value = try Value.init(6.7, allocator);
    defer allocator.destroy(value);

    try eq(@as(f64, 6.7), value.data);
    try eq(@as(f64, 0), value.grad);
    try eq(@as(?*Value, null), value.parents[0]);
    try eq(@as(?*Value, null), value.parents[1]);
    try eq(@as(Value.Op, Value.Op.leaf), value.op);
}

test " Value formatting" {
    const node = try Value.init(6.7, allocator);
    defer allocator.destroy(node);

    try testing.expectFmt("Value(data=6.7)" ,"{f}", .{node});
}

test "addition" {
    var node1 = try Value.init(1, allocator);
    const node2 = try Value.init(2, allocator);

    defer allocator.destroy(node1);
    defer allocator.destroy(node2);

    const result = try node1.add(allocator, node2);
    defer allocator.destroy(result);

    try eq(@as(f64, 1+2), result.data);
    try eq(@as(f64, 0), result.grad);
    try eq(@as(?*Value, node1), result.parents[0]);
    try eq(@as(?*Value, node2), result.parents[1]);
    try eq(@as(Value.Op, Value.Op.add), result.op);
}

test "subtraction" {
    var node1 = try Value.init(67, allocator);
    const node2 = try Value.init(6.7, allocator);

    defer allocator.destroy(node1);
    defer allocator.destroy(node2);

    const result = try node1.sub(allocator, node2);
    defer allocator.destroy(result);

    try eq(@as(f64, 67-6.7), result.data);
    try eq(@as(f64, 0), result.grad);
    try eq(@as(?*Value, node1), result.parents[0]);
    try eq(@as(?*Value, node2), result.parents[1]);
    try eq(@as(Value.Op, Value.Op.sub), result.op);

} 

test "multiplication" {
    var node1 = try Value.init(6, allocator);
    const node2 = try Value.init(7, allocator);

    defer allocator.destroy(node1);
    defer allocator.destroy(node2);

    const result = try node1.mul(allocator, node2);
    defer allocator.destroy(result);

    try eq(@as(f64, 6*7), result.data);
    try eq(@as(f64, 0), result.grad);
    try eq(@as(?*Value, node1), result.parents[0]);
    try eq(@as(?*Value, node2), result.parents[1]);
    try eq(@as(Value.Op, Value.Op.mul), result.op);

}

test "division" {
    var node1 = try Value.init(22.78, allocator);
    const node2 = try Value.init(3.4, allocator);

    defer allocator.destroy(node1);
    defer allocator.destroy(node2);

    const result = try node1.div(allocator, node2);
    defer allocator.destroy(result);

    try eq(@as(f64, 22.78/3.4), result.data);
    try eq(@as(f64, 0), result.grad);
    try eq(@as(?*Value, node1), result.parents[0]);
    try eq(@as(?*Value, node2), result.parents[1]);
    try eq(@as(Value.Op, Value.Op.div), result.op);

}

test "power" {
    var node = try Value.init(10, allocator);
    const exponent = 2;
    defer allocator.destroy(node);

    const result = try node.pow(allocator, exponent);
    defer allocator.destroy(result);

    try eq(@as(f64, std.math.pow(f64, 10, 2)), result.data);
    try eq(@as(f64, 0), result.grad);
    try eq(@as(?*Value, node), result.parents[0]);
    try eq(@as(?*Value, null), result.parents[1]);
    try eq(@as(Value.Op, Value.Op.pow), result.op);

}

test "ReLU with positive input" {
    var node = try Value.init(6.767, allocator);
    defer allocator.destroy(node);

    const result = try node.relu(allocator);
    defer allocator.destroy(result);

    try eq(@as(f64, 6.767), result.data); 
    try eq(@as(f64, 0), result.grad);
    try eq(@as(Value.Op, Value.Op.relu), result.op);
    try eq(@as(?*Value, node), result.parents[0]);
    try eq(@as(?*Value, null), result.parents[1]);
}

test "ReLU with negative input" {
    var node = try Value.init(-67.67, allocator);
    defer allocator.destroy(node);

    const result = try node.relu(allocator);
    defer allocator.destroy(result);

    try eq(@as(f64, 0), result.data);
    try eq(@as(f64, 0), result.grad);
    try eq(@as(Value.Op, Value.Op.relu), result.op);
    try eq(@as(?*Value, node), result.parents[0]);
    try eq(@as(?*Value, null), result.parents[1]);
}

test "tanh" {
    var node = try Value.init(-2, allocator);
    defer allocator.destroy(node);

    const result = try node.tanh(allocator);
    defer allocator.destroy(result);

    try approx(@as(f64, -0.9640), result.data, 0.0001);
    try eq(@as(f64, 0), result.grad);
    try eq(@as(Value.Op, Value.Op.tanh), result.op);
    try eq(@as(?*Value, node), result.parents[0]);
    try eq(@as(?*Value, null), result.parents[1]);
}

test "Leaky ReLU with positive input" {
    var node = try Value.init(6.767, allocator);
    defer allocator.destroy(node);

    const result = try node.leaky_relu(allocator, 0.01);
    defer allocator.destroy(result);

    try eq(@as(f64, 6.767), result.data);
    try eq(@as(f64, 0), result.grad);
    try eq(@as(Value.Op, Value.Op.leaky_relu), result.op);
    try eq(@as(?*Value, node), result.parents[0]);
    try eq(@as(?*Value, null), result.parents[1]);
}

test "Leaky ReLU with negative input" {
    var node = try Value.init(-67, allocator);
    defer allocator.destroy(node);

    const result = try node.leaky_relu(allocator, 0.01);
    defer allocator.destroy(result);

    try eq(@as(f64, -0.67), result.data);
    try eq(@as(f64, 0), result.grad);
    try eq(@as(Value.Op, Value.Op.leaky_relu), result.op);
    try eq(@as(?*Value, node), result.parents[0]);
    try eq(@as(?*Value, null), result.parents[1]);
}

test "sigmoid" {
    var node = try Value.init(0, allocator);
    defer allocator.destroy(node);

    const result = try node.sigmoid(allocator);
    defer allocator.destroy(result);

    try eq(@as(f64, 0.5), result.data);
}

test "GeLU" {
    var node = try Value.init(-1, allocator);
    defer allocator.destroy(node);

    const result = try node.gelu(allocator);
    defer allocator.destroy(result);

    try approx(@as(f64, -0.158808), result.data, 0.000001);
}
        
test "backward" {
    var x = try Value.init(3, allocator);
    defer allocator.destroy(x);

    const y = try x.mul(allocator, x);
    defer allocator.destroy(y);
    try y.backward(allocator);

    try approx(@as(f64, 6), x.grad, 0.0001);
}

test "MLP forward pass" {
    var arena = std. heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const arena_allocator = arena.allocator();

    var mlp = try MLP.init(arena_allocator, 3, &.{4, 3, 1});

    const input = [_]*Value{
        try Value.init(1, arena_allocator),
        try Value.init(2, arena_allocator),
        try Value.init(3, arena_allocator),
    };

    const output = try mlp.call(arena_allocator, &input);

    try std.testing.expectEqual(@as(usize, 3), mlp.layers.items.len);
    try std.testing.expectEqual(@as(usize, 4), mlp.layers.items[0].neurons.items.len);
    try std.testing.expectEqual(@as(usize, 3), mlp.layers.items[1].neurons.items.len);
    try std.testing.expectEqual(@as(usize, 1), mlp.layers.items[2].neurons.items.len);

    try std.testing.expectEqual(@as(usize, 1), output.items.len);
}
