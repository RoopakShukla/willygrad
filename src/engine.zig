const std = @import("std");
const print = std.debug.print;
const math = std.math;


pub const Value = struct {

    data: f64,
    grad: f64,
    op: Op,
    parents: [2]?*Value,


    pub fn init(data: f64) Value {
        return Value{
            .data = data,
            .grad = 0.0,
            .parents = .{ null, null },
            .op = Value.Op.leaf,
        };
    }
    

    pub const Op = enum {
        leaf,
        add,
        sub,
        mul,
        div,
        pow,
        relu,
        tanh,
        sigmoid,
        linear,
        leaky_relu
    };


    pub fn format(self: Value, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("Value(data={d})", .{self.data});
    }


    pub fn add(self: *Value, allocator: std.mem.Allocator, other: *Value) !*Value {

        const out = try allocator.create(Value);
        
        const data = self.*.data + other.*.data;
        out.* = Value.init(data);
        out.*.op = Value.Op.add;
        out.*.parents = .{ self, other};
        
        return out;
    }


    pub fn sub(self: *Value, allocator: std.mem.Allocator, other: *Value) !*Value {

        const out = try allocator.create(Value);

        const data = self.*.data - other.*.data;
        out.* = Value.init(data);
        out.*.op = Value.Op.sub;
        out.*.parents = .{ self, other};

        return out;
    }


    pub fn mul(self: *Value, allocator: std.mem.Allocator, other: *Value) !*Value {

        const out = try allocator.create(Value);
        
        const data = self.*.data * other.*.data;
        out.* = Value.init(data);
        out.*.op = Value.Op.mul;
        out.*.parents = .{ self, other};

        return out;
    }


    pub fn div(self: *Value, allocator: std.mem.Allocator, other: *Value) !*Value {

        const out = try allocator.create(Value);
        
        const data = self.*.data / other.*.data;
        out.* = Value.init(data);
        out.*.op = Value.Op.div;
        out.*.parents = .{ self, other};

        return out;
    }


    pub fn pow(self: *Value, allocator: std.mem.Allocator, other: f64) !*Value {

        const out = try allocator.create(Value);
        
        const data = math.pow(f64, self.*.data, other);
        out.* = Value.init(data);
        out.*.op = Value.Op.pow;
        out.*.parents = .{ self, null};

        return out;
    }


    // Activation functions 

    pub fn relu(self: *Value, allocator: std.mem.Allocator) !*Value {

        const out = try allocator.create(Value);

        const data = if (self.*.data > 0 ) self.*.data else 0;
        out.* = Value.init(data);
        out.*.op = Value.Op.relu;
        out.*.parents = .{ self, null };

        return out;
    }


    pub fn tanh(self: *Value, allocator: std.mem.Allocator) !*Value {

        const out = try allocator.create(Value);

        const data = math.tanh(self.*.data);
        out.* = Value.init(data);
        out.*.op = Value.Op.tanh;
        out.*.parents = .{ self, null };

        return out;
    }


    pub fn leaky_relu(self: *Value, allocator: std.mem.Allocator, alpha: f64) !*Value {

        const out = try allocator.create(Value);

        const data = if (self.*.data > 0) self.*.data else self.*.data*alpha ;
        out.* = Value.init(data);
        out.*.op = Value.Op.leaky_relu;
        out.*.parents = .{ self, null };

        return out ;
    }


};


