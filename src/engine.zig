const std = @import("std");
const print = std.debug.print;


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
        tanh
    };


    pub fn add(allocator: std.mem.Allocator, self: *Value, other: *Value) !*Value {

        const out = try allocator.create(Value);
        
        const data = self.*.data + other.*.data;
        out.* = Value.init(data);
        out.*.op = Value.Op.add;
        out.*.parents = .{ self, other};
        
        return out;
    }

    pub fn sub(allocator: std.mem.Allocator, self: *Value, other: *Value) !*Value {

        const out = try allocator.create(Value);

        const data = self.*.data - other.*.data;
        out.* = Value.init(data);
        out.*.op = Value.Op.sub;
        out.*.parents = .{ self, other};

        return out;
    }

    pub fn mul(allocator: std.mem.Allocator, self: *Value, other: *Value) !*Value {

        const out = try allocator.create(Value);
        
        const data = self.*.data * other.*.data;
        out.* = Value.init(data);
        out.*.op = Value.Op.mul;
        out.*.parents = .{ self, other};

        return out;
    }

    pub fn div(allocator: std.mem.Allocator, self: *Value, other: *Value) !*Value {

        const out = try allocator.create(Value);
        
        const data = self.*.data / other.*.data;
        out.* = Value.init(data);
        out.*.op = Value.Op.div;
        out.*.parents = .{ self, other};

        return out;
    }

    pub fn pow(allocator: std.mem.Allocator, self: *Value, other: f64) !*Value{

        const out = try allocator.create(Value);
        
        const data = std.math.pow(f64, self.*.data, other);
        out.* = Value.init(data);
        out.*.op = Value.Op.pow;
        out.*.parents = .{ self, null};

        return out;
    }

};


