const std = @import("std");
const print = std.debug.print;
const math = std.math;
const PI = math.pi;
const HashMap = std.AutoArrayHashMapUnmanaged;
const ArrayList = std.ArrayList;


pub const Value = struct {

    data: f64,
    grad: f64,
    op: Op,
    parents: [2]?*Value,
    tmp: f64,


    pub fn init(data: f64) Value {
        return Value{
            .data = data,
            .grad = 0,
            .parents = .{ null, null },
            .op = Value.Op.leaf,
            .tmp = 0,
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
        leaky_relu,
        gelu,
        linear,
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
        out.*.tmp = other;

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
        out.*.tmp = alpha;

        return out;
    }


    pub fn sigmoid(self: *Value, allocator: std.mem.Allocator) !*Value {

        const out = try allocator.create(Value);

        const data = 1 / (1 + math.exp(-self.data));
        out.* = Value.init(data);
        out.*.op = Value.Op.sigmoid;
        out.*.parents = .{ self, null };

        return out;
    }


    pub fn gelu(self: *Value, allocator: std.mem.Allocator) !*Value {

        const out = try allocator.create(Value);

        const data = (self.data / 2.0) * (1 + math.tanh(math.pow(f64, 2.0 / PI, 0.5) * ( self.data + 0.044715 * math.pow(f64, self.data, 3))));
        out.* = Value.init(data);
        out.*.op = Value.Op.gelu;
        out.*.parents = .{ self, null };

        return out;
    }


    pub fn linear(self: *Value, allocator: std.mem.Allocator) !*Value {

        const out = try allocator.create(Value);

        out.* = Value.init(self.*.data);
        out.*.op = Value.Op.linear;
        out.*.parents = .{ self, null };

        return out;
    }



    pub fn applyBackward(self: *Value) void {


        switch (self.*.op) {
            Value.Op.add => {

                const left = self.parents[0].?;
                const right = self.parents[1].?;
                
                left.grad += self.grad;
                right.grad += self.grad;
            },

            Value.Op.sub => {

                const left = self.parents[0].?;
                const right = self.parents[1].?;

                left.grad += self.grad;
                right.grad -= self.grad;
            },
            
            Value.Op.mul => {

                const left = self.parents[0].?;
                const right = self.parents[1].?;

                left.grad += right.data * self.grad;
                right.grad += left.data * self.grad;
            },

            Value.Op.div => {

                const left = self.parents[0].?;
                const right = self.parents[1].?;

                left.grad += (1 / right.data) * self.grad;
                right.grad += (-left.data / (right.data * right.data)) * self.grad;
            },

            Value.Op.pow => {
                const left = self.parents[0].?;

                left.grad += (self.tmp) * math.pow(f64, left.data, (self.tmp - 1)) * self.grad;
            },

            Value.Op.relu => {

                const left = self.parents[0].?;

                if (self.data > 0) {
                    left.grad += self.grad;
                }
            },

            Value.Op.tanh => {
                
                const left = self.parents[0].?;

                left.grad += (1 - self.data * self.data) * self.grad;
            },

            Value.Op.sigmoid => {

                const left = self.parents[0].?;

                left.grad += self.data * (1 - self.data) * self.grad;
            },

            Value.Op.leaky_relu => {

                const left = self.parents[0].?;

                if (left.data > 0) {
                    left.grad += self.grad;
                }
                else {
                    left.grad += self.tmp * self.grad;
                }
            },

            Value.Op.gelu => {

                const left = self.parents[0].?;
                
                const x: f64 = left.data;
                const tanhU: f64 = math.tanh(@sqrt(2.0 / PI) * ( x + 0.044715 * x * x * x ));
                
                left.grad += ((0.5 * (1 + tanhU)) + (x / 2.0) * (1 - tanhU * tanhU) * @sqrt(2.0 / PI) * (1 + 3 * 0.044715 * x * x))  * self.grad;
            },

            Value.Op.linear => {
                
                const left = self.parents[0].?;

                left.grad += self.grad;
            },

            Value.Op.leaf => {},

        }
            
    }

    pub fn backward(self: *Value, allocator: std.mem.Allocator) !void {
        var visited: HashMap(*Value, void) = .empty;
        defer visited.deinit(allocator);

        var topo: ArrayList(*Value) = .empty;
        defer topo.deinit(allocator);

        try self.buildTopo(&visited, &topo, allocator);

        self.grad = 1;

        var i = topo.items.len;
        while (i > 0) {
            i -= 1;
            topo.items[i].applyBackward();
        }

    }

    pub fn buildTopo(node: *Value, visited: *HashMap(*Value, void), topo: *ArrayList(*Value), allocator: std.mem.Allocator) !void {
        if (visited.contains(node)) return;

        try visited.put(allocator, node, {});

        for (node.parents) |parent| {
            if (parent != null) {
                try Value.buildTopo(parent.?, visited, topo, allocator);
            }
        }

        try topo.append(allocator, node);

    }


};


