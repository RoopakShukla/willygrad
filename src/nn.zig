const std = @import("std");
const engine = @import("engine.zig");
const Value = engine.Value;
var prng = std.Random.DefaultPrng.init(0);
const ArrayList = std.ArrayList;


pub fn uniform(min: f64, max: f64) f64 {
    return min + (max - min) * prng.random().float(f64);
}


pub const Neuron = struct {
    weights: ArrayList(*Value),
    bias: *Value,

    pub fn init(allocator: std.mem.Allocator, n_inputs: usize) !Neuron {
        var weights: ArrayList(*Value) = .empty;
        for (0 .. n_inputs) |_| {
            const w = try Value.init(uniform(-1, 1), allocator);
            try weights.append(allocator, w);
        }
        const bias = try Value.init(uniform(-1, 1), allocator);
        
        return Neuron{
            .weights = weights,
            .bias = bias
        };
    }

    pub fn call(self: *Neuron, allocator: std.mem.Allocator, input: []const *Value) !*Value {
        var sum = self.*.bias; 
        const length = input.len;

        if (input.len != self.weights.items.len) {
            return error.InputLengthMismatch;
        }


        for(0..length) |i| {
             sum = try sum.add(allocator, try self.*.weights.items[i].mul(allocator, input[i]));
        }

        return sum;
        

    }

    pub fn parameters(self: *Neuron, allocator: std.mem.Allocator) !ArrayList(*Value) {
        var params = try self.*.weights.clone(allocator);

        try params.append(allocator, self.*.bias);
        return params;
    }

};


pub const Layer = struct {
    neurons: ArrayList(Neuron),

    pub fn init(allocator: std.mem.Allocator, n_inputs: usize, n_neurons: usize) !Layer {
        var neurons: ArrayList(Neuron) = .empty;
        
        for (0 .. n_neurons) |_| {
            try neurons.append(allocator, try Neuron.init(allocator, n_inputs));
        }

        return .{
            .neurons = neurons
        };
    }

    pub fn call(self: *Layer, allocator: std.mem.Allocator, input: []const *Value) !ArrayList(*Value) {
        var out: ArrayList(*Value) = .empty;
        const length = self.neurons.items.len;

        for (0 .. length) |i| {
            try out.append(allocator, try self.*.neurons.items[i].call(allocator, input));
        }

        return out; 
    }
    

};


pub const MLP = struct {
    layers: ArrayList(Layer),

    pub fn init(allocator: std.mem.Allocator, n_inputs: usize, layer_sizes: []const usize) !MLP {
        var layers: ArrayList(Layer) = .empty;

        var current_inputs: usize = n_inputs;

        for (layer_sizes) |n_neurons| {
            try layers.append(allocator, try Layer.init(allocator, current_inputs, n_neurons)); 
            current_inputs = n_neurons;
        }
        return .{
            .layers = layers
        };
    }

    pub fn call(self: *MLP, allocator: std.mem.Allocator, input: []const *Value) !ArrayList(*Value) {
        var out: ArrayList(*Value) = .empty;
        var current_input = input;

        for (self.layers.items) |*layer| {
            out = try layer.call(allocator, current_input);
            current_input = out.items;
        }

        return out;
    }

};
