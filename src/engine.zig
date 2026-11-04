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
        mul,
        pow,
        relu,
        tanh
    };



    








};


