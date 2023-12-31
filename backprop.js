// every Unit corresponds to a wire in the diagrams
var Unit = function (value, grad) {
  // value computed in the forward pass
  this.value = value;
  // the derivative of circuit output w.r.t this unit, computed in backward pass
  this.grad = grad;
};

var multiplyGate = function () {};
multiplyGate.prototype = {
  forward: function (u0, u1) {
    // store pointers to input Units u0 and u1 and output unit utop
    this.u0 = u0;
    this.u1 = u1;
    this.utop = new Unit(u0.value * u1.value, 0.0);
    return this.utop;
  },
  backward: function () {
    // take the gradient in output unit and chain it with the
    // local gradients, which we derived for multiply gate before
    // then write those gradients to those Units.
    this.u0.grad += this.u1.value * this.utop.grad;
    this.u1.grad += this.u0.value * this.utop.grad;
  },
};


var addGate = function(){ };
addGate.prototype = {
  forward: function(u0, u1) {
    this.u0 = u0;
    this.u1 = u1; // store pointers to input units
    this.utop = new Unit(u0.value + u1.value, 0.0);
    return this.utop;
  },
  backward: function() {
    // add gate. derivative wrt both inputs is 1
    this.u0.grad += 1 * this.utop.grad;
    this.u1.grad += 1 * this.utop.grad;
  }
}



var sigmoidGate = function() {
    // helper function
    this.sig = function(x) { return 1 / (1 + Math.exp(-x)); };
  };
  sigmoidGate.prototype = {
    forward: function(u0) {
      this.u0 = u0;
      this.utop = new Unit(this.sig(this.u0.value), 0.0);
      return this.utop;
    },
    backward: function() {
      var s = this.sig(this.u0.value);
      this.u0.grad += (s * (1 - s)) * this.utop.grad;
    }
  }


// creating inputs from data and making a forward pass on the circuit


// create input units
var a = new Unit(1.0, 0.0);
var b = new Unit(2.0, 0.0);
var c = new Unit(-3.0, 0.0);
var x = new Unit(-1.0, 0.0);
var y = new Unit(3.0, 0.0);

// create the gates
var mulg0 = new multiplyGate();
var mulg1 = new multiplyGate();
var addg0 = new addGate();
var addg1 = new addGate();
var sg0 = new sigmoidGate();

// do the forward pass
var forwardNeuron = function() {
  ax = mulg0.forward(a, x); // a*x = -1
  by = mulg1.forward(b, y); // b*y = 6
  axpby = addg0.forward(ax, by); // a*x + b*y = 5
  axpbypc = addg1.forward(axpby, c); // a*x + b*y + c = 2
  s = sg0.forward(axpbypc); // sig(a*x + b*y + c) = 0.8808
};
forwardNeuron();

console.log('circuit output: ' + s.value); // prints 0.8808

// forward pass output of the circuit is 0.8808


// now setting the top level gradient that is df = 1 and doing a backward pass , gradient of the final output w.r.t itself


s.grad = 1.0;

sg0.backward(); // writes gradient into axpbypc
addg1.backward(); // writes gradients into axpby and c
addg0.backward(); // writes gradients into ax and by
mulg1.backward(); // writes gradients into b and y
mulg0.backward(); // writes gradients into a and x




// explaination -- check notes for detailed breakdown of what's happening above and how the gradient is got calculated in backward pass

var step_size = 0.01;
a.value += step_size * a.grad; // a.grad is -0.105
b.value += step_size * b.grad; // b.grad is 0.315
c.value += step_size * c.grad; // c.grad is 0.105
x.value += step_size * x.grad; // x.grad is 0.105
y.value += step_size * y.grad; // y.grad is 0.210

forwardNeuron();
console.log('circuit output after one backprop: ' + s.value); // prints 0.8825



// success we got higher value after adjusting the values with the gradients 0.8825 > 0.8808

// Now let's cross check if we have calculated the gradient correctly by checking the numerical gradient:

var forwardCircuitFast = function(a,b,c,x,y) {
  return 1/(1 + Math.exp( - (a*x + b*y + c)));
};
var a = 1, b = 2, c = -3, x = -1, y = 3;
var h = 0.0001;
var a_grad = (forwardCircuitFast(a+h,b,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h;
var b_grad = (forwardCircuitFast(a,b+h,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h;
var c_grad = (forwardCircuitFast(a,b,c+h,x,y) - forwardCircuitFast(a,b,c,x,y))/h;
var x_grad = (forwardCircuitFast(a,b,c,x+h,y) - forwardCircuitFast(a,b,c,x,y))/h;
var y_grad = (forwardCircuitFast(a,b,c,x,y+h) - forwardCircuitFast(a,b,c,x,y))/h;


console.log(a_grad , b_grad , c_grad , x_grad , y_grad)

// Indeed, these all give the same values as the backpropagated gradients [-0.105, 0.315, 0.105, 0.105, 0.210]. meaning we calculated them correctly



module.exports = {
  multiplyGate ,
  addGate,
  Unit
};