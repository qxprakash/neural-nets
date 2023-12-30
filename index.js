const forwardMultiplyGate = function(a, b){
    return a * b;
}

const forwardAddGate = function(a, b){
    return a + b;
}


const forwardCircuit = function(x , y, z){
    const q = forwardAddGate(x , y);
    const f = forwardMultiplyGate(q , z);
    return f;
}

// initial conditions
let x = -2, y = 5, z = -4;
let q = forwardAddGate(x, y); // q is 3
let f = forwardMultiplyGate(q, z); // output is -12

// gradient of the MULTIPLY gate with respect to its inputs
// wrt is short for "with respect to"
const derivative_f_wrt_z = q; // 3
const derivative_f_wrt_q = z; // -4

// derivative of the ADD gate with respect to its inputs
const derivative_q_wrt_x = 1.0;
const derivative_q_wrt_y = 1.0;

// chain rule
const derivative_f_wrt_x = derivative_q_wrt_x * derivative_f_wrt_q; // -4
const derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q; // -4



// final gradient, from above: [-4, -4, 3] , last gradient is 3 because derivative of f_wrt_z = q = 3 , see the code above
const gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]


// let the inputs respond to the force/tug:
let step_size = 0.01;
x = x + step_size * derivative_f_wrt_x; // -2.04
y = y + step_size * derivative_f_wrt_y; // 4.96
z = z + step_size * derivative_f_wrt_z; // -3.97



// Our circuit now better give higher output:
q = forwardAddGate(x, y); // q becomes 2.92
f = forwardMultiplyGate(q, z); // output is -11.59, up from -12! Nice!






// numerical gradient check -- to manually calcualte the gradient of the inputs x , y , z that is df/dx , df /dy  , df / dz

// thumb rule , --

// take a small step value in this case -- h = 0.0001
// calculate the circuit foward pass , with input + h , here input is that input which derivative you want to find out
// derivative  = (forwardPass(input + stepValue , ...otherInputs) - forwardPass(input , ...otherInputs)) / stepValue

// initial conditions variable names are changed to avoid conflicting names , but values are same as above inputs , x , y and z
// let a = -2, b = 5, c = -4;

var h = 0.0001;
var x_derivative = (forwardCircuit(x+h,y,z) - forwardCircuit(x,y,z)) / h; // -4
var y_derivative = (forwardCircuit(x,y+h,z) - forwardCircuit(x,y,z)) / h; // -4
var z_derivative = (forwardCircuit(x,y,z+h) - forwardCircuit(x,y,z)) / h; // 3

console.log(x_derivative , y_derivative , z_derivative)





