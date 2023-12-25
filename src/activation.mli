(* The type of our code base is essentially a float array array *)
type 'a t = float array array

(* The types of Activation functions we can select *)
type type_ = TanH | ReLU | Sigmoid

(* Implementation of Sech x *)
val sech : float -> float

(* Implementation of d/dx of Tanh x *)
val tanh_derivative : float -> float

(* Implementation of ReLU activation function *)
val relu : 'a t -> 'a t

(* Implementation of ReLU activation function's derivative *)
val relu_derive : 'a t -> 'a t

(* Implementation of TanH activation function *)
val tanh : 'a t -> 'a t

(* Implementation of TanH activation function's derivative *)
val tanh_derive : 'a t -> 'a t

(* Implementation of Sigmoid activation function *)
val sigmoid : 'a t -> 'a t

(* Implementation of Sigmoid activation function's derivative *)
val sigmoid_derive : 'a t -> 'a t

(* Implementation of Softmax activation function *)
val softmax : 'a t -> 'a t

(* Select the appropriate activation function based on user input *)
val activate : 'a t -> type_ -> 'a t

(* Select the appropriate activation's function derivative based on user input *)
val activate_derive : 'a t -> type_ -> 'a t

(* Convert the type string for pretty print *)
val act_to_string : type_ -> string