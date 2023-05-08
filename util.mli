(* The type of our code base is essentially a float array array *)
type 'a t = float array array

(* Get the tuple of rows and columns from the data provided *)
val shape : 'a t -> int * int

(* Return a 2D array, randomally init between 0 and 1 *)
val float_2d : int -> int -> 'a t

(* Return a 2D array, randomally init to zero *)
val float_2d_zeros : int -> int -> float array array

(* Tail recursive function to transpose the 2D array *)
val transpose : 'a array array -> 'a array array

(* Computes the dot product of the two matrices *)
val dot_product : 'a t -> 'a t -> 'a t

(* Computes the square root of the matrix *)
val sqrt : 'a t -> 'a t

(* Add a scalar to a 2D vector *)
val add_scalar : float -> 'a t -> 'a t

(* Subtract a scalar to a 2D vector *)
val sub_scalar :  'a t -> float -> 'a t

(* Multiply a scalar to a 2D vector *)
val mult_scalar : float -> 'a t -> 'a t

(* Add two 2D vectors together and return a 2D vector *)
val add_mat : 'a t -> 'a t -> 'a t

(* Sub one 2D vector from another *)
val sub_mat : 'a t -> 'a t -> 'a t

(* Multiply one 2D vector from another *)
val mult_mat : 'a t -> 'a t -> 'a t

(* Divide one 2D vector from another *)
val div_mat : 'a t -> 'a t -> 'a t

(* Sum all the values of a 2D vector and return a single float value *)
val sum_2d_matrix : 'a t -> float

(* One hot encode a array and return a 2D vector *)
val one_hot : int array -> 'a t

(* Random indices of a certain shape returned in a range *)
val random_indices : int -> int -> int -> int array

(* Use the random indices obtained to get rows from a vector *)
val select_indices : 'a array -> int array -> 'a array

(* Get the predictions from the vector input *)
val get_predictions : 'a t -> int array

(* Get the accuracy from the predictions and Ground Truth *)
val get_accuracy : 'a array -> 'a array -> float

(* Simple function to LOAD a csv file using the CSV module *)
val load_csv : string -> (string -> 'a) -> 'a array array

(* Return the training data with the path provided *)
val get_data : string -> 'a t

(* Return the testing data with the path provided *)
val get_labels : string -> int array

(* A simple assert function that can be used to check and throw errors *)
val asrt : bool * string -> unit