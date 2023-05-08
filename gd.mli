(* The algorithm type that we use will be one of the types *)
type algorithm_type = GD | SGD | MBGD

(* Return one of the gradient descent techniques using the types defined *)
val data : float array array -> float array array -> algorithm_type -> (float array array * float array array)

(* Convert the type string for pretty print *)
val gd_to_string : algorithm_type -> string