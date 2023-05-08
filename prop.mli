(* The type of our code base is essentially a float array array *)
type 'a t = float array array

(* Forward prop function *)
val forward_prop : 'a t -> 'a t -> 'a t -> 'a t -> 'a t -> Activation.type_ -> ('a t * 'a t * 'a t)

(* Back Prop function *)
val backward_prop : 'a t -> 'a t -> 'a t -> 'a t -> 'a t -> 'a t -> Activation.type_ -> ('a t * float * 'a t * float)

(* Update the parameters using Vanilla Gradient Descent *)
val update_params_gd : 'a t -> 'a t -> 'a t -> 'a t -> 'a t -> float -> 'a t -> float -> float -> ('a t * 'a t * 'a t * 'a t) 

(* Update the parameters using Gradient Descent with Momentum *)
val update_params_gdm : 'a t -> 'a t -> 'a t -> 'a t -> 'a t -> float -> 'a t -> float -> 'a t -> float -> 'a t -> float -> float -> float -> float -> ('a t * 'a t * 'a t * 'a t * 'a t * float * 'a t * float)

(* Update the parameters using RMS Prop *)
val update_params_rmsprop : 'a t -> 'a t -> 'a t -> 'a t -> 'a t -> float -> 'a t -> float -> 'a t -> float -> 'a t -> float -> float -> float -> float -> float -> ('a t * 'a t * 'a t * 'a t * 'a t * float * 'a t * float)