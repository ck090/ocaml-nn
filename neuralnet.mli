(* The type of our code base is essentially a float array array *)
type 'a t = float array array

(* Optimizer types *)
type optim = VGD | GDM | RMSProp

(* Convert the type string for pretty print *)
val optim_to_string : optim -> string 

(* MONAD type for the Gradient Descent Algorithm *)
module type MONAD = sig
  type 'a t
  val return : 'a -> 'a t
  val (>>=)  : 'a t -> ('a -> 'b t) -> 'b t
  val run : 'a t -> 'a
end

(* Module INIT that we will use *)
module Monad : MONAD

(* Return the init params which are nothing but hidden units and layers *)
val init_params : int -> ('a t * 'a t * 'a t * 'a t * 'a t * float * 'a t * float)

(* Gradient descent algorithm that takes in various inputs and returns four states *)
val fit : train_x:'a t -> train_y:'a t -> lr:float -> iter:int -> gd_type:Gd.algorithm_type -> optimizer:optim -> activation:Activation.type_ -> beta1:float -> beta2:float -> hidden_units:int -> epsilon:float -> ('a t * 'a t * 'a t * 'a t) Monad.t

(* Perform inference on the model trained by passing testing data and labels *)
val inference : test_x:'a t -> test_y:int array -> activation:Activation.type_ -> w1:'a t -> b1:'a t -> w2:'a t -> b2:'a t -> unit