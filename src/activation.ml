(**************************************************************************)
(*            OCaml implementation of Activation and Loss Functions       *)
(*                  1. ReLU  2. TanH 3. Softmax 4. Sigmoid                *)
(**************************************************************************)
type 'a t = float array array

type type_ = TanH | ReLU | Sigmoid
  
let sech x = 1.0 /. cosh x

let tanh_derivative x = sech x ** 2.0

let relu (z : 'a t) : 'a t =
  Array.map (fun arr -> Array.map (fun x -> max x 0.0) arr) z

let relu_derive (z : 'a t) : 'a t =
  Array.map (fun arr -> Array.map (fun x -> if x > 0.0 then 1.0 else 0.0) arr) z

let softmax (z : 'a t) : 'a t =
  let softmax_row row =
    let exp_row = Array.map exp row in
    let sum_exp_row = Array.fold_left (+.) 0.0 exp_row in
    Array.map (fun x -> x /. sum_exp_row) exp_row
  in
  let smax = Array.map softmax_row (Util.transpose z) in
  Util.transpose smax

let tanh (z : 'a t) : 'a t =
  Array.map (fun arr -> Array.map (fun x -> tanh x) arr) z

let tanh_derive (z : 'a t) : 'a t =
  Array.map (fun arr -> Array.map (fun x -> tanh_derivative x) arr) z

let sigmoid (z : 'a t) : 'a t =
  let sigmoid_scalar y = 1.0 /. (1.0 +. exp (-. y)) in
  let sigmoid_row row = Array.map sigmoid_scalar row in
  Array.map sigmoid_row z

let sigmoid_derive (z : 'a t) : 'a t =
  let sigmoid_scalar x = 1.0 /. (1.0 +. exp (-. x)) in
  Array.map (fun row -> Array.map (fun x -> let s = sigmoid_scalar x in s *. (1.0 -. s)) row) z

let activate (z : 'a t) (act : type_) : 'a t = 
  match act with
  | TanH -> tanh z
  | ReLU -> relu z
  | Sigmoid -> sigmoid z
  
let activate_derive (z : 'a t) (act : type_) : 'a t = 
  match act with
  | TanH -> tanh_derive z
  | ReLU -> relu_derive z
  | Sigmoid -> sigmoid_derive z

let act_to_string (act : type_) : string =
  match act with
  | TanH -> "TanH"
  | ReLU -> "ReLU"
  | Sigmoid -> "Sigmoid"