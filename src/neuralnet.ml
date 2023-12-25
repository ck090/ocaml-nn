(**************************************************************************)
(*    OCaml Monadic implementation of a FC Neural Network with Back-Prop  *)
(*    Here we define the InitParams function and all the Fit function     *)
(*    Accompanied with the Inference function that performs predictions   *)
(**************************************************************************)
type 'a t = float array array

module type MONAD = sig
  type 'a t
  val return : 'a -> 'a t
  val (>>=)  : 'a t -> ('a -> 'b t) -> 'b t
  val run : 'a t -> 'a
end

module Monad : MONAD = struct
  type 'a t = 'a
  let return x = x
  let (>>=) x f = f x
  let run x = x
end

let init_params (hidden_units : int) (output_units : int) (n_features : int) : 
  ('a t * 'a t * 'a t * 'a t * 'a t * float * 'a t * float) =
  let w1 = Util.float_2d hidden_units n_features in
  let b1 = Util.float_2d hidden_units 1 in
  let v_w1 = Util.float_2d_zeros hidden_units n_features in
  let v_b1 = 0.0 in
  let w2 = Util.float_2d output_units hidden_units in
  let b2 = Util.float_2d output_units 1 in
  let v_w2 = Util.float_2d_zeros output_units hidden_units in
  let v_b2 = 0.0 in
  (w1, b1, w2, b2, v_w1, v_b1, v_w2, v_b2)

type optim = VGD | GDM | RMSProp

let optim_to_string (opt : optim) : string =
  match opt with
  | VGD -> "VGD"
  | GDM -> "GDM"
  | RMSProp -> "RMSProp"

let fit ~(train_x : 'a t) ~(train_y : 'a t) ~(lr : float) ~(iter : int) ~(gd_type : Gd.algorithm_type) ~(optimizer : optim) ~(activation : Activation.type_) ~(beta1 : float) ~(beta2 : float) ~(hidden_units : int) ~(output_units : int) ~(epsilon : float) : ('a t * 'a t * 'a t * 'a t) Monad.t =
  Printf.printf "\n\n-------Loaded dataset-------\n";
  Printf.printf "Train dataset\t: (%d, %d)\n" (fst (Util.shape train_x)) (snd (Util.shape train_x));
  Util.asrt((snd (Util.shape train_x)) = (snd (Util.shape train_y)), "Shape error in Training data...");
  Printf.printf "\nModel hyper-parameters:\nEpochs: %d\nLearning Rate: %.2f\nOptimizer: %s\n" iter lr (optim_to_string optimizer);
  Printf.printf "Gradient Descent: %s\nActivation: %s\nHidden Units: %d\n" (Gd.gd_to_string gd_type) (Activation.act_to_string activation) hidden_units;
  Printf.printf "Beta1: %.3f\nBeta2: %.3f\nEpsilon: %f\n" beta1 beta2 epsilon;
  Printf.printf "\n-------Training Model-------\n";

  let open Monad in
  let w1, b1, w2, b2, v_w1, v_b1, v_w2, v_b2 = init_params hidden_units output_units (fst (Util.shape train_x)) in
  let rec batch_loop i w1 b1 w2 b2 v_w1 (v_b1: float) v_w2 (v_b2: float) iter train_x train_y (x, y) lr =
    if i = iter then return (w1, b1, w2, b2)
    else 
      Prop.forward_prop w1 b1 w2 b2 x activation
      |> return >>= fun (z1, a1, a2) ->
      Prop.backward_prop z1 a1 a2 w2 x y activation
      |> return >>= fun (dw1, db1, dw2, db2) ->
      if (i mod 50) = 0 then Printf.printf "Iteration: %d\n%!" i;
      if optimizer = GDM then 
        Prop.update_params_gdm w1 b1 w2 b2 dw1 db1 dw2 db2 v_w1 v_b1 v_w2 v_b2 lr beta1 beta2
        |> return >>= fun (w1', b1', w2', b2', v_dw1', v_db1', v_dw2', v_db2') ->
        batch_loop (i + 1) w1' b1' w2' b2' v_dw1' v_db1' v_dw2' v_db2' iter train_x train_y (Gd.data train_x train_y gd_type) lr
      else if optimizer = VGD then 
        Prop.update_params_gd w1 b1 w2 b2 dw1 db1 dw2 db2 lr 
        |> return >>= fun (w1', b1', w2', b2') ->
        batch_loop (i + 1) w1' b1' w2' b2' v_w1 v_b1 v_w2 v_b2 iter train_x train_y (Gd.data train_x train_y gd_type) lr
      else 
        Prop.update_params_rmsprop w1 b1 w2 b2 dw1 db1 dw2 db2 v_w1 v_b1 v_w2 v_b2 lr beta1 beta2 epsilon
        |> return >>= fun (w1', b1', w2', b2', v_dw1', v_db1', v_dw2', v_db2') ->
        batch_loop (i + 1) w1' b1' w2' b2' v_dw1' v_db1' v_dw2' v_db2' iter train_x train_y (Gd.data train_x train_y gd_type) lr
  in
  batch_loop 0 w1 b1 w2 b2 v_w1 v_b1 v_w2 v_b2 iter train_x train_y (Gd.data train_x train_y gd_type) lr

let inference ~(test_x : 'a t) ~(test_y : int array) ~(activation : Activation.type_) ~(w1 : 'a t) ~(b1 : 'a t) ~(w2 : 'a t) ~(b2 : 'a t) : unit =
  Printf.printf "\n-------Performing Inference-------\n";
  Printf.printf "Test dataset\t: (%d, %d)\n" (fst (Util.shape test_x)) (snd (Util.shape test_x));
  Util.asrt((snd (Util.shape test_x)) = Array.length test_y, "Shape error in Testing data...");

  let _, _, a2 = Prop.forward_prop w1 b1 w2 b2 test_x activation in
  let preds = Util.get_predictions (Util.transpose a2) in
  Printf.printf "Test accuracy: %f\nGT:\t" (Util.get_accuracy preds test_y);
  Array.iter (fun x -> Printf.printf "%d " x) test_y;
  Printf.printf "\nPred:\t";
  Array.iter (fun x -> Printf.printf "%d " x) preds;
  Printf.printf "\n";