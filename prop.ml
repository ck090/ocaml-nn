(**************************************************************************)
(*          OCaml implementation of Forward-Prop and Backward-Prop        *)
(*            Also update params using Vanilla Gradient Descent           *)
(*                   and Gradient Descent with Momentum                   *)
(**************************************************************************)

type 'a t = float array array

let forward_prop (w1: 'a t) (b1: 'a t) (w2: 'a t) (b2: 'a t) (train_x: 'a t) (activate: Activation.type_) :
  ('a t * 'a t * 'a t) = 
  let z1_dot = Util.dot_product w1 train_x in
  let z1 = Util.add_mat b1 z1_dot in
  let a1 = Activation.activate z1 activate in
  let z2 = Util.dot_product w2 a1
  |> Util.add_mat b2 in
  let a2 = Activation.sigmoid z2 in
  (z1, a1, a2)
  
let backward_prop (z1 : 'a t) (a1 : 'a t) (a2 : 'a t) (w2 : 'a t) (train_x : 'a t) (train_y : 'a t) (activate: Activation.type_):
  ('a t * float * 'a t * float) =
  let m_inv = (1. /. (float_of_int (Array.length train_y))) in
  let dz2 = Util.sub_mat a2 train_y in
  let dw2 = Util.dot_product dz2 (Util.transpose a1) 
  |> Util.mult_scalar m_inv in
  let db2 = m_inv *. Util.sum_2d_matrix dz2 in
  let dz1 = Util.dot_product (Util.transpose w2) dz2 
  |> Util.mult_mat (Activation.activate_derive z1 activate) in
  let dw1 = Util.dot_product dz1 (Util.transpose train_x) 
  |> Util.mult_scalar m_inv in
  let db1 = m_inv *. Util.sum_2d_matrix dz1 in
  (dw1, db1, dw2, db2)

  
let update_params_gd (w1 : 'a t) (b1 : 'a t) (w2 : 'a t) (b2 : 'a t) (dw1 : 'a t) (db1 : float) (dw2 : 'a t) (db2 : float) (lr : float) :
  ('a t * 'a t * 'a t * 'a t) = 
  let w1 = Util.mult_scalar lr dw1 
  |> Util.sub_mat w1 in
  let b1 = lr *. db1 
  |> Util.sub_scalar b1 in
  let w2 = Util.mult_scalar lr dw2 
  |> Util.sub_mat w2 in
  let b2 = lr *. db2 
  |> Util.sub_scalar b2 in
  (w1, b1, w2, b2)
  
let update_params_gdm (w1 : 'a t) (b1 : 'a t) (w2 : 'a t) (b2 : 'a t) (dw1 : 'a t) (db1 : float) (dw2 : 'a t) (db2 : float) (v_dw1 : 'a t) (v_db1 : float) (v_dw2 : 'a t) (v_db2 : float) (lr : float) (beta1 : float) (beta2 : float) :
  ('a t * 'a t * 'a t * 'a t * 'a t * float * 'a t * float) = 
  let v_dw1 = (Util.mult_scalar beta2 dw1)
  |> Util.add_mat (Util.mult_scalar beta1 v_dw1) in
  let v_db1 = (beta2 *. db1) +. (beta1 *. v_db1) in
  let v_dw2 = (Util.mult_scalar beta2 dw2)
  |> Util.add_mat (Util.mult_scalar beta1 v_dw2) in
  let v_db2 = (beta2 *. db2) +. (beta1 *. v_db2) in
  let w1 = (Util.mult_scalar lr v_dw1)
  |> Util.sub_mat w1 in
  let b1 = Util.sub_scalar b1 (lr *. v_db1) in
  let w2 = (Util.mult_scalar lr v_dw2)
  |> Util.sub_mat w2 in
  let b2 = Util.sub_scalar b2 (lr *. v_db2) in
  (w1, b1, w2, b2, v_dw1, v_db1, v_dw2, v_db2)

let update_params_rmsprop (w1 : 'a t) (b1 : 'a t) (w2 : 'a t) (b2 : 'a t) (dw1 : 'a t) (db1 : float) (dw2 : 'a t) (db2 : float) (v_dw1 : 'a t) (v_db1 : float) (v_dw2 : 'a t) (v_db2 : float) (lr : float) (beta1 : float) (beta2 : float) (epsilon : float) :
  ('a t * 'a t * 'a t * 'a t * 'a t * float * 'a t * float) = 
  let v_dw1 = (Util.mult_scalar beta2 (Util.mult_mat dw1 dw1))
  |> Util.add_mat (Util.mult_scalar beta1 v_dw1) in
  let v_db1 = (beta1 *. v_db1) +. (beta2 *. (db1 *. db1)) in
  let v_dw2 = (Util.mult_scalar beta2 (Util.mult_mat dw2 dw2))
  |> Util.add_mat (Util.mult_scalar beta1 v_dw2) in
  let v_db2 = (beta1 *. v_db2) +. (beta2 *. (db2 *. db2)) in
  let w1 = (Util.mult_scalar lr (Util.div_mat dw1 (Util.add_scalar epsilon (Util.sqrt v_dw1))))
  |> Util.sub_mat w1 in
  let b1 = Util.sub_scalar b1 (lr *. (db1 /. (epsilon +. (sqrt v_db1)))) in
  let w2 = (Util.mult_scalar lr (Util.div_mat dw2 (Util.add_scalar epsilon (Util.sqrt v_dw2))))
  |> Util.sub_mat w2 in
  let b2 = Util.sub_scalar b2 (lr *. (db2 /. (epsilon +. (sqrt v_db2)))) in
  (w1, b1, w2, b2, v_dw1, v_db1, v_dw2, v_db2)