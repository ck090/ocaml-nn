(**************************************************************************)
(*                  OCaml implementation of Gradient Descent              *)
(*                  1. Vanilla 2. Stochastic 3. Mini-Batch                *)
(**************************************************************************)
type algorithm_type = GD | SGD | MBGD

let data (x : float array array) (y : float array array) (type_ : algorithm_type) : (float array array * float array array) = 
  match type_ with
  | GD -> (x, y)
  | SGD -> 
    let size = snd (Util.shape x) in
    let _ = Util.asrt (snd (Util.shape x) = snd (Util.shape y), "Shape error..") in
    let ri = Util.random_indices 1 0 (size-1) in
    let x', y' = Util.select_indices (Util.transpose x) ri, Util.select_indices (Util.transpose y) ri in
    (Util.transpose x', Util.transpose y')
  | MBGD ->
    let size = snd (Util.shape x) in
    let _ = Util.asrt (snd (Util.shape x) = snd (Util.shape y), "Shape error..") in
    let ri = Util.random_indices 4 0 (size-1) in
    let x', y' = Util.select_indices (Util.transpose x) ri, Util.select_indices (Util.transpose y) ri in
    (Util.transpose x', Util.transpose y')

let gd_to_string (algo : algorithm_type) : string =
  match algo with
  | GD -> "GD"
  | SGD -> "SGD"
  | MBGD -> "MBGD"