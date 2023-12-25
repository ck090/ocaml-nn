(**************************************************************************)
(*        Utility functions that we will be using in our main code        *)
(*             The type is going to be 'a t = float list list             *)
(**************************************************************************)
type 'a t = float array array

let shape (a: 'a t) = 
  let rows = Array.length a in
  let cols = Array.length a.(0) in
  (rows, cols)

let float_2d rows cols = Array.init rows (fun _ -> Array.init cols (fun _ -> (Random.float 1.0) -. 0.5))

let float_2d_zeros rows cols = Array.make_matrix rows cols 0.0

let transpose (a : 'a array array) : 'a array array =
  let rows = Array.length a in
  let cols = Array.length a.(0) in
  let transposed = Array.make_matrix cols rows a.(0).(0) in
    for i = 0 to rows - 1 do
      for j = 0 to cols - 1 do
        transposed.(j).(i) <- a.(i).(j)
      done
    done;
  transposed

let dot_product (a1 : float array array) (a2 : float array array) : float array array =
  let rows_a1, _ = shape a1 in
  let _, cols_a2 = shape a2 in
  let transposed = transpose a2 in
  let result = Array.make_matrix rows_a1 cols_a2 0.0 in
    for i = 0 to rows_a1 - 1 do
      for j = 0 to cols_a2 - 1 do
        let dot_product = Array.mapi (fun k x -> x *. transposed.(j).(k)) a1.(i) |> Array.fold_left (+.) 0.0 in
        result.(i).(j) <- dot_product
      done
    done;
  result

let sqrt (arr : 'a t) : 'a t =
  let rows, cols = shape arr in
  let result = Array.make_matrix rows cols 0.0 in
  for i = 0 to rows - 1 do
    for j = 0 to cols - 1 do
      result.(i).(j) <- sqrt arr.(i).(j)
    done;
  done;
  result  

let add_scalar (s : float) (a : float array array) = Array.map (fun row -> Array.map (fun x -> x +. s) row) a

let sub_scalar (a: float array array) (s: float) = Array.map (fun row -> Array.map (fun x -> x -. s) row) a

let mult_scalar (s : float) (a : float array array) = Array.map (fun row -> Array.map (fun x -> x *. s) row) a

let add_mat (m1 : float array array) (m2 : float array array) : float array array =
  let n_rows1, n_cols1 = shape m1 in
  let n_rows2, n_cols2 = shape m2 in
  if n_rows1 <> n_rows2 || n_cols1 <> n_cols2 then
  let m1_resized = Array.init n_rows1 (fun i -> Array.make n_cols2 m1.(i).(0)) in
  for i = 0 to n_rows1 - 1 do
    for j = 0 to n_cols2 - 1 do
      m1_resized.(i).(j) <- m1.(i).(0)
    done
  done;
  Array.map2 (Array.map2 (+.)) m1_resized m2
  else
  Array.map2 (Array.map2 (+.)) m1 m2

let sub_mat (l1: float array array) (l2: float array array) = 
  Array.map2 (fun row1 row2 -> Array.map2 (fun x y -> x -. y) row1 row2) l1 l2

let mult_mat (l1: float array array) (l2: float array array) = 
  Array.map2 (fun row1 row2 -> Array.map2 (fun x y -> x *. y) row1 row2) l1 l2

let div_mat (l1: float array array) (l2: float array array) = 
  Array.map2 (fun row1 row2 -> Array.map2 (fun x y -> x /. y) row1 row2) l1 l2

let sum_2d_matrix (matrix : float array array) : float =
  Array.fold_left (fun acc row -> acc +. Array.fold_left (+.) 0.0 row) 0.0 matrix

let one_hot (y: int array): float array array =
  let n = Array.fold_left max 0 y + 1 in
  let temp = Array.init (Array.length y) (fun i -> Array.init n (fun j -> if j = y.(i) then 1.0 else 0.0)) in
  transpose temp
  
let random_indices n x y =
  let rec aux acc n =
    if n = 0 then acc
    else
      let r = Random.int (y - x + 1) + x in
      if Array.mem r acc then aux acc n
      else aux (Array.append [|r|] acc) (n-1)
  in aux [||] n

let select_indices matrix indices =
  Array.map (fun i -> matrix.(i)) indices

let get_predictions (a2 : float array array) : int array =
  let n_rows = Array.length a2 in
  let result = Array.make n_rows (-1) in
  for i = 0 to n_rows - 1 do
    let row = a2.(i) in
    let n_cols = Array.length row in
    let max_val = ref row.(0) in
    let max_index = ref 0 in
    for j = 1 to n_cols - 1 do
      let x = row.(j) in
      if x > !max_val then begin
        max_val := x;
        max_index := j;
      end
    done;
    result.(i) <- !max_index
  done;
  result

let get_accuracy (pred : 'a array) (y : 'a array) : float =
  let diff = Array.map2 (fun v1 v2 -> if v1 = v2 then 1 else 0) pred y in
  let sum_and_len = Array.fold_left (fun (s,l) x -> s+x, l+1) (0,0) diff in
  float_of_int (fst (sum_and_len)) /. float_of_int (snd (sum_and_len))

let load_csv (file_name: string) (f: string -> 'a) = 
  Csv.load file_name 
  |> List.map Array.of_list
  |> Array.of_list
  |> Array.map (Array.map f)

let get_data (path : string) : 'a t = 
  let temp = load_csv path float_of_string in
  transpose temp 
  |> Array.map (Array.map (fun x -> x /. 255.0))

let get_labels (path : string) : int array = 
  let temp = load_csv path int_of_string in
  Array.map (fun row -> row.(0)) temp

let asrt = function
  | (true,_) -> ()
  | (false, str) -> failwith ("Assertion failure: "^str)