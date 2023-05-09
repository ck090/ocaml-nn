(* Train the NN on any data *)

let () =
  (* Read in the train and test data *)
  let train_x = Util.get_data "./data/train_small.csv" in
  let train_y = Util.get_labels "./data/train_small_labels.csv" in

  let test_x = Util.get_data "./data/test_small.csv" in
  let test_y = Util.get_labels "./data/test_small_labels.csv" in

  (* Compute the gradients i.e. train the model *)
  let gradients = Neuralnet.fit ~train_x:train_x 
                                ~train_y:(Util.one_hot train_y) 
                                ~lr:0.09
                                ~iter:1000
                                ~gd_type:MBGD
                                ~optimizer:GDM
                                ~activation:ReLU
                                ~beta1:0.9 
                                ~beta2:0.1
                                ~hidden_units:10
                                ~epsilon:0.00001
  in

  (* Run inference on test data using the gradients *)
  let w1, b1, w2, b2 = Neuralnet.Monad.run gradients in
  Neuralnet.inference ~test_x:test_x
                      ~test_y:test_y
                      ~activation:ReLU
                      ~w1:w1
                      ~b1:b1
                      ~w2:w2
                      ~b2:b2

(**************************************************************************)
(*                      Train the NN on any data                          *)
(* train_x: Training data will be read as a Float Array Array format      *)
(* train_y: Training data labels will be read as a Float Array format     *)
(* lr : Represents the learning rate to set for the model                 *)
(* iter : Number of epochs to train the model on the provided data        *)
(* gd_type : Gradient Descent type choose: SGD or MBGD or GD              *)
(* optimizer : Optimizer choose: VGD or GDM or RMSProp                    *)
(* activation : Activation function choose: TanH or ReLU or Sigmoid       *)
(* beta1 and beta2 : Hyperparameters for GDM between 0 and 1              *)
(* hidden_units : Number of hidden units in the NN model                  *)
(**************************************************************************)
(*                       Test the NN on any data                          *)
(* test_x: Testing data will be read as a Float Array Array format        *)
(* test_y: Testing data labels will be read as a Float Array format       *)
(* activation : Activation function choose: TanH or ReLU                  *)
(**************************************************************************)