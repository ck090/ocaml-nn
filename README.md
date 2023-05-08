# OCAML-NN

A fully functional monadic implementation of a Fully-Connected Neural Network (FCNNs) in **OCaml**


## Details

Train and run a neural network on any dataset. We implement a fully-connected multi-layered neural network which learns network parameters using back propogation implemented as gradient descent algorithm. The data structure we use is OCaml Arrays. We further provide the following hyper-parameter customizabillity:

- Optimization Functions:  `Vanilla GD | GD w/Momentum | RMS Prop` <br>
- Activation Functions:  `ReLU | TanH | Sigmoid ` <br>
- Gradient Descent Type:  `Stochastic GD | Mini-Batch GD | Vanilla GD` <br>
- Learning Rate: `<any floating point number, ideally 0.05/0.1>` <br>
- Epochs: `<any integer greater than 0, ideally 1000>` <br>
- Beta1 and Beta2: `<any floating point number in [0,1]>` <br>
- Number of Hidden Units: `<any integer greater than 0, ideally 10>` <br>
- Epsilon: `<any floating point number, ideally 1e-8>` <br>

## Usage

There are two functions that can be called along with some utility functions that allows you to read datasets. An example training process is written in `train.ml`

`Neuralnet.fit` - Returns a trained model as a monadic state. This state contains four vectors which represent the final gradients of the model. To get the weights and biases pass the gradients through the `run` function of the monad. The function has the following arguments:

- `train_x:` input data as a `float array array`.
- `train_y:` input data labels as a `int array`.
- `lr:` learning rate for the model
- `iter:` number of epochs for training
- `gd_type:` gradient descent type choose from `SGD | MBGD | GD`
- `optimizer:` optimizier type choose from `VGD | GDM | RMSProp`
- `activation:` activation function choose from `TanH | ReLU | Sigmoid`
- `beta1:` regularization parameter used in GD with Momentum and RMSProp
- `beta2:` (1 - beta1) used for the same purpose
- `hidden_units:` number of hidden units per hidden layer in the model
- `epsilon:` small value used in RMSProp to avoid divide by zero errors

`Neuralnet.inference` - Peforms predictions on the test data provided. The function has the following arguments:

- `test_x:` test data as a `float array array`
- `test_y:` test data labels as `int array` used for computing `Accuracy`
- `activation:` activation function, should be the same as the one above
- `w1:` weights of the first layer, from `Neuralnet.fit`
- `b1:` biases of the first layer, from `Neuralnet.fit`
- `w2:` weights of the second layer, from `Neuralnet.fit`
- `b2:` biases of the second layer, from `Neuralnet.fit`

The entire OCaml code base is parameterized. One can always change them to customize it.

## Example

Run the code in `train.ml` to train the NeuralNet model on MNIST-10 dataset. To run do the following, from inside the folder:
```
dune build
dune exec ./train.exe
```
Run `dune clean`, to clean the `_build/` directory.


## Tests

