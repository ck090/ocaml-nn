# OCAML-NN

A fully functional implementation of a Fully-Connected Neural Network (FCNNs) in **OCaml**


## Details

Train and run a neural network on any dataset. We implement a fully-connected multi-layered neural network which learns network parameters using back propogation implemented as gradient descent algorithm. We further provide the following hyper-parameter customizabillity:

`Optimization Functions:  Vanilla GD | GD w/Momentum | RMS Prop` <br>
`Activation Functions:  ReLU | TanH | Sigmoid ` <br>
`Gradient Descent Type:  Stochastic GD | Mini-Batch GD | Vanilla GD` <br>
`Learning Rate: <any floating point number, ideally 0.05/0.1>` <br>
`Epochs: <any integer greater than 0, ideally 1000>` <br>
`Beta1 and Beta2: <any floating point number in [0,1]>` <br>
`# of Hidden Units: <any integer greater than 0, ideally 10>` <br>
`Epsilon: <any floating point number, ideally 1e-8>` <br>

## Usage

There are two functions that can be called along with some utility functions that allows you to read datasets.
