# simplelearn

A neural network library built on top of Theano.

To get started quickly, I would first suggest checking out the examples/ directory. While the code is reasonably well-documented, nothing beats an end-to-end example.

## Features:
* Theano's strengths: automated differentiation, dynamically compiles to C/CUDA/cuDNN code.
* Very to implement arbitrary DAGs as networks, not just linear stacks of "layers".
* Easy to optimize arbitrary output with respect to arbitrary inputs, not just the loss with respect to the weights. This is useful for visualizing network state, for example, by optimizing the input image with respect to a desired classification.
