Simplelearn: A library for machine learning research
====================================================

Author: [Matthew Koichi Grimes](http://mkg.cc)

Copyright 2015

Introduction
------------

Simplelearn is a machine learning library that allows you to define complex
models using simple parts. It it meant for those who want to quickly define new
models and training algorithms. The code is written to be easily readable,
understood, and extended.

All models in Simplelearn are directed acyclic graphs of function objects. It
is therefore easy to design models that are more complicated than a simple
stack of layers. Thanks to the Theano library, these models automatically
compile to C/CUDA code, and are automatically differentiable.

It is easy to optimize loss functions and other outputs with respect to
arbitrary variables, be they model parameters (training), input variables
(inference), or both. Optimizing w.r.t. inputs is useful for visualizing deep
features, or searching for pathological inputs that increase the loss function.

Currently Simplelearn only supports deterministic models (no RBMs).

Dependencies
------------
Required:

* [Theano](http://www.deeplearning.net/software/theano/)
* [nose](http://nose.readthedocs.org/en/latest/)
* [h5py](http://www.h5py.org/)

Optional:

* [CUDA](https://developer.nvidia.com/cuda-zone) To run on the GPU.
* [cuDNN](https://developer.nvidia.com/cuDNN) To run convnets on the GPU.
* [matplotlib](http://matplotlib.org/) To run many of the visualizers / examples.
* [sphinx](http://sphinx-doc.org/) To build the API docs yourself.

Documentation
-------------

Installation instructions are in simplelearn/INSTALL.md

Beginners should start by with the
[GitHub wiki](https://github.com/SuperElectric/simplelearn/wiki), then explore
the complete working examples in simplelearn/examples/.


License
-------

I have licensed this library under the Apache 2.0 license.
[(full text)](https://www.apache.org/licenses/LICENSE-2.0.html)
[(wikipedia)](http://en.wikipedia.org/wiki/Apache_License)
[(tl:dr)](https://tldrlegal.com/license/apache-license-2.0-%28apache-2.0%29).
