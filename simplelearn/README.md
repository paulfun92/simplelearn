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

It is easy to optimize loss functions with respect to arbitrary variables, be
they model parameters (training), input variables (inference), or
both. Optimizing w.r.t. inputs is useful for visualizing deep features, or
searching for pathological inputs that increase the loss function.

Currently Simplelearn only supports deterministic models (no RBMs).

Dependencies
------------
* [Theano](http://www.deeplearning.net/software/theano/)
* [nose](http://nose.readthedocs.org/en/latest/)

Documentation
-------------

Browsable API documentation is in docs/.

Beginners should start by browsing the complete working examples in examples/.

License
-------

The licence is Apache 2.0
[(full text)](https://www.apache.org/licenses/LICENSE-2.0.html)
[(wikipedia)](http://en.wikipedia.org/wiki/Apache_License)
[(tl:dr)](https://tldrlegal.com/license/apache-license-2.0-%28apache-2.0%29).
