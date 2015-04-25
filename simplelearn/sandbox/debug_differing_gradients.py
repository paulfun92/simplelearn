#! /usr/bin/env python

from pylearn2.models.mlp import MLP, Linear

def main():
    weights = numpy.asarray([[1, 2, 3],
                             [4, 5, 6]], dtype='floatX').T

    input_batch = numpy.asarray([[0, 1, 2]], dtype='floatX')
    label_batch = numpy.asarray([0, 1], dtype='int32')

    onehot_batch = numpy.zeros([input_batch.shape[0],
                                weights.shape[1]], dtype='floatX')
    onehot_batch[range(batch_size), label_batch] = 1.0

    pylearn2_mlp = make_pylearn2_mlp([weights])
    simplelearn_mlp = make_simplelearn_mlp


    # get cost_from_X from pylearn2_mlp

    # Build cross-entropy cost for simplelearn

    # compute gradients, compare



if __name__ == '__main__':
    main()
