#! /usr/bin/env python

'''
Steps through the iterations of a simple 2D->1D linear regression problem.
'''

import argparse

from matplotlib import pyplot
from nose.tools import (assert_equal,
                        assert_is_not,
                        assert_greater,
                        assert_greater_equal,
                        assert_is_instance)
import theano
from simplelearn.utils import check_is_subdtype

import pdb


def parse_args():
    parser = argparse.ArgumentParser(
                description=("Simple demo of stochastic gradient descent, "
                             "with and without Nesterov's accelerated "
                             "gradients."))

    def positive_int(arg):
        arg = int(arg)
        assert_greater(arg, 0)

        return arg

    parser.add_argument("--training-size",
                        type=positive_int,
                        default=100,
                        help=("The number of points in the training set"))

    parser.add_argument("--validation-size",
                        type=positive_int,
                        default=10,
                        help=("The number of points in the validation set."))

    result = parser.parse_args()
    return result


def main():
    args = parse_args()

    floatX = numpy.dtype(theano.config.floatX)

    ground_truth_node = AffineTransform(2, 1, dtype=floatX)
    ground_truth_node.weights.set_value(


if __name__ == '__main__':
    main()
