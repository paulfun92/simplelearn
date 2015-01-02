#! /usr/bin/env python

"""
Defines a quadratic cost over 2-D space, places a point in that space, and runs
gradient descent to move the point down the cost well. Plots the path of the
point as it's optimized using various values of momentum.
"""

import argparse
from matplotlib import pyplot
from simplelearn.nodes import Node, InputNode, make_shared_variable
from simplelearn.trainers import Sgd, LimitNumEpochsCallback
from simplelearn.datasets import DatasetIterator


class QuadraticCost2D(Node):
    def __init__(self):
        output_format = DenseFormat(axes=('b', 'f'),
                                    shape=(-1, 2),
                                    dtype=float)

        # use same format for input and output
        super(QuadraticCost2D, self).__init__(input=InputNode(output_format),
                                              output_format=output_format)

    def _define_function(**input_nodes):
        assert_equal(tuple(input_nodes.iterkeys()), ('input',))

        rng = numpy.random.RandomState(4924)
        mat = rng.uniform(2, dtype=float)
        covariance = numpy.dot(mat.T, mat)

        # X: input batch
        input_batch = input_nodes['input'].get_outut_symbol()

        # L: lower-triangular cholesky factor (covariance := LL^T)
        sqrt_covariance = numpy.linear.cholesky(covariance)

        # XL
        input_times_covariance = theano.dot(input_batch, sqrt_covariance)

        results = (input_times_covariance * input_times_covariance).sum(axis=1)
        assert_equal(results.ndim, 2)

        return results

    def get_compiled_function(self):
        if not hasattr(self, 'function'):
            input_symbol = self.inputs['input'].output_symbol
            self.function = theano.function([input_symbol], self.output_symbol)

        return self.function


# class SingleDatumIterator(DatasetIterator):
#     pass


def get_sgd_trainer():
    cost = QuadraticCost2D()

    def get_input_shared_variable():
        input_format = cost.inputs['input'].output_format
        numeric_batch = input_format.make_batch(is_symbolic=False,
                                                batch_size=1)
        return make_shared_variable(numeric_batch, 'input batch')

    input_symbol = get_input_shared_variable()

    sgd = Sgd(data_iterator=NullDatasetIterator(),
              cost_symbol=cost.output_symbol,
              parameter_symbols=(input_symbol, ),
              parameter_updaters=updater,
              cost_input_symbols=tuple(),
              LimitNumEpochsCallback(100))


def main():

    def parse_args():
        parser = argparse.ArgumentParser(description=("Plots 2-D SGD with "
                                                      "different values of "
                                                      "momentum."))

    parse_args()

    cost = QuadraticCost2D()

    initial_point = numpy.array((-2., -2.))
    momenta = numpy.linspace(0, .9, .1)

    figure, all_axes = pyplot.subplots(1,
                                       len(momenta),
                                       squeeze=False,
                                       figsize=(10, 3.5))

    for momentum, axes in safe_izip(momenta, all_axes):
        plot_sgd(cost.get_compiled_function(), initial_point, momentum, axes)

    pyplot.show()

if __name__ == '__main__':
    main()
