#! /usr/bin/env python

'''
Uses SgdParameterUpdater to perform gradient descent of a 2D point in a
quadratic energy basin, and plots the path of the point over time.
'''

import sys, argparse
import numpy
import theano
from matplotlib import pyplot
from nose.tools import (assert_equal,
                        assert_greater,
                        assert_greater_equal)
from simplelearn.utils import safe_izip
from simplelearn.data import DummyIterator
from simplelearn.formats import DenseFormat
from simplelearn.nodes import Node
from simplelearn.training import (MeanOverEpoch,
                                  LogsToLists,
                                  LimitsNumEpochs,
                                  StopsOnStagnation,
                                  SgdParameterUpdater,
                                  Sgd)
import pdb

floatX = theano.config.floatX  # pylint: disable=no-member


def parse_args():
    '''
    Returns command-line args as a namespace.
    '''

    parser = argparse.ArgumentParser(
        description=("Simple demo of stochastic gradient descent, "
                     "with and without Nesterov's accelerated "
                     "gradients."))

    parser.add_argument("--use-trainer",
                        type=bool,
                        default=True,
                        help=("If True, uses the simplelearn.training.Sgd "
                              "trainer. If False, uses a bare "
                              "simplelearn.training.SgdParameterUpdater "
                              "object, and --stagnation-threshold is "
                              "ignored."))

    parser.add_argument("--max-iters",
                        type=int,
                        default=100,
                        help=("Stop optimization after this many iterations."))

    parser.add_argument("--stagnation-iters",
                        type=int,
                        default=50,
                        help=("Stop optimization after this many iterations "
                              "without improving on the best cost thus far."))

    parser.add_argument("--stagnation-threshold",
                        type=int,
                        default=0.001,
                        help=("Stop optimization if the cost doesn't improve "
                              "more than this, after <stagnation_iters> "
                              "iterations."))

    parser.add_argument("--lambda-range",
                        nargs=2,
                        default=[0.001, .05],
                        help=("Min & max learning rate values."))

    parser.add_argument("--momentum-range",
                        nargs=2,
                        default=[0.0, 1.0],
                        help=("Min & max momentum values."))

    parser.add_argument("--angle",
                        type=float,
                        default=45,
                        help=("Angle, in degrees, by which to rotate the "
                              "cost measurement matrix."))

    parser.add_argument("--singular-values",
                        nargs=2,
                        default=[10, 2],
                        help=("The singular values of the cost measurement "
                              "matrix."))

    result = parser.parse_args()

    assert_greater(result.max_iters, 0)

    assert_greater(result.stagnation_iters, 0)
    assert_greater_equal(result.stagnation_threshold, 0.0)

    assert_greater(result.lambda_range[0], 0.0)
    assert_greater(result.lambda_range[1], result.lambda_range[0])

    assert_greater_equal(result.momentum_range[0], 0.0)
    assert_greater(result.momentum_range[1], result.momentum_range[0])

    for singular_value in result.singular_values:
        assert_greater(singular_value, 0.0)

    return result


def matrix_weighted_norm(covariance, batch_of_vectors):
    '''
    Returns v^T C v for a batch of vectors v.

    Parameters
    ----------
    covariance: theano.gof.Variable
      Represents an NxN covariance matrix.

    batch_of_vectors: theano.gof.Variable
      Represents a batch of M vectors, as a MxN matrix.

    Returns
    -------
    rval: theano.gof.Variable
      A Mx1 matrix, storing the v^T C v for the v's in batch_of_vectors' rows.
    '''
    assert_equal(covariance.ndim, 2)

    x_dot_c = theano.dot(batch_of_vectors, covariance)
    return (x_dot_c * batch_of_vectors).sum(axis=1)  # , keepdims=True)


def create_cost_batch(singular_values, angle, point):
    '''
    Returns squared mahalanobis norm V^T M V, where V is a point.

    Actually, V is a Nx2 matrix of N 2D points, and this returns a vector
    of norms.

    Parameters
    ----------
    singular_values: numpy.ndarray
      Singular values of the energy basin. shape=(2,)

    angle: float
      Rotation angle of the energy basin.
    '''
    assert_equal(point.ndim, 2)
    covariance = numpy.diag(singular_values)

    def make_rotation_matrix(angle):
        sa = numpy.sin(angle)
        ca = numpy.cos(angle)
        return numpy.array([[ca, -sa],
                            [sa, ca]])

    rotation_matrix = make_rotation_matrix(angle)

    # C <- R C R^T
    covariance = numpy.dot(rotation_matrix,
                           numpy.dot(covariance, rotation_matrix.T))

    batch_of_costs = matrix_weighted_norm(covariance, point)
    return theano.tensor.cast(batch_of_costs, floatX)


def optimize_without_trainer(point, update_function, num_iterations):
    '''
    Returns a Nx2 matrix where each row is the [x, y] coordinates of an
    iteration of gradient descent.
    '''

    outputs = numpy.zeros((num_iterations, 3))

    for output in outputs:
        output[:2] = point.get_value().flatten()
        output[2] = update_function()

    return outputs


def optimize_with_trainer(trainer,
                          point_logger,
                          loss_logger,
                          point_format,
                          loss_format):
    trainer.train()

    point_log, cost_log = (numpy.concatenate(logger.log,
                                             axis=fmt.axes.index('b'))
                           for logger, fmt
                           in safe_izip([point_logger, loss_logger],
                                        [point_format, loss_format]))

    cost_log = cost_log[:, numpy.newaxis]
    try:
        return numpy.hstack((point_log, cost_log))
    except ValueError:
        pdb.set_trace()


def main():
    args = parse_args()

    point = theano.shared(numpy.zeros((1, 2), dtype=floatX))
    cost_batch = create_cost_batch(singular_values=args.singular_values,
                                   angle=args.angle,
                                   point=point)
    cost_batch.name = 'cost_batch'

    gradient = theano.gradient.grad(cost_batch.sum(), point)

    # can't use shared variable <point> as an explicit input to a function;
    # must tell it to replace the shared variable's value with some non-shared
    # variable's
    non_shared_point = theano.tensor.matrix(dtype=floatX)
    cost_function = theano.function([non_shared_point, ],
                                    cost_batch,
                                    givens=[(point, non_shared_point)])

    cast_to_floatX = numpy.cast[floatX]

    learning_rates = cast_to_floatX(numpy.linspace(args.lambda_range[0],
                                                   args.lambda_range[1],
                                                   4))
    momenta = cast_to_floatX(numpy.linspace(args.momentum_range[0],
                                            args.momentum_range[1],
                                            3))

    figsize = (18, 12)

    figure, all_axes = pyplot.subplots(len(momenta) * 2,
                                       len(learning_rates),
                                       squeeze=False,
                                       figsize=figsize)

    # label subplot grid's rows with momenta
    pad = 5  # in points (the typographical unit of length)

    for axes, momentum in zip(all_axes[::2, 0], momenta):
        axes.annotate("momentum=%g" % momentum,
                      xy=(0, 0.5),
                      xytext=(-axes.yaxis.labelpad - pad, 0),
                      xycoords=axes.yaxis.label,
                      textcoords='offset points',
                      size='large',
                      ha='right',
                      va='center')

    for axes, learning_rate in zip(all_axes[0], learning_rates):
        axes.annotate("lambda=%g" % learning_rate,
                      xy=(0.5, 1),
                      xytext=(0, pad),
                      xycoords='axes fraction',
                      textcoords='offset points',
                      size='large',
                      ha='center',
                      va='baseline')

    def get_contour_data(cost_function, max_x, max_y):
        grid_x = numpy.linspace(-max_x, max_x, 20)
        grid_y = numpy.linspace(-max_y, max_y, 20)

        # pylint: disable=unbalanced-tuple-unpacking
        x_grid, y_grid = numpy.meshgrid(grid_x, grid_y)
        cast_to_floatX = numpy.cast[floatX]
        x_grid, y_grid = (cast_to_floatX(a) for a in (x_grid, y_grid))

        xys = numpy.vstack((x_grid.flat, y_grid.flat)).T
        zs = cost_function(xys)

        result = (x_grid, y_grid, zs.reshape(x_grid.shape))

        return tuple(cast_to_floatX(a) for a in result)

    initial_point = numpy.array([1.3, .5], dtype=floatX)
    aspect_ratio = float(figsize[1]) / float(figsize[0])
    aspect_ratio *= (float(len(momenta)) / float(len(learning_rates)))
    x_limit = numpy.sqrt((initial_point ** 2).sum()) * 2.0
    contour_data = get_contour_data(cost_function,
                                    x_limit,
                                    x_limit * aspect_ratio)

    for axes_row in all_axes[::2, :]:
        for axes in axes_row:
            # pylint: disable=star-args
            axes.contour(*contour_data)
            axes.plot(initial_point[0], initial_point[1], 'gx')
            axes.plot(0, 0, 'rx')

    for nesterov, line_style in safe_izip((False, True), ('r-,', 'g-,')):
        for momentum, axes_row, cost_axes_row in safe_izip(momenta,
                                                           all_axes[::2, :],
                                                           all_axes[1::2, :]):
            for learning_rate, axes, cost_axes in safe_izip(learning_rates,
                                                            axes_row,
                                                            cost_axes_row):
                param_updater = SgdParameterUpdater(point,
                                                    gradient,
                                                    0,
                                                    0,
                                                    nesterov)
                param_updater.momentum.set_value(momentum)
                param_updater.learning_rate.set_value(learning_rate)

                point.set_value(initial_point[numpy.newaxis, :])

                if args.use_trainer:
                    point_format = DenseFormat(axes=['b', 'f'],
                                               shape=[-1, 2],
                                               dtype=floatX)
                    point_node = Node(input_nodes=[],
                                      output_symbol=point,
                                      output_format=point_format)

                    point_logger = LogsToLists()

                    cost_format = DenseFormat(axes=['b'],
                                              shape=[-1],
                                              dtype=floatX)
                    cost_node = Node(input_nodes=[],
                                     output_symbol=cost_batch,
                                     output_format=cost_format)
                    cost_logger = LogsToLists()

                    cost_monitor = MeanOverEpoch(
                        cost_node,
                        [cost_logger,
                         StopsOnStagnation(
                             max_epochs=args.stagnation_iters,
                             min_proportional_decrease=args.stagnation_threshold)])


                    # We're not actually computing the mean over the epoch, but
                    # the point itself. When batch size is 1, they're the same
                    # thing.
                    point_monitor = MeanOverEpoch(point_node, [point_logger])
                    # cost_monitor = MeanOverEpoch(cost_node, [cost_logger])

                    trainer = Sgd(inputs=[],
                                  input_iterator=DummyIterator(),
                                  callbacks=[param_updater,
                                             point_monitor,
                                             cost_monitor,
                                             LimitsNumEpochs(args.max_iters)])

                    trajectory = optimize_with_trainer(trainer,
                                                       point_logger,
                                                       cost_logger,
                                                       point_format,
                                                       cost_format)
                else:
                    update_function = \
                        theano.function([],
                                        cost_batch,
                                        updates=param_updater.update_pairs)
                    trajectory = optimize_without_trainer(point,
                                                          update_function,
                                                          args.max_iters)

                axes.set_aspect('equal')
                axes.plot(trajectory[:, 0], trajectory[:, 1], line_style)

                # cost_axes.set_yscale('log')  # <-- this was a bad idea
                cost_axes.plot(numpy.arange(trajectory.shape[0]),
                               trajectory[:, 2],
                               line_style)

    # figure.tight_layout()
    figure.subplots_adjust(left=0.15, top=0.95)

    # add GUI callbacks
    def on_key_press(event):
        if event.key == 'q':
            sys.exit(0)

    figure.canvas.mpl_connect('key_press_event', on_key_press)

    pyplot.show()

if __name__ == '__main__':
    main()
