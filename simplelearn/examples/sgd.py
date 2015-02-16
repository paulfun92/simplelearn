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
                        assert_is_not,
                        assert_greater,
                        assert_greater_equal,
                        assert_is_instance)
from simplelearn.utils import safe_izip
from simplelearn.data import DataIterator
from simplelearn.formats import DenseFormat
from simplelearn.training import (TrainingMonitor,
                                  LimitsNumEpochs,
                                  StopsOnStagnation,
                                  SgdParameterUpdater,
                                  Sgd)
import pdb


def parse_args():
    parser = argparse.ArgumentParser(
                description=("Simple demo of stochastic gradient descent, "
                             "with and without Nesterov's accelerated "
                             "gradients."))

    parser.add_argument("--use-trainer",
                        type=bool,
                        default=False,
                        help=("If True, use simplelearn.training.Sgd trainer. "
                              "If False, use a bare simplelearn.training."
                              "SgdParameterUpdater object. These two should "
                              "produce equivalent results if --lambda-scaling "
                              "and --momentum-scaling aren't provided."))

    parser.add_argument("--max-iters",
                        type=int,
                        default=100,
                        help=("Stop optimization after this many iterations."))

    parser.add_argument("--stagnation-iters",
                        type=int,
                        default=10,
                        help=("Stop optimization after this many iterations "
                              "without improving on the best cost thus far."))

    parser.add_argument("--stagnation-threshold",
                        type=int,
                        default=.0001,
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


class DummyIterator(DataIterator):
    def __init__(self):
        super(DummyIterator, self).__init__()

    def next_is_new_epoch(self):
        return True

    def _next(self):
        return ()

class RecordingMonitor(TrainingMonitor):
    '''
    Logs the values of given symbolic expressions on each training batch.
    '''

    def __init__(self, values_to_record, formats):
        super(RecordingMonitor, self).__init__(values_to_record, formats)
        self.logs = None
        self.epoch_start_indices = None
        self.clear()  # initializes logs, epoch_start_indices

    def _on_batch(self, _, value_batches):
        assert_greater(len(value_batches), 0)
        self.num_batches += 1

        if self.logs is None:
            self.logs = [[batch] for batch in value_batches]
        else:
            for log, batch in safe_izip(self.logs, value_batches):
                log.append(batch)

    def on_epoch(self):
        if self.logs is not None:
            self.epoch_start_indices.append(len(self.logs[0]))

    def clear(self):
        self.logs = None

        # Note that these are batch indices, NOT example indices.
        self.epoch_start_indices = [0]

        self.num_batches = 0


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
    return (x_dot_c * batch_of_vectors).sum(axis=1)  #, keepdims=True)

def create_cost_batch(singular_values, angle, point):
    # assert_op = theano.tensor.opt.Assert()
    # eq_op = theano.tensor.eq

    # point = assert_op(point, eq_op(point.ndim, 2))
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
    return theano.tensor.cast(batch_of_costs, theano.config.floatX)


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


def optimize_with_trainer(trainer, recording_monitor):
    recording_monitor.clear()
    trainer.train()

    assert_is_not(recording_monitor.logs, None)
    assert_equal(recording_monitor.logs[0][0].ndim, 2)
    assert_equal(recording_monitor.logs[1][0].ndim, 1)

    (point_log,
     cost_log) = (numpy.concatenate(log, axis=fmt.axes.index('b'))
                  for log, fmt in safe_izip(recording_monitor.logs,
                                            recording_monitor._formats))

    cost_log = cost_log[:, numpy.newaxis]

    return numpy.hstack((point_log, cost_log))


def main():
    args = parse_args()

    point = theano.shared(numpy.zeros((1, 2), dtype=theano.config.floatX))
    cost_batch = create_cost_batch(singular_values=args.singular_values,
                                   angle=args.angle,
                                   point=point)
    cost_batch.name = 'cost_batch'

    cost_sum = cost_batch.sum()
    cost_sum.name = 'cost_sum'

    gradient = theano.gradient.grad(cost_sum, point)

    # can't use shared variable <point> as an explicit input to a function;
    # must tell it to replace the shared variable's value with some non-shared
    # variable's
    non_shared_point = theano.tensor.matrix(dtype=theano.config.floatX)
    cost_function = theano.function([non_shared_point, ],
                                    cost_batch,
                                    givens=[(point, non_shared_point)])

    floatX = numpy.dtype(theano.config.floatX)
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
    pad = 5 # in points

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

        x_grid, y_grid = numpy.meshgrid(grid_x, grid_y)
        cast_to_floatX = numpy.cast[theano.config.floatX]
        x_grid, y_grid = (cast_to_floatX(a) for a in (x_grid, y_grid))

        xys = numpy.vstack((x_grid.flat, y_grid.flat)).T
        zs = cost_function(xys)

        result = (x_grid, y_grid, zs.reshape(x_grid.shape))

        return tuple(cast_to_floatX(a) for a in result)


    initial_point = numpy.array([1.3, .5], dtype=theano.config.floatX)
    aspect_ratio = float(figsize[1]) / float(figsize[0])
    aspect_ratio *= (float(len(momenta)) / float(len(learning_rates)))
    x_limit = numpy.sqrt((initial_point**2).sum()) * 2.0
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
                    recording_monitor = \
                        RecordingMonitor((point, cost_batch),
                                         (DenseFormat(axes=('b', 'f'),
                                                      shape=(-1, 2),
                                                      dtype=floatX),
                                          DenseFormat(axes=('b',),
                                                      shape=(-1,),
                                                      dtype=floatX)))

                    monitors = [recording_monitor,
                                StopsOnStagnation(cost_batch,
                                                  DenseFormat(axes=['b'],
                                                              shape=[-1],
                                                              dtype=None),
                                                  args.stagnation_iters,
                                                  args.stagnation_threshold)]
                                                  # num_epochs=10,
                                                  # min_decrease=.0001)]

                    trainer = Sgd(cost=cost_sum,
                                  inputs=[],
                                  parameters=[point],
                                  parameter_updaters=[param_updater],
                                  input_iterator=DummyIterator(),
                                  monitors=monitors,
                                  epoch_callbacks=[
                                      LimitsNumEpochs(args.max_iters)])

                    trajectory = optimize_with_trainer(trainer,
                                                       recording_monitor)
                else:
                    update_function = \
                        theano.function([],
                                        cost_batch,
                                        updates=param_updater.updates)
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
