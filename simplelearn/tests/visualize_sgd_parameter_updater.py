#! /usr/bin/env python

'''
Uses SgdParameterUpdater to perform gradient descent of a 2D point in a
quadratic energy basin, and plots the path of the point over time.
'''

import sys
import numpy
import theano
from matplotlib import pyplot
from nose.tools import (assert_equal,
                        assert_is_not,
                        assert_greater,
                        assert_is_instance)
from simplelearn.utils import safe_izip
from simplelearn.data import DataIterator
from simplelearn.formats import DenseFormat
from simplelearn.training import (TrainingMonitor,
                                  LimitsNumEpochs,
                                  SgdParameterUpdater,
                                  Sgd)
import pdb


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
                # Extends log along the batch axis, then copies batch into it.
                log.append(batch)
                # batch_axis = fmt.axes.index('b')
                # new_shape = list(log.shape)
                # new_shape[batch_axis] += batch.shape[batch_axis]
                # log.resize(new_shape, refcheck=False)
                # batch_slice = ((slice(None), ) * batch_axis +
                #                (slice(batch_axis, None), ) +
                #                (slice(None), ) * (log.ndim - 1 - batch_axis))
                # log[batch_slice] = batch

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
    # assert_op = theano.tensor.opt.Assert()
    # eq_op = theano.tensor.eq

    # covariance = assert_op(covariance, eq_op(covariance.ndim, 2))
    # batch_of_vectors = assert_op(batch_of_vectors,
    #                              eq_op(batch_of_vectors.ndim, 2))

    x_dot_c = theano.dot(batch_of_vectors, covariance)
    return (x_dot_c * batch_of_vectors).sum(axis=1)  #, keepdims=True)

def create_cost_batch(singular_values, angle, point):
    # assert_op = theano.tensor.opt.Assert()
    # eq_op = theano.tensor.eq

    # point = assert_op(point, eq_op(point.ndim, 2))

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


def optimize_without_trainer(point, update_function):
    '''
    Returns a Nx2 matrix where each row is the [x, y] coordinates of an
    iteration of gradient descent.
    '''

    num_iterations = 100
    outputs = numpy.zeros((num_iterations, 3))

    for output in outputs:
        output[:2] = point.get_value().flatten()
        output[2] = update_function()

    return outputs
    # return numpy.vstack((numpy.arange(3), numpy.arange(3))).T

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

    # assert_is_instance(trainer._epoch_callbacks[0], LimitsNumEpochs)

    return numpy.hstack((point_log, cost_log))
    # cost_log = recording_monitor.logs[1][:, numpy.newaxis]
    # return numpy.hstack((recording_monitor.logs[0], cost_log))


def main():
    point = theano.shared(numpy.zeros((1, 2), dtype=theano.config.floatX))
    cost_batch = create_cost_batch(singular_values=(10, 2),
                                   angle=numpy.pi/4,
                                   point=point)
    gradient = theano.gradient.grad(cost_batch.sum(), point)

    # can't use shared variable <point> as an explicit input to a function;
    # must tell it to replace the shared variable's value with some non-shared
    # variable's
    non_shared_point = theano.tensor.matrix(dtype=theano.config.floatX)
    cost_function = theano.function([non_shared_point, ],
                                    cost_batch,
                                    givens=[(point, non_shared_point)])

    floatX = numpy.dtype(theano.config.floatX)
    cast_to_floatX = numpy.cast[floatX]

    momenta = cast_to_floatX(numpy.linspace(0.0, 1.0, 3))
    learning_rates = cast_to_floatX(numpy.linspace(.001, .05, 4))

    figure, all_axes = pyplot.subplots(len(momenta),
                                       len(learning_rates),
                                       squeeze=False,
                                       figsize=(18, 6))

    # label subplot grid's rows with momenta
    pad = 5 # in points

    for axes, momentum in zip(all_axes[:, 0], momenta):
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

    def get_contour_data(cost_function, window_bound):
        grid_x, grid_y = (numpy.linspace(-window_bound, window_bound, 20)
                          for i in range(2))

        x_grid, y_grid = numpy.meshgrid(grid_x, grid_y)
        cast_to_floatX = numpy.cast[theano.config.floatX]
        x_grid, y_grid = (cast_to_floatX(a) for a in (x_grid, y_grid))

        xys = numpy.vstack((x_grid.flat, y_grid.flat)).T
        zs = cost_function(xys)

        result = (x_grid, y_grid, zs.reshape(x_grid.shape))

        return tuple(cast_to_floatX(a) for a in result)


    initial_point = numpy.array([1.3, .5], dtype=theano.config.floatX)
    contour_data = get_contour_data(cost_function,
                                    numpy.abs(initial_point).max() * 1.5)

    for axes in all_axes.flat:
        # pylint: disable=star-args
        axes.contour(*contour_data)
        axes.plot(initial_point[0], initial_point[1], 'gx')
        axes.plot(0, 0, 'rx')

    use_trainer = True

    for nesterov, line_style in safe_izip((False, True), ('r-,', 'g-,')):
        for momentum, axes_row in safe_izip(momenta, all_axes):
            for learning_rate, axes in safe_izip(learning_rates, axes_row):
                param_updater = SgdParameterUpdater(point,
                                                    gradient,
                                                    0,
                                                    0,
                                                    nesterov)
                param_updater.momentum.set_value(momentum)
                param_updater.learning_rate.set_value(learning_rate)

                point.set_value(initial_point[numpy.newaxis, :])

                if use_trainer:
                    recording_monitor = \
                        RecordingMonitor((point, cost_batch),
                                         (DenseFormat(axes=('b', 'f'),
                                                      shape=(-1, 2),
                                                      dtype=floatX),
                                          DenseFormat(axes=('b',),
                                                      shape=(-1,),
                                                      dtype=floatX)))

                    trainer = Sgd(cost=cost_batch.sum(),
                                  inputs=[],
                                  parameters=[point],
                                  parameter_updaters=[param_updater],
                                  input_iterator=DummyIterator(),
                                  monitors=[recording_monitor],
                                  epoch_callbacks=[LimitsNumEpochs(200)])

                    trajectory = optimize_with_trainer(trainer,
                                                       recording_monitor)
                else:
                    update_function = \
                        theano.function([],
                                        cost_batch,
                                        updates=param_updater.updates)
                    trajectory = optimize_without_trainer(point,
                                                          update_function)

                axes.plot(trajectory[:, 0], trajectory[:, 1], line_style)

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
