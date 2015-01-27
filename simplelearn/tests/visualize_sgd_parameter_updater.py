#! /usr/bin/env python

'''
Uses SgdParameterUpdater to perform gradient descent of a 2D point in a
quadratic energy basin, and plots the path of the point over time.
'''

import sys
import numpy
import theano
from matplotlib import pyplot
from nose.tools import assert_equal
from simplelearn.utils import safe_izip
from simplelearn.training import SgdParameterUpdater
import pdb


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
    return (x_dot_c * batch_of_vectors).sum(axis=1, keepdims=True)

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
    return batch_of_costs


def optimize(point, update_function):
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

def main():
    point = theano.shared(numpy.zeros((1, 2)))
    cost_batch = create_cost_batch(singular_values=(10, 2),
                                   angle=numpy.pi/4,
                                   point=point)
    gradient = theano.gradient.grad(cost_batch.sum(), point)

    # can't use shared variable <point> as an explicit input to a function;
    # must tell it to replace the shared variable's value with some non-shared
    # variable's
    non_shared_point = theano.tensor.dmatrix()
    cost_function = theano.function([non_shared_point, ],
                                    cost_batch,
                                    givens=[(point, non_shared_point)])

    cast_to_floatX = numpy.cast[theano.config.floatX]

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
        grid_shape = x_grid.shape
        xys = numpy.vstack((x_grid.flat, y_grid.flat)).T
        zs = cost_function(xys)
        # pdb.set_trace()
        return tuple((x_grid, y_grid, zs.reshape(grid_shape)))


    initial_point = numpy.array([1.3, .5])
    contour_data = get_contour_data(cost_function,
                                    numpy.abs(initial_point).max() * 1.5)

    for axes in all_axes.flat:
        # pylint: disable=star-args
        axes.contour(*contour_data)
        axes.plot(initial_point[0], initial_point[1], 'gx')
        axes.plot(0, 0, 'rx')

    for nesterov, line_style in safe_izip((False, True), ('r-,', 'g-,')):
        param_updater = SgdParameterUpdater(point,
                                            gradient,
                                            0,
                                            0,
                                            nesterov)

        update_function = theano.function([],
                                          cost_batch,
                                          updates=param_updater.updates)

        for momentum, axes_row in safe_izip(momenta, all_axes):
            param_updater.momentum.set_value(momentum)

            for learning_rate, axes in safe_izip(learning_rates, axes_row):
                param_updater.learning_rate.set_value(learning_rate)
                point.set_value(initial_point[numpy.newaxis, :])

                trajectory = optimize(point, update_function)

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
