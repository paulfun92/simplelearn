#! /usr/bin/env python

'''
Steps through the iterations of a simple 2D->1D linear regression problem.
'''

import sys
import argparse
import numpy

from matplotlib import pyplot

# This line required for projection='3d' keyword to work
import mpl_toolkits.mplot3d   # pylint: disable=unused-import

from nose.tools import (assert_equal,
                        assert_greater)

import theano
from simplelearn.nodes import AffineTransform
from simplelearn.utils import safe_izip
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

    parser.add_argument("--testing-size",
                        type=positive_int,
                        default=10,
                        help=("The number of points in the testing set."))

    result = parser.parse_args()
    return result


def main():
    args = parse_args()

    floatX = numpy.dtype(theano.config.floatX)

    def affine_transform(matrix, bias, inputs):
        assert_equal(matrix.ndim, 2)
        assert_equal(bias.ndim, 2)
        assert_equal(bias.shape[0], matrix.shape[1])
        assert_equal(inputs.ndim, 2)

        result = numpy.dot(inputs, matrix) + bias
        assert_equal(result.shape, (inputs.shape[0], bias.shape[1]))
        return result

    def make_random_dataset(matrix,
                            bias,
                            min_input,
                            max_input,
                            output_variance,
                            rng,
                            num_points):
        num_dims = matrix.shape[0]
        assert_equal(len(min_input), num_dims)
        assert_equal(len(max_input), num_dims)

        inputs = numpy.vstack([rng.uniform(low=min_input[i],
                                           high=max_input[i],
                                           size=num_points)
                               for i in range(num_dims)]).T


        outputs = affine_transform(matrix, bias, inputs)

        output_noise = rng.normal(scale=output_variance, size=outputs.shape)

        return inputs, outputs + output_noise


    rng = numpy.random.RandomState(352351)

    # The "ground truth" affine transform
    matrix = rng.uniform(size=(2, 1))
    bias = rng.uniform(size=(1, 1))

    min_input = (-1.0, -1.0)
    max_input = (1.0, 1.0)
    output_variance = .1

    training_inputs, training_outputs = make_random_dataset(matrix,
                                                            bias,
                                                            min_input,
                                                            max_input,
                                                            output_variance,
                                                            rng,
                                                            args.training_size)

    testing_inputs, testing_outputs = make_random_dataset(matrix,
                                                          bias,
                                                          min_input,
                                                          max_input,
                                                          output_variance,
                                                          rng,
                                                          args.testing_size)

    def make_grid(matrix,
                  bias,
                  min_input,
                  max_input,
                  samples_per_dimension):
        assert_equal(matrix.shape, (2, 1))
        assert_equal(len(min_input), 2)
        assert_equal(len(max_input), 2)

        xs, ys = (numpy.linspace(min_input[i],
                                 max_input[i],
                                 samples_per_dimension)
                  for i in range(2))

        grid_xs, grid_ys = numpy.meshgrid(xs, ys)

        inputs = numpy.vstack((grid_xs.flat, grid_ys.flat)).T

        grid_zs = affine_transform(matrix,
                                   bias,
                                   inputs).reshape(grid_xs.shape)

        return grid_xs, grid_ys, grid_zs


    def get_bounding_cube(xs, ys, zs):
        '''
        Returns a tight bounding box around points given by xs, ys, and zs.

        Returns the x, y, and z coordinates of 8 corners of a tight
        bounding box around xs, ys, and zs. Plot these using some invisible
        color ('w') as a workaround to matplotlib's current inability to set
        the aspect ratio of 3d plots to be equal.
        '''

        mins = tuple(v.min() for v in (xs, ys, zs))
        max_range = max(v.max() - v.min() for v in (xs, ys, zs))

        cube_xs = []
        cube_ys = []
        cube_zs = []

        for xi in range(2):
            for yi in range(2):
                for zi in range(2):
                    cube_xs.append(mins[0] + max_range * xi)
                    cube_ys.append(mins[1] + max_range * yi)
                    cube_zs.append(mins[2] + max_range * zi)

        return tuple(numpy.asarray(vs) for vs in (cube_xs,
                                                  cube_ys,
                                                  cube_zs))

    figure = pyplot.gcf()
    figure.set_size_inches(18, 6, forward=True)
    points_axes = figure.add_subplot(1, 2, 1, projection='3d')

    # This does nothing in matplotlib 1.8.1.
    # It's apparently a long-standing bug.
    points_axes.set_aspect('equal')

    loss_axes = figure.add_subplot(1, 2, 2)

    ground_truth_plane = tuple(make_grid(matrix,
                                         bias,
                                         min_input,
                                         max_input,
                                         10))

    points_axes.plot_surface(*ground_truth_plane)

    z_ticks = 0.5 * numpy.arange(numpy.floor(ground_truth_plane[2].min() / 0.5),
                                 numpy.ceil(ground_truth_plane[2].max() / 0.5))
    points_axes.set_zticks(z_ticks)
    # A workaround to the fact that set_aspect('equal') doesn't yet work in
    # matplotlib 1.8.1. This was suggested here
    # bounding_cube = get_bounding_cube(*ground_truth_plane)
    # for bx, yx, zx in safe_izip(*bounding_cube):
    #     points_axes.plot([bx], [yx], [zx], 'w')

    def on_key_press(event):
        if event.key == 'q':
            sys.exit(0)

    figure.canvas.mpl_connect('key_press_event', on_key_press)

    pyplot.grid()
    pyplot.show()


if __name__ == '__main__':
    main()
