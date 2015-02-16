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
                        assert_greater,
                        assert_greater_equal)

import theano
from simplelearn.nodes import AffineTransform, AverageL2Loss
from simplelearn.utils import safe_izip
from simplelearn.data.dataset import Dataset
from simplelearn.formats import DenseFormat
from simplelearn.training import (SgdParameterUpdater,
                                  Sgd,
                                  LimitsNumEpochs,
                                  TrainingMonitor)
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

    def positive_float(arg):
        arg = float(arg)
        assert_greater(arg, 0)

        return arg

    def non_negative_float(arg):
        arg = float(arg)
        assert_greater_equal(arg, 0)

        return arg

    parser.add_argument("--training-size",
                        type=positive_int,
                        default=100,
                        help=("The number of points in the training set"))

    parser.add_argument("--testing-size",
                        type=positive_int,
                        default=10,
                        help=("The number of points in the testing set."))

    parser.add_argument("--learning-rate",
                        type=positive_float,
                        default=.1,
                        help="Learning rate for SGD")

    parser.add_argument("--momentum",
                        type=non_negative_float,
                        default=.5,
                        help="Momentum for SGD")

    parser.add_argument("--nesterov",
                        type=bool,
                        default=True,
                        help="Set to True to use Nesterov momentum.")

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
                            num_points,
                            dtype):
        assert_greater_equal(output_variance, 0.0)

        num_dims = matrix.shape[0]
        assert_equal(len(min_input), num_dims)
        assert_equal(len(max_input), num_dims)

        inputs = numpy.vstack([rng.uniform(low=min_input[i],
                                           high=max_input[i],
                                           size=num_points)
                               for i in range(num_dims)]).T

        outputs = affine_transform(matrix, bias, inputs)

        output_noise = (numpy.zeros(outputs.shape, dtype=dtype)
                        if output_variance == 0.0
                        else rng.normal(scale=output_variance,
                                        size=outputs.shape))

        cast = numpy.cast[dtype]

        return cast(inputs), cast(outputs + output_noise)

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
                                                            args.training_size,
                                                            floatX)

    testing_inputs, testing_outputs = make_random_dataset(matrix,
                                                          bias,
                                                          min_input,
                                                          max_input,
                                                          output_variance,
                                                          rng,
                                                          args.testing_size,
                                                          floatX)

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

    figure = pyplot.gcf()
    figure.set_size_inches(18, 6, forward=True)

    points_axes = figure.add_subplot(1, 2, 1, projection='3d')
    points_axes.set_aspect('equal')

    loss_axes = figure.add_subplot(1, 2, 2)

    ground_truth_plane = tuple(make_grid(matrix,
                                         bias,
                                         min_input,
                                         max_input,
                                         10))

    points_axes.plot_surface(ground_truth_plane[0],
                             ground_truth_plane[1],
                             ground_truth_plane[2],
                             color=[0, 1, 0, 0.5])  # translucent

    z_ticks = (0.5 *
               numpy.arange(numpy.floor(ground_truth_plane[2].min() / 0.5),
                            numpy.ceil(ground_truth_plane[2].max() / 0.5)))
    points_axes.set_zticks(z_ticks)

    points_axes.scatter(training_inputs[:, 0],
                        training_inputs[:, 1],
                        training_outputs,
                        c='blue')

    points_axes.scatter(testing_inputs[:, 0],
                        testing_inputs[:, 1],
                        testing_outputs,
                        c='red')

    training_set = Dataset(names=('inputs', 'targets'),
                           formats=(DenseFormat(axes=('b', 'f'),
                                                shape=(-1, 2),
                                                dtype=floatX),
                                    DenseFormat(axes=['b', 'f'],
                                                shape=[-1, 1],
                                                dtype=floatX)),
                           tensors=(training_inputs, training_outputs))

    input_node, label_node = training_set.make_input_nodes()

    affine_node = AffineTransform(input_node=input_node,
                                  output_format=DenseFormat(axes=['b', 'f'],
                                                            shape=[-1, 1],
                                                            dtype=None))

    cost = AverageL2Loss(affine_node, label_node)
    grad = theano.gradient.grad

    #DEBUG
    # cast = numpy.cast[affine_node.linear_node.params.get_value().dtype]
    # affine_node.linear_node.params.set_value(cast(matrix))
    # affine_node.bias_node.params.set_value(cast(bias))

    parameter_updaters = [SgdParameterUpdater(p,
                                              grad(cost.output_symbol, p),
                                              args.learning_rate / training_outputs.shape[0],
                                              args.momentum,
                                              args.nesterov)
                          for p in (affine_node.linear_node.params,
                                    affine_node.bias_node.params)]


    # DEBUG
    # affine_node.bias_node.params.set_value([[0.5]])
    # parameter_updaters = [SgdParameterUpdater(affine_node.linear_node.params,
    #                                           grad(cost.output_symbol,
    #                                                affine_node.linear_node.params),
    #                                           args.learning_rate / training_outputs.shape[0],
    #                                           args.momentum,
    #                                           args.nesterov)]

    # pdb.set_trace()

    def plot_model_surface():
        xs, ys, zs = \
            make_grid(affine_node.linear_node.params.get_value(),
                      affine_node.bias_node.params.get_value(),
                      min_input,
                      max_input,
                      10)
        model_surface = points_axes.plot_surface(xs, ys, zs,
                                                 color=[1, .5, .5, .5])
        print("model normal: %s, %s" %
              (str(affine_node.linear_node.params.get_value().flatten()),
               str(affine_node.bias_node.params.get_value().flatten())))
        return model_surface

    model_surface = [plot_model_surface()]

    class ModelSurfaceReplotter(TrainingMonitor):

        def __init__(self):
            super(ModelSurfaceReplotter, self).__init__([], [])

        def _on_batch(self, input_batches, monitored_value_batches):
            # pdb.set_trace()
            model_surface[0].remove()
            del model_surface[0]
            model_surface.append(plot_model_surface())
            # points_axes.autoscale_view(tight=True)
            # points_axes.autoscale(enable=True, axis='both')
            print("updated")
            points_axes.set_zlim(bottom=-1, top=1)

            # points_axes.relim()
            #points_axes.draw_idle()
            figure.canvas.draw()

        def on_epoch(self):
            pass

    class BatchPrinter(TrainingMonitor):

        def __init__(self):
            super(BatchPrinter, self).__init__([], [])
            self.batch_number = 0
            self.epoch_number = -1

            inputs = [input_node.output_symbol,
                      label_node.output_symbol]

            self.lin_func = theano.function(inputs[:1], affine_node.linear_node.output_symbol)
            self.bias_func = theano.function(inputs[:1], affine_node.bias_node.output_symbol)
            self.affine_func = theano.function(inputs[:1], affine_node.output_symbol)
            self.cost_func = theano.function(inputs, cost.output_symbol)

        def _on_batch(self, input_batches, monitored_value_batches):
            assert_equal(len(input_batches), 2)

            input_batches = tuple(b[:, numpy.newaxis, :]
                                  for b in input_batches)

            print("\nbatch %d:\n" % self.batch_number)

            # print("batch cost: %s" % self.cost_func(*input_batches))
            for input, label in safe_izip(*input_batches):
                print("x: %s L(x): %s B(L(x)): %s tgt: %s cost: %s" %
                      (str(input.flatten()),
                       str(self.lin_func(input).flatten()),
                       str(self.bias_func(input).flatten()),
                       str(label.flatten().flatten()),
                       str(self.cost_func(input, label))))

            self.batch_number += 1

        def on_epoch(self):
            self.batch_number = 0
            self.epoch_number += 1

            print("ending epoch %d" % self.epoch_number)

    sgd = Sgd(cost=cost.output_symbol,
              inputs=[n.output_symbol for n in (input_node, label_node)],
              # parameters=[],
              # parameters=[affine_node.linear_node.params],
              parameters=[affine_node.linear_node.params,
                          affine_node.bias_node.params],
              # parameter_updaters=[],
              parameter_updaters=parameter_updaters,
              input_iterator=training_set.iterator(
                  iterator_type='sequential',
                  batch_size=training_outputs.shape[0]),
                  #batch_size=2),
              # monitors=[ModelSurfaceReplotter(), BatchPrinter()],
              monitors=[ModelSurfaceReplotter()],
              epoch_callbacks=[LimitsNumEpochs(100)])  # revert to 100 once we've debugged

    def on_key_press(event):
        if event.key == 'q':
            sys.exit(0)
        if event.key == ' ':
            sgd.train()

    figure.canvas.mpl_connect('key_press_event', on_key_press)

    pyplot.grid()
    pyplot.show()


if __name__ == '__main__':
    main()
