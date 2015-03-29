#! /usr/bin/env python

'''
Demonstrates training a convolutional net on MNIST.
'''

import os
import argparse
import numpy
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict
from nose.tools import (assert_true,
                        assert_is_instance,
                        assert_greater,
                        assert_greater_equal,
                        assert_less_equal,
                        assert_equal)
from simplelearn.nodes import (Node,
                               Conv2D,
                               Pool2D,
                               ReLU,
                               Dropout,
                               AffineTransform,
                               CrossEntropy,
                               Misclassification,
                               Softmax,
                               RescaleImage,
                               FormatNode)
from simplelearn.utils import (safe_izip,
                               assert_floating,
                               assert_all_equal,
                               assert_all_greater,
                               assert_all_less_equal,
                               assert_all_integers)
from simplelearn.io import SerializableModel
from simplelearn.data.mnist import load_mnist
from simplelearn.formats import DenseFormat
from simplelearn.training import (SgdParameterUpdater,
                                  limit_param_norms,
                                  Sgd,
                                  LogsToLists,
                                  SavesAtMinimum,
                                  Monitor,
                                  AverageMonitor,
                                  LimitsNumEpochs,
                                  LinearlyInterpolatesOverEpochs,
                                  PicklesOnEpoch,
                                  ValidationCallback,
                                  StopsOnStagnation)
import pdb


def parse_args():
    '''
    Parses the command-line args.
    '''

    parser = argparse.ArgumentParser(
        description=("Trains multilayer perceptron to classify MNIST digits. "
                     "Default arguments are the ones used by Pylearn2's mlp "
                     "tutorial #3."))

    # pylint: disable=missing-docstring

    def positive_float(arg):
        result = float(arg)
        assert_greater(result, 0.0)
        return result

    def non_negative_float(arg):
        result = float(arg)
        assert_greater_equal(result, 0.0)
        return result

    def non_negative_int(arg):
        result = int(arg)
        assert_greater_equal(result, 0)
        return result

    def positive_int(arg):
        result = int(arg)
        assert_greater(result, 0)
        return result

    def legit_prefix(arg):
        abs_path = os.path.abspath(arg)
        assert_true(os.path.isdir(os.path.split(abs_path)[0]))
        assert_equal(os.path.splitext(abs_path)[1], "")
        return arg

    def positive_0_to_1(arg):
        result = float(arg)
        assert_greater(result, 0.0)
        assert_less_equal(result, 1.0)
        return result

    parser.add_argument("--output-prefix",
                        type=legit_prefix,
                        required=True,
                        help=("Directory and optional prefix of filename to "
                              "save the log to."))

    # Most of the default hyperparameter values below are taken from
    # Pylearn2, in pylearn2/scripts/tutorials/multilayer_perceptron/:
    #   multilayer_perceptron.ipynb
    #   mlp_tutorial_part_3.yaml
    #
    # Exceptions were made, since the original hyperparameters led to
    # divergence. These changes have been marked below.

    parser.add_argument("--learning-rate",
                        type=positive_float,
                        default=0.005,  # .01 used in pylearn2 demo
                        help=("Learning rate."))

    parser.add_argument("--initial-momentum",
                        type=non_negative_float,
                        default=0.5,
                        help=("Initial momentum."))

    parser.add_argument("--no-nesterov",
                        default=False,  # True used in pylearn2 demo
                        action="store_true",
                        help=("Don't use Nesterov accelerated gradients "
                              "(default: False)."))

    parser.add_argument("--batch-size",
                        type=non_negative_int,
                        default=100,
                        help="batch size")

    parser.add_argument("--dropout-include-rates",
                        default=(1.0, 1.0),  # i.e. no dropout
                        type=positive_0_to_1,
                        nargs=2,
                        help=("The dropout include rates for the outputs of "
                              "the first two layers. Must be in the range "
                              "(0.0, 1.0]. If 1.0, the Dropout node will "
                              "simply be omitted. For no dropout, use "
                              "1.0 1.0 (this is the default). Make sure to "
                              "lower the learning rate when using dropout. "
                              "I'd suggest a learning rate of 0.001 for "
                              "dropout-include-rates of 0.5 0.5."))

    parser.add_argument("--final-momentum",
                        type=positive_0_to_1,
                        default=.5,  # .99 used in pylearn2 demo
                        help="Value for momentum to linearly scale up to.")

    parser.add_argument("--epochs-to-momentum-saturation",
                        default=10,
                        type=positive_int,
                        help=("# of epochs until momentum linearly scales up "
                              "to --momentum_final_value."))

    max_norm = 1.9365 / 2  # 1.9365 used in pylearn2 demo

    parser.add_argument("--max-filter-norm",
                        type=positive_float,
                        default=max_norm,
                        help="Max. L2 norm of the convolutional filters.")

    parser.add_argument("--max-col-norm",
                        type=positive_float,
                        default=max_norm,
                        help="Max. L2 norm of weight matrix columns.")

    return parser.parse_args()


def build_conv_classifier(input_node,
                          filter_shapes,
                          filter_counts,
                          filter_init_uniform_ranges,
                          pool_shapes,
                          pool_strides,
                          affine_output_sizes,
                          affine_init_stddevs,
                          dropout_include_rates,
                          rng,
                          theano_rng):
    '''
    Builds a classification convnet on top of input_node.

    Returns
    -------
    rval: tuple
      (conv_nodes, affine_nodes, output_node), where:
         conv_nodes is a list of the Conv2D nodes.
         affine_nodes is a list of the AffineNodes.
         output_node is the final node, a Softmax.
    '''

    assert_is_instance(input_node, RescaleImage)

    conv_shape_args = (filter_shapes,
                       pool_shapes,
                       pool_strides)

    for conv_shapes in conv_shape_args:
        for conv_shape in conv_shapes:
            assert_all_integers(conv_shape)
            assert_all_greater(conv_shape, 0)

    conv_args = conv_shape_args + (filter_counts, filter_init_uniform_ranges)
    assert_all_equal([len(c) for c in conv_args])

    assert_equal(len(affine_output_sizes), len(affine_init_stddevs))

    assert_equal(len(dropout_include_rates),
                 len(filter_shapes) + len(affine_output_sizes) - 1)

    assert_equal(affine_output_sizes[-1], 10)  # for MNIST

    #
    # Converts from MNIST's ('b', '0', '1') to ('b', 'c', '0', '1')
    #

    assert_equal(input_node.output_format.axes, ('b', '0', '1'))

    input_shape = input_node.output_format.shape

    last_node = FormatNode(
        input_node,
        DenseFormat(axes=('b', 'c', '0', '1'),
                    shape=(input_shape[0],
                           1,
                           input_shape[1],
                           input_shape[2]),
                    dtype=None),
        {'1': ('1', 'c')})

    conv_dropout_include_rates = \
        dropout_include_rates[:len(filter_shapes)]

    # last_node = input_node

    #
    # Adds a conv-relu-maxpool-dropout stack for each element in filter_XXXX
    #

    conv_nodes = []

    def uniform_init(rng, params, init_range):
        '''
        Fills params with values uniformly sampled from
        [-init_range, init_range]
        '''

        assert_floating(init_range)
        assert_greater_equal(init_range, 0)

        values = params.get_value()
        values[...] = rng.uniform(low=-init_range,
                                  high=init_range,
                                  size=values.shape)
        params.set_value(values)

    for (filter_shape,
         filter_count,
         filter_init_range,
         pool_shape,
         pool_stride,
         conv_dropout_include_rate) in safe_izip(filter_shapes,
                                                 filter_counts,
                                                 filter_init_uniform_ranges,
                                                 pool_shapes,
                                                 pool_strides,
                                                 conv_dropout_include_rates):
        last_node = Conv2D(last_node,
                           filter_shape,
                           filter_count,
                           pads='valid')
        uniform_init(rng, last_node.filters, filter_init_range)
        conv_nodes.append(last_node)

        last_node = ReLU(last_node)

        last_node = Pool2D(last_node, pool_shape, pool_stride, mode='max')

        if conv_dropout_include_rate != 1.0:
            last_node = Dropout(last_node,
                                conv_dropout_include_rate,
                                theano_rng)

    affine_dropout_include_rates = \
        dropout_include_rates[len(filter_shapes):] + [None]

    affine_nodes = []

    def normal_distribution_init(rng, params, stddev):
        '''
        Fills params with values uniformly sampled from
        [-init_range, init_range]
        '''

        assert_floating(stddev)
        assert_greater_equal(stddev, 0)

        values = params.get_value()
        values[...] = rng.standard_normal(values.shape) * stddev
        params.set_value(values)

    #
    # Adds an affine-relu-dropout stack for each element in affine_XXXX,
    # except for the last one, where it omits the dropout.
    #

    for (affine_size,
         affine_init_stddev,
         affine_dropout_include_rate) in \
        safe_izip(affine_output_sizes,
                  affine_init_stddevs,
                  affine_dropout_include_rates):

        # The first affine node needs an axis map to collapse a feature map
        # (axes: 'b', 'c', '0', '1') into a feature vector (axes: 'b', 'f')
        axis_map = ({('c', '0', '1'): 'f'}
                    if len(affine_nodes) == 0
                    else None)

        last_node = AffineTransform(last_node,
                                    DenseFormat(axes=('b', 'f'),
                                                shape=(-1, affine_size),
                                                dtype=None),
                                    input_to_bf_map=axis_map)
        normal_distribution_init(rng,
                                 last_node.linear_node.params,
                                 affine_init_stddev)
        # stddev_init(rng, last_node.bias_node.params, affine_init_stddev)
        affine_nodes.append(last_node)

        last_node = ReLU(last_node)

        if len(affine_nodes) == len(affine_output_sizes):
            assert affine_dropout_include_rate is None
        elif affine_dropout_include_rate < 1.0:
            last_node = Dropout(last_node,
                                affine_dropout_include_rate,
                                theano_rng)

    last_node = Softmax(last_node)

    return conv_nodes, affine_nodes, last_node


def print_mcr(values, _):
    print("Misclassification rate: %s" % str(values))


def print_loss(values, _):  # 2nd argument: formats
    print("Average loss: %s" % str(values))


def main():
    '''
    Entry point of this script.
    '''

    args = parse_args()

    # Hyperparameter values taken from Pylearn2:
    # In pylearn2/scripts/tutorials/convolutional_network/:
    #   convolutional_network.ipynb

    filter_counts = [64, 64]
    filter_init_uniform_ranges = [.05] * len(filter_counts)
    dropout_include_rates = [.5] * len(filter_counts)
    filter_shapes = [(5, 5), (5, 5)]
    pool_shapes = [(4, 4), (4, 4)]
    pool_strides = [(2, 2), (2, 2)]
    affine_output_sizes = [10]
    affine_init_stddevs = [.05] * len(affine_output_sizes)
    dropout_include_rates += [.5] * (len(affine_output_sizes) - 1)

    assert_equal(affine_output_sizes[-1], 10)

    mnist_training, mnist_testing = load_mnist()

    image_uint8_node, label_node = mnist_training.make_input_nodes()
    image_node = RescaleImage(image_uint8_node)

    rng = numpy.random.RandomState(34523)
    theano_rng = RandomStreams(23845)

    (conv_nodes,
     affine_nodes,
     output_node) = build_conv_classifier(image_node,
                                          filter_shapes,
                                          filter_counts,
                                          filter_init_uniform_ranges,
                                          pool_shapes,
                                          pool_strides,
                                          affine_output_sizes,
                                          affine_init_stddevs,
                                          dropout_include_rates,
                                          rng,
                                          theano_rng)

    loss_node = CrossEntropy(output_node, label_node)
    scalar_loss = loss_node.output_symbol.mean()
    max_epochs = 500

    #
    # Makes parameter updaters
    #

    parameters = []
    parameter_updaters = []
    momentum_updaters = []

    def add_updaters(parameter,
                     scalar_loss,
                     parameter_updaters,
                     momentum_updaters):
        '''
        Adds a ParameterUpdater to parameter_updaters, and a
        LinearlyInterpolatesOverEpochs to momentum_updaters.
        '''
        gradient = theano.gradient.grad(scalar_loss, parameter)
        parameter_updaters.append(SgdParameterUpdater(parameter,
                                                      gradient,
                                                      args.learning_rate,
                                                      args.initial_momentum,
                                                      not args.no_nesterov))
        momentum_updaters.append(LinearlyInterpolatesOverEpochs(
            parameter_updaters[-1].momentum,
            args.final_momentum,
            args.epochs_to_momentum_saturation))

    for conv_node in conv_nodes:
        parameters.append(conv_node.filters)
        add_updaters(conv_node.filters,
                     scalar_loss,
                     parameter_updaters,
                     momentum_updaters)
        limit_param_norms(parameter_updaters[-1],
                          conv_node.filters,
                          args.max_filter_norm,
                          (1, 2, 3))

    for affine_node in affine_nodes:
        weights = affine_node.linear_node.params
        parameters.append(weights)
        add_updaters(weights,
                     scalar_loss,
                     parameter_updaters,
                     momentum_updaters)
        limit_param_norms(parameter_updater=parameter_updaters[-1],
                          params=weights,
                          max_norm=args.max_col_norm,
                          input_axes=[0])

        biases = affine_node.bias_node.params
        parameters.append(biases)
        add_updaters(biases,
                     scalar_loss,
                     parameter_updaters,
                     momentum_updaters)

    # updates = [updater.updates.values()[0] - updater.updates.keys()[0]
    #            for updater in parameter_updaters]
    # update_norm_monitors = [UpdateNormMonitor("layer %d %s" %
    #                                           (i // 2,
    #                                            "weights" if i % 2 == 0 else
    #                                            "bias"),
    #                                           update)
    #                         for i, update in enumerate(updates)]

    #
    # Makes batch and epoch callbacks
    #

    def make_misclassification_monitor():
        '''
        Returns an AverageMonitor of the misclassification rate.
        '''
        misclassification_node = Misclassification(output_node, label_node)
        mcr_logger = LogsToLists()
        training_stopper = StopsOnStagnation(max_epochs=10,
                                             min_proportional_decrease=0.0)
        return AverageMonitor(misclassification_node.output_symbol,
                              misclassification_node.output_format,
                              callbacks=[print_mcr,
                                         mcr_logger,
                                         training_stopper])

    mcr_monitor = make_misclassification_monitor()

    # batch callback (monitor)
    training_loss_logger = LogsToLists()
    training_loss_monitor = AverageMonitor(loss_node.output_symbol,
                                           loss_node.output_format,
                                           callbacks=[print_loss,
                                                      training_loss_logger])

    # print out 10-D feature vector
    # feature_vector_monitor = AverageMonitor(affine_nodes[-1].output_symbol,
    #                                         affine_nodes[-1].output_format,
    #                                         callbacks=[print_feature_vector])

    # epoch callbacks
    validation_loss_logger = LogsToLists()

    def make_output_filename(args, best=False):
        '''
        Constructs a filename that reflects the command-line params.
        '''
        assert_equal(os.path.splitext(args.output_prefix)[1], "")

        output_dir, output_prefix = os.path.split(args.output_prefix)
        if output_prefix != "":
            output_prefix = output_prefix + "_"

        output_prefix = os.path.join(output_dir, output_prefix)

        return ("%slr-%g_mom-%g_nesterov-%s_bs-%d%s.pkl" %
                (output_prefix,
                 args.learning_rate,
                 args.initial_momentum,
                 not args.no_nesterov,
                 args.batch_size,
                 "_best" if best else ""))

    model = SerializableModel([image_uint8_node], [output_node])
    saves_best = SavesAtMinimum(model, make_output_filename(args, best=True))

    validation_loss_monitor = AverageMonitor(
        loss_node.output_symbol,
        loss_node.output_format,
        callbacks=[validation_loss_logger, saves_best])

    validation_callback = ValidationCallback(
        inputs=[image_node.output_symbol, label_node.output_symbol],
        input_iterator=mnist_testing.iterator(iterator_type='sequential',
                                              batch_size=args.batch_size),
        monitors=[validation_loss_monitor, mcr_monitor])

    # pdb.set_trace()
    trainer = Sgd((image_node.output_symbol, label_node.output_symbol),
                  mnist_training.iterator(iterator_type='sequential',
                                          batch_size=args.batch_size),
                  parameters,
                  parameter_updaters,
                  monitors=[training_loss_monitor],
                  epoch_callbacks=[])

    stuff_to_pickle = OrderedDict(
        (('model', model),
         ('validation_loss_logger', validation_loss_logger)))

    # Pickling the trainer doesn't work when there are Dropout nodes.
    # stuff_to_pickle = OrderedDict(
    #     (('trainer', trainer),
    #      ('validation_loss_logger', validation_loss_logger),
    #      ('model', model)))

    trainer.epoch_callbacks = (momentum_updaters +
                               [PicklesOnEpoch(stuff_to_pickle,
                                               make_output_filename(args),
                                               overwrite=False),
                                validation_callback,
                                LimitsNumEpochs(max_epochs)])

    trainer.train()


if __name__ == '__main__':
    main()
