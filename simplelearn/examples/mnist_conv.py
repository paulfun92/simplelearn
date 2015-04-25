#! /usr/bin/env python

'''
Demonstrates training a convolutional net on MNIST.
'''

from __future__ import print_function

import os
import argparse
import timeit
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
from simplelearn.nodes import (Conv2DLayer,
                               AffineLayer,
                               Dropout,
                               CrossEntropy,
                               Misclassification,
                               Softmax,
                               RescaleImage,
                               FormatNode)
from simplelearn.utils import (safe_izip,
                               assert_floating,
                               assert_all_equal,
                               assert_all_greater,
                               assert_all_integers)
from simplelearn.io import SerializableModel
from simplelearn.data.dataset import Dataset
from simplelearn.data.mnist import load_mnist
from simplelearn.formats import DenseFormat
from simplelearn.training import (Monitor,
                                  SgdParameterUpdater,
                                  limit_param_norms,
                                  Sgd,
                                  LogsToLists,
                                  SavesAtMinimum,
                                  AverageMonitor,
                                  EpochCallback,
                                  LimitsNumEpochs,
                                  LinearlyInterpolatesOverEpochs,
                                  PicklesOnEpoch,
                                  ValidationCallback,
                                  StopsOnStagnation)

from simplelearn import debug_mnist

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

    def non_negative_0_to_1(arg):
        result = float(arg)
        assert_greater_equal(result, 0.0)
        assert_less_equal(result, 1.0)
        return result

    def max_norm_arg(arg):
        arg = float(arg)
        if arg < 0.0:
            return numpy.inf
        else:
            assert_greater(arg, 0.0)
            return arg

    parser.add_argument("--output-prefix",
                        type=legit_prefix,
                        required=True,
                        help=("Directory and optional prefix of filename to "
                              "save the log to."))

    # The default hyperparameter values below are taken from
    # Pylearn2's mlp tutorial, in pylearn2/scripts/tutorials/multilayer_perceptron/:
    #   multilayer_perceptron.ipynb
    #   mlp_tutorial_part_3.yaml
    #
    # Exceptions were made, since the original hyperparameters led to
    # divergence. These changes have been marked below.

    parser.add_argument("--learning-rate",
                        type=positive_float,
                        default=0.01,  # .01 used in pylearn2 demo
                        help=("Learning rate."))

    parser.add_argument("--initial-momentum",
                        type=non_negative_0_to_1,
                        default=0.5,
                        help=("Initial momentum."))

    parser.add_argument("--no-nesterov",
                        default=True,
                        action="store_true",
                        help=("Don't use Nesterov accelerated gradients "
                              "(default: False)."))

    parser.add_argument("--batch-size",
                        type=non_negative_int,
                        default=100,
                        help="batch size")

    parser.add_argument("--dropout",
                        action='store_true',
                        default=False,
                        help="Use dropout.")

    parser.add_argument("--final-momentum",
                        type=non_negative_0_to_1,
                        default=.99,
                        help="Value for momentum to linearly scale up to.")

    parser.add_argument("--epochs-to-momentum-saturation",
                        default=10,
                        type=positive_int,
                        help=("# of epochs until momentum linearly scales up "
                              "to --momentum_final_value."))

    default_max_norm = 1.9365

    parser.add_argument("--max-filter-norm",
                        type=max_norm_arg,
                        default=default_max_norm,
                        help=("Max. L2 norm of the convolutional filters. "
                              "Enter a negative number to not impose any "
                              "max norm."))

    parser.add_argument("--max-col-norm",
                        type=max_norm_arg,
                        default=default_max_norm,
                        help=("Max. L2 norm of weight matrix columns. Enter a "
                              "negative number to not imppose any max norm."))

    parser.add_argument("--weight-decay",
                        type=non_negative_float,
                        default=0.00005,
                        metavar="K",
                        help=("For each weight matrix or filters tensor W, "
                              "add K * sqr(W).sum() to each batch's cost, for "
                              "all weights and filters"))

    parser.add_argument("--validation-size",
                        type=non_negative_int,
                        default=10000,
                        metavar="V",
                        help=("If this is zero, use the test set as the "
                              "validation set. Otherwise, use the last V "
                              "elements of the training set as the validation "
                              "set."))

    return parser.parse_args()


class EpochTimer(EpochCallback):

    def on_start_training(self):
        self.start_time = timeit.default_timer()
        self.epoch_number = 0

    def on_epoch(self):
        end_time = timeit.default_timer()

        print("Epoch {} duration: {}".format(self.epoch_number,
                                             end_time - self.start_time))

        self.start_time = end_time
        self.epoch_number += 1


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
                 len(filter_shapes) + len(affine_output_sizes))

    assert_equal(affine_output_sizes[-1], 10)  # for MNIST

    assert_equal(input_node.output_format.axes, ('b', '0', '1'))


    #
    # Done sanity-checking args.
    #

    input_shape = input_node.output_format.shape

    # Converts from MNIST's ('b', '0', '1') to ('b', 'c', '0', '1')
    last_node = FormatNode(input_node,
                           DenseFormat(axes=('b', 'c', '0', '1'),
                                       shape=(input_shape[0],
                                              1,
                                              input_shape[1],
                                              input_shape[2]),
                                       dtype=None),
                           {'b': ('b', 'c')})
        # {'1': ('1', 'c')})

    conv_dropout_include_rates = \
        dropout_include_rates[:len(filter_shapes)]


    # Adds a dropout-conv-bias-relu-maxpool stack for each element in
    # filter_XXXX

    conv_layers = []

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
        if conv_dropout_include_rate != 1.0:
            assert_true(False,
                        "should never see this code if we're trying to "
                        "replicate pylearn2's MNIST demo")
            last_node = Dropout(last_node,
                                conv_dropout_include_rate,
                                theano_rng)

        last_node = Conv2DLayer(last_node,
                                filter_shape,
                                filter_count,
                                conv_pads='valid',
                                pool_window_shape=pool_shape,
                                pool_strides=pool_stride,
                                pool_pads='pylearn2')
        conv_layers.append(last_node)

        uniform_init(rng, last_node.conv2d_node.filters, filter_init_range)


    affine_dropout_include_rates = dropout_include_rates[len(filter_shapes):]

    affine_layers = []

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
    # Adds a dropout-affine-relu stack for each element in affine_XXXX,
    # except for the last one, where it omits the dropout.
    #

    for (affine_size,
         affine_init_stddev,
         affine_dropout_include_rate) in \
        safe_izip(affine_output_sizes,
                  affine_init_stddevs,
                  affine_dropout_include_rates):

        if affine_dropout_include_rate < 1.0:
            last_node = Dropout(last_node,
                                affine_dropout_include_rate,
                                theano_rng)

        # No need to supply an axis map for the first affine layer. By default,
        # it collapses all non-'b' axes into a feature vector, which is what we
        # want.

        last_node = AffineLayer(last_node,
                                DenseFormat(axes=('b', 'f'),
                                            shape=(-1, affine_size),
                                            dtype=None))

        normal_distribution_init(rng,
                                 last_node.affine_node.linear_node.params,
                                 affine_init_stddev)
        # stddev_init(rng, last_node.bias_node.params, affine_init_stddev)
        affine_layers.append(last_node)

    last_node = Softmax(last_node)

    return conv_layers, affine_layers, last_node


def print_mcr(values, _):
    print("Misclassification rate: %s" % str(values))


def print_loss(values, _):  # 2nd argument: formats
    print("Average loss: %s" % str(values))


class GradMonitor(Monitor):
    def __init__(self, grads, shapes):

        assert_equal(len(grads), len(shapes))
        assert_equal(len(grads), 3*2)

        formats = []

        for layer in range(2):
            formats.append(DenseFormat(axes=('oc', 'ic', '0', '1'),
                                       shape=shapes[layer * 2],
                                       dtype='floatX'))

            formats.append(DenseFormat(axes=('bias_singleton', 'bd'),
                                       shape=shapes[layer * 2 + 1],
                                       dtype='floatX'))


        formats.append(DenseFormat(axes=('if', 'of'),
                                   shape=shapes[-2],
                                   dtype='floatX'))

        formats.append(DenseFormat(axes=('bias_singleton', 'bd'),
                                   shape=shapes[-1],
                                   dtype='floatX'))

        super(GradMonitor, self).__init__(grads, formats, [])

    def _on_epoch(self):
        pass

    def _on_batch(self):
        pass

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
    filter_shapes = [(5, 5), (5, 5)]
    pool_shapes = [(4, 4), (4, 4)]
    pool_strides = [(2, 2), (2, 2)]
    affine_output_sizes = [10]
    affine_init_stddevs = [.05] * len(affine_output_sizes)
    dropout_include_rates = ([.5 if args.dropout else 1.0] *
                             (len(filter_counts) + len(affine_output_sizes)))

    assert_equal(affine_output_sizes[-1], 10)

    mnist_training, mnist_testing = load_mnist()

    if args.validation_size != 0:
        tensors = mnist_training.tensors
        training_tensors = [t[:-args.validation_size, ...] for t in tensors]
        testing_tensors = [t[args.validation_size:, ...] for t in tensors]
        mnist_training = Dataset(tensors=training_tensors,
                                 names=mnist_training.names,
                                 formats=mnist_training.formats)
        mnist_testing = Dataset(tensors=testing_tensors,
                                names=mnist_training.names,
                                formats=mnist_training.formats)

    mnist_testing_iterator = mnist_testing.iterator(iterator_type='sequential',
                                                    loop_style='divisible',
                                                    batch_size=args.batch_size)

    image_uint8_node, label_node = mnist_testing_iterator.make_input_nodes()
    image_node = RescaleImage(image_uint8_node)

    rng = numpy.random.RandomState(1234)
    theano_rng = RandomStreams(23845)

    (conv_layers,
     affine_layers,
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

    if args.weight_decay != 0.0:
        for conv_layer in conv_layers:
            filters = conv_layer.conv2d_node.filters
            filter_loss = args.weight_decay * theano.tensor.sqr(filters).sum()
            scalar_loss = scalar_loss + filter_loss

        for affine_layer in affine_layers:
            weights = affine_layer.affine_node.linear_node.params
            weight_loss = args.weight_decay * theano.tensor.sqr(weights).sum()
            scalar_loss = scalar_loss + weight_loss

    max_epochs = 500

    #
    # Makes parameter updaters
    #

    parameters = []
    DEBUG_grads = []
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
        DEBUG_grads.append(gradient)
        parameter_updaters.append(SgdParameterUpdater(parameter,
                                                      gradient,
                                                      args.learning_rate,
                                                      args.initial_momentum,
                                                      not args.no_nesterov))
        momentum_updaters.append(LinearlyInterpolatesOverEpochs(
            parameter_updaters[-1].momentum,
            args.final_momentum,
            args.epochs_to_momentum_saturation))

    for conv_layer in conv_layers:
        filters = conv_layer.conv2d_node.filters
        parameters.append(filters)
        add_updaters(filters,
                     scalar_loss,
                     parameter_updaters,
                     momentum_updaters)

        if args.max_filter_norm != numpy.inf:
            limit_param_norms(parameter_updaters[-1],
                              filters,
                              args.max_filter_norm,
                              (1, 2, 3))

        bias = conv_layer.bias_node.params
        parameters.append(bias)
        add_updaters(bias,
                     scalar_loss,
                     parameter_updaters,
                     momentum_updaters)

    for affine_layer in affine_layers:
        weights = affine_layer.affine_node.linear_node.params
        parameters.append(weights)
        add_updaters(weights,
                     scalar_loss,
                     parameter_updaters,
                     momentum_updaters)
        if args.max_col_norm != numpy.inf:
            limit_param_norms(parameter_updater=parameter_updaters[-1],
                              params=weights,
                              max_norm=args.max_col_norm,
                              input_axes=[0])

        biases = affine_layer.affine_node.bias_node.params
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
    gradients_monitor = GradMonitor(DEBUG_grads,
                                    [p.get_value().shape for p in parameters])

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

        if os.path.isdir(args.output_prefix):
            output_dir, output_prefix = args.output_prefix, ""
        else:
            output_dir, output_prefix = os.path.split(args.output_prefix)
            assert_true(os.path.isdir(output_dir))

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
        input_iterator=mnist_testing_iterator,
        monitors=[validation_loss_monitor, mcr_monitor])

    # trainer = Sgd((image_node.output_symbol, label_node.output_symbol),
    trainer = Sgd([image_uint8_node, label_node],
                  mnist_training.iterator(iterator_type='sequential',
                                          loop_style='divisible',
                                          batch_size=args.batch_size),
                  parameters,
                  parameter_updaters,
                  monitors=[gradients_monitor, training_loss_monitor],
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
                               [EpochTimer(),
                                PicklesOnEpoch(stuff_to_pickle,
                                               make_output_filename(args),
                                               overwrite=False),
                                validation_callback,
                                LimitsNumEpochs(max_epochs)])

    # start debugging code
    debug_mnist.mnist_weights = (
        [c.conv2d_node.filters for c in conv_layers] +
        [a.affine_node.linear_node.params for a in affine_layers])

    debug_mnist.mnist_biases = (
        [c.bias_node.params for c in conv_layers] +
        [a.affine_node.bias_node.params for a in affine_layers])

    def debug_formats():
        conv_output_syms = [c.conv2d_node.output_symbol for c in conv_layers]
        conv_formats = [c.conv2d_node.output_format for c in conv_layers]

        pool_output_syms = [c.pool2d_node.output_symbol for c in conv_layers]
        pool_formats = [c.pool2d_node.output_format for c in conv_layers]

        affine_output_syms = [a.affine_node.output_symbol for a in affine_layers]
        affine_formats = [a.affine_node.output_format for a in affine_layers]

        func = theano.function([image_uint8_node.output_symbol],
                               conv_output_syms +
                               pool_output_syms +
                               affine_output_syms)

        iterator = mnist_training.iterator(iterator_type='sequential',
                                           batch_size=10)
        image_batch = iterator.next()[0]
        outputs = func(image_batch)
        last_index = 0

        conv_outputs = outputs[last_index:len(conv_output_syms)]
        last_index += len(conv_outputs)

        pool_outputs = outputs[last_index:last_index + len(pool_output_syms)]
        last_index += len(pool_outputs)

        affine_outputs = outputs[last_index:last_index + len(affine_output_syms)]
        last_index += len(affine_outputs)

        def check_formats(outputs, formats):
            for output, fmt in safe_izip(outputs, formats):
                fmt.check(output)


        check_formats(conv_outputs, conv_formats)
        check_formats(pool_outputs, pool_formats)
        check_formats(affine_outputs, affine_formats)
        pdb.set_trace()

    # debug_formats()

    # end debugging code

    trainer.train()


if __name__ == '__main__':
    main()
