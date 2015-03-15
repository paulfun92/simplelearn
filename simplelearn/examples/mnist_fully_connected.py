#! /usr/bin/env python

import argparse
import numpy
import theano
from nose.tools import assert_greater, assert_equal
from simplelearn.nodes import (AffineTransform,
                               FormatNode,
                               ReLU,
                               CrossEntropy,
                               Misclassification,
                               Softmax,
                               RescaleImage)
from simplelearn.utils import safe_izip
from simplelearn.data.mnist import load_mnist
from simplelearn.formats import DenseFormat
from simplelearn.training import (SgdParameterUpdater,
                                  Sgd,
                                  LogsToLists,
                                  Monitor,
                                  AverageMonitor,
                                  LimitsNumEpochs,
                                  PicklesOnEpoch,
                                  ValidationCallback,
                                  StopsOnStagnation)
import pdb

def parse_args():
    parser = argparse.ArgumentParser(
        description=("Trains multilayer perceptron to classify MNIST digits."))

    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help=("Filename to save the trainer & log to."))

    return parser.parse_args()


def build_fc_classifier(input_node,
                        sizes,
                        sparse_init_counts,
                        rng):
    '''
    Builds a stack of fully-connected layers followed by a Softmax.

    Each hidden layer will be preceded by a ReLU.

    Initialization:

    Weights are initialized in the same way as in Pylearn2's MLP tutorial:
    pylearn2/scripts/tutorials/multilayer_perceptron/mlp_tutorial_part_3.yaml

    This means the following:

    Of the N affine layers, the weights of the first N-1 are to all 0.0, except
    for k randomly-chosen elements, which are set to some random number drawn
    from the normal distribution with stddev=1.0.

    The biases are all initialized to 0.0.
    The last layer's weights and biases are both set to 0.0.

    Parameters
    ----------
    input_node: Node
      The node to build the stack on.

    sizes: Sequence
      A sequence of ints, indicating the output sizes of each layer.
      The last int is the number of classes.

    sparse_init_counts:
      A sequence of ints, with length one shorter than <sizes>.
      Used to initialize the weights of the first N-1 layers.
      If the n'th element is x, this means that the n'th layer
      will have x nonzeros, with the rest initialized to zeros.
    '''
    assert_greater(len(sizes), 0)
    assert_equal(len(sparse_init_counts), len(sizes) - 1)

    assert_equal(input_node.output_format.dtype,
                 numpy.dtype(theano.config.floatX))

    # def get_flat_float_vector(image_node):
    #     # Convert uint8 image matrices to floatX vectors.
    #     image_size = numpy.prod(image_node.output_format.shape[1:])
    #     image_node = RescaleImage(image_node)
    #     return FormatNode(image_node,
    #                       DenseFormat(axes=('b', 'f'),
    #                                   shape=(-1, image_size),
    #                                   dtype=None),
    #                       axis_map={('0', '1'): 'f'})

    # image_node = get_flat_float_vector(image_node)

    affine_nodes = []  # do I need this?

    hidden_node = input_node
    for layer_index, size in enumerate(sizes):
        hidden_node = AffineTransform(hidden_node, DenseFormat(axes=('b', 'f'),
                                                             shape=(-1, size),
                                                             dtype=None))
        affine_nodes.append(hidden_node)
        if layer_index != (len(sizes) - 1):
            hidden_node = ReLU(hidden_node)

    output_node = Softmax(hidden_node, DenseFormat(axes=('b', 'f'),
                                                  shape=(-1, sizes[-1]),
                                                  dtype=None))

    # DEBUG
    # print_softmax_op = theano.printing.Print("softmax: ")
    # output_node.output_symbol = print_softmax_op(output_node.output_symbol)

    def init_sparse(shared_variable, num_nonzeros, rng):
        '''
        Mimics the sparse initialization in
        pylearn2.models.mlp.Linear.set_input_space()
        '''

        params = shared_variable.get_value()
        params[...] = 0.0

        indices = rng.choice(params.size,
                             size=num_nonzeros,
                             replace=False)

        # normal dist with stddev=1.0
        params.flat[indices] = rng.randn(num_nonzeros)

        shared_variable.set_value(params)


    # Initialize the first N-1 affine layer weights (not biases)
    for sparse_init_count, affine_node in safe_izip(sparse_init_counts,
                                                    affine_nodes[:-1]):
        for params in (affine_node.linear_node.params,
                       affine_node.bias_node.params):
            init_sparse(params, sparse_init_count, rng)

    # # Initialize the first N-1 affine layer weights (not biases)
    # for affine_node in affine_nodes:
    #     # normal-distributes weights
    #     weights = affine_node.linear_node.params.get_value()
    #     weights[...] = rng.normal(loc=0.0, scale=.005, size=weights.shape)
    #     affine_node.linear_node.params.set_value(weights)

    #     # Zeroes biases
    #     biases = affine_node.bias_node.params.get_value()
    #     biases[...] = 0.0
    #     affine_node.bias_node.params.set_value(biases)

    return affine_nodes, output_node


def print_loss(values, _):  # 2nd argument: formats
    print("Average loss: %s" % str(values))

def print_feature_vector(values, _):
    print("Average feature vector: %s" % str(values))

def print_mcr(values, _):
    print("Misclassification rate: %s" % str(values))

class UpdateNormMonitor(Monitor):
    def __init__(self, name, update):
        update = update.reshape(shape=(1, -1))
        update_norm = theano.tensor.sqrt((update**2).sum(axis=1))

        # just something to satisfy the checks of Monitor.__init__.
        # Because we overrride on_batch(), this is never used.
        dummy_fmt = DenseFormat(axes=('b',),
                                shape=(-1,),
                                dtype=update_norm.dtype)
        self.name = name
        super(UpdateNormMonitor, self).__init__([update_norm],
                                                [dummy_fmt],
                                                [])

    def _on_batch(self, input_batches, monitored_value_batches):
        print("%s update norm: %s" % (self.name, str(monitored_value_batches)))

    def _on_epoch(self):
        return tuple()

def main():
    args = parse_args()

    # Hyperparameter values taken from Pylearn2:
    # In pylearn2/scripts/tutorials/multilayer_perceptron/:
    #   multilayer_perceptron.ipynb
    #   mlp_tutorial_part_3.yaml

    sizes = [500, 500, 10]
    sparse_init_counts = [15, 15]

    assert_equal(sizes[-1], 10)

    mnist_training, mnist_testing = load_mnist()

    image_node, label_node = mnist_training.make_input_nodes()
    image_node = RescaleImage(image_node)

    rng = numpy.random.RandomState(34523)

    affine_nodes, output_node = build_fc_classifier(image_node,
                                                    sizes,
                                                    sparse_init_counts,
                                                    rng)

    loss_node = CrossEntropy(output_node, label_node)

    # DEBUG
    # print_loss_op = theano.printing.Print("cross_entropy: ")
    # loss_node.output_symbol = print_loss_op(loss_node.output_symbol)

    loss_sum = loss_node.output_symbol.mean()
    # loss_sum = loss_node.output_symbol.sum()

    learning_rate = .01
    momentum = 0 #.5
    use_nesterov = False
    max_epochs = 10000
    batch_size = 100

    #
    # Makes parameter updaters
    #

    parameters = []
    parameter_updaters = []
    for affine_node in affine_nodes:
        for params in (affine_node.linear_node.params,
                       affine_node.bias_node.params):
            parameters.append(params)
            gradients = theano.gradient.grad(loss_sum, params)
            parameter_updaters.append(SgdParameterUpdater(params,
                                                          gradients,
                                                          learning_rate,
                                                          momentum,
                                                          use_nesterov))

    updates = [updater.updates.values()[0] - updater.updates.keys()[0]
               for updater in parameter_updaters]
    update_norm_monitors = [UpdateNormMonitor("layer %d %s" %
                                              (i//2,
                                               "weights" if i % 2 == 0 else
                                               "bias"),
                                              update)
                            for i, update in enumerate(updates)]

    # pdb.set_trace()
    #
    # Makes batch and epoch callbacks
    #

    misclassification_node = Misclassification(output_node, label_node)
    mcr_logger = LogsToLists()
    mcr_monitor = AverageMonitor(misclassification_node.output_symbol,
                                 misclassification_node.output_format,
                                 callbacks=[print_mcr, mcr_logger])

    # batch callback (monitor)
    training_loss_logger = LogsToLists()
    training_loss_monitor = AverageMonitor(loss_node.output_symbol,
                                           loss_node.output_format,
                                           callbacks=[print_loss, training_loss_logger])

    # print out 10-D feature vector
    feature_vector_monitor = AverageMonitor(affine_nodes[-1].output_symbol,
                                            affine_nodes[-1].output_format,
                                            callbacks=[print_feature_vector])

    # epoch callbacks
    validation_loss_logger = LogsToLists()
    training_stopper = StopsOnStagnation(max_epochs=10,
                                         min_proportional_decrease=0.0)

    # pdb.set_trace()
    validation_loss_monitor = AverageMonitor(
        loss_node.output_symbol,
        loss_node.output_format,
        callbacks=[validation_loss_logger, training_stopper])

    validation_callback = ValidationCallback(
        inputs=[image_node.output_symbol, label_node.output_symbol],
        input_iterator=mnist_testing.iterator(iterator_type='sequential',
                                              batch_size=batch_size),
        monitors=[validation_loss_monitor, mcr_monitor])
        # monitors=[validation_loss_monitor, feature_vector_monitor])

    trainer = Sgd((image_node.output_symbol, label_node.output_symbol),
                  mnist_training.iterator(iterator_type='sequential',
                                          batch_size=batch_size),
                  parameters,
                  parameter_updaters,
                  monitors=[training_loss_monitor],
                  # monitors=[training_loss_monitor] + update_norm_monitors,
                  # monitors=[training_loss_monitor, feature_vector_monitor],
                  epoch_callbacks=[])

    stuff_to_pickle = {'trainer': trainer,
                       'validation_loss_logger': validation_loss_logger}

    trainer.epoch_callbacks = [PicklesOnEpoch(stuff_to_pickle,
                                              args.output_file,
                                              overwrite=False),
                               validation_callback,
                               LimitsNumEpochs(max_epochs)]

    trainer.train()


# def compare_weights(file0, file1):
#     def get_weights(file_path):
#         pickled = cPickle.load(file_path)
#         trainer = pickled['trainer']


if __name__ == '__main__':
    main()
