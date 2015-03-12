from simplelearn.nodes import AffineTransform, ReLU, CrossEntropy
from simplelearn.utils import safe_izip
from simplelearn.data.mnist import load_mnist
from simplelearn.formats import DenseFormat
from simplelearn.training import (SgdParameterUpdater,
                                  Sgd,
                                  LimitsNumEpochs,
                                  ValidationCallback,
                                  StopsOnStagnation)


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

    affine_nodes = []  # do I need this?

    for size in sizes:
        affine = AffineTransform(input_node, DenseFormat(axes=('b', 'f'),
                                                         shape=(-1, size),
                                                         dtype=None))
        affine_nodes.append(affine)
        input_node = ReLU(affine)

    def init_sparse(shared_variable, num_nonzeros, rng):
        '''
        Mimics the sparse initialization in
        pylearn2.models.mlp.Linear.set_input_space()
        '''

        params = shared_variable.get_values()
        params[...] = 0.0

        indices = rng.choice(params.size,
                             size=num_nonzeros,
                             replace=False)

        # normal dist with stddev=1.0
        params.flat[indices] = rng.randn(size=num_nonzeros)

        shared_variable.set_values(params)


    # Initialize the first N-1 affine layer weights (not biases)
    for sparse_init_count, affine_node in safe_izip(sparse_init_counts,
                                                    affine_nodes[:-1]):
        init_sparse(affine_node.weights, sparse_init_count, rng)

    output_node = Softmax(input_node, DenseFormat(axes=('b', 'f'),
                                                  shape=(-1, sizes[-1]),
                                                  dtype=None))

    return affine_nodes, output_node


def main():
    sizes = [500, 500, 10]
    sparse_init_counts = [15, 15]

    assert_equal(sizes[-1], 10)

    mnist_training, mnist_testing = load_mnist()

    image_node, label_node = dataset.get_input_nodes()

    affine_nodes, output_node = build_fc_classifier(input_node,
                                                    sizes,
                                                    sparse_init_counts)

    loss_node = CrossEntropy(output_node, label_node)
    loss_sum = loss_node.output_symbol.sum()

    # Values taken from Pylearn2:
    # In pylearn2/scripts/tutorials/multilayer_perceptron/:
    #   multilayer_perceptron.ipynb
    #   mlp_tutorial_part_3.yaml
    learning_rate = .01
    momentum = .05
    use_nesterov = False
    max_epochs = 10000


    #
    # Makes parameter updaters
    #

    parameters = []
    parameter_updaters = []
    for affine_node in affine_nodes:
        for params in (affine_node.linear_node.params,
                       affine_node.bias_node.params):
            parameters.append(params)
            gradients = theano.tensor.gradient(loss_sum, params)
            parameter_updaters.append(SgdParameterUpdater(params,
                                                          gradients,
                                                          learning_rate,
                                                          momentum,
                                                          use_neseterov))

    #
    # Makes batch and epoch callbacks
    #

    # batch callback (monitor)
    training_loss_logger = LogsToLists()
    training_loss_monitor = AverageMonitor(loss_node.output_symbol,
                                           loss_node.output_format,
                                           callbacks=[training_loss_logger])

    # epoch callbacks
    validation_loss_logger = LogsToLists()
    training_stopper = StopsOnStagnation(max_epochs=10, min_decrease=.01)

    validation_loss_monitor = AverageMonitor(
        loss_node.output_symbol,
        loss_node.output_format,
        callbacks=[validation_loss_logger, training_stopper])

    validation_callback = ValidationCallback(
        inputs=input_symbols,
        input_iterator=mnist_testing.iterator(iterator_type='sequential',
                                              batch_size=batch_size),
        monitors=[validation_loss_monitor])

    trainer = Sgd((image_node, label_node),
                  dataset.get_input_iterator(),
                  parameters,
                  parameter_updaters,
                  monitors=[training_loss_monitor],
                  epoch_callbacks=[])

    trainer.epoch_callbacks = [PicklesOnEpoch(trainer),
                               valiation_callback,
                               LimitsNumEpochs(max_epochs)]

if __name__ == '__main__':
    main()
