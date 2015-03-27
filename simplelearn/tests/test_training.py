import numpy
from numpy.testing import assert_allclose
import theano
import theano.tensor as T
from nose.tools import assert_equal, assert_raises_regexp
from simplelearn.utils import safe_izip
from simplelearn.training import (StopsOnStagnation,
                                  StopTraining,
                                  LimitsNumEpochs,
                                  LinearlyInterpolatesOverEpochs,
                                  LinearlyScalesOverEpochs)
from simplelearn.formats import DenseFormat
from simplelearn.nodes import Node
from simplelearn.data.dataset import Dataset

import pdb


class L2Norm(Node):

    def __init__(self, input_node):
        feature_axis = input_node.output_format.axes.index('f')
        input_symbol = input_node.output_symbol

        output_symbol = \
            T.sqrt((input_symbol * input_symbol).sum(axis=feature_axis))

        output_format = DenseFormat(axes=('b'), shape=(-1, ), dtype=None)
        super(L2Norm, self).__init__(output_symbol,
                                     output_format,
                                     input_node)


# def test_computes_average_over_epoch():
#     rng = numpy.random.RandomState(3851)
#     tensor = rng.uniform(-1.0, 1.0, size=(3, 10))
#     fmt = DenseFormat(axes=('b', 'f'), shape=(-1, 10), dtype=tensor.dtype)
#     dataset = Dataset(names=('x', ), formats=(fmt, ), tensors=(tensor, ))

#     l2norm_node = L2Norm(*dataset.make_input_nodes())
#     averages = []

#     l2norms = numpy.sqrt((tensor * tensor).sum(axis=1))
#     expected_average = l2norms.sum() / l2norms.size

#     averager = ComputesAverageOverEpoch(l2norm_node,
#                                         dataset.iterator('sequential',
#                                                          batch_size=1,
#                                                          loop_style="wrap"),
#                                         (lambda x: averages.append(x), ))

#     for ii in range(2):
#         averager()
#         assert_equal(len(averages), ii + 1)
#         assert_allclose(averages[ii], expected_average)


# def test_stops_on_stagnation():

#     def get_values(stepsize, kink_index):
#         descending_values = (numpy.arange(kink_index) * (-stepsize)) + 100.
#         flat_values = numpy.zeros(kink_index)
#         flat_values[:] = descending_values[-1]
#         return numpy.concatenate((descending_values, flat_values))

#     threshold = .1
#     kink_index = 20
#     values = get_values(threshold + .0001, kink_index)

#     num_epochs = 5
#     assert num_epochs < kink_index

#     index = 0
#     stops_on_stagnation = StopsOnStagnation("hockey stick",
#                                             num_epochs,
#                                             threshold)

#     try:
#         for value in values:
#             stops_on_stagnation(value)
#             index += 1
#     except StopTraining, st:
#         assert_equal(index, kink_index + num_epochs)
#         assert_equal(st.status, 'ok')
#         assert "didn't decrease for" in st.message
def test_limit_param_norms():
    '''
    A unit test for limit_param_norms().

    Optimizes a simple function f = ||W - x||, with a limit on W's norms.

    Initial value of W is 0. ||W - x|| is bigger than W's max norm. Therefore,
    we expect the final value of W to be k, scaled to max_norm.
    '''

    floatX = theano.config.floatX

    def make_single_example_dataset(norm, shape):
        axes = ('b', ) + tuple(str(i) for i in range(len(shape)))
        fmt = DenseFormat(axes=axes,
                          shape=(-1, ) + shape,
                          dtype=floatX)
        data = fmt.make_batch(batch_size=1, is_symbolic=False)
        sum_axes = range(1, len(shape) + 1)

        # Scale all data so that L2 norms = norm
        norms = (data ** 2.0).sum(axis=sum_axes, keepdims=True).sqrt()
        scales = norm / (norms + .00001)
        data *= scales

        return Dataset(tensors=[data],
                       formats=[fmt],
                       names=['data'])

    def make_costs_node(input_node, weights):
        input_node = input_nodes[0]

        diff = (input_node.output_symbol -
                weights.reshape((1, ) + weights.shape))
        sum_axes = range(1, len(diff.ndim) + 1)

        l2 = (diff * diff).sum(axis=sum_axes).sqrt()

        return Node([input_node],
                    l2,
                    DenseFormat(axes=['b'], shape=[-1], dtype=weights.dtype))

    cast = numpy.cast[floatX]

    dataset_norm = 3.0
    max_norm = 2.0
    learning_rate = .01

    for shape in ((2, ), (2, 3, 4)):
        dataset = make_single_example_dataset(dataset_norm, shape)

        weights = theano.tensor.shared(numpy.zeros((1, ) + shape,
                                                   dtype=floatX))

        input_nodes = dataset.make_input_nodes()
        assert_equal(len(input_nodes), 1)

        costs_node = make_costs_node(input_nodes[0], weights)
        gradients = theano.gradients.grad(costs_node.output_symbol.sum(),
                                          weights)
        param_updater = SgdParameterUpdater(parameter=weights,
                                            gradient=gradients,
                                            learning_rate=learning_rate,
                                            momentum=0.0,
                                            use_nesterov=False)

        input_axes = range(1, len(shape) + 1)
        limit_param_norms(param_updater, max_norm, input_axes)

        stops_on_stagnation = StopsOnStagnation(max_epochs=10)
        avg_cost_monitor = AverageMonitor(costs_node.output_symbol,
                                          costs_node.output_format,
                                          callbacks=[stops_on_stagnation])

        sgd = Sgd(inputs=input_nodes,
                  input_iterator=dataset.iterator(iterator_type='sequential',
                                                  batch_size=1),
                  parameters=[weights],
                  parameter_updaters=[param_updater],
                  monitors=[average_cost_monitor],
                  epoch_callbacks=[])

        sgd.train()

        weight_norms = (weights.get_value() ** 2.0).sum(input_axes,
                                                        keepdims=True).sqrt()
        assert_allclose(weight_norms, max_norm)

        # an optional sanity-check to confirm that the weights are on a
        # straight line between their initial value (0.0) and the data.
        normed_weights = weights.get_value() / (weight_norms + .00001)
        normed_data = dataset.tensors[0] / dataset_norm
        assert_allclose(normed_weights,
                        normed_data,
                        rtol=learning_rate)


def test_limits_num_epochs():
    max_num_epochs = 5
    limits_num_epochs = LimitsNumEpochs(max_num_epochs)

    limits_num_epochs.on_start_training()

    for _ in range(max_num_epochs - 1):
        limits_num_epochs.on_epoch()

    with assert_raises_regexp(StopTraining, "Reached max"):
        limits_num_epochs.on_epoch()


def test_linearly_scales_over_epochs():
    initial_value = 5.6
    final_scale = .012
    epochs_to_saturation = 23

    shared_variable = theano.shared(initial_value)
    assert_allclose(shared_variable.get_value(), initial_value)

    callback = LinearlyScalesOverEpochs(shared_variable,
                                        final_scale,
                                        epochs_to_saturation)
    expected_values = numpy.linspace(1.0,
                                     final_scale,
                                     epochs_to_saturation + 1) * initial_value
    flat_values = numpy.zeros(7)
    flat_values[:] = expected_values[-1]

    expected_values = numpy.concatenate((expected_values, flat_values))

    callback.on_start_training()
    for expected_value in expected_values:
        assert_allclose(shared_variable.get_value(),
                        expected_value)
        callback.on_epoch()


def test_linearly_interpolates_over_epochs():
    rng = numpy.random.RandomState(23452)
    dtype = numpy.dtype('float32')
    cast = numpy.cast[dtype]
    shape = (2, 3)

    initial_value = cast(rng.uniform(size=shape))
    final_value = cast(rng.uniform(size=shape))
    epochs_to_saturation = 23

    shared_variable = theano.shared(numpy.zeros(shape, dtype=dtype))

    callback = LinearlyInterpolatesOverEpochs(shared_variable,
                                              final_value,
                                              epochs_to_saturation)
    shared_variable.set_value(initial_value)

    callback.on_start_training()
    assert_allclose(shared_variable.get_value(), initial_value)

    initial_weights = numpy.concatenate(
        (numpy.linspace(1.0, 0.0, epochs_to_saturation + 1),
         numpy.zeros(7)))

    final_weights = 1.0 - initial_weights

    expected_values = [wi * initial_value + wf * final_value
                       for wi, wf
                       in safe_izip(initial_weights, final_weights)]

    for expected_value in expected_values:
        assert_allclose(shared_variable.get_value(),
                        expected_value,
                        atol=1e-06)
        callback.on_epoch()
