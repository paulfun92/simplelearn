'''
Unit tests for ../training.py
'''

from __future__ import print_function

__author__ = "Matthew Koichi Grimes"
__email__ = "mkg@alum.mit.edu"
__copyright__ = "Copyright 2015"
__license__ = "Apache 2.0"

import numpy
from numpy.testing import assert_allclose, assert_almost_equal
import theano
import theano.tensor as T
from nose.tools import (assert_true, 
                        assert_equal, 
                        assert_raises_regexp,
                        assert_is_instance)
from simplelearn.utils import safe_izip
from simplelearn.training import (StopsOnStagnation,
                                  StopTraining,
                                  LimitsNumEpochs,
                                  LinearlyInterpolatesOverEpochs,
                                  LinearlyScalesOverEpochs,
                                  SgdParameterUpdater,
                                  limit_param_norms,
                                  Sgd,
                                  MeanOverEpoch,
                                  EpochCallback)
from simplelearn.formats import DenseFormat
from simplelearn.nodes import Node
from simplelearn.data.dataset import Dataset

import pdb


class L2Norm(Node):
    '''
    Computes the L2 norm of a single vector.

    Unlike nodes.L2Loss, which is a function of two vectors.
    '''

    def __init__(self, input_node):
        assert_equal(frozenset(input_node.output_format.axes),
                     frozenset(['b', 'f']))

        feature_axis = input_node.output_format.axes.index('f')
        input_symbol = input_node.output_symbol

        output_symbol = \
            T.sqrt((input_symbol * input_symbol).sum(axis=feature_axis))

        output_format = DenseFormat(axes=['b'], shape=[-1], dtype=None)
        super(L2Norm, self).__init__([input_node],
                                     output_symbol,
                                     output_format)


def test_mean_over_epoch():

    rng = numpy.random.RandomState(3851)

    vectors = rng.uniform(-1.0, 1.0, size=(12, 10))
    fmt = DenseFormat(axes=('b', 'f'), shape=(-1, 10), dtype=vectors.dtype)
    dataset = Dataset(names=['vectors'], formats=[fmt], tensors=[vectors])
    iterator = dataset.iterator('sequential',
                                batch_size=2,
                                loop_style="divisible")

    input_node = iterator.make_input_nodes()[0]
    l2_norm_node = L2Norm(input_node)

    num_averages_compared = [0]

    def compare_with_expected_average(values, _):  # ignore format argument
        assert_equal(len(values), 1)
        average = values[0]

        assert_is_instance(fmt, DenseFormat)
        l2_norms = numpy.sqrt((vectors ** 2.0).sum(fmt.axes.index('f')))
        expected_average = l2_norms.sum() / l2_norms.size

        assert_allclose(average, expected_average)
        num_averages_compared[0] += 1

    average_monitor = MeanOverEpoch(l2_norm_node,
                                    [compare_with_expected_average])

    class DatasetRandomizer(EpochCallback):
        '''
        Fills the dataset with a fresh set of random values after each epoch.
        '''

        def on_start_training(self):
            pass

        def on_epoch(self):
            vectors[...] = rng.uniform(-1.0, 1.0, size=vectors.shape)

    trainer = Sgd([input_node],
                  iterator,
                  callbacks=[average_monitor,
                             LimitsNumEpochs(3),
                             DatasetRandomizer()])

    trainer.train()

    assert_equal(num_averages_compared[0], 3)


def test_stops_on_stagnation():
    '''
    Tests StopsOnStagnation.

    Presents the EpochCallback with a sequence of loss values in the shape
    of a hockey stick: constant downward slope, followed by a kink into
    a flat, stagnant section.
    '''

    def get_values(min_proportional_decrease, kink_index):
        descending_values = numpy.zeros(kink_index, dtype=float)
        descending_values[0] = 1.0
        scale = 1.0 - (min_proportional_decrease + .001)
        for i in range(1, len(descending_values)):
            descending_values[i] = descending_values[i - 1] * scale

        # descending_values = (numpy.arange(kink_index) * (-stepsize)) + 100.
        flat_values = numpy.zeros(kink_index)
        flat_values[:] = descending_values[-1]
        result = numpy.concatenate((descending_values, flat_values))
        return result[:, numpy.newaxis]

    min_proportional_decrease = .1
    kink_index = 20
    values = get_values(min_proportional_decrease, kink_index)

    num_epochs = 5
    assert num_epochs < kink_index

    index = 0
    stops_on_stagnation = StopsOnStagnation(num_epochs,
                                            min_proportional_decrease)
    fmt = DenseFormat(axes=['b'], shape=[-1], dtype=None)

    training_stopped = False
    try:
        for value in values:
            index += 1
            stops_on_stagnation(value, fmt)
    except StopTraining, st:
        assert_equal(index, kink_index + num_epochs)
        assert_equal(st.status, 'ok')
        assert "Value did not lower" in st.message
        training_stopped = True

    assert_true(training_stopped)


def test_limit_param_norms():
    '''
    A unit test for limit_param_norms().

    Optimizes a simple function f = ||W - x||, with a limit on W's norms.

    Initial value of W is 0. ||W - x|| is bigger than W's max norm. Therefore,
    we expect the final value of W to be k, scaled to max_norm.
    '''

    floatX = theano.config.floatX

    def make_single_example_dataset(norm, shape, rng):
        '''
        Returns a Dataset with a single datum with a given L2 norm.

        Parameters
        ----------
        norm: float
          The L2 norm that the flattened datum should have.

        shape: Sequence
          The shape of the datum.

        Returns
        -------
        rval: Dataset
        '''
        axes = ('b', ) + tuple(str(i) for i in range(len(shape)))
        fmt = DenseFormat(axes=axes,
                          shape=(-1, ) + shape,
                          dtype=floatX)
        data = fmt.make_batch(batch_size=1, is_symbolic=False)
        data[...] = rng.uniform(low=-1.0, high=1.0, size=data.shape)

        sum_axes = tuple(range(1, len(shape) + 1))

        # Scale all data so that L2 norms = norm
        norms = numpy.sqrt((data ** 2.0).sum(axis=sum_axes, keepdims=True))
        scales = norm / (norms + .00001)
        data *= scales

        return Dataset(tensors=[data],
                       formats=[fmt],
                       names=['data'])

    def make_costs_node(input_node, weights):
        '''
        Returns a Node that computes the squared distance between input_node
        and weights.
        '''
        assert_is_instance(input_node, Node)
        flat_shape = (input_node.output_symbol.shape[0], -1)

        input_vectors = input_node.output_symbol.reshape(flat_shape)
        weight_vectors = weights.reshape((weights.shape[0], -1))

        diff = input_vectors - weight_vectors
        costs = T.sqr(diff).sum(axis=1)

        return Node([input_node],
                    costs,
                    DenseFormat(axes=['b'], shape=[-1], dtype=weights.dtype))

    dataset_norm = .3
    max_norm = .2
    learning_rate = .001
    rng = numpy.random.RandomState(325)

    def print_cost(monitored_value, fmt):
        print("avg cost: %s" % monitored_value)

    def print_weight_norm(monitored_values, fmt):
        assert_equal(len(monitored_values), 1)
        weights = monitored_values[0]
        norm = numpy.sqrt((weights.get_value() ** 2.0).sum())
        print("weights' norm: %s" % norm)

    for shape in ((2, ), (2, 3, 4)):
        dataset = make_single_example_dataset(dataset_norm, shape, rng)

        weights = theano.shared(numpy.zeros((1, ) + shape, dtype=floatX))

        training_iterator = dataset.iterator(iterator_type='sequential',
                                             batch_size=1)
        input_nodes = training_iterator.make_input_nodes()
        assert_equal(len(input_nodes), 1)

        costs_node = make_costs_node(input_nodes[0], weights)
        gradients = theano.gradient.grad(costs_node.output_symbol.mean(),
                                         weights)
        param_updater = SgdParameterUpdater(parameter=weights,
                                            gradient=gradients,
                                            learning_rate=learning_rate,
                                            momentum=0.0,
                                            use_nesterov=False)

        input_axes = tuple(range(1, len(shape) + 1))
        limit_param_norms(param_updater, weights, max_norm, input_axes)

        stops_on_stagnation = StopsOnStagnation(max_epochs=10)
        average_cost_monitor = MeanOverEpoch(costs_node,
                                             callbacks=[stops_on_stagnation])

        sgd = Sgd(inputs=input_nodes,
                  input_iterator=training_iterator,
                  callbacks=[param_updater, average_cost_monitor])
        sgd.train()

        weight_norm = numpy.sqrt((weights.get_value() ** 2.0).sum())
        assert_almost_equal(weight_norm, max_norm, decimal=6)

        # an optional sanity-check to confirm that the weights are on a
        # straight line between their initial value (0.0) and the data.
        normed_weights = weights.get_value() / weight_norm
        normed_data = dataset.tensors[0] / dataset_norm
        assert_allclose(normed_weights,
                        normed_data,
                        rtol=learning_rate * 10)


def test_limits_num_epochs():
    '''
    Unit test for simplelearn.training.LimitsNumEpochs.
    '''
    max_num_epochs = 5
    limits_num_epochs = LimitsNumEpochs(max_num_epochs)

    limits_num_epochs.on_start_training()

    for _ in range(max_num_epochs - 1):
        limits_num_epochs.on_epoch()

    with assert_raises_regexp(StopTraining, "Reached max"):
        limits_num_epochs.on_epoch()


def test_linearly_scales_over_epochs():
    '''
    Unit test for simplelearn.training.LinearlyScalesEpochs.
    '''
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
    '''
    Unit test for simplelearn.training.LinearlyInterpolatesOverEpochs.
    '''
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
