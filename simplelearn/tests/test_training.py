import numpy
from numpy.testing import assert_allclose
import theano.tensor as T
from nose.tools import assert_equal, assert_raises_regexp

from simplelearn.training import (ComputesAverageOverEpoch,
                                  StopsOnStagnation,
                                  StopTraining,
                                  LimitsNumEpochs)
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


def test_computes_average_over_epoch():
    rng = numpy.random.RandomState(3851)
    tensor = rng.uniform(-1.0, 1.0, size=(3, 10))
    fmt = DenseFormat(axes=('b', 'f'), shape=(-1, 10), dtype=tensor.dtype)
    dataset = Dataset(names=('x', ), formats=(fmt, ), tensors=(tensor, ))

    l2norm_node = L2Norm(*dataset.make_input_nodes())
    averages = []

    l2norms = numpy.sqrt((tensor * tensor).sum(axis=1))
    expected_average = l2norms.sum() / l2norms.size

    averager = ComputesAverageOverEpoch(l2norm_node,
                                        dataset.iterator('sequential',
                                                         batch_size=1,
                                                         loop_style="wrap"),
                                        (lambda x: averages.append(x), ))

    for ii in range(2):
        averager()
        assert_equal(len(averages), ii + 1)
        assert_allclose(averages[ii], expected_average)


def test_stops_on_stagnation():

    def get_values(stepsize, kink_index):
        descending_values = (numpy.arange(kink_index) * (-stepsize)) + 100.
        flat_values = numpy.zeros(kink_index)
        flat_values[:] = descending_values[-1]
        return numpy.concatenate((descending_values, flat_values))

    threshold = .1
    kink_index = 20
    values = get_values(threshold + .0001, kink_index)

    num_epochs = 5
    assert num_epochs < kink_index

    index = 0
    stops_on_stagnation = StopsOnStagnation("hockey stick",
                                            num_epochs,
                                            threshold)

    try:
        for value in values:
            stops_on_stagnation(value)
            index += 1
    except StopTraining, st:
        assert_equal(index, kink_index + num_epochs)
        assert_equal(st.status, 'ok')
        assert "didn't decrease for" in st.message


def test_limits_num_epochs():
    limits_num_epochs = LimitsNumEpochs(5)
    for index in range(4):
        limits_num_epochs()

    assert_raises_regexp(StopTraining,
                         "Reached max \# of epochs",
                         limits_num_epochs())
