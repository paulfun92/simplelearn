import numpy
from numpy.testing import assert_allclose
from nose.tools import assert_equal
import theano.tensor as T

from simplelearn.training import ComputesAverageOverEpoch
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

    l2norm_node = L2Norm(*dataset.get_input_nodes())
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
        assert_equal(len(averages), ii)
        assert_allclose(averages[ii], expected_average)
