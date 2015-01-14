from simplelearn.training import ComputesAverageOverEpoch
from simplelearn.formats import DenseFormat
from simplelearn.nodes import Node
from numpy.testing import assert_allclose


class L2Norm(Node):

    def __init__(self, input_node):
        feature_axis = input_node.axes.index('f')
        input_symbol = input_node.output_symbol

        output_symbol = \
            (input_symbol * input_symbol).sum(axis=feature_axis).sqrt()

        output_format = DenseFormat(axes=('b'), shape=(-1, ))
        super(L2Norm, self).__init__(output_symbol,
                                     output_format,
                                     input_node)


def test_computes_average_over_epoch():
    tensor = rng.uniform(3, 10)
    fmt = DenseFormat(axes=('b', 'f'), shape=(-1, 10))
    dataset = Dataset(names=('x', ), formats=(fmt, ), tensors=(tensor, ))

    l2norm_node = L2Norm(*dataset.get_input_nodes())
    averages = []

    l2norms = numpy.sqrt((tensor * tensor).sum(axis=1))
    expected_average = l2norms.sum() / l2norms.size

    averager = ComputesAverageOverEpoch(l2norm_node,
                                        dataset.iterator(batch_size=1),
                                        lambda x: averages.append(x))

    assert_array_near(averager(), expected_average)
