"""
Symbolic functions, which can be composed into a DAG to form a model.
"""

__author__ = "Matthew Koichi Grimes"
__email__ = "mkg@alum.mit.edu"
__copyright__ = "Copyright 2014"
__license__ = "Apache 2.0"


import collections
import numpy
import theano
from nose.tools import (assert_false,
                        assert_equal,
                        assert_is_instance,
                        assert_in)
from simplelearn.utils import safe_izip
from simplelearn.formats import Format, DenseFormat
import pdb

# pylint: disable=too-few-public-methods


class Node(object):

    """
    A Node represents a function.

    The function's input nodes are in self.inputs
    The function's output is self.output_symbol.
    The function's output format is self.output_format.

    A model is a directed acyclic graph (DAG) of Nodes.
    """

    def __init__(self, input_nodes, output_symbol, output_format):
        '''
        Parameters
        ----------
        input_nodes: Node, or Sequence of Nodes

        output_symbol: theano.gof.Variable
          A function of the input_nodes' output_symbols.

        output_format: Format
          The output_symbol's data format.
        '''
        assert_is_instance(input_nodes, (Node, collections.Sequence))
        assert_is_instance(output_symbol, theano.gof.Variable)
        assert_is_instance(output_format, Format)

        if isinstance(input_nodes, Node):
            input_nodes = (input_nodes, )
        else:
            input_nodes = tuple(input_nodes)

        for input_node in input_nodes:
            assert_is_instance(input_node, Node)

        self.inputs = input_nodes
        self.output_format = output_format
        self.output_symbol = output_symbol


class FormatNode(Node):
    '''
    A node that just performs a format conversion.
    '''
    def __init__(self, input_node, output_format, axis_map):
        input_symbol = input_node.output_symbol
        input_format = input_node.output_format
        output_symbol = input_format.convert(input_symbol,
                                             output_format=output_format,
                                             axis_map=axis_map)

        super(FormatNode, self).__init__(input_node,
                                         output_symbol,
                                         output_format)


class InputNode(Node):
    """
    Represents the input to a model.

    Doesn't have any inputs. This just stores an output_symbol and its Format.
    """

    def __init__(self, fmt):
        '''
        Parameters
        ----------
        fmt: Format
          Format of self.output_symbol
        '''
        output_symbol = fmt.make_batch(is_symbolic=True)
        super(InputNode, self).__init__(input_nodes=(),
                                        output_symbol=output_symbol,
                                        output_format=fmt)


class Function1dTo1d(Node):
    """
    Represents a vector-valued function of vectors.

    Takes the input, reshapes it into a matrix[batch_index, feature_index],
    and computes the output using a vector-valued function of the rows.

    This output matrix is then reshaped to the desired output_format.
    """

    @staticmethod
    def _get_bf_shape(fmt):
        """
        Returns the shape of an equivalent format with ('b', 'f') axes.

        Does this by collapsing all axes other than 'b' into a single axis 'f'.
        """

        non_b_sizes = [fmt.shape[fmt.axes.index(a)]
                       for a in fmt.axes if a != 'b']
        f_size = 1 if len(non_b_sizes) == 0 else numpy.prod(non_b_sizes)
        b_size = fmt.shape[fmt.axes.index('b')]
        return (b_size, f_size)

    def __init__(self,
                 input_node,
                 output_format,
                 input_to_bf_map=None,
                 bf_to_output_map=None):
        """
        Input node's format and output_format must both contain 'b' and 'f'
        axes.

        Parameters
        ----------
        input_node: Node
          Node that provides input to this Node.

        output_format: Format
          Format of this node's output_symbol.

        input_to_bf_map: dict
          An axis map from input_node's axes to the ('b', 'f') axes used
          internally.

          See docs for Format.convert() for more detail on axis maps.

          If omitted, this will keep the 'b' axis, and collapse all remaining
          axes into a single 'f' axis.

        bf_to_output_map: dict
          An axis map from the internal ('b', 'f') axes to the output_format.

          See docs for Format.convert() for more detail on axis maps.

          If omitted, this will keep the 'b' axis, and expand the 'f' axis to
          all remaining axes in output_format.
        """

        assert_is_instance(output_format, Format)
        assert_is_instance(input_node, Node)

        input_format = input_node.output_format
        input_axes = input_format.axes
        output_axes = output_format.axes

        #
        # Checks that input and output axes contain 'b', the batch axis.
        #

        assert_in('b', input_axes)
        assert_in('b', output_axes)

        #
        # Creates equivalent bf format to input format.
        #

        get_bf_shape = Function1dTo1d._get_bf_shape

        input_bf_format = DenseFormat(axes=('b', 'f'),
                                      shape=get_bf_shape(input_format),
                                      dtype=input_format.dtype)

        #
        # Creates axis mappings to and from input_bf_format
        #

        if input_to_bf_map is None:
            input_non_b_axes = tuple(a for a in input_axes if a != 'b')
            if len(input_non_b_axes) == 0:
                input_to_bf_map = {'b': ('b', 'f')}
            else:
                input_to_bf_map = {input_non_b_axes: 'f',
                                   'b': 'b'}

        if bf_to_output_map is None:
            output_non_b_axes = tuple(a for a in output_format.axes
                                      if a != 'b')
            if len(output_non_b_axes) == 0:
                bf_to_output_map = {('b', 'f'): 'b'}
            else:
                bf_to_output_map = {'f': output_non_b_axes,
                                    'b': 'b'}

        input_to_bf_node = FormatNode(input_node,
                                      input_bf_format,
                                      input_to_bf_map)

        # format to output format
        output_bf_format = DenseFormat(axes=('b', 'f'),
                                       shape=get_bf_shape(output_format),
                                       dtype=output_format.dtype)

        # compute function of feature vectors
        output_bf_node = self._get_output_bf_node(input_to_bf_node,
                                                  output_bf_format)

        assert_is_instance(output_bf_node, Node)

        bf_to_output_node = FormatNode(output_bf_node,
                                       output_format,
                                       bf_to_output_map)

        super(Function1dTo1d, self).__init__(input_node,
                                             bf_to_output_node.output_symbol,
                                             bf_to_output_node.output_format)

    def _get_output_bf_node(self,
                            input_bf_node,
                            output_bf_format):
        '''
        Returns a Node that takes input with axes=('b', 'f'), and
        yields output with axes=('b', 'f').
        '''
        raise NotImplementedError("%s._get_output_bf_node() not yet "
                                  "implemented." % type(self))


class Linear(Function1dTo1d):
    '''
    Applies a linear transformation to the input.
    '''

    def __init__(self, input_node, output_format):
        get_bf_shape = Function1dTo1d._get_bf_shape

        num_input_features = get_bf_shape(input_node.output_format)[1]
        num_output_features = get_bf_shape(output_format)[1]

        params = numpy.zeros((num_input_features, num_output_features),
                             dtype=input_node.output_symbol.dtype)
        self.params = theano.shared(params)

        super(Linear, self).__init__(input_node, output_format)

    def _get_output_bf_node(self,
                            input_bf_node,
                            output_bf_format):
        output_symbol = theano.tensor.dot(input_bf_node.output_symbol,
                                          self.params)
        return Node(input_bf_node, output_symbol, output_bf_format)


class Bias(Function1dTo1d):
    '''
    Adds a bias to the input.
    '''

    def __init__(self, input_node, output_format):
        get_bf_shape = Function1dTo1d._get_bf_shape

        num_input_features = get_bf_shape(input_node.output_format)[1]
        num_output_features = get_bf_shape(output_format)[1]

        assert_equal(num_output_features, num_input_features)

        params = numpy.zeros((1, num_output_features),
                             dtype=input_node.output_symbol.dtype)

        self.params = theano.shared(params, broadcastable=[True, False])
        super(Bias, self).__init__(input_node, output_format)

    def _get_output_bf_node(self,
                            input_bf_node,
                            output_bf_format):
        output_symbol = input_bf_node.output_symbol + self.params
        return Node(input_bf_node, output_symbol, output_bf_format)


class AffineTransform(Function1dTo1d):
    '''
    Implements dot(X, M) + B (multiplying by a matrix, then adding a bias)
    '''

    def __init__(self,
                 input_node,
                 output_format,
                 input_to_bf_map=None,
                 bf_to_output_map=None):
        super(AffineTransform, self).__init__(input_node,
                                              output_format,
                                              input_to_bf_map,
                                              bf_to_output_map)

    def _get_output_bf_node(self,
                            input_bf_node,
                            output_bf_format):
        self.linear_node = Linear(input_bf_node, output_bf_format)

        # bias node's output format is the same as its input format
        self.bias_node = Bias(self.linear_node,
                              output_format=output_bf_format)

        return self.bias_node


class Softmax(Function1dTo1d):
    '''
    Softmax node.
    '''

    def __init__(self,
                 input_node,
                 output_format=None,
                 input_to_bf_map=None,
                 bf_to_output_map=None):

        if output_format is None:
            output_format = input_node.output_format

        super(Softmax, self).__init__(input_node,
                                      output_format,
                                      input_to_bf_map,
                                      bf_to_output_map)

    def _get_output_bf_node(self, input_bf_node, output_bf_format):
        softmaxes = theano.tensor.nnet.softmax(input_bf_node.output_symbol)
        return Node(input_bf_node, softmaxes, output_bf_format)


class ReLU(Node):
    '''
    Elementwise linear rectifier (zeros any negative elements).

    Preserves format.
    '''
    def __init__(self, input_node):
        input_symbol = input_node.output_symbol
        output_symbol = theano.tensor.switch(input_symbol > 0.0,
                                             input_symbol,
                                             0.0)
        super(ReLU, self).__init__([input_node],
                                   output_symbol,
                                   input_node.output_format)


class L2Loss(Node):
    '''
    Computes ||x_i - y_i||^2 for each x_i, y_i in a batch of data.

    Input nodes must have a batch axis.

    Output axes: ['b']
    '''

    def __init__(self, input_node_a, input_node_b):
        format_a = input_node_a.output_format
        format_b = input_node_b.output_format

        if format_a.shape != format_b.shape or \
           format_a.axes != format_b.axes:
            raise ValueError("Can't take the L2 loss between different "
                             "formats: %s vs %s" % (format_a, format_b))

        def convert_to_bf(node):
            '''
            Returns a node's output_symbol reshaped to have axes=('b', 'f').
            '''

            def make_bf_format(fmt):
                batch_size = fmt.shape[fmt.axes.index('b')]
                feature_size = numpy.prod(tuple(size
                                                for size, axis
                                                in safe_izip(fmt.shape,
                                                             fmt.axes)
                                                if axis != 'b'))

                non_b_axes = tuple(axis for axis in fmt.axes if axis != 'b')
                return DenseFormat(shape=(batch_size, feature_size),
                                   axes=('b', 'f'),
                                   dtype=fmt.dtype), non_b_axes

            bf_format, non_b_axes = make_bf_format(node.output_format)
            axis_map = {'b': 'b',
                        non_b_axes: 'f'}

            return node.output_format.convert(node.output_symbol,
                                              bf_format,
                                              axis_map=axis_map)


        symbol_a_bf = convert_to_bf(input_node_a)
        symbol_b_bf = convert_to_bf(input_node_b)

        diff = symbol_a_bf - symbol_b_bf
        output_symbol = (diff * diff).sum(axis=1)
        output_format = DenseFormat(axes=['b'], shape=[-1], dtype=None)

        super(L2Loss, self).__init__((input_node_a, input_node_b),
                                     output_symbol,
                                     output_format)

class CrossEntropy(Node):
    '''
    Computes the cross-entropy between model outputs and target labels.

    Target labels can either be one-hot vectors (vectors with all zeros except
    for a single 1), or ints. In the latter case, the ints are interpreted as
    the index of the one in a one-hot vector.
    '''

    def __init__(self, softmax_node, targets_node):
        assert_equal(softmax_node.output_format.axes, ('b', 'f'))
        assert_in(targets_node.output_format.axes, [('b', 'f'), 'b'])

        output = theano.tensor.nnet.categorical_crossentropy(
            softmax_node.output_symbol,
            target_node.output_symbol)

        super(CrossEntropy, self).__init__((softmax_node, targets_node),
                                           output,
                                           softmax_node.output_format)

# TODO: cross-entropy
