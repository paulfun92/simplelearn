"""
Symbolic functions, which can be composed into a DAG to form a model.
"""

__author__ = "Matthew Koichi Grimes"
__email__ = "mkg@alum.mit.edu"
__copyright__ = "Copyright 2014"
__license__ = "Apache 2.0"


import numpy
import theano
from nose.tools import assert_equal, assert_is_instance, assert_in
from simplelearn.utils import safe_izip
from simplelearn.formats import Format, DenseFormat
import pdb

class Node(object):

    """
    A Node represents a function.

    The function's input nodes are in self.inputs
    The function's output is self.output_symbol.
    The function's outptu format is self.output_format.

    A model is a directed acyclic graph (DAG) of Nodes.
    """

    def __init__(self, output_symbol, output_format, *input_nodes):
        assert_is_instance(output_symbol, theano.gof.Variable)
        assert_is_instance(output_format, Format)

        for input_node in input_nodes:
            assert_is_instance(input_node, Node)

        self.inputs = input_nodes
        self.output_format = output_format
        self.output_symbol = output_symbol


class InputNode(Node):
    """
    Represents the input to a model.

    Doesn't have any inputs. This just stores an output_symbol and its Format.
    """

    def __init__(self, fmt):
        output_symbol = fmt.make_batch(is_symbolic=True)
        super(InputNode, self).__init__(output_symbol=output_symbol,
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
                 output_format,
                 input_node,
                 input_to_bf_map=None,
                 bf_to_output_map=None):
        """
        Input node's format and output_format must both contain 'b' and 'f'
        axes.

        Parameters
        ----------
        output_format: Format
          Format of this node's output_symbol.

        input_node: Node
          Node that provides input to this Node.

        input_to_bf_map: dict
          A mapping from
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
                input_to_bf_map = {'b' : ('b', 'f')}
            else:
                input_to_bf_map = {input_non_b_axes: 'f',
                                   'b': 'b'}

        if bf_to_output_map is None:
            output_non_b_axes = tuple(a for a in output_format.axes
                                      if a != 'b')
            if len(output_non_b_axes) == 0:
                bf_to_output_map = {('b', 'f') : 'b'}
            else:
                bf_to_output_map = {'f': output_non_b_axes,
                                    'b': 'b'}

        #
        # Creates this node's function/output symbol.
        #

        input_symbol = input_node.output_symbol

        # format input to bf_format
        input_bf_symbol = input_format.convert(input_symbol,
                                               input_bf_format,
                                               axis_map=input_to_bf_map)

        # compute function of feature vectors
        output_bf_symbol = self._get_function_of_rows(input_bf_symbol)

        # format to output format
        output_bf_format = DenseFormat(axes=('b', 'f'),
                                       shape=get_bf_shape(output_format),
                                       dtype=output_format.dtype)

        output_symbol = output_bf_format.convert(output_bf_symbol,
                                                 output_format,
                                                 axis_map=bf_to_output_map)

        super(Function1dTo1d, self).__init__(output_symbol,
                                             output_format,
                                             input_node)

    def _get_function_of_rows(self, rows_symbol):
        """
        Returns a symbol f(x), where f is a function of rows of x.

        rows_symbol: theano variable
          A matrix where each row is an input to f.
        """
        raise NotImplementedError("%s._get_function_of_features() not yet "
                                  "implemented." % type(self))


class Linear(Function1dTo1d):
    '''
    Applies a linear transformation to the input.
    '''

    def __init__(self, output_format, input_node):
        get_bf_shape = Function1dTo1d._get_bf_shape

        num_input_features = get_bf_shape(input_node.output_format)[1]
        num_output_features = get_bf_shape(output_format)[1]

        params = numpy.zeros((num_input_features, num_output_features),
                             dtype=output_format.dtype)
        self.params = theano.shared(params)

        super(Linear, self).__init__(output_format, input_node)

    def _get_function_of_rows(self, rows_symbol):
        """
        Returns dot(X, W), where X is the input, and W are self.params.

        Parameters
        ----------
        rows_symbol: X, a theano 2D tensor

        Returns
        -------
        rval: a theano 2D tensor
          rval[i, :] := dot(rows_symbol[i, :], W)
        """
        return theano.tensor.dot(rows_symbol, self.params)


class Bias(Function1dTo1d):
    '''
    Adds a bias to the input.
    '''

    def __init__(self, output_format, input_node):
        get_bf_shape = Function1dTo1d._get_bf_shape

        num_input_features = get_bf_shape(input_node.output_format)[1]
        num_output_features = get_bf_shape(output_format)[1]

        assert_equal(num_output_features, num_input_features)

        params = numpy.zeros((1, num_output_features),
                             dtype=output_format.dtype)

        self.params = theano.shared(params, broadcastable=[True, False])
        super(Bias, self).__init__(output_format, input_node)

    def _get_function_of_rows(self, rows_symbol):
        return rows_symbol + self.params


class L2Loss(Node):
    def __init__(self, input_node_a, input_node_b):
        diff = input_node_a.output_symbol - input_node_b.output_symbol
        output_symbol = (diff * diff).sum()
        output_format = DenseFormat(axes=(),
                                    shape=(),
                                    dtype=None)

        super(L2Loss, self).__init__(output_symbol,
                                     output_format,
                                     input_node_a,
                                     input_node_b)

class AffineTransform(Function1dTo1d):
    def __init__(self,
                 output_format,
                 input_node,
                 input_to_bf_map=None,
                 bf_to_output_map=None):
        self.linear_node = Linear(input_node,
                                  input_to_bf_map=input_to_bf_map)

        # bias node's output format is the same as its input format
        self.bias_node = Bias(self.linear_node,
                              self.linear_node.output_format,
                              bf_to_output_map=bf_to_output_map)

        super(AffineTransform, self).__init__(bias_node.output_format,
                                              bias_node,
                                              bf_to_output_map)

    def _get_function_of_rows(self, input_bf_symbol):
        return self.bias_node.output_symbol


class Softmax(Function1dTo1d):
    def __init__(self, output_format, input_node):
        super(Softmax, self).__init__(output_format, input_node)

    def _get_function_of_rows(self, features_symbol):
        raise NotImplementedError()
