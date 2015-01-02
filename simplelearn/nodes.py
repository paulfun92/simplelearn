"""
A typical workflow for setting up a node B and getting its function is:

1) Set up all Nodes that will serve as inputs to B.

  B.set_inputs(argument_name_0=node_0, argument_name_1=node_1, ...)

2) Get B's output symbol (theano variable) y:

  y = Y.get_output_symbol(batch_size=n)

3) As usual, get an evaluatable function w.r.t. input symbols a1 ... aN
   by doing:

  y_func = theano.function([a1, ..., aN], y)
"""

import numpy
import theano
from simplelearn.formats import Format, DenseFormat


def make_shared_variable(initial_value, name=None):
    """
    Creates a shared variable, initializing its contents with initial_value.

    Parameters
    ----------
    initial_value: numpy array
      Determines shape, dtype, and initial value of the return value.

    name: str or None
      If provided, the return value's name will be set to this.

    Returns
    -------
    rval: theano shared variable.
    """
    # pylint: disable=protected-access
    return theano.shared(theano._asarray(initial_value,
                                         dtype=initial_value.dtype),
                         name=name)


class Node(object):

    """
    A Node represents a function that outputs a theano tensor.

    It takes input(s) from other nodes, listed in self._inputs.

    A model is a directed acyclic graph (DAG) of Nodes.
    """

    def __init__(self, output_symbol, output_format, *input_nodes):
        if not isinstance(output_symbol, theano.gof.Variable):
            raise TypeError("Expected output_symbol to be a theano variable, "
                            "not a %s." % type(output_symbol))

        if not isinstance(output_format, Format):
            raise TypeError("Expected output_format to be an instance of "
                            "Format, not %s." % type(output_format))

        for input_node in input_nodes:
            if not isinstance(input_node, Node):
                raise TypeError("Expected input_nodes to be Nodes, but got a "
                                "%s." % type(input_node))

        self.inputs = input_nodes
        self.output_format = output_format
        self.output_symbol = output_symbol


class InputNode(Node):
    """
    Represents the input to a model.

    Doesn't have any inputs. This just stores an output_symbol and its Format.
    """

    def __init__(self, format):
        output_symbol = format.get_batch(is_symbolic=True)
        super(InputNode, self).__init__(output_symbol=output_symbol,
                                        output_format=format)


class Function1dTo1d(Node):
    """
    Represents a 1-D function of 1-D vectors.

    Internally reshapes input to a 2-D matrix, independently computes
    a function of each row, then reshapes the result to the output format.
    """

    @staticmethod
    def _get_bf_shape(fmt):
        """
        Returns the shape of an equivalent format with ('b', 'f') axes.
        """

        non_b_sizes = [fmt.shape[fmt.axes.index(a)] for a in fmt.axes]
        f_size = numpy.prod(non_b_sizes)
        b_size = fmt.axes.index('b')
        return (b_size, f_size)

    def __init__(self, output_format, input_node):
        """
        Input node's format and output_format must both contain 'b' and 'f'
        axes.
        """

        #
        # Checks that input and output axes contain 'b' and 'f'.
        #

        input_axes = input_node.format.axes
        output_axes = output_format.axes

        for expected_axis in ('b', 'f'):
            for (format_axes,
                 format_name) in safe_izip((input_axes, output_axes),
                                           ('input', 'output')):
                if expected_axis not in format_axes:
                    raise ValueError("Expected %s format's axes to contain "
                                     "axis '%s', but it didn't: %s." %
                                     (format_name,
                                      expected_axis,
                                      str(format_axes)))

        #
        # Creates equivalent bf format to input format.
        #

        get_bf_shape = Function1dTo1d._get_bf_shape

        bf_shape = get_bf_shape(input_node.output_format)
        if get_bf_shape(output_format) != bf_shape:
            raise ValueError("Total size of non-batch axes of input and "
                             "output formats don't match (%d vs %d)."
                             % (bf_shape.shape[0],
                                get_bf_shape(output_format)[0]))

        bf_format = DenseFormat(axes=('b', 'f'),
                                shape=bf_shape,
                                dtype=input_format.dtype)

        #
        # Creates axis mappings to and from bf_format
        #

        input_b_axes = tuple(a for a in input_axes if a != 'f')
        input_to_bf_map = {'f': 'f',
                           input_b_axes: 'b'}
        output_b_axes = tuple(a for a in output_format.axes if a != 'f')
        bf_to_output_map = {'f': 'f',
                            'b': output_b_axes}

        #
        # Creates this node's function/output symbol.
        #

        input_symbol = input_node.output_symbol
        input_format = input_node.output_format

        # format input to bf_format
        bf_symbol = input_format.convert(input_symbol,
                                         bf_format,
                                         axis_map=input_to_bf_map)

        # compute function of feature vectors
        bf_symbol = self._get_function_of_rows(bf_symbol)

        # format to output format
        output_symbol = bf_format.convert(bf_symbol,
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

    def __init__(self, output_format, input_node):
        super(Linear, self).__init__(output_format, input_node)

        num_input_features = get_bf_shape(input_node.format)[1]
        num_output_features = get_bf_shape(output_format)[1]

        weights_shape = (num_input_features, num_output_features)
        self.weights = make_shared_variable(numpy.zeros(weights_shape,
                                                        dtype=output_dtype))

    def _get_function_of_rows(self, rows_symbol):
        """
        Returns dot(X, W), where X is the input, and W are self.weights.

        Parameters
        ----------
        rows_symbol: X, a theano 2D tensor
        """
        return theano.tensor.dot(rows_symbol, self.weights)


class Softmax(FunctionOfFeatures):
    def __init__(self, output_format, input_node):
        super(Softmax, self).__init__(output_format, input_node)

    def _get_function_of_rows(self, features_symbol):
        raise NotImplementedError()
