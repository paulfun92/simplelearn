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

    def __init__(self, output_format, **input_nodes):
        if not isinstance(output_format, Format):
            raise TypeError("Expected output_format to be an instance of "
                            "Format, not %s." % type(output_format))

        self.inputs = input_nodes
        self.output_format = output_format
        self.output_symbol = self.define_function(**inputs)

    def _define_function(**input_nodes):
        raise NotImplementedError()


class InputNode(object):
    """
    Represents the input to a model.
    """

    def __init__(self, format):
        super(IdentityNode, self).__init__(output_format=format)
        self.output_symbol = format.get_batch(is_symbolic=True)

    def _define_function(**input_nodes):
        if len(input_nodes) > 0:
            raise ValueError("Unexpected arguments: %s" %
                             str(tuple(input_nodes.iterkeys())))

        return self.output_symbol


class Softmax(Node):
    def __init__(self, output_size, output_dtype='floatX'):
        super(Softmax, self).__init__()

        if not numpy.issubdtype(type(output_size), numpy.integer):
            raise TypeError("The output_size must be an integer, not a %s." %
                            type(output_size))

        self._output_format = DenseFormat(axes=('b', 'f'),
                                          shape=(-1, output_size),
                                          dtype=output_dtype)

        raise NotImplementedError("This class isn't finished")


class Linear(Node):

    def __init__(self, output_size, output_dtype='floatX'):
        super(Linear, self).__init__()

        if not numpy.issubdtype(type(output_size), numpy.integer):
            raise TypeError("The output_size must be an integer, not a %s." %
                            type(output_size))

        if output_size < 0:
            raise ValueError("Got negative output_size: %d" % output_size)

        self.weights = None
        self._output_format = DenseFormat(axes=('b', 'f'),
                                          shape=(-1, output_size),
                                          dtype=output_dtype)

    def get_output_symbol(self, **kwargs):
        """
        Returns dot(X, W), where X is the input, and W are self._weights.

        Parameters
        ----------
        input: theano 2D tensor

        """

        if len(kwargs) != 1:
            raise ValueError("Expected exactly one argument to "
                             "%s.get_output_symbol(), but got %d." %
                             (type(self), len(kwargs)))

        if 'input' not in kwargs:
            raise ValueError("Expected kwarg 'input', but got '%s' instead." %
                             kwargs.iterkeys()[0])

        input_node = kwargs['input']
        if not isinstance(input_node, Node):
            raise TypeError("Expected a subclass of %s as a input, but got a "
                            "%s." % (Node, type(input_node)))

        arg_format = input_node.get_output_format()

        if not isinstance(arg_format, DenseFormat):
            raise TypeError("Expected input's format to be a DenseFormat, "
                            "not a %s." % type(arg_format))

        if frozenset(arg_format.axes) != frozenset(('b', 'f')):
            raise ValueError("Expected input format to consist of 'b' and 'f' "
                             "axes. Instead, got %s." % str(arg_format.axes))

        input_size = arg_format.shape[arg_format.axes.index('f')]
        output_dtype = self._output_format.dtype

        input_format = DenseFormat(axes=('b', 'f'),
                                   shape=(-1, input_size),
                                   dtype=output_dtype)
        input_symbol = arg_format.convert(input_node.get_output_symbol(),
                                          input_format)

        output_size = \
            self._output_format.shape[self._output_format.axes.index('f')]

        weights_shape = (input_size, output_size)
        self.weights = make_shared_variable(numpy.zeros(weights_shape,
                                                        dtype=output_dtype))

        return theano.tensor.dot(input_symbol, self.weights)
