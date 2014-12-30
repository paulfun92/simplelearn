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

    def __init__(self):
        super(Node, self).__init__()
        self._inputs = None
        self._output_format = None

    def get_output_symbol(self, **kwargs):
        """
        Returns the output symbol, defined as a function of the inputs.

        Parameters
        ----------

        kwargs: dict
          The named inputs. Each input must be a Node. The argument name
          must match the argument names expected by this Node.
        """

        raise NotImplementedError("%s.get_output_symbol() not yet implemented."
                                  % type(self))

    def get_output_format(self):
        if self._output_format is None:
            raise RuntimeError("self._output_format not defined.")

        assert isinstance(self._output_format, Format)

        return self._output_format


class Softmax(Node):
    def __init__(self, output_size, output_dtype='floatX'):
        super(Softmax, self).__init__()

        if not numpy.issubdtype(type(output_size), numpy.integer):
            raise TypeError("The output_size must be an integer, not a %s." %
                            type(output_size))

        self._output_format = DenseFormat(axes=('b', 'f'),
                                          shape=(-1, output_size),
                                          dtype=output_dtype)


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
