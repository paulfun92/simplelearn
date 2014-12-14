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
from simplelearn.formats import DataFormat


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

    def set_inputs(self, **kwargs):
        """
        Connects some Nodes as inputs.

        Parameters
        ----------
        kwargs: dict
          A dict with strings as keys and Nodes as values.

          The strings are input variable names, mapping to input Nodes.

          The variable names must be the same as those returned by
          _get_input_names().
        """
        if not hasattr(self, 'inputs'):
            raise RuntimeError("self._inputs not found, meaning that "
                               "Node's constructor was probably not called. "
                               "Did your constructor forget to call its "
                               "superclass constructor?")

        expected_input_names = self._get_input_names()
        if not isinstance(expected_input_names, frozenset):
            raise TypeError("%s._get_input_names() returned a %s, "
                            "not a frozenset." %
                            (self.__class__, type(expected_input_names)))

        input_names = frozenset(kwargs.iterkeys())
        if input_names != expected_input_names:
            raise ValueError("Expected arguments %s, but got %s." %
                             (str(expected_input_ames), str(input_names)))

        for input_name, input_node in kwargs.iteritems():
            if not isinstance(input_node, Node):
                raise TypeError("input '%s' was a %s, which isn't a subclass "
                                "of Node." % (input_name, type(input_node)))

        self._inputs.clear()
        self._inputs.update(kwargs)

    def _get_input_names(self):
        """
        Returns the expected input names as a set.

        Returns
        -------
        rval: frozenset
          A frozenset of strings.
        """
        raise NotImplementedException("%s.get_input_names() not "
                                      "implemented." % self.__class__)

    def get_output_symbol(self, batch_size):
        """
        Returns a theano symbol that represents the output.

        Also allocates any internal parameters (e.g. weights),
        sized according to batch_size and the input size.
        """
        if not numpy.issubdtype(type(batch_size), 'int'):
            raise TypeError("batch_size was a %s, not an int." %
                            type(batch_size))

        if batch_size < 0:
            raise ValueError("Expected batch_size to be non-negative, but got "
                             "%d." % batch_size)

        input_vars = dict(name, node.get_output_var(batch_size)
                          for name, node in self._inputs.iteritems())
        result = self._get_output_symbol(**input_vars)
        if not isinstance(result, T.TensorVariable):
            raise TypeError("get_output_symbol_impl() returned a %s, not a "
                            "theano.tensor.TensorVariable")

        return result

    def _get_output_symbol(self, **kwargs):
        """
        Implementation of get_output_symbol().

        All subclasses of Node must implement this method.

        Parameters
        ----------

        kwargs: dict
          A dict with strings as keys and Theano symbols as values.
          The strings are variable names (as specified by _get_input_names).
          The values are Theano symbols, representing the inputs' outputs.

        Returns
        -------

        rval: theano.TensorVariable
        """
        raise NotImplementedError("%s.symbolically_evaluate()() not "
                                  "implemented." % self.__class__)

    def get_output_format(self):
        return self._output_format

    def get_output_shape(self, batch_size):
        """
        Returns the output tensor's shape.

        This is likely to be a function of batch_size, the shape of internal
        parameters, and the output sizes of connected inputs, listed in
        self._inputs.

        Returns
        -------
        rval: tuple
          A tuple of nonnegative ints.
        """

        result = get_output_shape_impl()

        if not isinstance(result, tuple):
            raise TypeError("get_output_shape_impl() returned a %s, not a "
                            "tuple." % type(result))

        if any(not numpy.issubdtype(s, 'int') for s in result):
            raise TypeError("%s._get_output_shape() contained some non-ints: "
                            "%s" % (self.__class__, str(result)))

        if any(s < 0 for s in result):
            raise ValueError("%s._get_output_shape() returned some negative "
                             "numbers: %s" % (self.__class__, str(result)))

        return result

    def _get_output_shape(self, batch_size):
        """
        Implementation of Node.get_output_shape() (same method signature).
        """

        raise NotImplementedError("%s.get_output_shape_impl() not "
                                  "implemented." % self.__class__)


class TensorNode(Node):
    """
    A Node with tensor inputs and outputs in specific DataFormats.
    """

    def __init__(self, format):
        if not isinstance(format, DataFormat):
            raise TypeError("%s() expected a DataFormat argument, but got a "
                            "%s." % (type(self), type(format)))

        self.format = format

    def set_inputs(self, **kwargs):
        for value in kwargs.valueiter():
            check

    def _get_input_names(self):
        return frozenset(('input',))

    def _get_output_symbol(self):
        input_node = self._inputs['input']
        input_format = input_node.get_output_format()
        input_symbol = input_node.get_output_symbol()

        return input_format.format(input_symbol)


class Linear(Node):

    def __init__(self, output_size):
        super(Linear, self).__init__()
        self.weights = None
        self.output_size = output_size

    def _get_input_names(self):
        return frozenset(("input", ))

    def _get_output_shape(self, batch_size):
        return (batch_size, self.output_size)

    def get_output_symbol(self, batch_size):
        assert len(self._inputs) == 1

        def allocate_weight(input_shape, output_shape):
        return T.dot(self._inputs.input, self.weights)
