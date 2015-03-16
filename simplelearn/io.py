'''
Functionality related to serialization and data I/O.
'''

import theano
from nose.tools import assert_is_instance
from collections import Sequence

class SerializableModel(object):
    def __init__(self, input_nodes, output_nodes):
        assert_is_instance(input_nodes, Sequence)
        assert_is_instance(output_nodes, Sequence)

        self._input_nodes = input_nodes
        self._output_nodes = output_nodes

    def compile_function(self):
        def get_symbols(nodes):
            return [n.output_symbol for n in nodes]

        return theano.function(get_symbols(self.input_nodes),
                               get_symbols(self.output_nodes))
