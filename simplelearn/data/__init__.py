"""
Data sources (a dataset is a static data source).
"""

__author__ = "Matthew Koichi Grimes"
__email__ = "mkg@alum.mit.edu"
__copyright__ = "Copyright 2014"
__license__ = "Apache 2.0"

import os
from simplelearn.formats import Format

def _get_data_path():
    env_variable = 'SIMPLELEARN_DATA_PATH'

    if env_variable not in os.environ:
        raise RuntimeError("Environment variable %s not "
                           "defined." % env_variable)

    result = os.environ[env_variable]

    if not os.path.isdir(result):
        raise RuntimeError("Couldn't find %s directory "
                           "(%s). Please create it." %
                           (env_variable, result))

    return result


data_path = _get_data_path()


class DataSource(object):
    """
    A source of data.
    """

    def iterator(self, iterator_type, batch_size, **kwargs):
        raise NotImplementedError("%s.iterator() not yet implemented." %
                                  type(self))

    def make_input_nodes(self):
        """
        Returns
        -------
        rval: instance of a collections.namedtuple type.
          A named tuple of InputNodes. The names must correspond to the names
          used by iterators' next() method (e.g. 'model_input', 'label').
        """
        raise NotImplementedError("%s.make_input_nodes() not yet implemented."
                                  % type(self))


class DataIterator(object):
    """
    Yields data batches from a data source.
    """

    def __init__(self):
        # Used for ensuring that next_is_new_epoch() is implemented correctly.
        self._next_was_called = False

    def __iter__(self):
        return self

    def next_is_new_epoch(self):
        """
        Returns True if next() will return a datum in a new epoch.

        The first epoch counts as a new epoch.
        """
        raise NotImplementedError("%s.next_is_new_epoch() not yet implemented."
                                  % type(self))

    def next(self):
        """
        Returns a batch of data, and increments the iterator.

        The returned batch is in a collections.namedtuple type.  For example, a
        common one will be collections.namedtuple('DataAndLabel', ('data',
        'label')). A returned batch can be indexed as a tuple::

          batch[0]  # data
          batch[1]  # label

        ...or using named fields::

          batch.data  # data
          batch.label  # label

        These field names must match the names returned by the dataset's
        make_input_nodes.

        Returns
        -------

        rval: instance of a collections.namedtuple type.
        """

        if not self._next_was_called:
            if not self.next_is_new_epoch():
                raise ValueError("%s.next_is_new_epoch() implemented "
                                 "incorrectly: if next() hasn't yet been "
                                 "called, next_is_new_epoch() must return "
                                 "True. (It returned False.)")

        # self._batch = self._next()
        result = self._next()

        if not isinstance(result, tuple) or \
           not all(Format.is_numeric(r) for r in result):
            raise TypeError("%s._next() implemented incorrectly: It must "
                            "return a tuple of numeric arrays, but got "
                            "something else.")

        self._next_was_called = True
        return result

    def _next(self):
        """
        Implements next(). See that method's docs for details.
        """
        raise NotImplementedError("%s._next() not yet implemented." %
                                  type(self))
