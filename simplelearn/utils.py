"""
Utility functions used throughout simplelearn
"""

__author__ = "Matthew Koichi Grimes"
__email__ = "mkg@alum.mit.edu"
__copyright__ = "Copyright 2015"
__license__ = "Apache 2.0"


import collections
import numpy
from itertools import chain
from nose.tools import assert_is_instance, assert_equal

def safe_izip(*iterables):
    """
    Like izip, except this raises an IndexError if not all the arguments
    have the same length.
    """

    sentinel = object()

    iterators = tuple(chain(x, (sentinel,)) for x in iterables)

    while iterators:
        items = tuple(next(iterator) for iterator in iterators)

        if all(item is sentinel for item in items):  # all are sentinels
            raise StopIteration()
        elif any(item is sentinel for item in items):  # some are sentinels
            raise IndexError("Can't safe_izip over sequences of different "
                             "lengths.")
        else:
            yield items


def flatten(iterable):
    """
    Returns an iterator that goes through a nested iterable in
    depth-first order.
    """
    for element in iterable:
        if isinstance(element, collections.Iterable) and \
           not isinstance(element, basestring):
            for sub_element in flatten(element):
                yield sub_element
        else:
            yield element


def assert_is_integer(arg):
    '''
    Checks that arg is a scalar of integral type.
    '''
    assert_true(numpy.issubdtype(type(arg), numpy.integer),
                "%s is not an integer" % type(arg))


def assert_are_integers(arg, size=None):
    '''
    Checks that arg is a Sequence of integral-typed scalars.

    Parameters
    ----------
    args: Sequence

    size: int
      Optional. If specified, this checks that len(arg) == size.
    '''
    if size is not None:
        assert_equal(len(arg), size)

    assert_is_instance(arg, Sequence)

    for element, index in enumerate(arg):
        assert_is_integer(arg,
                          "Element %d (%s) is not an integer." %
                          (index, element))


def assert_are_greater_equal(arg, scalar):
    '''
    Checks that arg is a Sequence of scalars greater than or equal to <scalar>.
    '''

    for element, index in enumerate(arg):
        assert_greater_equal(arg,
                             scalar,
                             "Element %d (%s) was less than %s." %
                             (index, element, scalar))


def check_is_subdtype(arg, name, expected_dtype):
    """
    Throws a TypeError if arg is not a sub-dtype of expected_type.

    This function accepts a number of arg types:

    Works on                         Example
    -----------------------------------------------
    scalars                          1.0
    dtypes                           numpy.float32
    strings                          'float32'
    objects with a 'dtype' member    numpy.zeros(3)


    Usage example:

      check_is_subdtype(1.0, "some_float", numpy.integer)

    This throws a TypeError with message "Expected some_float to be a
    <type 'numpy.integer'>, but got a <type 'float'>."

    Parameters
    ----------
    arg: numpy.dtype, its str representation (e.g. 'float32'), a numeric
         scalar, or an object with a dtype field.

    name: str
      arg's argument name. Used in the error message.

    expected_dtype: numpy.dtype or its str representation (e.g. 'float32').
      Expected super-dtype of arg. Examples: numpy.floating, numpy.integer
    """
    if isinstance(arg, numpy.dtype):
        dtype = arg
    elif isinstance(arg, str):
        dtype = numpy.dtype(arg)
    elif hasattr(arg, 'dtype'):
        dtype = arg.dtype
    elif numpy.isscalar(arg):
        dtype = type(arg)
    else:
        raise TypeError("Expected arg to be a dtype, dtype string, numeric "
                        "scalar, or an object wth a .dtype attribute, but "
                        "instead got a %s." % type(arg))

    if not numpy.issubdtype(dtype, expected_dtype):
        raise TypeError("Expected %s to be a %s, but got a %s."
                        % (name, expected_dtype, dtype))
