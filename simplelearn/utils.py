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


def check_is_subdtype(arg, name, expected_dtype):
    """
    Throws a TypeError if arg is not a sub-dtype of expected_type.

    For example, this:

      check_is_subdtype(1.0, "some_float", numpy.integer)

    Throws a TypeError with message "Expected some_float to be a
    <type 'numpy.integer'>, but got a <type 'float'>."
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
        raise TypeError("Expected arg to be a scalar, or an object wth a"
                        " .dtype attribute, but got a %s instead." %
                        type(arg))

    if not numpy.issubdtype(dtype, expected_dtype):
        raise TypeError("Expected %s to be a %s, but got a %s."
                        % (name, expected_dtype, dtype))
