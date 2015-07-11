"""
Run-time sanity-checks with informative error messages.

All members of this module are assert_* functions, so it should be safe
to import them all as "from simplelearn.asserts import *".
"""

import itertools
import collections
import numpy
import h5py
import theano

from simplelearn.utils import safe_izip

__array_like_types = (numpy.ndarray,
                      numpy.memmap,
                      h5py.Dataset,
                      theano.sandbox.cuda.type.CudaNdarrayType)


from nose.tools import (assert_is_instance,
                        assert_equal,
                        assert_greater,
                        assert_greater_equal,
                        assert_less,
                        assert_less_equal,
                        assert_true)

def assert_is_subdtype(dtype, super_dtype):
    '''
    Checks that <dtype> is a subdtype of <super_dtype>
    '''
    dtype = numpy.dtype(dtype)
    assert_true(numpy.issubdtype(dtype, super_dtype),
                "{} is not a sub-dtype of {}.".format(dtype, super_dtype))


def assert_integer(scalar):
    '''
    Checks that scalar is a scalar of integral type

    Parameters
    ----------
    scalar: a numeric primitive (e.g. int, float, etc).
    '''

    dtype = type(scalar)

    assert_is_subdtype(dtype, numpy.integer)


def assert_floating(scalar):
    '''
    Checks that <scalar> is a scalar of floating-point type.

    Parameters
    ----------
    scalar: a numerical primitive type (e.g. int, float, etc)
    '''
    dtype = type(scalar)

    assert_is_subdtype(dtype, numpy.floating)


def assert_number(scalar):
    '''
    Checks that <scalar> is a number of integer or floating-point type.

    Parameters
    ----------
    scalar: a numerical primitive type (e.g. int, float, etc)
    '''

    dtype = type(scalar)
    assert_is_subdtype(dtype, numpy.number)


def assert_integer_dtype(type):
    assert_is_subdtype(dtype, numpy.integer)


def assert_floating_dtype(dtype):
    assert_is_subdtype(dtype, numpy.floating)


def assert_all_subdtype(iterable, super_dtype):
    '''
    Checks that iterable is an Iterable of integral-typed scalars.

    Parameters
    ----------
    iterable: Iterable

    size: int
      Optional. If specified, this checks that len(iterable) == size.
    '''
    assert_is_instance(iterable, collections.Iterable)

    global __array_like_types

    if isinstance(iterable, __array_like_types):
        assert_is_subdtype(iterable.dtype, super_dtype)
    else:
        for index, element in enumerate(iterable):
            assert_true(numpy.issubdtype(type(element), numpy.integer),
                        "Element %d (%s) is not an integer, but a %s." %
                        (index, element, type(element)))


def assert_all_integer(iterable):
    '''
    Checks that iterable is an Iterable of integral-typed scalars.

    Parameters
    ----------
    iterable: Iterable

    size: int
      Optional. If specified, this checks that len(iterable) == size.
    '''
    assert_all_subdtype(iterable, numpy.integer)


def assert_all_floating(iterable, size=None):
    '''
    Checks that iterable is an Iterable of floating-point scalars.

    Parameters
    ----------
    iterable: Iterable

    size: int
      Optional. If specified, this checks that len(iterable) == size.
    '''
    assert_all_subdtype(iterable, numpy.floating)


def assert_all_true(arg):
    '''
    Checks that all elements of arg are true.
    '''
    assert_is_instance(arg, collections.Iterable)

    global __array_like_types

    if isinstance(iterable, __array_like_types):
        assert_true(numpy.all(arg))

    for index, element in enumerate(arg):
        assert_true(element,
                    "Element {} ({}) is not True.".format(index, element))


def assert_all_is_instance(arg, expected_type):
    '''
    Checks that arg is an Iterable of integral-typed scalars.

    Parameters
    ----------
    arg: Iterable

    expected_type: type
      The type that all elements of arg should be an instance of.
    '''
    assert_is_instance(arg, collections.Iterable)

    for index, element in enumerate(arg):
        assert_is_instance(element, expected_type,
                           "Element %d (%s) is not an integer, but a %s." %
                           (index, element, type(element)))


def assert_all_equal(arg0, arg1=None):
    '''
    Asserts that all elements are equal.

    This can be called with 1 or 2 arguments, as follows:

    1 argument: checks that all elements are equal to each other.
      assert_all_equal([1, 1, 1])

    2 arguments: checks that all elements of arg0 are equal to
                 a scalar arg1.
      assert_all_equal([1, 1, 1], 1)

    2 arguments: checks that all corresponding elements of arg0 and
                 arg1 are equal to each other.
      assert_all_equal([1, 2, 3], (1, 2, 3))


    Parameters
    ----------
    arg0: Sequence
    arg1: scalar, or Sequence (optional)
    '''
    assert_is_instance(arg0, collections.Sequence)

    global __array_like_types

    if isinstance(arg0, __array_like_types):
        if arg1 is None:
            assert_all_equal(arg0[1:], arg0[0])
        else:
            assert_true(numpy.all(arg0 == arg1))


    if arg1 is None:
        first_value = arg0[0]
        for a0 in arg0[1:]:
            assert_equal(a0, first_value)
    elif isinstance(arg1, collections.Iterable):
        for a0, a1 in safe_izip(arg0, arg1):
            assert_equal(a0, a1)
    else:
        for a0 in arg0:
            assert_equal(a0, arg1)

def assert_all_greater_equal(arg0, arg1):
    '''
    Checks that arg0 is a Iterable of scalars greater than or equal to arg1.

    arg1 may be a scalar, or an Iterable of equal length as arg0.
    '''

    global __array_like_types

    if isinstance(arg0, __array_like_types):
        return numpy.all(arg0 >= arg1)

    for (index,
         elem0,
         elem1) in safe_izip(xrange(len(arg0)),
                             arg0,
                             (arg1 if isinstance(arg1, collections.Iterable)
                              else itertools.repeat(arg1, len(arg0)))):
        assert_greater_equal(elem0,
                             elem1,
                             "Element %d: %s was less than %s." %
                             (index, elem0, elem1))


def assert_all_greater(arg0, arg1):
    '''
    Checks that all elements of arg0 are less than arg1.

    arg1 may be a scalar or an Iterable of equal length as arg0.
    '''

    global __array_like_types

    if isinstance(arg0, __array_like_types):
        return numpy.all(arg0 > arg1)

    for (index,
         elem0,
         elem1) in safe_izip(xrange(len(arg0)),
                             arg0,
                             arg1 if isinstance(arg1, collections.Iterable)
                             else itertools.repeat(arg1, len(arg0))):
        assert_greater(elem0,
                       elem1,
                       "Element %d: %s was not greater than %s." %
                       (index, elem0, elem1))


def assert_all_less(arg0, arg1):
    '''
    Checks that all elements of arg0 are less than arg1.

    arg1 may be a scalar or an Iterable of equal length as arg0.
    '''

    global __array_like_types

    if isinstance(arg0, __array_like_types):
        return numpy.all(arg0 < arg1)


    for (index,
         elem0,
         elem1) in safe_izip(xrange(len(arg0)),
                             arg0,
                             arg1 if isinstance(arg1, collections.Iterable)
                             else itertools.repeat(arg1, len(arg0))):
        assert_less(elem0,
                    elem1,
                    "Element %d: %s was not less than %s." %
                    (index, elem0, elem1))


def assert_all_less_equal(arg0, arg1):
    '''
    Checks that all elements of arg0 are less than arg1.

    arg1 may be a scalar or an Iterable of equal length as arg0.
    '''

    global __array_like_types

    if isinstance(arg0, __array_like_types):
        return numpy.all(arg0 <= arg1)

    for (index,
         elem0,
         elem1) in safe_izip(xrange(len(arg0)),
                             arg0,
                             arg1 if isinstance(arg1, collections.Iterable)
                             else itertools.repeat(arg1, len(arg0))):
        assert_less_equal(elem0,
                          elem1,
                          "Element %d: %s was greater than %s." %
                          (index, elem0, elem1))

def assert_parent_dir_exists(file_path):
    '''
    Checks that the file path's parent dir exists.
    '''

    parent_dir = os.path.split(os.path.abspath(file_path))[0]
    assert_true(os.path.isdir(parent_dir),
                "Couldn't find parent directory of '{}'".format(file_path))
