'''
Utility functions used throughout simplelearn
'''

from __future__ import print_function

__author__ = "Matthew Koichi Grimes"
__email__ = "mkg@alum.mit.edu"
__copyright__ = "Copyright 2015"
__license__ = "Apache 2.0"


import os
import collections
import urllib2
import itertools
import numpy
import h5py
import theano
from nose.tools import assert_not_equal, assert_true

import pdb

import theano.sandbox.cuda.type


def safe_izip(*iterables):
    """
    Like izip, except this raises an IndexError if not all the arguments
    have the same length.
    """

    sentinel = object()

    iterators = tuple(itertools.chain(x, (sentinel,)) for x in iterables)

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


def human_readable_memory_size(size, precision=2):
    '''
    Converts a number (# of bytes) to a human-readable string, e.g. "2.4 GB".
    '''
    if size < 0:
        raise ValueError("Size must be non-negative (was %g)." % size)

    suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
    suffix_index = 0
    while size > 1024:
        suffix_index += 1  # increment the index of the suffix
        size = size / 1024.0  # apply the division
    return "%.*f %s" % (precision, size, suffixes[suffix_index])


def human_readable_duration(seconds):
    '''
    Converts a duration in secs to a human-readable string, e.g. "15 m 29.2s".
    '''

    minutes = seconds // 60
    seconds = seconds % 60
    result = "%g s" % seconds

    if minutes > 0:
        hours = minutes // 60
        minutes = minutes % 60
        result = ("%d m " % minutes) + result

        if hours > 0:
            days = hours // 24
            hours = hours % 24
            result = ("%d h " % hours) + result

            if days > 0:
                result = ("%d d " % days) + result

    return result


def download_url(url,
                 local_directory=None,
                 local_filepath=None,
                 show_progress=True):
    '''
    Downloads a file from a URL.

    Parameters
    ----------
    url: str
      The web address of the file to download.

    local_directory: str
      The directory to download to. Mutually exclusive to local_filepath.

    local_filepath: str
      The full filepath to download to. Mutually exclusive to local_directory.

    local_filepath: str
      The file path to download to. If this is a directory, the original
      filename will be used. If this is a file path, the file name will be
      changed.

    show_progress: bool
      If true, print a progress bar.
    '''
    assert_not_equal(local_directory is None, local_filepath is None)

    if local_directory is not None:
        assert_true(os.path.isdir(local_directory))
        local_filepath = os.path.join(local_directory,
                                      url.split('/')[-1])
    else:
        local_directory = os.path.split(local_filepath)[0]
        assert_true(os.path.isdir(local_directory))

    file_name = os.path.split(local_filepath)[1]

    url_handle = urllib2.urlopen(url)

    with open(local_filepath, 'wb') as file_handle:
        metadata = url_handle.info()
        file_size = int(metadata.getheaders("Content-Length")[0])

        if show_progress:
            # print("Downloading: %s Bytes: %s" % (file_name, file_size))
            print("Downloading {} ({})".format(
                file_name,
                human_readable_memory_size(file_size)))

        downloaded_size = 0
        block_size = 8192
        while True:
            buffer = url_handle.read(block_size)
            if not buffer:
                break

            downloaded_size += len(buffer)
            file_handle.write(buffer)
            if show_progress:
                print("{:10d}  [{:3.2f}%]".format(
                    downloaded_size,
                    downloaded_size * 100. / file_size),
                      end='\r')


def cudnn_available(arg=None):
    '''
    Returns True iff cuDNN, or a particular component of it, is available.

    Available means that cuDNN is installed, and a particular component
    hasn't been deactivated by setting theano.config.optimizer_excluding
    to 'cudnn', 'conv_dnn', or 'pool_dnn'.
    '''

    if not theano.sandbox.cuda.dnn.dnn_available:
        return False
    elif 'cudnn' in theano.config.optimizer_excluding:
        return False
    elif arg == 'conv':
        if 'conv_dnn' in theano.config.optimizer_excluding:
            return False
    elif arg == 'pool' and 'pool_dnn':
        if theano.config.optimizer_excluding:
            return False
    elif arg is not None:
        raise ValueError("Expected cudnn_available() argument to be 'conv' "
                         "or 'pool', but got '{}'.".format(arg))

    return True



# def check_is_subdtype(arg, name, expected_dtype):
#     """
#     Throws a TypeError if arg is not a sub-dtype of expected_type.

#     This function accepts a number of arg types:

#     Works on                         Example
#     -----------------------------------------------
#     scalars                          1.0
#     dtypes                           numpy.float32
#     strings                          'float32'
#     objects with a 'dtype' member    numpy.zeros(3)


#     Usage example:

#       check_is_subdtype(1.0, "some_float", numpy.integer)

#     This throws a TypeError with message "Expected some_float to be a
#     <type 'numpy.integer'>, but got a <type 'float'>."

#     Parameters
#     ----------
#     arg: numpy.dtype, its str representation (e.g. 'float32'), a numeric
#          scalar, or an object with a dtype field.

#     name: str
#       arg's argument name. Used in the error message.

#     expected_dtype: numpy.dtype or its str representation (e.g. 'float32').
#       Expected super-dtype of arg. Examples: numpy.floating, numpy.integer
#     """
#     if isinstance(arg, numpy.dtype):
#         dtype = arg
#     elif isinstance(arg, str):
#         dtype = numpy.dtype(arg)
#     elif hasattr(arg, 'dtype'):
#         dtype = arg.dtype
#     elif numpy.isscalar(arg):
#         dtype = type(arg)
#     else:
#         raise TypeError("Expected arg to be a dtype, dtype string, numeric "
#                         "scalar, or an object wth a .dtype attribute, but "
#                         "instead got a %s." % type(arg))

#     if not numpy.issubdtype(dtype, expected_dtype):
#         raise TypeError("Expected %s to be a %s, but got a %s."
#                         % (name, expected_dtype, dtype))
