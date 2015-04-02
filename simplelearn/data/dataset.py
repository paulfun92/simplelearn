"""
Static dataset.
"""

__author__ = "Matthew Koichi Grimes"
__email__ = "mkg@alum.mit.edu"
__copyright__ = "Copyright 2014"
__license__ = "Apache 2.0"

import collections
import numpy
from simplelearn.utils import assert_all_equal, assert_integer
from numpy.testing import assert_equal
from nose.tools import (assert_is_instance,
                        assert_less,
                        assert_greater,
                        assert_less_equal,
                        assert_not_equal)
from simplelearn.data import DataSource, DataIterator
from simplelearn.utils import safe_izip
from simplelearn.nodes import InputNode
from simplelearn.formats import Format


class Dataset(DataSource):
    """
    A finite set of data.

    Stores data internally in one or more sliceable multidimensional
    arrays (e.g. numpy.ndarray, numpy.memmap, h5py.Dataset, etc).
    """

    def __init__(self, names, tensors, formats):
        if len(names) != len(formats) or len(names) != len(tensors):
            raise ValueError("Names, formats, and tensors must all have the "
                             "same length, but got %d, %d, and %d "
                             "respectively." %
                             tuple(len(names), len(formats), len(tensors)))

        for name in names:
            if not isinstance(name, basestring):
                raise TypeError("names must be strings, not %s." % type(name))

        for tensor, fmt in safe_izip(tensors, formats):
            if not isinstance(fmt, Format):
                raise TypeError("formats must be Formats, not %s." %
                                type(fmt))

            if 'b' not in fmt.axes:
                raise ValueError("Expected format to contain a 'b' axis "
                                 "(batch axis).")

            fmt.check(tensor)

        self.names = names
        self.formats = formats
        self.tensors = tensors

    @property
    def size(self):
        '''
        The number of examples contained in this Dataset.

        Throws a RuntimeException if this Dataset contains no tensors.
        '''
        if len(self.tensors) == 0:
            raise RuntimeError("This dataset has no tensors, so its "
                               "'size' is undefined.")

        sizes = [t.shape[f.axes.index('b')]
                 for t, f in safe_izip(self.tensors, self.formats)]
        assert_all_equal(sizes)
        return sizes[1]

    @staticmethod
    def _getitem_arg_to_slice(arg):
        '''
        Turns arg into a slice, if it isn't one.
        '''
        if numpy.isscalar(arg):
            assert_integer(arg)
            if arg == -1:
                return slice(arg, None)
            else:
                return slice(arg, arg + 1)
        else:
            assert_is_instance(arg, slice)
            return arg

    def __getitem__(self, arg):
        '''
        Returns a tuple of slices along the batch axis.

        Examples:

          self[slice(3, 10)] returns a 6-element batch from each tensor.

          self[3] is equivalent to self[slice(3, 4)] (batch axis is intact)


        Parameters
        ----------
        arg: integer or slice

        Returns
        -------
        rval: tuple
          A tuple of views into each tensor.
        '''

        batch_slice = self._getitem_arg_to_slice(arg)

        def get_slice_tuple(fmt, batch_slice):
            '''
            Returns a tuple for slicing a tensor along its batch axis.
            '''
            return tuple(batch_slice if axis == 'b'
                         else slice(None)
                         for axis in fmt.axes)

        return tuple(tensor[get_slice_tuple(fmt, batch_slice)]
                     for tensor, fmt
                     in safe_izip(self.tensors, self.formats))

    def iterator(self, iterator_type, batch_size, **kwargs):
        if iterator_type == 'sequential':
            return SequentialIterator(batch_size,
                                      names=self.names,
                                      tensors=self.tensors,
                                      formats=self.formats,
                                      **kwargs)
        else:
            raise NotImplementedError("'%s' iterator type not supported." %
                                      iterator_type)


class SequentialIterator(DataIterator):
    """
    Iterates through samples in a Dataset in memory order.
    """

    def __init__(self, batch_size, names, formats, tensors, **kwargs):
        """
        Parameters
        ----------
        batch_size: int

        names: sequence of strings
          tensors' names.

        formats: sequence of Formats
          tensors' formats. All must have a 'b' (batch) axis.

        tensors: sequence of numpy tensors
          All must have the same number of samples.

        loop_style: str (optional)
          How to handle the case where the number of samples in the dataset
          isn't divisible by the batch_size.

          Choose one of 'wrap', 'truncate', or 'divisible' (default='wrap'):

          'wrap': Wrap off the end of the data, back to the beginning. The
                  final batch of the epoch consists of the last few samples,
                  followed by the first few samples.

          'truncate': Skip the last few samples if there aren't enough to make
                      a batch_size'd batch.

          'divisible': If the number of samples isn't divisible by batch_size,
                       raise a ValueError in the constructor.
        """

        #
        # Sanity-checks arguments
        #

        if not numpy.issubdtype(type(batch_size), numpy.integer):
            raise TypeError("batch_size must be an integer, not a %s." %
                            type(batch_size))

        if not batch_size > 0:
            raise ValueError("batch_size must be positive, not %d." %
                             batch_size)

        if len(names) != len(formats) or len(names) != len(tensors):
            raise ValueError("Expected equal # of names, formats and tensors, "
                             "not %d names, %d formats, and %d tensors." %
                             (len(names), len(formats), len(tensors)))

        if len(formats) == 0:
            raise ValueError("Got empty sequences for 'names', 'formats' & "
                             "'tensors' arguments.")

        for name in names:
            if not isinstance(name, basestring):
                raise TypeError("Expected names to be strings, but got a %s."
                                % type(name))

        for fmt in formats:
            if not isinstance(fmt, Format):
                raise TypeError("Expected formats to be Formats, but got a "
                                "%s.", type(fmt))

        for tensor in tensors:
            if not Format.is_numeric(tensor):
                raise TypeError("Expected tensors to be numeric arrays, but "
                                "got a %s." % type(tensor))

        self._loop_style = kwargs.get('loop_style', None)
        if self._loop_style is None:
            self._loop_style = 'wrap'

        loop_style_values = ('truncate', 'wrap', 'divisible')

        if self._loop_style not in loop_style_values:
            raise ValueError("'loop_style' argument must be one of %s, not "
                             "'%s'." % (str(loop_style_values),
                                        self._loop_style))

        def get_num_samples(tensor, fmt):
            '''
            Returns the size along the batch axis.
            '''
            batch_index = fmt.axes.index('b')
            return tensor.shape[batch_index]

        sample_counts = tuple(get_num_samples(t, f)
                              for t, f in safe_izip(tensors, formats))
        if not all(sc == sample_counts[0] for sc in sample_counts[1:]):
            raise ValueError("Expected all tensors to have the same number of "
                             "samples, but got %s." % str(sample_counts))

        num_samples = sample_counts[0]

        if num_samples < batch_size:
            raise ValueError("# of samples %d must be greater than "
                             "batch_size %d." %
                             (num_samples, batch_size))

        if self._loop_style == 'divisible' and \
           numpy.mod(num_samples, batch_size) != 0:
            raise ValueError("# of samples %d is not divisible by "
                             "batch_size %d (remainder = %d)." %
                             (num_samples,
                              batch_size,
                              numpy.mod(num_samples, batch_size)))

        self._next_batch_start = 0
        self._batch_size = batch_size
        self._names = names
        self._formats = formats
        self._tensors = tensors
        # self.Batch = collections.namedtuple('Batch', names)

        super(SequentialIterator, self).__init__()

    def make_input_nodes(self):
        NamedTupleOfNodes = collections.namedtuple('NamedNodes', self._names)
        nodes = tuple(InputNode(fmt) for fmt in self._formats)

        return NamedTupleOfNodes(*nodes)

    def next_is_new_epoch(self):
        num_samples = self._tensors[0].shape[self._formats[0].axes.index('b')]
        assert_less(self._next_batch_start, num_samples)

        if self._loop_style == 'wrap':
            return self._next_batch_start < self._batch_size
        else:
            return self._next_batch_start == 0

    def _next(self):
        num_samples = \
            self._tensors[0].shape[self._formats[0].axes.index('b')]

        if self._loop_style == 'truncate':
            num_samples = num_samples - numpy.mod(num_samples,
                                                  self._batch_size)

        assert_less(self._next_batch_start, num_samples)

        def get_range(tensor, fmt, start, end):
            """
            Returns a tuple of batches from batch index = start to end.
            """
            assert_less_equal(start, end)
            assert 'b' in fmt.axes
            index = tuple(slice(start, end) if axis == 'b'
                          else slice(None)
                          for axis in fmt.axes)
            result = tensor[index]
            assert_equal(result.shape[fmt.axes.index('b')], end - start)
            return result

        if self._next_batch_start + self._batch_size > num_samples:
            assert_not_equal(self._loop_style,
                             'divisible',
                             "Number of samples %d wasn't divisible by "
                             "batch size %d. This should've been caught "
                             "in the %s constructor." %
                             (num_samples,
                              self._batch_size,
                              type(self)))

            assert_not_equal(self._loop_style,
                             'truncated',
                             "Truncated number of samples %d wasn't divisible "
                             "by batch size %d. It must've been coded wrong." %
                             (num_samples, self._batch_size))

            if self._loop_style == 'wrap':
                batch = tuple(fmt.make_batch(is_symbolic=False,
                                             batch_size=self._batch_size)
                              for fmt in self._formats)

                chunk_size = num_samples - self._next_batch_start
                assert_greater(chunk_size, 0)

                for subbatch, tensor, fmt in safe_izip(batch,
                                                       self._tensors,
                                                       self._formats):
                    get_range(subbatch,
                              fmt,
                              0,
                              chunk_size)[...] = \
                        get_range(tensor,
                                  fmt,
                                  self._next_batch_start,
                                  num_samples)

                    get_range(subbatch,
                              fmt,
                              chunk_size,
                              self._batch_size)[...] = \
                        get_range(tensor,
                                  fmt,
                                  0,
                                  self._batch_size - chunk_size)

                self._next_batch_start = self._batch_size - chunk_size
                return batch
            else:
                raise ValueError("Unrecognized loop_style '%s'. This "
                                 "should've been caught in %s's "
                                 "constructor."
                                 % (self._loop_style, type(self)))

        subbatches = tuple(get_range(tensor,
                                     fmt,
                                     self._next_batch_start,
                                     self._next_batch_start +
                                     self._batch_size)
                           for tensor, fmt
                           in safe_izip(self._tensors, self._formats))

        self._next_batch_start += self._batch_size
        assert_less_equal(self._next_batch_start, num_samples)

        if self._next_batch_start == num_samples:
            self._next_batch_start = 0

        return subbatches
