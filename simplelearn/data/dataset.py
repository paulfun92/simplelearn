"""
Static dataset.
"""

__author__ = "Matthew Koichi Grimes"
__email__ = "mkg@alum.mit.edu"
__copyright__ = "Copyright 2014"
__license__ = "Apache 2.0"

import collections
import numpy
from numpy.testing import assert_equal
from nose.tools import (assert_less,
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

    Stores data internally in one or more indexable numpy arrays (e.g. ndarray,
    memmap).
    """

    def __init__(self, names, formats, tensors):
        if not all(len(x) == len(names) for x in (formats, tensors)):
            raise ValueError("Names, formats, and tensors must all have the "
                             "same length, but got %d, %d, and %d "
                             "respectively." %
                             tuple(len(names), len(formats), len(tensors)))

        for name in names:
            if not isinstance(name, str):
                raise TypeError("names must be strings, not %s." % type(name))

        for tensor, fmt in safe_izip(tensors, formats):
            if not isinstance(fmt, Format):
                raise TypeError("formats must be Formats, not %s." %
                                type(fmt))

            if 'b' not in fmt.axes:
                raise ValueError("Expected format to contain a 'b' axis "
                                 "(batch axis).")

            fmt.check(tensor)

        DataTuple = collections.namedtuple('DataTuple', names)
        NodeAndTensor = collections.namedtuple('NodeAndTensor',
                                               ('node', 'tensor'))

        self.data = DataTuple(*tuple(NodeAndTensor(InputNode(fmt), tensor)
                                     for fmt, tensor
                                     in safe_izip(formats, tensors)))

    def iterator(self, iterator_type, batch_size, **kwargs):
        if iterator_type == 'sequential':
            tensors = tuple(d.tensor for d in self.data)
            formats = tuple(d.node.output_format for d in self.data)
            return SequentialIterator(batch_size,
                                      names=self.data._fields,
                                      tensors=tensors,
                                      formats=formats,
                                      **kwargs)
        else:
            raise NotImplementedError("'%s' iterator type not supported." %
                                      iterator_type)


class SequentialIterator(DataIterator):
    """
    Iterates through samples in a Dataset in memory order.
    """

    def __init__(self, batch_size, names, tensors, formats, **kwargs):
        """
        Parameters
        ----------
        batch_size: int

        names: sequence of strings
          tensors' names.

        tensors: sequence of numpy tensors
          All must have the same number of samples.

        formats: sequence of Formats
          tensors' formats. All must have a 'b' (batch) axis.

        mode: str (optional)
          How to handle the case where the number of samples in the dataset
          isn't divisible by the batch_size.

          Choose one of 'loop', 'truncate', or 'divisible' (default='loop'):

          'loop': Loop off the end (the final batch consists of the last few
                  samples, followed by the first few samples).

          'truncate': Skip the last few samples if there aren't enough to make
                      a batch_size'd batch.

          'divisible': If the number of samples isn't divisible by batch_size,
                       raise a ValueError in the constructor.

        """
        if not numpy.issubdtype(type(batch_size), numpy.integer):
            raise TypeError("batch_size must be an integer, not a %s." %
                            type(batch_size))

        if not batch_size > 0:
            raise ValueError("batch_size must be positive, not %d." %
                             batch_size)

        if len(formats) != len(tensors):
            raise ValueError("Expected equal # of formats and tensors, "
                             "not %d formats and %d tensors." %
                             (len(formats), len(tensors)))

        if len(formats) == 0:
            assert_equal(len(tensors), 0)
            raise ValueError("Got empty sequence for 'formats' & "
                             "'tensors' arguments.")

        for tensor in tensors:
            if not Format.is_numeric(tensor):
                raise TypeError("Expected tensors to be numeric arrays, but "
                                "got a %s." % type(tensor))

        for fmt in formats:
            if not isinstance(fmt, Format):
                raise TypeError("Expected formats to be Formats, but got a "
                                "%s.", type(fmt))

        self._mode = kwargs.get('mode', None)
        if self._mode is None:
            self._mode = 'loop'

        mode_values = ('truncate', 'loop', 'divisible')

        if self._mode not in mode_values:
            raise ValueError("'mode' argument must be one of %s, not '%s'." %
                             (str(mode_values), self._mode))

        def get_num_samples(tensor, fmt):
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

        if self._mode == 'divisible' and \
           numpy.mod(num_samples, batch_size) != 0:
            raise ValueError("# of samples %d is not divisible by "
                             "batch_size %d (remainder = %d)." %
                             (num_samples,
                              batch_size,
                              numpy.mod(num_samples, batch_size)))

        self._epoch = -1
        self._next_batch_start = 0
        self._batch_size = batch_size
        self._formats = formats
        self._tensors = tensors
        self.Batch = collections.namedtuple('Batch', names)

    def epoch(self):
        return self._epoch

    def next(self):
        num_samples = \
            self._tensors[0].shape[self._formats[0].axes.index('b')]

        if self._mode == 'truncate':
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
            assert_not_equal(self._mode,
                             'divisible',
                             "Number of samples %d wasn't divisible by "
                             "batch size %d. This should've been caught "
                             "in the %s constructor." %
                             (num_samples,
                              self._batch_size,
                              type(self)))

            assert_not_equal(self._mode,
                             'truncated',
                             "Truncated number of samples %d wasn't divisible "
                             "by batch size %d. It must've been coded wrong." %
                             (num_samples, self._batch_size))

            if self._mode == 'loop':
                batch = self.Batch(*(fmt.make_batch(is_symbolic=False,
                                                    batch_size=self._batch_size)
                                     for fmt in self._formats))
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
            # elif self._mode == 'truncate':
            #     self._next_batch_start = 0
            else:
                raise ValueError("Unrecognized iteration mode '%s'. This "
                                 "should've been caught in %s's "
                                 "constructor."
                                 % (self._mode, type(self)))

        subbatches = tuple(get_range(tensor,
                                     fmt,
                                     self._next_batch_start,
                                     self._next_batch_start +
                                     self._batch_size)
                           for tensor, fmt
                           in safe_izip(self._tensors, self._formats))

        # If _mode != 'loop', this could be "if self._next_batch_start == 0"
        if self._next_batch_start < self._batch_size:
            self._epoch += 1

        self._next_batch_start += self._batch_size
        assert_less_equal(self._next_batch_start, num_samples)

        if self._next_batch_start == num_samples:
            self._next_batch_start = 0

        return self.Batch(*subbatches)
