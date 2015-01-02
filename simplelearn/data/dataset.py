"""
Static dataset.
"""

import collections
import numpy
from numpy.testing import assert_equal
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

    def get_input_nodes(self):
        names = self.data._fields
        nodes = tuple(x.node for x in self.data)
        InputNodes = collections.namedtuple('InputNodes', names)
        return InputNodes(nodes)

    def iterator(self, iterator_type, batch_size, **kwargs):
        if iterator_type == 'sequential':
            formats = tuple(d.node.output_format for d in self.data)
            return SequentialIterator(batch_size,
                                      formats,
                                      tuple(d.tensor for d in self.data),
                                      **kwargs)
        else:
            raise NotImplementedError("'%s' iterator type not supported." %
                                      iterator_type)


class SequentialIterator(DataIterator):
    """
    Iterates through samples in a Dataset in memory order.
    """

    def __init__(self, batch_size, tensors, formats, **kwargs):
        """
        Parameters
        ----------
        batch_size: int

        tensors: sequence
          A sequence of tensors, all with the same number of samples.

        formats: sequence
          The tensors' Formats. These must all have a 'b' (batch) axis.

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
        if not numpy.issubdtype(batch_size, numpy.integer):
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

        self._mode = kwargs.get('mode', None)
        if self._mode is None:
            self._mode = 'loop'

        mode_values = ('truncate', 'loop', 'divisible')

        if self._mode not in mode_values:
            raise ValueError("'mode' argument must be one of %s." %
                             str(mode_values))

        def get_num_samples(tensor, fmt):
            batch_index = fmt.axes.index('b')
            return tensor.shape[batch_index]

        sample_counts = tuple(get_num_samples(t, f)
                              for t, f in safe_izip(tensors, formats))
        assert all(sc == sample_counts[0] for sc in sample_counts[1:])

        num_samples = sample_counts[0]
        if self._mode == 'divisible' and \
           numpy.mod(num_samples, batch_size) != 0:
            raise ValueError("# of samples %d is not divisible by "
                             "batch_size %d (remainder = %d)." %
                             (num_samples,
                              batch_size,
                              numpy.mod(num_samples, batch_size)))

        self._epoch = 0
        self._next_batch_start = 0
        self._batch_size = batch_size
        self._formats = formats
        self._tensors = tensors

    def epoch(self):
        return self._epoch

    def next(self):
        num_samples = \
            self._tensors[0].shape[self._formats[0].axes.index('b')]

        assert self._next_batch_start < num_samples

        num_remaining_samples = num_samples - self._next_batch_start

        def get_range(tensor, fmt, start, end):
            index = tuple(slice(start, end) if axis == 'b'
                          else slice(None)
                          for axis in fmt.axes)
            return tensor[index]

        if num_remaining_samples < self._batch_size:
            if self._mode == 'truncate':
                self._next_batch_start = 0
                # don't return here
            elif self._mode == 'loop':
                result = [f.make_batch(batch_size=self._batch_size,
                                       is_symbolic=False)
                          for f in self._formats]

                for batch, data in safe_izip(result, self._tensors):
                    tensor = data.tensor
                    fmt = data.node.format

                    batch_slice = get_range(batch,
                                            fmt,
                                            0,
                                            num_remaining_samples)
                    data_slice = get_range(tensor,
                                           fmt,
                                           self._next_batch_start,
                                           num_samples)
                    batch_slice[...] = data_slice

                    batch_slice = get_range(batch,
                                            fmt,
                                            num_remaining_samples,
                                            self._batch_size)
                    data_slice = get_range(tensor,
                                           fmt,
                                           0,
                                           self._batch_size -
                                           num_remaining_samples)
                    batch_slice[...] = data_slice

                self._next_batch_start = (self._batch_size -
                                          num_remaining_samples)

                return result
            elif self._mode == 'divisible':
                raise RuntimeError("number of samples not divisible by "
                                   "batch size while iteration mode == "
                                   "'divisible'. This should've been "
                                   "caught in the iterator constructor.")
            else:
                raise RuntimeError("self._mode had unrecognized value "
                                   "'%s'. This should've been caught in "
                                   "the iterator constructor." %
                                   self._mode)

        batch_end = self._next_batch_start + self._batch_size
        result = tuple(get_range(tensor,
                                 fmt,
                                 self._next_batch_start,
                                 batch_end))
        self._next_batch_start = batch_end
