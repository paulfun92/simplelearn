"""
Static dataset.
"""

__author__ = "Matthew Koichi Grimes"
__email__ = "mkg@alum.mit.edu"
__copyright__ = "Copyright 2014"
__license__ = "Apache 2.0"

import collections
import numpy
from simplelearn.asserts import assert_all_equal, assert_integer
from numpy.testing import assert_equal
from nose.tools import (assert_is_instance,
                        assert_less,
                        assert_greater,
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

            if fmt.dtype is None:
                raise ValueError("Expected all formats to specify a dtype.")

            fmt.check(tensor)

        self.names = tuple(names)
        self.formats = tuple(formats)
        self.tensors = tuple(tensors)

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
            iterator_class = SequentialIterator
        elif iterator_type == 'random':
            iterator_class = RandomIterator
        else:
            raise NotImplementedError("'%s' iterator type not supported." %
                                      iterator_type)

        return iterator_class(self, batch_size, **kwargs)


class DatasetIterator(DataIterator):
    '''
    An abstract superclass of iterators for Dataset, a fixed-size DataSource.
    '''

    def __init__(self, dataset, batch_size):
        super(DatasetIterator, self).__init__()

        assert_is_instance(dataset, Dataset)
        assert_greater(batch_size, 0)
        assert_integer(batch_size)

        self.dataset = dataset
        self.batch_size = batch_size

    def _next_batch_indices(self):
        '''
        Returns
        -------
        rval: slice or tuple of ints

          Used to index into the batch axis to select the next batch of
          examples.
        '''
        raise NotImplementedError("{}._next_batch_indices() not yet "
                                  "implemented".format(type(self)))

    def _next(self):

        def get_batch(tensor, fmt, batch_indices):
            """
            Returns a batch containing selected examples.

            Parameters
            ----------
            tensor: numpy.ndarray, or similar
              The tensor to select a batch from.

            fmt: simplelearn.format.DenseFormat
              The tensor's format

            batch_indices: Sequence
              an array of example indices.
            """

            assert 'b' in fmt.axes
            index = tuple(batch_indices if axis == 'b'
                          else slice(None)
                          for axis in fmt.axes)
            return tensor[index]

        batch_indices = self._next_batch_indices()

        result = []
        for tensor, fmt in safe_izip(self.dataset.tensors,
                                     self.dataset.formats):
            batch = get_batch(tensor, fmt, batch_indices)
            batch_index = fmt.axes.index('b')
            assert_equal(batch.shape[batch_index], self.batch_size)
            result.append(batch)

        return tuple(result)

    def make_input_nodes(self):
        NamedTupleOfNodes = collections.namedtuple('NamedNodes',
                                                   self.dataset.names)
        nodes = tuple(InputNode(fmt) for fmt in self.dataset.formats)
        return NamedTupleOfNodes(*nodes)


def _get_num_examples(dataset):
    '''
    Returns the number of examples in a Dataset.
    '''
    assert_is_instance(dataset, Dataset)

    example_counts = tuple(tensor.shape[fmt.axes.index('b')]
                           for tensor, fmt
                           in safe_izip(dataset.tensors, dataset.formats))

    if not all(sc == example_counts[0] for sc in example_counts[1:]):
        raise ValueError("Expected all tensors to have the same number of "
                         "samples, but got {}.".format(example_counts))

    return example_counts[0]


class SequentialIterator(DatasetIterator):
    """
    Iterates through samples in a Dataset in memory order.
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 loop_style='wrap'):
        """
        Parameters
        ----------
        dataset: Dataset

        batch_size: int

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
        super(SequentialIterator, self).__init__(dataset, batch_size)

        loop_style_values = ('truncate', 'wrap', 'divisible')

        if loop_style not in loop_style_values:
            raise ValueError("'loop_style' argument must be one of %s, not "
                             "'%s'." % (str(loop_style_values), loop_style))

        self._loop_style = loop_style

        num_examples = _get_num_examples(dataset)

        if num_examples < batch_size:
            raise ValueError("# of samples %d must be greater than "
                             "batch_size %d." %
                             (num_examples, batch_size))

        if self._loop_style == 'divisible' and \
           numpy.mod(num_examples, batch_size) != 0:
            raise ValueError("# of samples %d is not divisible by "
                             "batch_size %d (remainder = %d)." %
                             (num_examples,
                              batch_size,
                              numpy.mod(num_examples, batch_size)))

        self._next_batch_start = 0

    def _get_num_examples(self):
        '''
        Returns the number of examples yielded by this iter,
        which will be less than the # of examples in self.dataset
        if self._loop_style is 'truncate'.
        '''

        tensor = self.dataset.tensors[0]
        fmt = self.dataset.formats[0]

        num_examples = tensor.shape[fmt.axes.index('b')]

        if self._loop_style == 'truncate':
            num_examples = (num_examples -
                            numpy.mod(num_examples, self.batch_size))

        assert_less(self._next_batch_start, num_examples)

        return num_examples

    def make_input_nodes(self):
        NamedTupleOfNodes = collections.namedtuple('NamedNodes',
                                                   self.dataset.names)
        nodes = tuple(InputNode(fmt) for fmt in self.dataset.formats)

        return NamedTupleOfNodes(*nodes)

    def next_is_new_epoch(self):
        num_examples = self._get_num_examples()
        assert_less(self._next_batch_start, num_examples)

        if self._loop_style == 'wrap':
            return self._next_batch_start < self.batch_size
        else:
            return self._next_batch_start == 0

    def _next_batch_indices(self):

        num_examples = self._get_num_examples()

        if self._next_batch_start + self.batch_size > num_examples:
            assert_not_equal(self._loop_style,
                             'divisible',
                             "Number of samples %d wasn't divisible by "
                             "batch size %d. This should've been caught "
                             "in the %s constructor." %
                             (num_examples,
                              self.batch_size,
                              type(self)))

            assert_not_equal(self._loop_style,
                             'truncated',
                             "Truncated number of samples %d wasn't divisible "
                             "by batch size %d. It must've been coded wrong." %
                             (num_examples, self.batch_size))

            if self._loop_style == 'wrap':
                tail_indices = numpy.arange(self._next_batch_start,
                                            num_examples)
                head_indices = numpy.arange(self.batch_size -
                                            len(tail_indices))
                assert_equal(len(tail_indices) + len(head_indices),
                             self.batch_size)

                self._next_batch_start = len(head_indices)
                return numpy.concatenate((tail_indices, head_indices))
            else:
                raise ValueError("Unrecognized loop_style '%s'. This "
                                 "should've been caught in %s's "
                                 "constructor."
                                 % (self._loop_style, type(self)))
        else:
            next_batch_end = self._next_batch_start + self.batch_size
            result = slice(self._next_batch_start, next_batch_end)

            self._next_batch_start = next_batch_end % num_examples

            return result

    # def _next(self):

    #     def get_batch(tensor, fmt, start, end):
    #         """
    #         Returns a batch from batch index = start to end.
    #         """
    #         assert_less_equal(start, end)
    #         assert 'b' in fmt.axes
    #         index = tuple(slice(start, end) if axis == 'b'
    #                       else slice(None)
    #                       for axis in fmt.axes)
    #         result = tensor[index]
    #         assert_equal(result.shape[fmt.axes.index('b')], end - start)
    #         return result

    #     num_examples = \
    #         self._tensors[0].shape[self._formats[0].axes.index('b')]

    #     if self._loop_style == 'truncate':
    #         num_examples = num_examples - numpy.mod(num_examples,
    #                                                 self._batch_size)

    #     assert_less(self._next_batch_start, num_examples)

    #     if self._next_batch_start + self._batch_size > num_examples:
    #         assert_not_equal(self._loop_style,
    #                          'divisible',
    #                          "Number of samples %d wasn't divisible by "
    #                          "batch size %d. This should've been caught "
    #                          "in the %s constructor." %
    #                          (num_examples,
    #                           self._batch_size,
    #                           type(self)))

    #         assert_not_equal(self._loop_style,
    #                          'truncated',
    #                          "Truncated number of samples %d wasn't divisible "
    #                          "by batch size %d. It must've been coded wrong." %
    #                          (num_examples, self._batch_size))

    #         if self._loop_style == 'wrap':
    #             batch = tuple(fmt.make_batch(is_symbolic=False,
    #                                          batch_size=self._batch_size)
    #                           for fmt in self._formats)

    #             chunk_size = num_examples - self._next_batch_start
    #             assert_greater(chunk_size, 0)

    #             for subbatch, tensor, fmt in safe_izip(batch,
    #                                                    self._tensors,
    #                                                    self._formats):
    #                 get_batch(subbatch,
    #                           fmt,
    #                           0,
    #                           chunk_size)[...] = \
    #                     get_batch(tensor,
    #                               fmt,
    #                               self._next_batch_start,
    #                               num_examples)

    #                 get_batch(subbatch,
    #                           fmt,
    #                           chunk_size,
    #                           self._batch_size)[...] = \
    #                     get_batch(tensor,
    #                               fmt,
    #                               0,
    #                               self._batch_size - chunk_size)

    #             self._next_batch_start = self._batch_size - chunk_size
    #             return batch
    #         else:
    #             raise ValueError("Unrecognized loop_style '%s'. This "
    #                              "should've been caught in %s's "
    #                              "constructor."
    #                              % (self._loop_style, type(self)))

    #     subbatches = tuple(get_batch(tensor,
    #                                  fmt,
    #                                  self._next_batch_start,
    #                                  self._next_batch_start +
    #                                  self._batch_size)
    #                        for tensor, fmt
    #                        in safe_izip(self._tensors, self._formats))

    #     self._next_batch_start += self._batch_size
    #     assert_less_equal(self._next_batch_start, num_examples)

    #     if self._next_batch_start == num_examples:
    #         self._next_batch_start = 0

    #     return subbatches


class RandomIterator(DatasetIterator):
    '''
    Iterates through samples in a Dataset in random order.

    By default, all examples in the dataset have an equal probability of
    being selected for the next batch. To change this, edit self.probabilities.
    Make sure they add up to 1.0, or else numpy will throw an error.
    '''

    def __init__(self, dataset, batch_size, rng):
        super(RandomIterator, self).__init__(dataset, batch_size)
        assert_is_instance(rng, numpy.random.RandomState)
        self._rng = rng

        num_examples = _get_num_examples(dataset)

        self.probabilities = (numpy.ones(num_examples, dtype=float) /
                              num_examples)

        self.batches_per_epoch = (num_examples // batch_size +
                                  0 if num_examples % batch_size == 0 else 1)
        self.num_batches_shown = 0

    def next_is_new_epoch(self):
        return (self.num_batches_shown % self.batches_per_epoch) == 0

    def _next_batch_indices(self):
        self._rng.choice(self.probabilities.shape[0],
                         size=self.batch_size,
                         replace=True,
                         p=self.probabilities)

    def _next(self):
        result = super(RandomIterator, self)._next()

        self.num_batches_shown += 1

        return result
