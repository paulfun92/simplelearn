'''
A dataset that lives on disk as an HDF5 file.
'''

import os
import collections
import h5py
import numpy
from nose.tools import (assert_true,
                        assert_equal,
                        assert_in,
                        assert_is_instance,
                        assert_greater)
import simplelearn
from simplelearn.utils import (safe_izip,
                               assert_integer,
                               assert_all_is_instance,
                               assert_all_integers,
                               assert_all_greater_equal)
from simplelearn.formats import DenseFormat
from simplelearn.data.dataset import Dataset
from simplelearn.data import DataIterator
from simplelearn.nodes import InputNode

import pdb


def make_h5_file(path,
                 partition_names,
                 partition_sizes,
                 tensor_names,
                 tensor_formats):
    '''
    Creates a h5py.File with groups that can be wrapped by H5Dataset.

    Usage
    -----

    h5_file = make_hf_file(file_path, p_names, p_sizes, t_names, t_formats)
      1: Call this function to create a h5py.File object
      2: Fill the h5py.File's data tensors with appropriate data.
      3: Close the h5py.File, then re-open it using H5Dataset,
         a read-only dataset interface.

    Parameters
    ----------
    partition_names: Sequence
      Names of the sub-datasets, e.g. ['test', 'train'].

    partition_sizes: Sequence
      Number of examples in each sub-dataset, e.g. [50000, 10000] for
      MNIST.

    tensor_names: Sequence
      Names of the data tensors, e.g. ['images', 'labels']. Each
      sub-tensor uses the same tensor_names.

    tensor_formats: Sequence
      The DataFormats of the data tensors, e.g. (for MNIST):
      [DataFormat(axes=['b', '0', '1'], shape=[-1, 28, 28], dtype='uint8'),
       DataFormat(axes=['b'], shape=[-1], dtype='uint8')]

    The example parameter values above would create an h5py.File
    with the following hierarchical structure:

    hfpy.File/
      'partition_names': an h5py.Dataset of strings, ['test', 'train']
      'tensor_names': an h5py.Dataset of strings, ['images', 'labels']
      'partitions': an h5py.Group with the following members:
        'train': an h5py.Group, with the following members:
          'images': an h5py.Dataset tensor, with shape given by
                    partition_sizes[0] and tensor_formats[0].
          'labels': an h5py.Dataset tensor, with shape given by
                    partition_sizes[0] and tensor_formats[1].
        'test': an h5py.Group, with the following members:
          'images': an h5py.Dataset tensor, with shape given by
                    partition_sizes[1] and tensor_formats[0].
          'labels': an h5py.Dataset tensor, with shape given by
                    partition_sizes[1] and tensor_formats[1].
    '''

    assert_is_instance(path, basestring)
    assert_equal(os.path.splitext(path)[1], '.h5')
    absolute_path = os.path.abspath(path)
    assert_true(absolute_path.startswith(simplelearn.data.data_path))

    assert_all_is_instance(partition_names, basestring)
    assert_equal(len(frozenset(partition_names)), len(partition_names))

    assert_all_integers(partition_sizes)
    assert_all_greater_equal(partition_sizes, 0)

    assert_all_is_instance(tensor_names, basestring)
    assert_equal(len(frozenset(tensor_names)), len(tensor_names))

    assert_all_is_instance(tensor_formats, DenseFormat)
    for tensor_format in tensor_formats:
        assert_in('b', tensor_format.axes)

    # Done sanity-checking args

    h5_file = h5py.File(absolute_path, mode='w')

    # Add ordered lists of tensor/partition names, since h5py.Group.keys()
    # can't be trusted to list group members in the order that they were
    # added in.

    def add_ordered_names(list_name, names, group):
        '''
        Adds a list of names to a group, as a h5py.Dataset of strings.
        '''
        max_name_length = max([len(n) for n in tensor_names])
        string_dtype = 'S{}'.format(max_name_length)
        result = group.create_dataset(list_name,
                                      (len(names), ),
                                      dtype=string_dtype)
        for n, name in enumerate(names):
            result[n] = name

    # Not sure if storing partition order is necessary, but why not.
    add_ordered_names('partition_names', partition_names, h5_file)

    # Storing tensor order is definitely necessary.
    add_ordered_names('tensor_names', tensor_names, h5_file)

    partitions = h5_file.create_group('partitions')

    for partition_name, partition_size in safe_izip(partition_names,
                                                    partition_sizes):
        partition = partitions.create_group(partition_name)

        for tensor_name, tensor_format in safe_izip(tensor_names,
                                                    tensor_formats):
            tensor_shape = list(tensor_format.shape)
            tensor_shape[tensor_format.axes.index('b')] = partition_size

            # fletcher32: checksum against data corruption with tiny overhead.
            # http://docs.h5py.org/en/latest/high/dataset.html#fletcher32-filter
            tensor = partition.create_dataset(tensor_name,
                                              tensor_shape,
                                              tensor_format.dtype,
                                              fletcher32=True)

            # Label the tensor axes by their axis names in fmt.
            for index, axis in enumerate(tensor_format.axes):
                tensor.dims[index].label = axis

    return h5_file


class H5Dataset(Dataset):
    '''
    A read-only Dataset that wraps a h5py.Group, in a h5py.File created by
    make_h5_file.

    When pickled, this saves the HDF5 file's path, and the group name of
    the partition within it.
    '''

    def __init__(self, path, partition_name):
        '''
        Opens (as read-only) a group in a HDF5 file created with open_h5_file.

        Parameters
        ----------
        path: str
          Path to the .h5 file
        '''

        assert_is_instance(path, basestring)
        assert_is_instance(partition_name, basestring)

        path = os.path.abspath(path)

        def is_subdir(sub_dir, root_dir):
            '''
            Return True iff sub_dir is a subdirectory of root_dir.
            '''

            sub_dir = os.path.abspath(sub_dir)
            root_dir = os.path.abspath(root_dir)
            common_prefix = os.path.commonprefix([root_dir, sub_dir])
            return common_prefix == root_dir

        assert_true(is_subdir(path, simplelearn.data.data_path))

        self.h5_file = h5py.File(path, mode='r')

        partitions = self.h5_file['partitions']
        tensor_names = self.h5_file['tensor_names']

        partition = partitions[partition_name]

        self._partition_name = partition_name

        tensors = [partition[n] for n in tensor_names]

        def get_format(tensor):
            '''
            Returns the DenseFormat of an h5py.Dataset with labeled dims.
            '''

            axes = tuple(dim.label for dim in tensor.dims)
            shape = list(tensor.shape)
            shape[axes.index('b')] = -1

            return DenseFormat(axes=axes,
                               shape=shape,
                               dtype=tensor.dtype)

        formats = [get_format(t) for t in tensors]

        super(H5Dataset, self).__init__(tensor_names, tensors, formats)

    def __getstate__(self):
        '''
        Saves just the file path (relative to data_dir) and the partition name.
        '''
        absolute_path = os.path.abspath(self.h5_file.filename)
        relative_path = os.path.relpath(absolute_path,
                                        simplelearn.data.data_path)
        return (relative_path, self._partition_name)

    def __setstate__(self, state):
        self.__init__(os.path.join(simplelearn.data.data_path, state[0]),
                      state[1])

class RandomIterator(DataIterator):
    '''
    Iterates through samples in an H5Dataset in random order.

    Within each batch, the examples will be sorted in memory order. This is
    because h5py requires that when indexing into h5py.Dataset with an array of
    indices, the indices must be sorted. This should not affect the outcome of
    SGD, but if it is problematic for any other reason, you can shuffle the
    batch contents before using them.

    Initially, all examples in the h5_dataset have an equal probability of
    being selected for the next batch. To change this, edit self.probabilities.
    Make sure they add up to 1.0, or else numpy will throw an error.
    '''

    def __init__(self, h5_dataset, batch_size, rng):
        assert_is_instance(h5_dataset, Dataset)  # not necessarily an H5Dataset
        assert_greater(len(h5_dataset.tensors), 0)

        assert_integer(batch_size)
        assert_greater(batch_size, 0)

        assert_is_instance(rng, numpy.random.RandomState)

        super(RandomIterator, self).__init__()

        def get_num_examples():
            tensor = h5_dataset.tensors[0]
            fmt = h5_dataset.formats[0]
            return tensor.shape[fmt.axes.index('b')]

        self._h5_dataset = h5_dataset
        self._batch_size = batch_size
        self._rng = rng

        num_examples = get_num_examples()

        self.probabilities = numpy.empty(get_num_examples(), dtype=float)
        self.probabilities[:] = 1.0 / float(self.probabilities.shape[0])
        self.batches_per_epoch = (num_examples // batch_size +
                                  0 if num_examples % batch_size == 0 else 1)
        self.num_batches_shown = 0

    def make_input_nodes(self):
        NamedTupleOfNodes = collections.namedtuple('NamedNodes',
                                                   self._h5_dataset.names)
        nodes = tuple(InputNode(fmt) for fmt in self._h5_dataset.formats)
        return NamedTupleOfNodes(*nodes)  # pylint: disable=star-args

    def next_is_new_epoch(self):
        return ((self.num_batches_shown % self.batches_per_epoch) == 0)

    def _next(self):

        def get_batch(tensor, fmt, example_indices):
            """
            Returns a batch containing selected examples.

            Parameters
            ----------
            tensor: numpy.ndarray, or similar
              The tensor to select a batch from.

            fmt: simplelearn.format.DenseFormat
              The tensor's format

            example_indices: Sequence
              an array of example indices.
            """

            assert 'b' in fmt.axes
            index = tuple(example_indices if axis == 'b'
                          else slice(None)
                          for axis in fmt.axes)
            result = tensor[index]
            assert_equal(result.shape[fmt.axes.index('b')],
                         len(example_indices))
            return result

        example_indices = self._rng.choice(self.probabilities.shape[0],
                                           size=self._batch_size,
                                           replace=True,
                                           p=self.probabilities)

        # h5py.Datasets requires that index arrays be sorted. This should not
        # affect the outcome of SGD, which is invariant to the order of
        # examples within a batch.
        example_indices.sort()  # in-place quicksort

        result = tuple(get_batch(tensor, fmt, example_indices)
                       for tensor, fmt
                       in safe_izip(self._h5_dataset.tensors,
                                    self._h5_dataset.formats))

        self.num_batches_shown += 1

        return result

def load_h5_dataset(h5_path, partition=None, mode='r'):
    '''
    Returns all the H5Datasets contained in a file created with make_h5_file().

    Parameters
    ----------

    h5_path: str
      Path to .h5 file created with make_h5_file().

    partition: str or None
      Optional. If supplied, this returns just that partition (e.g. 'test',
      'train'). If omitted, this returns all partitions.

    mode: the file mode to give to h5py.File().

    Returns
    -------

    rval: tuple, or H5Dataset
      If <partition> is omitted, this returns a tuple of H5Datasets.
      Otherwise, this returns just the H5Dataset of the specified partition.
    '''

    h5_file = h5py.File(h5_path, mode=mode)
    partition_names = h5_file['partition_names']
    if partition is None:
        with h5py.File(h5_path, mode='r') as h5_file:
            partition_names = list(h5_file['partition_names'])

        return tuple(H5Dataset(h5_path, n) for n in partition_names)
    else:
        return H5Dataset(h5_path, partition)
