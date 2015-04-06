'''
A dataset that lives on disk as an HDF5 file.
'''

import os
import copy
import h5py
from nose.tools import (assert_true,
                        assert_equal,
                        assert_not_equal,
                        assert_in,
                        assert_is_instance)
import simplelearn
from simplelearn.utils import (safe_izip,
                               assert_all_is_instance,
                               assert_all_integers,
                               assert_all_greater_equal)
from simplelearn.formats import DenseFormat
from simplelearn.data.dataset import Dataset

import pdb

def make_h5_file(path,
                 partition_names,
                 partition_sizes,
                 tensor_names,
                 tensor_formats,
                 dtypes=None):
    '''
    Creates a h5py.File with groups that can be wrapped by H5Dataset.

    For example, an h5py.File of labeled images with a test and train set would
    have the following internal structure:

    file: an h5py.File, containing the following named members:
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

    if dtypes is not None:
        tensor_formats = [copy.deepcopy(f) for f in tensor_formats]
        for fmt, dtype in safe_izip(tensor_formats, dtypes):
            assert_not_equal(fmt.dtype is None, dtype is None)
            if fmt.dtype is None:
                fmt.dtype = dtype

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
    A Dataset that wraps a h5py.Group, in a h5py.File created by open_h5_file.

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
