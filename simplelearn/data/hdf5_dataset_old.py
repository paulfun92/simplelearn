'''
A Dataset that stores its data in an HDF5 file.
'''

import os
import h5py
import numpy
from nose.tools import (assert_true,
                        assert_is_instance,
                        assert_in,
                        assert_not_equal)
from simplelearn.data.dataset import Dataset
from simplelearn.formats import DenseFormat

import pdb


# TODO: hdf5 reform:
# Hdf5Datasets should just a collection of tensors, with some partition row
# indices that demarcate where the test, train, or validation sets begin.
# Each tensor will have an 'axes' attr, so that we can recover its DenseFormat.
#
# When serializing, it just stores the file path of the hdf file, relative to
# simplelearn.data.data_dir.
#
# An Hdf5Dataset is just a range of row_indices of an HdfDatasets.
# In its __init__, it stores a self._hdf_datasets and a self._slice, then
# calls super().__init__(subtensors, names, formats), where subtensors
# are sliced from HdfDatasets' tensors.
#
# It overrides Dataset's serialization to only serialize self._hdf_datasets,
# and self._slice

# class Hdf5DatasetWriter(object):
#     def __init__(self, dataset_path, size, description=''):
#         self.partition_bounds = []
#         self.partition_names = []
#         self.tensors = []
#         self.formats = []
#         self.hdf = h5py.File(dataset_path, 'w')
#         self.hdf.attrs['description'] = description
#         self.size = size

#     def add_tensor(self, name, fmt):
#         '''
#         Adds a tensor to an HDF5 group, in a manner compatible with Hdf5Dataset.

#         Parameters
#         ----------
#         name: str
#           The tensor's name in the group.

#         fmt: simplelearn.formats.DenseFormat
#           The tensor's format.

#         hdf_group: h5py.Group
#           The Group to add this tensor to under the name <name>. Note that
#           h5py.Files are Groups.
#         '''
#         shape = list(self.fmt.shape)
#         shape[self.fmt.index('b')] = self.size

#         dataset = self.hdf.create_dataset(name, shape, fmt.dtype)
#         dataset.attrs['axes'] = numpy.asarray(axes, 'S')

#         self.tensors.append(dataset)
#         self.formats.append(fmt)

#         return dataset

#     def add_partition(self, size, name):
#         '''
#         Parameters
#         ----------
#         size: int
#           A positive int, or -1. If the latter, make this the final partition,
#           using all remaining space.

#         name: str
#           The name of this partition (e.g. 'train', 'valid', 'test').
#         '''
#         assert_integer(size)
#         assert_not_equal(size, 0)
#         assert_greater(size, -2)

#         assert_is_instance(name, basestring)
#         name = unicode(name)

#         partition_start = (0 if len(self.partition_bounds) == 0
#                            else self.partition_bounds[-1])

#         if size == -1:
#             size = self.size - partition_start

#         assert_less_equal(partition_end, self.size)

#         self.partition_ends.append(partition_start + size)
#         self.partition_names.append(name)

#     def get_partition(self, partition):
#         '''
#         Parameters
#         ----------
#         partition: str or int
#           Either the partition name (choose from self.partition_names)
#           or the partition's index (as used by self.partition_names).

#         Returns
#         -------
#         rval: Dataset
#         '''
#         if isinstance(partition, basestring):
#             partition = tuple(self.partition_names).index(partition)
#         else:
#             assert_integer(partition)
#             assert_greater_equal(partition, 0)
#             assert_less(partition, len(self.partition_names))

#         partition_tensors = []
#         for fmt, hdf_tensor in safe_izip(self.formats, self.tensors):

def add_tensor(name, shape, dtype, axes, hdf_group):
    '''
    Adds a tensor to an HDF5 group, in a manner compatible with Hdf5Dataset.

    Parameters
    ----------
    name: str
      The tensor's name in the group.

    shape: Sequence
      The tensor's shape.

    dtype: str, numpy.dtype
      The tensor's dtype.

    axes: Sequence
      The axis names.

    hdf_group: h5py.Group
      The Group to add this tensor to under the name <name>. Note that
      h5py.Files are Groups.
    '''
    dataset = hdf_group.create_dataset(name, shape, dtype)
    dataset.attrs['axes'] = numpy.asarray(axes, 'S')
    return dataset


# __file_path_to_hdf = {}

# def load_h5py(file_path):
#     class

# class Hdf5Wrapper(object):

#     __paths_to_handles = {}

#     def __init__(self, file_path):
#         assert_true(os.path.isfile(file_path))

#         file_path = abspath(file_path)

#         if file_path not in self.__paths_to_handles:
#             self.__paths_to_handles[file_path] = h5py.File(file_path, 'r')

#         self.__paths_to_handles[file_path]

#     def __del__

class Hdf5Dataset(Dataset):
    '''
    A Dataset that's saved in an HDF5 file.

    Pickling this dataset just pickles the HDF5 file's path.

    Unpickling this dataset loads the HDF5 file and wraps it with this Dataset.

    Note that Hdf5Dataset loads the HDF5 file in read-only mode. See
    ./mnist.py for an example of how to actually create the HDF5 file
    in the first place.
    '''

    def __init__(self, hdf5_filepath, group_path=u''):
        assert_true(os.path.isfile(hdf5_filepath))
        group_path = unicode(group_path)
        if group_path[0] == '/':
            group_path = group_path[1:]

        assert_not_equal(group_path[0], '/')

        hdf = h5py.File(hdf5_filepath, 'r')
        assert_in(group_path, hdf.keys())
        hdf = hdf[group_path]

        self.hdf5_filepath = hdf5_filepath
        self.group_path = group_path

        # args to superclass constructor
        names = []
        tensors = []
        formats = []

        def make_dense_format(tensor):
            '''
            Constructs a DenseFormat object from an annotaeted h5py.Dataset.

            Parameters
            ----------
            tensor: h5py.Dataset
              Must be annotated with 'axes', as done by this module's
              add_tensor() function.

            Returns
            -------
            rval: DenseFormat
            '''

            axes = tuple(tensor.attrs['axes'])
            shape = tuple(tensor.shape)

            # Sets batch size to -1
            if 'b' in axes:
                b_index = axes.index('b')
                shape = list(shape)
                shape[b_index] = -1
                shape = tuple(shape)

            return DenseFormat(shape=shape, axes=axes, dtype=tensor.dtype)

        for name, tensor in hdf.items():
            assert_is_instance(tensor, h5py.Dataset)

            axes = tensor.attrs['axes']
            assert_is_instance(axes, numpy.ndarray)  #h5py.Dataset)
            axes = tuple(axes)

            fmt = make_dense_format(tensor)

            names.append(name)
            tensors.append(tensor)
            formats.append(fmt)

        super(Hdf5Dataset, self).__init__(names, tensors, formats)

    # def _get_dataset_group(self, hdf5):
    #     '''
    #     Returns the h5py.Group that stores this dataset.

    #     Override this method in subclasses if the dataset is not stored in the
    #     top-level group of the HDF5 file.

    #     Parameters
    #     ----------
    #     hdf5: h5py.File
    #     '''
    #     return hdf5

    def __getinitargs__(self):
        '''
        For unpickling.
        '''
        return (self.hdf5_filepath, self.group_path)

    def __getstate__(self):
        return self.hdf5_filepath

    def __setstate__(self, state):
        '''
        The acutal work of unpickling is done by
        self.__init__(__getinitargs__())
        '''
        assert_is_instance(state, str)
        self.hdf5_filepath = state

    # def _construct_superclass_from_hdf5(self, hdf5_filepath):
    #     assert_true(os.path.isfile(hdf5_filepath))

    #     hdf = h5py.File(hdf5_filepath, 'r')

    #     hdf = self._get_dataset_group(hdf)

    #     # args to superclass constructor
    #     names = []
    #     tensors = []
    #     formats = []

    #     def make_dense_format(tensor):
    #         '''
    #         Constructs a DenseFormat object from an annotaeted h5py.Dataset.

    #         Parameters
    #         ----------
    #         tensor: h5py.Dataset
    #           Must be annotated with 'axes', as done by this module's
    #           add_tensor() function.

    #         Returns
    #         -------
    #         rval: DenseFormat
    #         '''

    #         axes = tuple(tensor['axes'])
    #         shape = tuple(tensor.shape)

    #         # Sets batch size to -1
    #         if 'b' in axes:
    #             b_index = shape.index('b')
    #             shape = list(shape)
    #             shape[b_index] = -1
    #             shape = tuple(shape)

    #         return DenseFormat(shape=shape, axes=axes, dtype=tensor.dtype)

    #     for name, group in hdf.items():
    #         tensor = group['tensor']
    #         assert_is_instance(tensor, h5py.Dataset)

    #         axes = tensor['axes']
    #         assert_is_instance(axes, h5py.Dataset)
    #         axes = tuple(axes)

    #         fmt = make_dense_format(tensor)

    #         names.append(name)
    #         tensors.append(tensor)
    #         formats.append(fmt)

    #     super(Hdf5Dataset, self).__init__(names, tensors, formats)
