'''
A Dataset that stores its data in an HDF5 file.
'''

import os
import h5py
import numpy
from nose.tools import assert_true, assert_is_instance, assert_in
from simplelearn.data.dataset import Dataset
from simplelearn.formats import DenseFormat

import pdb

def add_tensor(name, shape, dtype, axes, hdf):
    '''
    Adds a tensor to an HDF5 file, in a manner compatible with Hdf5Dataset.
    '''
    dataset = hdf.create_dataset(name, shape, dtype)
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
    '''

    def __init__(self, hdf5_filepath, group_path=u'/'):
        assert_true(os.path.isfile(hdf5_filepath))
        assert_is_instance(group_path, unicode)

        hdf = h5py.File(hdf5_filepath, 'r')
        assert_in(group_path[1:], hdf.keys())
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
