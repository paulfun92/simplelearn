'''
Datasets that live on an HDF5 file under simplelearn.data.data_path's subtree.
'''

import os
import numpy
import h5py
from nose.tools import (assert_false,
                        assert_is,
                        assert_in,
                        assert_not_in,
                        assert_is_instance,
                        assert_equal,
                        assert_not_equal,
                        assert_less_equal,
                        assert_greater,
                        assert_greater_equal)
import simplelearn
from simplelearn.data.dataset import Dataset
from simplelearn.utils import safe_izip, assert_integer
from simplelearn.formats import DenseFormat

import pdb


# pylint: disable=too-few-public-methods
class Hdf5Datasets(object):
    '''
    Wraps an HDF5 file. Slice this with Hdf5Dataset to get a Dataset.

    On pickling, this just saves the relative path from
    simplelearn.data.data_path to the HDF5 file.
    '''

    def __init__(self, path, mode, size=None):
        assert_is_instance(path, basestring)
        assert_is_instance(mode, basestring)
        assert_in(mode,
                  ('r', 'w', 'w-'),
                  "This class only supports 'r', 'w', and 'w-', not %s."
                  % mode)

        self.hdf = h5py.File(path, mode=mode)

        assert_equal(os.path.abspath(self.hdf.filename),
                     os.path.abspath(path))

        if mode == 'r':
            assert_is(size, None)
        else:
            assert_integer(size)
            assert_greater(size, 0)
            if mode == 'r':
                raise ValueError("Can't specify a size when using read-only "
                                 "mode '%s'" % mode)

            self.hdf.attrs['size'] = size

            assert_not_in('slice_names', self.hdf)
            assert_not_in('slice_ends', self.hdf)

            self.hdf.create_dataset('slice_names',
                                    (0, ),
                                    maxshape=(None, ),
                                    dtype='S100')

            self.hdf.create_dataset('slice_ends',
                                    (0, ),
                                    maxshape=(None, ),
                                    dtype=int)

            self.hdf.create_group('tensors')

    @property
    def size(self):
        '''
        Number of 'rows' in each of the tensors.

        Returns
        -------
        rval: int
        '''
        return int(self.hdf.attrs['size'])

    @property
    def default_slice_names(self):
        '''
        Names of the default row slices (e.g. 'train', 'test').

        Returns
        -------
        rval: tuple
          A tuple of unicode strings.
        '''
        return tuple(unicode(s) for s in self.hdf['slice_names'])

    @property
    def default_slices(self):
        '''
        The default row slices corresponding to self.default_slice_names.

        Returns
        -------
        rval: tuple
          A tuple of slice objects.
        '''
        ends = list(self.hdf['slice_ends'])
        if len(ends) == 0:
            return ()

        starts = [0] + ends[:-1]
        return tuple(slice(s, e) for s, e in safe_izip(starts, ends))

    def get_default_slice(self, name):
        '''
        Get one of the default slices by name.

        O(N) in the number of default slices, since it does a linear search for
        <name>.

        Returns
        -------
        rval: slice
        '''

        slice_names = numpy.asarray(self.hdf['slice_names'])
        name = numpy.asarray([name], dtype=self.hdf['slice_names'].dtype)
        slice_index = numpy.nonzero(slice_names == name)[0]
        if len(slice_index) == 0:
            raise ValueError("%s is not a known default slice. Choose from %s"
                             % tuple(slice_names))

        pdb.set_trace()
        return self.default_slices[slice_index[0]]

    def add_default_slice(self, name, size):
        '''
        Add to the list of default slices.

        Most dataset come with designated training and testing sets.
        These are stored as default slices.

        Parameters
        ----------
        name: str
          Name of slice (e.g. 'test', 'valid', 'train')

        size: int
          Size of slice, or -1 to use all remaining data unclaimed by
          previous default slices.

        Returns
        -------
        rval: Hdf5DatasetsSlice
          A Dataset view into the slice that was just added.
          Don't use this if you plan on adding more tensors.
        '''
        assert_is_instance(name, basestring)
        # if we need longer names, change the 'S100' dtype above.
        assert_less_equal(len(name), 100)
        assert_not_in(name, self.hdf['slice_names'])

        assert_integer(size)
        assert_greater(size, -2)

        slice_names = self.hdf['slice_names']
        slice_ends = self.hdf['slice_ends']

        previous_end = 0 if slice_ends.shape[0] == 0 else slice_ends[-1]

        if size == -1:
            size = self.size - previous_end
        else:
            assert_integer(size)
            assert_greater_equal(size, 0)

        new_end = previous_end + size
        assert_less_equal(new_end, self.size)

        num_slices = slice_names.shape[0] + 1
        assert_equal(slice_ends.shape[0] + 1, num_slices)

        slice_names.resize([num_slices])
        slice_names[-1] = name

        slice_ends.resize([num_slices])
        slice_ends[-1] = new_end

        return Hdf5DatasetsSlice(self, self.default_slices[-1])

    def add_tensor(self, name, fmt, dtype=None):
        '''
        Parameters
        ----------
        name: str
          Tensor name (e.g. 'images', 'labels', etc).

        fmt: simplelearn.formats.DenseFormat
          Tensor format. Must contain a 'b' axis.

        dtype: numpy.dtype
          Required if fmt.dtype is None.
        '''
        assert_is_instance(name, basestring)
        name = unicode(name)
        assert_false(name in self.hdf)

        assert_is_instance(fmt, DenseFormat)
        assert_not_equal(fmt.dtype is None, dtype is None)
        dtype = fmt.dtype if fmt.dtype is not None else numpy.dtype(dtype)

        shape = list(fmt.shape)
        shape[fmt.axes.index('b')] = self.hdf.attrs['size']

        group = self.hdf['tensors']

        # fletcher32: checksum against data corruption with almost no overhead.
        # http://docs.h5py.org/en/latest/high/dataset.html#fletcher32-filter
        tensor = group.create_dataset(name, shape, dtype, fletcher32=True)

        # Label the tensor axes by their axis names in fmt.
        for index, axis in enumerate(fmt.axes):
            tensor.dims[index].label = axis

        return tensor

    def __getinitargs__(self):
        path = os.path.relpath(os.path.abspath(self.hdf.filename),
                               start=simplelearn.data.data_path)

        return {'path': path, 'mode': 'r'}

    # pylint: disable=no-self-use
    def __getstate__(self):
        return False  # prevents __setstate__ from being called


class Hdf5DatasetsSlice(Dataset):
    '''
    A Dataset that's a slice of the rows of an Hdf5Datasets object.

    On pickling, this just saves the row slice and a reference to the
    HdfDatsets object.
    '''

    def __init__(self, hdf5_datasets, row_slice):
        assert_is_instance(hdf5_datasets, Hdf5Datasets)

        if isinstance(row_slice, basestring):
            row_slice = hdf5_datasets.get_default_slice(row_slice)
        else:
            assert_is_instance(row_slice, slice)

        def get_format(tensor):
            '''
            Returns the tensor's format as a new DenseFormat object.
            '''
            pdb.set_trace()
            axes = tuple(dim.label for dim in tensor.dims)
            shape = list(tensor.shape)
            shape[axes.index('b')] = -1

            return DenseFormat(axes=axes,
                               shape=shape,
                               dtype=tensor.dtype)

        def slice_tensor(tensor, fmt, row_slice):
            '''
            Slices the tensor along the 'b' (batch) axis.
            '''

            assert_is_instance(row_slice, slice)

            b_index = fmt.axes.index('b')
            if len(fmt.axes) == 1:
                return tensor[row_slice]
            else:
                slices = tuple(row_slice if i == b_index else slice(None)
                               for i in range(len(fmt.axes)))
                return tensor[slices]

        tensor_group = hdf5_datasets['tensors']

        names = tensor_group.keys()
        hdf5_tensors = tensor_group.values()
        formats = [get_format(t) for t in hdf5_tensors]
        tensors = [slice_tensor(t, f, row_slice)
                   for t, f in safe_izip(hdf5_tensors, formats)]

        self._init_args = {'row_slice': row_slice,
                           'hdf5_datasets': hdf5_datasets}
        super(Hdf5DatasetsSlice, self).__init__(names=names,
                                                tensors=tensors,
                                                formats=formats)

    def __getinitargs__(self):
        '''
        Called on pickling.

        On unpickling, this causes __init__ to be called with the given args.
        '''

        return self._init_args

    # pylint: disable=no-self-use
    def __getstate__(self):
        return False  # prevents __setstate__ from being called on unpickling
