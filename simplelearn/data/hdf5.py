'''
Datasets that live on an HDF5 file under simplelearn.data.data_path's subtree.
'''

import os
from collections import Iterable
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
class Hdf5Data(object):
    '''
    Wraps an HDF5 file. The [] operator returns a Hdf5Dataset, which contains
    a slice along the batch axis from each of the tensors in this object.

    On pickling, this just saves the relative path from
    simplelearn.data.data_path to the HDF5 file.
    '''

    def __init__(self, path, mode, size=None):
        '''
        Wraps an HDF5 file with this dataset.

        When creating and editing new Hdf5Dataset (i.e. mode = 'w' or 'w-'),
        it's safe practice to close it after you're done editing,
        then re-open it in read-only mode (mode 'r') for actual use.

        Parameters
        ----------
        path: str
          file path to the HDF5 file.

        mode: str
          'r': read-only
          'w': create new for writing, overwriting any file of the same name.
          'w-: create new for writing, fail if a file of the same name exists.

        size: integer
          Number of examples the dataset will store.
          Only specify if <mode> is 'w' or 'w-'.
        '''
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
    def tensors(self):
        '''
        Returns the tensors in a tuple.
        '''
        return tuple(self.hdf['tensors'].values())

    @property
    def names(self):
        '''
        Returns the tensor names in a tuple.
        '''
        return tuple(self.hdf['tensors'].keys())

    @property
    def formats(self):
        '''
        Returns the tensor formats in a tuple.
        '''
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

        return tuple(get_format(t) for t in self.tensors)

    @property
    def size(self):
        '''
        Number of examples in each of the tensors.

        This is fixed on instantiation, so it's defined even if no tensors have
        been added yet.

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
        Returns a default slice.

        Parameters
        ----------
        name: str
          The name of the default slice.

        Returns
        -------
        rval: slice
        '''

        slice_names = numpy.asarray(self.hdf['slice_names'])
        name = numpy.asarray([name],
                             dtype=self.hdf['slice_names'].dtype)
        slice_index = numpy.nonzero(slice_names == name)[0]

        if len(slice_index) == 0:
            raise ValueError("%s is not a known default slice. Choose "
                             "from %s" % (name, tuple(slice_names)))

        return self.default_slices[slice_index[0]]

    def __getitem__(self, arg):
        '''
        Like Dataset.__getitem__, except:
          * Returns an Hdf5Dataset, which is a picklable view of this HdfData.
          * The arg can be a default slice name.

        Example:

          self['test'] returns the testing set, if 'test' is one of the
          default slice names.

        Note that this is O(N) in the number of default slices, since it does a
        linear search for <name>.

        Returns
        -------
        rval: HdfDatasetSlice
        '''

        return Hdf5Dataset(self, arg)

    def __setitem__(self, key, batches):
        '''
        For writing to Hdf5Data objects.

        Example:
          self['train'] = (training_images, training_labels)

        Parameters
        ----------
        key: str, integer, or slice
          See doc for __getitem__()

        batches: Iterable
          An Iterable of numpy array-like objects that can be assigned to
          slices of self.tensors. Must have the same length as self.tensors.
        '''
        assert_is_instance(batches, Iterable)

        batch_slice = (self.get_default_slice(key)
                       if isinstance(key, basestring)
                       else Dataset._getitem_arg_to_slice(key))

        def get_slice_tuple(fmt, batch_slice):
            '''
            Returns a tuple for slicing a single tensor along its batch axis.
            '''
            return tuple(batch_slice if axis == 'b' else slice(None)
                         for axis in fmt.axes)

        for tensor, fmt, batch in safe_izip(self.tensors,
                                            self.formats,
                                            batches):
            slice_tuple = get_slice_tuple(fmt, batch_slice)
            tensor[slice_tuple] = batch

    def add_default_slice(self, name, size):
        '''
        Add to the list of default slices.

        Most datasets come with designated training and testing sets.
        These are stored as default slices.

        Parameters
        ----------
        name: str
          Name of slice (e.g. 'test', 'valid', 'train')

        size: int
          Size of slice, or -1 to use all remaining data unclaimed by
          previous default slices.
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

    def __getstate__(self):
        relative_path = os.path.relpath(os.path.abspath(self.hdf.filename),
                                        start=simplelearn.data.data_path)
        return relative_path

    def __setstate__(self, relative_path):
        absolute_path = os.path.join(simplelearn.data.data_path,
                                     relative_path)

        self.__init__(absolute_path, 'r')


class Hdf5Dataset(Dataset):
    '''
    A Dataset that's a slice of the rows of a Hdf5Data object.

    On pickling, this just saves the row slice and a reference to the
    HdfDatsets object.

    Writing to this dataset's data will not affect the underlying HDF5 file!
    If you wish to edit the HDF5 data, use Hdf5Data.__setitem__().
    '''

    def __init__(self, hdf5_data, slice_arg):
        '''
        Equivalent to hdf5_data[slice_arg].

        See docs for Hdf5Data.__getitem__() for details.
        '''
        assert_is_instance(hdf5_data, Hdf5Data)

        if isinstance(slice_arg, basestring):
            slice_arg = hdf5_data.get_default_slice(slice_arg)

        full_dataset = Dataset(names=hdf5_data.names,
                               tensors=hdf5_data.tensors,
                               formats=hdf5_data.formats)

        super(Hdf5Dataset, self).__init__(names=full_dataset.names,
                                          tensors=full_dataset[slice_arg],
                                          formats=full_dataset.formats)

        self._init_args = (hdf5_data, slice_arg)

    def __getinitargs__(self):
        '''
        Called on pickling.

        On unpickling, this causes __init__ to be called with the given args.
        '''

        return self._init_args
