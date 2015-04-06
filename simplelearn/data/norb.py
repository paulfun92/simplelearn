'''
Function for loading NORB and Small NORB datasets.
'''

from __future__ import print_function
import os
import h5py
import gzip
import numpy
from nose.tools import assert_equal, assert_in
import simplelearn
from simplelearn.utils import safe_izip, download_url

from simplelearn.nodes import Node
from simplelearn.data.hdf5_dataset import Hdf5Dataset, add_tensor

import pdb


def _read_norb_file(filepath):
    '''
    Reads a NORB file and returns a numpy.ndarray with its contents.

    Parameters
    ----------
    A raw NORB file, of '.mat' or '.mat.gz' format. Each file contains
    a dense tensor with a particular shape and dtype.

    Returns
    -------
    rval: numpy.ndarray
      The tensor contained in the NORB file.
    '''
    if not (filepath.endswith('.mat') or filepath.endswith('.mat.gz')):
        raise ValueError("Expected filename extension '.mat' or '.mat.gz', "
                         "but got %s." % filepath)

    def read_numbers(file_handle, dtype, count):
        '''
        Reads some numbers from a binary file and returns them as a
        numpy.ndarray.

        Parameters
        ----------

        file_handle : file handle
          The file handle from which to read the numbers.

        num_type : str, numpy.dtype
          The dtype of the numbers.

        count : int
          Reads off this many numbers.

        Returns
        -------
        rval: numpy.ndarray
          An array with dtype=dtype and shape=(count, )
        '''
        dtype = numpy.dtype(dtype)
        num_bytes = count * dtype.itemsize
        string = file_handle.read(num_bytes)
        return numpy.fromstring(string, dtype=dtype)

    def read_header(file_handle):
        '''
        Reads the header of a NORB file, describing the tensor dtype and shape.

        Parameters
        ----------
        file_handle: file, gzip.GzipFile, bz2.BZ2File
          A handle to a binary NORB file (*-cat-*, *-dat-*, or *-info-*).

        Returns
        -------
        rval: tuple
          (dtype, shape), where dtype is the numpy.dtype of the tensor,
          and shape is a numpy.ndarray of int32s describing its shape.
        '''

        # Maps dtype keys to dtypes and their NORB isizes in bytes.
        # This is missing one mapping: 0x1E3D4C52 -> 'packed matrix'.
        # While this is mentioned in the NORB spec, it's never actually
        # used by the NORB datasets, so it's not implemented here.
        dtype_dict = {0x1E3D4C51: (numpy.dtype('float32'), 4),
                      0x1E3D4C53: (numpy.dtype('float64'), 8),
                      0x1E3D4C54: (numpy.dtype('int32'), 4),
                      0x1E3D4C55: (numpy.dtype('uint8'), 1),
                      0x1E3D4C56: (numpy.dtype('int16'), 2)}

        dtype_key = read_numbers(file_handle, 'int32', 1)[0]
        dtype, dsize = dtype_dict[dtype_key]
        assert_equal(dsize, dtype.itemsize)

        ndim = read_numbers(file_handle, 'int32', 1)[0]
        # expected ndim values for cat, info, and dat files, respectively
        assert_in(ndim, (1, 2, 4))
        shape = read_numbers(file_handle, 'int32', ndim)

        return dtype, shape

    file_handle = (open(filepath, 'rb')
                   if filepath.endswith('.mat')
                   else gzip.open(filepath))

    dtype, shape = read_header(file_handle)
    size = numpy.prod(shape)
    return read_numbers(file_handle, dtype, size).reshape(shape)


def _make_hdf(which_norb):
    '''
    Caches NORB data to an HDF5 file.

    Reads the original NORB files from a standard designated
    directory, downloading them there first if necessary.

    Then, reads the NORB files, caching their contents to an HDF5 file.

    Parameters
    ----------
    which_norb: 'big' or 'small'

    Returns
    -------
    rval: None
    '''

    assert_in(which_norb, ('big', 'small'))

    norb_directory = os.path.join(simplelearn.data.data_path,
                                  '%s_norb' % which_norb)
    if not os.path.isdir(norb_directory):
        os.mkdir(norb_directory)

    norb_originals_directory = os.path.join(norb_directory, 'original_files')

    if not os.path.isdir(norb_originals_directory):
        os.mkdir(norb_originals_directory)

    def get_norb_filenames(which_norb, which_set):
        '''
        Returns the norb filenames. Downloads them if they're absent.

        If the download fails, this throws an IOError.

        This checks for both expected file suffixes (.mat, .mat.gz).

        Returns
        -------
        rval: tuple
          (dat_files, cat_files, info_files), where each is a
          list of filenames.
        '''
        def get_norb_filenames_gz(which_norb, which_set):
            '''
            Returns the list of files that contain the given NORB dataset.

            The filenames use the 'mat.gz' suffix. Does not check whether the
            files exist.

            Parameters
            ----------
            which_norb: 'big' or 'small'
            which_set: 'test' or 'train'

            Returns
            -------
            rval: tuple
              (dat_files, cat_files, info_files), where each is a
              list of filenames.
            '''
            if which_norb == 'small':
                prefix = 'smallnorb'
                dim = 96
                file_num_strs = ['']
            else:
                prefix = 'norb'
                dim = 108
                file_nums = range(1, 3 if which_set == 'test' else 11)
                file_num_strs = ['-%02d' % n for n in file_nums]

            instances = '01235' if which_set == 'test' else '46789'
            suffix = 'mat.gz'

            cat_files = []
            dat_files = []
            info_files = []

            for (file_type,
                 file_list) in safe_izip(('cat', 'dat', 'info'),
                                         (cat_files, dat_files, info_files)):
                for file_num_str in file_num_strs:
                    filename = ('{prefix}-5x{instances}x9x18x6x2x{dim}x{dim}'
                                '-{which_set}ing{file_num_str}-{file_type}.'
                                '{suffix}').format(prefix=prefix,
                                                   instances=instances,
                                                   dim=dim,
                                                   which_set=which_set,
                                                   file_num_str=file_num_str,
                                                   file_type=file_type,
                                                   suffix=suffix)
                    filepath = os.path.join(norb_originals_directory,
                                            filename)
                    file_list.append(filepath)

            return cat_files, dat_files, info_files

        mat_gz_files = get_norb_filenames_gz(which_norb, which_set)

        def all_files_exist(dat_files, cat_files, info_files):
            '''
            Returns True if all the given NORB file paths exist on disk.
            '''
            for files in (dat_files, cat_files, info_files):
                if not all(os.path.isfile(f) for f in files):
                    return False

            return True

        if all_files_exist(*mat_gz_files):
            return mat_gz_files
        else:  # .mat.gz files weren't there; check for .mat files.
            # chop off the .gz suffix
            mat_files = ([f[:-3] for f in files] for files in mat_gz_files)

            if all_files_exist(*mat_files):
                return mat_files
            else:
                print("Couldn't find all the raw %s NORB %ing set files in "
                      "%s. Downloading..." % (which_norb,
                                              which_set,
                                              norb_originals_directory))

                # If the .mat files don't exist either, download .mat.gz files
                for file_list in mat_gz_files:
                    for filepath in file_list:
                        url = ('http://www.cs.nyu.edu/~ylclab/data/norb-v1.0/'
                               + os.path.split(filepath)[1])
                        try:
                            download_url(url,
                                         local_filepath=filepath,
                                         show_progress=True)
                        except IOError, io_error:
                            raise IOError("IOError: %s\n"
                                          "Error attempting to download %s to "
                                          "%s. Please download the NORB files "
                                          "manually, and put them in %s." %
                                          (io_error,
                                           url,
                                           filepath,
                                           os.path.split(filepath)[0]))

                return mat_gz_files

    def add_dataset(which_norb, which_set, hdf_file):
        '''
        Copies a NORB dataset into a group in a HDF5 file.

        Adds an h5py.Group named <which_set> to hdf_file,
        and gives it 'images' and 'labels' tensors.

        Parameters
        ----------
        which_norb: 'big' or 'small'
        which_set: 'train' or 'test'
        hdf_file: h5py.File

        Returns
        -------
        rval: None
        '''
        assert_in(which_norb, ('big', 'small'))
        assert_in(which_set, ('train', 'test'))

        group = hdf_file.create_group(which_set)

        examples_per_file = 24300 if which_norb == 'small' else 29160

        if which_norb == 'small':
            num_examples = examples_per_file
        else:
            num_examples = examples_per_file * (10 if which_set == 'train'
                                                else 2)

        image_dim = 96 if which_norb == 'small' else 108

        images = add_tensor('images',
                            (num_examples, 2, image_dim, image_dim),
                            'uint8',
                            ('b', 's', '0', '1'),
                            group)

        label_size = 5 if which_norb == 'small' else 11

        labels = add_tensor('labels',
                            (num_examples, label_size),
                            'int32',
                            ('b', 'f'),
                            group)

        cat_files, dat_files, info_files = get_norb_filenames(which_norb,
                                                              which_set)

        index = 0
        for cat_file, dat_file, info_file in safe_izip(cat_files,
                                                       dat_files,
                                                       info_files):
            index_range = slice(index, index + examples_per_file)

            print("Caching %s" % dat_file)
            dat = _read_norb_file(dat_file)
            assert_equal(dat.dtype, images.dtype)
            images[index_range, :, :, :] = dat

            print("Caching %s" % cat_file)
            cat = _read_norb_file(cat_file)
            assert_equal(cat.dtype, labels.dtype)
            labels[index_range, 0] = cat

            print("Caching %s" % info_file)
            info = _read_norb_file(info_file)
            assert_equal(info.dtype, labels.dtype)
            labels[index_range, 1:] = info

            index += examples_per_file

        assert_equal(index, num_examples)

    hdf_path = os.path.join(norb_directory, 'cache.h5')
    print("Caching %s NORB dataset to an HDF5 cache file. "
          "This will only happen once.")

    with h5py.File(hdf_path, 'w') as hdf_file:  # w for overwrite, w- for no
        add_dataset(which_norb, 'train', hdf_file)
        add_dataset(which_norb, 'test', hdf_file)


def load_norb(which_norb, which_set):
    '''
    Loads a NORB dataset.

    Parameters
    ----------
    which_norb: str
      'big' or 'small'

    which_set: str
      'train' or 'test'
    '''
    assert_in(which_norb, ('big', 'small'))
    assert_in(which_set, ('train', 'test'))

    norb_directory = os.path.join(simplelearn.data.data_path,
                                  '%s_norb' % which_norb)
    if not os.path.isdir(norb_directory):
        os.mkdir(norb_directory)
        # need_to_download = True

    cache_file_path = os.path.join(norb_directory, 'cache.h5')
    if not os.path.isfile(cache_file_path):
        _make_hdf(which_norb)

    return Hdf5Dataset(cache_file_path, which_set)
