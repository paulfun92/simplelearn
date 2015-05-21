'''
Function for loading NORB and Small NORB datasets.
'''

from __future__ import print_function
import os
import gzip
import numpy
from nose.tools import assert_equal, assert_in
import simplelearn
from simplelearn.formats import DenseFormat
from simplelearn.utils import safe_izip, download_url
from simplelearn.data.h5_dataset import (make_h5_file,
                                         load_h5_dataset,
                                         H5Dataset)

import pdb


def _get_norb_dir(which_norb):
    assert_in(which_norb, ('small', 'big'))
    return os.path.join(simplelearn.data.data_path,
                        '{}_norb'.format(which_norb))


def _get_partition_sizes(which_norb):
    assert_in(which_norb, ('small', 'big'))

    if which_norb == 'small':
        examples_per_file = 24300
        num_train_files = 1
        num_test_files = 1
    else:
        examples_per_file = 29160
        num_train_files = 10
        num_test_files = 2

    return (num_train_files * examples_per_file,
            num_test_files * examples_per_file)


def _get_tensor_formats(which_norb):
    assert_in(which_norb, ('small', 'big'))

    if which_norb == 'small':
        label_size = 5
        image_dim = 96
    else:
        label_size = 11
        image_dim = 108

    return (DenseFormat(axes=('b', 's', '0', '1'),
                        shape=(-1, 2, image_dim, image_dim),
                        dtype='uint8'),
            DenseFormat(axes=('b', 'f'),
                        shape=(-1, label_size),
                        dtype='int32'))


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
                         "but got {}.".format(filepath))

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
        # (In fact, the NORB datasets don't use anything other than uint8 and
        # uint32).
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

        # shape field consists of at least 3 int32's, even when ndim < 3.
        shape = read_numbers(file_handle, 'int32', max(ndim, 3))[:ndim]

        return dtype, shape

    file_handle = (open(filepath, 'rb') if filepath.endswith('.mat')
                   else gzip.open(filepath))

    dtype, shape = read_header(file_handle)
    size = numpy.prod(shape)
    result = read_numbers(file_handle, dtype, size).reshape(shape)
    return result


def _get_official_norb_cache_path(which_norb):
    '''
    Returns the path to the .h5 file that stores an official NORB dataset.

    The first time this is called, this will create a directory D:

      D == <simplelearn.data.data_path>/[big_norb, small_norb]

    In this, it will create a cache file called 'cache.h5', and copy data into
    it from the original NORB files, which are expected to be in

      D/original_files

    If they're not, they will be downloaded to there from the web.

    Parameters
    ----------
    which_norb: 'big' or 'small'

    Returns
    -------
    rval: tuple
      Two H5Datasets (train, test).
    '''

    assert_in(which_norb, ('big', 'small'))

    norb_dir = _get_norb_dir(which_norb)
    if not os.path.isdir(norb_dir):
        os.mkdir(norb_dir)

    originals_dir = os.path.join(norb_dir, 'original_files')

    if not os.path.isdir(originals_dir):
        os.mkdir(originals_dir)

    def get_norb_path_lists(which_norb, which_set):
        '''
        Returns paths to orignal NORB files. Downloads the files if needed.

        If the download fails, this throws an IOError.

        This checks for both possible file suffixes (.mat and .mat.gz).

        Returns
        -------
        rval: tuple
          (dat_files, cat_files, info_files), where each is a
          list of filenames.
        '''
        def get_norb_filename_lists_gz(which_norb, which_set):
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
                file_num_strs = ['-{:02d}'.format(n) for n in file_nums]

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
                    file_list.append(filename)

            return cat_files, dat_files, info_files

        gz_filename_lists = get_norb_filename_lists_gz(which_norb, which_set)
        assert_equal(len(gz_filename_lists), 3)

        actual_path_lists = ([], [], [])

        def get_actual_path(gz_filename):
            '''
            Returns the path to an original NORB file.

            Given a 'foo.gz' filename, this checks if either foo or foo.gz
            exist on disk. If so, this returns the path to that file. If not,
            this downloads the file, then returns the path to the downloaded
            file.
            '''
            assert_equal(os.path.split(gz_filename)[0], '')

            if os.path.isfile(os.path.join(originals_dir, gz_filename)):
                return gz_filename
            elif os.path.isfile(os.path.join(originals_dir, gz_filename[:-3])):
                return gz_filename[:-3]
            else:
                url_root = 'http://www.cs.nyu.edu/~ylclab/data/'
                url = url_root + 'norb-v1.0{}/{}'.format(
                    '' if which_norb == 'big' else '-small',
                    gz_filename)

                gz_path = os.path.join(originals_dir, gz_filename)

                try:
                    download_url(url,
                                 local_filepath=gz_path,
                                 show_progress=True)
                except IOError, io_error:
                    raise IOError("IOError: {}\n"
                                  "Error attempting to download {} to "
                                  "{}. Please download the NORB files "
                                  "manually, and put them in {}.".format(
                                      io_error,
                                      url,
                                      gz_path,
                                      os.path.split(gz_path)[0]))

                return gz_filename

        for gz_filenames, actual_paths in safe_izip(gz_filename_lists,
                                                    actual_path_lists):
            for gz_filename in gz_filenames:
                actual_paths.append(os.path.join(originals_dir,
                                                 get_actual_path(gz_filename)))

        return actual_path_lists

    h5_path = os.path.join(norb_dir, 'cache.h5')

    partition_names = ('train', 'test')
    tensor_names = ('images', 'labels')
    tensor_formats = _get_tensor_formats(which_norb)
    partition_sizes = _get_partition_sizes(which_norb)

    if not os.path.isfile(h5_path):
        print("Caching {} NORB dataset to an HDF5 cache file. "
              "This will only happen once.".format(which_norb))

        with make_h5_file(h5_path,
                          partition_names,
                          partition_sizes,
                          tensor_names,
                          tensor_formats) as h5_file:

            partitions = h5_file['partitions']

            for partition_name in partition_names:
                (cat_files,
                 dat_files,
                 info_files) = get_norb_path_lists(which_norb, partition_name)

                partition = partitions[partition_name]

                images = partition['images']
                labels = partition['labels']

                index = 0

                for cat_file, dat_file, info_file in safe_izip(cat_files,
                                                               dat_files,
                                                               info_files):

                    print("Caching {}".format(dat_file))
                    dat = _read_norb_file(dat_file)
                    index_range = slice(index, index + dat.shape[0])
                    assert_equal(dat.dtype, images.dtype)
                    images[index_range, ...] = dat

                    print("Caching {}".format(cat_file))
                    cat = _read_norb_file(cat_file)
                    assert_equal(cat.shape[0], dat.shape[0])
                    assert_equal(cat.dtype, labels.dtype)
                    labels[index_range, 0] = cat

                    print("Caching {}".format(info_file))
                    info = _read_norb_file(info_file)
                    assert_equal(info.shape[0], dat.shape[0])
                    assert_equal(info.dtype, labels.dtype)
                    labels[index_range, 1:] = info

                    index += dat.shape[0]

        print("Wrote cache to {}".format(h5_path))

    return h5_path


def load_norb(which_norb, partition=None):
    '''
    Returns one or all partitions in a NORB dataset.

    Parameters
    ----------
    which_norb: string
      'big' for Big NORB, 'small' for Small NORB, or the path to a .h5 file
      for custom NORBs.

    partition: string or None
      Optional. If omitted, this returns all partitions in a tuple. If
      supplied, this returns just the named partition.
    '''
    if which_norb in ('big', 'small'):
        h5_path = _get_official_norb_cache_path(which_norb)
    else:
        h5_path = which_norb

    return load_h5_dataset(h5_path, partition)
