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


def _load_official_norb(which_norb):
    '''
    Loads one of the official NORB dataset (big or small).

    The first time this is called, it will copy the NORB dataset to
    an HDF5 file called 'cache.h5' in the norb_directory, which is
    simplelearn.data.data_path/[big_norb, small_norb].

    If the original NORB files aren't in norb_directory/original_files,
    they will be downloaded from the web.

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
                    # filepath = os.path.join(originals_dir,
                    #                         filename)
                    # file_list.append(filepath)

            return cat_files, dat_files, info_files

        gz_filename_lists = get_norb_filename_lists_gz(which_norb, which_set)
        assert_equal(len(gz_filename_lists), 3)

        # all_gz_files = mat_gz_files[0] + mat_gz_files[1] + mat_gz_files[2]
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
                # url = 'http://www.cs.nyu.edu/~ylclab/data/norb-v1.0{}/{}'.format(
                #        + gz_filename)
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
        # def all_files_exist(dat_files, cat_files, info_files):
        #     '''
        #     Returns True if all the given NORB file paths exist on disk.
        #     '''
        #     for files in (dat_files, cat_files, info_files):
        #         if not all(os.path.isfile(f) for f in files):
        #             return False

        #     return True

        # if all_files_exist(*mat_gz_files):
        #     return mat_gz_files
        # else:  # .mat.gz files weren't there; check for .mat files.
        #     # chop off the .gz suffix
        #     mat_files = ([f[:-3] for f in files] for files in mat_gz_files)

        #     if all_files_exist(*mat_files):
        #         return mat_files
        #     else:
        #         print("Couldn't find all the raw %s NORB %ing set files in "
        #               "%s. Downloading..." % (which_norb,
        #                                       which_set,
        #                                       originals_dir))

        #         # If the .mat files don't exist either, download .mat.gz files
        #         for file_list in mat_gz_files:
        #             for filepath in file_list:
        #                 url = ('http://www.cs.nyu.edu/~ylclab/data/norb-v1.0/'
        #                        + os.path.split(filepath)[1])
        #                 try:
        #                     download_url(url,
        #                                  local_filepath=filepath,
        #                                  show_progress=True)
        #                 except IOError, io_error:
        #                     raise IOError("IOError: %s\n"
        #                                   "Error attempting to download %s to "
        #                                   "%s. Please download the NORB files "
        #                                   "manually, and put them in %s." %
        #                                   (io_error,
        #                                    url,
        #                                    filepath,
        #                                    os.path.split(filepath)[0]))

        #         return mat_gz_files

    # def add_dataset(which_norb, which_set, hdf_file):
    #     '''
    #     Copies a NORB dataset into a group in a HDF5 file.

    #     Adds an h5py.Group named <which_set> to hdf_file,
    #     and gives it 'images' and 'labels' tensors.

    #     Parameters
    #     ----------
    #     which_norb: 'big' or 'small'
    #     which_set: 'train' or 'test'
    #     hdf_file: h5py.File

    #     Returns
    #     -------
    #     rval: None
    #     '''
    #     assert_in(which_norb, ('big', 'small'))
    #     assert_in(which_set, ('train', 'test'))

    #     group = hdf_file.create_group(which_set)

    #     examples_per_file = 24300 if which_norb == 'small' else 29160

    #     if which_norb == 'small':
    #         num_examples = examples_per_file
    #     else:
    #         num_examples = examples_per_file * (10 if which_set == 'train'
    #                                             else 2)

    #     image_dim = 96 if which_norb == 'small' else 108

    #     images = add_tensor('images',
    #                         (num_examples, 2, image_dim, image_dim),
    #                         'uint8',
    #                         ('b', 's', '0', '1'),
    #                         group)

    #     label_size = 5 if which_norb == 'small' else 11

    #     labels = add_tensor('labels',
    #                         (num_examples, label_size),
    #                         'int32',
    #                         ('b', 'f'),
    #                         group)

    #     cat_files, dat_files, info_files = get_norb_filenames(which_norb,
    #                                                           which_set)

    #     index = 0
    #     for cat_file, dat_file, info_file in safe_izip(cat_files,
    #                                                    dat_files,
    #                                                    info_files):
    #         index_range = slice(index, index + examples_per_file)

    #         print("Caching %s" % dat_file)
    #         dat = _read_norb_file(dat_file)
    #         assert_equal(dat.dtype, images.dtype)
    #         images[index_range, :, :, :] = dat

    #         print("Caching %s" % cat_file)
    #         cat = _read_norb_file(cat_file)
    #         assert_equal(cat.dtype, labels.dtype)
    #         labels[index_range, 0] = cat

    #         print("Caching %s" % info_file)
    #         info = _read_norb_file(info_file)
    #         assert_equal(info.dtype, labels.dtype)
    #         labels[index_range, 1:] = info

    #         index += examples_per_file

    #     assert_equal(index, num_examples)

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


    result = tuple(H5Dataset(h5_path, p) for p in ('train', 'test'))

    for dataset, partition_size in safe_izip(result, partition_sizes):
        assert_equal(len(dataset.tensors), 2)
        assert_equal(tuple(dataset.names), tensor_names)
        assert_equal(dataset.formats, tensor_formats)
        for tensor in dataset.tensors:
            assert_equal(tensor.shape[0], partition_size)

    return result


def load_norb(norb_spec, partition=None):
    '''
    Returns one or all partitions in a NORB dataset.

    Parameters
    ----------
    norb_spec: string
      'big' for Big NORB, 'small' for Small NORB, or the path to a .h5 file
      for custom NORBs.

    partition: string or None
      Optional. If omitted, this returns all partitions in a tuple. If
      supplied, this returns just the named partition.
    '''
    if norb_spec in ('big', 'small'):
        h5_path = os.path.join(_get_norb_dir(norb_spec), 'cache.h5')
    else:
        h5_path = norb_spec

    return load_h5_dataset(h5_path, partition)
