'''
Function for loading MNIST as a Dataset.
'''

import os
import gzip
import struct
import numpy
from nose.tools import assert_equal, assert_in
import simplelearn
from simplelearn.utils import safe_izip, download_url
from simplelearn.data.h5_dataset import make_h5_file, H5Dataset
from simplelearn.formats import DenseFormat

import pdb


def _read_mnist_file(raw_file_path):
    '''
    Returns the tensor encoded in an original MNIST file as a numpy.ndarray.
    '''

    def open_file(file_path):
        '''
        Returns a readable file handle to an MNIST file, either in the original
        gzipped form, or uncompressed form.
        '''
        extension = os.path.splitext(file_path)[1]
        if extension == '.gz':
            return gzip.open(file_path, mode='rb')
        else:
            return open(file_path, mode='rb')

    def read_tensor(file_handle, shape):
        '''
        Reads the non-header part of an MNIST file.

        Works for both .gz and un-gzipped files, unlike numpy.fromfile().
        '''
        data_string = file_handle.read(-1)  # read all
        result = numpy.fromstring(data_string, dtype='uint8')
        return result.reshape(shape)

    def read_mnist_image_file(file_path):
        '''
        Reads a MNIST image file, returns it as a numpy array.
        '''
        with open_file(file_path) as raw_images_file:
            (magic_number,
             num_images,
             num_rows,
             num_cols) = struct.unpack('>iiii', raw_images_file.read(16))

            if magic_number != 2051:
                raise ValueError("Wrong magic number in MNIST images file %s" %
                                 file_path)

            return read_tensor(raw_images_file,
                               (num_images, num_rows, num_cols))

    def read_mnist_label_file(file_path):
        '''
        Reads a MNIST label file, returns it as a numpy array.
        '''
        with open_file(file_path) as raw_labels_file:

            (magic_number,
             num_labels) = struct.unpack('>ii', raw_labels_file.read(8))

            if magic_number != 2049:
                raise ValueError("Wrong magic number in MNIST labels file %s" %
                                 file_path)

            return read_tensor(raw_labels_file, [num_labels])
            # result = numpy.fromfile(raw_labels_file, dtype='uint8')
            # assert_equal(result.ndim, 1)
            # assert_equal(result.size, num_labels)
            # return result

    if '-idx3-' in raw_file_path:
        return read_mnist_image_file(raw_file_path)
    else:
        assert_in('-idx1-', raw_file_path)
        return read_mnist_label_file(raw_file_path)


def load_mnist():
    '''
    Returns the train and test sets of MNIST.

    Downloads and copies to a local HDF5 cache file if necessary.

    Returns
    -------
    rval: tuple
      (train, test), where each is an H5Dataset
    '''

    mnist_dir = os.path.join(simplelearn.data.data_path, 'mnist')
    if not os.path.isdir(mnist_dir):
        os.mkdir(mnist_dir)

    h5_path = os.path.join(mnist_dir, 'cache.h5')

    (train_filenames,
     test_filenames) = (['{}-images-idx3-ubyte.gz'.format(prefix),
                         '{}-labels-idx1-ubyte.gz'.format(prefix)]
                        for prefix in ('train', 't10k'))

    partition_names = ('train', 'test')
    tensor_names = ('images', 'labels')
    tensor_formats = (DenseFormat(axes=('b', '0', '1'),
                                  shape=(-1, 28, 28),
                                  dtype='uint8'),
                      DenseFormat(axes=('b',),
                                  shape=(-1, ),
                                  dtype='uint8'))

    if not os.path.isfile(h5_path):
        # Create the .h5 file and copy MNIST into it, downloading MNIST files
        # if necessary.
        with make_h5_file(h5_path,
                          partition_names,
                          (60000, 10000),
                          tensor_names,
                          tensor_formats) as h5_file:

            originals_dir = os.path.join(mnist_dir, 'original_files')
            if not os.path.isdir(originals_dir):
                os.mkdir(originals_dir)

            partitions = h5_file['partitions']

            for (partition_name,
                 filenames) in safe_izip(partition_names,
                                         [train_filenames, test_filenames]):
                partition = partitions[partition_name]
                h5_tensors = [partition[n] for n in tensor_names]

                for h5_tensor, original_filename in safe_izip(h5_tensors,
                                                              filenames):
                    filepath = os.path.join(originals_dir, original_filename)
                    if not os.path.isfile(filepath):
                        url_root = "http://yann.lecun.com/exdb/mnist/"
                        url = url_root + original_filename
                        download_url(url,
                                     local_filepath=filepath,
                                     show_progress=True)

                    h5_tensor[...] = _read_mnist_file(filepath)

    result = tuple(H5Dataset(h5_path, p) for p in ('train', 'test'))

    for dataset in result:
        assert_equal(len(dataset.tensors), 2)
        assert_equal(tuple(dataset.names), tensor_names)
        assert_equal(dataset.formats, tensor_formats)

    return result