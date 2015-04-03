'''
Function for loading MNIST as a Dataset.
'''

from os.path import join, isdir, isfile
import struct
import numpy
from nose.tools import (assert_true,
                        assert_false,
                        assert_equal,
                        assert_in)
from simplelearn.data import data_path
from simplelearn.formats import DenseFormat
from simplelearn.utils import safe_izip
from simplelearn.data.hdf5 import Hdf5Data

import pdb


def _copy_raw_mnist_to_hdf5(raw_mnist_dir, hdf5_path):
    '''
    Copies the MNIST train and test data to a new hdf5 file.

    Parameters
    ----------
    raw_mnist_dir: str
      The path containing the raw MNIST files:
        train-images-idx3-ubyte
        train-labels-idx1-ubyte
        t10k-images-idx3-ubyte
        t10k-labels-idx1-ubyte

    Returns
    -------
    rval: Hdf5Data
      The default slices are set to 'train' and 'test'.
      The tensors are 'images' and 'labels'.

      Extract train and test slices as:

        training_set = Hdf5DatasetSlice(rval, 'train')
        testing_set = Hdf5DatasetSlice(rval, 'test')
    '''

    assert_true(isdir(raw_mnist_dir))
    assert_false(isfile(hdf5_path))

    def read_raw_images(raw_images_file_path):
        '''
        Reads a MNIST images file, returns it as a numpy array.
        '''
        raw_images_file = open(raw_images_file_path, 'rb')
        (magic_number,
         num_images,
         num_rows,
         num_cols) = struct.unpack('>iiii', raw_images_file.read(16))

        if magic_number != 2051:
            raise ValueError("Wrong magic number in MNIST images file %s" %
                             raw_images_file_path)

        data = numpy.fromfile(raw_images_file, dtype='uint8')
        return data.reshape((num_images, num_rows, num_cols))

    def read_raw_labels(raw_labels_file_path):
        '''
        Reads a MNIST labels file, returns it as a numpy array.
        '''
        raw_labels_file = open(raw_labels_file_path, 'rb')
        (magic_number,
         num_labels) = struct.unpack('>ii', raw_labels_file.read(8))

        if magic_number != 2049:
            raise ValueError("Wrong magic number in MNIST labels file %s" %
                             raw_labels_file_path)

        result = numpy.fromfile(raw_labels_file, dtype='uint8')
        assert_equal(result.ndim, 1)
        assert_equal(result.size, num_labels)
        return result

    def get_data(train_or_test, images_or_labels):
        '''
        Returns the train/test set's images/labels as a numpy array.
        '''

        assert_in(train_or_test, ('train', 'test'))
        assert_in(images_or_labels, ('images', 'labels'))

        prefix = 'train' if train_or_test == 'train' else 't10k'

        if images_or_labels == 'images':
            filepath = join(raw_mnist_dir,
                            '%s-images-idx3-ubyte' % prefix)
            return read_raw_images(filepath)
        else:
            filepath = join(raw_mnist_dir,
                            '%s-labels-idx1-ubyte' % prefix)
            return read_raw_labels(filepath)

    image_data = [get_data(t, 'images') for t in ('train', 'test')]
    label_data = [get_data(t, 'labels') for t in ('train', 'test')]
    for images, labels, expected_size in safe_izip(image_data,
                                                   label_data,
                                                   (60000, 10000)):
        assert_equal(images.shape[0], expected_size)
        assert_equal(labels.shape[0], expected_size)

    hdf5_data = Hdf5Data(hdf5_path, mode='w-', size=70000)
    hdf5_data.add_tensor('images',
                         DenseFormat(axes=('b', '0', '1'),
                                     shape=(-1, ) + image_data[0].shape[1:],
                                     dtype=image_data[0].dtype))
    hdf5_data.add_tensor('labels',
                         DenseFormat(axes=('b'),
                                     shape=(-1, ) + label_data[0].shape[1:],
                                     dtype=label_data[0].dtype))

    training_set = hdf5_data.add_default_slice('train', 60000)
    testing_set = hdf5_data.add_default_slice('test', -1)

    hdf5_data['train'] = [image_data[0], label_data[0]]
    hdf5_data['test'] = [image_data[1], label_data[1]]

    return Hdf5Data(hdf5_path, mode='r')


def load_mnist():
    '''
    Loads MNIST data.

    If this is the first time this function is called, the data will be read
    from the raw MNIST files, and copied to a single HDF5 file, stored in
    $SIMPLELEARN_DATA_PATH/mnist/mnist_cache.h5. This is a one-time operation.

    Returns
    -------
    rval: Hdf5Data
    '''
    default_mnist_dir = join(data_path, 'mnist')
    cache_path = join(default_mnist_dir, 'cache.h5')

    if isfile(cache_path):
        return Hdf5Data(cache_path, mode='r')
    else:  # Construct cache file
        raw_mnist_dir = join(default_mnist_dir, 'original_files')
        return _copy_raw_mnist_to_hdf5(raw_mnist_dir, cache_path)
