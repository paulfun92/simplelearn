'''
Function for loading MNIST as a Dataset.
'''

import warnings
from os.path import join, isdir, isfile
import struct
from collections import OrderedDict
import h5py
import numpy
from nose.tools import (assert_true,
                        assert_false,
                        assert_is_instance,
                        assert_equal,
                        assert_in,
                        assert_is_instance)
from simplelearn.formats import DenseFormat
from simplelearn.data import data_path
from simplelearn.data.dataset import Dataset


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
    rval: h5py.File
      An h5py file handle with the following structure:
        rval['train']['images'] := training images
        rval['train']['labels'] := training labels
        rval['test']['images'] := test images
        rval['test']['labels'] := test labels
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

    def add_dataset(train_or_test, hdf5_file):
        '''
        Adds the training or testing set to the hdf5 file.
        '''
        assert_in(train_or_test, ('train', 'test'))

        group = hdf5_file.create_group(train_or_test)

        prefix = 'train' if train_or_test == 'train' else 't10k'

        raw_images_path = join(raw_mnist_dir, '%s-images-idx3-ubyte' % prefix)
        raw_images = read_raw_images(raw_images_path)
        images = group.create_dataset('images',
                                      raw_images.shape,
                                      dtype=raw_images.dtype)
        images[...] = raw_images

        raw_labels_path = join(raw_mnist_dir, '%s-labels-idx1-ubyte' % prefix)
        raw_labels = read_raw_labels(raw_labels_path)
        labels = group.create_dataset('labels',
                                      raw_labels.shape,
                                      dtype=raw_labels.dtype)
        labels[...] = raw_labels

    with h5py.File(hdf5_path, 'w-') as writeable_hdf5_file:
        add_dataset('train', writeable_hdf5_file)
        add_dataset('test', writeable_hdf5_file)

    return h5py.File(hdf5_path, 'r')


def load_mnist(raw_mnist_dir=None):
    '''
    Loads MNIST data.

    If this is the first time this function is called, the data will be read
    from the raw MNIST files, and copied to a single HDF5 file, stored in
    $SIMPLELEARN_DATA_PATH/mnist/mnist_cache.h5. This is a one-time operation.
    Subsequent calls to load_mnist will ignore the raw_mnist_dir


    Parameters
    ----------
    raw_mnist_dir: str or None
      The directory in which the raw MNIST files are to be found.
      If None, this will use "$SIMPLELEARN_DATA_PATH/mnist"

    Returns
    -------
    rval: tuple
      A tuple of two simplelearn.data.Datasets, the training and testing set.
    '''
    default_mnist_dir = join(data_path, 'mnist')
    cache_path = join(default_mnist_dir, 'mnist_cache.h5')

    if isfile(cache_path):
        if raw_mnist_dir is not None:
            warnings.warn("Ignoring raw_mnist_dir argument '%s', since a "
                          "cached copy of MNIST already exists at %s." %
                          (raw_mnist_dir, cache_path))

        hdf_file = h5py.File(cache_path, 'r')
    else:

        if raw_mnist_dir is None:
            raw_mnist_dir = default_mnist_dir
        else:
            assert_is_instance(raw_mnist_dir, basestring)

        hdf_file = _copy_raw_mnist_to_hdf5(raw_mnist_dir, cache_path)

    def group_to_dataset(group):
        assert_equal(group.keys(), ['images', 'labels'])
        images = group['images']
        labels = group['labels']

        formats = [DenseFormat(axes=('b', '0', '1'),
                               shape=[-1, images.shape[1], images.shape[2]],
                               dtype=images.dtype),
                   DenseFormat(axes=['b'],
                               shape=[-1],
                               dtype=labels.dtype)]

        return Dataset(group.keys(), group.values(), formats)

    return tuple(group_to_dataset(hdf_file[set_name])
                 for set_name in ('train', 'test'))
