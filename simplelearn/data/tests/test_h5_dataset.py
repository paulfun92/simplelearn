'''
nosetests for ../h5_dataset.py.

These tests require MNIST to be installed in:
  <simplelearn.data.data_path>/mnist/original_files/
'''

import os
import cPickle
import numpy
from nose.tools import (assert_equal, assert_less)
from numpy.testing import assert_array_equal, assert_allclose
from simplelearn.utils import safe_izip
from simplelearn.data.mnist import load_mnist
from simplelearn.data.dataset import Dataset
from simplelearn.formats import DenseFormat
from simplelearn.data.h5_dataset import RandomIterator


def _file_size_in_bytes(file_path):
    '''
    Returns the file size.
    '''

    return os.stat(file_path).st_size


def test_pickle_h5_dataset():
    '''
    Tests pickling and unpickling of H5Dataset.
    '''

    # Path for the pickle file, not the .h5 file.
    file_path = '/tmp/test_mnist_test_pickle_hdf5_data.pkl'

    def make_pickle(file_path):
        '''
        Pickles the MNIST dataset.
        '''
        hdf5_data = load_mnist()
        with open(file_path, 'wb') as pickle_file:
            cPickle.dump(hdf5_data,
                         pickle_file,
                         protocol=cPickle.HIGHEST_PROTOCOL)

    make_pickle(file_path)
    assert_less(_file_size_in_bytes(file_path), 1024 * 5)

    def load_pickle(file_path):
        '''
        Loads the MNIST dataset pickled above.
        '''
        with open(file_path, 'rb') as pickle_file:
            return cPickle.load(pickle_file)

    mnist_datasets = load_mnist()
    pickled_mnist_datasets = load_pickle(file_path)

    for (mnist_dataset,
         pickled_mnist_dataset) in safe_izip(mnist_datasets,
                                             pickled_mnist_datasets):
        for (name,
             expected_name,
             fmt,
             expected_fmt,
             tensor,
             expected_tensor) in safe_izip(pickled_mnist_dataset.names,
                                           mnist_dataset.names,
                                           pickled_mnist_dataset.formats,
                                           mnist_dataset.formats,
                                           pickled_mnist_dataset.tensors,
                                           mnist_dataset.tensors):
            assert_equal(name, expected_name)
            assert_equal(fmt, expected_fmt)
            assert_array_equal(tensor, expected_tensor)


def test_random_iterator():
    num_classes = 3
    dataset_size = 100 * num_classes

    vectors = numpy.arange(dataset_size).reshape((1, dataset_size))
    labels = numpy.arange(dataset_size) % num_classes

    dataset = Dataset(tensors=(vectors, labels),
                      names=('vectors', 'labels'),
                      formats=(DenseFormat(axes=('f', 'b'),
                                           shape=(1, -1),
                                           dtype=int),
                               DenseFormat(axes=['b'],
                                           shape=[-1],
                                           dtype=int)))

    seed = 225234
    batch_size = 231

    iterator = RandomIterator(dataset,
                              batch_size=batch_size,
                              rng=numpy.random.RandomState(seed))

    rng = numpy.random.RandomState(seed)

    # Set each label's probability to be proportional to the label value.
    iterator.probabilities[:] = labels / float(labels.sum())

    counts = numpy.zeros(num_classes, dtype=int)

    for batch_index in xrange(iterator.batches_per_epoch * 100):
        batch_vectors, batch_labels = iterator.next()

        expected_indices = rng.choice(iterator.probabilities.shape[0],
                                      size=batch_size,
                                      replace=True,
                                      p=iterator.probabilities)
        expected_indices.sort()
        expected_vectors = dataset.tensors[0][:, expected_indices]
        expected_labels = dataset.tensors[1][expected_indices]

        assert_array_equal(batch_vectors, expected_vectors)
        assert_array_equal(batch_labels, expected_labels)

        assert_equal(iterator.next_is_new_epoch(),
                     (batch_index + 1) % iterator.batches_per_epoch == 0)

        for label_value in range(num_classes):
            counts[label_value] += \
                numpy.count_nonzero(expected_labels == label_value)

    probabilities = counts / float(counts.sum())
    expected_probabilities = (numpy.arange(num_classes) /
                              float(numpy.arange(num_classes).sum()))
    assert_allclose(probabilities, expected_probabilities, atol=.01)

    assert_equal(counts[0], 0)
