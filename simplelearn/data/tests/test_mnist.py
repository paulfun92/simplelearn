'''
nosetests for ../mnist.py
'''

import numpy
from simplelearn.formats import DenseFormat
from simplelearn.data.mnist import load_mnist
from simplelearn.utils import safe_izip
from nose.tools import assert_equal, assert_true

import pdb


def test_mnist():
    mnist_datasets = load_mnist()
    assert_equal(mnist_datasets.keys(), ['train', 'test'])

    training_set, test_set = mnist_datasets.values()

    expected_formats =  [DenseFormat(shape=[-1, 28, 28],
                                     axes=['b', '0', '1'],
                                     dtype='uint8'),
                         DenseFormat(shape=[-1],
                                     axes=['b'],
                                     dtype='uint8')]
    expected_names = [u'images', u'labels']
    expected_sizes = [60000, 10000]

    for dataset, expected_size in safe_izip((training_set, test_set),
                                            expected_sizes):
        assert_equal(dataset._names, expected_names)
        assert_equal(dataset._formats, expected_formats)

        for tensor, fmt in safe_izip(dataset._tensors, dataset._formats):
            fmt.check(tensor)
            assert_equal(tensor.shape[0], expected_size)

        labels = dataset._tensors[dataset._names.index('labels')]
        assert_true(numpy.logical_and(labels[...] >= 0,
                                      labels[...] < 10).all())
