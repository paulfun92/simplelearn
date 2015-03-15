'''
nosetests for ../mnist.py
'''

import theano
import numpy
from numpy.testing import assert_allclose, assert_array_equal
from nose.tools import assert_equal, assert_true
from simplelearn.formats import DenseFormat
from simplelearn.data.mnist import load_mnist
from simplelearn.utils import safe_izip
from simplelearn.nodes import RescaleImage

try:
    from pylearn2.datasets.mnist import MNIST as Pylearn2Mnist
    pylearn2_is_installed = True
except ImportError:
    pylearn2_is_installed = False

import pdb


def test_mnist():
    mnist_datasets = load_mnist()
    for mnist, expected_size in safe_izip(mnist_datasets, (60000, 10000)):
        assert_equal(mnist._tensors[0].shape[0], expected_size)

    training_set, test_set = mnist_datasets

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


def test_mnist_against_pylearn2():
    if not pylearn2_is_installed:
        return

    simplelearn_datasets = load_mnist()
    pylearn2_datasets = [Pylearn2Mnist(which_set=w) for w in ('train', 'test')]

    def get_convert_function():
        input_nodes = simplelearn_datasets[0].make_input_nodes()
        sl_batch_converter = RescaleImage(input_nodes[0])
        return theano.function([input_nodes[0].output_symbol],
                               sl_batch_converter.output_symbol)

    convert_sl_batch = get_convert_function()

    def check_equal(sl, pl):
        batch_size = 100

        sl_iter = sl.iterator(batch_size=100,
                              iterator_type='sequential',
                              loop_style='divisible')

        pl_iter = pl.iterator(batch_size=100,
                              mode='sequential',
                              topo=True,
                              targets=True)

        keep_going = True

        count = 0

        while keep_going:
            sl_image, sl_label = sl_iter.next()
            pl_image, pl_label = pl_iter.next()

            sl_image = convert_sl_batch(sl_image)[..., numpy.newaxis]
            sl_label = sl_label[:, numpy.newaxis]

            assert_allclose(sl_image, pl_image)
            assert_array_equal(sl_label, pl_label)
            count += sl_image.shape[0]
            keep_going = not sl_iter.next_is_new_epoch()

        assert_equal(count, sl._tensors[0].shape[0])
        assert_equal(count, pl.X.shape[0])

    for sl, pl in safe_izip(simplelearn_datasets, pylearn2_datasets):
        check_equal(sl, pl)
