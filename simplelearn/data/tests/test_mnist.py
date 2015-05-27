'''
nosetests for ../mnist.py
'''

import theano
import numpy
from numpy.testing import assert_allclose, assert_array_equal
import nose
from nose.tools import assert_equal, assert_true
from simplelearn.formats import DenseFormat
from simplelearn.data.mnist import load_mnist
from simplelearn.utils import safe_izip
from simplelearn.asserts import assert_all_equal
from simplelearn.nodes import RescaleImage

try:
    from pylearn2.datasets.mnist import MNIST as Pylearn2Mnist
    from pylearn2.space import Conv2DSpace, IndexSpace, CompositeSpace
    PYLEARN2_IS_INSTALLED = True
except ImportError:
    PYLEARN2_IS_INSTALLED = False


def test_mnist():
    '''
    Tests load_mnist().

    Checks test & train sets' formats and sizes, but not content.
    '''
    train_set, test_set = load_mnist()

    for mnist, expected_size in safe_izip((train_set, test_set),
                                          (60000, 10000)):
        assert_equal(mnist.size, expected_size)

    expected_formats = [DenseFormat(shape=[-1, 28, 28],
                                    axes=['b', '0', '1'],
                                    dtype='uint8'),
                        DenseFormat(shape=[-1],
                                    axes=['b'],
                                    dtype='uint8')]
    expected_names = [u'images', u'labels']
    expected_sizes = [60000, 10000]

    for dataset, expected_size in safe_izip((train_set, test_set),
                                            expected_sizes):
        assert_all_equal(dataset.names, expected_names)
        assert_all_equal(dataset.formats, expected_formats)

        for tensor, fmt in safe_izip(dataset.tensors, dataset.formats):
            fmt.check(tensor)
            assert_equal(tensor.shape[0], expected_size)

        labels = dataset.tensors[dataset.names.index('labels')]
        assert_true(numpy.logical_and(labels[...] >= 0,
                                      labels[...] < 10).all())


def test_mnist_against_pylearn2():
    '''
    Tests the content of the MNIST dataset loaded by load_mnist().

    Compares against pylearn2's MNIST wrapper. No-op if pylearn2 is not
    installed.
    '''
    if not PYLEARN2_IS_INSTALLED:
        raise nose.SkipTest()

    train_set, test_set = load_mnist()

    simplelearn_datasets = (train_set, test_set)
    pylearn2_datasets = [Pylearn2Mnist(which_set=w) for w in ('train', 'test')]

    def get_convert_function():
        '''
        Converts simplelearn's mnist data (uint8, [0, 255]), to
        pylearn2's iterator output (float32, [0.0, 1.0]).
        '''
        iterator = simplelearn_datasets[0].iterator(iterator_type='sequential',
                                                    batch_size=1)
        input_nodes = iterator.make_input_nodes()
        sl_batch_converter = RescaleImage(input_nodes[0])
        return theano.function([input_nodes[0].output_symbol],
                               sl_batch_converter.output_symbol)

    convert_s_batch = get_convert_function()

    def check_equal(s_mnist, p_mnist):
        '''
        Compares simplelearn and pylearn2's MNIST datasets.
        '''
        batch_size = 100

        s_iter = s_mnist.iterator(batch_size=batch_size,
                                  iterator_type='sequential',
                                  loop_style='divisible')

        def get_pylearn2_iterator():
            image_space = Conv2DSpace(shape=[28, 28],
                                      num_channels=1,
                                      axes=('b', 0, 1, 'c'),
                                      # pylearn2.MNIST forces this dtype
                                      dtype='float32')
            # label_space = VectorSpace(dim=10, dtype='uint8')
            label_space = IndexSpace(max_labels=10, dim=1, dtype='uint8')
            space = CompositeSpace([image_space, label_space])
            source = ('features', 'targets')
            specs = (space, source)

            return p_mnist.iterator(batch_size=batch_size,
                                    mode='sequential',
                                    data_specs=specs)

        p_iter = get_pylearn2_iterator()
        keep_going = True
        count = 0

        while keep_going:
            s_image, s_label = s_iter.next()
            p_image, p_label = p_iter.next()

            s_image = convert_s_batch(s_image)[..., numpy.newaxis]
            s_label = s_label[:, numpy.newaxis]

            assert_allclose(s_image, p_image)
            assert_array_equal(s_label, p_label)
            count += s_image.shape[0]
            keep_going = not s_iter.next_is_new_epoch()

        assert_equal(count, s_mnist.tensors[0].shape[0])
        assert_equal(count, p_mnist.X.shape[0])

    for s_mnist, p_mnist in safe_izip(simplelearn_datasets,
                                      pylearn2_datasets):
        check_equal(s_mnist, p_mnist)
