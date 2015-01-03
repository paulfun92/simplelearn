import numpy
from nose.tools import assert_raises_regexp, assert_less
from numpy.testing import assert_equal
from simplelearn.utils import safe_izip
from simplelearn.formats import DenseFormat
from simplelearn.data.dataset import Dataset


def test_sequential_iterator_next():
    formats = (DenseFormat(axes=('f', '0', '1', 'b'),  # mono RGB
                           shape=(3, 32, 32, -1),
                           dtype='float32'),
               DenseFormat(axes=('f', 'b', 's', '0', '1'),  # stereo grayscale
                           shape=(1, -1, 2, 8, 8),
                           dtype='int32'),
               DenseFormat(axes=('b', 'f'),  # 5-D label
                           shape=(-1, 5),
                           dtype='int32'),
               DenseFormat(axes=('b',),   # scalar label
                           shape=(-1,),
                           dtype='int'))

    names = ('mono_RGB',
             'stereo_grayscale',
             'label_vector',
             'label_scalar')

    def make_data(fmt, num_samples):
        data = fmt.make_batch(is_symbolic=False, batch_size=num_samples)
        data[...] = numpy.arange(data.size,
                                 dtype=fmt.dtype).reshape(data.shape)
        return data

    num_samples = 100

    tensors = tuple(make_data(f, num_samples) for f in formats)

    dataset = Dataset(names=names,
                      formats=formats,
                      tensors=tensors)

    assert_raises_regexp(ValueError,
                         "is not divisible by batch_size",
                         dataset.iterator,
                         iterator_type='sequential',
                         batch_size=11,
                         mode='divisible')

    def get_batch(sample_index, batch_size):
        batches = []
        for tensor, fmt in safe_izip(tensors, formats):
            index = tuple(slice(sample_index, sample_index + batch_size)
                          if a == 'b'
                          else slice(None)
                          for a in fmt.axes)
            batches.append(tensor[index])

        return tuple(batches)

    batch_size = 10
    sample_index = 0
    epoch = 0
    batch_number = 0

    assert_equal(numpy.mod(num_samples, batch_size), 0)

    iterators = tuple(dataset.iterator(iterator_type='sequential',
                                       batch_size=batch_size,
                                       mode=m) for m in ('truncate',
                                                         'loop',
                                                         'divisible'))

    # pylint: disable=star-args
    for truncate_batch, loop_batch, divisible_batch in safe_izip(*iterators):
        expected_batch = get_batch(sample_index, batch_size)
        for (iterator_batch,
             iterator_name) in safe_izip((truncate_batch,
                                          loop_batch,
                                          divisible_batch),
                                         ('truncate',
                                          'loop',
                                          'divisible')):
            assert_equal(iterator_batch,
                         expected_batch,
                         "iterator %s yielded unexpected batch %d." %
                         (iterator_name, batch_number))

        sample_index += batch_size
        if sample_index == num_samples:
            epoch += 1
            sample_index = 0
        else:
            assert_less(sample_index, num_samples)

        for iterator in iterators:
            assert_equal(iterator.epoch(), epoch)

        batch_number += 1

        if epoch == 3:
            break
