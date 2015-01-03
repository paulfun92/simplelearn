import numpy
from nose.tools import assert_raises_regexp
from numpy.testing import assert_equal
from simplelearn.utils import safe_izip
from simplelearn.formats import DenseFormat
from simplelearn.data.dataset import Dataset


def test_iterator():
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

    rng = numpy.random.RandomState(3233)

    def make_data(rng, fmt, num_samples):
        data = fmt.make_batch(is_symbolic=False, batch_size=num_samples)
        data[...] = numpy.arange(data.size,
                                 dtype=fmt.dtype).reshape(data.shape)

        # if numpy.issubdtype(fmt.dtype, numpy.floating):
        #     data[...] = rng.uniform(low=-2, high=2, size=data.shape)
        # elif numpy.issubdtype(fmt.dtype, numpy.integer):
        #     data[...] = rng.randint(low=-127, high=128, size=data.shape)
        # else:
        #     raise RuntimeError("This line should never be reached.")
        return data

    num_samples = 100

    tensors = tuple(make_data(rng, f, num_samples) for f in formats)

    dataset = Dataset(names=('mono_RGB',
                             'stereo_grayscale',
                             'label_vector',
                             'label_scalar'),
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

        return batches

    batch_size = 10
    sample_index = 0
    epoch = 0

    assert_equal(numpy.mod(num_samples, batch_size), 0)

    iterators = tuple(dataset.iterator(iterator_type='sequential',
                                       batch_size=batch_size,
                                       mode=m) for m in ('truncate',
                                                         'loop',
                                                         'divisible'))

    # pylint: disable=star-args
    for sequential_batch, loop_batch, divisible_batch in safe_izip(*iterators):
        batch = get_batch(sample_index, batch_size)
        for b in sequential_batch, loop_batch, divisible_batch:
            assert_equal(batch, tuple(b))

        sample_index += batch_size
        if sample_index == num_samples:
            epoch += 1
            sample_index = 0

        for iterator in iterators:
            assert_equal(iterator.epoch(), epoch)

        if epoch == 3:
            break
