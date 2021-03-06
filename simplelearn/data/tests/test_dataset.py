import numpy
from nose.tools import (assert_raises_regexp,
                        assert_less,
                        assert_less_equal)
from numpy.testing import assert_equal
from simplelearn.utils import safe_izip
from simplelearn.formats import DenseFormat
from simplelearn.data.dataset import Dataset

import pdb


def test_sequential_iterator_next():
    formats = (DenseFormat(axes=('b',),   # scalar label
                           shape=(-1,),
                           dtype='int'),
               DenseFormat(axes=('b', 'f'),  # 5-D label
                           shape=(-1, 5),
                           dtype='int32'),
               DenseFormat(axes=('f', '0', '1', 'b'),  # mono RGB
                           shape=(3, 32, 32, -1),
                           dtype='float32'),
               DenseFormat(axes=('f', 'b', 's', '0', '1'),  # stereo grayscale
                           shape=(1, -1, 2, 8, 8),
                           dtype='int32'))

    names = ('label_scalar',
             'label_vector',
             'mono_RGB',
             'stereo_grayscale')

    max_epochs = 3

    def make_data(fmt, num_samples):
        data = fmt.make_batch(is_symbolic=False, batch_size=num_samples)
        data[...] = numpy.arange(data.size,
                                 dtype=fmt.dtype).reshape(data.shape)
        return data

    num_samples = 103

    tensors = tuple(make_data(f, num_samples) for f in formats)

    dataset = Dataset(names=names,
                      formats=formats,
                      tensors=tensors)

    assert_raises_regexp(ValueError,
                         "is not divisible by batch_size",
                         dataset.iterator,
                         iterator_type='sequential',
                         batch_size=11,
                         loop_style='divisible')

    def get_batch(sample_index, batch_size, tensors):
        batches = []
        for tensor, fmt in safe_izip(tensors, formats):
            index = tuple(slice(sample_index, sample_index + batch_size)
                          if a == 'b'
                          else slice(None)
                          for a in fmt.axes)
            assert_less_equal(sample_index + batch_size,
                              tensor.shape[fmt.axes.index('b')])
            batches.append(tensor[index])

        return tuple(batches)

    batch_size = 10

    #
    # Tests 'truncate' loop_style
    #

    sample_index = 0
    truncated_num_samples = num_samples - numpy.mod(num_samples, batch_size)

    iterator = dataset.iterator(iterator_type='sequential',
                                batch_size=batch_size,
                                loop_style='truncate')

    assert iterator.next_is_new_epoch()

    epoch = 0
    for batch_number, iterator_batch in enumerate(iterator):
        row_index = numpy.mod(sample_index, truncated_num_samples)
        expected_batch = get_batch(row_index, batch_size, tensors)
        sample_index += batch_size
        assert_equal(iterator.next_is_new_epoch(),
                     sample_index % truncated_num_samples == 0)

        assert_equal(iterator_batch,
                     expected_batch,
                     "'truncated' iterator yielded unexpected batch %d." %
                     batch_number)

        if iterator.next_is_new_epoch():
            epoch += 1
            if epoch == max_epochs:
                break

    #
    # Tests 'wrap' loop_style
    #

    sample_index = 0
    iterator = dataset.iterator(iterator_type='sequential',
                                batch_size=batch_size,
                                loop_style='wrap')
    assert iterator.next_is_new_epoch()

    looped_tensors = []
    for tensor, fmt in safe_izip(tensors, formats):
        tile_pattern = [1, ] * len(fmt.axes)
        tile_pattern[fmt.axes.index('b')] = max_epochs + 1
        looped_tensors.append(numpy.tile(tensor, tile_pattern))

    epoch = 0

    for batch_number, iterator_batch in enumerate(iterator):
        expected_batch = get_batch(sample_index, batch_size, looped_tensors)
        assert_equal(iterator_batch, expected_batch)

        sample_index += batch_size

        assert_equal(iterator.next_is_new_epoch(),
                     sample_index % num_samples < batch_size)

        if iterator.next_is_new_epoch():
            epoch += 1
            if epoch == max_epochs:
                break

    #
    # Tests 'divisible' loop_style
    #

    sample_index = 0
    assert_raises_regexp(ValueError,
                         ("# of samples %d is not divisible by batch_size" %
                          num_samples),
                         dataset.iterator,
                         iterator_type='sequential',
                         batch_size=batch_size,
                         loop_style='divisible')

    def crop_tensors(tensors, num_samples):
        result = []
        for tensor, fmt in safe_izip(tensors, formats):
            selector = tuple(slice(0, num_samples)
                             if axis == 'b'
                             else slice(None)
                             for axis in fmt.axes)
            result.append(tensor[selector])

        return tuple(result)

    cropped_tensors = crop_tensors(tensors, truncated_num_samples)
    cropped_dataset = Dataset(names=names,
                              formats=formats,
                              tensors=cropped_tensors)

    iterator = cropped_dataset.iterator(iterator_type='sequential',
                                        batch_size=batch_size,
                                        loop_style='divisible')

    assert iterator.next_is_new_epoch()

    epoch = 0
    for batch_number, iterator_batch in enumerate(iterator):
        row_index = sample_index % truncated_num_samples
        expected_batch = get_batch(row_index, batch_size, cropped_tensors)
        assert_equal(iterator_batch, expected_batch)

        sample_index += batch_size

        assert_equal(sample_index % truncated_num_samples == 0,
                     iterator.next_is_new_epoch())

        if iterator.next_is_new_epoch():
            epoch += 1

            if epoch == max_epochs:
                break
