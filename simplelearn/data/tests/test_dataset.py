import numpy
from nose.tools import (assert_raises_regexp,
                        assert_less,
                        assert_less_equal)
from numpy.testing import assert_equal
from simplelearn.utils import safe_izip
from simplelearn.formats import DenseFormat
from simplelearn.data.dataset import Dataset


def test_sequential_iterator_next():
    # formats = (DenseFormat(axes=('f', '0', '1', 'b'),  # mono RGB
    #                        shape=(3, 32, 32, -1),
    #                        dtype='float32'),
    #            DenseFormat(axes=('f', 'b', 's', '0', '1'),  # stereo grayscale
    #                        shape=(1, -1, 2, 8, 8),
    #                        dtype='int32'),
    #            DenseFormat(axes=('b', 'f'),  # 5-D label
    #                        shape=(-1, 5),
    #                        dtype='int32'),
    #            DenseFormat(axes=('b',),   # scalar label
    #                        shape=(-1,),
    #                        dtype='int'))

    # names = ('mono_RGB',
    #          'stereo_grayscale',
    #          'label_vector',
    #          'label_scalar')

    formats = (DenseFormat(axes=('b',),   # scalar label
                           shape=(-1,),
                           dtype='int'), )

    names = ('label_scalar', )

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
                         mode='divisible')

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
    # Tests 'truncate' mode
    #

    sample_index = 0
    truncated_num_samples = num_samples - numpy.mod(num_samples, batch_size)

    iterator = dataset.iterator(iterator_type='sequential',
                                batch_size=batch_size,
                                mode='truncate')

    for batch_number, iterator_batch in enumerate(iterator):
        row_index = numpy.mod(sample_index, truncated_num_samples)
        expected_batch = get_batch(row_index, batch_size, tensors)
        expected_epoch = sample_index // truncated_num_samples
        sample_index += batch_size
        assert_equal(iterator.epoch(),
                     expected_epoch,
                     "sample_index: %d, batch_number: %d" %
                     (sample_index, batch_number))

        assert_equal(iterator_batch,
                     expected_batch,
                     "'truncated' iterator yielded unexpected batch %d." %
                     batch_number)

        if expected_epoch == max_epochs:
            break

    #
    # Tests 'loop' mode
    #

    sample_index = 0
    iterator = dataset.iterator(iterator_type='sequential',
                                batch_size=batch_size,
                                mode='loop')

    looped_tensors = []
    for tensor, fmt in safe_izip(tensors, formats):
        tile_pattern = [1, ] * len(fmt.axes)
        tile_pattern[fmt.axes.index('b')] = max_epochs + 1
        looped_tensors.append(numpy.tile(tensor, tile_pattern))

    expected_epoch = -1
    for batch_number, iterator_batch in enumerate(iterator):
        expected_batch = get_batch(sample_index, batch_size, looped_tensors)
        assert_equal(iterator_batch, expected_batch)

        if sample_index % num_samples < batch_size:
            expected_epoch += 1

        assert_equal(iterator.epoch(), expected_epoch)
        sample_index += batch_size

        if expected_epoch == max_epochs:
            break


    #
    # Tests ''
    #
        # row_index = numpy.mod(sample_index, num_samples)
        # if row_index < batch_size:
        #     expected_epoch += 1

        # if row_index + batch_size > num_samples:

        #     expected_batch = tuple(fmt.make_batch(is_symbolic=False)
        #                            for fmt in formats)

        #     chunk_size = num_samples - row_index
        #     expected_chunk = get_batch(row_index, chunk_size)
        #     for b, c, f in safe_izip(expected_batch, expected_chunk, formats):
        #         b[f.axes.index('b')]

        #     for (iterator_subbatch,
        #          expected_subchunk) for safe_izip(iterator_batch,
        #                                           expected_chunk):
        #         assert_equal(iterator_subbatch[:
        # else:
        #     expected_batch = get_batch(row_index, batch_size)

        # sample_index += batch_size
        # assert_equal(truncated_iterator.epoch(),
        #              expected_epoch,
        #              "sample_index: %d, batch_number: %d" %
        #              (sample_index, batch_number))

        # assert_equal(iterator_batch,
        #              expected_batch,
        #              "'truncated' iterator yielded unexpected batch %d." %
        #              batch_number)

        # if epoch == max_epochs:
        #     break


    ################################



    # iterators = tuple(dataset.iterator(iterator_type='sequential',
    #                                    batch_size=batch_size,
    #                                    mode=m) for m in ('truncate',
    #                                                      'loop',
    #                                                      'divisible'))

    # # pylint: disable=star-args
    # for truncate_batch, loop_batch, divisible_batch in safe_izip(*iterators):
    #     expected_truncated_batch = get_batch(
    #     expected_loop_batch = get_batch(numpy.mod(sample_index % num_samples),
    #                                          batch_size)
    #     for (iterator_batch,
    #          iterator_name) in safe_izip((truncate_batch,
    #                                       loop_batch,
    #                                       divisible_batch),
    #                                      ('truncate',
    #                                       'loop',
    #                                       'divisible')):
    #         assert_equal(iterator_batch,
    #                      expected_batch,
    #                      "iterator %s yielded unexpected batch %d." %
    #                      (iterator_name, batch_number))

    #     sample_index += batch_size
    #     if sample_index == num_samples:
    #         epoch += 1
    #         sample_index = 0
    #     else:
    #         assert_less(sample_index, num_samples)

    #     for iterator in iterators:
    #         assert_equal(iterator.epoch(), epoch)

    #     batch_number += 1

    #     if epoch == 3:
    #         break
