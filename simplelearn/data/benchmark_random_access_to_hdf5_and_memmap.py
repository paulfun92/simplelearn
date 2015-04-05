#! /usr/bin/env python

from __future__ import print_function

from timeit import default_timer
import h5py
import numpy
from nose.tools import assert_true, assert_is_instance
from numpy.lib.format import open_memmap
from simplelearn.utils import (human_readable_memory_size,
                               human_readable_duration,
                               assert_integer)


def main():
    # modeled after big NORB's test set images
    shape = (29160 * 2, 2, 108, 108)
    dtype = numpy.dtype('uint8')
    dtype_max = numpy.iinfo(dtype).max

    batch_size = 128
    num_batches = int(numpy.ceil(shape[0] / float(batch_size)))

    path_prefix = '/tmp/benchmark_random_access_to_hdf5_and_memmap'
    # path_prefix = '/home/mkg/tmp/benchmark_random_access_to_hdf5_and_memmap'
    h5_path = path_prefix + '.h5'
    mm_path = path_prefix + '.npy'

    def get_expected_values(start_row, end_row=None):
        if end_row is None:
            assert_is_instance(start_row, numpy.ndarray)
            values = start_row
        else:
            assert_integer(start_row)
            assert_integer(end_row)
            values = numpy.arange(start_row, end_row)

        values = values % dtype_max
        values = values.reshape((values.shape[0], ) +
                                ((1, ) * (len(shape) - 1)))

        return numpy.tile(values, shape[1:])

    def fill_tensor(tensor):
        '''
        Fill each row with its batch index.
        '''

        row_index = 0

        while row_index < shape[0]:
            print("writing {} of {} rows".format(row_index, shape[0]),
                  end='\r')

            next_row_index = min(shape[0], row_index + batch_size)
            values = get_expected_values(row_index, next_row_index)

            tensor[row_index:next_row_index, ...] = values
            row_index = next_row_index

    memory_size = human_readable_memory_size(numpy.prod(shape))

    start_time = default_timer()
    with h5py.File(h5_path, mode='w') as h5_file:
        print("Allocating %s HDF5 tensor to %s." % (memory_size, h5_path))
        h5_tensor = h5_file.create_dataset('tensor', shape, dtype)
        print("Filling HDF5 tensor.")
        fill_tensor(h5_tensor)

    duration = default_timer() - start_time
    print("HDF5 sequential write time: " + human_readable_duration(duration))
    print("{:.2g} secs per {}-sized batch".format(duration / num_batches,
                                                  batch_size))

    print("Allocating %s memmap tensor to %s." % (memory_size, mm_path))
    start_time = default_timer()
    fill_tensor(open_memmap(mm_path, 'w+', dtype, shape))
    duration = default_timer() - start_time
    print('Memmap sequential write time: %s' %
          human_readable_duration(duration))
    print("{:.2g} secs per {}-sized batch".format(duration / num_batches,
                                                  batch_size))

    rng = numpy.random.RandomState(1413)

    shuffled_indices = rng.choice(shape[0], size=shape[0], replace=False)

    def random_reads(tensor):
        row_index = 0

        is_hdf5 = isinstance(tensor, h5py.Dataset)

        while row_index < shape[0]:
            print("read {} of {} rows".format(row_index, shape[0]), end='\r')

            next_row_index = min(shape[0], row_index + batch_size)
            indices = shuffled_indices[row_index:next_row_index]
            if is_hdf5:
                indices = numpy.sort(indices)

            expected_values = get_expected_values(indices)
            assert_true((tensor[indices, ...] == expected_values).all())

            row_index = next_row_index

    print("Randomly reading from " + h5_path)
    start_time = default_timer()
    with h5py.File(h5_path, mode='r') as h5_file:
        h5_tensor = h5_file['tensor']
        random_reads(h5_tensor)

    duration = default_timer() - start_time
    print('HDF5 random read time: ' + human_readable_duration(duration))
    print("{:.2g} secs per {}-sized batch".format(duration / num_batches,
                                                  batch_size))

    print("Randomly reading from " + mm_path)
    start_time = default_timer()
    random_reads(open_memmap(mm_path, 'r', dtype, shape))
    duration = default_timer() - start_time
    print('Memmap random read time: ' + human_readable_duration(duration))
    print("{:.2g} secs per {}-sized batch".format(duration / num_batches,
                                                  batch_size))


if __name__ == '__main__':
    main()
