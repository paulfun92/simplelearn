#! /usr/bin/env python

from __future__ import print_function

import argparse
from timeit import default_timer
import h5py
import numpy
from nose.tools import assert_true, assert_is_instance
from numpy.lib.format import open_memmap
from simplelearn.utils import (human_readable_memory_size,
                               human_readable_duration)
from simplelearn.asserts import assert_integer


def main():

    def parse_args():
        parser = argparse.ArgumentParser(
            description=("Compares sequential write and random read times for "
                         "HDF5 vs memmap datasets."))

        parser.add_argument("--output-dir",
                            default="/tmp/",
                            help=("The directory to output to. Handy for "
                                  "comparing HDD vs SDD performance."))

        parser.add_argument("--no-memmap",
                            action='store_true',
                            default=False,
                            help="Don't test memmaps.")

        parser.add_argument("--no-h5",
                            action='store_true',
                            default=False,
                            help="Don't test HDF5.")

        parser.add_argument("--batch-size",
                            default=128,
                            help="Number of images per batch")

        parser.add_argument("--dtype",
                            type=numpy.dtype,
                            default='uint8',
                            help="Data dtype.")

        parser.add_argument("--image-dim",
                            default=108,  # from big NORB dataset
                            help=("Size of one side of the random square "
                                  "images."))

        parser.add_argument("--num-gb",
                            type=float,
                            default=1.0,
                            help="File size, in GB")

    args = parse_args()

    # modeled after big NORB's test set images
    example_shape = (args.image_dim, args.image_dim)
    example_size = numpy.prod(example_shape) * args.dtype.itemsize
    num_examples = numpy.floor(num_GB * (1024 ** 3) / example_size)

    shape = (num_examples, args.image_dim, args.image_dim)
    dtype_max = numpy.iinfo(args.dtype).max

    # batch_size = 128
    num_batches = int(numpy.ceil(shape[0] / float(args.batch_size)))

    path_prefix = os.path.join(args.output_dir,
                               '/benchmark_random_access_to_hdf5_and_memmap')
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

            next_row_index = min(shape[0], row_index + args.batch_size)
            values = get_expected_values(row_index, next_row_index)

            tensor[row_index:next_row_index, ...] = values
            row_index = next_row_index

    memory_size = human_readable_memory_size(numpy.prod(shape))

    if not args.no_h5:
        start_time = default_timer()
        with h5py.File(h5_path, mode='w') as h5_file:
            print("Allocating %s HDF5 tensor to %s." % (memory_size, h5_path))
            h5_tensor = h5_file.create_dataset('tensor', shape, args.dtype)
            print("Filling HDF5 tensor.")
            fill_tensor(h5_tensor)

        duration = default_timer() - start_time
        print("HDF5 sequential write time: " + human_readable_duration(duration))
        print("{:.2g} secs per {}-sized batch".format(duration / num_batches,
                                                      args.batch_size))

    if not args.no_memmap:
        print("Allocating %s memmap tensor to %s." % (memory_size, mm_path))
        start_time = default_timer()
        fill_tensor(open_memmap(mm_path, 'w+', args.dtype, shape))
        duration = default_timer() - start_time
        print('Memmap sequential write time: %s' %
              human_readable_duration(duration))
        print("{:.2g} secs per {}-sized batch".format(duration / num_batches,
                                                      args.batch_size))

    rng = numpy.random.RandomState(1413)

    shuffled_indices = rng.choice(shape[0], size=shape[0], replace=False)

    def random_reads(tensor):
        row_index = 0

        is_hdf5 = isinstance(tensor, h5py.Dataset)

        while row_index < shape[0]:
            print("read {} of {} rows".format(row_index, shape[0]), end='\r')

            next_row_index = min(shape[0], row_index + args.batch_size)
            indices = shuffled_indices[row_index:next_row_index]
            if is_hdf5:
                indices = numpy.sort(indices)

            expected_values = get_expected_values(indices)
            assert_true((tensor[indices, ...] == expected_values).all())

            row_index = next_row_index

    if not args.no_h5:
        print("Randomly reading from " + h5_path)
        start_time = default_timer()
        with h5py.File(h5_path, mode='r') as h5_file:
            h5_tensor = h5_file['tensor']
            random_reads(h5_tensor)

        duration = default_timer() - start_time
        print('HDF5 random read time: ' + human_readable_duration(duration))
        print("{:.2g} secs per {}-sized batch".format(duration / num_batches,
                                                      args.batch_size))

    if not args.no_memmap:
        print("Randomly reading from " + mm_path)
        start_time = default_timer()
        random_reads(open_memmap(mm_path, 'r', args.dtype, shape))
        duration = default_timer() - start_time
        print('Memmap random read time: ' + human_readable_duration(duration))
        print("{:.2g} secs per {}-sized batch".format(duration / num_batches,
                                                      args.batch_size))


if __name__ == '__main__':
    main()
