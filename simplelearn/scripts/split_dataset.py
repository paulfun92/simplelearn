#! /usr/bin/env python

'''
Splits a dataset into two. Useful for creating training and testing sets.
'''

from __future__ import print_function

import os
import sys
import argparse
import string
import numpy
from nose.tools import (assert_true,
                        assert_greater_equal,
                        assert_less_equal)
from simplelearn.utils import safe_izip
from simplelearn.data.h5_dataset import make_h5_file, load_h5_dataset
from simplelearn.data.memmap_dataset import make_memmap_file, MemmapDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Splits a .npy or .h5 dataset into two.\n"
                     "\n"
                     "If the input is a .h5 dataset, it must have only one "
                     "partition."
                     "\n"
                     "The output will be either a single .h5 file with two "
                     "partitions, or two .npy files, one for each partition."))

    # def dataset_path(arg):
    #     if not arg.endswith('.h5') and not arg.endswith('.npy'):
    #         print("Expected dataset to be a .h5 or .npy file, but got " + arg)
    #         sys.exit(1)

    #     parent_dir = os.path.dirname(os.path.abspath(arg))
    #     assert_true(os.path.isdir(parent_dir))

    #     return arg

    def partition_name(arg):
        # a limitation of h5
        allowed_characters = string.ascii_letters + string.digits + "_"

        assert_true(all(c in allowed_characters for c in arg),
                    "parition name {} may only contain alphanumeric "
                    "characters and '_'".format(arg))
        return arg


    def existing_dataset_path(arg):
        if not arg.endswith('.h5') and not arg.endswith('.npy'):
            print("Expected dataset to be a .h5 or .npy file, but got " + arg)
            sys.exit(1)

        parent_dir = os.path.dirname(os.path.abspath(arg))
        assert_true(os.path.isdir(parent_dir))

        assert_true(os.path.isfile(arg))
        return arg


    def existing_dir(arg):
        assert_true(os.path.isdir(arg))
        return arg

    def nonnegative_fraction(arg):
        arg = float(arg)
        assert_greater_equal(arg, 0.0)
        assert_less_equal(arg, 1.0)
        return arg

    parser.add_argument("-i",
                        "--input",
                        type=existing_dataset_path,
                        help=("The dataset to split. Must be a .h5 or .npy "
                              "file. If a .h5 file, it must be of a "
                              "single-partition H5Dataset."))

    parser.add_argument("-o",
                        "--output-dir",
                        default=None,
                        type=existing_dir,
                        help="The directory to write the output file(s) to.")

    parser.add_argument("--output-format",
                        choices=('h5', 'npy'),
                        default=None,
                        help=("Output file format. Choose from h5 or npy. "
                              "If omitted, the output format is the same as "
                              "the input format."))

    parser.add_argument("--ratio",
                        type=nonnegative_fraction,
                        help=("The fraction of input data to use for the "
                              "first output dataset. The remainder will go "
                              "in the second output dataset."))

    parser.add_argument("--seed",
                        default=1234,
                        type=int,
                        help=("Seed for the RNG that chooses which examples "
                              "to use for the training and testing sets."))

    parser.add_argument("--partition-names",
                        default=('train', 'valid'),
                        type=partition_name,
                        nargs=2,
                        help="Partition names of the output dataset.")

    args = parser.parse_args()

    if args.output_format is None:
        args.output_format = 'npy' if args.input.endswith('.npy') else "h5"

    if args.output_dir is None:
        args.output_dir = os.path.split(args.input)[0]

    return args


def get_batch_slice(fmt, batch_index_slice):
    '''
    Returns a tuple T such that A[T] returns a slice along the batch dimension.
    '''
    return tuple(batch_index_slice if axis == 'b' else slice(None)
                 for axis in fmt.axes)


def get_partition_masks(full_dataset, ratio, rng):
    num_examples = full_dataset.num_examples()
    num_training_examples = int(numpy.round(num_examples * ratio))
    training_example_indices = rng.choice(num_examples,
                                          size=num_training_examples,
                                          replace=False)
    partition_mask = numpy.zeros(num_examples, dtype=bool)
    partition_mask[training_example_indices] = True

    return partition_mask, numpy.logical_not(partition_mask)


def write_memmaps(full_dataset, args, rng):

    def get_partition_path(output_dir, input_path, partition_name):
        basename = os.path.splitext(os.path.split(input_path)[1])[0]
        return os.path.join(output_dir,
                            "{}_split_{}_{}.npy".format(basename,
                                                        args.ratio,
                                                        partition_name))

    partition_masks = get_partition_masks(full_dataset, args.ratio, rng)

    for partition_name, partition_mask in safe_izip(args.partition_names,
                                                    partition_masks):
        partition_path = get_partition_path(args.output_dir,
                                            args.input,
                                            partition_name)

        memmap = make_memmap_file(partition_path,
                                  numpy.count_nonzero(partition_mask),
                                  full_dataset.names,
                                  full_dataset.formats)


        for full_tensor, name, fmt in safe_izip(full_dataset.tensors,
                                                full_dataset.names,
                                                full_dataset.formats):
            partition_tensor = memmap[name]
            batch_slice = get_batch_slice(fmt, partition_mask)
            partition_tensor[...] = full_tensor[batch_slice]

def write_h5(full_dataset, args, rng):
    partition_masks = get_partition_masks(full_dataset, args.ratio, rng)
    partition_sizes = [numpy.count_nonzero(mask) for mask in partition_masks]

    basename = os.path.splitext(os.path.split(args.input)[1])[0]

    output_path = os.path.join(args.output_dir,
                               "{}_split_{}.h5".format(basename, args.ratio))

    h5_file = make_h5_file(output_path,
                           args.partition_names,
                           partition_sizes,
                           full_dataset.names,
                           full_dataset.formats)

    for partition_name, partition_mask in safe_izip(args.partition_names,
                                                    partition_masks):
        partition = h5_file['partitions'][partition_name]

        for full_tensor, name, fmt in safe_izip(full_dataset.tensors,
                                                full_dataset.names,
                                                full_dataset.formats):
            partition_tensor = partition[name]
            batch_slice = get_batch_slice(fmt, partition_mask)
            partition_tensor[...] = full_tensor[batch_slice]


def main():
    args = parse_args()

    if args.input.endswith('.h5'):
        full_dataset, = load_h5_dataset(args.input)
    else:
        full_dataset = MemmapDataset(args.input)

    rng = numpy.random.RandomState(args.seed)

    if args.output_format == 'npy':
        write_memmaps(full_dataset, args, rng)
    elif args.output_format == 'h5':
        write_h5(full_dataset, args, rng)
    else:
        raise ValueError("Expected --output to end in .h5 or .npy. "
                         "This should've been caught in parse_args. "
                         "This line should never have been reached.")


if __name__ == '__main__':
    main()
