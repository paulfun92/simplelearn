#! /usr/bin/env python

from __future__ import print_function

import os
import argparse
import numpy
from nose.tools import (assert_true,
                        assert_greater,
                        assert_greater_equal,
                        assert_less_equal)
from simplelearn.utils import safe_izip
from simplelearn.data.h5_dataset import make_h5_file, load_h5_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Splits a single-partition .h5 dataset into two "
                     "partitions of a new dataset (training and testing). "
                     "Saves the partitions to a new .h5 file."))

    def existing_h5_path(arg):
        assert_true(arg.endswith('.h5'))
        assert_true(os.path.isfile(arg))
        return arg

    def new_h5_path(arg):
        assert_true(arg.endswith('.h5'))
        parent_dir = os.path.dirname(os.path.abspath(arg))
        assert_true(os.path.isdir(parent_dir))
        return arg

    def fraction(arg):
        arg = float(arg)
        assert_greater_equal(arg, 0.0)
        assert_less_equal(arg, 1.0)
        return arg

    parser.add_argument("-i",
                        "--input",
                        type=existing_h5_path,
                        help="The .h5 file of a single-partition H5Dataset")

    parser.add_argument("-o",
                        "--output",
                        type=new_h5_path,
                        help="The .h5 file to create.")

    parser.add_argument("--training-ratio",
                        type=fraction,
                        help=("The fraction of data to use for the training "
                              "set."))

    parser.add_argument("--seed",
                        default=1234,
                        type=int,
                        help=("Seed for the RNG that chooses which examples "
                              "to use for the training and testing sets."))

    return parser.parse_args()


def main():
    args = parse_args()

    full_dataset, = load_h5_dataset(args.input)

    rng = numpy.random.RandomState(args.seed)

    num_examples = full_dataset.num_examples()
    num_training_examples = int(numpy.round(num_examples *
                                            args.training_ratio))
    assert_greater(num_training_examples, 0)

    num_testing_examples = num_examples - num_training_examples
    assert_greater(num_testing_examples, 0)

    training_example_indices = rng.choice(num_examples,
                                          size=num_training_examples,
                                          replace=False)
    training_mask = numpy.zeros(num_examples, dtype=bool)
    training_mask[training_example_indices] = True
    testing_mask = numpy.logical_not(training_mask)

    partition_names = ['train', 'test']

    print("Allocating output file...")

    h5_file = make_h5_file(args.output,
                           partition_names=partition_names,
                           partition_sizes=[num_training_examples,
                                            num_testing_examples],
                           tensor_names=full_dataset.names,
                           tensor_formats=full_dataset.formats)

    print("Copying data...")

    for (partition_name,
         examples_mask) in safe_izip(partition_names,
                                     [training_mask, testing_mask]):
        partition = h5_file['partitions'][partition_name]

        for (full_tensor,
             tensor_name,
             fmt) in safe_izip(full_dataset.tensors,
                               full_dataset.names,
                               full_dataset.formats):
            index = tuple(examples_mask if axis == 'b' else slice(None)
                          for axis in fmt.axes)

            partial_tensor = partition[tensor_name]
            partial_tensor[...] = full_tensor[index]

    print("Done. Wrote output to {}.".format(args.output))


if __name__ == '__main__':
    main()
