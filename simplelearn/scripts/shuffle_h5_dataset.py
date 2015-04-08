#! /usr/bin/env python

import os
import argparse
from nose.tools import assert_true, assert_equal
import numpy
from simplelearn.utils import safe_izip
from simplelearn.data.h5_dataset import load_h5_dataset


def main():
    def parse_args():
        parser = argparse.ArgumentParser(
            description=("Shuffles a .h5 dataset file in-place. \n"
                         "\n"
                         "Due to the limitations of numpy.shuffle, this can "
                         "only shuffle datasets where the batch axis is the "
                         "first axis for all tensors. Will throw an error and "
                         "do nothing if this is not the case.\n"
                         "\n"
                         "Shuffling on hard drives is very slow. Therefore "
                         "it's recommended that you first copy the dataset to "
                         "a SSD or other high-speed storage, if available."))

        def path_to_h5_file(arg):
            assert_true(os.path.isfile(arg))
            assert_equal(os.path.splitext(arg)[1], '.h5')

        parser.add_argument('-i',
                            '--input',
                            type=path_to_h5_file,
                            required=True,
                            help=("The .h5 file to shuffle in-place. It "
                                  "should have been created by "
                                  "simplelearn.data.make_h5_file()."))

        parser.add_argument('-s',
                            '--seed',
                            type=int,
                            default=1234,
                            help="RNG seed to use for shuffling")

        return parser.parse_args()

    args = parse_args()

    partitions = load_h5_dataset(args.input, mode='r+')

    for partition_index, partition in enumerate(partitions):
        for (tensor_index,
             fmt,
             tensor) in safe_izip(range(len(partition.tensors)),
                                  partition.formats,
                                  partition.tensors):
            if fmt.axes[0] != 'b':
                raise ValueError("Can't shuffle this dataset. Partition {}, "
                                 "tensor {} 's first axis is not the batch "
                                 "axis (axes = {}).".format(partition_index,
                                                            tensor_index,
                                                            str(fmt.axes)))

            rng = numpy.random.RandomState(args.seed)
            print("Shuffling partition {} tensor {}.".format(partition_index,
                                                             tensor_index))
            rng.shuffle(tensor)

if __name__ == '__main__':
    main()
