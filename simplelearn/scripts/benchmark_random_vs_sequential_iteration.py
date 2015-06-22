#! /usr/bin/env python

from __future__ import print_function

import argparse
import os
from timeit import default_timer
from nose.tools import assert_greater
import numpy
from simplelearn.data.h5_dataset import load_h5_dataset
from simplelearn.data.memmap_dataset import MemmapDataset
from simplelearn.utils import safe_izip, human_readable_duration

import pdb

def parse_args():
    parser = argparse.ArgumentParser(
        description=("Benchmarks running a loop through the big "
                     "NORB dataset, using sequential and random "
                     "iterators."))

    def positive_int(arg):
        arg = int(arg)
        assert_greater(arg, 0)
        return arg

    parser.add_argument("-i",
                        "--input",
                        help="Path to .h5 or .npy file of dataset.")

    parser.add_argument("-b",
                        "--batch-size",
                        default=128,
                        type=positive_int,
                        help="Batch size (default=128).")

    parser.add_argument("-e",
                        "--num-epochs",
                        default=5,
                        type=positive_int,
                        help="Number of epochs (default=5).")

    parser.add_argument("-m",
                        "--memory",
                        action="store_true",
                        help="load dataset into memory before benchmarking.")

    return parser.parse_args()

def main():

    def load_training_dataset(path):
        if path.endswith('.h5'):
            return load_h5_dataset(args.input)[0]
        elif path.endswith('.npy'):
            return MemmapDataset(path)
        else:
            raise ValueError("Unexpected file extension {}".format(
                os.path.splitext(path)[1]))


    args = parse_args()

    dataset = load_training_dataset(args.input)
    if args.memory:
        start_time = default_timer()
        dataset = dataset.load_to_memory()
        print("Loaded dataset to memory in {}".format(
            human_readable_duration(default_timer() - start_time)))

    sequential_iter = dataset.iterator(iterator_type='sequential',
                                       batch_size=args.batch_size)
    random_iter = dataset.iterator(iterator_type='random',
                                   batch_size=args.batch_size,
                                   rng=numpy.random.RandomState(4829))

    for iterator, iterator_type in safe_izip((sequential_iter, random_iter),
                                             ('sequential', 'random')):
        print("Timing {} iterator:".format(iterator_type))

        epochs_seen = 0
        old_num_batches = 0
        num_batches = 0
        start_time = default_timer()

        while epochs_seen < args.num_epochs:
            iterator.next()
            num_batches += 1

            if iterator.next_is_new_epoch():
                epochs_seen += 1
                print("  completed epoch {}, which had {} batches.".format(
                    epochs_seen,
                    num_batches - old_num_batches))
                old_num_batches = num_batches


        duration = default_timer() - start_time

        print(("{} epochs of {} iterator: {}, ({} per batch of {} samples "
               "each.").format(epochs_seen,
                               iterator_type,
                               human_readable_duration(duration),
                               human_readable_duration(duration / num_batches),
                               args.batch_size))

if __name__ == '__main__':
    main()
