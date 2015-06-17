#! /usr/bin/env python

from __future__ import print_function

import argparse
from timeit import default_timer
from nose.tools import assert_greater
import numpy
# from simplelearn.data.norb import load_norb
# from simplelearn.data.mnist import load_mnist
from simplelearn.data.h5_dataset import load_h5_dataset
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
                        help="Path to .h5 file of dataset.")

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

    return parser.parse_args()

def main():

    args = parse_args()

    # dataset = load_mnist()[1]
    #dataset = load_norb(which_norb='big')[0]
    dataset = load_h5_dataset(args.input)[0]

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
