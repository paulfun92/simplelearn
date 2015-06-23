#! /usr/bin/env python

from __future__ import print_function

import argparse
import os
from timeit import default_timer
from nose.tools import assert_greater
import numpy
import theano
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

    parser.add_argument("-g",
                        "--num-gpu-batches",
                        default=None,
                        type=int,
                        help="# of batches to transfer to GPU at a time.")

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

    batch_queues = []
    gpu_batch_queues = []
    batch_symbols = []
    for fmt in [i.output_format for i in random_iter.make_input_nodes()]:
        batches = fmt.make_batch(is_symbolic=False, batch_size=args.batch_size)
        batches.resize((args.num_gpu_batches, ) + batches.shape)
        # batches = numpy.outer(numpy.ones(args.num_gpu_batches), batch)
        batch_queues.append(batches)
        gpu_batch_queues.append(theano.shared(batch_queues[-1]))
        batch_symbols.append(fmt.make_batch(is_symbolic=True))

    # batch_number_symbol = theano.scalar.iscalar()
    # gpu_batches = [q[batch_number_symbol, ...] for q in gpu_batch_queues]
    # theano_function = theano.function([theano.scalar.iscalar()],
    #                                   batch_symbols,
    #                                   givens=dict(safe_izip(batch_symbols,
    #                                                         gpu_batches)))

    for iterator, iterator_type in safe_izip((sequential_iter, random_iter),
                                             ('sequential', 'random')):
        print("Timing {} iterator:".format(iterator_type))

        epochs_seen = 0
        old_num_batches = 0
        num_batches = 0
        start_time = default_timer()

        while epochs_seen < args.num_epochs:
            batches = iterator.next()
            num_batches += 1

            for batch, batch_queue in safe_izip(batches, batch_queues):
                queue_index = (num_batches - 1) % args.num_gpu_batches
                batch_queue[queue_index, ...] = batch

            if args.num_gpu_batches is not None and \
               (num_batches % args.num_gpu_batches) == 0:
                for (batch_queue,
                     gpu_batch_queue) in safe_izip(batch_queues,
                                                   gpu_batch_queues):
                    gpu_batch_queue.set_value(batch_queue)

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
