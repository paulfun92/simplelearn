#! /usr/bin/env python

'''
Times copying batches of images to the GPU.
'''

from __future__ import print_function

import os
import sys
import argparse
from timeit import default_timer
import numpy
import theano


def parse_args():
    parser = argparse.ArgumentParser(
        description="Times copying batches of images to the GPU.")

    def positive_int(arg):
        arg = int(arg)
        if arg <= 0:
            print("Expected a positive int, got {}.".format(arg))
            sys.exit(1)

        return arg

    parser.add_argument("--image-shape",
                        default=(108, 108),
                        type=positive_int,
                        nargs=2,
                        metavar=('R', 'C'),
                        help="The shape of the images (# rows, # columns).")

    parser.add_argument("--batch-size",
                        default=128,
                        type=positive_int,
                        help="Number of images per batch")

    parser.add_argument("--batches-per-transfer",
                        default=10,
                        type=positive_int,
                        help="Number of batches to copy at a time.")


    parser.add_argument("--num-batches",
                        default=200,
                        type=positive_int,
                        help=("Total number of batches to transfer (should be "
                              "roughly an epoch's worth of batches)."))

    return parser.parse_args()

def main():
    args = parse_args()

    if theano.config.device[:3] != "gpu" or \
       os.environ['CUDA_LAUNCH_BLOCKING'] != '1':
        print(('Run this script as: CUDA_LAUNCH_BLOCKING=1 THEANO_FLAGS='
               '"device_gpu" {}'.format(sys.argv[0])))
        sys.exit(1)

    floatX = theano.config.floatX  # pylint: disable=no-member

    batches = numpy.zeros((args.batches_per_transfer,
                           args.batch_size,
                           args.image_shape[0],
                           args.image_shape[1],
                           3),
                          dtype=floatX)

    num_transfers = ((args.num_batches - 1) // args.batches_per_transfer) + 1

    gpu_memory = theano.shared(batches)

    start_time = default_timer()
    for _ in xrange(num_transfers):
        gpu_memory.set_value(batches)

    duration = default_timer() - start_time

    total_num_batches = num_transfers * args.batches_per_transfer

    print("Copied {} batches, {} at a time.".format(
        total_num_batches,
        args.batches_per_transfer))
    print("Total time: {} s".format(duration))
    print("Time per batch: {} s".format(duration / total_num_batches))


if __name__ == '__main__':
    main()
