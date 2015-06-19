#! /usr/bin/env python

import os
import argparse
from nose.tools import assert_equal
from simplelearn.data.h5_dataset import load_h5_dataset
from simplelearn.utils import safe_izip
from simplelearn.data.memmap_dataset import make_memmap_file


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Converts a H5Dataset's partitions to MemmapDatasets."))

    def h5_filename(arg):
        assert_equal(os.path.splitext(arg)[1], '.h5')
        return arg

    def npy_filename(arg):
        assert_equal(os.path.splitext(arg)[1], '.npy')
        return arg

    parser.add_argument('-i',
                        '--input',
                        type=h5_filename,
                        required=True,
                        help=("The .h5 file."))

    parser.add_argument('-o',
                        '--output',
                        type=npy_filename,
                        help=("The output file template. For example, "
                              "'-o dir/mnist.npy' will create two files: "
                              "dir/mnist_train.npy and dir/mnist_test.py"))



    result = parser.parse_args()

    if result.output is None:
        result.output = os.path.splitext(result.input)[0] + '.npy'

    return result


def main():
    args = parse_args()

    h5_datasets = load_h5_dataset(args.input)

    def get_output_filepath(input_name, partition_name):
        dirname, filename = os.path.split(input_name)
        basename, extension = os.path.splitext(filename)
        assert_equal(extension, '.h5')

        return os.path.join(dirname, "{}_{}{}".format(basename,
                                                      partition_name,
                                                      '.npy'))

    for dataset, partition_name in safe_izip(h5_datasets, h5_datasets._fields):
        output_filepath = get_output_filepath(args.input, partition_name)
        memmap = make_memmap_file(output_filepath,
                                  dataset.num_examples(),
                                  dataset.names,
                                  dataset.formats)

        memmap_tensors = [memmap[name] for name in dataset.names]
        for in_tensor, out_tensor in safe_izip(dataset.tensors,
                                               memmap_tensors):
            assert_equal(out_tensor.shape, in_tensor.shape)
            out_tensor[...] = in_tensor


        print("Wrote {}".format(output_filepath))

if __name__ == '__main__':
    main()
