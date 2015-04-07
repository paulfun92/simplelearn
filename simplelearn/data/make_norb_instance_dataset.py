#! /usr/bin/env python

import os
import argparse
from collections import Sequence
import numpy
from nose.tools import (assert_equal,
                        assert_is_instance,
                        assert_greater,
                        assert_greater_equal)
import simplelearn
from simplelearn.utils import (safe_izip,
                               assert_integer,
                               assert_all_equal,
                               assert_all_true,
                               assert_all_integers,
                               assert_all_greater_equal)

from simplelearn.formats import DenseFormat
from simplelearn.data.dataset import Dataset
from simplelearn.data.norb import load_norb
from simplelearn.data.h5_dataset import make_h5_file

import pdb


def make_instance_dataset(norb_name,
                          a_norb,
                          b_norb,
                          test_elevation_stride,
                          test_azimuth_stride,
                          objects=None,
                          rng_seed=None):
    '''
    Creates instance recognition datasets from category recognition datasets.

    Merges two category recognition datasets (with disjoint object instances),
    and re-partitions them into instance recognition datasets (with disjoint
    camera views).

    The instance recognition dataset consists of a train and test set.

    All objects not selected by <objects> are ignored.

    Of the remaining images, he test set consists of all images that satisfy
    both the test_elevation_stride and test_azimuth_stride. The other
    images are used for the training set.

    If the category datset is in stereo, only the left stereo images are used.

    Parameters
    ----------
    norb_name: str
      The name of the category recognition dataset (e.g. 'big_norb'). Used to
      build the name of the instance recognition dataset. Alphanumeric
      characters and '_' only.

    a_norb: NORB Dataset
      One of the category recognition datasets (i.e. training set).

    b_norb: NORB Dataset
      The other category recognition dataset (i.e. testing set).

    test_elevation_stride: int
      Use every M'th elevation as a test image.

    test_azimuth_stride: int
      Use every N'th azimuth as a test image.

    objects: Sequence
      [(c0, i0), (c1, i1), ..., (cN, iN)]
      Each (cx, ix) pair specifies an object to include, by their
      class and instance labels cx and ix.

    rng_seed: int
      Used to seed the RNG for shuffling examples.

    Returns
    -------
    rval: str
      The path to the newly created .h5 file.
    '''

    assert_is_instance(norb_name, basestring)
    assert_all_true(c.isalnum() or c == '_' for c in norb_name)

    assert_is_instance(a_norb, Dataset)
    assert_is_instance(b_norb, Dataset)
    assert_all_equal(a_norb.names, b_norb.names)
    assert_all_equal(a_norb.formats, b_norb.formats)

    assert_integer(test_elevation_stride)
    assert_greater(test_elevation_stride, 0)

    assert_integer(test_azimuth_stride)
    assert_greater(test_azimuth_stride, 0)

    if objects is not None:
        assert_is_instance(objects, Sequence)
        for id_pair in objects:
            assert_equal(len(id_pair), 2)
            assert_all_integers(id_pair)
            assert_all_greater_equal(id_pair, 0)

    if rng_seed is None:
        rng_seed = 3252
    else:
        assert_integer(rng_seed)
        assert_greater_equal(rng_seed, 0)

    #
    # Done sanity-checking args
    #

    def get_row_indices(labels,
                        test_elevation_stride,
                        test_azimuth_stride,
                        objects):
        '''
        Returns row indices or training and testing sets.
        '''

        logical_and = numpy.logical_and

        if objects is not None:
            objects = numpy.asarray(objects)
            object_mask = (labels[:, 0:2] == objects).all(axis=1)
        else:
            object_mask = numpy.ones(labels.shape[0], dtype=bool)

        test_mask = logical_and(object_mask,
                                (labels[:, 3] % test_elevation_stride) == 0)

        test_mask = logical_and(
            test_mask,
            (labels[:, 4] % (test_azimuth_stride * 2)) == 0)

        train_mask = logical_and(object_mask, numpy.logical_not(test_mask))

        return tuple(numpy.nonzero(m)[0] for m in (train_mask, test_mask))

    a_train_indices, a_test_indices = get_row_indices(a_norb.tensors[1],
                                                      test_elevation_stride,
                                                      test_azimuth_stride,
                                                      objects)

    b_train_indices, b_test_indices = get_row_indices(b_norb.tensors[1],
                                                      test_elevation_stride,
                                                      test_azimuth_stride,
                                                      objects)

    def create_h5_filepath(norb_name,
                           test_elevation_stride,
                           test_azimuth_stride,
                           objects):
        '''
        Creates an hdf filepath based on the args.

        For which-norb: "big_norb", elevation_stride: 2, azimuth_stride: 1:
          <data_dir>/big_norb_instance/e2_a1_all.h5

        For same as above, but with objects: [[1, 2], [3, 7], [4, 1]]
          <data_dir>/big_norb_instance/e2_a1_1-2_3-7_4-1.h5
        '''
        output_dir = os.path.join(simplelearn.data.data_path,
                                  '{}_norb_instance'.format(norb_name))

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        filename = "e{:02d}_a{:02d}_o_".format(test_elevation_stride,
                                               test_azimuth_stride)

        if objects is None:
            filename = filename + 'all'
        else:
            for id_pair in objects:
                filename = filename + "%d-%d" % tuple(id_pair)

        filename = filename + '.h5'

        return os.path.join(output_dir, filename)

    h5_path = create_h5_filepath(norb_name,
                                 test_elevation_stride,
                                 test_azimuth_stride,
                                 objects)

    def get_mono_format(input_image_format):
        axes = input_image_format.axes
        shape = input_image_format.shape

        if 's' in axes:
            s_index = axes.index('s')
            axes = list(axes)
            del axes[s_index]

            shape = list(shape)
            del shape[s_index]

        return DenseFormat(axes=axes,
                           shape=shape,
                           dtype=input_image_format.dtype)

    mono_image_format = get_mono_format(a_norb.formats[0])
    label_format = a_norb.formats[1]
    partition_names = ['train', 'test']
    partition_sizes = [len(a_train_indices) + len(b_train_indices),
                       len(a_test_indices) + len(b_test_indices)]
    train_indices = (a_train_indices, b_train_indices)
    test_indices = (a_test_indices, b_test_indices)

    with make_h5_file(h5_path,
                      partition_names,
                      partition_sizes,
                      a_norb.names,
                      [mono_image_format, label_format]) as h5_file:
        '''
        Creates a .h5 file, copies repartitioned data into it, and shuffles.
        '''
        partitions = h5_file['partitions']

        for partition_name, (a_indices, b_indices) \
            in safe_izip(partition_names, [train_indices, test_indices]):

            partition = partitions[partition_name]

            a_images = a_norb.tensors[0]
            b_images = b_norb.tensors[0]
            out_images = partition['images']

            print("Copying {} partition.".format(partition_name))

            if 's' in a_norb.formats[0].axes:
                assert_equal(a_norb.formats[0].axes.index('s'), 1)

                out_images[:len(a_indices), ...] = a_images[a_indices, 0, ...]
                out_images[len(a_indices):, ...] = b_images[b_indices, 0, ...]
            else:
                out_images[:len(a_indices), ...] = a_images[a_indices, ...]
                out_images[len(a_indices):, ...] = b_images[b_indices, ...]

            a_labels = a_norb.tensors[1]
            b_labels = b_norb.tensors[1]
            out_labels = partition['labels']

            out_labels[:len(a_indices), :] = a_labels[a_indices, :]
            out_labels[len(a_indices):, :] = b_labels[b_indices, :]

            # Shuffles the output images and labels in the same way
            print("Shuffling...")
            for out_tensor in (out_images, out_labels):
                # same seed for each loop means same shuffle order
                rng = numpy.random.RandomState(rng_seed)
                rng.shuffle(out_tensor)

    return h5_path


def main():

    def parse_args():
        parser = argparse.ArgumentParser(
            description=("Merges the test and training sets of a "
                         "NORB dataset, shuffles them, and re-splits "
                         "into testing and training data according "
                         "to camera angle. If the NORB dataset is "
                         "stereo, this will use only the left stereo "
                         "images."))

        parser.add_argument('-i',
                            "--input",
                            required=True,
                            help=("'big', 'small', or the .h5 file "
                                  "of the NORB dataset."))

        def positive_int(arg):
            arg = int(arg)
            assert_greater(arg, 0)
            return arg

        parser.add_argument("-e",
                            "--test-elevation-stride",
                            type=positive_int,
                            required=True,
                            metavar='M',
                            help=("Select every M'th elevation for "
                                  "the test set."))

        parser.add_argument('-a',
                            '--test-azimuth-stride',
                            type=positive_int,
                            required=True,
                            metavar='N',
                            help=("Select every N'th azimuth for the "
                                  "test set"))

        def non_negative_int(arg):
            arg = int(arg)
            assert_greater_equal(arg, 0)
            return arg

        parser.add_argument('--objects',
                            type=non_negative_int,
                            nargs='+',
                            default=None,
                            help=("Objects to include in the "
                                  "instance dataset. Omit to include "
                                  "all objects. Otherwise, provide "
                                  "a sequence of integers c1 i1 c2 "
                                  "i2 ... cZ nZ, which specify each "
                                  "object using category and "
                                  "instance labels cx ix"))

        def ends_with_h5(arg):
            assert_equal(os.path.splitext(arg)[1], '.h5')
            return arg

        # parser.add_argument('-o',
        #                     '--output',
        #                     type=ends_with_h5,
        #                     required=True,
        #                     help=('.h5 file to save to'))

        args = parser.parse_args()

        if args.objects is not None:
            assert_equal(len(args.objects) % 2, 0)
            args.objects = numpy.asarray(args.objects).reshape(-1, 2)

        return args

    args = parse_args()

    norb_datasets = load_norb(args.input)
    if args.input in ('big', 'small'):
        norb_name = args.input + "_norb"
    else:
        # strip dir and .h5 extension from args.input
        norb_name = os.path.splitext(os.path.split(args.input)[1])[0]

    output_path = make_instance_dataset(norb_name,
                                        norb_datasets[0],
                                        norb_datasets[1],
                                        args.test_elevation_stride,
                                        args.test_azimuth_stride,
                                        args.objects)

    print("Saved instance dataset to '{}'.".format(output_path))


if __name__ == '__main__':
    main()
