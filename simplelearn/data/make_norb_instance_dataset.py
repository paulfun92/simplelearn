#! /usr/bin/env python

import os
import argparse
from collections import Sequence
import numpy
import h5py
from nose.tools import (assert_equal,
                        assert_in,
                        assert_true,
                        assert_is_instance,
                        assert_greater,
                        assert_greater_equal)
import simplelearn
from simplelearn.utils import (safe_izip,
                               assert_integer,
                               assert_all_equal)

from simplelearn.formats import DenseFormat
from simplelearn.data.norb import load_norb
from simplelearn.data.hdf5_dataset import add_tensor

import pdb


def make_instance_dataset(norb,
                          test_elevation_stride,
                          test_azimuth_stride,
                          objects=None,
                          rng_seed=None):
    '''
    Creates an instance recognition dataset NORB images.

    Parameters
    ----------
    test_elevation_stride: int
      Use every M'th elevation as a test image.

    test_azimuth_stride: int
      Use every N'th azimuth as a test image.

    objects: Sequence
      [(c0, i0), (c1, i1), ..., (cN, iN)]
      Each (cx, ix) pair specifies an object to include, by their
      class and instance labels cx and ix.

    The test set consists of all images that satisfy both the
    test_elevation_stride and test_azimuth_stride. All other images are used
    for the training set.

    Only the left stereo images are used.
    '''
    assert_in(which_norb, ('big', 'small'))

    assert_integer(test_elevation_stride)
    assert_greater(test_elevation_stride, 0)

    assert_integer(test_azimuth_stride)
    assert_greater(test_azimuth_stride, 0)

    if objects is not None:
        assert_is_instance(objects, Sequence)
        for id_pair in objects:
            assert_equal(len(id_pair), 2)
            category, instance = id_pair
            assert_integer(category)
            assert_greater_equal(category, 0)
            assert_integer(instance)
            assert_greater_equal(instance, 0)

    if rng_seed is None:
        rng_seed = 3252
    else:
        assert_integer(rng_seed)
        assert_greater_equal(rng_seed, 0)

    a_norb, b_norb = (load_norb(which_norb, s) for s in ('train', 'test'))

    def get_row_indices(labels,
                        test_elevation_stride,
                        test_azimuth_stride,
                        objects):
        '''
        Returns row masks for training and testing data.
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

    a_train_indices, a_test_indices = get_row_indices(a_norb._tensors[1],
                                                      test_elevation_stride,
                                                      test_azimuth_stride,
                                                      objects)

    b_train_indices, b_test_indices = get_row_indices(b_norb._tensors[1],
                                                      test_elevation_stride,
                                                      test_azimuth_stride,
                                                      objects)

    a_size = a_norb._tensors[0].shape[0]
    b_train_indices += a_size
    b_test_indices += a_size

    rng = numpy.random.RandomState(rng_seed)

    def concat_and_shuffle(a_indices, b_indices, rng):
        assert_equal(a_indices.ndim, 1)
        assert_equal(b_indices.ndim, 1)
        indices = numpy.concatenate((a_indices, b_indices), axis=0)
        rng.shuffle(indices)
        return indices

    train_indices = concat_and_shuffle(a_train_indices, b_train_indices, rng)
    test_indices = concat_and_shuffle(a_test_indices, b_test_indices, rng)

    def create_hdf_filepath(which_norb,
                            test_elevation_stride,
                            test_azimuth_stride,
                            objects):
        '''
        Creates an hdf filepath based on the args.

        For --which-norb big --elevation-stride 2, --azimuth-stride 1:
          <data_dir>/big_norb_instance/e2_a1_all.h5

        For same as above, but with --objects 1 2 3 7 4 1
          <data_dir>/big_norb_instance/e2_a1_1-2_3-7_4-1.h5
        '''
        norb_directory = os.path.join(simplelearn.data.data_path,
                                      '%s_norb_instance' % which_norb)
        if not os.path.isdir(norb_directory):
            os.mkdir(norb_directory)

        filename = "e%02d_a%02d_o_" % (test_elevation_stride,
                                       test_azimuth_stride)

        if objects is None:
            filename = filename + 'all'
        else:
            for id_pair in objects:
                filename = filename + "%d-%d" % tuple(id_pair)

        filename = filename + '.h5'

        return os.path.join(norb_directory, filename)

    hdf_path = create_hdf_filepath(which_norb,
                                   test_elevation_stride,
                                   test_azimuth_stride,
                                   objects)

    assert_all_equal(a_norb._names, b_norb._names)
    assert_all_equal(a_norb._formats, b_norb._formats)

    def copy_examples(a_tensor, b_tensor, indices, hdf_tensor):
        a_indices = indices[indices < a_size]
        b_indices = indices[indices >= a_size]
        hdf_tensor[a_indices, ...] = a_tensor[a_indices]
        hdf_tensor[b_indices, ...] = b_tensor[b_indices - a_size]

    example_shapes = [a_norb._tensors[0].shape[2:],
                      a_norb._tensors[1].shape[1:]]

    # Remove the right stereo images
    def get_mono_tensors(norb):
        return [norb._tensors[0][:, 0, :, :], norb._tensors[1]]

    a_tensors, b_tensors = (get_mono_tensors(n) for n in (a_norb, b_norb))

    def get_mono_formats(norb):
        stereo_format = norb._formats[0]
        assert_equal(stereo_format.axes, ('b', 's', '0', '1'))
        mono_format = DenseFormat(axes=('b', '0', '1'),
                                  shape=(-1, ) + stereo_format.shape[1:],
                                  dtype=stereo_format.dtype)

        return [mono_format, norb._formats[1]]

    formats = get_mono_formats(a_norb)

    with h5py.File(hdf_path, 'w') as hdf_file:
        for group_name, indices in safe_izip(['train', 'test'],
                                             [train_indices, test_indices]):
            group = hdf_file.Group(group_name)

            num_examples = len(indices)
            for (tensor_a,
                 tensor_b,
                 tensor_name,
                 fmt,
                 example_shape) in safe_izip(a_tensors,
                                             b_tensors,
                                             a_norb._names,
                                             formats,
                                             example_shapes):
                shape = (num_examples, ) + example_shape
                hdf_tensor = add_tensor(tensor_name,
                                        shape,
                                        fmt.dtype,
                                        fmt.axes,
                                        group)
                copy_examples(tensor_a, tensor_b, indices, hdf_tensor)


def _main():
    def parse_args():
        parser = argparse.ArgumentParser(
            description=("Merges the test and training sets of a "
                         "NORB dataset, shuffles them, and re-splits "
                         "into testing and training data according "
                         "to camera angle. If the NORB dataset is "
                         "stereo, this will use only the left stereo "
                         "images."))

        def norb_path(arg):
            if arg == 'big':
                return load_norb(

        parser.add_argument('-i',
                            "--input",
                            type=norb_path,
                            required=True,
                            help=("'big', 'small', or the .h5 file "
                                  "of the NORB dataset."))

        parser.add_argument("-e",
                            "--test-elevation-stride",
                            required=True,
                            metavar='M',
                            help=("Select every M'th elevation for "
                                  "the test set."))

        parser.add_argument('-a',
                            '--test-azimuth-stride',
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

        parser.add_argument('-o',
                            '--output',
                            required=True,
                            help=('.h5 file to save to'))

        args = parser.parse_args()

        if args.objects is not None:
            assert_equal(len(args.objects) % 2, 0)
            args.objects = numpy.asarray(args.objects).reshape(-1, 2)

        return args

    args = parse_args()

    make_instance_dataset(norb_dataset,
                          args.test_elevation_stride,
                          args.test_azimuth_stride,
                          args.objects,
                          args.output)


if __name__ == '__main__':
    _main()
