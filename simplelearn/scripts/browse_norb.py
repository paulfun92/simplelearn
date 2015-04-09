#! /usr/bin/env python

'''
Script for browsing a NORB dataset by label value.
'''

import sys
import argparse
import copy
import numpy
import matplotlib
from matplotlib import pyplot
from nose.tools import assert_false, assert_in, assert_equal
import theano
from simplelearn.data.norb import load_norb
from simplelearn.utils import safe_izip
from simplelearn.nodes import InputNode, FormatNode, RescaleImage, Lcn
from simplelearn.formats import DenseFormat

import pdb

def parse_args():
    parser = argparse.ArgumentParser(
        description="Browse NORB images by label.")

    parser.add_argument('--which-norb',
                        help="'big', 'small', or a path to a .h5 file.'")

    parser.add_argument('--which-set',
                        required=True,
                        help=("Which partition to use (typically 'test' "
                              "or 'train')."))

    parser.add_argument('--lcn',
                        default=False,
                        action='store_true',
                        help="Preprocess images with LCN.")

    return parser.parse_args()


class LabelIndexMap(object):
    '''
    Maps between 5-D NORB labels and dense 5-D indices.
    '''

    def __init__(self, dataset_labels):
        assert_equal(len(dataset_labels.shape), 2)
        labels_5d = dataset_labels[:, :5]

        self.label5d_values = tuple(sorted(frozenset(label_column))
                                    for label_column in labels_5d.T)

        indices = self.label5d_to_index(labels_5d)

        self._index_to_rows = dict()
        for row, index in enumerate(indices):
            key = tuple(index)
            if key not in self._index_to_rows:
                self._index_to_rows[key] = ([row], [0])
            else:
                self._index_to_rows[key][0].append(row)

        self.index_sizes = indices.max(axis=0) + 1

    def label5d_to_index(self, labels):
        assert_in(len(labels.shape), (1, 2))

        was_1D = False
        if len(labels.shape) == 1:
            labels = labels[numpy.newaxis, :]
            was_1D = True

        labels = numpy.asarray(labels[:, :5])

        indices = numpy.empty_like(labels)
        indices[...] = -1

        for (label5d_values,
             label_column,
             index_column) in safe_izip(self.label5d_values,
                                        labels.T,
                                        indices.T):

            for ind, value in enumerate(label5d_values):
                mask = (label_column == value)
                index_column[mask] = ind

        assert_false((indices == -1).any())

        if was_1D:
            assert_equal(indices.shape[0], 1)
            return indices[0, :]
        else:
            return indices

    def index_to_label_5d(self, indices):
        assert_in(len(indices.shape), (1, 2))

        indices = numpy.asarray(indices)

        was_1D = False
        if len(indices.shape) == 1:
            indices = indices[numpy.newaxis, :]
            was_1D = True

        assert_equal(indices.shape[1], 5)

        labels = numpy.zeros_like(indices)

        for (index_column,
             label_column,
             label5d_values) in safe_izip(indices.T,
                                          labels.T,
                                          self.label5d_values):
            label_column[:] = label5d_values[index_column]

        if was_1D:
            return labels[0, :]
        else:
            return labels

    def index_to_rows(self, indices):
        assert_equal(indices.shape, (5, ))
        key = tuple(indices)
        return self._index_to_rows.get(key, (None, -1))


def main():
    args = parse_args()

    dataset = load_norb(args.which_norb, args.which_set)
    dataset_labels = dataset.tensors[1]
    assert_in(dataset_labels.shape[1], (5, 11))

    label_index_map = LabelIndexMap(dataset_labels)

    # The multidimensional index. Corrsponds to the first 5 label dimensions.
    # index[i] indexes into the list of all known values of labels[i].
    # Multiple images can have the label pointed to by index[i]. A list of
    # dataset rows corresponding to those images is given by
    # label_index_map.index_to_rows(index). This returns (rows, row_index),
    # where rows is a list of dataset rows that fit the label pointed to by
    # index, and rows_index is an index into rows, pointing to the last
    # such image displayed.
    index = numpy.zeros(5, dtype=int)

    # Some labels have -1 as a possible value, meaning "not applicable".
    # For such labels, place the index to point at the first real value.
    for label_dim in range(len(index)):
        if label_index_map.label5d_values[label_dim][0] == -1:
            index[label_dim] = 1

    # The current index dimension being edited.
    index_dim = [0]

    num_axes = 3 if 's' in dataset.formats[0].axes else 2
    figure, all_axes = pyplot.subplots(1,
                                       num_axes,
                                       squeeze=True,
                                       figsize=(4 * num_axes + 2 , 4))

    # Hides x and y tick marks from all axes
    for axes in all_axes:
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)

    window_title = ("{} NORB, {}ing set".format(args.which_norb,
                                                args.which_set))
    figure.canvas.set_window_title(window_title)

    label_names = ['category',
                   'instance',
                   'elevation',
                   'azimuth',
                   'lighting']

    categories = ['animal', 'human', 'plane', 'truck', 'car']

    converters = [lambda x: categories[x],
                  lambda x: None if x == -1 else x,
                  lambda x: None if x == -1 else 30 + x * 5,
                  lambda x: None if x == -1 else x * 10,
                  lambda x: None if x == -1 else x]

    if dataset_labels.shape[1] == 11:
        label_names.extend(['horiz. shift',
                            'vert. shift',
                            'lumination change',
                            'contrast',
                            'object scale',
                            'rotation (deg)'])
        categories.append('blank')

        contrasts = (0.8, 1.3)
        scales = (0.78, 1.0)

        converters.extend([lambda x: x,
                           lambda x: x,
                           lambda x: x,
                           lambda x: contrasts[x],
                           lambda x: scales[x],
                           lambda x: x])
    def draw_menu():
        lines = []

        rows, rows_index = label_index_map.index_to_rows(index)

        if rows is not None:
            row = rows[rows_index[0]]
            label = dataset_labels[row]

            for name, value, converter in safe_izip(label_names[:5],
                                                    label[:5],
                                                    converters[:5]):
                lines.append('{}: {}'.format(name, converter(value)))

            lines.append("image: {} of {}".format(rows_index[0] + 1,
                                                  len(rows)))
            lines.append('')

            if dataset_labels.shape[1] == 11:
                for name, value, converter in safe_izip(label_names[5:],
                                                        label[5:],
                                                        converters[5:]):
                    lines.append('{}: {}'.format(name, converter(value)))
        else:
            label_5d = label_index_map.index_to_label_5d(index)

            for name, value, converter in safe_izip(label_names[:5],
                                                    label_5d,
                                                    converters[:5]):
                lines.append('{}: {}'.format(name, converter(value)))

            lines.append("image: (no such image)")
            lines.append('')

            if dataset_labels.shape[1] == 11:
                for name in label_names[5:]:
                    lines.append('{}: N/A'.format(name))

        lines[index_dim[0]] = "==> " + lines[index_dim[0]]

        # "transAxes": 0, 0 = bottom-left, 1, 1 at upper-right.
        text_axes = all_axes[0]
        text_axes.clear()
        text_axes.text(0.1,  # x
                       0.5,  # y
                       '\n'.join(lines),
                       verticalalignment='center',
                       transform=text_axes.transAxes)

    dataset_images = dataset.tensors[0]
    image_format = dataset.formats[0]

    def make_lcn_func():
        '''
        Returns a function that takes a NORB image and applies LCN to it.
        '''

        image_node = InputNode(image_format)
        assert_in(image_node.output_format.axes,
                  (('b', 's', '0', '1'), ('b', '0', '1')))

        if 's' in image_node.output_format.axes:
            b01_shape = (-1, ) + image_node.output_format.shape[2:]
            b01_format = DenseFormat(axes=('b', '0', '1'),
                                     shape=b01_shape,
                                     dtype=image_node.output_format.dtype)
            b01_node = FormatNode(image_node,
                                  b01_format,
                                  axis_map={('b', 's'): 'b'})
        else:
            b01_node = image_node

        b01_shape = b01_node.output_format.shape
        bc01_format = DenseFormat(axes=('b', 'c', '0', '1'),
                                  shape=(-1, 1) + b01_shape[1:],
                                  dtype=b01_node.output_format.dtype)
        # image_shape = image_node.output_format.shape
        # bc01_shape = (-1, 1, image_shape[0], image_shape[1])
        # bc01_format = DenseFormat(axes=('b', 'c', '0', '1'),
        #                           shape=bc01_shape,
        #                           dtype=image_node.output_format.dtype)

        bc01_node = FormatNode(b01_node,
                               bc01_format,
                               axis_map={'b': ('b', 'c')})

        float_image_node = RescaleImage(bc01_node)
        lcn = Lcn(float_image_node)

        # Re-shape to original shape
        output_format = copy.deepcopy(image_node.output_format)
        output_format.dtype = lcn.output_format.dtype
        if 's' in image_node.output_format.axes:
            output_axis_map = {('b', 'c') : ('b', 's')}
        else:
            output_axis_map = {('b', 'c') : 'b'}

        output_node = FormatNode(lcn,
                                 output_format,
                                 axis_map=output_axis_map)

        batch_function = theano.function([image_node.output_symbol],
                                         output_node.output_symbol)

        def single_example_function(image):
            '''
            Takes a single example and adds a singleton batch axis to it before
            feeding it to batch_function.
            '''
            image = image[numpy.newaxis, ...]
            result_batch = batch_function(image)
            return result_batch[0]

        return single_example_function

    lcn = make_lcn_func() if args.lcn else None

    def draw_images():
        def draw_image_impl(image, axes):
            if args.lcn:
                axes.imshow(image, cmap='gray')
            else:
                axes.imshow(image,
                            cmap='gray',
                            norm=matplotlib.colors.NoNorm())


        rows, rows_index = label_index_map.index_to_rows(index)

        image_axes = all_axes[1:]  # could be 1 or two axes

        if rows is None:
            for axes in image_axes:
                axes.clear()
        else:
            row = rows[rows_index[0]]
            image = dataset_images[row]
            if args.lcn:
                image = lcn(image)

            if 's' in image_format.axes:
                assert_equal(image_format.axes.index('s'), 1)
                for sub_image, axes in safe_izip(image, image_axes):
                    draw_image_impl(sub_image, axes)
            else:
                draw_image_impl(image, all_axes[1])


    def on_key_press(event):
        """
        Keyboard callback.
        """
        def add_mod(arg, step, size):
            """
            Increments arg by step, and loops if it exits [0, size-1].
            """
            return (arg + size + step) % size

        def incr_index_dim(step):
            """
            Moves the cursor up (step=1) or down (step=-1).
            """
            assert_in(step, (0, -1, 1))

            # Add one for the image index
            num_dimensions = len(label_index_map.index_sizes) + 1
            index_dim[0] = add_mod(index_dim[0], step, num_dimensions)

        def incr_index(step):
            """
            Increments/decrements the label currently pointed to.
            """
            assert_in(step, (0, -1, 1))

            if index_dim[0] == 5:  # i.e. the image index
                rows, rows_index = label_index_map.index_to_rows(index)
                if rows is not None:
                    rows_index[0] = add_mod(rows_index[0],
                                            step,
                                            len(rows))
            else:
                # increment/decrement one of the indices
                index_size = label_index_map.index_sizes[index_dim[0]]
                index[index_dim[0]] = add_mod(index[index_dim[0]],
                                              step,
                                              index_size)

        # Disables left/right key if we're currently showing a blank,
        # and the current index type is neither 'category' (0) nor
        # 'image number' (5)
        disable_left_right = False


        if index[0] == 5 and index_dim[0] not in (0, 5):
            # Current class is 'blank', and we're trying to adjust something
            # other than class or image number
            disable_left_right = True
        elif (label_index_map.index_to_rows(index)[0] is None
              and index_dim[0] == 5):
            # We're trying to adjust the image number when there are no images.
            disable_left_right = True

        if event.key == 'up':
            incr_index_dim(-1)
            draw_menu()
        elif event.key == 'down':
            incr_index_dim(1)
            draw_menu()
        elif event.key == 'q':
            sys.exit(0)
        elif event.key in ('left', 'right') and not disable_left_right:
            incr_index(-1 if event.key == 'left' else 1)
            draw_menu()
            draw_images()
        else:
            return  # nothing changed, so don't bother drawing

        figure.canvas.draw()


    figure.canvas.mpl_connect('key_press_event', on_key_press)

    draw_menu()
    draw_images()
    pyplot.show()


if __name__ == '__main__':
    main()
