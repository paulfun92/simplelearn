#! /usr/bin/env python

'''
Script for browsing a NORB dataset by label value.
'''

import sys
import argparse
import numpy
import matplotlib
from matplotlib import pyplot
from nose.tools import assert_false, assert_in, assert_equal
from simplelearn.data.norb import load_norb
from simplelearn.utils import safe_izip

class LabelIndexMap(object):
    '''
    Maps between NORB labels and dense 5-D indices.
    '''

    def __init__(self, dataset_labels):
        dataset_labels = numpy.asarray(dataset_labels)
        self.label_values = tuple(sorted(frozenset(label_column))
                                  for label_column in dataset_labels.T)

        indices = self.label_to_index(dataset_labels)

        self._index_to_rows = dict()
        for row, index in enumerate(indices):
            key = tuple(index)
            if key not in self._index_to_rows:
                self._index_to_rows[key] = ([row], [0])
            else:
                self._index_to_rows[key][0].append(row)

        self.index_sizes = indices.max(axis=0) + 1

        # # Maps index to a list of dataset row indices.
        # def make_index_to_rows(labels):
        #     '''
        #     Returns a dict that maps index to (rows, row_index).
        #     '''
        #     dense_labels = labels[:5]
        #     assert_is_instance(dense_labels, numpy.ndarray)

        #     # Divide azimuth label by 2 to make them dense.
        #     assert_true((numpy.mod(dense_labels[:, 3], 0) == 0).all())
        #     dense_labels[:, 3] /= 2

        #     for c, label_column in enumerate(dense_labels.T):
        #         index_sizes[c] = label_column.max() + 1

        #         # check that dense_labels are actually dense
        #         assert_equal(sorted(frozenset(label_column)),
        #                      range(index_sizes[c]))

        #     result = dict()
        #     for row, dense_label in enumerate(dense_labels):
        #         dense_label = tuple(dense_label)

        #         if dense_label not in result:
        #             result[dense_label] = ([row], [0])
        #         else:
        #             rows = result[dense_label][0]
        #             rows.append(row)

        #     return result

        # self._index_to_rows = make_index_to_rows(dataset_labels)

    def label_to_index(self, labels):
        assert_in(len(labels.shape), (1, 2))

        labels = numpy.asarray(labels)

        was_1D = False
        if len(labels.shape) == 1:
            labels = labels[numpy.newaxis, :]
            was_1D = True

        assert_equal(labels.shape[1], 5)

        indices = numpy.empty_like(labels)
        indices[...] = -1

        for (label_values,
             label_column,
             index_column) in safe_izip(self.label_values,
                                        labels.T,
                                        indices.T):

            for ind, value in enumerate(label_values):
                mask = (label_column == value)
                index_column[mask] = ind

        assert_false((indices == -1).any())

        if was_1D:
            assert_equal(indices.shape[0], 1)
            return indices[0, :]
        else:
            return indices

    def index_to_label(self, indices):
        assert_in(len(indices.shape), (1, 2))

        indices = numpy.asarray(indices)

        was_1D = False
        if len(indices.shape) == 1:
            indices = indices[indices.newaxis, :]
            was_1D = True

        assert_equal(indices.shape[1], 5)

        labels = numpy.zeros_like(indices)

        for (index_column,
             label_column,
             label_values) in safe_izip(indices.T,
                                        labels.T,
                                        self.label_values):
            label_column[:] = label_values[index_column]

        if was_1D:
            return labels[0, :]
        else:
            return labels

    def index_to_rows(self, indices):
        assert_equal(indices.shape, (5, ))
        key = tuple(indices)
        return self._index_to_rows.get(key, (None, -1))


def main():
    def parse_args():
        parser = argparse.ArgumentParser(
            description="Browse NORB images by label.")

        parser.add_argument('--which-norb',
                            help="'big', 'small', or a path to a .h5 file.'")

        parser.add_argument('--which-set',
                            required=True,
                            help=("Which partition to use (typically 'test' "
                                  "or 'train')."))

        return parser.parse_args()

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

    # The current index dimension being edited.
    index_dim = [0]

    figure, all_axes = pyplot.subplots(1, 3, squeeze=True, figsize=(16, 4))

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
                  lambda x: x,
                  lambda x: 30 + x * 5,
                  lambda x: x * 20,
                  lambda x: x]

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

            lines[index_dim[0]] = "==> " + lines[index_dim[0]]

            if dataset_labels.shape[1] == 11:
                for name, value, converter in safe_izip(label_names[5:],
                                                        label[5:],
                                                        converters[5:]):
                    lines.append('{}: {}'.format(name, converter(value)))
        else:
            label = label_index_map.index_to_label(index)

            for name, value, converter in safe_izip(label_names[:5],
                                                    label[:5],
                                                    converters[:5]):
                lines.append('{}: {}'.format(name, converter(value)))

            lines.append("image: (no such image)")
            lines.append('')

            if dataset_labels.shape[1] == 11:
                for name in label_names[5:]:
                    lines.append('{}: N/A'.format(name))

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

    def draw_images():
        def draw_image_impl(image, axes):
            axes.imshow(image, cmap='gray', norm=matplotlib.colors.NoNorm())


        rows, rows_index = label_index_map.index_to_rows(index)

        if rows is not None:
            row = rows[rows_index[0]]

            image = dataset_images[row]
            if 's' in image_format.axes:
                assert_equal(image_format.axes.index('s'), 1)
                for sub_image, axes in safe_izip(image, all_axes[1:]):
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

        if index[0] == 5 and index_dim[0] in (0, 5):
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
