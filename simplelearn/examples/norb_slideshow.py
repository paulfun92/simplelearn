#! /usr/bin/env python

'''
Steps through the images of NORB or Small NORB.

This is intended as a minimal example of using the NORB dataset.
For an actually useful browser, use simplelearn/scripts/browse_norb.py.
'''

import sys
import argparse
import matplotlib
from matplotlib import pyplot
from nose.tools import assert_in
from simplelearn.utils import safe_izip
from simplelearn.data.norb import load_norb

import pdb


def main():
    '''
    Entry point of script.
    '''

    def parse_args():
        '''
        Parses command-line args.
        '''

        parser = argparse.ArgumentParser(
            description="Viewer for MNIST's images and labels.")

        parser.add_argument('--which-norb',
                            required=True,
                            choices=['big', 'small'],
                            help="Which NORB dataset (big or small)")

        parser.add_argument('--which-set',
                            required=True,
                            choices=['test', 'train'],
                            help="Which set to view (test or train)")

        return parser.parse_args()

    args = parse_args()

    figure, all_axes = pyplot.subplots(1, 3, squeeze=True, figsize=(16, 4))

    # Hides x and y tick marks from all axes
    for axes in all_axes:
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)

    window_title = ("{} NORB, {}ing set".format(args.which_norb,
                                                args.which_set))

    figure.canvas.set_window_title(window_title)

    dataset = load_norb(args.which_norb)[0 if args.which_set == 'train' else 1]
    iterator = dataset.iterator(batch_size=1, iterator_type='sequential')

    index = [0]

    label_axes, left_image_axes, right_image_axes = all_axes

    def draw_image(image, axes):
        axes.imshow(image, cmap='gray', norm=matplotlib.colors.NoNorm())

    def draw_label(labels, axes):
        assert_in(len(labels), (5, 11))
        axes.clear()

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

        if len(labels) == 11:
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

        lines = ['{}: {}'.format(name, converter(label))
                 for name, converter, label
                 in safe_izip(label_names, converters, labels)]

        # "transAxes": 0, 0 = bottom-left, 1, 1 at upper-right.
        axes.text(0.1,  # x
                  0.5,  # y
                  '\n'.join(lines),
                  verticalalignment='center',
                  transform=axes.transAxes)

    index = [0]

    def show_next():
        '''
        Shows the next image and label.
        '''

        stereo_images, labels = iterator.next()
        stereo_image = stereo_images[0]
        label = labels[0]

        draw_label(label, label_axes)
        draw_image(stereo_image[0], left_image_axes)
        draw_image(stereo_image[1], right_image_axes)

        # image_num = index[0] % dataset.size

        figure.canvas.draw()
        index[0] += 1

    show_next()

    def on_key_press(event):
        if event.key == 'q':
            sys.exit(0)
        elif event.key == ' ':
            show_next()

    figure.canvas.mpl_connect('key_press_event', on_key_press)
    pyplot.show()

if __name__ == '__main__':
    main()
