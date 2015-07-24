#! /usr/bin/env python

'''
Simple script to step through and view the MNIST dataset's contents.
'''

import argparse
import sys
from matplotlib import pyplot
from simplelearn.data.mnist import load_mnist

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

        parser.add_argument('--which-set',
                            required=True,
                            choices=['test', 'train'],
                            help=("Which set to view ('test' or 'train'"))

        return parser.parse_args()

    args = parse_args()

    figure, image_axes = pyplot.subplots(1,
                                         1,
                                         squeeze=True,
                                         figsize=(4, 4))

    figure.canvas.set_window_title("MNIST's %sing set" % args.which_set)

    image_axes.get_xaxis().set_visible(False)
    image_axes.get_yaxis().set_visible(False)

    dataset = load_mnist()[0 if args.which_set == 'train' else 1]
    iterator = dataset.iterator(batch_size=1, iterator_type='sequential')

    index = [0]

    def show_next():
        '''
        Shows the next image and label.
        '''

        images, labels = iterator.next()
        image = images[0]
        label = labels[0]

        image_axes.imshow(image, cmap='gray')

        image_num = index[0] % dataset.num_examples()

        image_axes.set_title('label: %d (%d of %d)' %
                             (label, image_num + 1, dataset.num_examples()))

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
