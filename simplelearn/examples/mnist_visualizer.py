#! /usr/bin/env python

import sys
import argparse
import cPickle
import theano
import numpy
import matplotlib
from matplotlib import pyplot
from nose.tools import (assert_greater,
                        assert_greater_equal,
                        assert_is_instance,
                        assert_equal)
from simplelearn.io import SerializableModel
from simplelearn.data import DummyIterator
from simplelearn.data.mnist import load_mnist
from simplelearn.nodes import Softmax, RescaleImage, InputNode, CrossEntropy
from simplelearn.training import Sgd, SgdParameterUpdater, LimitsNumEpochs
from simplelearn.utils import safe_izip

import pdb


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Visualizes the thoughts of the best model outputted by "
                     "./mnist_fully_connected.py. Default arguments are "
                     "hand-tuned to get good results on the initially "
                     "displayed digit (a '7'). For good results on other "
                     "digits, try playing with --max-iterations and "
                     "--learning-rate"))

    def pkl_file(arg):
        pickle_file = file(arg, 'rb')
        return cPickle.load(pickle_file)

    def positive_float(arg):
        result = float(arg)
        assert_greater(result, 0.0)
        return result

    def non_negative_float(arg):
        result = float(arg)
        assert_greater_equal(result, 0.0)
        return result

    def positive_int(arg):
        result = int(arg)
        assert_greater(result, 0)
        return result

    parser.add_argument("--model",
                        type=pkl_file,
                        required=True,
                        help=("The '..._best.pkl' file outputted by "
                              "./mnist_fully_connected.py"))

    parser.add_argument("--learning-rate",
                        type=positive_float,
                        default=0.8,
                        help=("The learning rate used to optimize images."))

    parser.add_argument("--max-iterations",
                        type=positive_int,
                        default=200,
                        help="The max # of iterations when optimizing images.")

    parser.add_argument("--momentum",
                        type=non_negative_float,
                        default=.5,
                        help=("The momentum used to optimize images."))

    parser.add_argument("--nesterov",
                        type=bool,
                        default=True,
                        help=("Use Nesterov accelerated gradients if True."))

    return parser.parse_args()


def main():
    args = parse_args()
    model = args.model

    assert_is_instance(model, SerializableModel)
    assert_equal(len(model.output_nodes), 1)
    assert_equal(len(model.input_nodes), 1)

    output_node = model.output_nodes[0]
    assert_is_instance(output_node, Softmax)
    assert_equal(output_node.output_format.axes, ('b', 'f'))

    input_uint8_node = model.input_nodes[0]
    function = model.compile_function()

    def get_input_float_node(output_node):
        '''
        Crawls back the chain from output_node towards inputs, and returns
        the RescaleImage node if found.
        '''
        assert_equal(len(output_node.inputs), 1)
        while not isinstance(output_node, RescaleImage):
            if isinstance(output_node, InputNode):
                raise RuntimeError("Expected model to contain a RescaleImage "
                                   "node, but didn't find one.")
            output_node = output_node.inputs[0]

        return output_node

    input_float_node = get_input_float_node(output_node)

    mnist_train = load_mnist()[1]
    mnist_train_iterator = mnist_train.iterator(iterator_type='sequential',
                                                batch_size=1)
    label_node = mnist_train_iterator.make_input_nodes()[1]

    cross_entropy = CrossEntropy(output_node, label_node)

    #
    # Create shared-variable versions of the float image and label nodes,
    # and swap them into the computational graph.
    #

    shared_input_float = theano.shared(
        input_float_node.output_format.make_batch(is_symbolic=False,
                                                  batch_size=1))

    shared_label = theano.shared(
        label_node.output_format.make_batch(batch_size=1, is_symbolic=False))

    cross_entropy.output_symbol = theano.clone(
        cross_entropy.output_symbol,
        replace={input_float_node.output_symbol: shared_input_float,
                 label_node.output_symbol: shared_label})

    loss_symbol = cross_entropy.output_symbol.mean()
    output_node.output_format.check(output_node.output_symbol)

    gradient_symbol = theano.gradient.grad(loss_symbol, shared_input_float)

    def get_optimized_images(float_image):

        optimized_images = input_float_node.output_format.make_batch(
            is_symbolic=False,
            batch_size=10)

        for i in xrange(model.output_nodes[0].output_format.shape[1]):
            print("optimizing image w.r.t. '%d' label" % i)
            param_updater = SgdParameterUpdater(
                shared_input_float,
                gradient_symbol,
                learning_rate=args.learning_rate,
                momentum=args.momentum,
                use_nesterov=args.nesterov)

            sgd = Sgd(inputs=[],
                      parameters=[shared_input_float],
                      parameter_updaters=[param_updater],
                      input_iterator=DummyIterator(),
                      monitors=[],
                      epoch_callbacks=[LimitsNumEpochs(args.max_iterations)])

            shared_input_float.set_value(float_image)
            shared_label.set_value(numpy.asarray([i],
                                                 dtype=shared_label.dtype))
            sgd.train()

            optimized_images[i, ...] = shared_input_float.get_value()[0, ...]

        return optimized_images

    figure = pyplot.figure(figsize=numpy.array([5, 5]) * [3, 5])

    image_axes = figure.add_subplot(3, 5, 1)

    optimized_image_axes = []
    for r in range(2, 4):
        for c in range(1, 6):
            optimized_image_axes.append(figure.add_subplot(3,
                                                           5,
                                                           (r - 1) * 5 + c))

    get_float_image = theano.function([input_uint8_node.output_symbol],
                                      input_float_node.output_symbol)

    def update_display(uint8_image, target_label):
        float_image = get_float_image(uint8_image)
        normalize_images = False
        norm = (None if normalize_images
                else matplotlib.colors.NoNorm())

        image_axes.imshow(float_image[0, ...], cmap='gray', norm=norm)

        optimized_images = get_optimized_images(float_image)
        for image, axes in safe_izip(optimized_images, optimized_image_axes):
            axes.imshow(image, cmap='gray', norm=matplotlib.colors.NoNorm())

        figure.canvas.draw()

    def on_key_press(event):
        if event.key == ' ':
            update_display(*mnist_train_iterator.next())
        elif event.key == 'q':
            sys.exit(0)

    figure.canvas.mpl_connect('key_press_event', on_key_press)

    update_display(*mnist_train_iterator.next())
    pyplot.show()


if __name__ == '__main__':
    main()
