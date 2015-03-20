"""
Symbolic functions, which can be composed into a DAG to form a model.
"""

__author__ = "Matthew Koichi Grimes"
__email__ = "mkg@alum.mit.edu"
__copyright__ = "Copyright 2014"
__license__ = "Apache 2.0"


import copy
import collections
import numpy
import theano
from theano.tensor import Rebroadcast
from nose.tools import (assert_true,
                        assert_equal,
                        assert_is_instance,
                        assert_in)
from simplelearn.utils import (safe_izip,
                               assert_is_integer,
                               assert_are_integers,
                               assert_are_greater_equal)
from simplelearn.formats import Format, DenseFormat
import pdb

# pylint: disable=too-few-public-methods


class Node(object):

    """
    A Node represents a function.

    The function's input nodes are in self.inputs
    The function's output is self.output_symbol.
    The function's output format is self.output_format.

    A model is a directed acyclic graph (DAG) of Nodes.
    """

    def __init__(self, input_nodes, output_symbol, output_format):
        '''
        Parameters
        ----------
        input_nodes: Node, or Sequence of Nodes

        output_symbol: theano.gof.Variable
          A function of the input_nodes' output_symbols.

        output_format: Format
          The output_symbol's data format.
        '''
        assert_is_instance(input_nodes, (Node, collections.Sequence))
        assert_is_instance(output_symbol, theano.gof.Variable)
        assert_is_instance(output_format, Format)

        if isinstance(input_nodes, Node):
            input_nodes = (input_nodes, )
        else:
            input_nodes = tuple(input_nodes)

        for input_node in input_nodes:
            assert_is_instance(input_node, Node)

        self.inputs = input_nodes
        self.output_format = output_format
        self.output_symbol = output_symbol


class FormatNode(Node):
    '''
    A node that just performs a format conversion.
    '''
    def __init__(self, input_node, output_format, axis_map):
        input_symbol = input_node.output_symbol
        input_format = input_node.output_format
        output_symbol = input_format.convert(input_symbol,
                                             output_format=output_format,
                                             axis_map=axis_map)

        super(FormatNode, self).__init__(input_node,
                                         output_symbol,
                                         output_format)


class InputNode(Node):
    """
    Represents the input to a model.

    Doesn't have any inputs. This just stores an output_symbol and its Format.
    """

    def __init__(self, fmt):
        '''
        Parameters
        ----------
        fmt: Format
          Format of self.output_symbol
        '''
        output_symbol = fmt.make_batch(is_symbolic=True)
        super(InputNode, self).__init__(input_nodes=(),
                                        output_symbol=output_symbol,
                                        output_format=fmt)


class RescaleImage(Node):
    '''
    Converts a int image to a floating-point image.
    Remaps pixel value range from [0, 255] to [0.0, 1.0].
    '''

    def __init__(self, input_node, output_dtype='floatX'):
        #
        # Sanity-check args
        #

        # Check number of axes, to see if it's really an image
        non_batch_axes = tuple(a for a in input_node.output_format.axes
                               if a != 'b')
        if len(non_batch_axes) not in (2, 3):
            raise ValueError("Expected input_node.format to have 2 or 3 "
                             "non-batch axes, but got %d: %s" %
                             (len(non_batch_axes), str(input_node.axes)))

        assert_true(numpy.issubdtype(input_node.output_symbol.dtype,
                                     numpy.integer))

        dtype = str(theano.config.floatX
                    if str(output_dtype) == 'floatX'
                    else output_dtype)

        assert_true(numpy.issubdtype(dtype, numpy.floating))

        # TODO: theano-assert that all values are between 0 and 255.

        #
        # Actual work starts here
        #

        output_format = copy.deepcopy(input_node.output_format)
        output_format.dtype = dtype

        normalizer = numpy.asarray(255, dtype=dtype)
        output_symbol = (theano.tensor.cast(input_node.output_symbol, dtype) /
                         normalizer)

        super(RescaleImage, self).__init__(input_node,
                                           output_symbol,
                                           output_format)


class Function1dTo1d(Node):
    """
    Represents a vector-valued function of vectors.

    Takes the input, reshapes it into a matrix[batch_index, feature_index],
    and computes the output using a vector-valued function of the rows.

    This output matrix is then reshaped to the desired output_format.
    """

    @staticmethod
    def _get_bf_shape(fmt):
        """
        Returns the shape of an equivalent format with ('b', 'f') axes.

        Does this by collapsing all axes other than 'b' into a single axis 'f'.
        """

        non_b_sizes = [fmt.shape[fmt.axes.index(a)]
                       for a in fmt.axes if a != 'b']
        f_size = 1 if len(non_b_sizes) == 0 else numpy.prod(non_b_sizes)
        b_size = fmt.shape[fmt.axes.index('b')]
        return (b_size, f_size)

    def __init__(self,
                 input_node,
                 output_format,
                 input_to_bf_map=None,
                 bf_to_output_map=None):
        """
        Input node's format and output_format must both contain 'b' and 'f'
        axes.

        Parameters
        ----------
        input_node: Node
          Node that provides input to this Node.

        output_format: Format
          Format of this node's output_symbol.

        input_to_bf_map: dict
          An axis map from input_node's axes to the ('b', 'f') axes used
          internally.

          See docs for Format.convert() for more detail on axis maps.

          If omitted, this will keep the 'b' axis, and collapse all remaining
          axes into a single 'f' axis.

        bf_to_output_map: dict
          An axis map from the internal ('b', 'f') axes to the output_format.

          See docs for Format.convert() for more detail on axis maps.

          If omitted, this will keep the 'b' axis, and expand the 'f' axis to
          all remaining axes in output_format.
        """

        assert_is_instance(output_format, Format)
        assert_is_instance(input_node, Node)

        input_format = input_node.output_format
        input_axes = input_format.axes
        output_axes = output_format.axes

        #
        # Checks that input and output axes contain 'b', the batch axis.
        #

        assert_in('b', input_axes)
        assert_in('b', output_axes)

        #
        # Creates equivalent bf format to input format.
        #

        get_bf_shape = Function1dTo1d._get_bf_shape

        input_bf_format = DenseFormat(axes=('b', 'f'),
                                      shape=get_bf_shape(input_format),
                                      dtype=input_format.dtype)

        #
        # Creates axis mappings to and from input_bf_format
        #

        if input_to_bf_map is None:
            input_non_b_axes = tuple(a for a in input_axes if a != 'b')
            if len(input_non_b_axes) == 0:
                input_to_bf_map = {'b': ('b', 'f')}
            else:
                input_to_bf_map = {input_non_b_axes: 'f',
                                   'b': 'b'}

        if bf_to_output_map is None:
            output_non_b_axes = tuple(a for a in output_format.axes
                                      if a != 'b')
            if len(output_non_b_axes) == 0:
                bf_to_output_map = {('b', 'f'): 'b'}
            else:
                bf_to_output_map = {'f': output_non_b_axes,
                                    'b': 'b'}

        input_to_bf_node = FormatNode(input_node,
                                      input_bf_format,
                                      input_to_bf_map)

        # format to output format
        output_bf_format = DenseFormat(axes=('b', 'f'),
                                       shape=get_bf_shape(output_format),
                                       dtype=output_format.dtype)

        # compute function of feature vectors
        output_bf_node = self._get_output_bf_node(input_to_bf_node,
                                                  output_bf_format)

        assert_is_instance(output_bf_node, Node)

        bf_to_output_node = FormatNode(output_bf_node,
                                       output_format,
                                       bf_to_output_map)

        super(Function1dTo1d, self).__init__(input_node,
                                             bf_to_output_node.output_symbol,
                                             bf_to_output_node.output_format)

    def _get_output_bf_node(self,
                            input_bf_node,
                            output_bf_format):
        '''
        Returns a Node that takes input with axes=('b', 'f'), and
        yields output with axes=('b', 'f').
        '''
        raise NotImplementedError("%s._get_output_bf_node() not yet "
                                  "implemented." % type(self))


class Linear(Function1dTo1d):
    '''
    Applies a linear transformation to the input.
    '''

    def __init__(self, input_node, output_format):
        get_bf_shape = Function1dTo1d._get_bf_shape

        num_input_features = get_bf_shape(input_node.output_format)[1]
        num_output_features = get_bf_shape(output_format)[1]

        params = numpy.zeros((num_input_features, num_output_features),
                             dtype=input_node.output_symbol.dtype)
        self.params = theano.shared(params)

        super(Linear, self).__init__(input_node, output_format)

    def _get_output_bf_node(self,
                            input_bf_node,
                            output_bf_format):
        output_symbol = theano.tensor.dot(input_bf_node.output_symbol,
                                          self.params)
        return Node(input_bf_node, output_symbol, output_bf_format)


class Bias(Function1dTo1d):
    '''
    Adds a bias to the input.
    '''

    def __init__(self, input_node, output_format):
        get_bf_shape = Function1dTo1d._get_bf_shape

        num_input_features = get_bf_shape(input_node.output_format)[1]
        num_output_features = get_bf_shape(output_format)[1]

        assert_equal(num_output_features, num_input_features)

        params = numpy.zeros((1, num_output_features),
                             dtype=input_node.output_symbol.dtype)

        self.params = theano.shared(params, broadcastable=[True, False])
        super(Bias, self).__init__(input_node, output_format)

    def _get_output_bf_node(self,
                            input_bf_node,
                            output_bf_format):
        output_symbol = input_bf_node.output_symbol + self.params
        return Node(input_bf_node, output_symbol, output_bf_format)


class AffineTransform(Function1dTo1d):
    '''
    Implements dot(X, M) + B (multiplying by a matrix, then adding a bias)
    '''

    def __init__(self,
                 input_node,
                 output_format,
                 input_to_bf_map=None,
                 bf_to_output_map=None):
        super(AffineTransform, self).__init__(input_node,
                                              output_format,
                                              input_to_bf_map,
                                              bf_to_output_map)

    def _get_output_bf_node(self,
                            input_bf_node,
                            output_bf_format):
        self.linear_node = Linear(input_bf_node, output_bf_format)

        # bias node's output format is the same as its input format
        self.bias_node = Bias(self.linear_node,
                              output_format=output_bf_format)

        return self.bias_node


def _make_bc01_format_node(i_node, i2t_axis_map):
    '''
    Returns a FormatNode that converts a 4-D input to "bc01-floatX" format.

    "bc01" is the axis order used by cuDNN: batch-channel-row-column.
    floatX is the dtype specified by theano.config.floatX

    Parameters
    ----------
    i_node: Node
      A Node with a 4-dimensional output.

    i2t_axis_map: None or dict
       The maps axis names in i_node.output_format to 'b', 'c', '0', and '1'.
       You can omit this if i_node.ouptut_format.axes already uses those names.
    '''
    # Naming convention used in this function:
    # 'i': input
    # 't': the input, transposed to standard 'bc01' order.
    i_axes = i_node.output_format.axes
    if i2t_axis_map is None:
        i2t_axis_map = {'b':'b', 'c':'c', '0':'0', '1':'1'}

    t2i_axis_map = dict((v, k) for k, v in axis_map)

    t_axes = ('b', 'c', '0', '1')
    t_axes_with_i_names = [t2i_axis_map[a] for a in t_axes]

    i_shape = input_node.output_format.axes
    t_shape = [i_shape[i_axes.index(a)] for a in t_axes_with_i_names]

    assert_equal(t_shape[0], -1)
    for s in t_shape:
        assert_greater_equal(s, 0)

    t_format = DenseFormat(axes=t_axes,
                           shape=t_shape,
                           dtype=theano.config.floatX)

    return FormatNode(i_node, t_format, i2t_axis_map)


def _assert_is_shape2d(arg):
    assert_are_integers(arg, 2)
    assert_are_greater_equal(arg, 0)


def _make_shape_reserving_pad(window_shape):
    '''
    Returns padding amount that preserves the image shape under convolution.
    '''
    _assert_is_shape2d(window_shape)

    return tuple((ws - 1) // 2 if ws % 2 == 0 else ws // 2
                 for ws in window_shape)

def _make_bc01_output_format(bc01_input_format,
                             strides,
                             filter_shape,
                             num_filters,
                             pad):
    '''
    Constructs an appropriately-sized output format for a convolution-like
    operator.

    Returns
    -------
    rval: DenseFormat
      Output format of this Conv2D node.
    '''

    assert_equal(bc01_input_format.axes, ('b', 'c', '0', '1'))
    _assert_is_shape2d(strides)
    _assert_is_shape2d(filter_shape)

    strides = numpy.asarray(strides)
    i_img_shape = numpy.asarray(bc01_input_format.shape[2:]) // strides
    filter_shape = numpy.asarray(filter_shape)

    if pad == 'valid':
        o_img_shape = i_img_shape - filter_shape + 1
    elif pad == 'full':
        o_img_shape = i_img_shape + filter_shape + 1
    elif pad = 'same_shape':
        o_img_shape = copy.deepcopy(i_img_shape)
        pad = _make_shape_preserving_pad(filter_shape)
    else:
        assert_is_shape2d(pad)
        o_img_shape = i_img_shape - filter_shape + 1 + pad

    return DenseFormat(axes=('b', 'c', '0', '1'),
                       shape=(-1,
                              num_filters,
                              o_img_shape[0],
                              o_img_shape[1]),
                       dtype=theano.config.floatX)

class Conv2D(Node):
    '''
    Returns a convolution over 2D space.

    The output axis format is ('b', 'c', '0', '1'), or batch, channel, row,
    column. Output dtype is theano.config.floatX.

    The input may be in any 4D DenseFormat. This will internally be converted
    to the above axis ordering and dtype.
    '''

    def __init__(self,
                 input_node,
                 filter_shape,
                 num_filters,
                 pad,
                 strides=(1, 1),
                 axis_map=None):
        '''
        Parameters
        ----------

        pad: str or tuple
          'valid': no zero-padding.
                   output_shape: input_shape - filter_shape + 1
          'full': maximal zero-padding.
                  output_shape: input_shape + filter_shape + 1
          'same_size': Enough padding to preserve image shape.
                       output_shape = input_shape
          (R, C): Pad rows and columns by this many zeros on each side.
                  output_shape = (input_shape[0] + 2R, input_shape[1] + 2C)

        strides: Sequence
          A Sequence of length two. Default: (1, 1).
          A stride of (R, C) means we apply the filters once every R rows
          and every C columns.

        axis_map: dict
          Maps axis names from those used by input_node.output_format.axes
          to ('b', 'c', '0', '1'), i.e. batch, channel, row, column.
          See docs for axis_map parameter to simplelearn.format.convert()
        '''

        #
        # Sanity-checks args
        #

        assert_is_instance(input_node, Node)
        _assert_is_shape2d(filter_shape)
        assert_is_integer(num_filters)

        if isinstance(pad, basestring):
            assert_in(pad, ('valid', 'full', 'same_size'))
        else:
            assert_is_shape(pad)

        if stride is not None:
            _assert_is_shape2d(stride)

        if axis_map is not None:
            assert_is_instance(axis_map, dict)

        input_format_node = _make_bc01_format_node(input_node, axis_map)

        output_format = _make_bc01_output_format(
            input_format_node.output_format,
            strides,
            filter_shape,
            num_filters,
            pad)

        self.filters = theano.shared(
            numpy.zeros(([num_filters] +
                         input_format_node.output_format.shape[1] +
                         list(filter_shape)),
                        dtype=theano.config.floatX))


        def make_output_symbol(t_node, filters, pad, strides):
            image_shape = copy.deepcopy(t_node.output_format.shape)
            assert_equal(image_shape[0], -1)
            image_shape[0] = None

            return theano.tensor.nnet.conv2d(input=t_node.output_symbol,
                                             filters=filters,
                                             image_shape=image_shape,
                                             filter_shape=filters.shape,
                                             border_mode=pad,
                                             subsample=strides,
                                             **kwargs)

        output = make_output_symbol(input_format_node,
                                    self.filters,
                                    pad,
                                    strides)


        super(Conv2D, self).__init__([input_node], output, output_format)


class Pool2D(Node):

    def __init__(self,
                 input_node,
                 window_shape,
                 strides,
                 mode=None,
                 pad=None,
                 axis_map=None):
        '''
        cuDNN spatial pooling over image maps.

        Parameters
        ----------
        input_node: Node
          input_node.output_format may be any 4-D input format.

        window_shape: Sequence
          The shape of the pooling window, given as 2 ints: (# rows, # cols)

        strides: Sequence
          A sequence of 2 ints: the row and column strides.

        mode: str
          'max' or 'average' (default: 'max')

        pad: str, or Sequence
          'valid', 'full', 'same_size', or a Sequence of 2 ints.

          'valid': no zero-padding
                   output_shape: input_shape - window_shape + 1

          'full': maximal zero-padding
                  output_shape: input_shape + filter_shape + 1

          'same_size': Enough padding to preserve image shape.
                       output_shape: image_shape

          (R, C): Pad rows and colums by this many zeros on each side.
                  output_shape = (input_shape[0] + 2R, input_shape[1] + 2C)

        axis_map: dict
          Maps the axis names in input_node.output_format.axes to
          'b', 'c', '0', and '1' (batch, channel, row, column).
          You can omit this argument those are the axis names
          being used by input_node.output_format.axes.
        '''
        assert_is_instance(input_node, Node)
        _assert_is_shape2d(window_shape)
        _assert_is_shape2d(strides)
        assert_in(mode, ('max', 'average'))


        input_format_node = _make_bc01_format_node(input_node, axis_map)

        output_format = _make_bc01_output_format(
            input_format_node.output_format,
            strides,
            window_shape,
            input_format_node.output_format.shape[1],  # num channels unchanged
            pad)

        if isinstance(pad, basestring):
            if pad == 'valid':
                pad = (0, 0)
            elif pad == 'full':
                pad = window_shape - 1
            elif pad == 'same_size':
                pad = _make_shape_preserving_pad(window_shape)
            else:
                _assert_is_shape2d(pad)

        output_symbol = theano.sandbox.cuda.dnn.dnn_pool(
            img=input_format_node.output_symbol,
            ws=window_shape,
            stride=strides,
            mode=mode,
            pad=pad)

        super(Pool2D, self).__init__([input_node],
                                     output_symbol,
                                     output_format)


class Softmax(Function1dTo1d):
    '''
    Softmax node.
    '''

    def __init__(self,
                 input_node,
                 output_format=None,
                 input_to_bf_map=None,
                 bf_to_output_map=None):

        if output_format is None:
            output_format = input_node.output_format

        super(Softmax, self).__init__(input_node,
                                      output_format,
                                      input_to_bf_map,
                                      bf_to_output_map)

    def _get_output_bf_node(self, input_bf_node, output_bf_format):
        softmaxes = theano.tensor.nnet.softmax(input_bf_node.output_symbol)
        return Node(input_bf_node, softmaxes, output_bf_format)


class ReLU(Node):
    '''
    Elementwise linear rectifier (zeros any negative elements).

    Preserves format.
    '''
    def __init__(self, input_node):
        input_symbol = input_node.output_symbol
        output_symbol = theano.tensor.switch(input_symbol > 0.0,
                                             input_symbol,
                                             0.0)
        super(ReLU, self).__init__([input_node],
                                   output_symbol,
                                   input_node.output_format)

class Dropout(Node):
    '''
    Randomly zeros some fraction (1-P) of the input, and scales it by 1/P.

    P is chosen as the include_probability constructor arg.

    It is the user's responsibility to also scale the learning rate of the
    input node's weights by P^2. The model-constructing functions in
    simplelearn/models.py do this.

    The same mask is applied to all elements in a batch.
    (To apply different dropout masks, we'd need to know the batch size at
    theano funciton compilation time, which is possible, but would make the
    code messier. I'm therefore not implementing it for now.)
    '''
    def __init__(self, input_node, include_probability, theano_rng):
        '''
        Parameters
        ----------
        input_node: Node
        include_probability: float
        theano_rng: theano.rng.shared_randomstreams.RandomStreams
        '''
        assert_true(numpy.issubdtype(type(include_probability),
                                     numpy.floating))
        assert_is_instance(theano_rng,
                           theano.tensor.shared_randomstreams.RandomStreams)
        assert_in('b', input_node.output_format.axes)

        self._include_probability = float(include_probability)

        example_shape = list(input_node.output_format.shape)
        b_index = input_node.output_format.axes.index('b')
        example_shape[b_index] = 1

        input_symbol = input_node.output_symbol

        # Evaluates to a different random mask on every function evaluation.
        # Within a single evaluation, the mask is the same even if it comes up
        # in multiple places.
        mask = theano_rng.binomial(p=self._include_probability,
                                   size=example_shape,
                                   dtype=input_symbol.dtype)

        # makes the mask's batch axis broadcastable
        def rebroadcast_batch_axis(arg, fmt):
            rebroadcast = Rebroadcast(*tuple(enumerate(axis == 'b'
                                                       for axis
                                                       in fmt.axes)))
            return rebroadcast(arg)

        self.mask = rebroadcast_batch_axis(mask, input_node.output_format)

        cast = numpy.cast[input_symbol.dtype]
        input_scale = cast(1.0 / self._include_probability)

        output_symbol = input_symbol * self.mask * input_scale

        super(Dropout, self).__init__([input_node],
                                      output_symbol,
                                      input_node.output_format)


class L2Loss(Node):
    '''
    Computes ||x_i - y_i||^2 for each x_i, y_i in a batch of data.

    Input nodes must have a batch axis.

    Output axes: ['b']
    '''

    def __init__(self, input_node_a, input_node_b):
        format_a = input_node_a.output_format
        format_b = input_node_b.output_format

        if format_a.shape != format_b.shape or \
           format_a.axes != format_b.axes:
            raise ValueError("Can't take the L2 loss between different "
                             "formats: %s vs %s" % (format_a, format_b))

        def convert_to_bf(node):
            '''
            Returns a node's output_symbol reshaped to have axes=('b', 'f').
            '''

            def make_bf_format(fmt):
                batch_size = fmt.shape[fmt.axes.index('b')]
                feature_size = numpy.prod(tuple(size
                                                for size, axis
                                                in safe_izip(fmt.shape,
                                                             fmt.axes)
                                                if axis != 'b'))

                non_b_axes = tuple(axis for axis in fmt.axes if axis != 'b')
                return DenseFormat(shape=(batch_size, feature_size),
                                   axes=('b', 'f'),
                                   dtype=fmt.dtype), non_b_axes

            bf_format, non_b_axes = make_bf_format(node.output_format)
            axis_map = {'b': 'b',
                        non_b_axes: 'f'}

            return node.output_format.convert(node.output_symbol,
                                              bf_format,
                                              axis_map=axis_map)


        symbol_a_bf = convert_to_bf(input_node_a)
        symbol_b_bf = convert_to_bf(input_node_b)

        diff = symbol_a_bf - symbol_b_bf
        output_symbol = (diff * diff).sum(axis=1)
        output_format = DenseFormat(axes=['b'], shape=[-1], dtype=None)

        super(L2Loss, self).__init__((input_node_a, input_node_b),
                                     output_symbol,
                                     output_format)

class CrossEntropy(Node):
    '''
    Computes the cross-entropy between model outputs and target labels.

    Target labels can either be one-hot vectors (vectors with all zeros except
    for a single 1), or ints. In the latter case, the ints are interpreted as
    the index of the one in a one-hot vector.
    '''

    def __init__(self, softmax_node, target_node):
        assert_equal(softmax_node.output_format.axes, ('b', 'f'))
        assert_in(target_node.output_format.axes, (('b', 'f'), ('b', )))

        target_symbol = theano.tensor.cast(target_node.output_symbol,
                                           'int64')

        output = theano.tensor.nnet.categorical_crossentropy(
            softmax_node.output_symbol,
            target_symbol)

        output_format = DenseFormat(axes=['b'], shape=[-1], dtype=None)
        super(CrossEntropy, self).__init__((softmax_node, target_node),
                                           output,
                                           output_format)

class Misclassification(Node):
    '''
    Returns 1 if a softmax and a target label disagree, and a 0 otherwise.

    Not smoothly differentiable. Don't use this as a loss function to minimize.

    The average value of this is useful as a value to monitor.
    '''

    def __init__(self, softmax_node, target_node):
        assert_equal(softmax_node.output_format.axes, ('b', 'f'))
        assert_in(target_node.output_format.axes, (('b', 'f'), ('b', )))

        target_symbol = theano.tensor.cast(target_node.output_symbol,
                                           'int64')

        # If targets are one-hot vectors, convert them to target indices
        if len(target_node.output_format.axes) == 2:
            target_indices = theano.tensor.argmax(target_node.output_symbol,
                                                  axis=1)
        else:
            assert_equal(target_node.output_symbol.ndim, 1)
            target_indices = target_node.output_symbol

        softmax_indices = theano.tensor.argmax(softmax_node.output_symbol,
                                               axis=1)

        result = theano.tensor.neq(softmax_indices, target_indices)
        result_format = DenseFormat(axes=['b'], shape=[-1], dtype='int8')
        super(Misclassification, self).__init__((softmax_node, target_node),
                                                result,
                                                result_format)
