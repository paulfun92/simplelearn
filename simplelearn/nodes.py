"""
Symbolic functions, which can be composed into a DAG to form a model.
"""

__author__ = "Matthew Koichi Grimes"
__email__ = "mkg@alum.mit.edu"
__copyright__ = "Copyright 2015"
__license__ = "Apache 2.0"


import copy
import collections
import numpy
import theano
from theano.tensor import Rebroadcast
from nose.tools import (assert_true,
                        assert_equal,
                        assert_greater,
                        assert_greater_equal,
                        assert_is_instance,
                        assert_is,
                        assert_in,
                        assert_not_in)
from numpy.testing import assert_array_equal
from simplelearn.utils import (safe_izip,
                               assert_integer,
                               assert_floating,
                               assert_all_equal,
                               assert_all_integers,
                               assert_all_is_instance,
                               assert_all_greater,
                               assert_all_greater_equal)
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

    # Set to True to check consistency between output_symbol.shape and
    # output_format.shape at runtime.
    DEBUG_check_output_shape = False

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
        output_format.check(output_symbol)

        if isinstance(input_nodes, Node):
            input_nodes = (input_nodes, )
        else:
            input_nodes = tuple(input_nodes)

        for input_node in input_nodes:
            assert_is_instance(input_node, Node)

        self.inputs = input_nodes
        self.output_format = output_format
        self.output_symbol = output_symbol

        if self.DEBUG_check_output_shape:
            assert_op = theano.tensor.opt.Assert("Expected shape {}".format(str(self.output_format.shape)))
            eq_op = theano.tensor.eq

            self.output_symbol = assert_op(self.output_symbol,
                                           eq_op(self.output_symbol.ndim,
                                                 len(self.output_format.axes)))

            for i in range(len(self.output_format.axes)):
                if self.output_format.axes[i] != 'b' or \
                   self.output_format.shape[i] != -1:

                    self.output_symbol = \
                        assert_op(self.output_symbol,
                                  eq_op(self.output_symbol.shape[i],
                                        self.output_format.shape[i]))


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


class RescaleImage(Node):  # TODO: rename to be a noun
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
    def _get_bf_shape(input_format, input_to_bf_map):
        """
        Returns the shape of an equivalent format with ('b', 'f') axes.

        The shape is (-1, N), where N is the product of all input axes that
        input_to_bf_map maps to 'f'. If input_to_bf_map is omitted, N is
        the product of all input axes other than 'b'.
        """
        assert_is_instance(input_to_bf_map, dict)  # no Nones allowed

        # No duplicate values
        assert_equal(len(frozenset(input_to_bf_map.itervalues())),
                     len(input_to_bf_map))

        bf_to_input_map = dict()
        for key, value in input_to_bf_map.iteritems():
            bf_to_input_map[value] = key

        if 'f' not in bf_to_input_map:
            # Return input 'f' axis' size unchanged.
            f_index = input_format.axes.index('f')
            return (-1, input_format.shape[f_index])
        else:
            # Return the product of sizes of axes mapped to 'f'.
            axes_mapped_to_f = bf_to_input_map['f']
            if isinstance(axes_mapped_to_f, basestring):
                axes_mapped_to_f = (axes_mapped_to_f, )

            # This would be complicated to code around, and is probably
            # user error anyway.
            assert_not_in('b', axes_mapped_to_f)

            dims_mapped_to_f = \
                [input_format.shape[input_format.axes.index(a)]
                 for a in axes_mapped_to_f]

            return (-1, numpy.prod(dims_mapped_to_f))

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
        # Creates axis mappings to and from ('b', 'f') format
        #

        if input_to_bf_map is None:
            input_non_b_axes = tuple(a for a in input_axes if a != 'b')
            if len(input_non_b_axes) == 0:
                input_to_bf_map = {'b': ('b', 'f')}
            else:
                input_to_bf_map = {'b': 'b',
                                   input_non_b_axes: 'f'}

        if bf_to_output_map is None:
            output_non_b_axes = tuple(a for a in output_format.axes
                                      if a != 'b')
            if len(output_non_b_axes) == 0:
                bf_to_output_map = {('b', 'f'): 'b'}
            else:
                bf_to_output_map = {'b': 'b',
                                    'f': output_non_b_axes}

        #
        # Creates equivalent bf format to input format.
        #

        get_bf_shape = Function1dTo1d._get_bf_shape

        input_bf_format = DenseFormat(axes=('b', 'f'),
                                      shape=get_bf_shape(input_format,
                                                         input_to_bf_map),
                                      dtype=input_format.dtype)

        input_bf_node = FormatNode(input_node,
                                   input_bf_format,
                                   input_to_bf_map)

        # format to output format
        output_to_bf_map = None
        if bf_to_output_map is not None:
            output_to_bf_map = dict()
            for key, value in bf_to_output_map.iteritems():
                output_to_bf_map[value] = key

        output_bf_format = DenseFormat(axes=('b', 'f'),
                                       shape=get_bf_shape(output_format,
                                                          output_to_bf_map),
                                       dtype=output_format.dtype)

        # compute function of feature vectors
        output_bf_node = self._get_output_bf_node(input_bf_node,
                                                  input_bf_format,
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
                            input_bf_format,
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

    def __init__(self, input_node, output_format, **kwargs):
        get_bf_shape = Function1dTo1d._get_bf_shape

        # set in _get_output_bf_node()
        self.params = None

        super(Linear, self).__init__(input_node, output_format, **kwargs)

    def _get_output_bf_node(self,
                            input_bf_node,
                            input_bf_format,
                            output_bf_format):
        assert_is(self.params, None)

        params = numpy.zeros((input_bf_format.shape[1],
                              output_bf_format.shape[1]),
                             dtype=input_bf_node.output_symbol.dtype)
        self.params = theano.shared(params)
        output_symbol = theano.tensor.dot(input_bf_node.output_symbol,
                                          self.params)

        if output_bf_format.dtype is None:
            output_dtype = input_bf_node.output_symbol.dtype
        else:
            output_dtype = output_bf_format.dtype

        output_symbol = theano.tensor.cast(output_symbol, str(output_dtype))

        return Node(input_bf_node, output_symbol, output_bf_format)


class Bias(Function1dTo1d):
    '''
    Adds a bias to the input.
    '''

    def __init__(self, input_node, output_format, **kwargs):

		# set in _get_output_bf_node()
        self.params = None
        super(Bias, self).__init__(input_node, output_format, **kwargs)

    def _get_output_bf_node(self,
                            input_bf_node,
                            input_bf_format,
                            output_bf_format):
        assert_is(self.params, None)
        assert_equal(input_bf_format.shape, output_bf_format.shape)

        params = numpy.zeros(output_bf_format.shape[1],
                             dtype=input_bf_node.output_symbol.dtype)

        self.params = theano.shared(params)

        output_symbol = (input_bf_node.output_symbol +
                         self.params.dimshuffle('x', 0))  # reshapes N to 1xN

        if output_bf_format.dtype is None:
            output_dtype = input_bf_node.output_symbol.dtype
        else:
            output_dtype = output_bf_format.dtype

        output_symbol = theano.tensor.cast(output_symbol, str(output_dtype))

        return Node(input_bf_node, output_symbol, output_bf_format)


class AffineTransform(Function1dTo1d):
    '''
    Implements dot(X, M) + B (multiplying by a matrix, then adding a bias)
    '''

    def __init__(self, input_node, output_format, **kwargs):
        super(AffineTransform, self).__init__(input_node,
                                              output_format,
                                              **kwargs)

    def _get_output_bf_node(self,
                            input_bf_node,
                            input_bf_format,
                            output_bf_format):
        self.linear_node = Linear(input_bf_node, output_bf_format)

        # bias node's output format is the same as its input format
        self.bias_node = Bias(self.linear_node, output_bf_format)

        return self.bias_node


def _assert_is_shape2d(arg):
    assert_all_integers(arg)
    assert_equal(len(arg), 2)
    assert_all_greater_equal(arg, 0)


def _make_bc01_format_node(i_node, i_to_bc01_axis_map):
    '''
    Returns a FormatNode that converts a 4-D input to "bc01-floatX" format.

    "bc01" is the axis order used by cuDNN: batch-channel-row-column.
    floatX is the dtype specified by theano.config.floatX

    Parameters
    ----------
    i_node: Node
      A Node with a 4-dimensional output.

    i_to_bc01_axis_map: None or dict
       The maps axis names in i_node.output_format to 'b', 'c', '0', and '1'.
       You can omit this if i_node.ouptut_format.axes already uses those names.
    '''
    i_axes = i_node.output_format.axes
    assert_all_is_instance(i_axes, basestring)
    assert_equal(len(i_axes), 4)

    bc01_axes = ('b', 'c', '0', '1')

    if i_to_bc01_axis_map is None:
        i_to_bc01_axis_map = dict((a, a) for a in bc01_axes)
    else:
        # keys are all strings (as opposed to tuples of strings)
        assert_all_is_instance(i_to_bc01_axis_map.iterkeys(), basestring)

        # values are some permutation of bc01_axes
        assert_equal(set(i_to_bc01_axis_map.itervalues()),
                     set(bc01_axes))

    bc01_to_i_axis_map = dict((v, k)
                              for k, v
                              in i_to_bc01_axis_map.iteritems())

    bc01_axes_with_i_names = [bc01_to_i_axis_map[a] for a in bc01_axes]

    i_shape = i_node.output_format.shape
    bc01_shape = [i_shape[i_axes.index(a)] for a in bc01_axes_with_i_names]

    assert_equal(bc01_shape[0], -1)
    for size, axis in safe_izip(bc01_shape, bc01_axes):
        if axis != 'b':
            assert_greater_equal(size, 0)

    bc01_format = DenseFormat(axes=bc01_axes,
                              shape=bc01_shape,
                              dtype=theano.config.floatX)
    return FormatNode(i_node, bc01_format, i_to_bc01_axis_map)


def _get_pads(pad_arg, image_shape, window_shape, strides):
    '''
    Converts pad argument to an ndarray of two ints.
    '''
    _assert_is_shape2d(image_shape)
    _assert_is_shape2d(window_shape)
    _assert_is_shape2d(strides)

    image_shape = numpy.asarray(image_shape)
    window_shape = numpy.asarray(window_shape)
    strides = numpy.asarray(strides)

    if pad_arg == 'valid':
        pads = [0, 0]
    elif pad_arg == 'full':
        pads = window_shape - 1
    elif pad_arg == 'same_shape':
        for window_size in window_shape:
            assert_equal(window_size % 2,
                         1,
                         "when pad_arg = 'same_shape', window_shape must have "
                         "only odd numbers. Instead, got %s." %
                         str(window_shape))
        pads = window_shape // 2
    elif pad_arg == 'min':
        pads = numpy.ceil((image_shape - window_shape) % strides / 2.0)
        pads = numpy.cast[int](pads)
    elif isinstance(pad_arg, basestring):
        raise ValueError("Unrecognized pad_arg value '%s'" % pad_arg)
    else:
        _assert_is_shape2d(pad_arg)
        pads = pad_arg

    return tuple(pads)

def _make_bc01_output_format(bc01_input_format,
                             strides,
                             window_shape,
                             num_filters,
                             pad):
    """
    Constructs an appropriately-sized output format for a convolution-like
    operator.

    Parameters
    ----------
    pad: str or Sequence
      'valid': zero padding
      'full': maximal padding (window_shape - 1)
      'same_shape': Pad just enough so that convolution doesn't change
                    the resolution (striding may still change the resolution).
                    The window_shape must have odd-numbered sizes.
      [R, C]: Pad by R rows and C columns of zeros on each side of the image.

    Returns
    -------
    rval: tuple
      (fmt, pads), where
        fmt: a DenseFormat
          Output format of this Conv2D node.
        pads: str, or tuple
          A suitable padding argument to pass to Theano's dnn_conv or dnn_pool.
          If pad was 'valid' or 'full', it's returned as-is to take advantage
          of Theano's efficient special-casing of those values. Otherwise
          this will be two ints, the row and column pad amounts.
    """

    assert_equal(bc01_input_format.axes, ('b', 'c', '0', '1'))
    assert_equal(bc01_input_format.shape[0], -1)
    assert_all_greater(bc01_input_format.shape[1:], 0)
    _assert_is_shape2d(strides)
    _assert_is_shape2d(window_shape)

    i_img_shape, strides, window_shape = (
        numpy.asarray(x) for x in (bc01_input_format.shape[2:],
                                   strides,
                                   window_shape))

    pad_pair = _get_pads(pad_arg=pad,
                         image_shape=i_img_shape,
                         window_shape=window_shape,
                         strides=strides)

    padded_input_shape = i_img_shape + numpy.asarray(pad_pair) * 2
    o_img_shape = (padded_input_shape - window_shape + 1 - 1) // strides + 1

    # Confirm that output sizes work out to be the predicted sizes from
    # Theano's docs, when stride == 1
    for window_size, o_img_size, i_img_size, stride in safe_izip(window_shape,
                                                                 o_img_shape,
                                                                 i_img_shape,
                                                                 strides):
        if stride == 1:
            if pad == 'valid':
                assert_equal(o_img_size, i_img_size - window_size + 1)
            elif pad == 'full':
                assert_equal(o_img_size, i_img_size + window_size - 1)
            elif pad == 'same_shape':
                assert_equal(o_img_size, i_img_size)

    output_format = DenseFormat(axes=('b', 'c', '0', '1'),
                                shape=(-1,
                                       num_filters,
                                       o_img_shape[0],
                                       o_img_shape[1]),
                                dtype=theano.config.floatX)

    return output_format

class Conv2D(Node):
    '''
    Returns a convolution over 2D space.

    The output axis order is batch-channel-row-column ('b', 'c', '0', '1').
    Output dtype is theano.config.floatX.

    The input may be in any 4D DenseFormat. This will internally be converted
    to the above axis ordering and dtype.
    '''

    def __init__(self,
                 input_node,
                 filter_shape,
                 num_filters,
                 pads,
                 strides=(1, 1),
                 axis_map=None,
                 **kwargs):
        '''
        Parameters
        ----------

        input_node: Node
          A node with 4D output format.

        filter_shape: Sequence
          [NUM_ROWS, NUM_COLUMNS]

        num_filters: int

        pads: str or tuple
          'valid': no zero-padding.
                   output_shape: input_shape - filter_shape + 1
          'full': maximal zero-padding.
                  output_shape: input_shape + filter_shape + 1
          'min': minimal zero-padding.
                 Just enough padding to ensure that all input pixels get used
                 in the output.
          'same_shape': (cuDNN only) Enough padding to preserve image shape.
                       output_shape = input_shape.
          (R, C): (cuDNN only) Pad rows and columns by this many zeros on each
                  side.
                  output_shape = (input_shape[0] + 2R, input_shape[1] + 2C)

        strides: Sequence
          [R, C], default: (1, 1).
          A stride of (R, C) means we apply the filters once every R rows
          and every C columns.

        axis_map: dict, or None.
          Maps axis names from those used by input_node.output_format.axes
          to ('b', 'c', '0', '1'), i.e. batch, channel, row, column.
          Default: None, meaning the input node's axes are already some
                   permutation of ('b', 'c', '0', '1').
        '''

        #
        # Sanity-checks args
        #

        assert_is_instance(input_node, Node)
        _assert_is_shape2d(filter_shape)
        assert_integer(num_filters)
        assert_greater(num_filters, 0)

        filter_shape = numpy.asarray(filter_shape)
        dnn_available = theano.sandbox.cuda.dnn.dnn_available

        if dnn_available():
            if isinstance(pads, basestring):
                assert_in(pads, ('valid', 'full', 'same_shape'))
                if pads == 'same_shape':
                    assert_true((filter_shape % 2 == 1).all(),
                                "If pads == 'same_shape', then filter_shape "
                                "must be odd in both dimensions, but got %s." %
                                str(filter_shape))
                    pads = tuple(numpy.asarray(filter_shape) // 2)
            else:
                _assert_is_shape2d(pads)
                pads = tuple(pads)
        else:
            assert_in(pads, ('valid', 'full'),
                      "cuDNN not found, so falling back to "
                      "theano.tensor.nnet.conv2d(), which only supports "
                      "pads argument values of 'valid' and 'full', "
                      "not %s." % str(pads))

        if strides is not None:
            _assert_is_shape2d(strides)

        if axis_map is not None:
            assert_is_instance(axis_map, dict)
            # axis_map = get_full_axis_map(axis_map)

        if dnn_available():
            assert_equal(len(kwargs), 0,
                         "cuDNN implementation does not accept kwargs. Switch "
                         "to default Theano implementation using Theano "
                         "config option 'optimizer_excluding=conv_dnn'")

        input_format_node = _make_bc01_format_node(input_node, axis_map)

        output_format = _make_bc01_output_format(
            input_format_node.output_format,
            strides,
            filter_shape,
            num_filters,
            pads)

        self.filters = theano.shared(
            # n_filters, n_input_channels, filter_shape[0], filter_shape[1]
            numpy.zeros(([num_filters,
                          input_format_node.output_format.shape[1]] +
                         list(filter_shape)),
                        dtype=theano.config.floatX))


        def make_output_symbol(t_node, filters, pads, strides):
            assert_is_instance(pads, (basestring, tuple))

            strides = tuple(strides)

            if dnn_available():
                dnn_conv = theano.sandbox.cuda.dnn.dnn_conv
                return dnn_conv(img=t_node.output_symbol,
                                kerns=filters,
                                border_mode=pads,
                                subsample=strides,
								conv_mode=kwargs.get('conv_mode', 'conv'))
            else:
                image_shape = list(copy.deepcopy(t_node.output_format.shape))
                assert_equal(image_shape[0], -1)
                image_shape[0] = None
                image_shape = tuple(image_shape)

                conv2d = theano.tensor.nnet.conv2d
                return conv2d(input=t_node.output_symbol,
                              filters=filters,
                              image_shape=image_shape,
                              filter_shape=filters.get_value().shape,  # sic
                              border_mode=pads,
                              subsample=strides,
                              **kwargs)  # pylint: disable=star-args


        if pads not in ('valid', 'full'):
            pads = _get_pads(pads,
                             image_shape=output_format.shape[2:],
                             window_shape=filter_shape,
                             strides=strides)

            assert_true((numpy.asarray(pads) <
                         numpy.asarray(filter_shape)).all())

        output = make_output_symbol(input_format_node,
                                    self.filters,
                                    pads,
                                    strides)

        super(Conv2D, self).__init__([input_node], output, output_format)


# TODO: Add unit tests to make sure that the pad is being handled correctly.
#       Confirm that:
#
#       1) cuDNN pads with -inf when max-pooling, and 0 when mean-pooling?
#       2) The 'min' padding mode works as expected
#       3) the 'pylearn2' padding mode works as expected.
#       4) all cases work fine even when (strides > window_shape).any()
# 2) and 3) can be tested with 7x7 input, 4x4 pool window, and 2x2 pool stride.
# Both 'min' and 'pylearn2' should produce a 3x3 output, wheras pad='valid'
# would produce a 2x2 output.
#
# Fill the first row of input with -5, the last with -6, and the middle with
# -10. When using 'min', the output pool should be
class Pool2D(Node):

    def __init__(self,
                 input_node,
                 window_shape,
                 strides,
                 mode,
                 pad,
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

        pad: string or Sequence
          Either a pair of ints, or one of the following strings:
            'valid': zero padding
            'full': maximal padding
            'min': minimal padding needed to ensure that all input pixels find
                   themselves in some pooling window.
            'same_size': Enough padding to preserve size.
                         Requires odd-numbered rows and cols in window_shape.
            'pylearn2': Similar in intent to 'min', except this is how
                        pylearn2.mlp.ConvElemwise does it. It adds padding to
                        the far end of the input (maximal rows and columns), as
                        much as is needed to ensure that all input pixels
                        affect the pool output. This is different from 'min',
                        which adds this padding to both sides of the input
                        image.

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

        if pad != 'pylearn2':
            output_format = _make_bc01_output_format(
                bc01_input_format=input_format_node.output_format,
                strides=strides,
                window_shape=window_shape,
                num_filters=input_format_node.output_format.shape[1],
                pad=pad)

            pads = _get_pads(pad,
                             image_shape=input_format_node.output_format.shape[2:],
                             window_shape=window_shape,
                             strides=strides)

            assert_true((numpy.asarray(pads) < numpy.asarray(window_shape)).all())

            output_symbol = theano.sandbox.cuda.dnn.dnn_pool(
                img=input_format_node.output_symbol,
                ws=tuple(window_shape),
                stride=tuple(strides),
                mode=mode,
                pad=pads)

            super(Pool2D, self).__init__([input_node],
                                         output_symbol,
                                         output_format)

        else:
            image_shape = numpy.asarray(input_format_node.output_format.shape[2:])
            window_shape = numpy.asarray(window_shape)
            strides = numpy.asarray(strides)
            overflow = ((image_shape - window_shape) + 1 - 1) % strides
            single_sided_pads = strides - overflow
            single_sided_pads[single_sided_pads == strides] = 0

            bc01_symbol = input_format_node.output_symbol
            T = theano.tensor
            floatX = theano.config.floatX
            padded_image = T.alloc(T.constant(-numpy.inf if mode == 'max' else 0,
                                              dtype=floatX),
                                   bc01_symbol.shape[0],
                                   bc01_symbol.shape[1],
                                   bc01_symbol.shape[2] + single_sided_pads[0],
                                   bc01_symbol.shape[3] + single_sided_pads[1])

            padded_image = T.set_subtensor(padded_image[:,
                                                        :,
                                                        :bc01_symbol.shape[2],
                                                        :bc01_symbol.shape[3]],
                                           bc01_symbol)

            self.DEBUG_padded_image_symbol = padded_image

            output_symbol = theano.sandbox.cuda.dnn.dnn_pool(
                img=padded_image,
                ws=tuple(window_shape),
                stride=tuple(strides),
                mode=mode,
                pad=(0, 0))

            bc01_shape = input_format_node.output_format.shape
            padded_image_shape = (numpy.asarray(bc01_shape[2:]) +
                                  single_sided_pads)

            output_image_shape = (((padded_image_shape - window_shape) + 1 - 1) //
                                  strides) + 1
            # pdb.set_trace()
            # assert_array_equal(output_image_shape * strides - 1 + window_shape,
            #                    padded_image_shape)
            output_format = DenseFormat(axes=('b', 'c', '0', '1'),
                                        shape=(bc01_shape[0],
                                               bc01_shape[1],
                                               output_image_shape[0],
                                               output_image_shape[1]),
                                        dtype=floatX)

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

    def _get_output_bf_node(self,
                            input_bf_node,
                            input_bf_format,
                            output_bf_format):
        softmaxes = theano.tensor.nnet.softmax(input_bf_node.output_symbol)
        return Node(input_bf_node, softmaxes, output_bf_format)


class ReLU(Node):
    '''
    Elementwise linear rectifier (zeros any negative elements).

    Preserves format.
    '''
    def __init__(self, input_node):
        input_symbol = input_node.output_symbol
        # As in pylearn2/models/mlp.py, which uses:
        # "linear_response * (linear_response > 0.) + self.left_slope * \
        #  linear_response * (linear_response < 0.)"
        output_symbol = input_symbol * (input_symbol > 0.0)

        assert_equal(output_symbol.dtype, input_symbol.dtype)
        # output_symbol = theano.tensor.switch(input_symbol > 0.0,
        #                                      input_symbol,
        #                                      0.0)
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


class AffineLayer(Function1dTo1d):
    '''
    A sequence of affine -> ReLU.
    '''

    def __init__(self,
                 input_node,
                 output_format,
                 input_to_bf_map=None,
                 bf_to_output_map=None):
        '''
        Parameters
        ----------
        See docs for simplelearn.nodes.AffineTransform constructor.
        '''
        super(AffineLayer, self).__init__(input_node,
                                          output_format,
                                          input_to_bf_map,
                                          bf_to_output_map)


    def _get_output_bf_node(self,
                            input_bf_node,
                            input_bf_format,
                            output_bf_format):
        self.affine_node = AffineTransform(input_bf_node, output_bf_format)
        self.relu_node = ReLU(self.affine_node)
        return self.relu_node


class SoftmaxLayer(Function1dTo1d):
    '''
    A sequence of affine -> softmax
    '''

    def __init__(self,
                 input_node,
                 output_format,
                 input_to_bf_map=None,
                 bf_to_output_map=None):
        super(SoftmaxLayer, self).__init__(input_node,
                                           output_format,
                                           input_to_bf_map,
                                           bf_to_output_map)

    def _get_output_bf_node(self,
                            input_bf_node,
                            input_bf_format,
                            output_bf_format):
        self.affine_node = AffineTransform(input_bf_node, output_bf_format)
        self.softmax_node = Softmax(self.affine_node)
        return self.softmax_node


class Conv2DLayer(Node):
    '''
    A sequence of conv2d -> channel-wise bias -> ReLU -> pool2d
    '''
    def __init__(self,
                 input_node,
                 filter_shape,
                 num_filters,
                 conv_pads,
                 pool_window_shape,
                 pool_strides,
                 pool_mode='max',
                 pool_pads='min',
                 filter_strides=(1, 1),
                 channel_axis='c',
                 axis_map=None,
                 **kwargs):
        '''
        Parameters
        ----------
        input_node, filter_shape, num_filters, filter_strides, conv_pads,
        axis_map, kwargs:
          See equivalent arguments of simplelearn.nodes.Conv2D constructor.

        pool_window_shape, pool_strides, pool_mode:
          See equivalent arguments of simplelearn.nodes.Pool2D constructor.
        '''
        assert_in(channel_axis, input_node.output_format.axes)

        self.conv2d_node = Conv2D(input_node,
                                  filter_shape,
                                  num_filters,
                                  conv_pads,
                                  strides=filter_strides,
                                  axis_map=axis_map,
                                  **kwargs)

        # Implements channel-wise bias by collapsing non-channel axes into
        # batch axes in the bias node. Bias node then adds a bias per
        # channel, then reshapes output to original shape.

        # TODO: replace this with a solution that doesn't involve a potentially
        # expensive transpose? We just need to take a <num_channels>-sized
        # bias vector, dimshuffle it to add singleton axes along other axes,
        # then add it. No need to transpose an entire feature map.
        non_channel_axes = tuple(axis for axis
                                 in self.conv2d_node.output_format.axes
                                 if axis != channel_axis)
        self.bias_node = Bias(self.conv2d_node,
                              output_format=self.conv2d_node.output_format,
                              input_to_bf_map={non_channel_axes: 'b',
                                               channel_axis: 'f'},
                              bf_to_output_map={'b': non_channel_axes,
                                                'f': channel_axis})
        assert_equal(self.bias_node.params.get_value().shape, (num_filters, ))

        self.relu_node = ReLU(self.bias_node)
        self.pool2d_node = Pool2D(input_node=self.relu_node,
                                  window_shape=pool_window_shape,
                                  strides=pool_strides,
                                  mode=pool_mode,
                                  pad=pool_pads)

        super(Conv2DLayer, self).__init__([input_node],
                                          self.pool2d_node.output_symbol,
                                          self.pool2d_node.output_format)


def _normal_distribution_pdf(inputs, mean, covariance):
    '''
    Computes the normal distribution PDF (aka Gaussian).

    Parameters
    ----------
    inputs: numpy.ndarray
      Shape: (N, ) or (N, M), dtype: floating-point

    mean: numpy.ndarray
      Shape: (N, ), dtype: floating-point

    covariance: numpy.ndarrray
      Shape: (N, N) dtype: floating-point
    '''
    cast_float = numpy.cast[float]

    if len(inputs.shape) == 1:
        inputs = inputs[numpy.newaxis, :]
    else:
        assert_equal(len(inputs.shape), 2)

    inputs = cast_float(inputs)
    ndim = inputs.shape[1]

    if len(mean.shape) == 1:
        mean = mean[numpy.newaxis, :]
    else:
        assert_equal(len(mean.shape), 2)

    assert_equal(mean.shape[1], ndim)

    mean = cast_float(mean)

    assert_all_equal(covariance.shape, ndim)
    # No need to check for positive definiteness. If it isn't, numpy.linalg.inv
    # will throw an exception below.

    #
    # Done sanity-checking inputs
    #

    covariance = cast_float(covariance)
    inv_covariance = numpy.linalg.inv(covariance)

    rt = inputs - mean
    rt_ic = numpy.dot(rt, inv_covariance)
    rt_ic_r = (rt_ic * rt).sum(axis=1)
    exp_term = numpy.exp(-0.5 * rt_ic_r)

    denominator = numpy.sqrt((2 * numpy.pi) ** ndim *
                             numpy.linalg.det(covariance))

    return exp_term / denominator

def _make_2d_gaussian_filter(filter_shape, covariance=None, dtype=None):
    '''
    Returns a 2D Gaussian filter.
    '''
    assert_equal(len(filter_shape), 2)
    assert_all_integers(filter_shape)
    assert_all_greater(filter_shape, 0)

    filter_shape = numpy.asarray(filter_shape, dtype=int)

    if covariance is None:
        # Set std. deviation so that filter window is 4 std.deviations across.
        # This keeps most of the probability mass within the window.
        std_deviations = numpy.asarray(filter_shape, dtype=dtype) / 4.0

        covariance = numpy.diag(std_deviations ** 2.0)
    else:
        assert_equal(covariance.shape, (2, 2))

    if dtype is None:
        dtype = theano.config.floatX


    xys = numpy.indices(filter_shape)      # shape: (2, nrows, ncols)
    xys = numpy.transpose(xys, (1, 2, 0))  # shape: (nrows, ncols, 2)
    xys = xys.reshape((-1, 2))             # shape: (nrows * ncols, 2)
    mean = (filter_shape // 2)             # shape: (2, )

    # shape: (nrows * ncols, )
    gaussian_elems = _normal_distribution_pdf(xys, mean, covariance)

    cast = numpy.cast[dtype]
    return cast(gaussian_elems.reshape(filter_shape))  # shape: (nrows, ncols)


class Lcn(Node):
    '''
    LeCun-style local contrast normalization.

    Internally uses Conv2D, so the output axis order is 'bc01'.

    Requires floating-point input.

    Performs LCN on each input channel independently.
    '''

    def __init__(self,
                 input_node,
                 filter_shape=(7, 7),
                 threshold=1e-4,
                 axis_map=None):
        assert_floating(input_node.output_symbol)

        assert_equal(len(filter_shape), 2)
        assert_all_integers(filter_shape)
        assert_all_greater(filter_shape, 0)

        assert_greater_equal(threshold, 0.0)

        if axis_map is not None:
            assert_is_instance(axis_map, dict)

        #
        # Done sanity-checking args
        #

        # Transposes to bc01 axis order
        bc01_node = _make_bc01_format_node(input_node, axis_map)

        # Collapses b and c axes into b, so that the different channels are
        # treated as separate single-channel images.
        #
        # Leaves the 'c' axis intact as a singleton dimension.
        def make_channel_separator(node):
            bc01 = ('b', 'c', '0', '1')
            assert_equal(node.output_format.axes, bc01)

            shape = node.output_format.shape
            fmt = DenseFormat(axes=bc01,
                              shape=(-1, 1, shape[2], shape[3]),
                              dtype=node.output_format.dtype)
            return FormatNode(node, fmt, axis_map={('b', 'c'): ('b', 'c')})

        separated_channels_node = make_channel_separator(bc01_node)
        separated_channels = separated_channels_node.output_symbol

        # Apply single-channel 2D convolution with a Gaussian filter
        filters = _make_2d_gaussian_filter(filter_shape)[numpy.newaxis,
                                                         numpy.newaxis,
                                                         :,
                                                         :]

        filters /= filters.sum()

        # 1-channel to 1-channel convolution with a gaussian filter
        blur_node = Conv2D(input_node=separated_channels_node,
                           filter_shape=filter_shape,
                           num_filters=1,
                           pads='same_shape')
        blur_node.filters.set_value(filters)

        high_pass = separated_channels - blur_node.output_symbol

        squares_node = Node(separated_channels_node,
                            theano.tensor.sqr(separated_channels),
                            separated_channels_node.output_format)

        # Same convolution as above, but applied on squares_node
        blur_squares_node = Conv2D(input_node=squares_node,
                                   filter_shape=filter_shape,
                                   num_filters=1,
                                   pads='same_shape')
        blur_squares_node.filters.set_value(filters)
        local_norms = theano.tensor.sqrt(blur_squares_node.output_symbol)

        # shape: n_images * n_channels, 1, 1, 1
        global_norms = local_norms.mean(axis=[2, 3], keepdims=True)

        denominator = theano.tensor.largest(global_norms, local_norms)
        denominator = theano.tensor.maximum(denominator, threshold)
        result_with_separated_channels = high_pass / denominator

        result_with_separated_channels_node = Node(
            blur_squares_node,  # choice of node's input doesn't matter here
            result_with_separated_channels,
            separated_channels_node.output_format)

        # Monochrome -> multi-channel images
        def make_channel_gatherer(node):
            bc01 = ('b', 'c', '0', '1')
            assert_equal(node.output_format.axes, bc01)
            assert_equal(node.output_format.shape[1], 1)

            output_shape = bc01_node.output_format.shape

            fmt = DenseFormat(axes=bc01,
                              shape=output_shape,
                              dtype=node.output_format.dtype)
            return FormatNode(node, fmt, axis_map={('b', 'c'): ('b', 'c')})

        result = make_channel_gatherer(result_with_separated_channels_node)

        super(Lcn, self).__init__(input_node,
                                  result.output_symbol,
                                  result.output_format)


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

                return (DenseFormat(shape=(batch_size, feature_size),
                                    axes=('b', 'f'),
                                    dtype=fmt.dtype),
                        non_b_axes)

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
        assert_integer(target_node.output_symbol)
        assert_is_instance(softmax_node, (Softmax, SoftmaxLayer))

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
