'''
Higher-level Nodes commonly used in building models.
'''

from simplelearn.nodes import AffineTransform, ReLU, Function1dTo1d, Dropout

class AffineLayer(Function1dTo1d):
    '''
    A sequence of [dropout ->] affine -> ReLU.
    '''

    def __init__(self,
                 theano_rng,
                 input_node,
                 output_format,
                 input_to_bf_map=None,
                 bf_to_output_map=None,
                 dropout_include_rate=0.5):
        '''
        Parameters
        ----------
        input_node, output_format, input_to_bf_map, bf_to_output_map:
          See docs for simplelearn.nodes.Function1dTo1d constructor.

        theano_rng, dropout_include_rate:
          See docs for simplelearn.nodes.Dropout constructor.
        '''
        super(AffineLayer, self).__init__(
            input_node,
            output_format,
            input_to_bf_map,
            bf_to_output_map,
            theano_rng=theano_rng,
            dropout_include_rate=dropout_include_rate)

    def _get_output_bf_node(self,
                            input_bf_node,
                            output_bf_format,
                            **kwargs):
        dropout_include_rate = kwargs['dropout_include_rate']

        input_to_affine = input_bf_node

        if dropout_include_rate == 1.0:
            self.dropout_node = None
        else:
            self.dropout_node = Dropout(input_bf_node,
                                        dropout_include_rate,
                                        theano_rng)
            input_to_affine = self.dropout_node

        self.affine_node = AffineTransform(input_to_affine, output_bf_format)
        self.relu_node = ReLU(self.affine_node)

        return self.relu_node


class Conv2DLayer(Node):
    '''
    A sequence of [dropout ->] conv2d -> channel-wise bias -> ReLU -> pool2d
    '''
    def __init__(self,
                 theano_rng,
                 input_node,
                 filter_shape,
                 num_filters,
                 conv_pads,
                 pool_window_shape,
                 pool_strides,
                 pool_mode=None,
                 filter_strides=(1, 1),
                 dropout_include_rate=0.5,
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

        theano_rng, dropout_include_rate:
          See equivalent arg of simplelearn.nodes.Dropout constructor.
        '''

        input_to_conv = input_node

        if dropout_include_rate == 1.0:
            self.dropout_node = None
        else:
            self.dropout_node = Dropout(last_node,
                                        dropout_include_rate,
                                        theano_rng)
            input_to_conv = self.dropout_node

        self.conv2d_node = Conv2D(input_to_conv,
                                  filter_shape,
                                  num_filters,
                                  pads,
                                  filter_strides=(1, 1),
                                  axis_map=None,
                                  **kwargs)

        non_channel_axes = tuple(axis for axis
                                 in self.conv2d_node.output_format.axes
                                 if axis != channel_axis)
        self.bias_node = Bias(self.conv2d_node,
                              output_format=top_layer.output_format,
                              input_to_bf_map={non_channel_axes: 'b',
                                               channel_axis: 'f'},
                              bf_to_output_map={'b': non_channel_axes,
                                                'f': channel_axis})
        assert_equal(top_layer.params.get_value().shape[1], num_filters)

        self.relu_node = ReLU(self.bias_node)
        self.pool2d_node = Pool2D(input_node=self.relu_node,
                                  window_shape=pool_shape,
                                  strides=pool_strides,
                                  mode=pool_mode)

        super(Conv2DLayer, self).__init__([input_node],
                                          self.pool2d_node.output_symbol,
                                          self.pool2d_node.output_format)

def add_affine_layer(input_node, output_size_or_format):
    '''
    Stacks a new AffineTransform and ReLU node onto the input_node.

    Parameters
    ----------
    input_node: Node
      The node to build the layers on top of.

    output_size_or_format: int or DenseFormat
      If an int, the output will have format ('b', 'f'), (-1, output_size).
      If a DenseFormat, this will be used as the output format.

    Returns
    -------
    rval: ReLU
      The new top node.
    '''
    if isinstance(output_size_or_format, DenseFormat):
        output_format = output_size_or_format
    elif numpy.issubdtype(output_size_or_format, numpy.integer):
        output_format = DenseFormat(shape=(-1, output_size_or_format),
                                    axes=('b', 'f'),
                                    dtype=None)
    else:
        TypeError("Expected output_size_or_format to be an int or a "
                  "DenseFormat, not a %s." % type(output_size_or_format))

    affine = AffineTransform(input_node, output_format)
    return ReLU(affine.output_node)


def build_fc_classifier(input_node, sizes):
    '''
    Builds a stack of fully-connected layers followed by a Softmax.

    Parameters
    ----------
    input_node: Node
      The node to build the stack on.

    sizes: Sequence
      A sequence of ints, indicating the output sizes of each layer.
      The last int is the number of classes.
    '''

    for size in sizes:
        input_node = add_affine_layer(size)

    return Softmax(input_node, DenseFormat(axes=('b', 'f'),
                                           shape=(-1, sizes[-1]),
                                           dtype=None))

def build_fc_trainer(input_node, label_node, sizes, weight_ranges):
    softmax = build_fc_classifer(input_node, sizes)

    # go down the stack from softmax to input_node, initializing the weights
    raise NotImplementedError()
