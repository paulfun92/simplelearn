'''
Higher-level Nodes commonly used in building models.
'''

from simplelearn.nodes import AffineTransform, ReLU, Function1dTo1d, Dropout


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
