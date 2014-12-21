"""
Classes that represent different data formats.

Here, "data format" can include things like memory layout, axis order,
and floating point precision, which don't qualitatively change the data.

Conversion between two formats should be approximately reversible. For
example, converting from float64 to float32 is lossy but permitted, while
converting from float to int, or complex to float, is not.
"""

import numpy
import theano
from theano.gof.op import get_debug_values
from theano.tensor import TensorType
from theano.sandbox.cuda.type import CudaNdarrayType
from simplelearn.utils import safe_izip, flatten


class Format(object):
    """
    Abstract class. Represents the format of a numeric or symbolic data batch.

    "format" means things like axis order, and optionally, dtype.

    Parameters
    ----------
    dtype: str or numpy.dtype or NoneType

      default: None.

      If None, this format won't specify a dtype. Converting a batch from
      another format will leave the batch's existing dtype unchanged. On the
      other hand, you will need to specify a dtype when calling make_batch.
    """

    def __init__(self, dtype=None):
        self.dtype = dtype

    @property
    def dtype(self):
        if str(self._dtype) == 'floatX':
            result = numpy.dtype(theano.config.floatX)
        else:
            result = self._dtype

        assert result is None or isinstance(result, numpy.dtype)
        return result

    @dtype.setter
    def dtype(self, new_dtype):
        if new_dtype is None or str(new_dtype) == 'floatX':
            self._dtype = new_dtype
        else:
            self._dtype = numpy.dtype(new_dtype)

    @staticmethod
    def is_symbolic(batch):
        """
        Returns True i.f.f, batch is a (Theano) symbol.

        Returns False if batch is a (numpy) numeric array.

        Raises a TypeError if data is some other type.

        Parameters
        ----------
        data: numpy.ndarray, numpy.memmap, or theano.gof.Variable

        Returns
        -------
        rval: str
           "symbolic", "numeric", or "unrecognized".
        """

        if isinstance(batch, theano.gof.Variable):
            return True
        elif (isinstance(batch, (numpy.ndarray, numpy.memmap)) or
              type(batch) == 'CudaNdarray'):
            return False
        else:
            raise TypeError("Unrecognized batch type %s." % type(batch))

    def _is_equivalent(self, target_format):
        """
        Returns True if converting from self to target_format is a no-op.
        """
        raise NotImplementedError("%s._is_equivalent() not yet implemented." %
                                  type(self))

    def convert(self, batch, target_format, output=None):
        """
        Formats a data batch in this format to the target_format.

        Output argument only supported for numeric batches.

        This function just calls self._convert(), check()-ing its
        inputs and outputs.

        Note that if output is None and self._is_equivalent(target_format), the
        batch is returned as-is without calling self._convert().

        Parameters
        ----------
        batch: numpy.ndarray-like or theano.gof.Variable

        target_format: DataFormat

        output: NoneType, or numpy.ndarray-like
          Optional output variable. Can only be supplied if batch is numeric.
          If provided, this function won't return anything.
          If omitted, this function will return the formatted batch.

        Returns
        -------
        rval: numpy.ndarray or theano.gof.Variable
          The formatted batch. When formatting between equivalent formats, this
          will be <batch>.
        """

        self.check(batch)

        if output is None and self._is_equivalent(target_format):
            return batch

        if target_format.dtype is not None and \
           not numpy.can_cast(batch.dtype,
                              target_format.dtype,
                              casting='same_kind'):  # Allows float64->float32
            raise TypeError("Can't cast from %s to %s." %
                            (batch.dtype, target_format.dtype))

        if self.is_symbolic(batch) and output is not None:
            raise ValueError("You can't provide an output argument when "
                             "data is symbolic.")

        result = self._convert(batch, target_format, output)

        if self.is_symbolic(batch) != self.is_symbolic(result):
            def symbolic_or_numeric(batch):
                return "symbolic" if self.is_symbolic(batch) else "numeric"

            raise TypeError("Expected %(self_type)s._convert(<%(data_type)s "
                            "tensor>) to return a <%(data_type)s tensor>, but "
                            "got a %(result_type)s instead." %
                            dict(self_type=type(self),
                                 data_type=symbolic_or_numeric(batch),
                                 result_type=symbolic_or_numeric(result)))

        target_format.check(result)

        return result

    def _convert(self, batch, target_format, output):
        """
        Implementation of format(). See that method's description for specs.

        Must throw an exception (typically TypeError or ValueError) if
        conversion to target_format is not supported.
        """

        raise NotImplementedError("%s._convert() not yet implemented." %
                                  self.__class__)

    def check(self, batch):
        """
        Checks to see if batch fits this format's specs.

        If not, throws a ValueError or TypeError with an informative message.

        Parameters
        ----------
        batch: theano or numpy variable.

        Returns
        -------
        nothing.
        """

        # Checks whether the batch is a known symbolic or numeric type.
        self.is_symbolic(batch)

        if self.dtype is not None:
            if batch.dtype != self.dtype:
                raise TypeError("batch's dtype (%s) doesn't match this %s's "
                                "dtype (%s)." %
                                (batch.dtype, type(self), self.dtype))
        self._check(batch)

    def _check(self, batch):
        """
        Implements check(). See that method's documentation.
        """
        raise NotImplementedError("%s._check() not yet implemented." %
                                  self.__class__)

    def make_batch(self, is_symbolic, batch_size=None, dtype=None):
        """
        Makes a numeric or symbolic batch.

        Parameters
        ----------
        is_symbolic: bool
          if True, return a symbolic batch. Otherwise return a numeric batch.

        batch_size: int
          Number of examples in numeric batch. Omit if is_symbolic is True.

        dtype: str/numpy.dtype, or NoneType
          A numpy/theano dtype. Required if self.dtype is None.
          Prohibited otherwise.
        """

        # Sanity-checks batch_size
        if batch_size is not None:
            if not numpy.issubdtype(type(batch_size), numpy.integer):
                raise TypeError("batch_size must be an integer, not an %s." %
                                type(batch_size))
            elif batch_size < 0:
                raise ValueError("batch_size must be non-negative, not %d." %
                                 batch_size)

        # checks is_symbolic vs batch_size
        if is_symbolic and batch_size is not None:
            raise ValueError("Can't supply a batch_size when is_symbolic "
                             "is True.")
        if not is_symbolic and batch_size is None:
            raise ValueError("Must supply a batch_size when is_symbolic "
                             "is False.")

        # Checks dtype vs self.dtype
        if dtype is None:
            if self.dtype is None:
                raise TypeError("Since this %s doesn't specify a dtype, you "
                                "must provide a dtype argument." % type(self))
        elif self.dtype is not None:
            raise TypeError("Can't supply a dtype argument because this %s's "
                            "dtype is not None (it's %s)." %
                            (type(self), self.dtype))
        elif dtype == 'floatX':
            dtype = theano.config.floatX
        else:
            dtype = numpy.dtype(dtype)

        result = self._make_batch(is_symbolic, batch_size, dtype)

        self.check(result)
        return result

    def _make_batch(self, is_symbolic, batch_size, dtype):
        """
        Implements make_batch. See that method's documentation().

        Parameters
        ----------
        dtype: numpy.dtype
          The dtype of the batch to create. Unlike the argument to
          make_batch(), this will never be None, or 'floatX'.
        """
        raise NotImplementedError("%s._make_batch() not yet implemented." %
                                  type(self))


class DenseFormat(Format):
    """
    Format for fixed-sized dense data.

    Examples:

      # vectors of size 100:
      vector_format = DenseFormat(axes=('b', 'f'),
                                  shape=(100, -1))

      # 640x480 RGB images, indexed as [channel, row, column, batch]:
      image_format = DenseFormat(axes=('f', '0', '1', 'b'),
                                 shape=(3, 480, 640, -1))

    Parameters
    ----------

    axes: sequence
      A sequence of strings. Each string is the canonical name of an axis.

    shape: sequence
      A sequence of dimension sizes. Batch axis gets the dummy size -1.

    dtype: see superclass' docs.
    """

    def __init__(self, axes, shape, dtype):
        super(DenseFormat, self).__init__(dtype=dtype)

        if not all(isinstance(axis, str) for axis in axes):
            raise TypeError("axes contained non-strings: %s" %
                            str(tuple(axes)))

        if len(frozenset(axes)) < len(axes):
            raise ValueError("axes contained duplicate elements: %s" %
                             str(tuple(axes)))

        if not all(numpy.issubdtype(type(size), 'int') for size in shape):
            raise TypeError("shape contained non-ints: %s" % str(shape))

        if len(axes) != len(shape):
            raise ValueError("axes and shape's lengths differ (%s vs %s)." %
                             (str(axes), str(shape)))

        if 'b' in axes:
            b_size = shape[axes.index('b')]

            if b_size != -1:
                raise ValueError("Shape element corresonding to 'b' axis must "
                                 "be given the dummy size -1, not %d. "
                                 "shape: %s axes: %s" %
                                 (b_size, str(shape), str(axes)))

        if any(size < 0 and axis is not 'b'
               for size, axis
               in safe_izip(shape, axes)):
            raise ValueError("Negative size in non-batch dimension. "
                             "shape: %s, axes: %s" %
                             (str(shape), str(axes)))

        self.axes = tuple(axes)
        self.shape = tuple(shape)

    def _make_batch(self, is_symbolic, batch_size, dtype=None):
        if 'b' not in self.axes:
            raise ValueError("This format has no batch ('b') axis.")

        if is_symbolic:
            raise NotImplementedError()
        else:
            shape = list(self.shape)
            shape[self.axes.index('b')] = batch_size

            dtype = dtype if dtype is not None else self.dtype
            if dtype is None:
                raise ValueError("When self.dtype is None, you must provide a "
                                 "dtype argument to make_batch")

            return numpy.zeros(shape, dtype)

    def _check(self, batch):
        super(DenseFormat, self)._check(batch)

        is_symbolic = self.is_symbolic(batch)

        if is_symbolic:
            if not isinstance(batch.type, (TensorType, CudaNdarrayType)):
                raise TypeError("Expected a TensorType or CudaNdarrayType, "
                                "not a %s." % batch.type)

            for val in get_debug_values(batch):
                self._check(val)

        if batch.ndim != len(self.axes):
            raise ValueError("Expected a 2-D tensor, but found %d" %
                             batch.ndim)

        if not is_symbolic:
            for expected_size, size, axis in safe_izip(self.shape,
                                                       batch.shape,
                                                       self.axes):
                if axis != 'b' and size != expected_size:
                    raise ValueError("Mismatch between this format' size of "
                                     "axis %s (%d) and batch's corresponding "
                                     "size %d." %
                                     (expected_size,
                                      axis,
                                      size))

    def _convert(self, batch, target_format, output_batch, **kwargs):
        """
        Converts a batch to another format.

        When converting to another DenseFormat, if the axes are the same,
        but in a different order, this will transpose the axes for you.

        Example 1: same axes, different order

          from = DenseFormat(axes=('a, 'b', 'c'), sizes=(3, 3, 3))
          to = DenseFormat(axes=('b', 'c', 'a'), sizes=(3, 3, 3))
          from.format(batch, to)  # transposes axes correctly

        If the axes are different, you must supply an "axis_map" dict to
        clarify which axes of self correspond with which axes of target_format.

        Example 2: same # of axes, different names.

          from = DenseFormat(axes=('a', 'b'), sizes=(3, 3))
          to = DenseFormat(axes('c', 'd'), sizes(3, 3))
          from.format(batch, to, axis_map={'a': 'd',
                                           'b': 'c'})

        Example 3: different # of axes.

          images = DenseFormat(axes=('f', '0', '1', 'b'),
                               sizes=(1, 10, 10, 100))

          vectors = DenseFormat(axes=('b', 'f'),
                                sizes=(100, 100))

          images.format(batch, vectors, axis_map={'b': 'b',
                                                  'f': ('0', '1', 'f')}

          axis_map always maps from the axis names of the format with
          fewer axes, to the names of the format with more axes. In this
          case, it's mapping from vectors' names to images' names, even
          though we're formatting from images to vectors.

          The line "'f': ('0', '1', 'f')" indicates that the image batch's
          ('f', '0', '1') dimensions will first be transposed to ('0', '1',
          'f'), before being flattened to a vector with a single dimension 'f'.

        Parameters
        ----------
        axis_map: dict
          If mapping from self.axes to target_format.axes is ambiguous,
          you must supply axis_map. This is a dict that maps
        """

        if isinstance(target_format, DenseFormat):
            if numpy.prod(self.shape) != numpy.prod(target_format.shape):
                raise ValueError("Total batch size of self and target_format "
                                 "differ (%d vs %d)." %
                                 (numpy.prod(self.shape),
                                  numpy.prod(target_format.shape)))

            if frozenset(self.axes) != frozenset(target_format.axes):
                if 'axis_map' not in kwargs:
                    raise ValueError("self.axes contain different axes than "
                                     "target_format.axes. You therefore must "
                                     "supply an 'axis_map' argument.")

                axis_map = kwargs['axis_map']

                if len(self.axes) > len(target_format.axes):
                    more, fewer = (self, target_format)
                else:
                    more, fewer = (target_format, self)

                if frozenset(fewer.axes) != frozenset(axis_map.iterkeys()):
                    raise ValueError("axis_map's keys %s don't correspond to "
                                     "the axes of the format with fewer "
                                     "dimensions %s" %
                                     (frozenset(axis_map.iterkeys()),
                                      frozenset(fewer.axes)))

                expanded_fewer_axes = flatten(axis_map[x] for x in fewer.axes)

                if self.is_symbolic(batch):
                    raise NotImplementedError()
                else:

                    def transpose(batch, from_axes, to_axes):
                        transposed_indices = tuple(from_axes.index(x)
                                                   for x in to_axes)
                        return batch.transpose(transposed_indices)

                    if len(self.axes) > len(target_format.axes):  # more->fewer
                        result = transpose(batch,
                                           self.axes,
                                           expanded_fewer_axes)
                        result = result.reshape(target_format.shape)
                    else:  # fewer->more
                        expanded_fewer_shape = (more.shape[more.axes.index(x)]
                                                for x in expanded_fewer_axes)
                        batch = batch.reshape(expanded_fewer_shape)
                        result = transpose(batch,
                                           expanded_fewer_axes,
                                           target_format.axes)

                    if output_batch is not None:
                        output_batch[...] = result
                        return output_batch
                    else:
                        return result

        else:
            raise NotImplementedError("Converting from %s to %s not yet "
                                      "implemented." % (type(self),
                                                        type(target_format)))
