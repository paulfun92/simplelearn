"""Classes that represent different data formats.

"Data format" includes things that don't qualitatively change the
data, such as memory layout, axis order, and floating point precision.

Conversion between two formats should be approximately reversible. For
example, converting from float64 to float32 is lossy but permitted, while
converting from float to int, or complex to float, is not.
"""

__author__ = "Matthew Koichi Grimes"
__email__ = "mkg@alum.mit.edu"
__copyright__ = "Copyright 2014"
__license__ = "Apache 2.0"


import numpy
from nose.tools import assert_equal, assert_greater_equal, assert_is_instance
import theano
from theano.gof.op import get_debug_values
from theano.tensor import TensorType
from theano.sandbox.cuda.type import CudaNdarrayType
from simplelearn.utils import safe_izip, flatten, check_is_subdtype


class Format(object):
    """
    The format of a numeric or symbolic data batch.

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
        # pylint: disable=attribute-defined-outside-init

        if new_dtype is None or str(new_dtype) == 'floatX':
            self._dtype = new_dtype
        else:
            self._dtype = numpy.dtype(new_dtype)

    @staticmethod
    def is_symbolic(batch):
        """
        Returns True i.f.f, batch is a (Theano) symbol.

        Returns False i.f.f. batch is a (numpy) numeric array.

        Raises a TypeError if data is some other type.

        Parameters
        ----------
        data: numpy.ndarray, numpy.memmap, or theano.gof.Variable

        Returns
        -------
        rval: bool
        """

        if isinstance(batch, theano.gof.Variable):
            return True
        elif (isinstance(batch, (numpy.ndarray, numpy.memmap)) or
              type(batch) == 'CudaNdarray'):
            return False
        else:
            raise TypeError("Unrecognized batch type %s." % type(batch))

    @staticmethod
    def is_numeric(batch):
        """
        Returns True i.f.f. batch is a (numpy) numeric array.

        Returns False i.f.f. batch is a (Theano) symbol.

        Raises a TypeError if data is some other type.

        Parameters
        ----------
        data: numpy.ndarray, numpy.memmap, or theano.gof.Variable

        Returns
        -------
        rval: bool
        """
        return not Format.is_symbolic(batch)

    def requires_conversion(self, target_format):
        """
        Returns True if converting from self to target_format is a no-op.
        """
        raise NotImplementedError("%s.requires_conversion() not yet implemented." %
                                  type(self))

    def convert(self, batch, output_format, output=None, **kwargs):
        """
        Formats a data batch in this format to the output_format.

        Output argument only supported for numeric batches.

        This function just calls self._convert(), check()-ing its
        inputs and outputs.

        Note that if output is None and self.requires_conversion(output_format), the
        batch is returned as-is without calling self._convert().

        Parameters
        ----------
        batch: numpy.ndarray-like or theano.gof.Variable

        output_format: DataFormat

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

        if output is None and not self.requires_conversion(output_format):
            return batch

        if output_format.dtype is not None and \
           not numpy.can_cast(batch.dtype,
                              output_format.dtype,
                              casting='same_kind'):  # Allows float64->float32
            raise TypeError("Can't cast from %s to %s." %
                            (batch.dtype, output_format.dtype))

        if self.is_symbolic(batch) and output is not None:
            raise ValueError("You can't provide an output argument when "
                             "data is symbolic.")

        result = self._convert(batch, output_format, output, **kwargs)

        output_format.check(result)

        if self.is_symbolic(batch) != self.is_symbolic(result):
            def symbolic_or_numeric(batch):
                return "symbolic" if self.is_symbolic(batch) else "numeric"

            raise TypeError("Expected %(self_type)s._convert(<%(data_type)s "
                            "tensor>) to return a <%(data_type)s tensor>, but "
                            "got a %(result_type)s instead." %
                            dict(self_type=type(self),
                                 data_type=symbolic_or_numeric(batch),
                                 result_type=symbolic_or_numeric(result)))

        return result

    def _convert(self, batch, output_format, output, **kwargs):
        """
        Implementation of format(). See that method's description for specs.

        Must throw an exception (typically TypeError or ValueError) if
        conversion to output_format is not supported.
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
                                  type(self))

    def make_batch(self, is_symbolic, batch_size=None, dtype=None, name=None):
        """
        Makes a numeric or symbolic batch.

        May later split this into two functions: make_numeric_batch and
        make_symbolic_batch. Keep it like this for now. If implementations of
        _make_batch end up not sharing much logic between is_symbolic=True
        and is_symbolic=False, then maybe split it.

        Parameters
        ----------
        is_symbolic: bool
          if True, return a symbolic batch. Otherwise return a numeric batch.

        batch_size: int
          Number of examples in numeric batch. Omit if is_symbolic is True.

        dtype: str/numpy.dtype, or NoneType
          A numpy/theano dtype. Required if self.dtype is None.
          Prohibited otherwise.

        name: str, or NoneType
          Name of the symbolic batch.
          Optional when is_symbolic is True.
          Omit if is_symbolic is False.
        """

        assert_is_instance(is_symbolic, bool)

        # checks is_symbolic vs batch_size
        assert_equal(is_symbolic, batch_size is None,
                     "Must not supply a batch_size when is_symbolic is True."
                     if is_symbolic
                     else
                     "Must supply a batch_size when is_symbolic is False.")

        # Sanity-checks batch_size
        if not is_symbolic:
            check_is_subdtype(batch_size, 'batch_size', numpy.integer)
            assert_greater_equal(batch_size, 0)

        # checks is_symbolic vs name
        if not is_symbolic and name is not None:
            raise ValueError("Can't supply a name when is_symbolic is False.")

        # Checks dtype vs self.dtype
        if dtype is None:
            if self.dtype is None:
                raise TypeError("Must supply a dtype argument, because this "
                                "%s has no dtype of its own." % type(self))
            else:
                dtype = self.dtype
        elif self.dtype is not None:
            raise TypeError("Can't supply a dtype argument because this %s's "
                            "dtype is not None (it's %s)." %
                            (type(self), self.dtype))
        elif dtype == 'floatX':
            dtype = theano.config.floatX
        else:
            dtype = numpy.dtype(dtype)

        result = self._make_batch(is_symbolic, batch_size, dtype, name)

        self.check(result)
        return result

    def _make_batch(self, is_symbolic, batch_size, dtype, name):
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
    """Format for fixed-sized dense data.

    Example::

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
            raise TypeError("axes contain non-strings: %s" %
                            str(tuple(axes)))

        if len(frozenset(axes)) < len(axes):
            raise ValueError("axes contain duplicate elements: %s" %
                             str(tuple(axes)))

        if not all(numpy.issubdtype(type(size), 'int') for size in shape):
            raise TypeError("shape contains non-ints: %s" % str(shape))

        if len(axes) != len(shape):
            raise ValueError("axes and shape's lengths differ (%s vs %s)." %
                             (str(axes), str(shape)))

        if 'b' in axes:
            b_size = shape[axes.index('b')]

            if b_size != -1:
                raise ValueError("Shape element corresponding to 'b' axis "
                                 "must be given the dummy size -1, not %d. "
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

    def __str__(self):
        return ("DenseFormat{shape: %s, axes: %s, dtype: %s}" %
                (str(self.shape), str(self.axes), self.dtype))

    def _make_batch(self, is_symbolic, batch_size, dtype, name):

        if 'b' not in self.axes:
            raise ValueError("This format has no batch ('b') axis.")

        if is_symbolic:
            shape = list(self.shape)

            # ok if batch_size is None
            shape[self.axes.index('b')] = batch_size

            broadcastable = [False] * len(self.axes)
            # broadcastable = tuple(size == 1 for size in shape)

            # broadcastable = [False] * len(self.axes)
            # if 'f' in self.axes:
            #     f_index = self.axes.index('f')
            #     broadcastable[f_index] = (self.shape[f_index] == 1)

            # if 'b' in self.axes:
            #     broadcastable[self.axes.index('b')] = (batch_size == 1)

            # broadcastable = tuple(broadcastable)

            tensor_type = TensorType(dtype=dtype, broadcastable=broadcastable)
            result = tensor_type.make_variable(name=name)

            if theano.config.compute_test_value != 'off':
                if batch_size is None:
                    raise ValueError("When theano.config.compute_test_values "
                                     "is not 'off', you must supply a "
                                     "batch_size argument even when making"
                                     "symbolic batches.")
                else:
                    result.tag.test_value = \
                        self.make_batch(is_symbolic=False,
                                        batch_size=batch_size,
                                        dtype=dtype)
                # Don't understand this, from
                # pylearn2.space.ConvSpace2D.make_theano_batch, but keep it
                # here in case it becomes clear later:

                # if batch_size == 1:
                #     n = 1
                # else:
                #     batch_size
                #     # TODO: try to extract constant scalar value from batch_size
                #     n = 4
                # rval.tag.test_value = self.get_origin_batch(batch_size=n,
                #                                             dtype=dtype)
            return result

            # This is what pylearn2.space.VectorSpace does, for efficiency
            # reasons, but IIRC people on the mailing list were often
            # complaining of breakages caused by batch type that changed from
            # tensor.row to tensor.matrix depending on the value of batch_size.
            # whether a batch was a row or a matrix. Seems like any
            # efficiency gains may be more trouble that they're worth.

            # if batch_size == 1:
            #     return theano.tensor.row(name=name, dtype=dtype)
            # else:
            #     return theano.tensor.matrix(name=name, dtype=dtype)
        else:
            shape = list(self.shape)
            shape[self.axes.index('b')] = batch_size

            dtype = dtype if dtype is not None else self.dtype
            if dtype is None:
                raise ValueError("When self.dtype is None, you must provide a "
                                 "dtype argument to make_batch")

            return numpy.zeros(shape, dtype)

    def _check(self, batch):
        is_symbolic = self.is_symbolic(batch)

        if is_symbolic:
            if not isinstance(batch.type, (TensorType, CudaNdarrayType)):
                raise TypeError("Expected a TensorType or CudaNdarrayType, "
                                "not a %s." % batch.type)

            for val in get_debug_values(batch):
                self._check(val)

        if batch.ndim != len(self.axes):
            raise ValueError("Expected a %d-D tensor, but batch had %d "
                             "dimensions" % (len(self.axes), batch.ndim))

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

    def _convert(self, batch, output_format, output_batch, **kwargs):
        """
        Converts a batch to another format.

        Implemented by methods '_convert_to_XXX()' for target format XXX.
        See those methods' documentation for details.
        """

        if isinstance(output_format, DenseFormat):
            return self._convert_to_DenseFormat(batch,
                                                output_format,
                                                output_batch,
                                                **kwargs)
        else:
            raise NotImplementedError("Conversion from format %s to format %s "
                                      "not supported." %
                                      (type(self), type(output_format)))

    def _convert_to_DenseFormat(self,
                                batch,
                                output_format,
                                output_batch,
                                **kwargs):
        """
        Converts batch in this format to a DenseFormat.

        If the axes of <self> and <output_format> are the same,
        but in a different order, this will transpose the axes for you.

        Example 1: same axes, different order

          source = DenseFormat(axes=('a', 'b', 'c'), sizes=(3, 3, 3))
          target = DenseFormat(axes=('b', 'c', 'a'), sizes=(3, 3, 3))
          source.format(batch, target)  # transposes batch to target.axes

        If the axes are different, you must supply an "axis_map" argument to
        specify which axes of self correspond with which axes of output_format.

        Example 2: same # of axes, different names.

          from = DenseFormat(axes=('a', 'b'), sizes=(3, 3))
          to = DenseFormat(axes('c', 'd'), sizes(3, 3))
          from.format(batch, to, axis_map={'a': 'd',
                                           'b': 'c'})

        You can use axis_map to collapse multiple axes into a single axis:

        Example 3: stereo RGB to mono RGB

          stereo = DenseFormat(axes=('b', 's', '0', '1', 'f'),
                               sizes=(-1, 2, 32, 32, 3))
          mono = DenseFormat(axes=('b', '0', '1', 'f'),
                             sizes=(-1, 32, 32, 3))

          # No need to specify identity mappings like '0':'0' and '1':'1'.

          mono_batch = stereo.convert(stereo_batch,
                                      mono,
                                      axis_map={('b', 's'): 'b'})

        You can also expand a single axis into multiple axes:

        Example 4: mono RGB back to stereo

          stereo_batch_2 = mono.convert(mono_batch,
                                        stereo,
                                        axis_map={'b': ('b', 's')})

        ... or both expand some axes, and collapse others:

        Example 5: mono RGB to stereo flattened image vector

          stereo_vectors = DenseFormat(axes=('b', 's', 'f'),
                                       sizes=(-1, 2, 32*32*3))
          stereo_vector_batch = mono.convert(mono_batch,
                                             stereo_vectors,
                                             axis_map={'b': ('b', 's'),
                                                       ('0', '1', 'f') : 'f'})

        When expanding/collapsing a group of axes, the order of the axes in the
        tuple matters. In example 5 above, ('0', '1', 'f') : 'f' means that
        the image vectors (output channel 'f') will have row-major ('0'-major)
        element layout.

        Parameters
        ----------

        axis_map: dict
          A dict mapping strings or tuples to strings or tuples. Maps input
          batch axes to output batch axes. See examples above for usage.

          Identity mappings (e.g. 'b':'b') may be omitted. If all mappings are
          identity mappings, the axis_map may be omitted entirely.
        """

        if frozenset(self.axes) != frozenset(output_format.axes) and \
           'axis_map' not in kwargs:
            raise ValueError("If self.axes and output_format.axes don't "
                             "contain the same axes, then you must supply an "
                             "'axis_map' keyword argument.")

        if 'axis_map' in kwargs:
            axis_map = kwargs['axis_map']
            if not (axis_map is None or isinstance(axis_map, dict)):
                raise TypeError("Expected axis_map to be None or a dict, not "
                                "a %s." % type(axis_map))

            if axis_map is None:
                axis_map = dict()
        else:
            axis_map = dict()

        def get_standardized_axis_map(axis_map, source_axes, target_axes):
            """
            Returns a copy of target_axes, with any omitted identity mappings
            (e.g. 'b':'b') added back in.
            """

            flat_keys = frozenset(flatten(axis_map.iterkeys()))
            flat_values = frozenset(flatten(axis_map.itervalues()))
            source_axes = frozenset(source_axes)
            target_axes = frozenset(target_axes)

            if not flat_keys.issubset(source_axes):
                raise ValueError("axes_map's keys %s aren't a subset of the "
                                 "axes in source format %s." %
                                 (str(flat_keys), str(source_axes)))

            if not flat_values.issubset(target_axes):
                raise ValueError("axes_map's values %s aren't a subset of the "
                                 "axes in target format %s." %
                                 (str(flat_values), str(target_axes)))

            implicit_keys = source_axes - flat_keys
            implicit_values = target_axes - flat_values

            if implicit_keys != implicit_values:
                raise ValueError("Can't infer implicit identity axis "
                                 "mappings. The axis_map's missing implicit "
                                 "keys %s don't match its missing implicit "
                                 "values %s." % (str(implicit_keys),
                                                 str(implicit_values)))

            result = dict(axis_map)

            # Adds missing identity mappings
            for implicit_axis in implicit_keys:
                result[implicit_axis] = implicit_axis

            assert_equal(frozenset(flatten(result.iterkeys())), source_axes)
            assert_equal(frozenset(flatten(result.itervalues())), target_axes)

            return result

        def transpose(batch, from_axes, to_axes):
            assert_equal(frozenset(from_axes), frozenset(to_axes))

            for axes in (from_axes, to_axes):
                assert all(isinstance(axis, str) for axis in axes)

            return batch.transpose([from_axes.index(a) for a in to_axes])

        axis_map = get_standardized_axis_map(axis_map,
                                             self.axes,
                                             output_format.axes)

        grouped_source_axes = tuple(axis_map.iterkeys())
        ungrouped_source_axes = tuple(flatten(grouped_source_axes))
        batch = transpose(batch, self.axes, ungrouped_source_axes)

        grouped_target_axes = tuple(axis_map[k] for k in grouped_source_axes)
        ungrouped_target_axes = tuple(flatten(grouped_target_axes))

        # The sizes of target axes, in the order that they appear in
        # ungrouped_target_axes
        ungrouped_target_shape = \
            tuple(output_format.shape[output_format.axes.index(a)]
                  for a in ungrouped_target_axes)
        batch = batch.reshape(ungrouped_target_shape)
        batch = transpose(batch,
                          ungrouped_target_axes,
                          output_format.axes)

        if output_batch is not None:
            output_batch[...] = batch
            return output_batch
        elif output_format.dtype is None:
            return batch
        else:
            if self.is_symbolic(batch):
                return theano.tensor.cast(batch, str(output_format.dtype))
            else:
                return numpy.cast[output_format.dtype](batch)

    def requires_conversion(self, target_format):
        for fmt in (self, target_format):
            if fmt.dtype is not None:
                assert_is_instance(fmt.dtype, numpy.dtype)

            assert_is_instance(fmt.axes, tuple)
            assert_is_instance(fmt.shape, tuple)

        if target_format.dtype is not None and \
           self.dtype != target_format.dtype:
            return True

        return (self.axes != target_format.axes or
                self.shape != target_format.shape)
