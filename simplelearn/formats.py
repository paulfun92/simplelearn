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
from theano.sandbox.cuda.type import CudaNdarrayType


class DataFormat(object):
    """
    Specifies the data format of a single tensor.

    Data format includes things like dtype, shape (not including batch size),
    and axis order / memory layout.

    Primary methods:

      format():    Converts a tensor from this format to another.
      validate():  Returns True if a tensor fits this format.
      new_theano_tensor(): Creates a Theano tensor in this format.
      new_numpy_tensor(): Creates a numpy tensor in this format.
    """

    @staticmethod
    def get_variable_type(data):
        """
        Returns whether the data is a numeric or symbolic variable.

        Parameters
        ----------
        data: numpy.ndarray, numpy.memmap, or theano.gof.Variable

        Returns
        -------
        rval: str
           "symbolic", "numeric", or "unrecognized".
        """

        if isinstance(data, theano.gof.Variable):
            return "symbolic"
        elif (isinstance(data, (numpy.ndarray, numpy.memmap)) or
              type(data) == 'CudaNdarray'):
            return "numeric"
        else:
            return "unrecognized"

    def format(self, data, target_format, output=None):
        """
        Formats numeric or symbolic data in this format to the target_format.

        Output argument only supported for numeric data.

        This function just calls self._format(), check()-ing its
        inputs and outputs.

        Parameters
        ----------
        data: numpy.ndarray-like or theano.gof.Variable

        target_format: DataFormat

        output: NoneType, or numpy.ndarray-like
          Optional output variable. Can only be supplied if data is numeric.
          If provided, this function won't return anything.
          If omitted, this function will return the formatted data.

        Returns
        -------
        rval: NoneType, numpy.ndarray or theano.gof.Variable
          If output is None, rval is the formatted data. When formatting
          between two equivalent formats, rval will be data, returned as-is.

          If output is not None, rval will be None.
        """

        self.check(data)
        if output is None:
            result = self._format(data, target_format)
            target_format.check(result)
        else:
            if data_type == 'symbolic':
                raise ValueError("You can't provide an output argument when "
                                 "data is symbolic.")
            target_format.check(output)
            self._format(data, target_format, output)

        # Checks that:  numeric data -> numeric result,
        #              symbolic data -> symbolic result

        data_type = self.get_variable_type(data)
        result_type = self.get_variable_type(result)

        if data_type != result_type:
            raise TypeError("Expected %{classname}s._format(<%{data_type}s "
                            "tensor>) to return a <%{data_type}s tensor>, but "
                            "got a %{result_type}s instead." %
                            dict(self_type=type(self),
                                 data_type=data_type,
                                 result_type=result_type))

        if output is None:
            return result

    def _format(self, data, target_format):
        """
        Implementation of format(). See that method's description for specs.

        Must throw an exception (typically TypeError or ValueError) if
        conversion to target_format is not supported.
        """

        raise NotImplementedError("%s._format() not implemented." %
                                  self.__class__)

    def check(self, data):
        """
        Checks to see if data fits this format's specs.

        If not, throws a ValueError or TypeError with an informative message.

        Parameters
        ----------
        data: theano or numpy variable.

        Returns
        -------
        nothing.
        """

        # Type-checks data.
        if self.get_variable_type(data) == 'unrecognized':
            raise TypeError("Expected data to be a numpy or theano variable, "
                            "not a %s." % type(data))

        self._check(data)

    def _check(self, data):
        """
        Implements check(). See that method's documentation.
        """
        raise NotImplementedError("%s.is_valid() not implemented." %
                                  self.__class__)

    def make_batch(self, batch_type, dtype=None):
        """
        Makes a numeric or symbolic batch.

        Parameters
        ----------
        batch_type: str
          'numeric' or 'symbolic'

        dtype: str
          A numpy/theano dtype.
        """
        if dtype is None and self.dtype is None:
            raise TypeError("Since this %s doesn't specify a dtype, you must "
                            "provide a dtype argument." % type(self))

        return self._make_batch(batch_type=batch_type)

    def _make_batch(self, batch_type, dtype=None):
        """
        Implements make_batch. See that method's documentation().
        """
        raise NotImplementedError("%s._make_batch() not yet implemented." %
                                  type(self))


class OptionallyTypedFormat(DataFormat):
    """
    A DataFormat that optionally specifies the dtype of a variable.

    Not yet sure if this merits being a separate class from Dataformat. I may
    merge it into DataFormat.
    """

    def __init__(self, dtype):
        """
        Parameters
        ----------
        dtype: str/numpy.dtype, or None
          If not None, this Format will insist that data be of this dtype.
        """
        if dtype is None:
            self.dtype = dtype
        else:
            self.dtype = numpy.dtype(dtype)  # checks that dtype str is legit

    def _check(self, data):
        super(OptionallyTypedFormat, self)._check(data)

        if self.dtype is not None and not str(data.dtype) == self.dtype:
            raise TypeError("Data's dtype (%s) doesn't match this %s's dtype "
                            "(%s)" % (data.dtype, type(self), self.dtype))

    def _check_target_dtype(self, target_dtype):
        """
        Throws an exception if casting from this.dtype to target_dtype would
        entail a significant loss of precision (e.g. float->int,
        complex->float).
        """

        # checks that target_dtype is a legit dtype
        target_dtype = numpy.dtype(target_dtype)

        if numpy.issubdtype(self.dtype, numpy.complex):
            if not numpy.issubdtype(target_dtype, numpy.complex):
                raise TypeError("Can't convert from complex to "
                                "non-complex (in this case, %s to %s)."
                                (self.dtype, target_dtype))
        elif numpy.issubdtype(self.dtype, numpy.float):
            if not numpy.issubdtype(target_dtype, (numpy.float,
                                                   numpy.complex)):
                raise TypeError("Can't convert %s to %s." %
                                (self.dtype, target_dtype))


class VectorFormat(OptionallyTypedFormat):
    """
    Fixed-size dense vectors.

    Parameters
    ----------
    size: int
      Vector length.

    dtype: numpy.dtype/str
      Data type.

    layout_order: sequence
      A sequence of at least two strings, where the strings are axis
      names. Specifies the axis order of memory layout. Used when converting to
      topological formats.  Example: ('b', 'c', '0', '1'). 'b' axis must always
      be first.
    """

    def __init__(self, size, dtype, layout_order=('b', 'c', '0', '1')):
        super(VectorFormat, self).__init__(dtype)

        if not numpy.issubdtype(size, 'int'):
            raise TypeError("size should be an int, not a %s." % type(size))

        if size < 0:
            raise ValueError("size must be non-negative, not %d." % size)

        self.size = size

        if len(layout_order) < 2:
            raise ValueError("layout_order %s must have at least two elements."
                             % str(layout_order))

        if layout_order[0] != 'b':
            raise ValueError("The 'b' axis must be first in layout_order %s." %
                             str(layout_order))

        if not all(isinstance(s, str) for s in layout_order):
            raise TypeError("layout_order %s must be a sequence of strings." %
                            str(layout_order))

        self.layout_order = layout_order

    def _check(self, data):
        super(VectorFormat, self)._check(data)

        variable_type = self.get_variable_type(data)

        if variable_type is 'symbolic':
            if not isinstance(data.type, (TensorType, CudaNdarrayType)):
                raise TypeError("Expected a TensorType or CudaNdarrayType, "
                                "not a %s." % data.type)

            for val in get_debug_values(data):
                self._check(val)

        if data.ndim != 2:
            raise ValueError("Expected a 2-D tensor, but found %d" %
                             data.ndim)

        if variable_type is 'numeric':
            if batch.shape[1] != self.size:
                raise ValueError("Expected batch.shape[1] to equal self.size "
                                 "(%d), but it was %d." %
                                 (self.size, batch.shape[1]))

    def _format(self, data, target_format, output_data):
        self._check_target_dtype(target_format.dtype)

        if isinstance(target_format, VectorFormat):
            if target_format.size != self.size:
                raise ValueError("vector sizes don't match (self: %d, "
                                 "target: %d)." %
                                 (self.size, target_format.size))

            if output_data is None:
                return data
            else:
                output_data[...] = data
                return
        else:
            raise NotImplementedError("%s.format(data, %s) not yet "
                                      "implemented." %
                                      (type(self), type(target_format)))
