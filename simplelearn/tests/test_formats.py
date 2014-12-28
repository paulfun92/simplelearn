"""
Tests ../formats.py
"""

import os, pdb
import tempfile
import itertools
import numpy
from numpy.testing import assert_allclose
import theano
import theano.tensor as T
from simplelearn.formats import Format, DenseFormat
from simplelearn.utils import safe_izip
from nose.tools import assert_raises, assert_raises_regexp, assert_equal

_all_dtypes = tuple(numpy.dtype(t.dtype) for t in theano.scalar.all_types)


def _make_symbol(dtype, name=None):
    tensor_type = T.TensorType(dtype=dtype, broadcastable=(False, False))
    return tensor_type.make_variable(name)


def test_is_symbolic():
    tmp_dir = tempfile.mkdtemp(prefix="test_formats-test_get_variable_type-"
                               "temp_files-")

    numeric_batches = (numpy.zeros(3),
                       numpy.memmap(mode='w+',
                                    dtype='float32',
                                    shape=(2, 3),
                                    filename=os.path.join(tmp_dir,
                                                          'memmap.dat')))

    symbolic_batches = [T.dmatrix('dmatrix'), ]
    symbolic_batches.append(symbolic_batches[0].sum())

    for batch in numeric_batches:
        assert not Format.is_symbolic(batch)

    for batch in symbolic_batches:
        assert Format.is_symbolic(batch)

    # delete temporary memmap file & its temporary directory
    memmap_filename = numeric_batches[1].filename
    del numeric_batches
    os.remove(memmap_filename)
    os.rmdir(os.path.split(memmap_filename)[0])


def test_format_constructor():
    for dtype in _all_dtypes:
        batch_format = Format(dtype=dtype)
        if dtype is None:
            assert batch_format.dtype is None
        else:
            assert isinstance(batch_format.dtype, numpy.dtype)
            assert str(batch_format.dtype) == str(dtype)

    assert_raises(TypeError, Format, "not an legit dtype")


def test_format_abstract_methods():
    # pylint: disable=protected-access,anomalous-backslash-in-string

    batch_format = Format()
    assert_raises_regexp(NotImplementedError,
                         "_is_equivalent\(\) not yet implemented.",
                         batch_format._is_equivalent,
                         batch_format)

    assert_raises_regexp(NotImplementedError,
                         "_convert\(\) not yet implemented.",
                         batch_format._convert,
                         batch_format,
                         batch_format,
                         None)

    assert_raises_regexp(NotImplementedError,
                         "_check\(\) not yet implemented.",
                         batch_format._check,
                         None)

    assert_raises_regexp(NotImplementedError,
                         "_make_batch\(\) not yet implemented.",
                         batch_format._make_batch,
                         is_symbolic=True,
                         batch_size=2,
                         dtype=None,
                         name=None)


class DummyFormat(Format):
    def __init__(self, dtype=None):
        super(DummyFormat, self).__init__(dtype=dtype)

    def _is_equivalent(self, target_format):
        return self == target_format

    def _convert(self, batch, target_format, output):
        output_dtype = (batch.dtype if target_format.dtype is None
                        else target_format.dtype)

        if self.is_symbolic(batch):
            return T.cast(batch, str(output_dtype))
        else:
            if output is not None:
                output[...] = batch
                return output
            else:
                return numpy.asarray(batch, dtype=str(output_dtype))

    def _check(self, batch):
        pass

    def _make_batch(self, is_symbolic, batch_size, dtype, name):
        if dtype is None:
            dtype = self.dtype

        if is_symbolic:
            # return theano.tensor.tensor(dtype=dtype)
            return _make_symbol(dtype, name)
        else:
            return numpy.zeros((batch_size, 2), dtype=dtype)


def test_format_check():
    all_dtypes = _all_dtypes + (None, )
    batch_dtypes = [dt for dt in _all_dtypes if dt is not 'floatX']

    for format_dtype in all_dtypes:
        fmt = DummyFormat(format_dtype)

        for batch_dtype in batch_dtypes:
            for batch in (numpy.zeros(2, dtype=batch_dtype),
                          _make_symbol(batch_dtype)):
                if fmt.dtype == batch_dtype or fmt.dtype is None:
                    fmt.check(batch)
                else:
                    assert_raises_regexp(TypeError,
                                         "batch's dtype ",
                                         fmt.check,
                                         batch)


def test_format_make_batch():
    floatX_format = Format('floatX')
    none_format = Format(None)

    assert_raises_regexp(TypeError,
                         "batch_size must be an integer, not a ",
                         floatX_format.make_batch,
                         False,
                         1.)

    assert_raises_regexp(ValueError,
                         "batch_size must be non-negative, not ",
                         floatX_format.make_batch,
                         False,
                         -2)

    # Leave this commented-out block in for now.  Batch size is used even when
    # creating symbolic batches, to make batch axis broadcastable when
    # batch_size is 1. We may later decide that this is an unneeded bit of
    # complexity that we just remove, in which case we can go back to making
    # batch_size mutually exclusive with is_symbolic=True.

    # assert_raises_regexp(ValueError,
    #                     "Can't supply a batch_size when is_symbolic is True",
    #                      floatX_format.make_batch,
    #                      True,
    #                      2)

    assert_raises_regexp(ValueError,
                         "Must supply a batch_size when is_symbolic is False",
                         floatX_format.make_batch,
                         False)

    assert_raises_regexp(ValueError,
                         "Must supply a batch_size when is_symbolic is False",
                         floatX_format.make_batch,
                         is_symbolic=False,
                         name="foo")

    assert_raises_regexp(ValueError,
                         "Can't supply a name when is_symbolic is False",
                         floatX_format.make_batch,
                         is_symbolic=False,
                         batch_size=2,
                         name="foo")

    assert_raises_regexp(TypeError,
                         "Must supply a dtype argument, because this ",
                         none_format.make_batch,
                         True)

    assert_raises_regexp(TypeError,
                         "Can't supply a dtype argument because this ",
                         floatX_format.make_batch,
                         True,
                         dtype='float64')


class BadFormat(DummyFormat):

    def __init__(self, dtype):
        super(BadFormat, self).__init__(dtype)

    def _convert(self, batch, target_format, output):
        """
        Converts numeric batches to symbolic batches, and vice versa.
        """
        if self.is_symbolic(batch):
            return numpy.array([1, 2], dtype=target_format.dtype)
        else:
            return _make_symbol(target_format.dtype)


def test_format_convert():

    # Does a N^2 check of formatting all dtypes to all other dtypes.

    all_dtypes = _all_dtypes + (None, )

    for from_dtype in all_dtypes:
        from_format = DummyFormat(dtype=from_dtype)
        batch_dtype = 'float32' if from_dtype is None else from_format.dtype

        symbolic_batch = _make_symbol(batch_dtype)
        numeric_batch = numpy.zeros((10, 2), dtype=batch_dtype)

        for to_dtype in all_dtypes:
            to_format = DummyFormat(dtype=to_dtype)
            output_dtype = batch_dtype if to_dtype is None else to_format.dtype

            for batch in (symbolic_batch, numeric_batch):
                if numpy.can_cast(batch_dtype,
                                  output_dtype,
                                  casting='same_kind'):
                    output = from_format.convert(batch, to_format)
                    assert_equal(output.dtype, output_dtype)
                else:
                    assert_raises_regexp(TypeError,
                                         "Can't cast from ",
                                         from_format.convert,
                                         batch,
                                         to_format)

    # Checks that a format that takes a numeric batch and formats it to a
    # symbolic one (or vice-versa) will cause an exception.
    bad_int_format = BadFormat('int32')
    bad_float_format = BadFormat('float32')
    numeric_int_batch = numpy.zeros(3, dtype='int32')
    symbolic_int_batch = _make_symbol('int32')

    for int_batch in (numeric_int_batch, symbolic_int_batch):
        assert_raises_regexp(TypeError,
                             "Expected",
                             bad_int_format.convert,
                             int_batch,
                             bad_float_format)

    # Checks that providing an output argument when converting symbols raises
    # an exception.
    dummy_int_format = DummyFormat('int32')
    dummy_float_format = DummyFormat('float32')
    symbolic_float_batch = _make_symbol('float32')
    assert_raises_regexp(ValueError,
                         "You can't provide an output argument when data is "
                         "symbolic.",
                         dummy_int_format.convert,
                         symbolic_int_batch,
                         dummy_float_format,
                         symbolic_float_batch)


def test_denseformat_init():
    assert_raises_regexp(TypeError,
                         "axes contain non-strings",
                         DenseFormat,
                         ('b', 0, 1, 'c'),
                         (-1, 1, 1, 1),
                         'floatX')

    assert_raises_regexp(ValueError,
                         "axes contain duplicate elements",
                         DenseFormat,
                         ('b', '0', '1', '0'),
                         (-1, 1, 1, 1),
                         'floatX')

    assert_raises_regexp(TypeError,
                         "shape contains non-ints",
                         DenseFormat,
                         ('b', '0', '1', 'c'),
                         (-1, 1, 1, 1.),
                         'floatX')

    assert_raises_regexp(ValueError,
                         "axes and shape's lengths differ",
                         DenseFormat,
                         ('b', '0', '1', 'c'),
                         (-1, 1, 1),
                         'floatX')

    assert_raises_regexp(ValueError,
                         "Shape element corresponding to 'b' axis must be "
                         "given the dummy size -1",
                         DenseFormat,
                         ('b', '0', '1', 'c'),
                         (1, 1, 1, 1),
                         'floatX')

    assert_raises_regexp(ValueError,
                         "Negative size in non-batch dimension",
                         DenseFormat,
                         ('b', '0', '1', 'c'),
                         (-1, 1, 1, -1),
                         'floatX')

    axes = ('b', '0', '1', 'c')
    shape = (-1, 2, 3, 4)
    dense_format = DenseFormat(axes, shape, 'floatX')

    assert_equal(dense_format.axes, axes)
    assert_equal(dense_format.shape, shape)
    assert_equal(dense_format.dtype, numpy.dtype(theano.config.floatX))


def test_denseformat_make_batch():
    batchless_format = DenseFormat(('a', 'c', 'd'), (1, 1, 1), 'floatX')

    assert_raises_regexp(ValueError,
                         "This format has no batch \('b'\) axis",
                         batchless_format.make_batch,
                         True)

    assert_raises_regexp(ValueError,
                         "This format has no batch \('b'\) axis",
                         batchless_format.make_batch,
                         False,
                         2)

    axes = ('b', '0', '1', 'c')
    shape = (-1, 2, 3, 4)
    floatX_format = DenseFormat(axes, shape, 'floatX')
    none_format = DenseFormat(axes, shape, None)

    symbolic_batch_0 = floatX_format.make_batch(is_symbolic=True, name="woo")
    symbolic_batch_1 = none_format.make_batch(is_symbolic=True,
                                              name="woo",
                                              dtype='floatX')

    # Different symbolic batches are by definition unequal, so this can't work
    # assert_equal(symbolic_batch_0, symbolic_batch_1)

    floatX = numpy.dtype(theano.config.floatX)

    for batch in (symbolic_batch_0, symbolic_batch_1):
        assert_equal(batch.dtype, floatX)
        assert_equal(batch.name, 'woo')

    batch_shape = list(shape)
    batch_size = 7
    batch_shape[axes.index('b')] = batch_size

    numeric_batch_0 = floatX_format.make_batch(batch_size=batch_size,
                                               is_symbolic=False)
    numeric_batch_1 = none_format.make_batch(batch_size=batch_size,
                                             is_symbolic=False, dtype='floatX')
    zero_batch = numpy.zeros(batch_shape, dtype=floatX)

    for batch in (numeric_batch_0, numeric_batch_1):
        assert_equal(batch.dtype, floatX)
        numpy.testing.assert_equal(batch, zero_batch)


def test_denseformat_convert_to_denseformat():

    def make_patterned_batch(batch_format):
        """
        Makes a batch whose elements are a function of the indices and axes.
        """

        batch = batch_format.make_batch(batch_size=3, is_symbolic=False)

        def make_element(indices, axes):
            indices = numpy.array(indices)
            axes = numpy.array(tuple(ord(a) for a in axes))
            result = 0

            for index, axis in safe_izip(indices, axes):
                index = numpy.mod(index, 10)
                axis = numpy.mod(int(axis), 5)
                result += index * axis

            return result

        iterator = numpy.nditer(batch, flags=['multi_index'],
                                op_flags=['readwrite'])
        while not iterator.finished:
            indices = iterator.multi_index
            iterator[0] = make_element(indices, batch_format.axes)
            # batch[indices] = make_element(indices, batch_format.axes)
            iterator.iternext()

        assert (batch != numpy.zeros(batch.shape)).any()
        return batch

    # test simple permutations; i.e. no difference in the set of axes, or their
    # corresponding dimension sizes.
    dense_formats = [DenseFormat(axes=axes, shape=shape, dtype=int)
                     for axes, shape
                     in safe_izip(itertools.permutations(('c', '0', '1', 'b')),
                                  itertools.permutations((2, 3, 4, -1)))]

    patterned_batches = [make_patterned_batch(fmt) for fmt in dense_formats]
    # pdb.set_trace()
    # for source, target in itertools.product(dense_formats, repeat=2):
    for source, source_batch in safe_izip(dense_formats, patterned_batches):
        source_batch = make_patterned_batch(source)

        for target, expected_target_batch in safe_izip(dense_formats,
                                                       patterned_batches):
            target_batch = source.convert(source_batch, target)
            assert_allclose(target_batch, expected_target_batch)
