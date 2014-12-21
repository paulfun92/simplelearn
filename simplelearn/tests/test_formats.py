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
                         True,
                         2,
                         None)


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

    def _make_batch(is_symbolic, batch_size, dtype):
        if dtype is None:
            dtype = self.dtype

        if is_symbolic:
            # return theano.tensor.tensor(dtype=dtype)
            return T.TensorType(dtype=dtype).make_variable()
        else:
            return numpy.zeros((batch_size, 2), dtype=dtype)


def test_format_format():

    # Does a N^2 check of formatting all dtypes to all other dtypes.

    all_dtypes = _all_dtypes + (None, )

    for from_dtype, to_dtype in itertools.product(all_dtypes, repeat=2):
        from_format = DummyFormat(dtype=from_dtype)
        to_format = DummyFormat(dtype=to_dtype)

        batch_dtype = 'float32' if from_dtype is None else from_format.dtype
        output_dtype = batch_dtype if to_dtype is None else to_format.dtype

        # symbolic_batch = theano.tensor.dtensor4(dtype=batch_dtype)
        tensor_type = T.TensorType(dtype=batch_dtype,
                                   broadcastable=(False, False))
        symbolic_batch = tensor_type.make_variable()
        numeric_batch = numpy.zeros((10, 2), dtype=batch_dtype)
        for batch in (symbolic_batch, numeric_batch):
            if numpy.can_cast(batch_dtype, output_dtype, casting='same_kind'):
                output = from_format.convert(batch, to_format)
                assert_equal(output.dtype, output_dtype)
            else:
                assert_raises_regexp(TypeError,
                                     "Can't cast from ",
                                     from_format.convert,
                                     batch,
                                     to_format)



    # float32_format = DummyFormat(dtype='float32')
    # none_format = DummyFormat()

    # for fmt in (float32_format, none_format):
    #     assert_raises(TypeError, fmt.check, object())

    # bad_dtype = numpy.dtype('int')

    # def make_batches(dtype=None):
    #     """
    #     Returns a numeric batch and a symbolic batch.
    #     """
    #     return (numpy.zeros((2, 3), dtype=dtype),
    #             theano.Tensor.dtensor4(dtype=dtype))

    # for format_dtype

    # for int_batch in make_batches('int'):
    #     assert_raises(TypeError, float32_format.check, int_batch)

    # for dtype in _all_dtypes:
    #     for batch in (numpy.zeros((2, 3), dtype=dtype),
    #                   theano.Tensor.dtensor4(dtype=dtype)):
    #         assert_raises(ValueError, none_format.check, batch)




def notest_denseformat_make_batch_numeric():

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

        iterator = numpy.nditer(batch, flags=['multi_index'])
        while not iterator.finished:
            indices = iterator.multi_index
            batch[indices] = make_element(indices, batch_format.axes)

        assert batch != numpy.zeros(batch.shape)
        return batch

    # test simple permutations; i.e. no difference in the set of axes, or their
    # corresponding dimension sizes.
    # pdb.set_trace()
    dense_formats = (DenseFormat(axes=axes, shape=shape, dtype=int)
                     for axes, shape
                     in safe_izip(itertools.permutations(('c', '0', '1', 'b')),
                                  itertools.permutations((2, 3, 4, -1))))
    # pdb.set_trace()
    for source, target in itertools.product(dense_formats, repeat=2):
        source_batch = make_patterned_batch(source)
        target_batch = source.convert(source_batch, target)
        expected_target_batch = make_patterned_batch(target)
        pdb.set_trace()
        assert_allclose(target_batch, expected_target_batch)


def notest_denseformat_format_numeric():
    pass
