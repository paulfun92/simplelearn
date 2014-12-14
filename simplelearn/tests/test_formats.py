"""
Tests ../formats.py
"""

import sys
import os.path
import tempfile
import numpy
import theano
from nose.tools import assert_is
from simplelearn.formats import DataFormat


def test_get_variable_type():
    tmp_dir = tempfile.mkdtemp(prefix="test_formats-test_get_variable_type-"
                               "temp_files-")

    numeric_batches = (numpy.zeros(3),
                       numpy.memmap(mode='w+',
                                    dtype='float32',
                                    shape=(2, 3),
                                    filename=os.path.join(tmp_dir,
                                                          'memmap.dat')))

    symbolic_batches = [theano.tensor.dmatrix('dmatrix'), ]
    symbolic_batches.append(symbolic_batches[0].sum())

    for batch in numeric_batches:
        assert_is(DataFormat.get_variable_type(batch), 'numeric')

    for batch in symbolic_batches:
        assert_is(DataFormat.get_variable_type(batch), 'symbolic')

    # delete temporary memmap file & its temporary directory
    memmap_filename = numeric_batches[1].filename
    del numeric_batches
    os.remove(memmap_filename)
    sys.rmdir(os.path.split(memmap_filename)[0])


# def test_make_batch(format):
#     format.make_batch()


# def test_reversible(from_format, to_format, error_tolerance=0.0):
#     from_format.make_batch()
