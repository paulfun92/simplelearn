'''
A Dataset that lives on disk in a numpy memmap file.
'''

import os
import numpy
from nose.tools import (assert_true,
                        assert_equal,
                        assert_is_instance,
                        assert_greater)
import simplelearn
from simplelearn.utils import safe_izip
from simplelearn.formats import DenseFormat
from simplelearn.data.dataset import Dataset
from simplelearn.asserts import assert_all_is_instance


def make_memmap_file(path, num_examples, tensor_names, tensor_formats):
    '''
    Allocates a memmap file on disk. Overwrites if necessary.

    Parameters
    ----------

    path: str
      Path to file.

    num_examples: int
      # of examples in this dataset.

    tensor_names: Iterable of strings
      Names of the tensors in this dataset.

    tensor_formats: Iterable of simplelearn.format.DenseFormats
      Formats of the tensors. MemmapDataset requires that the batch axis be the
      first axis.
    '''
    assert_is_instance(path, basestring)
    assert_greater(num_examples, 0)
    assert_equal(len(tensor_names), len(tensor_formats))

    assert_all_is_instance(tensor_names, basestring)
    assert_all_is_instance(tensor_formats, DenseFormat)

    # We store datasets in a single structured array, so the batch axis must
    # be the first axis for all tensors.
    for tensor_format in tensor_formats:
        assert_equal(tensor_format.axes.index('b'), 0)

    dtype_dict = {
        'names': tensor_names,
        'formats': [(fmt.dtype, fmt.shape[1:]) for fmt in tensor_formats],
        'titles': [fmt.axes for fmt in tensor_formats]}

    memmap_file = numpy.lib.format.open_memmap(path,
                                               mode='w+',
                                               dtype=dtype_dict,
                                               shape=(num_examples, ))

    return memmap_file


class MemmapDataset(Dataset):
    '''
    A read-only Dataset that wraps a .npy file, loaded as a numpy.memmap.
    '''

    def __init__(self, path):
        assert_is_instance(path, basestring)

        path = os.path.abspath(path)

        assert_true(path.startswith(simplelearn.data.data_path),
                    ("{} is not a subdirectory of simplelearn.data.data_path "
                     "{}").format(path, simplelearn.data.data_path))

        # pylint can't see memmap members
        # pylint: disable=no-member
        self.memmap = numpy.lib.format.open_memmap(path, mode='r')
        num_examples = self.memmap.shape[0]

        names = self.memmap.dtype.names
        tensors = [self.memmap[name] for name in names]
        axes_list = [field[2] for field
                     in self.memmap.dtype.fields.itervalues()]

        def replace_element(arg, index, new_value):
            assert_is_instance(arg, tuple)
            result = list(arg)
            result[index] = new_value
            return tuple(result)

        formats = [DenseFormat(axes=axes,
                               shape=replace_element(tensor.shape, 0, -1),
                               dtype=tensor.dtype)
                   for axes, tensor in safe_izip(axes_list, tensors)]

        super(MemmapDataset, self).__init__(names=names,
                                            formats=formats,
                                            tensors=tensors)

    def __getstate__(self):
        '''
        Saves just the file path (relative to data_dir) and the partition name.
        '''
        absolute_path = os.path.abspath(self.memmap.filename)
        relative_path = os.path.relpath(absolute_path,
                                        simplelearn.data.data_path)
        return relative_path

    def __setstate__(self, state):
        self.__init__(os.path.join(simplelearn.data.data_path, state[0]))
