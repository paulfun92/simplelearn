import itertools
import numpy
from simplelearn.utils import safe_izip, flatten
from nose.tools import assert_raises_regexp, assert_equal


def test_safe_izip():
    rng = numpy.random.RandomState(3423)

    sequence_length = 5

    expected_values = rng.uniform(size=(2, sequence_length))

    for index, value0, value1 in safe_izip(range(sequence_length),
                                           expected_values[0, :],
                                           expected_values[1, :]):
        assert_equal(value0, expected_values[0, index])
        assert_equal(value1, expected_values[1, index])

    def iterate_over_unequal_lengths():
        for x0, x1 in safe_izip(xrange(3), xrange(2)):
            pass

    assert_raises_regexp(IndexError,
                         "Can't safe_izip over sequences of different lengths",
                         iterate_over_unequal_lengths)


# def test_safe_izip2():
#     pairs = safe_izip(itertools.permutations(('c', '0', '1', 'b')),
#                       itertools.permutations((2, 3, 4, -1)))
#     for pair in pairs:
#         print pair


def test_flatten():
    nested_list = [xrange(3),
                   3,
                   ((4, 5), (6, ()), 7),
                   8,
                   (9, ),
                   10,
                   ()]

    for value, expected_value in safe_izip(flatten(nested_list), xrange(11)):
        assert_equal(value, expected_value)
