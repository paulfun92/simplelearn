"""
Utility functions used throughout simplelearn
"""

import collections
from itertools import izip_longest, takewhile, chain


def safe_izip(*iterables):
    sentinel = object()

    iterators = tuple(chain(x, (sentinel,)) for x in iterables)

    while iterators:
        items = tuple(map(next, iterators))

        if all(item is sentinel for item in items):  # all are sentinels
            raise StopIteration()
        elif any(item is sentinel for item in items):  # some are sentinels
            raise IndexError("Can't safe_izip over sequences of different "
                             "lengths.")

        else:
            yield items


def flatten(iterable):
    """
    Returns an iterator that goes through a nested iterable in
    depth-first order.
    """
    for element in iterable:
        if isinstance(element, collections.Iterable) and \
           not isinstance(element, basestring):
            for sub_element in flatten(element):
                yield sub_element
        else:
            yield element
