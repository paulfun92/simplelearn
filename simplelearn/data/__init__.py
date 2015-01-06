"""
Data sources (a dataset is a static data source).
"""

__author__ = "Matthew Koichi Grimes"
__email__ = "mkg@alum.mit.edu"
__copyright__ = "Copyright 2014"
__license__ = "Apache 2.0"


class DataSource(object):
    """
    A source of data.
    """

    def iterator(self, iterator_type, batch_size, **kwargs):
        raise NotImplementedError("%s.iterator() not yet implemented." %
                                  type(self))

    def get_input_nodes(self):
        """
        Returns
        -------
        rval: instance of a collections.namedtuple type.
          A named tuple of InputNodes. The names must correspond to the names
          used by iterators' next() method (e.g. 'model_input', 'label').
        """
        raise NotImplementedError("%s.get_input_nodes() not yet implemented."
                                  % type(self))


class DataIterator(object):
    """
    Yields data batches from a data source.
    """

    def __init__(self):
        # Used for ensuring that _epoch() is implemented correctly
        # (i.e. that epochs start at -1, and increase by at most one when
        # next() is called).
        self._next_was_called = False

    def epoch(self):
        """
        Returns the epoch number of the last yielded batch.

        If no batches have been yielded yet, this returns -1.

        An epoch is defined as a loop through a "representative sample" of the
        data distribution. For example::

          * For many fixed-sized datasets, the N'th epoch is simply the N'th
            loop through the entire dataset.
          * If sampling randomly, you might trigger an epoch every S samples
            (where S is the dataset size), even if you might've skipped a few
            of the samples on any particular epoch.

        Epochs need not be of fixed length.

        Incrementing the epoch number serves only to trigger
        simplelearn.Trainer's epoch callbacks. These can include costly
        functions such as computing average error over a validation set, so in
        general epochs shouldn't be frequent.

        Example epoch callbacks include::

          * Computing the average error over a validation dataset.
          * Reshuffling the training set iteration order.
          * Redefining the training set to bias sampling towards examples that
            the model misclassed in the previous epoch.

        Returns
        -------
        rval: int
          The epoch number, starting at 0.
        """
        raise NotImplementedError("%s._epoch() not yet implemented." %
                                  type(self))

    def __iter__(self):
        return self

    def next(self):
        """
        Returns a batch of data, and increments the iterator.

        The returned batch is in a collections.namedtuple type.  For example, a
        common one will be collections.namedtuple('DataAndLabel', ('data',
        'label')). A returned batch can be indexed as a tuple::

          batch[0]  # data
          batch[1]  # label

        ...or using named fields::

          batch.data  # data
          batch.label  # label

        These field names must match the names returned by the dataset's
        get_input_nodes.

        Returns
        -------

        rval: instance of a collections.namedtuple type.
        """
        prev_epoch = self.epoch()
        if not numpy.issubdtype(type(prev_epoch), numpy.integer):
            raise TypeError("Expected epoch() to return an integer, not a %s."
                            % type(prev_epoch))

        if not self._next_was_called:
            if prev_epoch != -1:
                raise ValueError("%s implemented incorrectly: "
                                 "Expected epoch() to return -1 before "
                                 "yielding first batch, but got %d." %
                                 prev_epoch)

        result = self._next()

        if not isinstance(result, tuple) or \
           not all(Format.is_numeric(r) for r in result):
            raise TypeError("%s implemented incorrectly: Must return a tuple "
                            "of numeric arrays.")

        curr_epoch = self.epoch()
        if not self._next_was_called:
            if curr_epoch != 0:
                raise VaueError("%s implemented incorrectly: "
                                "Expected epoch() to return 0 after "
                                "first call to next(), but got %d."
                                % curr_epoch)

        if (curr_epoch - prev_epoch) not in (0, 1):
            raise ValueError("%s implemented incorrectly: Expected epoch() "
                             "to increase by 0 or 1 when calling next(), but "
                             "it went from %d to %d."
                             % (prev_epoch, curr_epoch))

        self._next_was_called = True
        return result

    def _next(self):
        """
        Implements next(). See that method's docs for details.
        """
        raise NotImplementedError("%s._next() not yet implemented." %
                                  type(self))
