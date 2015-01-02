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

    def epoch(self):
        """
        Returns the current epoch number, starting at 0.

        An epoch is defined as a loop through a "representative sample" of the
        data distribution. For example:

          * For many fixed-sized datasets, the N'th epoch is simply the N'th
            loop through the entire dataset.
          * If sampling randomly, you might trigger an epoch every S samples
            (where S is the dataset size), even if you might've skipped a few
            of the samples on any particular epoch.

        Epochs need not be of fixed length.

        Incrementing the epoch number serves only to trigger
        simplelearn.Trainer's epoch callbacks. Such callback functions can be
        costly, so epochs shouldn't generally be frequent. Example epoch
        callbacks include:

        * Computing validation set error to see if the training has converged.
        * Reshuffling the training set iteration order.
        * Redefining the training set to bias sampling towards examples that
          the model misclassed in the previous epoch.

        Returns
        -------
        rval: int
          The epoch number, starting at 0.
        """
        raise NotImplementedError("%s.epoch() not yet implemented." %
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
        raise NotImplementedError("%s.next() not yet implemented." %
                                  type(self))
