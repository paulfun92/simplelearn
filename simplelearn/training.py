"""
Training algorithms, and callbacks for monitoring their progress.
"""

__author__ = "Matthew Koichi Grimes"
__email__ = "mkg@alum.mit.edu"
__copyright__ = "Copyright 2015"
__license__ = "Apache 2.0"

import warnings
from collections import Sequence, OrderedDict
import numpy
import theano
from nose.tools import (assert_true,
                        assert_equal,
                        assert_not_equal,
                        assert_less_equal,
                        assert_greater,
                        assert_greater_equal,
                        assert_is_instance,
                        assert_is_not,
                        assert_in)
from simplelearn.data import DataIterator
from simplelearn.utils import safe_izip, check_is_subdtype
from simplelearn.formats import Format
import pdb

# pylint: disable=too-few-public-methods

# Sketch:

# SGD takes N params and N GradientBasedUpdaters. One of the callbacks may
# optionally also be given the GBAs, and update their learning rate and
# momentum.
#
# pros: Pretty flexible, need not have a different SGD class for every
# different kind of update.
#
# cons: Wordy on creation.
#
# weight_updater = MomentumBasedUpdater(initial_learning_rate,
#                                       initial_momentum)
# weight_updater_updater = LinearDecay(updater.learning_rate,
#                                      saturation_fraction=.01,
#                                      epochs_to_saturation=200)
# sgd = SGD(dataset_iterator,
#           cost.get_output_symbol(),
#           model.get_parameters(),
#           weight_updater,
#           weight_updater_updater)
#
# hmm... not so bad.


class EpochCallback(object):

    def on_start_training(self):
        '''
        Called at the beginning of training, before processing any batches.
        '''
        raise NotImplementedError("%s.on_start_training() not yet implemented."
                                  % type(self))

    def on_epoch(self):
        raise NotImplementedError("%s.on_epoch() not yet implemented." %
                                  type(self))


class LimitsNumEpochs(EpochCallback):
    """
    Throws a StopTraining exception after a fixed number of epochs.
    """

    def __init__(self, max_num_epochs):
        if not numpy.issubdtype(type(max_num_epochs), numpy.integer):
            raise TypeError("Expected max_num_epochs to be an integer, not a "
                            "%s." % type(max_num_epochs))

        if max_num_epochs < 0:
            raise ValueError("max_num_epochs must be non-negative, got %d." %
                             max_num_epochs)

        self._max_num_epochs = max_num_epochs
        self._num_epochs_seen = None

    def on_start_training(self):
        self._num_epochs_seen = 0

    def on_epoch(self):
        assert self._num_epochs_seen >= 0

        self._num_epochs_seen += 1

        if self._num_epochs_seen >= self._max_num_epochs:
            raise StopTraining(status='ok',
                               message=('Reached max # of epochs %d.' %
                                        self._max_num_epochs))


class ValidationCallback(EpochCallback):
    '''
    Runs monitors over a validation dataset after each training epoch.
    '''

    def __init__(self, inputs, input_iterator, monitors):
        self._epoch_limiter = LimitsNumEpochs(1)
        epoch_callbacks = [self._epoch_limiter, ]
        self.trainer = Sgd(inputs=inputs,
                           parameters=[],
                           parameter_updaters=[],
                           input_iterator=input_iterator,
                           monitors=monitors,
                           epoch_callbacks=epoch_callbacks)

    def on_start_training(self):
        self.on_epoch()

    def on_epoch(self):
        try:
            self.trainer.train()
        except StopTraining, stop_training:
            if stop_training.status == 'ok' and \
               stop_training.message.startswith("Reached max # of epochs "):
                self._epoch_limiter._num_epochs_seen = 0
            else:
                raise


class LinearlyScalesOverEpochs(EpochCallback):
    """
    An epoch callback that linearly scales a theano shared variable over time.

    Parameters
    ----------

    shared_value: a Theano shared variable
      This value will be scaled in-place by a factor S that decreases from 1.0
      to final_scale over <epochs_to_saturation> epochs.

    final_scale: float
      Final value of S.

    epochs_to_saturation: int
      self._scale should decay to final_value after this many epochs.
    """

    def __init__(self, shared_value, final_scale, epochs_to_saturation):
        assert_is_instance(shared_value,
                           theano.tensor.sharedvar.SharedVariable)

        check_is_subdtype(final_scale, "final_scale", numpy.floating)

        check_is_subdtype(epochs_to_saturation,
                          "epochs_to_saturation",
                          numpy.integer)

        self.shared_value = shared_value
        self._initial_value = self.shared_value.get_value()

        self._final_scale = final_scale
        self._epochs_to_saturation = epochs_to_saturation

        self._num_epochs_seen = None

    def on_start_training(self):
        self._num_epochs_seen = 0

    def on_epoch(self):
        assert_greater_equal(self._num_epochs_seen, 0)

        # interpolation parameter
        alpha = min(1.0,
                    float(self._num_epochs_seen) / self._epochs_to_saturation)

        scale = (1.0 - alpha) + alpha * self._final_scale

        self.shared_value.set_value(scale * self._initial_value)


class Monitor(EpochCallback):
    """
    On each epoch, this reports statistics about some monitored value Y.

    Examples: Y might be the output of layer 3 of a 6-layer net.

              MaxMonitor reports the elementwise maximum of Y encountered
              over the epoch.

              AverageMonitor reports Y, elementwise-averaged over the epoch.
    """

    def __init__(self, values_to_monitor, formats, callbacks):
        '''
        Parameters
        ----------
        values_to_monitor: theano expression, or a Sequence of them
          A sequence of theano expressions to monitor. These should be
          functions of the input variables.

          Must not be empty.

        formats: a Format, or a Sequence of them
          A sequence of the above values' Formats.

        callbacks: a __call__-able, or a Sequence of them
          The values returned by self._on_epoch() get fed to these callbacks.
          These must have the call signature f(values, formats).
          Values is the Sequence returned by self._on_epoch().
          Formats are the values' formats, also a Sequence.
        '''

        #
        # Checks args
        #

        if isinstance(values_to_monitor, theano.gof.Variable):
            values_to_monitor = [values_to_monitor]

        if isinstance(formats, Format):
            formats = [formats]

        if not isinstance(callbacks, Sequence):
            callbacks = [callbacks]

        assert_is_instance(values_to_monitor, Sequence)
        assert_is_instance(formats, Sequence)
        assert_is_instance(callbacks, Sequence)

        assert_equal(len(values_to_monitor), len(formats))
        assert_equal(len(values_to_monitor),
                     len(frozenset(values_to_monitor)),
                     "values_to_monitor contains repeated elements: %s" %
                     str(values_to_monitor))

        for value, fmt in safe_izip(values_to_monitor, formats):
            assert_is_instance(value, theano.gof.Variable)
            assert_is_instance(fmt, Format)

            if 'b' not in fmt.axes:
                raise ValueError("format.axes doesn't contain batch axis "
                                 "'b': %s" % str(fmt.axes))

        #
        # Sets members
        #

        self.monitored_values = tuple(values_to_monitor)
        self._formats = tuple(formats)
        self._callbacks = list(callbacks)

    def on_batch(self, input_batches, monitored_value_batches):
        '''
        Updates the values to report at the end of the epoch.

        Parameters
        ----------
        input_batches: Sequence of numpy.ndarrays
          The input batches coming in from the dataset's iterator.
          Typically these are (values, labels)

        monitored_value_batches: Sequence of numpy.ndarrays
          The numerical values, for this batch, of the values_to_monitor
          arguments to __init__().
        '''
        assert_equal(len(monitored_value_batches), len(self._formats))

        for batch, fmt in safe_izip(monitored_value_batches, self._formats):
            fmt.check(batch)

        self._on_batch(tuple(input_batches),
                       tuple(monitored_value_batches))

    def _on_batch(self, input_batches, monitored_value_batches):
        '''
        Implementation of self.on_batch(). See that method's docs.

        Parameters
        ----------
        input_batches: tuple of numpy.ndarrays.

        monitored_value_batches: tuple of numpy.ndarrays.
        '''
        raise NotImplementedError("%s._on_batch() not yet implemented." %
                                  type(self))

    def on_start_training(self):
        pass

    def on_epoch(self):
        '''
        Feeds monitored values to self._callbacks
        '''
        # compute values to report
        values_to_report = self._on_epoch()

        if not isinstance(values_to_report, tuple):
            raise ValueError("%s._on_epoch() implemented incorrectly. It "
                             "should return a tuple, but it returned %s."
                             % (type(self), type(values_to_report)))

        for callback in self._callbacks:
            callback(values_to_report, self._formats)

    def _on_epoch(self):
        '''
        Returns a tuple of values to feed to self._callbacks as arguments.

        Returns
        -------
        rval: tuple of numpy.ndarrays
           Arguments to feed to self._callbacks' __call__(self, *args)
        '''
        raise NotImplementedError("%s._on_epoch() not yet implemented")


# class AverageMonitor(Monitor):

#     def __init__(self, values_to_monitor, formats):
#         super(AverageMonitor, self).__init__(values_to_monitor, formats)

#         # # values must have names
#         # for value in values_to_monitor:
#         #     assert_in('name', value.__dict__)
#         #     assert_is_instance(value.name, str)

#         # # value names must be unique
#         # if len(frozenset(v.name for v in values_to_monitor)) != \
#         #    len(values_to_monitor):
#         #     raise ValueError("There were some duplicate variable names: %s" %
#         #                      str(tuple(v.name for v in values_to_monitor)))

#         for fmt in formats:
#             assert_in('b', fmt.axes)

#         self._totals = None
#         self._count = 0

#     def _on_batch(self, input_batches, monitored_value_batches):

#         def get_sums_and_batch_size(batches, formats):
#             batch_axes = [fmt.axes.index('b') for fmt in formats]

#             sums = [numpy.sum(batch, axis=batch_axis)
#                     for batch, batch_axis
#                     in safe_izip(batches, batch_axes)]

#             batch_sizes = numpy.array([batch.shape[batch_axis]
#                                        for batch, batch_axis
#                                        in safe_izip(batches, batch_axes)])

#             # Sic. Numpy.all() returns True for empty arrays.
#             assert_true(numpy.all(batch_sizes[0] == batch_sizes[1:]))

#             return sums, batch_sizes[0]

#         sums, batch_size = get_sums_and_batch_sizes(monitored_value_batches,
#                                                     self._formats)
#         if self._totals is None:
#             self._totals = sums
#             self._count = batch_size
#         else:
#             for total, subtotal in safe_izip(self._totals, sums):
#                 total += subtotal

#             self._count += batch_size

#     def _on_epoch(self):
#         totals = self._totals
#         count = self._count
#         self._totals = None
#         self._count = None
#         return [total / count for total in totals]


# class MaxMonitor(Monitor):
#     def __init__(self, values_to_monitor, formats):
#         super(MaxMonitor, self).__init__(values_to_monitor, formats)

#         for fmt in formats:
#             assert_in('b', fmt.axes)

#         self._maxes = None

#     def _on_batch(self, input_batches, monitored_value_batches):
#         batch_axes = [fmt.axes.index('b') for fmt in self._formats]

#         new_maxes = [numpy.max(batch, axis=batch_axis)
#                      for batch, batch_axis
#                      in safe_izip(batches, batch_axes)]

#         for new_max, old_max, batch_axis in safe_izip(new_maxes,
#                                                       self._maxes,
#                                                       batch_axes):
#             stack = numpy.concatenate((new_max, old_max), axis=batch_axis)
#             old_max[...] = numpy.max(stack, axis=batch_axis, keepdims=True)

#     def _on_epoch(self):
#         result = self._maxes
#         self._maxes = None
#         return result


# class MinMonitor(Monitor):
#     def __init__(self, values_to_monitor, formats):
#         super(MinMonitor, self).__init__(values_to_monitor, formats)

#         for fmt in formats:
#             assert_in('b', fmt.axes)

#         self._mins = None

#     def _on_batch(self, input_batches, monitored_value_batches):
#         batch_axes = [fmt.axes.index('b') for fmt in self._formats]

#         new_mins = [numpy.min(batch, axis=batch_axis)
#                     for batch, batch_axis
#                     in safe_izip(batches, batch_axes)]

#         for new_max, old_max, batch_axis in safe_izip(new_mins,
#                                                       self._mins,
#                                                       batch_axes):
#             stack = numpy.concatenate((new_max, old_max), axis=batch_axis)
#             old_max[...] = numpy.max(stack, axis=batch_axis, keepdims=True)

#     def _on_epoch(self):
#         result = self._mins
#         self._mins = None
#         return result


class ReduceMonitor(Monitor):
    '''
    An abstract superclass of monitors like MaxMonitor, MinMonitor,
    that operate by applying a reduction operator (e.g. max, min)
    along the batch axis for each batch.
    '''

    def __init__(self, values_to_monitor, formats, callbacks):
        super(ReduceMonitor, self).__init__(values_to_monitor,
                                            formats,
                                            callbacks)

        assert_greater(len(self._formats), 0)
        assert_greater(len(self._callbacks), 0)

        for fmt in self._formats:
            assert_in('b', fmt.axes)

        self._tallies = None

    def _reduce_batch(self, input_batch, batch_axis):
        '''
        Reduce input_batch along its batch_axis, and return the result.
        '''
        raise NotImplementedError("%s._reduce_batch() not yet implemented." %
                                  type(self))

    def _update_tally(self, reduced_value, batch_axis, tally):
        '''
        Updates a tally (one of self._tallies) using a reduced batch.
        '''
        raise NotImplementedError("%s._update_tally() not yet implemented." %
                                  type(self))

    def _on_batch(self, input_batches, monitored_value_batches):
        batch_axes = [fmt.axes.index('b') for fmt in self._formats]

        new_tallies = []
        for batch, fmt, batch_axis in safe_izip(monitored_value_batches,
                                                self._formats,
                                                batch_axes):
            new_tally = self._reduce_batch(batch, batch_axis)
            fmt.check(new_tally)
            assert_equal(new_tally.shape[batch_axis], 1)

            new_tallies.append(new_tally)

        new_tallies = tuple(new_tallies)
        # new_tallies = [self._reduce_batch(batch, batch_axis)
        #                for batch, batch_axis
        #                in safe_izip(batches, batch_axes)]

        if self._tallies is None:
            self._tallies = new_tallies
        else:
            for new_tally, old_tally, batch_axis in safe_izip(new_tallies,
                                                              self._tallies,
                                                              batch_axes):
                self._update_tally(new_tally, batch_axis, old_tally)

    def _on_epoch(self):
        assert_is_not(self._tallies, None)

        result = self._tallies
        self._tallies = None

        return result


class MaxMonitor(ReduceMonitor):
    '''
    Computes the elementwise maximum of monitored values, over the batch axis.
    '''

    def __init__(self, values_to_monitor, formats, callbacks):
        super(AverageMonitor, self).__init__(values_to_monitor, formats)

    def _reduce_batch(self, input_batch, batch_axis):
        return numpy.max(input_batch, axis=batch_axis)

    def _update_tally(self, reduced_value, batch_axis, tally):
        stack = numpy.concatenate((reduced_value, tally), axis=batch_axis)
        tally[...] = numpy.max(stack, axis=batch_axis, keepdims=True)


class MinMonitor(ReduceMonitor):
    '''
    Computes the elementwise minimum of monitored values, over the batch axis.
    '''

    def __init__(self, values_to_monitor, formats, callbacks):
        super(AverageMonitor, self).__init__(values_to_monitor, formats)

    def _reduce_batch(self, input_batch, batch_axis):
        return numpy.min(input_batch, axis=batch_axis)

    def _update_tally(self, reduced_value, batch_axis, tally):
        stack = numpy.concatenate((reduced_value, tally), axis=batch_axis)
        tally[...] = numpy.min(stack, axis=batch_axis, keepdims=True)


class SumMonitor(ReduceMonitor):
    '''
    Computes the elementwise sum of monitored values over the batch axis.
    '''

    def __init__(self, values_to_monitor, formats, callbacks):
        super(SumMonitor, self).__init__(values_to_monitor,
                                         formats,
                                         callbacks)
        self._count = None

    def _reduce_batch(self, input_batch, batch_axis):
        return numpy.sum(input_batch, axis=batch_axis, keepdims=True)

    def _update_tally(self, reduced_value, batch_axis, tally):
        tally += reduced_value


class AverageMonitor(SumMonitor):
    '''
    Computes the elementwise average of monitored values over the batch axis.
    '''

    def __init__(self, values_to_monitor, formats, callbacks):
        super(AverageMonitor, self).__init__(values_to_monitor,
                                             formats,
                                             callbacks)
        self._count = 0

    def _on_batch(self, input_batches, monitored_value_batches):
        # Update self._tallies
        super(AverageMonitor, self)._on_batch(input_batches,
                                              monitored_value_batches)
        assert_is_instance(self._tallies, Sequence)

        batch_axes = [fmt.axes.index('b') for fmt in self._formats]

        # Update self._count
        batch_sizes = numpy.asarray([batch.shape[batch_axis]
                                     for batch, batch_axis
                                     in safe_izip(monitored_value_batches,
                                                  batch_axes)])
        assert_true(numpy.all(batch_sizes[0] == batch_sizes[1:]),
                    "Unequal batch sizes: %s" % str(batch_sizes))
        self._count += batch_sizes[0]

    def _on_epoch(self):
        totals = super(AverageMonitor, self)._on_epoch()
        assert_is_instance(totals, Sequence)

        result = tuple(total / self._count for total in totals)
        self._count = 0

        return result


# class AverageMonitor(ReduceMonitor):
#     def __init__(self, values_to_monitor, formats):
#         super(AverageMonitor).__init__(values_to_monitor, formats)
#         self._count = None

#     def _reduce_batch(self, input_batch, batch_axis):
#         batch_size = input_batch.shape[batch_axis]

#         if self._batch_size is None:
#             self._batch_size = batch_size
#         else:
#             assert_equal(self._batch_size, batch_size)

#         return numpy.sum(input_batch, axis=batch_axis)

#     def _update_tally(self, reduced_value, batch_axis, tally):
#         tally += reduced_value

#     def _on_batch(self, input_batches, monitored_value_batches):
#         # Update self._tallies
#         super(self, AverageMonitor)._on_batch(input_batches,
#                                               monitored_value_batches)

#         # Update self._count
#         batch_sizes = [batch.shape[batch_axis]
#                        for batch, batch_axis
#                        in safe_izip(monitored_value_batches, batch_axes)]
#         assert_true(numpy.all(batch_sizes[0] == batch_sizes[1:]))
#         self._count += batch_sizes[0]

#     def _on_epoch(self):
#         totals = super(AverageMonitor, self)._on_epoch()
#         result = [total / self._count for total in totals]
#         self._count = None

#         return result



# class AverageMonitor_old(Monitor):
#     def __init__(self, values_to_monitor, formats):
#         super(AverageMonitor, self).__init__(values_to_monitor, formats)
#         assert_greater_equal(len(values_to_monitor), 0)

#         # values must have names
#         for value in values_to_monitor:
#             assert_in('name', value.__dict__)
#             assert_is_instance(value.name, str)

#         # value names must be unique
#         if len(frozenset(v.name for v in values_to_monitor)) != \
#            len(values_to_monitor):
#             raise ValueError("There were some duplicate variable names: %s" %
#                              str(tuple(v.name for v in values_to_monitor)))

#         self._totals = None
#         self._count = 0
#         self.averages = None

#     def _on_batch(self, input_batches, monitored_value_batches):

#         # Create _totals if this is the first call in an epoch.
#         if self._totals is None:
#             self._totals = []
#             for batch, fmt in safe_izip(monitored_value_batches,
#                                         self._formats):
#                 batch_axis = fmt.axes.index('b')
#                 self._totals.append(numpy.sum(batch, axis=batch_axis))

#         # Otherwise, accumulate monitored values into _totals.
#         else:
#             for total, batch, fmt in safe_izip(self._totals,
#                                                monitored_value_batches,
#                                                self._formats):
#                 batch_axis = fmt.axes.index('b')
#                 total += numpy.sum(batch, axis=batch_axis)

#         def get_num_examples(batches, formats):
#             '''
#             Returns the # of examples in a batch of sub-batches.

#             Checks that all sub-batches contain the same # of examples.
#             '''

#             result = None

#             # Checks that all batches have the same number of examples
#             for batch, fmt in safe_izip(batches, formats):
#                 batch_axis = fmt.axes.index('b')
#                 batch_size = batch.shape[batch_axis]

#                 if result is None:
#                     result = batch_size
#                 else:
#                     assert_equal(batch_size, result)

#             assert_is_not(result, None)

#             return result

#         self._count += get_num_examples(monitored_value_batches, self._formats)

#     def on_epoch(self):
#         if self._count != 0:
#             self.averages = tuple(total / self._count
#                                   for total in self._totals)

#         self._totals = None
#         self._count = 0


# class MaxMonitor(TrainingMonitor):
#     """
#     Keeps track of the N largest values of some f(x), along with inputs x.

#     The list of values gets cleared after each epoch.
#     """

#     def __init__(self, value_to_monitor, top_n=1):
#         fmt = DenseFormat(shape=(), axes=('b',), dtype=value_to_monitor.dtype)
#         super(MaxMonitor, self).__init__(value_to_monitor, fmt)

#         # assert_equal(value_to_monitor.ndim, 1)
#         assert_greater(top_n, 0)

#         self.maxes = fmt.make_batch(is_symbolic=False, batch_size=0)
#         self._top_n = top_n

#     def _on_batch(self, input_batch, *monitored_value_batches):
#         batch_axis = self._format.axes.index('b')
#         batch_max = numpy.max(monitored_value_batch, axis=batch_axis)

#         indices = numpy.searchsorted(self.maxes, batch_max)
#         assert len(indices) == 1

#         if len(self.maxes) < self._top_n or indices[0] > 0:
#             self.maxes = numpy.insert(self.maxes, indices, (batch_max, ))
#             if len(self.maxes) == self._top_n:
#                 self.maxes = self.maxes[1:]

#         assert len(self.maxes) <= self._top_n

#     def on_epoch(self):
#         self.maxes = self._format.make_batch(is_symbolic=False, batch_size=0)


# class ComputesAverageOverEpoch_old(object):
#     """
#     Epoch callback. Computes the average of a function over an epoch of data.

#     On call, this loops over a data iterator, computing f(x) for each
#     datum x, where f is given in the constructor. After an epoch's worth
#     of data, this sums all the f(x)'s, divides by the number of samples,
#     and passes the result to any interested callbacks.
#     """
#     def __init__(self, function_node, data_iterator, callbacks):
#         """
#         Parameters
#         ----------

#         function_node: Node
#           A Node whose inputs are a DataSource's output nodes.

#         data_iterator: DataIterator
#           Iterates over the DataSource connected to function_node.

#         callbacks: sequence
#           A sequence of callables. Call signature must be f(x), where x
#           is a numeric batch of outputs from function_node.
#         """

#         assert_is_instance(function_node, Node)
#         assert_true(data_iterator.next_is_new_epoch(),
#                     "iterator doesn't point to the beginning of an epoch.")
#         assert_is_isntance(callbacks, collections.Sequence)

#         self._function_batch_axis = function_node.output_format.axes.index('b')

#         input_symbols = tuple(input_node.output_symbol
#                               for input_node in function_node.inputs)
#         self._function = theano.function(input_symbols,
#                                          function_node.output_symbol)
#         self._iterator = data_iterator
#         self._callbacks = callbacks

#     def __call__(self):
#         if not self._iterator.next_is_new_epoch():
#             raise ValueError("self._iterator doesn't point to a fresh epoch.")

#         count = 0
#         total = None

#         batch = self._function(*self._iterator.next())
#         count += batch.shape[self._function_batch_axis]
#         total = batch.sum(axis=self._function_batch_axis)

#         while not self._iterator.next_is_new_epoch():
#             batch = self._function(*self._iterator.next())
#             count += batch.shape[self._function_batch_axis]
#             total += batch.sum(axis=self._function_batch_axis)

#         average = total / count

#         for callback in self._callbacks:
#             callback(average)

class StopTraining(Exception):
    """
    An exception thrown to signal the end of training.

    Analogous to the built-in exception StopIteration.
    """
    def __init__(self, status, message):
        if status not in ('ok', 'error'):
            raise ValueError("Expected StopTraining status to be 'ok' or "
                             "'error', but got '%s'." % str(status))

        self.status = status
        super(StopTraining, self).__init__(message)


class StopsOnStagnation(object):
    '''
    A callback to Monitor that stops training if the monitored value
    (e.g. average loss over the epoch) doesn't decrease for N epochs.
    '''

    def __init__(self, max_epochs, min_decrease):
        assert_greater(max_epochs, 0)
        assert_true(numpy.issubdtype(type(max_epochs), numpy.integer))

        assert_greater_equal(min_decrease, 0.0)

        self._max_epochs_since_min = max_epochs
        self._min_decrease = min_decrease
        self._epochs_since_min = 0
        self._min_value = None

    def __call__(self, values, formats):
        assert_equal(len(values), 1)
        assert_equal(len(values), len(formats))

        fmt = formats[0]
        assert_equal(fmt.axes, ('b', ))

        assert_equal(values[0].shape, (1, ))
        value = values[0][0]

        if self._min_value is None:
            self._min_value = value
        elif value + self._min_decrease < self._min_value:
            self._epochs_since_min = 0
            self._min_value = value
        else:
            self._epochs_since_min += 1

        if self._epochs_since_min >= self._max_epochs_since_min:
            raise StopTraining("ok",
                               "%s stopping training. Value did not lower "
                               "below last min value of %g for %d epochs." %
                               (type(self),
                                self._min_value,
                                self._epochs_since_min))


class LogsToLists(object):
    '''
    A callback to Monitor that logs monitored values to lists.
    '''
    def __init__(self):
        self.logs = None

    def __call__(self, values, formats):
        assert_equal(len(values), len(formats))
        assert_greater(len(values), 0)

        if self.logs is None:
            self.logs = [list() for value in values]
        else:
            assert_equal(len(self.logs), len(values))

        for log, value in safe_izip(self.logs, values):
            log.append(value)

# class StopsOnStagnation(AverageMonitor):
#     """
#     Halts training if the average of f(x) over an epoch stops decreasing.
#     """

#     def __init__(self,
#                  value_to_monitor,
#                  value_format,
#                  num_epochs,
#                  min_decrease=0.0,
#                  value_name=None):
#         '''
#         Parameters
#         ----------
#         value_to_monitor: theano expression
#           Some f(x) (x is the data iterator output batch).

#         value_format: Format
#           Data format of value_to_monitor

#         num_epochs: int
#           maximum number of epochs to wait for average f to decrease.

#         min_decrease: float
#           minimum decrease in avg f needed for it to count as a decrease.

#         value_name: str
#           If value_to_monitor doesn't have a name, specify it here.
#         '''

#         check_is_subdtype(num_epochs, 'num_epochs', numpy.integer)
#         assert_greater(num_epochs, 0)
#         check_is_subdtype(min_decrease, 'min_decrease', numpy.floating)
#         assert_greater_equal(min_decrease, 0.0)
#         assert_not_equal(value_to_monitor.name is None,
#                          value_name is None,
#                          ("value_to_monitor.name was specified. No need to "
#                           "provide value_name argument"
#                           if value_to_monitor.name is not None
#                           else ("when value_to_monitor has no name, you must "
#                                 "provide a <name> argument.")))

#         super(StopsOnStagnation, self).__init__([value_to_monitor],
#                                                 [value_format])

#         self._max_epochs_since_min = num_epochs
#         self._epochs_since_min = 0
#         self._min_decrease = min_decrease
#         self._min_value = numpy.inf
#         self._value_name = (value_to_monitor.name
#                             if value_to_monitor.name is not None
#                             else value_name)

#     def on_epoch(self):
#         # Computes average value over epoch
#         super(StopsOnStagnation, self).on_epoch()

#         # There were no batches in the previous epoch
#         if self.averages is None:
#             return
#         else:
#             assert_equal(len(self.averages), 1)
#             average = self.averages[0]

#         if self._min_value - average > self._min_decrease:
#             self._min_value = average
#             self._epochs_since_min = 0
#         else:
#             self._epochs_since_min += 1

#         if self._epochs_since_min > self._max_epochs_since_min:
#             raise StopTraining(status='ok',
#                                message=("%s didn't decrease for %d epochs." %
#                                         (self.monitored_values[0].name,
#                                          self._epochs_since_min)))


# class StopsOnStagnation_old(object):
#     """
#     A callback to give ComputesAverageOverEpoch.

#     Stops the training when the average f(x_i) over epoch x stops decreasing.
#     """

#     def __init__(self, name, num_epochs, min_decrease=0.0):
#         """
#         Parameters
#         ----------

#         name: str
#           Name of the quantity being monitored.
#         """

#         #
#         # Sanity-checks args.
#         #

#         assert_is_instance(name, str)
#         check_is_subdtype(num_epochs, 'num_epochs', numpy.integer)

#         check_greater(num_epochs, 0)

#         if not numpy.issubdtype(type(min_decrease), numpy.floating):
#             raise TypeError("Expected a floating-point value for "
#                             "min_decrease, but got a %s." % type(min_decrease))

#         if min_decrease < 0.0:
#             raise ValueError("Expected min_decrease to be non-negative, but "
#                              "got %g." % min_decrease)

#         #
#         # Sets members
#         #

#         self._name = name
#         self._max_epochs_since_min = num_epochs
#         self._epochs_since_min = 0
#         self._min_decrease = min_decrease
#         self._min_value = numpy.inf

#     def __call__(self, average_over_epoch):
#         if average_over_epoch < self._min_value:
#             self._min_value = average_over_epoch
#             self._epochs_since_min = 0
#         else:
#             self._epochs_since_min += 1

#         if self._epochs_since_min > self._max_epochs_since_min:
#             raise StopTraining(status='ok',
#                                message=("%s didn't decrease for %d epochs." %
#                                         (self._name, self._epochs_since_min)))


class SgdParameterUpdater(object):
    """
    Defines how to update parameters using SGD with momentum.

    You can set the learning rate and momentum dynamically during the
    optimization.

    Fields
    ------
    learning_rate: theano.tensor.SharedScalarVariable
      Call set_value() on this to change the learning rate.

    momentum:  theano.tensor.SharedScalarVariable
      Call set_value() on this to change the momentum.

    updates: dict
      A dictionary with (var: new_var) pairs, where var and new_var are
      Theano expressions. At each training update, var's value will be
      replaced with new_var.

      This contains the update for not just a parameter, but also the internal
      state, such as the as the momentum-averaged update direction.
    """

    def __init__(self,
                 parameter,
                 gradient,  # see (*) below
                 learning_rate,
                 momentum,
                 use_nesterov):

        # (*): We pass in the gradient, rather than the cost, since there are
        # different ways to generate the gradient expression, and we want to
        # allow the user to choose different ones, rather than generating the
        # gradient here ourselves. In particular, the 'consider_constant'
        # argument to theano.gradient.grad() could be of interest to the user.
        # (It's a list of symbols to consider constant, and thus not
        # backpropagate through.)
        """
        Parameters
        ----------
        parameter: A theano symbol
          A parameter being optimized by an Sgd trainer.

        gradient: A theano symbol
          The gradient of the loss function w.r.t. the above parameter.

        learing_rate: float
          The initial value of the learning rate.

        momentum: float
          A parameter affecting how smeared the update direction is over
          multiple batches. Use 0.0 for momentum-less SGD.

        use_nesterov: bool
          If true, use Nesterov momentum. (See "Advances in Optimizing
          Recurrent Networks", Yoshua Bengio, et al.)
        """

        #
        # sanity-check args
        #

        assert_is_instance(parameter, theano.tensor.sharedvar.SharedVariable)
        assert_is_instance(gradient, theano.gof.Variable)
        assert_equal(parameter.broadcastable, gradient.broadcastable,
                     "If an Op's .grad() method is buggy, it can return "
                     "broadcast masks.")
        check_is_subdtype(gradient, 'gradient', numpy.floating)
        assert_greater_equal(learning_rate, 0)
        assert_greater_equal(momentum, 0)
        assert_is_instance(use_nesterov, bool)

        floatX = theano.config.floatX

        if str(gradient.dtype) != str(floatX):
            gradient = theano.tensor.cast(gradient, floatX)

        #
        # define updates, set members
        #

        def concat(str0, str1):
            if str0 is None or str1 is None:
                return None
            else:
                return str0 + str1

        def make_shared_floatX(numeric_var, name, **kwargs):
            return theano.shared(numpy.asarray(numeric_var, dtype=floatX),
                                 name=name,
                                 **kwargs)

        self.learning_rate = make_shared_floatX(learning_rate,
                                                concat(parameter.name,
                                                       ' learning rate'))

        self.momentum = make_shared_floatX(momentum,
                                           concat(parameter.name, ' momentum'))

        self._velocity = make_shared_floatX(
            0.0 * parameter.get_value(),
            concat(parameter.name, ' velocity'),
            broadcastable=parameter.broadcastable)

        new_velocity = (self.momentum * self._velocity -
                        self.learning_rate * gradient)
        new_velocity.name = concat('new ', self._velocity.name)

        assert_equal(str(new_velocity.dtype), str(floatX))
        assert_equal(self._velocity.broadcastable, new_velocity.broadcastable)

        step = (self.momentum * new_velocity - self.learning_rate * gradient
                if use_nesterov
                else new_velocity)

        assert_equal(parameter.broadcastable,
                     step.broadcastable)

        new_parameter = parameter + step
        new_parameter.name = concat('new ', parameter.name)

        self.updates = OrderedDict([(parameter, new_parameter),
                                    (self._velocity, new_velocity)])


class Sgd(object):

    """
    Uses stochastic gradient descent to optimize a cost w.r.t. parameters.

    The parameters and the inputs may be the same.

    At each iteration this computes the gradients of each parameter with
    respect to the cost function, then updates the parameter value using
    the gradients. How this update is performed (e.g. learning rate,
    momentum value & type, etc) is up to the GradientBasedParameterUpdater
    objects passed into the constructor.
    """

    def __init__(self,
                 # cost,
                 inputs,
                 parameters,
                 parameter_updaters,
                 input_iterator,
                 monitors,
                 epoch_callbacks):

        """
        Parameters
        ----------

        cost: theano.gof.Variable or None
          The cost to be reduced. Must be a scalar.
          May be None if parameters and parameter_updaters are both empty.

        inputs: sequence of theano.gof.Variables
          The inputs to the cost (e.g. [images, labels]), in the order yielded
          by the input_iterator.

        input_iterator: simplelearn.datasets.Iterator
          Yields training data (inputs' values).

        parameters: sequence of theano.tensor.sharedvar.SharedVariables
          What this trainer modifies to lower the cost. These are typically
          model weights, though they could also be inputs (e.g. for optimizing
          input images).

        parameter_updaters: sequence of SgdParameterUpdaters
          updaters for the corresponding elements in <parameters>.
          These are defined using the loss function to be minimized.

        inputs: sequence of theano.gof.Variables
          These are the inputs to cost.

        monitors: (optional) sequence of Monitors.
          These are also used as epoch callbacks.

        epoch_callbacks: sequence of EpochCallbacks
          One of these must throw a StopTraining exception for the training to
          halt.
        """

        #
        # sanity-checks the arguments.
        #

        # if cost is None:
        #     assert_equal(len(parameters), 0)
        #     assert_equal(len(parameter_updaters), 0)
        # else:
        #     assert_isinstance(cost, theano.gof.Variable)

        assert_is_instance(inputs, Sequence)
        for input_symbol in inputs:
            assert_is_instance(input_symbol, theano.gof.Variable)

        assert_is_instance(parameters, Sequence)
        assert_is_instance(parameter_updaters, Sequence)
        for parameter, updater in safe_izip(parameters, parameter_updaters):
            assert_is_instance(parameter,
                               theano.tensor.sharedvar.SharedVariable)

            assert_is_instance(updater, SgdParameterUpdater)

            assert_in(parameter, updater.updates)

        assert_is_instance(input_iterator, DataIterator)

        assert_true(input_iterator.next_is_new_epoch())

        assert_is_instance(monitors, Sequence)
        for monitor in monitors:
            assert_is_instance(monitor, Monitor)

        assert_is_instance(epoch_callbacks, Sequence)
        for epoch_callback in epoch_callbacks:
            assert_is_instance(epoch_callback, EpochCallback)
            if isinstance(epoch_callback, Monitor):
                warnings.warn("You've passed a Monitor subclass %s "
                              "as one of the epoch_callbacks. If you want the "
                              ".on_batch() method to be called on this, you "
                              "need to pass it in as one of the monitors." %
                              str(epoch_callback))

        #
        # Sets members
        #

        self._input_iterator = input_iterator
        self._parameters = tuple(parameters)
        self._parameter_updaters = tuple(parameter_updaters)
        self._monitors = tuple(monitors)

        def compile_update_function():
            outputs = []
            for monitor in monitors:
                outputs.extend(monitor.monitored_values)

            updates = OrderedDict()
            for updater in parameter_updaters:
                assert_is_instance(updater.updates, OrderedDict)
                updates.update(updater.updates)

            return theano.function(inputs, outputs, updates=updates)

        self._update_function = compile_update_function()

        repeated_callbacks = frozenset(monitors).intersection(epoch_callbacks)
        assert_equal(len(repeated_callbacks),
                     0,
                     "There were duplicate entries between monitors and "
                     "epoch_callbacks: %s" % str(repeated_callbacks))

        # These get called once before any training, and after each epoch
        # thereafter. One of them must halt the training at some point by
        # throwing a StopTraining exception.
        self._epoch_callbacks = tuple(epoch_callbacks)

        self._train_called = False

    def train(self):
        """
        Runs training until a StopTraining exception is raised.

        Training runs indefinitely until one of self.epoch_callbacks raises
        a StopTraining exception.
        """

        if self._train_called and len(self._parameter_updaters) > 0:
            raise RuntimeError("train() has already been called on this %s. "
                               "Re-running train() risks inadvertently "
                               "carrying over implicit state from the "
                               "previous training run, such as the direction "
                               "of parameter updates (via the momentum "
                               "term). Instead, instantiate a new copy of "
                               "this %s and run train() on that." %
                               (type(self), type(self)))

        self._train_called = True

        if len(self._epoch_callbacks) + len(self._monitors) == 0:
            raise RuntimeError("self._monitors and self._epoch_callbacks are "
                               "both empty, so this will "
                               "iterate through the training data forever. "
                               "Please add an EpochCallback or "
                               "Monitor that will throw a "
                               "StopTraining exception at some point.")
        try:
            all_callbacks = self._monitors + self._epoch_callbacks
            for callback in all_callbacks:
                callback.on_start_training()

            while True:

                # gets batch of data
                cost_arguments = self._input_iterator.next()

                # fprop-bprop, updates parameters
                # pylint: disable=star-args
                outputs = self._update_function(*cost_arguments)

                # updates monitors
                output_index = 0
                for monitor in self._monitors:
                    new_output_index = (output_index +
                                        len(monitor.monitored_values))
                    assert_less_equal(new_output_index, len(outputs))
                    monitored_values = outputs[output_index:new_output_index]

                    monitor.on_batch(cost_arguments, monitored_values)

                    output_index = new_output_index

                # calls epoch callbacks, if we've iterated through an epoch
                if self._input_iterator.next_is_new_epoch():
                    for callback in all_callbacks:
                        callback.on_epoch()

        except StopTraining, exception:
            if exception.status == 'ok':
                print("Stopped training with message: %s" % exception.message)
                return
            else:
                raise
