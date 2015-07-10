"""
Training algorithms, and callbacks for monitoring their progress.
"""

from __future__ import print_function

__author__ = "Matthew Koichi Grimes"
__email__ = "mkg@alum.mit.edu"
__copyright__ = "Copyright 2015"
__license__ = "Apache 2.0"

import os
import copy
import warnings
import cPickle
from collections import Sequence, OrderedDict
import numpy
import theano
import theano.tensor as T
from nose.tools import (assert_true,
                        assert_is,
                        assert_equal,
                        assert_less,
                        assert_less_equal,
                        assert_greater,
                        assert_greater_equal,
                        assert_is_instance,
                        assert_is_not,
                        assert_in)
from simplelearn.asserts import (assert_integer,
                                 assert_floating,
                                 assert_all_less,
                                 assert_all_greater_equal,
                                 assert_all_integer,
                                 assert_is_subdtype,
                                 assert_all_is_instance)
from simplelearn.data import DataIterator
from simplelearn.utils import safe_izip
from simplelearn.formats import DenseFormat
from simplelearn.nodes import Node
import pdb

# pylint: disable=too-few-public-methods


class StopTraining(Exception):
    '''
    An exception thrown to signal the end of training.

    Analogous to the built-in exception StopIteration.
    '''
    def __init__(self, status, message):
        if status not in ('ok', 'error'):
            raise ValueError("Expected StopTraining status to be 'ok' or "
                             "'error', but got '%s'." % str(status))

        self.status = status
        super(StopTraining, self).__init__(message)


class EpochCallback(object):
    '''
    Abstract class for callbacks to call between training epochs.
    '''

    def on_start_training(self):
        '''
        Called at the beginning of training, before processing any batches.
        '''
        raise NotImplementedError("%s.on_start_training() not yet implemented."
                                  % type(self))

    def on_epoch(self):
        '''
        Called after each epoch of training.
        '''
        raise NotImplementedError("%s.on_epoch() not yet implemented." %
                                  type(self))


class LimitsNumEpochs(EpochCallback):
    '''
    Throws a StopTraining exception after a fixed number of epochs.
    '''

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

class PicklesOnEpoch(EpochCallback):
    '''
    A callback that saves a list of objects at the start of training, and
    again on each epoch.
    '''
    def __init__(self, objects, filepath, overwrite=True):
        '''
        Parameters
        ----------
        objects: OrderedDict
          Maps names to picklable objects. This dict is pickled at each epoch.
          Note that dynamically created functions (e.g. inner functions that
          aren't publically accessible by name) are not picklable.
          module-level An object, or a sequence of objects, to pickle at each
          epoch.

        filepath: str
          The file path to save the objects to. Must end in '.pkl'

        overwrite: bool
          Overwrite the file at each epoch. If False, and filepath is
          'path/to/file.pkl', this saves a separate file for each epoch,
          of the form 'path/to/file_00000.pkl', 'path/to/file_00001.pkl',
          etc. The first file stores the state of <objects> before any epochs.
        '''
        assert_is_instance(objects, OrderedDict)
        for key in objects.keys():
            assert_is_instance(key, basestring)

        if os.path.isdir(filepath):
            path = filepath
            filename = ""
        else:
            path, filename = os.path.split(filepath)

        assert_true(os.path.isdir(path), "{} isn't a directory".format(path))
        assert_equal(os.path.splitext(filename)[1], '.pkl')


        self._objects_to_pickle = objects
        self._filepath = filepath
        self._overwrite = overwrite
        self._num_epochs_seen = 0

    def on_start_training(self):
        self.on_epoch()

    def on_epoch(self):
        if self._overwrite:
            filepath = self._filepath
        else:
            path, filename = os.path.split(self._filepath)
            extension = os.path.splitext(filename)[1]
            filename = '%s_%05d%s' % (filename,
                                      self._num_epochs_seen,
                                      extension)

            filepath = os.path.join(path, filename)

        with file(filepath, 'wb') as pickle_file:

            cPickle.dump(self._objects_to_pickle,
                         pickle_file,
                         protocol=cPickle.HIGHEST_PROTOCOL)

        self._num_epochs_seen += 1

class ValidationCallback(EpochCallback):
    '''
    Evaluates some EpochCallbacks over validation data between training epochs.
    '''

    def __init__(self, inputs, input_iterator, epoch_callbacks):
        '''
        Parameters
        ----------

        inputs: sequence of theano.gof.Variables
          Symbols for the outputs of the input_iterator.

        input_iterator: simplelearn.data.DataIterator
          Yields tuples of validation set batches, such as (values, labels).

        epoch_callbacks: Sequence of EpochCallbacks.
        '''

        #
        # Checks inputs
        #

        assert_is_instance(inputs, Sequence)
        assert_all_is_instance(inputs, theano.gof.Variable)

        assert_is_instance(input_iterator, DataIterator)
        assert_true(input_iterator.next_is_new_epoch())

        assert_is_instance(epoch_callbacks, Sequence)
        assert_greater(len(epoch_callbacks), 0)
        assert_all_is_instance(epoch_callbacks, EpochCallback)

        #
        # Sets members
        #

        self._input_iterator = input_iterator

        symbols_to_compute = []
        update_pairs = OrderedDict()

        for epoch_callback in epoch_callbacks:
            if isinstance(epoch_callback, IterationCallback):
                callback_symbols = [node.output_symbol for node
                                    in epoch_callback.nodes_to_compute]
                symbols_to_compute.extend(callback_symbols)
                update_pairs.update(epoch_callback.update_pairs)

        if len(update_pairs) > 0:
            warnings.warn("Are you sure you meant to pass IterationCallbacks "
                          "with update pairs to a ValidationCallback? "
                          "ValidationCallbacks are generally supposed to "
                          "operate without side-effects.")

        if any(not isinstance(c, IterationCallback) for c in epoch_callbacks):
            warnings.warn("It's rare to pass in a non-IterationCallback to "
                          "ValidationCallback. Did you mean to do this?")

        self._epoch_callbacks = epoch_callbacks

        self._update_function = theano.function(inputs,
                                                symbols_to_compute,
                                                updates=update_pairs)

    def on_start_training(self):
        self.on_epoch()

    def on_epoch(self):
        '''
        Loops through an epoch of the validation dataset.
        '''

        # Calls epoch_callbacks' on_start_training()
        for epoch_callback in self._epoch_callbacks:
            epoch_callback.on_start_training()

        # Repeatedly calls epoch_callbacks' on_batch()
        keep_going = True

        while keep_going:
            input_batches = self._input_iterator.next()
            keep_going = not self._input_iterator.next_is_new_epoch()

            # pylint: disable=star-args
            computed_values = self._update_function(*input_batches)

            value_index = 0
            for epoch_callback in self._epoch_callbacks:
                if isinstance(epoch_callback, IterationCallback):
                    new_value_index = (value_index +
                                       len(epoch_callback.nodes_to_compute))
                    assert_less_equal(new_value_index, len(computed_values))

                    values = computed_values[value_index:new_value_index]
                    epoch_callback.on_iteration(values)

                    value_index = new_value_index

        # Calls epoch_callbacks' on_epoch() methods.
        for epoch_callback in self._epoch_callbacks:
            epoch_callback.on_epoch()


class LinearlyInterpolatesOverEpochs(EpochCallback):
    '''
    Linearly interpolates a theano shared variable over epochs.
    '''

    def __init__(self,
                 shared_value,
                 final_value,
                 epochs_to_saturation):
        assert_is_instance(shared_value,
                           theano.tensor.sharedvar.SharedVariable)
        assert_is_subdtype(shared_value.dtype, numpy.floating)

        assert_equal(shared_value.ndim == 0, numpy.isscalar(final_value))

        if numpy.isscalar(final_value):
            assert_floating(final_value)
        else:
            assert_is_subdtype(final_value.dtype, numpy.floating)
            assert_equal(final_value.shape,
                         shared_value.get_value().shape)

        assert_integer(epochs_to_saturation)
        assert_greater(epochs_to_saturation, 0)

        self.shared_value = shared_value

        cast = numpy.cast[shared_value.dtype]
        self._final_value = cast(final_value)

        self._epochs_to_saturation = epochs_to_saturation

        self._num_epochs_seen = None
        self._initial_value = None

    def on_start_training(self):
        self._num_epochs_seen = 0
        self._initial_value = self.shared_value.get_value()

    def on_epoch(self):
        assert_greater_equal(self._num_epochs_seen, 0)
        self._num_epochs_seen += 1

        cast = numpy.cast[self.shared_value.dtype]

        # interpolation parameter
        end_weight = cast(min(
            1.0,
            float(self._num_epochs_seen) / self._epochs_to_saturation))

        start_weight = cast(1.0) - end_weight

        self.shared_value.set_value(
            start_weight * self._initial_value +
            end_weight * self._final_value)


class LinearlyScalesOverEpochs(EpochCallback):
    '''
    Linearly scales a theano shared variable over epochs.

    Parameters
    ----------

    shared_value: a Theano shared variable
      This value will be scaled in-place by a factor S that decreases from 1.0
      to final_scale over <epochs_to_saturation> epochs.

    final_scale: float
      Final value of S. Mutually exclusive with final_value.

    final_value: numpy.ndarray
      A numpy array of the same shape as shared_value.get_value().shape.
      Mutually exclusive with final_scale.

    epochs_to_saturation: int
      self._scale should decay to final_value after this many epochs.
    '''

    def __init__(self,
                 shared_value,
                 final_scale,
                 epochs_to_saturation):
        assert_is_instance(shared_value,
                           theano.tensor.sharedvar.SharedVariable)
        assert_floating(final_scale)
        assert_greater_equal(final_scale, 0.0)
        assert_less_equal(final_scale, 1.0)
        assert_integer(epochs_to_saturation)
        assert_greater(epochs_to_saturation, 0)

        self.shared_value = shared_value
        self._final_scale = final_scale
        self._epochs_to_saturation = epochs_to_saturation

        # initialized in on_start_training()
        self._initial_value = None
        self._num_epochs_seen = None

    def on_start_training(self):
        self._num_epochs_seen = 0
        self._initial_value = self.shared_value.get_value()

    def on_epoch(self):
        assert_greater_equal(self._num_epochs_seen, 0)
        self._num_epochs_seen += 1

        # interpolation parameter
        alpha = min(1.0,
                    float(self._num_epochs_seen) / self._epochs_to_saturation)

        scale = (1.0 - alpha) + alpha * self._final_scale

        self.shared_value.set_value(scale * self._initial_value)


class IterationCallback(EpochCallback):
    '''
    Gets called by Sgd after every training iteration.

    Can optionally provide symbolic expressions for the Sgd update
    function to compute, and/or to update in-place.
    '''

    def __init__(self, nodes_to_compute=None, update_pairs=None):
        '''
        Parameters
	----------
	nodes_to_compute: None, or Sequence of Nodes

	  Nodes to compute on each iteration. Sgd's update function will
          output their values at each iteration. These values will then be
          passed to this IterationCallback's on_iteration().

	update_pairs: None, or Sequence of pairs

          A sequence of pairs ((S0, N0), (S1, N1), ...) of Theano shared
          variables Si and symbolic expressions of their new values Ni.
          See docs for 'updates' kwarg for theano.function().
        '''
        if nodes_to_compute is None:
	    nodes_to_compute = list()

        self.nodes_to_compute = list(nodes_to_compute)

        if update_pairs is None:
            update_pairs = list()

        self.update_pairs = update_pairs

    def on_iteration(self, computed_values):
        # sanity-check formats of computed_values
        for value, node in safe_izip(computed_values,
                                     self.nodes_to_compute):
            node.output_format.check(value)

        rval = self._on_iteration(computed_values)
        assert_is(rval,
                  None,
                  ("{}._on_iteration implemented incorrectly. It "
                   "shouldn't return anything.".format(type(self))))

    def _on_iteration(self, computed_values):
        raise NotImplementedError(
            '{}._on_iteration() not implemented.'.format(type(self)))

    def on_start_training(self):
        pass  # ok to leave unimplemented

    def on_epoch(self):
        pass  # ok to leave unimplemented


class ParameterUpdater(IterationCallback):
    '''
    An IterationCallback limited to just updating shared variables.
    '''

    def __init__(self, update_pairs):
        assert_greater(len(update_pairs), 0)
        super(ParameterUpdater, self).__init__(update_pairs=update_pairs)

    def _on_iteration(self, computed_values):
        assert_equal(len(computed_values), 0)


# class Monitor(IterationCallback):
#     '''
#     Monitors some value over an epoch (e.g. average loss)
#     '''
#     # On each epoch, this reports statistics about some monitored value Y.

#     # Examples: Y might be the output of layer 3 of a 6-layer net.

#     #           MaxMonitor reports the elementwise maximum of Y encountered
#     #           over the epoch.

#     #           AverageMonitor reports Y, elementwise-averaged over the epoch.
#     # '''

#     def __init__(self, nodes_to_monitor, callbacks):
#         '''
#         Parameters
#         ----------
#         nodes_to_monitor: Sequence of Nodes.
#           Nodes whose values you want to monitor. These must lie upstream
#           in the computational graph from the data iterator's nodes.

#           Must not be empty.

#         callbacks: a __call__-able, or a Sequence of them.

#           The call signature of these must be f(values, formats), where:

#           values: Sequence of numpy.ndarrays
#             The values computed by nodes_to_monitor.

#           formats: Sequence of DenseFormats
#             The corresponding formats of the above values.
#         '''

#         #
#         # Checks args
#         #

#         if isinstance(nodes_to_monitor, Node):
#             nodes_to_monitor = [nodes_to_monitor]
#         else:
#             nodes_to_monitor = tuple(nodes_to_monitor)
#             assert_all_is_instance(nodes_to_monitor, Node)

#         assert_equal(len(nodes_to_monitor), len(frozenset(nodes_to_monitor)),
#                      "nodes_to_monitor contains repeated elements: %s" %
#                      str(nodes_to_monitor))

#         if not isinstance(callbacks, Sequence):
#             callbacks = [callbacks]
#         else:
#             callbacks = tuple(callbacks)

#         for callback in callbacks:
#             assert_true(callable(callback))

#         #
#         # Sets members
#         #

#         self._monitored_nodes = nodes_to_monitor
#         self._callbacks = callbacks

#     def on_batch(self, input_batches, monitored_value_batches):
#         '''
#         Updates the values to report at the end of the epoch.

#         Parameters
#         ----------
#         input_batches: Sequence of numpy.ndarrays
#           The input batches coming in from the dataset's iterator.
#           Typically these are (values, labels)

#         monitored_value_batches: Sequence of numpy.ndarrays
#           The numerical values, for this batch, of the values_to_monitor
#           arguments to __init__().
#         '''
#         assert_equal(len(monitored_value_batches),
#                      len(self._monitored_nodes))

#         for batch, node in safe_izip(monitored_value_batches,
#                                      self._monitored_nodes):
#             node.output_format.check(batch)

#         self._on_batch(tuple(input_batches), tuple(monitored_value_batches))

#     def _on_batch(self, input_batches, monitored_value_batches):
#         '''
#         Implementation of self.on_batch(). See that method's docs.

#         Parameters
#         ----------
#         input_batches: tuple of numpy.ndarrays.

#         monitored_value_batches: tuple of numpy.ndarrays.
#         '''
#         raise NotImplementedError("%s._on_batch() not yet implemented." %
#                                   type(self))

#     def on_start_training(self):
#         pass

#     def on_epoch(self):
#         '''
#         Feeds monitored values to self._callbacks
#         '''
#         # compute values to report
#         values_to_report = self._on_epoch()

#         if not isinstance(values_to_report, tuple):
#             raise ValueError("%s._on_epoch() implemented incorrectly. It "
#                              "should return a tuple, but it returned %s."
#                              % (type(self), type(values_to_report)))

#         for callback in self._callbacks:
#             formats = [node.output_format for node in self._monitored_nodes]
#             callback(values_to_report, formats)

#     def _on_epoch(self):
#         '''
#         Returns a tuple of values to feed to self._callbacks as arguments.

#         Returns
#         -------
#         rval: tuple of numpy.ndarrays
#            Arguments to feed to self._callbacks' __call__(self, *args)
#         '''
#         raise NotImplementedError("%s._on_epoch() not yet implemented" %
#                                   type(self))


class ReduceOverEpoch(IterationCallback):
    '''
    Superclass of IterationCallbacks like MaxOverEpoch, MeanOverEpoch.

    At each iteration, this computes a per-example quantity X, and updates some
    reduction Y of X (e.g. max(X), mean(X)) over the batch axis.

    For example, X could be the per-example loss, and Y the average loss over
    the epoch.

    We're using the term "reduce" in the sense of "summarize many values as a
    single value" (a la mapreduce), not in the sense of "lessen in value".

    Specify X in the constructor.
    Specify Y by overriding update_reduction and _on_epoch().
    Y's format is stored in self.reduction_format.
    '''

    def __init__(self,
                 node_to_reduce,
                 callbacks,
                 reduction_format=None,
                 output_format=None):
        '''
        Parameters
        ----------
        node_to_reduce: Node
           The node whose computed value will be passed to
           self._update_reduction(). Must have a 'b' axis.

        callbacks: a callable, or a Sequence of callables
           The reduction will be passed to each of these callables F as
           F(reduction, reduction_format)

        reduction_format: DenseFormat
           The format of self._reduction. If omitted, this is assumed to be the
           same as per_batch_node.output_format.

           Example: in SumOverFormat, if the batches have dtype=uint8,
                    reduction_format has dtype=int64, to prevent overflow.

        output_format: DenseFormat
           The format of the value passed to the callbacks at the end of each
           epoch. If omitted, this is assumed to be the same as reduction_format.

           Example: in MeanOverFormat, if the batches have integer dtype, the
                    reduction_format (sum over batches) also has integer dtype,
                    but the output (average over batches) must have floating-
                    point dtype, since we divide by the number of examples.
        '''

        #
        # Sanity-check args
        #

        assert_is_instance(node_to_reduce, Node)
        assert_in('b', node_to_reduce.output_format.axes)

        assert_is_instance(callbacks, Sequence)
        for callback in callbacks:
            assert_true(callable(callback))

        if reduction_format is None:
            reduction_format = node_to_reduce.output_format
        else:
            assert_is_instance(reduction_format, DenseFormat)

        if output_format is None:
            output_format = reduction_format
        else:
            assert_is_instance(reduction_format, DenseFormat)

        #
        # Done sanity-checking args
        #

        self.node_to_reduce = node_to_reduce
        self._callbacks = callbacks
        self.reduction_format = reduction_format
        self.output_format = output_format
        self._reduction = None

        super(ReduceOverEpoch, self).__init__([node_to_reduce])

    def on_start_training(self):
        assert_is(self._reduction,
                  None,
                  "self._reductions should've been set to None in the "
                  "constructor. Something weird is going on.")

        # self._reductions = [None] * len(self._monitored_nodes)

    def _reduce(self, batch_and_reduction, axis, keepdims):
        '''
        Reduces a concatenated batch and reduction, along the batch axis.

        You can usually just replace this method with numpy.max, numpy.sum,
        etc, rather than overriding it.

        Parameters:
        -----------
        batch_and_reduction: numpy.ndarray
          A batch of outputs from self.node_to_reduce, with self._reduction
          appended to the end.

        axis: int
          The axis index of the batch axis.

        keepdims: bool
          Don't collapse the batch axis. Must be True.

        Returns:
        --------
        rval: numpy.ndarray
          The new value to replace old_reduction with. Must be the same
          shape and dtype as old_reduction.
        '''
        raise NotImplementedError("%s._update_reduction() not yet implemented."
                                  % type(self))

    def _on_iteration(self, batches_to_reduce):
        assert_equal(len(batches_to_reduce), 1)
        batch_to_reduce = batches_to_reduce[0]
        self.node_to_reduce.output_format.check(batch_to_reduce)

        batch_axis = self.node_to_reduce.output_format.axes.index('b')

        if self._reduction is not None:
            batch_to_reduce = numpy.concatenate([batch_to_reduce,
                                                 self._reduction],
                                                axis=batch_axis)

        self._reduction = self._reduce(batch_to_reduce,
                                       axis=batch_axis,
                                       keepdims=True)

        # self._reduction = self._update_reduction(batch_to_reduce,
        #                                          self._reduction)

        self.reduction_format.check(self._reduction)

    def on_epoch(self):
        output = self._on_epoch()
        self._reduction = None

        self.output_format.check(output)
        for callback in self._callbacks:
            callback(output, self.output_format)

    def _on_epoch(self):
        '''
        Returns the output value to pass to self._callbacks.

        Override if this value isn't just self._reduction.
        '''
        return self._reduction
        # raise NotImplementedError("{}._on_epoch() not "
        #                           "implemented.".format(type(self)))

        # result = self._reduction
        # self._reduction = None

        # for callback in self._callbacks:
        #     callback(result, self.reduction_format)


class MaxOverEpoch(ReduceOverEpoch):
    '''
    Computes the elementwise maximum of monitored values, over the batch axis.
    '''

    def __init__(self, node, callbacks):
        super(MaxOverEpoch, self).__init__(node, callbacks)
        self._reduce = numpy.max

    # def _update_reduction(self, batch, reduction):
    #     batch_axis = self.nodes_to_compute[0].axes.index('b')

    #     if reduction is None:
    #         stack = batch
    #     else:
    #         stack = numpy.concatenate((batch, reduction), axis=batch_axis)

    #     return numpy.max(stack, axis=batch_axis, keepdims=True)


class MinOverEpoch(ReduceOverEpoch):
    '''
    Computes the elementwise minimum of monitored values, over the batch axis.
    '''

    def __init__(self, node, callbacks):
        super(MinOverEpoch, self).__init__(node, callbacks)
        self._reduce = numpy.min

    # def _update_reduction(self, batch, reduction):
    #     batch_axis = self.nodes_to_compute[0].axes.index('b')

    #     if reduction is None:
    #         stack = batch
    #     else:
    #         stack = numpy.concatenate((batch, reduction), axis=batch_axis)

    #     return numpy.min(stack, axis=batch_axis, keepdims=True)


class SumOverEpoch(ReduceOverEpoch):
    '''
    Computes the elementwise sum of monitored values over the batch axis.

    Integer dtypes smaller than int64 will be cast to int64 before summing,
    to avoid overflow.
    '''

    def __init__(self, node, callbacks, output_format=None):
        # numpy.sum automatically upcasts the accumulator to 'int', if the
        # thing being summed is of a lesser precision integral type. This is
        # exactly what we want, but we need to adjust the reduction_format's
        # dtype accordingly.
        batch_dtype = node.output_symbol.dtype

        if numpy.issubdtype(batch_dtype, numpy.integer) and \
           batch.dtype.itemsize < numpy.dtype('int').itemsize:

            reduction_format = copy.deepcopy(node.output_format)
            reduction_format.dtype = numpy.dtype('int')
        else:
            reduction_format = None

        super(SumOverEpoch, self).__init__(node,
                                           callbacks,
                                           reduction_format=reduction_format,
                                           output_format=output_format)

        self._reduce = numpy.sum


        # self._count = None


    # def _update_reduction(self, batch, fmt, old_reduction):
    #     def is_small_int(dtype):
    #         dtype = numpy.dtype(dtype)
    #         return (numpy.issubdtype(dtype, numpy.integer) and
    #                 dtype.itemsize < numpy.dtype('int').itemsize)

    #     # Upcast small int dtypes to int, to avoid overflow.
    #     if is_small_int(batch.dtype):
    #         batch = numpy.cast['int'](batch)

    #         if old_reduction is not None:
    #             assert_equal(old_reduction.dtype, numpy.dtype('int'))

    #     batch_axis = fmt.axes.index('b')

    #     if old_reduction is None:
    #         stack = batch
    #     else:
    #         stack = numpy.concatenate((batch, old_reduction), axis=batch_axis)

    #     return numpy.sum(stack, axis=batch_axis, keepdims=True)


class MeanOverEpoch(SumOverEpoch):
    '''
    Computes the elementwise mean of monitored values over the batch axis.
    '''

    def __init__(self, node, callbacks):
        output_format = None

        # If we're summing integers, we have to specify that the output
        # (their averages) will be floats, not integers.
        if numpy.issubdtype(node.output_symbol.dtype, numpy.integer):
            output_format = copy.deepcopy(node.output_format)
            output_format.dtype = numpy.dtype('float')

        super(MeanOverEpoch, self).__init__(node,
                                            callbacks,
                                            output_format=output_format)
        self._count = 0

    def _on_iteration(self, batches_to_reduce):
        # Update self._reduction
        super(MeanOverEpoch, self)._on_iteration(batches_to_reduce)

        # Update self._count
        batch_axis = self.node_to_reduce.output_format.axes.index('b')
        assert_equal(len(batches_to_reduce), 1)
        self._count += batches_to_reduce[0].shape[batch_axis]

        # batch_sizes = []
        # for (monitored_value_batch,
        #      monitored_node) in safe_izip(monitored_value_batches,
        #                                   self._monitored_nodes):
        #     batch_axis = monitored_node.output_format.axes.index('b')
        #     batch_sizes.append(monitored_value_batch.shape[batch_axis])

        # batch_sizes = numpy.asarray(batch_sizes)

        # assert_true(numpy.all(batch_sizes[0] == batch_sizes[1:]),
        #             "Unequal batch sizes: %s" % str(batch_sizes))
        # self._count += batch_sizes[0]

    def _on_epoch(self):
        total = super(MeanOverEpoch, self)._on_epoch()
        mean = total / float(self._count)
        self._count = 0

        return mean

class DoesSomethingAtMinimum(object):
    '''
    Does something when some scalar hits a minimum during training.

    Provide this as one of the callbacks to a ReduceOverEpoch that
    computes a scalar value.
    '''

    def __init__(self):
        self._min_value = None

    def __call__(self, value, fmt):
        assert_is_instance(value, numpy.ndarray)
        assert_equal(value.shape, (1, ))
        value = value[0]

        assert_is_instance(fmt, DenseFormat)
        assert_equal(fmt.axes, ('b',))

        # old_min_value = self._min_value

        if self._min_value is None or value < self._min_value:
            self._min_value = value

            self._on_minimum()

    def _on_minimum(self):
        raise NotImplementedError("{}._on_minimum() not implemented.".format(
            type(self)))


class SavesAtMinimum(DoesSomethingAtMinimum):
    '''
    Saves an object to a file when some scalar hits a new low during training.

    Overwrites the file at each new low.

    Useful for saving models when the mean validation loss / misclassification
    rate hits a new low.

    Provide this as one of the callbacks to a ReduceOverEpoch, like
    MeanOverEpoch.
    '''

    def __init__(self, object_to_save, output_filepath):
        '''
        Parameters
        ----------
        object_to_save: A picklable object

        output_filepath: string
          The file path to save object_to_save to.
        '''
        super(SavesAtMinimum, self).__init__()

        assert_true(os.path.isdir(os.path.dirname(output_filepath)))

        self._object_to_save = object_to_save
        self._output_filepath = output_filepath

    def _on_minimum(self):
        pickle_file = file(self._output_filepath, 'wb')
        cPickle.dump(self._object_to_save,
                     pickle_file,
                     protocol=cPickle.HIGHEST_PROTOCOL)

    # def __call__(self, value, fmt):
    #     assert_is_instance(value, numpy.ndarray)
    #     assert_equal(value.shape, (1, ))
    #     value = value[0]

    #     assert_is_instance(fmt, DenseFormat)
    #     assert_equal(fmt.axes, ('b',))
    #     # assert_equal(len(values), 1)
    #     # assert_equal(len(values), len(formats))

    #     # fmt = formats[0]
    #     # assert_equal(fmt.axes, ('b', ))

    #     # assert_equal(values[0].shape, (1, ))
    #     # value = values[0][0]

    #     old_min_value = self._min_value

    #     if self._min_value is None or value < self._min_value:
    #         self._min_value = value

    #     if old_min_value != self._min_value:
    #         pickle_file = file(self._output_filepath, 'wb')
    #         cPickle.dump(self._object_to_save,
    #                      pickle_file,
    #                      protocol=cPickle.HIGHEST_PROTOCOL)


class StopsOnStagnation(DoesSomethingAtMinimum):
    '''
    Stops training when some scalar stops decreasing during training.

    Useful for halting training when the mean validation loss /
    misclassification rate stagnates or starts to rise (i.e. when
    training starts to overfit).

    Provide this as one of the callbacks to a ReduceOverEpoch, like
    MeanOverEpoch.
    '''

    def __init__(self, max_epochs, min_proportional_decrease=0.0):
        '''
        max_epochs: int
          Stop training if the monitored value doesn't decrease for
          this many epochs.

        min_proportional_decrease: float
          If this value is T, the monitored value is V, and the last known
          minimum of V is Vm, then V is considered a decrease only if
          V < (1.0 - T) * Vm
        '''
        super(StopsOnStagnation, self).__init__()

        assert_greater(max_epochs, 0)
        assert_true(numpy.issubdtype(type(max_epochs), numpy.integer))

        assert_greater_equal(min_proportional_decrease, 0.0)

        self._max_epochs_since_min = max_epochs
        self._min_proportional_decrease = min_proportional_decrease
        self._epochs_since_min = 0

        # This gets set to self._min_value at each siginificant decrese.
        # A "significant decrease" is a decrease in self._min_value
        # by more than min_proportional_decrease relative to
        # _significant_min_value.
        self._significant_min_value = None

    def _on_minimum(self):
        assert_is_not(self._min_value, None)

        if self._significant_min_value is None:
            self._significant_min_value = self._min_value
        else:
            threshold = ((1.0 - self._min_proportional_decrease) *
                         self._significant_min_value)

            if self._min_value < threshold:
                self._epochs_since_min = 0
                self._significant_min_value = self._min_value

    def __call__(self, value, fmt):
        self._epochs_since_min += 1
        assert_less_equal(self._epochs_since_min, self._max_epochs_since_min)

        # Calls self._on_minimum() if needed
        super(StopsOnStagnation, self).__call__(value, fmt)

        if self._epochs_since_min == self._max_epochs_since_min:
            message = ("{} stopping training. Value did not lower by "
                       "a fraction exceeding {} for {} epochs.".format(
                           self._min_proportional_decrease,
                           self._min_value,
                           self._epochs_since_min))

            raise StopTraining("ok", message)



        # fmt = formats[0]
        # assert_equal(fmt.axes, ('b', ))

        # assert_equal(values[0].shape, (1, ))
        # value = values[0][0]

        # if self._min_value is None or \
        #    value < (1.0 - self._min_proportional_decrease) * self._min_value:
        #     self._epochs_since_min = 0
        #     self._min_value = value
        # else:
        #     self._epochs_since_min += 1

        # if self._epochs_since_min >= self._max_epochs_since_min:
        #     message = ("{} stopping training. Value did not lower by "
        #                "a fraction exceeding {} for {} epochs.".format(
        #                    self._min_proportional_decrease,
        #                    self._min_value,
        #                    self._epochs_since_min))

        #     raise StopTraining("ok", message)

# TODO: replace with EpochLogger
class LogsToLists(object):
    '''
    A callback to Monitor that logs monitored values to lists.
    '''
    def __init__(self):
        self.log = None

    def __call__(self, value, fmt):
        # assert_equal(len(values), len(formats))
        # assert_greater(len(values), 0)

        if self.log is None:
            self.log = list()

        log.append(value)


class SgdParameterUpdater(ParameterUpdater):
    '''
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
    '''

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
        # backpropagate through to their inputs.)
        '''
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
        '''

        #
        # sanity-check args
        #

        assert_is_instance(parameter, theano.tensor.sharedvar.SharedVariable)
        assert_is_instance(gradient, theano.gof.Variable)
        assert_equal(parameter.broadcastable, gradient.broadcastable,
                     "If an Op's .grad() method is buggy, it can return "
                     "broadcast masks.")
        assert_is_subdtype(gradient.dtype, numpy.floating)
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
            '''
            Like str0 + str1, except returns None if either is None.
            '''
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

        updates = OrderedDict([(parameter, new_parameter),
                               (self._velocity, new_velocity)])

        super(SgdParameterUpdater, self).__init__(updates)


def limit_param_norms(parameter_updater, param, max_norm, input_axes):
    '''
    Modifies the update of an SgdParameterUpdater to limit param L2 norms.

    Parameter norms are computed by summing over the input_axes, provided.
    These are so named because you typically want to sum over the axes
    that get dotted with the input to the node (e.g. input_axes=[0] for Linear,
    input_axes=[1, 2, 3] for Conv2D).

    Parameters
    ----------

    parameter_updater: simplelearn.training.SgdParameterUpdater
      The parameter updater whose updates this will modify.

    param: theano shared variable

      The parameter being updated by parameter_updater.

      (No way to get this from SgdParameterUpdater at present; it updates the
      parameter and its velocity, and there's no way to safely distinguish them
      in parameter_updates.update_pairs)

    max_norm: floating-point scalar
      The maximum L2 norm to be permitted for the parameters.

    input_axes: Sequence
      A Sequence of ints. The indices to sum over when computing the
      L2 norm of the updated params.
    '''

    assert_is_instance(parameter_updater, SgdParameterUpdater)
    assert_in(param, parameter_updater.update_pairs)

    assert_floating(max_norm)
    assert_greater(max_norm, 0.0)

    assert_greater(len(input_axes), 0)
    assert_all_integer(input_axes)
    assert_all_greater_equal(input_axes, 0)
    assert_all_less(input_axes, param.ndim)

    input_axes = numpy.asarray(input_axes)
    updated_param = parameter_updater.update_pairs[param]

    norms = T.sqrt(T.sum(T.sqr(updated_param),
                         axis=input_axes,
                         keepdims=True))
    desired_norms = T.clip(norms, 0, max_norm)

    broadcast_mask = numpy.zeros(param.ndim, dtype=bool)
    broadcast_mask[input_axes] = True
    scales = T.patternbroadcast(desired_norms / (1e-7 + norms),
                                broadcast_mask)

    parameter_updater.update_pairs[param] = updated_param * scales


class Sgd(object):

    '''
    Uses stochastic gradient descent to optimize a cost w.r.t. parameters.

    The parameters and the inputs may be the same.

    At each iteration this computes the gradients of each parameter with
    respect to the cost function, then updates the parameter value using
    the gradients. How this update is performed (e.g. learning rate,
    momentum value & type, etc) is up to the SgdParameterUpdater
    objects passed into the constructor.
    '''

    def __init__(self,
                 inputs,
                 input_iterator,
                 callbacks,
                 theano_function_mode=None):

        '''
        Parameters
        ----------

        inputs: sequence of Nodes.
          Symbols for the outputs of the input_iterator.
          These should come from input_iterator.make_input_nodes()

        input_iterator: simplelearn.data.DataIterator
          Yields tuples of training set batches, such as (values, labels).

        callbacks: Sequence of EpochCallbacks
          This includes subclasses like IterationCallback &
          ParameterUpdater. One of these callbacks must throw a StopTraining
          exception for the training to halt.

        theano_function_mode: theano.compile.Mode
          Optional. The 'mode' argument to pass to theano.function().
          An example: pylearn2.devtools.nan_guard.NanGuard()
        '''

        #
        # sanity-checks the arguments.
        #

        assert_all_is_instance(inputs, Node)
        assert_is_instance(input_iterator, DataIterator)
        assert_true(input_iterator.next_is_new_epoch())

        for (input,
             iterator_input) in safe_izip(inputs,
                                          input_iterator.make_input_nodes()):
            assert_equal(input.output_format, iterator_input.output_format)

        # assert_is_instance(parameter_updaters, Sequence)
        # assert_all_is_instance(parameter_updaters, ParameterUpdater)
        # for parameter_updater in parameter_updaters:
        #     assert_is_instace(parameter_updater, IterationCallback)
        #     assert_equal(len(parameter_updater.nodes_to_compute), 0)
        # assert_all_is_instance(parameter_updaters, IterationCallback)

        # assert_is_instance(parameters, Sequence)
        # assert_is_instance(parameter_updaters, Sequence)
        # for parameter, updater in safe_izip(parameters, parameter_updaters):
        #     assert_is_instance(parameter,
        #                        theano.tensor.sharedvar.SharedVariable)

        #     assert_is_instance(updater, SgdParameterUpdater)

        #     assert_in(parameter, updater.updates)

        assert_equal(len(callbacks),
                     len(frozenset(callbacks)),
                     "There were duplicate callbacks.")

        assert_all_is_instance(callbacks, EpochCallback)

        #
        # Sets members
        #

        self._inputs = inputs
        self._input_iterator = input_iterator
        # self._parameters = tuple(parameters)
        # self._parameter_updaters = tuple(parameter_updaters)
        self._theano_function_mode = theano_function_mode
        self.epoch_callbacks = list(callbacks)
        self._train_called = False

    def _compile_update_function(self):
        input_symbols = [i.output_symbol for i in self._inputs]

        iteration_callbacks = [e for e in self.epoch_callbacks
                               if isinstance(e, IterationCallback)]

        output_symbols = []
        for iteration_callback in iteration_callbacks:
            for node_to_compute in iteration_callback.nodes_to_compute:
                output_symbols.append(node_to_compute.output_symbol)

        update_pairs = OrderedDict()

        for iteration_callback in iteration_callbacks:
            update_pairs.update(iteration_callback.update_pairs)

        return theano.function(input_symbols,
                               output_symbols,
                               updates=update_pairs,
                               mode=self._theano_function_mode)

    def train(self):
        '''
        Runs training until a StopTraining exception is raised.

        Training runs indefinitely until one of self.epoch_callbacks raises
        a StopTraining exception.
        '''

        if self._train_called:
            raise RuntimeError("train() has already been called on this %s. "
                               "Re-running train() risks inadvertently "
                               "carrying over implicit state from the "
                               "previous training run, such as the direction "
                               "of parameter updates (via the momentum "
                               "term), or the internal state of the Monitors "
                               "or EpochCallbacks. Instead, instantiate a new "
                               "copy of this %s and run train() on that." %
                               (type(self), type(self)))

        self._train_called = True

        if len(self.epoch_callbacks) == 0:
            raise RuntimeError("self.epoch_callbacks is empty, so Sgd will "
                               "iterate through the training data forever. "
                               "Please add an EpochCallback that will throw a "
                               "StopTraining exception at some point.")


        assert_all_is_instance(self.epoch_callbacks, EpochCallback)

        #
        # End sanity checks
        #

        update_function = self._compile_update_function()

        # Overlaps with self.epoch_callbacks
        iteration_callbacks = [c for c in self.epoch_callbacks
                               if isinstance(c, IterationCallback)]

        try:
            for epoch_callback in self.epoch_callbacks:
                epoch_callback.on_start_training()

            while True:

                # gets batch of data
                cost_arguments = self._input_iterator.next()

                # fprop-bprops, updates parameters, computes callback outputs.
                # pylint: disable=star-args
                all_callback_outputs = update_function(*cost_arguments)

                # calls iteration_callbacks' on_iteration() method, passing
                # in their output values, if any.
                output_index = 0
                for iteration_callback in iteration_callbacks:
                    num_outputs = len(iteration_callback.nodes_to_compute)
                    new_output_index = output_index + num_outputs

                    assert_less_equal(new_output_index,
                                      len(all_callback_outputs))

                    outputs = \
                        all_callback_outputs[output_index:new_output_index]

                    iteration_callback.on_iteration(outputs)

                    output_index = new_output_index

                assert_equal(output_index, len(all_callback_outputs))

                # if we've iterated through an epoch, call epoch_callbacks'
                # on_epoch() methods.
                if self._input_iterator.next_is_new_epoch():
                    for epoch_callback in self.epoch_callbacks:
                        epoch_callback.on_epoch()

        except StopTraining, exception:
            if exception.status == 'ok':
                print("Training halted normally with message: {}".format(
                    exception.message))
                return
            else:
                raise
