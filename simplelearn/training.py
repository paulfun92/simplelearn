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
                        assert_equal,
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
from simplelearn.formats import Format
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
    Evaluates some Monitors over validation data in between training epochs.
    '''

    def __init__(self, inputs, input_iterator, monitors):
        '''
        Parameters
        ----------

        inputs: sequence of theano.gof.Variables
          Symbols for the outputs of the input_iterator.

        input_iterator: simplelearn.data.DataIterator
          Yields tuples of validation set batches, such as (values, labels).

        monitors: sequence of Monitors.
          These are also used as epoch callbacks.
        '''

        #
        # Checks inputs
        #

        assert_is_instance(inputs, Sequence)
        for input_symbol in inputs:
            assert_is_instance(input_symbol, theano.gof.Variable)

        assert_is_instance(input_iterator, DataIterator)
        assert_true(input_iterator.next_is_new_epoch())

        assert_is_instance(monitors, Sequence)
        assert_greater(len(monitors), 0)

        for monitor in monitors:
            assert_is_instance(monitor, Monitor)

        #
        # Sets members
        #

        self._input_iterator = input_iterator

        monitored_symbols = []
        for monitor in monitors:
            for node in monitor._monitored_nodes:
                monitored_symbols.append(node.output_symbol)

        self._monitors = monitors

        self._update_function = theano.function(inputs, monitored_symbols)

    def on_start_training(self):
        self.on_epoch()

    def on_epoch(self):
        '''
        Loops through an epoch of the validation dataset.
        '''

        # Calls monitors' on_start_training()
        for monitor in self._monitors:
            monitor.on_start_training()

        # Repeatedly calls monitors' on_batch()
        keep_going = True

        while keep_going:
            input_batches = self._input_iterator.next()
            keep_going = not self._input_iterator.next_is_new_epoch()

            # pylint: disable=star-args
            monitored_values = self._update_function(*input_batches)

            value_index = 0
            for monitor in self._monitors:
                new_value_index = value_index + len(monitor._monitored_nodes)
                assert_less_equal(new_value_index, len(monitored_values))

                values = monitored_values[value_index:new_value_index]
                monitor.on_batch(input_batches, values)

                value_index = new_value_index

        # Calls monitors' on_epoch() methods.
        for monitor in self._monitors:
            monitor.on_epoch()


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


class Monitor(EpochCallback):
    '''
    On each epoch, this reports statistics about some monitored value Y.

    Examples: Y might be the output of layer 3 of a 6-layer net.

              MaxMonitor reports the elementwise maximum of Y encountered
              over the epoch.

              AverageMonitor reports Y, elementwise-averaged over the epoch.
    '''

    def __init__(self, nodes_to_monitor, callbacks):
        '''
        Parameters
        ----------
        nodes_to_monitor: Sequence of Nodes.
          Nodes whose values you want to monitor. These must lie upstream
          in the computational graph from the data iterator's nodes.

          Must not be empty.

        callbacks: a __call__-able, or a Sequence of them.

          The call signature of these must be f(values, formats), where:

          values: Sequence of numpy.ndarrays
            The values computed by nodes_to_monitor.

          formats: Sequence of DenseFormats
            The corresponding formats of the above values.
        '''

        #
        # Checks args
        #

        if isinstance(nodes_to_monitor, Node):
            nodes_to_monitor = [nodes_to_monitor]
        else:
            nodes_to_monitor = tuple(nodes_to_monitor)
            assert_all_is_instance(nodes_to_monitor, Node)

        assert_equal(len(nodes_to_monitor), len(frozenset(nodes_to_monitor)),
                     "nodes_to_monitor contains repeated elements: %s" %
                     str(nodes_to_monitor))

        if not isinstance(callbacks, Sequence):
            callbacks = [callbacks]
        else:
            callbacks = tuple(callbacks)

        for callback in callbacks:
            assert_true(callable(callback))

        #
        # Sets members
        #

        self._monitored_nodes = nodes_to_monitor
        self._callbacks = callbacks

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
        assert_equal(len(monitored_value_batches),
                     len(self._monitored_nodes))

        for batch, node in safe_izip(monitored_value_batches,
                                     self._monitored_nodes):
            node.output_format.check(batch)

        self._on_batch(tuple(input_batches), tuple(monitored_value_batches))

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
            formats = [node.output_format for node in self._monitored_nodes]
            callback(values_to_report, formats)

    def _on_epoch(self):
        '''
        Returns a tuple of values to feed to self._callbacks as arguments.

        Returns
        -------
        rval: tuple of numpy.ndarrays
           Arguments to feed to self._callbacks' __call__(self, *args)
        '''
        raise NotImplementedError("%s._on_epoch() not yet implemented" %
                                  type(self))


class ReduceMonitor(Monitor):
    '''
    An abstract superclass of monitors like MaxMonitor, MinMonitor,
    that operate by applying a reduction operator (e.g. max, min)
    along the batch axis for each batch.

    Subclasses must override _update_reduction, which updates the current
    epoch's reduction with the newest batch.

    The dtypes of self._reductions may not be the same as the dtypes of the
    corresponding batches. For example, AverageMonitor will compute the
    average values of int batches as floats.
    '''

    def __init__(self, nodes_to_monitor, callbacks):
        for node_to_monitor in nodes_to_monitor:
            assert_in('b', node_to_monitor.output_format.axes)

        super(ReduceMonitor, self).__init__(nodes_to_monitor, callbacks)

        self._reductions = None

        self._expected_reduction_shapes = []  # for sanity-checking
        for node_to_monitor in nodes_to_monitor:
            fmt = node_to_monitor.output_format
            batch_axis = fmt.axes.index('b')
            expected_shape = list(fmt.shape)
            expected_shape[batch_axis] = 1
            self._expected_reduction_shapes.append(tuple(expected_shape))

    def on_start_training(self):
        # initialize with zero-sized batches
        self._reductions = None

    def _update_reduction(self, batch, fmt, old_reduction):
        '''
        Returns the value of <old_reduction>, updated by <batch>.

        Parameters:
        -----------
        batch: numpy.ndarray
          A batch of outputs from one of self._monitored_nodes.

        fmt: DenseFormat
          The format of batch.

        old_reduction: numpy.ndarray or None
          The old reduced value in self.reductions to update.
          This will be None if this is the first call in an epoch.

        Returns:
        --------
        rval: numpy.ndarray
          The new value to replace old_reduction with. Must be the same
          shape and dtype as old_reduction.
        '''
        raise NotImplementedError("%s._update_reduction() not yet implemented."
                                  % type(self))

    def _on_batch(self, input_batches, monitored_value_batches):
        batch_axes = [node.output_format.axes.index('b')
                      for node in self._monitored_nodes]

        old_reductions = ([None] * len(self._monitored_nodes)
                          if self._reductions is None
                          else self._reductions)
        new_reductions = []

        for batch, old_reduction, expected_reduction_shape \
            in safe_izip(monitored_value_batches,
                         old_reductions,
                         self._expected_reduction_shapes):
            new_reduction = self._update_reduction(batch,
                                                   node.output_format,
                                                   old_reduction)
            assert_equal(new_reduction.shape, expected_reduction_shape)

            new_reductions.append(new_reduction)

        self._reductions = new_reductions

    def _on_epoch(self):
        assert_is_not(self._reductions, None)

        result = self._reductions
        self._reductions = None

        return result


class MaxMonitor(ReduceMonitor):
    '''
    Computes the elementwise maximum of monitored values, over the batch axis.
    '''

    def __init__(self, nodes_to_monitor, callbacks):
        super(MaxMonitor, self).__init__(nodes_to_monitor, callbacks)

    def _update_reduction(self, batch, fmt, reduction):
        batch_axis = fmt.axes.index('b')
        stack = numpy.concatenate((batch, reduction), axis=batch_axis)
        return numpy.max(stack, axis=batch_axis, keepdims=True)


class MinMonitor(ReduceMonitor):
    '''
    Computes the elementwise minimum of monitored values, over the batch axis.
    '''

    def __init__(self, nodes_to_monitor, callbacks):
        super(MinMonitor, self).__init__(nodes_to_monitor, callbacks)

    def _update_reduction(self, batch, fmt, reduction):
        batch_axis = fmt.axes.index('b')

        if reduction is None:
            stack = batch
        else:
            stack = numpy.concatenate((batch, reduction), axis=batch_axis)

        return numpy.min(stack, axis=batch_axis, keepdims=True)


class SumMonitor(ReduceMonitor):
    '''
    Computes the elementwise sum of monitored values over the batch axis.

    Integer dtypes smaller than int64 will be cast to int64 before summing,
    to avoid overflow.
    '''

    def __init__(self, nodes_to_monitor, callbacks):
        if isinstance(nodes_to_monitor, Node):
            nodes_to_monitor = [nodes_to_monitor]

        super(SumMonitor, self).__init__(nodes_to_monitor, callbacks)
        self._count = None

    def _update_reduction(self, batch, fmt, old_reduction):
        def is_small_int(dtype):
            dtype = numpy.dtype(dtype)
            return (numpy.issubdtype(dtype, numpy.integer) and
                    dtype.itemsize < numpy.dtype('int').itemsize)

        # Upcast small int dtypes to int, to avoid overflow.
        if is_small_int(batch.dtype):
            batch = numpy.cast['int'](batch)

            if old_reduction is not None:
                assert_equal(old_reduction.dtype, numpy.dtype('int'))

        batch_axis = fmt.axes.index('b')

        if old_reduction is None:
            stack = batch
        else:
            stack = numpy.concatenate((batch, old_reduction), axis=batch_axis)

        return numpy.sum(stack, axis=batch_axis, keepdims=True)


class AverageMonitor(SumMonitor):
    '''
    Computes the elementwise average of monitored values over the batch axis.
    '''

    def __init__(self, nodes_to_monitor, callbacks):
        super(AverageMonitor, self).__init__(nodes_to_monitor, callbacks)
        self._count = 0

    def _on_batch(self, input_batches, monitored_value_batches):
        # Update self._reductions
        super(AverageMonitor, self)._on_batch(input_batches,
                                              monitored_value_batches)

        batch_sizes = []
        for (monitored_value_batch,
             monitored_node) in safe_izip(monitored_value_batches,
                                          self._monitored_nodes):
            batch_axis = monitored_node.output_format.axes.index('b')
            batch_sizes.append(monitored_value_batch.shape[batch_axis])

        batch_sizes = numpy.asarray(batch_sizes)

        assert_true(numpy.all(batch_sizes[0] == batch_sizes[1:]),
                    "Unequal batch sizes: %s" % str(batch_sizes))
        self._count += batch_sizes[0]

    def _on_epoch(self):
        totals = super(AverageMonitor, self)._on_epoch()
        assert_is_instance(totals, Sequence)

        result = tuple(total / float(self._count) for total in totals)
        self._count = 0

        return result


class SavesAtMinimum(object):
    '''
    A callback to Monitor that pickles an object (typically the model)
    when some monitored scalar value hits an all-time low.
    '''

    def __init__(self, object_to_save, output_filepath):
        '''
        Parameters
        ----------
        object_to_save: A picklable object

        output_filepath: string
          The file path to save object_to_save to.
        '''
        assert_true(os.path.isdir(os.path.dirname(output_filepath)))

        self._object_to_save = object_to_save
        self._output_filepath = output_filepath
        self._min_value = None


    def __call__(self, values, formats):
        assert_equal(len(values), 1)
        assert_equal(len(values), len(formats))

        fmt = formats[0]
        assert_equal(fmt.axes, ('b', ))

        assert_equal(values[0].shape, (1, ))
        value = values[0][0]

        old_min_value = self._min_value

        if self._min_value is None or value < self._min_value:
            self._min_value = value

        if old_min_value != self._min_value:
            pickle_file = file(self._output_filepath, 'wb')
            cPickle.dump(self._object_to_save,
                         pickle_file,
                         protocol=cPickle.HIGHEST_PROTOCOL)

class StopsOnStagnation(object):
    '''
    A callback to Monitor that stops training if the monitored value
    (e.g. average loss over the epoch) doesn't decrease for N epochs.
    '''

    def __init__(self, max_epochs, min_proportional_decrease=0.0):
        '''
        max_epochs: int
          Wait for max this many epochs for the monitored value to decrease.

        min_proportional_decrease: float
          If this value is T, the monitored value is V, and the last known
          minimum of V is Vm, then V is considered a decrease only if
          V < (1.0 - T) * Vm
        '''
        assert_greater(max_epochs, 0)
        assert_true(numpy.issubdtype(type(max_epochs), numpy.integer))

        assert_greater_equal(min_proportional_decrease, 0.0)

        self._max_epochs_since_min = max_epochs
        self._min_proportional_decrease = min_proportional_decrease
        self._epochs_since_min = None
        self._min_value = None

    def __call__(self, values, formats):
        assert_equal(len(values), 1)
        assert_equal(len(values), len(formats))

        fmt = formats[0]
        assert_equal(fmt.axes, ('b', ))

        assert_equal(values[0].shape, (1, ))
        value = values[0][0]

        if self._min_value is None or \
           value < (1.0 - self._min_proportional_decrease) * self._min_value:
            self._epochs_since_min = 0
            self._min_value = value
        else:
            self._epochs_since_min += 1

        if self._epochs_since_min >= self._max_epochs_since_min:
            message = ("{} stopping training. Value did not lower by "
                       "a fraction exceeding {} for {} epochs.".format(
                           self._min_proportional_decrease,
                           self._min_value,
                           self._epochs_since_min))

            raise StopTraining("ok", message)


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


class EpochLogger(Monitor):
    """
    Logs the outputs of some Monitors to an HDF5 file.

    Runs the same monitors on training and / or testing sets in between each
    training epoch.
    """

    def _log_batch(self, values, _):  # ignore formats
        assert_equal(len(values), len(formats))

        for name, value, in safe_izip(self.names, self.values):
            self.h5_file[name]log.append(value)

        self.h5_file.flush()

    def __init__(self,
                 nodes,
                 names,
                 file_path,
                 log_testing=True,
                 log_validation=True):
        '''
        Parameters
        ----------
        nodes: Sequence of Node

        names: Sequence of basestring

        file_path: basestring
          Path to save the .h5 file to.

        log_testing: bool
          Default: True. Evaluate monitors over the testing set if True.

        log_vaidation: bool
          Default: True. Evaluate monitors over the training set if True.
        '''

        assert_all_is_instance(monitors, Monitor)
        assert_all_is_instance(names, basestring)
        assert_equal(len(nodes), len(names))

        self._names = names
        self._nodes = nodes
        self.h5_file = h5py.File(file_path, mode='w+')

        logs = self.h5_file.create_group("logs")

        for name, node in safe_izip(names, nodes):
            batch_dim = node.output_format.axes.index('b')

            initial_shape = list(node.output_format.shape)
            initial_shape[batch_dim] = 0

            max_shape = list(initial_shape)
            max_shape[batch_dim] = None

            logs.create_dataset(name,
                                initial_shape,
                                max_shape=max_shape,
                                dtype=node.output_symbol.dtype)

        super(EpochLogger, self).__init__(nodes, callbacks=_log_batch)

    def _on_epoch(self):
        # process monitors on training data
        super(EpochLogger, self)._on_epoch(self)


class SgdParameterUpdater(object):
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
        # backpropagate through.)
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

        self.updates = OrderedDict([(parameter, new_parameter),
                                    (self._velocity, new_velocity)])


def limit_param_norms(parameter_updater, params, max_norm, input_axes):
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

    max_norm: floating-point scalar
      The maximum L2 norm to be permitted for the parameters.

    input_axes: Sequence
      A Sequence of ints. The indices to sum over when computing the
      L2 norm of the updated params.
    '''

    assert_is_instance(parameter_updater, SgdParameterUpdater)

    assert_in(params, parameter_updater.updates)

    assert_floating(max_norm)
    assert_greater(max_norm, 0.0)

    assert_greater(len(input_axes), 0)
    assert_all_integer(input_axes)
    assert_all_greater_equal(input_axes, 0)
    assert_all_less(input_axes, params.ndim)

    input_axes = numpy.asarray(input_axes)
    updated_params = parameter_updater.updates[params]

    norms = T.sqrt(T.sum(T.sqr(updated_params),
                         axis=input_axes,
                         keepdims=True))
    desired_norms = T.clip(norms, 0, max_norm)

    broadcast_mask = numpy.zeros(params.ndim, dtype=bool)
    broadcast_mask[input_axes] = True
    scales = T.patternbroadcast(desired_norms / (1e-7 + norms),
                                broadcast_mask)

    parameter_updater.updates[params] = updated_params * scales


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
                 parameters,
                 parameter_updaters,
                 epoch_callbacks,
                 theano_function_mode=None):

        '''
        Parameters
        ----------

        inputs: sequence of Nodes.
          Symbols for the outputs of the input_iterator.
          These should come from input_iterator.make_input_nodes()

        input_iterator: simplelearn.data.DataIterator
          Yields tuples of training set batches, such as (values, labels).

        parameters: sequence of theano.tensor.sharedvar.SharedVariables
          What this trainer modifies to lower the cost. These are typically
          model weights, though they could also be inputs (e.g. for optimizing
          input images).

        parameter_updaters: sequence of SgdParameterUpdaters
          updaters for the corresponding elements in <parameters>.
          These are defined using the loss function to be minimized.

        epoch_callbacks: sequence of EpochCallbacks
          One of these must throw a StopTraining exception for the training to
          halt. Monitors go here.

        theano_function_mode: theano.compile.Mode
          Optional. The 'mode' argument to pass to theano.function().
          An example: pylearn2.devtools.nan_guard.NanGuard()
        '''

        #
        # sanity-checks the arguments.
        #

        assert_is_instance(inputs, Sequence)
        for input in inputs:
            assert_is_instance(input, Node)

        assert_is_instance(input_iterator, DataIterator)
        assert_true(input_iterator.next_is_new_epoch())

        for (input,
             iterator_input) in safe_izip(inputs,
                                          input_iterator.make_input_nodes()):
            assert_equal(input.output_format, iterator_input.output_format)

        assert_is_instance(parameters, Sequence)
        assert_is_instance(parameter_updaters, Sequence)
        for parameter, updater in safe_izip(parameters, parameter_updaters):
            assert_is_instance(parameter,
                               theano.tensor.sharedvar.SharedVariable)

            assert_is_instance(updater, SgdParameterUpdater)

            assert_in(parameter, updater.updates)

        assert_equal(len(epoch_callbacks), len(frozenset(epoch_callbacks)))

        assert_is_instance(epoch_callbacks, Sequence)
        assert_all_is_instance(epoch_callbacks, EpochCallback)

        #
        # Sets members
        #

        self._inputs = inputs
        self._input_iterator = input_iterator
        self._parameters = tuple(parameters)
        self._parameter_updaters = tuple(parameter_updaters)
        self._theano_function_mode = theano_function_mode

        # These get called once before any training, and after each epoch
        # thereafter. One of them must halt the training at some point by
        # throwing a StopTraining exception.
        self.epoch_callbacks = list(epoch_callbacks)

        self._train_called = False

    def _compile_update_function(self):
        input_symbols = [i.output_symbol for i in self._inputs]

        monitored_symbols = []
        for epoch_callback in self.epoch_callbacks:
            if isinstance(epoch_callback, Monitor):
                for monitored_node in epoch_callback._monitored_nodes:
                    monitored_symbols.append(monitored_node.output_symbol)

        parameter_updates = OrderedDict()
        for parameter_updater in self._parameter_updaters:
            parameter_updates.update(parameter_updater.updates)

        return theano.function(input_symbols,
                               monitored_symbols,
                               updates=parameter_updates,
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

        monitors = [c for c in self.epoch_callbacks if isinstance(c, Monitor)]

        try:
            all_callbacks = tuple(self.epoch_callbacks)
            for callback in all_callbacks:
                callback.on_start_training()

            while True:

                # gets batch of data
                cost_arguments = self._input_iterator.next()

                # fprop-bprop, updates parameters
                # pylint: disable=star-args
                monitored_values = update_function(*cost_arguments)

                # updates monitors
                value_index = 0
                for monitor in monitors:
                    new_value_index = (value_index +
                                        len(monitor._monitored_nodes))
                    assert_less_equal(new_value_index, len(monitored_values))
                    values = monitored_values[value_index:new_value_index]

                    monitor.on_batch(cost_arguments, values)

                    value_index = new_value_index

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


    # def __getstate__(self):
    #     result = dict()
    #     result.update(self.__dict__)
    #     result['_update_function'] = "left unserialized"
    #     return result

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     assert_equal(self._update_function, "left unserialized")
    #     self._update_function = self._compile_update_function(
    #         **self._compile_update_function_args)
