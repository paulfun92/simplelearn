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
from nose.tools import (assert_true,
                        assert_equal,
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

        assert_true(os.path.isdir(path))
        assert_equal(os.path.splitext(filename)[1], '.pkl')

        # if isinstance(objects, Sequence) and \
        #    not isinstance(objects, basestring):
        #     self.objects = objects
        # else:
        #     self.objects = [objects]

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
            basename, extension = os.path.splitext(filename)
            filename = '%s_%05d%s' % (filename,
                                      self._num_epochs_seen,
                                      extension)

            filepath = os.path.join(path, filename)

        pickle_file = file(filepath, 'wb')

        cPickle.dump(self._objects_to_pickle,
                     pickle_file,
                     protocol=cPickle.HIGHEST_PROTOCOL)

        # for obj in self.objects:
        #     try:
        #         cPickle.dump(obj,
        #                      pickle_file,
        #                      protocol=cPickle.HIGHEST_PROTOCOL)
        #     except cPickle.PicklingError, pe:
        #         print("error pickling %s" % str(obj))
        #         raise

        # for obj in self.objects:
        #     try:
        #         cPickle.dump(obj,
        #                      pickle_file,
        #                      protocol=cPickle.HIGHEST_PROTOCOL)
        #     except cPickle.PicklingError, pe:
        #         print("error pickling %s" % str(obj))
        #         raise


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
          Yields tuples of training set batches, such as (values, labels).

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

        outputs = []
        for monitor in monitors:
            outputs.extend(monitor.monitored_values)

        self._monitors = monitors

        self._update_function = theano.function(inputs, outputs)

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
            outputs = self._update_function(*input_batches)

            output_index = 0
            for monitor in self._monitors:
                new_output_index = (output_index +
                                    len(monitor.monitored_values))
                assert_less_equal(new_output_index, len(outputs))
                monitored_values = outputs[output_index:new_output_index]

                monitor.on_batch(input_batches, monitored_values)

                output_index = new_output_index

        # Calls monitors' on_epoch() methods.
        for monitor in self._monitors:
            monitor.on_epoch()


class LinearlyScalesOverEpochs(EpochCallback):
    '''
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
    '''

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
    '''
    On each epoch, this reports statistics about some monitored value Y.

    Examples: Y might be the output of layer 3 of a 6-layer net.

              MaxMonitor reports the elementwise maximum of Y encountered
              over the epoch.

              AverageMonitor reports Y, elementwise-averaged over the epoch.
    '''

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
        raise NotImplementedError("%s._on_epoch() not yet implemented" %
                                  type(self))


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

    def on_start_training(self):
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
        super(MaxMonitor, self).__init__(values_to_monitor, formats, callbacks)

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
        super(MinMonitor, self).__init__(values_to_monitor, formats, callbacks)

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
        if not isinstance(formats, Sequence):
            formats = [formats]
            assert not isinstance(values_to_monitor, Sequence)
            values_to_monitor = [values_to_monitor]

        # _reduce_batch() upgrades small int dtypes (e.g. uint8) to larger int
        # dtypes to avoid over/underflow when summing large numbers of them.
        # We need to make their corresponding formats agnostic to dtype, so
        # that they don't raise a stink about batch/tally dtypes being
        # different from the format's expected dtype.
        def remove_small_int_dtype(fmt):
            if fmt.dtype is not None and numpy.issubdtype(fmt.dtype,
                                                          numpy.integer):
                result = copy.deepcopy(fmt)
                result.dtype = None
                return result
            else:
                return fmt

        formats = [remove_small_int_dtype(fmt) for fmt in formats]

        super(SumMonitor, self).__init__(values_to_monitor,
                                         formats,
                                         callbacks)
        self._count = None

    def _reduce_batch(self, input_batch, batch_axis):

        # Lower risk of integer over/underflow (esp. if dtype is uint8)
        def upcast_if_integer(input_batch):
            if numpy.issubdtype(input_batch.dtype, numpy.integer):
                return numpy.cast['int64'](input_batch)
            else:
                return input_batch

        return numpy.sum(upcast_if_integer(input_batch),
                         axis=batch_axis,
                         keepdims=True)

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

        result = tuple(total / float(self._count) for total in totals)
        self._count = 0

        return result


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


class SavesAtMinimum(object):
    '''
    A callback to Monitor that pickles an object (typically the model)
    when some monitored scalar value hits an all-time low.
    '''

    def __init__(self, object_to_save, output_filepath):
        def check_path(path):
            abspath = os.path.abspath(path)

        check_path(output_filepath)


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

    def __init__(self, max_epochs, min_proportional_decrease):
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
        elif value < (1.0 - self._min_proportional_decrease) * self._min_value:
            self._epochs_since_min = 0
            self._min_value = value
        else:
            self._epochs_since_min += 1

        if self._epochs_since_min >= self._max_epochs_since_min:
            message = ("%s stopping training. Value did not lower %s"
                       "below last min value of %g for %d epochs." %
                       (type(self),
                        ("more than %g " % self._min_proportional_decrease
                         if self._min_proportional_decrease > 0.0
                         else ""),
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
                 monitors,
                 epoch_callbacks,
                 theano_function_mode=None):

        '''
        Parameters
        ----------

        inputs: sequence of theano.gof.Variables
          Symbols for the outputs of the input_iterator.

        input_iterator: simplelearn.data.DataIterator
          Yields tuples of training set batches, such as (values, labels).

        parameters: sequence of theano.tensor.sharedvar.SharedVariables
          What this trainer modifies to lower the cost. These are typically
          model weights, though they could also be inputs (e.g. for optimizing
          input images).

        parameter_updaters: sequence of SgdParameterUpdaters
          updaters for the corresponding elements in <parameters>.
          These are defined using the loss function to be minimized.

        monitors: (optional) sequence of Monitors.
          These are also used as epoch callbacks.

        epoch_callbacks: sequence of EpochCallbacks
          One of these must throw a StopTraining exception for the training to
          halt.

        theano_function_mode: theano.compile.Mode
          Optional. The 'mode' argument to pass to theano.function().
          An example: pylearn2.devtools.nan_guard.NanGuard()
        '''

        #
        # sanity-checks the arguments.
        #

        assert_is_instance(inputs, Sequence)
        for input_symbol in inputs:
            assert_is_instance(input_symbol, theano.gof.Variable)

        assert_is_instance(input_iterator, DataIterator)
        assert_true(input_iterator.next_is_new_epoch())

        assert_is_instance(parameters, Sequence)
        assert_is_instance(parameter_updaters, Sequence)
        for parameter, updater in safe_izip(parameters, parameter_updaters):
            assert_is_instance(parameter,
                               theano.tensor.sharedvar.SharedVariable)

            assert_is_instance(updater, SgdParameterUpdater)

            assert_in(parameter, updater.updates)

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
            '''
            Compiles the function that computes the monitored values.
            '''

            outputs = []
            for monitor in monitors:
                outputs.extend(monitor.monitored_values)

            updates = OrderedDict()
            for updater in parameter_updaters:
                assert_is_instance(updater.updates, OrderedDict)
                updates.update(updater.updates)

            return theano.function(inputs,
                                   outputs,
                                   updates=updates,
                                   mode=theano_function_mode)

        self._update_function = compile_update_function()

        repeated_callbacks = frozenset(monitors).intersection(epoch_callbacks)
        assert_equal(len(repeated_callbacks),
                     0,
                     "There were duplicate entries between monitors and "
                     "epoch_callbacks: %s" % str(repeated_callbacks))

        # These get called once before any training, and after each epoch
        # thereafter. One of them must halt the training at some point by
        # throwing a StopTraining exception.
        self.epoch_callbacks = tuple(epoch_callbacks)

        self._train_called = False

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

        if len(self.epoch_callbacks) + len(self._monitors) == 0:
            raise RuntimeError("self._monitors and self.epoch_callbacks are "
                               "both empty, so this will "
                               "iterate through the training data forever. "
                               "Please add an EpochCallback or "
                               "Monitor that will throw a "
                               "StopTraining exception at some point.")
        try:
            all_callbacks = self._monitors + tuple(self.epoch_callbacks)
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
