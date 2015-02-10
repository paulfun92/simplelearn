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
        self._epochs_seen = -1

    def on_epoch(self):
        self._epochs_seen += 1

        assert self._epochs_seen >= 0

        if self._epochs_seen >= self._max_num_epochs:
            raise StopTraining(status='ok',
                               message=('Reached max # of epochs %d.' %
                                        self._max_num_epochs))


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

        self._num_epochs_seen = -1

    def on_epoch(self):
        self._num_epochs_seen += 1

        assert self._num_epochs_seen >= 0

        # interpolation parameter
        alpha = min(1.0,
                    float(self._num_epochs_seen) / self._epochs_to_saturation)

        scale = (1.0 - alpha) + alpha * self._final_scale

        self.shared_value.set_value(scale * self._initial_value)


class TrainingMonitor(EpochCallback):
    """
    Monitors some function of an input batch during training.
    """

    def __init__(self, values_to_monitor, formats):
        '''
        Parameters
        ----------
        values_to_monitor: sequence
          A sequence of theano variables to monitor

        formats: sequence
          A sequence of the above values' Formats
        '''

        #
        # Checks args
        #

        assert_is_instance(values_to_monitor, Sequence)
        assert_is_instance(formats, Sequence)

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

    def on_batch(self, input_batches, monitored_value_batches):
        assert_equal(len(monitored_value_batches), len(self._formats))

        for batch, fmt in safe_izip(monitored_value_batches, self._formats):
            fmt.check(batch)

        self._on_batch(input_batches, monitored_value_batches)

    def _on_batch(self, input_batches, monitored_value_batches):
        raise NotImplementedError("%s._on_batch() not yet implemented." %
                                  type(self))


class AverageMonitor(TrainingMonitor):
    def __init__(self, values_to_monitor, formats):
        super(AverageMonitor, self).__init__(values_to_monitor, formats)
        assert_greater(values_to_monitor, 0)

        self._totals = None
        self._count = 0
        self.averages = None

    def _on_batch(self, input_batches, monitored_value_batches):

        if self._totals is None:
            self._totals = []
            for batch, fmt in safe_izip(monitored_value_batches,
                                        self._formats):
                batch_axis = fmt.axes.index('b')
                self._totals.append(numpy.sum(batch, axis=batch_axis))
        else:
            for total, batch, fmt in safe_izip(self._totals,
                                               monitored_value_batches,
                                               self._formats):
                batch_axis = fmt.axes.index('b')
                total += numpy.sum(batch, axis=batch_axis)

        def get_num_examples(batches, formats):
            '''
            Returns the # of examples in a batch of sub-batches.

            Checks that all sub-batches contain the same # of examples.
            '''

            result = None

            # Checks that all batches have the same number of examples
            for batch, fmt in safe_izip(batches, formats):
                batch_axis = fmt.axes.index('b')
                batch_size = batch.shape[batch_axis]

                if result is None:
                    result = batch_size
                else:
                    assert_equal(batch_size, result)

            assert_is_not(result, None)

            return result

        self._count += get_num_examples(monitored_value_batches, self._formats)

    def on_epoch(self):
        if self._count != 0:
            self.averages = tuple(total / self._count for total in self._totals)

        self._totals = None
        self._count = 0


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


class StopsOnStagnation(AverageMonitor):
    """
    Halts training if the average of f(x) over an epoch stops decreasing.
    """

    def __init__(self,
                 value_to_monitor,
                 value_format,
                 num_epochs,
                 min_decrease=0.0,
                 value_name=None):
        '''
        Parameters
        ----------
        value_to_monitor: theano expression
          Some f(x) (x is the data iterator output batch).

        value_format: Format
          Data format of value_to_monitor

        num_epochs: int
          maximum number of epochs to wait for average f to decrease.

        min_decrease: float
          minimum decrease in avg f needed for it to count as a decrease.

        value_name: str
          If value_to_monitor doesn't have a name, specify it here.
        '''

        check_is_subdtype(num_epochs, 'num_epochs', numpy.integer)
        assert_greater(num_epochs, 0)
        check_is_subdtype(min_decrease, 'min_decrease', numpy.floating)
        assert_greater_equal(min_decrease, 0.0)
        assert_not_equal(value_to_monitor.name is None,
                         value_name is None,
                         ("value_to_monitor.name was specified. No need to "
                          "provide value_name argument"
                          if value_to_monitor.name is not None
                          else ("when value_to_monitor has no name, you must "
                                "provide a <name> argument.")))

        super(StopsOnStagnation, self).__init__([value_to_monitor],
                                                [value_format])

        self._max_epochs_since_min = num_epochs
        self._epochs_since_min = 0
        self._min_decrease = min_decrease
        self._min_value = numpy.inf
        self._value_name = (value_to_monitor.name
                            if value_to_monitor.name is not None
                            else value_name)

    def on_epoch(self):
        # Computes average value over epoch
        super(StopsOnStagnation, self).on_epoch()

        # There were no batches in the previous epoch
        if self.averages is None:
            return
        else:
            assert_equal(len(self.averages), 1)
            average = self.averages[0]

        if self._min_value - average > self._min_decrease:
            self._min_value = average
            self._epochs_since_min = 0
        else:
            self._epochs_since_min += 1

        if self._epochs_since_min > self._max_epochs_since_min:
            raise StopTraining(status='ok',
                               message=("%s didn't decrease for %d epochs." %
                                        (self.monitored_values[0].name,
                                         self._epochs_since_min)))


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

        def make_shared_floatX(numeric_var, name):
            return theano.shared(numpy.asarray(numeric_var, dtype=floatX),
                                 name=name)

        self.learning_rate = make_shared_floatX(learning_rate,
                                                concat(parameter.name,
                                                       ' learning rate'))

        self.momentum = make_shared_floatX(momentum,
                                           concat(parameter.name, ' momentum'))

        self._velocity = make_shared_floatX(0.0 * parameter.get_value(),
                                            concat(parameter.name,
                                                   ' velocity'))

        new_velocity = (self.momentum * self._velocity -
                        self.learning_rate * gradient)

        assert_equal(str(new_velocity.dtype), str(floatX))
        new_velocity.name = concat('new ', self._velocity.name)

        step = (self.momentum * new_velocity - self.learning_rate * gradient
                if use_nesterov
                else new_velocity)

        new_parameter = parameter + step
        new_parameter.name = concat('new ', parameter.name)

        self.updates = OrderedDict([(parameter, parameter)])  # no problem with this
        #self.updates = OrderedDict([(parameter, new_parameter)])  # problem persists
        # self.updates = OrderedDict([(parameter, new_parameter),
        #                             (self._velocity, new_velocity)])


# class GradientBasedParameterUpdater(object):
#     """
#     Updates parameters using their gradients.

#     This is a support class for gradient-based optimizers such as Sgd.

#     Subclasses must override _update_parameters().
#     """

#     def update_parameters(self, gradients, parameters):
#         """
#         Updates parameters in-place, based on their gradients.

#         Parameters
#         ----------
#         gradients: numpy array
#           The gradients of the training cost with respect to parameters.

#         parameters: theano shared variable
#           The parameters to be updated in-place.
#         """

#         assert_equal(gradients.shape, parameters.shape)
#         assert_equal(gradients.dtype, parameters.dtype)

#         self._update_parameters(gradients, parameters)

#     def _update_parameters(self, gradients, parameters):
#         raise NotImplementedError("%s.update_parameters() not yet implemented."
#                                   % type(self))


# class SgdParameterUpdater(GradientBasedParameterUpdater):
#     """
#     Optimizes parameters Implements momentum-based gradient descent.

#     The momentum and learning_rate are stored as numpy scalars, meaning you can
#     modify them in-place, for example using a callback called at the end of
#     each epoch.
#     """

#     def __init__(self, initial_learning_rate, initial_momentum):
#         def check_arg(arg, name):
#             if not isinstance(arg, float):
#                 raise TypeError("Expected %s to be a float, not a %s." %
#                                 name, type(arg))

#             if arg < 0.0 or arg > 1.0:
#                 raise ValueError("Expected %s to be in the range [0.0, 1.0], "
#                                  "but got %g." % (name, arg))

#         check_arg(initial_learning_rate, 'initial_learning_rate')
#         check_arg(initial_momentum, 'initial_momentum')

#         floatX = numpy.dtype(theano.config.floatX)

#         self.learning_rate = numpy.asarray(initial_learning_rate, dtype=floatX)
#         self.momentum = numpy.asarray(initial_momentum, dtype=floatX)
#         self._previous_update = None

#     def _update_parameters(self, gradients, parameters):
#         """
#         Updates parameters in-place, based on their gradients.

#         Parameters
#         ----------
#         gradients: numpy array
#           The gradients of the training cost with respect to parameters.

#         parameters: theano shared variable
#           The parameters to be updated in-place.
#         """

#         new_update = gradients * (-self.learning_rate)

#         if self._previous_update is not None:
#             new_update = (new_update * (1.0 - self.momentum) +
#                           self._previous_update * self.momentum)

#         parameters += new_update
#         self._previous_update = new_update


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
                 cost,
                 inputs,
                 parameters,
                 parameter_updaters,
                 input_iterator,
                 monitors,
                 epoch_callbacks):

        """
        Parameters
        ----------

        cost: theano.gof.Variable
          The cost to be reduced. Get as cost_node.get_output_symbol().

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

        inputs: sequence of theano.gof.Variables
          These are the inputs to cost.

        monitors: (optional) sequence of TrainingMonitors.
          These are also used as epoch callbacks.

        epoch_callbacks: sequence of EpochCallbacks
          One of these must throw a StopTraining exception for the training to
          halt.
        """

        #
        # sanity-checks the arguments.
        #

        assert_is_instance(cost, theano.gof.Variable)

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
            assert_is_instance(monitor, TrainingMonitor)

        assert_is_instance(epoch_callbacks, Sequence)
        for epoch_callback in epoch_callbacks:
            assert_is_instance(epoch_callback, EpochCallback)
            if isinstance(epoch_callback, TrainingMonitor):
                warnings.warn("You've passed a TrainingMonitor subclass %s "
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
            outputs = [cost]
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

        if self._train_called:
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
                               "TrainingMonitor that will throw a "
                               "StopTraining exception at some point.")
        try:
            all_callbacks = self._monitors + self._epoch_callbacks
            for callback in all_callbacks:
                callback.on_epoch()

            while True:

                # gets batch of data
                cost_arguments = self._input_iterator.next()

                # fprop-bprop, updates parameters
                # pylint: disable=star-args
                outputs = self._update_function(*cost_arguments)

                # updates monitors
                output_index = 1
                for monitor in self._monitors:
                    new_output_index = (output_index +
                                        len(monitor.monitored_values))
                    assert_less_equal(new_output_index, len(outputs))
                    monitored_values = outputs[output_index:new_output_index]

                    monitor.on_batch(cost_arguments, monitored_values)

                    output_index = new_output_index

                # for monitor, monitored_value in safe_izip(self._monitors,
                #                                           outputs[1:]):
                #     monitor.on_batch(monitored_value)

                # calls epoch callbacks, if we've iterated through an epoch
                if self._input_iterator.next_is_new_epoch():
                    for callback in all_callbacks:
                        callback.on_epoch()

        except StopTraining, exception:
            if exception.status == 'ok':
                return
            else:
                raise
