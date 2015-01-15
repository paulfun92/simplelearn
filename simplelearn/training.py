"""
Training algorithms, and callbacks for monitoring their progress.
"""

__author__ = "Matthew Koichi Grimes"
__email__ = "mkg@alum.mit.edu"
__copyright__ = "Copyright 2014"
__license__ = "Apache 2.0"

import collections
import numpy
import theano
import theano.tensor as T
from nose.tools import assert_equal
from simplelearn.data import DataIterator
from simplelearn.nodes import Node

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

class ComputesAverageOverEpoch(object):
    """
    Epoch callback. Computes the average of a function over an epoch of data.

    On call, this loops over a data iterator, computing f(x) for each
    datum x, where f is given in the constructor. After an epoch's worth
    of data, this sums all the f(x)'s, divides by the number of samples,
    and passes the result to any interested callbacks.
    """
    def __init__(self, function_node, data_iterator, callbacks):
        """
        Parameters
        ----------

        function_node: Node
          A Node whose inputs are a DataSource's output nodes.

        data_iterator: DataIterator
          Iterates over the DataSource connected to function_node.

        callbacks: sequence
          A sequence of callables. Call signature must be f(x), where x
          is a numeric batch of outputs from function_node.
        """
        if not isinstance(function_node, Node):
            raise TypeError("Expected function_node to be a Node, but got a "
                            "%s." % type(function_node))

        if not data_iterator.next_is_new_epoch():
            raise ValueError("iterator doesn't point to the beginning of an "
                             "epoch.")

        if not isinstance(callbacks, collections.Sequence):
            raise TypeError("callbacks argument must be a sequence.")

        self._function_batch_axis = function_node.output_format.axes.index('b')

        input_symbols = tuple(input_node.output_symbol
                              for input_node in function_node.inputs)
        self._function = theano.function(input_symbols,
                                         function_node.output_symbol)
        self._iterator = data_iterator
        self._callbacks = callbacks

    def __call__(self):
        if not self._iterator.next_is_new_epoch():
            raise ValueError("self._iterator doesn't point to a fresh epoch.")

        count = 0
        total = None

        batch = self._function(*self._iterator.next())
        count += batch.shape[self._function_batch_axis]
        total = batch.sum(axis=self._function_batch_axis)

        while not self._iterator.next_is_new_epoch():
            batch = self._function(*self._iterator.next())
            count += batch.shape[self._function_batch_axis]
            total += batch.sum(axis=self._function_batch_axis)

        average = total / count

        for callback in self._callbacks:
            callback(average)


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
        super(StopTraining).__init__(message)


class StopsOnStagnation(object):
    """
    A callback to give ComputesAverageOverEpoch.

    Stops the training when the average f(x_i) over epoch x stops decreasing.
    """

    def __init__(self, name, num_epochs, min_decrease=0.0):
        """
        Parameters
        ----------

        name: str
          Name of the quantity being monitored.
        """

        #
        # Sanity-checks args.
        #

        if not numpy.issubdtype(type(num_epochs), numpy.integer):
            raise TypeError("num_epochs must be an integer, but got a %s."
                            % type(num_epochs))

        if num_epochs < 1:
            raise ValueError("num_epochs must be at least 1, but got %d." %
                             num_epochs)

        if not numpy.issubdtype(type(min_decrease), numpy.floating):
            raise TypeError("Expected a floating-point value for "
                            "min_decrease, but got a %s." % type(min_decrease))

        if min_decrease < 0.0:
            raise ValueError("Expected min_decrease to be non-negative, but "
                             "got %g." % min_decrease)

        #
        # Sets members
        #

        self._max_epochs_since_min = num_epochs
        self._epochs_since_min = 0
        self._min_decrease = min_decrease
        self._min_value = numpy.inf

    def __call__(self, average_over_epoch):
        if average_over_epoch < self._min_value:
            self._min_value = average_over_epoch
            self._epochs_since_min = 0
        else:
            self._epochs_since_min += 1

        if self._epochs_since_min >= self._max_epochs_since_min:
            raise StopTraining(status='ok',
                               message=("%s didn't decrease for %d epochs." %
                                        (self._name, self._epochs_since_min)))


class GradientBasedParameterUpdater(object):
    """
    Updates parameters using their gradients.

    This is a support class for gradient-based optimizers such as Sgd.

    Subclasses must override _update_parameters().
    """

    def update_parameters(self, gradients, parameters):
        """
        Updates parameters in-place, based on their gradients.

        Parameters
        ----------
        gradients: numpy array
          The gradients of the training cost with respect to parameters.

        parameters: theano shared variable
          The parameters to be updated in-place.
        """

        assert_equal(gradients.shape, parameters.shape)
        assert_equal(gradients.dtype, parameters.dtype)

        self._update_parameters(gradients, parameters)

    def _update_parameters(self, gradients, parameters):
        raise NotImplementedError("%s.update_parameters() not yet implemented."
                                  % type(self))


class SgdParameterUpdater(GradientBasedParameterUpdater):
    """
    Implements momentum-based gradient descent.

    The momentum and learning_rate are stored as numpy scalars, meaning you can
    modify them in-place, for example using a callback called at the end of
    each epoch.
    """

    def __init__(self, initial_learning_rate, initial_momentum):
        def check_arg(arg, name):
            if not isinstance(arg, float):
                raise TypeError("Expected %s to be a float, not a %s." %
                                name, type(arg))

            if arg < 0.0 or arg > 1.0:
                raise ValueError("Expected %s to be in the range [0.0, 1.0], "
                                 "but got %g." % (name, arg))

        check_arg(initial_learning_rate, 'initial_learning_rate')
        check_arg(initial_momentum, 'initial_momentum')

        floatX = numpy.dtype(theano.config.floatX)

        self.learning_rate = numpy.asarray(initial_learning_rate, dtype=floatX)
        self.momentum = numpy.asarray(initial_momentum, dtype=floatX)
        self._previous_update = None

    def _update_parameters(self, gradients, parameters):
        """
        Updates parameters in-place, based on their gradients.

        Parameters
        ----------
        gradients: numpy array
          The gradients of the training cost with respect to parameters.

        parameters: theano shared variable
          The parameters to be updated in-place.
        """

        new_update = gradients * (-self.learning_rate)

        if self._previous_update is not None:
            new_update = (new_update * (1.0 - self.momentum) +
                          self._previous_update * self.momentum)

        parameters += new_update
        self._previous_update = new_update


class LinearlyDecayingCallback(object):
    """
    Linearly decays a scalar down to some final value over N epochs.

    Parameters
    ----------

    value: numpy.ndarray scalar
      A 0-dimensional numpy array, with floating-point dtype. This value
      will be decayed in-place.

    saturated_value: float
      Final value of <value>.

    epochs_to_saturation: int
      <value> should decay to <saturated_value> after this many epochs.
    """

    def __init__(self, value, saturated_value, epochs_to_saturation):
        if not isinstance(value, numpy.ndarray) or value.ndim != 0:
            raise TypeError("value must be a 0-dimensional numpy array.")

        if not numpy.issubdtype(value.dtype, numpy.floating):
            raise TypeError("value.dtype must be a floating-point dtype, not "
                            "%s." % value.dtype)

        if value < saturated_value:
            raise ValueError("The value (%g) is expected to be bigger than "
                             "its saturated_value (%g)." % (value,
                                                            saturated_value))

        self.value = value
        self._initial_value = float(value)
        self._saturated_value = saturated_value
        self._epochs_to_saturation = epochs_to_saturation
        self._num_epochs_seen = -1

    def __call__(self):
        self._num_epochs_seen += 1
        assert self._num_epochs_seen >= 0

        alpha = min(1.0, (float(self._num_epochs_seen) /
                          float(self._epochs_to_saturation)))

        self.value[...] = (self._initial_value * (1.0 - alpha) +
                           self._saturated_value * alpha)


class LimitNumEpochsCallback(object):
    """
    Throws a StopTraining exception after a fixed number of epochs.
    """

    def __init__(self, max_num_epochs):
        if not numpy.issubdtype(max_num_epochs, numpy.integer):
            raise TypeError("Expected max_num_epochs to be an integer, not a "
                            "%s." % type(max_num_epochs))

        if max_num_epochs < 0:
            raise ValueError("max_num_epochs must be non-negative, got %d." %
                             max_num_epochs)

        self._max_num_epochs = max_num_epochs
        self._epochs_seen = -1

    def __call__(self):
        self._epochs_seen += 1
        if self._epochs_seen >= self.max_num_epochs:
            raise StopTraining(status='ok',
                               message=('Reached max # of epochs %d.' %
                                        self._max_num_epochs))


class Sgd(object):

    """
    A trainer that performs stochastic gradient descent.

    At each iteration this computes the gradients of each parameter with
    respect to the cost function, then updates the parameter value using
    the gradients. How this update is performed (e.g. learning rate,
    momentum value & type, etc) is up to the GradientBasedParameterUpdater
    objects passed into the constructor.
    """

    def __init__(self,
                 data_iterator,
                 cost_symbol,
                 parameter_symbols,
                 parameter_updaters,
                 cost_input_symbols):
        """
        Parameters
        ----------

        data_iterator: simplelearn.datasets.Iterator
          Provides the next training datum (list of arguments to cost) when
          polled.

        cost_symbol: theano.gof.Variable
          The cost to be reduced. Get as cost_node.get_output_symbol().

        parameter_symbols: sequence of shared Theano variables
          What this trainer modifies to lower the cost. These are typically
          model weights, though they could also be inputs (e.g. for optimizing
          input images).

        parameter_updaters: sequence of GradientBasedParameterUpdaters.
          One of these per symbol in parameter_symbols.

        cost_input_symbols: sequence of theano.gof.Variables
          These are the inputs to cost.
        """

        #
        # sanity-checks the arguments.
        #

        if not isinstance(data_iterator, DataIterator):
            raise TypeError("Expected data_iterator to be a DataIterator, but "
                            "got a %s." % type(data_iterator))

        if not isinstance(cost_symbol, theano.gof.Variable):
            raise TypeError("Expected cost_symbol to be a theano symbol, but "
                            "got a %s." % type(cost_symbol))

        for parameter_symbol in parameter_symbols:
            if not isinstance(parameter_symbol, theano.gof.Variable):
                raise TypeError("Expected parameter_symbols to be theano "
                                "symbols, but got a %s." %
                                type(parameter_symbol))

        assert_equal(len(parameter_symbols), len(parameter_updaters))

        for updater in parameter_updaters:
            if not all(isinstance(updater, GradientBasedParameterUpdater)):
                raise TypeError("Expected all elements of parameter_updaters "
                                "to be GradientBasedParameterUpdater "
                                "instances, but got a %s." % type(updater))

        for cost_input_symbol in parameter_symbols:
            if not isinstance(cost_input_symbol, theano.gof.Variable):
                raise TypeError("Expected cost_input_symbols to be theano "
                                "symbols, but got a %s." %
                                type(cost_input_symbol))

        self._data_iterator = data_iterator

        # Parameters to update
        self._parameter_symbols = tuple(parameter_symbols)

        # a list of gradient functions, one for each parameter
        gradient_symbols = [T.grad(cost_symbol, p) for p in parameter_symbols]
        self._gradient_functions = tuple(T.function(cost_input_symbols, g)
                                         for g in gradient_symbols)

        # a list of parameter updaters.
        self._parameter_updaters = parameter_updaters

        # These get called once before any training, and after each epoch
        # thereafter. One of them must halt the training at some point by
        # throwing a StopTraining exception.
        self.epoch_callbacks = []

    def train(self):
        """
        Runs training until a StopTraining exception is raised.

        Training runs indefinitely until one of self.epoch_callbacks raises
        a StopTraining exception.
        """

        if len(self.epoch_callbacks) == 0:
            raise RuntimeError("self.epoch_callbacks is empty, so this will "
                               "iterate through the training data forever. "
                               "Please add a callback that will throw a "
                               "StopTraining exception at some point.")
        try:
            for callback in self.epoch_callbacks:
                callback()

            while True:
                epoch_of_prev_batch = self._data_iterator.epoch()

                # gets batch of data
                cost_arguments = self._data_iterator.get_next_batch()

                epoch_of_curr_batch = self._data_iterator.epoch()

                assert epoch_of_curr_batch in (epoch_of_prev_batch,
                                               epoch_of_prev_batch + 1)

                # calls epoch callbacks, if we've iterated through an epoch
                if epoch_of_curr_batch > epoch_of_prev_batch:
                    for callback in self.epoch_callbacks:
                        callback()

                # computes gradients of cost w.r.t. parameters
                gradients = [g(cost_arguments)
                             for g in self._gradient_functions]

                # Updates parameters using their gradients.
                for (parameter,
                     gradient,
                     updater) in safe_izip(self._parameters,
                                           gradients,
                                           self._parameter_updaters):
                    updater.update_parameters(gradients=gradient,
                                              parameters=parameter)

        except StopTraining, exception:
            if exception.status == 'ok':
                return
            else:
                raise
