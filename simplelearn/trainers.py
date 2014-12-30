import theano.tensor as T


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


class GradientBasedParameterUpdater(object):
    """
    Support class for gradient-based trainers.

    The update_parameters() method updates a set of parameters, given
    their gradients.

    Subclasses must override its implementation, _update_parameters().
    """
    def __init__(self):
        pass

    def update_parameters(gradients, parameters):
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

    def _update_parameters(gradients, parameters):
        raise NotImplementedError("%s.update_parameters() not yet implemented."
                                  % type(self))


class SgdParameterUpdater(GradientBasedParameterUpdater):
    """
    Implements momentum-based gradient descent.

    The momentum and learning_rate are stored as numpy scalars, meaning you can
    modify them in-place, for example using a callback called at the end of
    each epoch.
    """

    def __init__(initial_learning_rate, initial_momentum):
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

    def _update_parameters(gradients, parameters):
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

    def __init__(value, saturated_value, epochs_to_saturation):
        if not isinstance(value, numpy.ndarray) or value.ndim != 0:
            raise TypeError("value must be a 0-dimensional numpy array.")

        if not numpy.issubdtype(value.dtype, numpy.floating):
            raise TypeError("value.dtype must be a floating-point dtype, not "
                            "%s." % value.dtype)

        if value < saturated_value:
            raise ValueError("The value (%g) is expected to be bigger than "
                             "its saturated_value (%g)." % (value,
                                                            saturated_value))

        self._initial_value = float(value)
        self._saturated_value = saturated_value
        self._epochs_to_saturation = epochs_to_saturation
        self._num_epochs_seen = -1

    def __call__(self):
        self._num_epochs_seen += 1
        assert self._num_epochs_seen >= 0

        alpha = min(1.0, (float(self._num_epochs_seen) /
                          float(self._epochs_to_saturation)))

        value[...] = (self._initial_value * (1.0 - alpha) +
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

    At each iteration this updates the parametrs Pi as:

      Pi -= dC/dPi * Li

    where C is the cost, and Li is Pi's learning rate (see constructor
    parameters below).

    Parameters
    ----------
    data_iterator: simplelearn.datasets.Iterator
      Provides the next training datum (list of arguments to cost) when polled.

    cost_symbol: theano.gof.Variable
      The cost to be reduced. Get as cost_node.get_output_symbol().

    parameter_symbols: sequence of shared Theano variables
      What this trainer modifies to lower the cost. These are typically model
      weights, though they could also be inputs (e.g. for optimizing input
      images).

    learning_rates: sequence of floats
      The learning rates of parameter_symbols.

    cost_input_symbols: sequence of theano.gof.Variables
      These are the inputs to cost.

    epoch_callbacks: sequence of callables.
      These get called once before the initial epoch, then after each epoch.
      They are called in the order in which they're listed.
      At least one of them must halt the training at some point by raising an
      StopTraining.
    """
    def __init__(self,
                 data_iterator,
                 cost_symbol,
                 parameter_symbols,
                 parameter_updaters,
                 cost_input_symbols,
                 epoch_callbacks):

        gradient_symbols = [T.grad(cost_symbol, p) for p in parameter_symbols]

        # Parameters to update
        self._parameter_symbols = tuple(parameter_symbols)
        for p in self.parameter_symbols:
            if not isinstance(p, theano.gof.Variable):
                raise TypeError("Expected all parameter_symbols to be "
                                "theano.gof.Variables, but found a %s." %
                                type(p))

        # a list of gradient functions, one for each parameter
        self._gradient_functions = tuple(T.function(cost_input_symbols, g)
                                         for g in gradient_symbols)
        self._learning_rates = tuple(learning_rates)
        self._momenta = tuple(momenta)

        num_params = tuple(len(x) for x in (self._parameter_symbols,
                                            self._learning_rates,
                                            self._momenta))
        if not (num_params[0] == num_params[1:]).all():
            raise ValueError("Expected parameter_symbols, learning_rates, and "
                             "momenta arguments to have the same length, not "
                             "%d, %d, and %d" % num_params)

        self._epoch_callbacks = tuple(epoch_callbacks)

    def train(self):
        """
        Runs training until a StopTraining exception is raised.

        Training runs indefinitely until one of self._epoch_callbacks() raises
        a StopTraining exception.
        """
        try:
            for callback in self._epoch_callbacks:
                callback()

            while True:
                cost_arguments = self._data_iterator.get_next_batch()
                gradients = [g(cost_arguments)
                             for g in self._gradient_functions]

                for callback in self._epoch_callbacks:
                    callback()

        except StopTraining, exception:
            if exception.status = 'ok':
                return
            else:
                raise
