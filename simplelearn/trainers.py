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

class GradientBasedParameterUpdater(object):
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
        """
        Implementation of update_parameters.

        See docs of that method for details.
        """
        raise NotImplementedError("%s._update_parameter() not yet implemented."
                                  % type(self))


class MomentumBasedParameterUpdater(GradientBasedParameterUpdater):

    def __init__(initial_learning_rate, initial_momentum):
        self._learning_rate = copy(initial_learning_rate)
        self._momentum = copy(initial_momentum)
        self._previous_update = None

    def _update_parameters(gradients, parameters):
        new_update = gradients * (-self._learning_rate)
        if self._previous_update is not None:
            new_update = (new_update * (1.0 - self.momentum) +
                          self._previous_update * self.momentum)

        parameters += new_update
        self._previous_update = new_update


class LinearlyDecayingCallback(object):
    def __init__(value, saturated_value, epochs_to_saturation):
        if not isinstance(value, numpy.ndarray) or value.ndim != 0:
            raise TypeError("value must be a 0-dimensional numpy array.")

        if not numpy.issubdtype(value.dtype, numpy.floating):
            raise TypeError("value.dtype must be a floating-point dtype, not "
                            "%s." % value.dtype)

        self._initial_value = float(value)
        self._saturated_value = saturated_value
        self._epochs_to_saturation = epochs_to_saturation
        self._num_epochs_seen = -1

    def __call__(self):
        self._num_epochs_seen += 1
        alpha = (float(self._num_epochs_seen) /
                 float(self._epochs_to_saturation))
        alpha = max(1.0, alpha)

        value[...] = (self._initial_value * (1.0 - alpha) +
                      self._saturated_value * alpha)


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
      EndTrainingException.
    """
    def __init__(self,
                 data_iterator,
                 cost_symbol,
                 parameter_symbols,
                 learning_rates,
                 momenta,
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
        Runs training until an EndTrainingException is raised.

        Training runs indefinitely until one of self._epoch_callbacks() raises
        an EndTrainingException.
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

        except EndTrainingException, exception:
            if exception.status = 'ok':
                return
            else:
                raise
