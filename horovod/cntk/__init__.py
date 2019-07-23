import cntk as C
from cntk.learners import Learner

from horovod.cntk.mpi_ops import allreduce, broadcast


# CTNK uses the term "learner" rather than optimizer. All optimizer functions return a "Learner" base class. 
# The interface is defined here https://www.cntk.ai/pythondocs/cntk.learners.html?highlight=learner
# using mxnet as a guide implementation as a guide. 
class DistributedOptimizer(cntk.learners.Learner):

    """Construct a new DistributedOptimizer, which uses another optimizer (learner for CNTK)
    under the hood for computing single-process gradient values and
    applying gradient updates after the gradient values have been averaged
    across all the Horovod ranks.
    """
    def __init__(self, learner):
        self._optimizer = learner

    # Parent class inheritance
    def learning_rate(self):
        return self._optimizer.learning_rate()

    def parameters(self):
        return self._optimizer.parameters

    def reset_learning_rate(self, learning_rate):
        # Resets the learning rate for this optimizer (does this need to be broadcasted?)
        self._optimizer.reset_learning_rate(learning_rate)

    def update(self, gradient_values, training_sample_count, is_sweep_end=False):
        """
        From docs: 
        Update the parameters associated with this learner.
        Parameters: 

            gradient_values (dict) – maps Parameter to a NumPy array containing the first order gradient values for the Parameter w.r.t. the training objective.
            training_sample_count (int) – number of samples in the minibatch
            is_sweep_end (bool) – a flag indicating whether it is at the end of a sweep of data

        Hvd additional implementation: 
            - must all reduce the gradients, then set its value on this class.
        """


    def do_allreduce(self, gradient_values):
        # all reduce the gradients




