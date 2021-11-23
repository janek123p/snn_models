from synapse_models.synapse import Synapse
import numpy as np


class PostsynapticSpikeData:
    """ Small wrapper class for postsynaptic spike
    containing the timestep and a boolean to indicate, whether
    this postsynaptic spike has already had an impact on the weight. """

    def __init__(self, ts):
        """ Initialize data 
        ts: Timestep of the postsynaptic spike
        """
        self.ts = ts
        self.potentiation_was_performed = False

    def is_potentiation_performed(self):
        """ Return if potentiation with this postsynaptic spike has already been performed. """
        return self.potentiation_was_performed

    def set_potentiation_performed(self):
        """ Mark potentation as performed """
        self.potentiation_was_performed = True

    def get_timestep(self):
        """ Get timestep of the postsynaptic spike. """
        return self.ts


class STDP_NN_SymmSnyapse(Synapse):
    """ Class for STDP synapse with nearest neighbour pairing scheme. """

    def __init__(self, network, source, target, init_weight, delay, params=None):
        """ Initialize stdp_nn_synapse 
        network: Network instance the synapse belongs to
        source: Source neuron or source neuron id
        target: Target neuron of target neuron id
        init_weight: Inital weight of the synapse
        delay: Delay of the synapse
        params: Dictionary specifying the following parameters:
            -lambda (scaling factor)[0.01]
            -tau_plus (time constant of the presynaptic trace)[20 ms]
            -tau_minus (time constant of the postsynaptic trace)[20 ms]
            -alpha (factor fo possible asymmetry in weight change)[1.0]
            -mu_plus (exponent for potentiation)[1.0]
            -mu_minu (expontent for depression)[1.0]
            -w_max (maximal allowed weight, may be negative)[1200.0]
        """

        super().__init__(network, source, target, init_weight, delay)

        std_params = {"lambda": 0.01, "tau_plus": 20., "tau_minus": 20., "alpha": 1.0,
                      "mu_plus": 1., "mu_minus": 1., "w_max": 1200.}

        self.last_presynaptic_spiketimestep = 0

        self.postsynaptic_spikedata = []

        if params is not None:
            std_params.update(params)
        params = std_params

        self.lambda_val = params["lambda"]
        self.tau_plus = params["tau_plus"]
        self.tau_minus = params["tau_minus"]
        self.alpha = params["alpha"]
        self.mu_plus = params["mu_plus"]
        self.mu_minus = params["mu_minus"]
        self.w_max = params["w_max"]

        self.eps_time = self.network.get_resolution() / 2.

    # OVERRIDE
    def handle_presynaptic_spike(self):
        """ Handling of the presynaptic spike. """  

        # get delay of synapse in steps, current timestep and
        delay_steps = self.delay_steps
        t_pre = self.network.get_timestep()
        t_pre_last = self.last_presynaptic_spiketimestep

        ### POTENTIATION ############################################
        # perform all postsynaptic weight potentiations
        for post_data in self.postsynaptic_spikedata:
            t_post = post_data.get_timestep()
            # check if spike is in range to have an impact on the weight
            if t_post > t_pre_last - delay_steps and t_post <= t_pre - delay_steps:
                minus_dt = (t_pre_last - t_post - delay_steps) * \
                    self.network.get_resolution()
                w_norm = self.weight/self.w_max + self.lambda_val * \
                    pow(1 - self.weight/self.w_max, self.mu_plus) * \
                    np.exp(minus_dt / self.tau_plus)

                # facilitate weight, clipping it to bounds if necessary
                self.weight = w_norm * self.w_max if w_norm < 1 else self.w_max
                post_data.set_potentiation_performed()

        ### DEPRESSION ##############################################
        # filter all items from self.postsynaptic_spiketimes that are no longer needed
        # get all spikes till (now - delay)
        # and grab the latest to perform weight change
        t_post = self.filter_unnecessary_postsyn_spikes_and_get_latest_in_range()
        if t_post >= 0:
            # calculate minus delta t which must be negative
            minus_dt = (t_post - t_pre + delay_steps) * \
                self.network.get_resolution()

            # depression
            w_norm = self.weight/self.w_max - self.lambda_val * self.alpha * \
                pow(self.weight/self.w_max, self.mu_minus) * \
                np.exp(minus_dt / self.tau_minus)

            # updating weight, clipping it to bounds if necessary
            self.weight = w_norm * self.w_max if w_norm > 0. else 0.

        ##############################################################

        # send signal to target neuron
        self.network.get_neuron_by_id(
            self.target_id).handle_incoming_spike(self.weight, self.delay)

        # update last spike
        self.last_presynaptic_spiketimestep = self.network.get_timestep()

    # OVERRIDE
    def handle_postsynaptic_spike(self):
        """ Handling of the postsynaptic spike. """  

        self.postsynaptic_spikedata.append(
            PostsynapticSpikeData(self.network.get_timestep()))

    def filter_unnecessary_postsyn_spikes_and_get_latest_in_range(self):
        """ Return latest postsynaptic spike that occured strictly before
        current_timstep - delay and filter all postsynaptic spikes from the list
        that are no longer needed. """
        
        # get current timestep
        t = self.network.get_timestep()
        # get all spikes before now - delay and grab last one
        filtered_spikes = [
            x.get_timestep() for x in self.postsynaptic_spikedata if x.get_timestep() < t - self.delay_steps]
        if len(filtered_spikes) > 0:
            last_postsyn_spike_before_delay = filtered_spikes[-1]
            # remove all spikes whose potentiation have been performed and that are no longer needed
            self.postsynaptic_spikedata = list(filter(lambda data: not (data.is_potentiation_performed() and
                data.get_timestep() < last_postsyn_spike_before_delay), self.postsynaptic_spikedata))
            return last_postsyn_spike_before_delay
        else:
            return -1
