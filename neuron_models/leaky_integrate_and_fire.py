from neuron_models.neuron import Neuron
import numpy as np


class lif_neuron_euler(Neuron):
    """ Implementation of approximation of an integrate and fire neuron 
    with exponentially shaped postsynaptic current with euler method """

    def __init__(self, network, params=None):
        """ Initialize lif_psc_exp_euler neuron. 
        network: Network instance the neuron belongs to
        params: Dictionary specifying the following paramters of the neuron: 
            -V_th (threshold voltage)[-55.0 mV]
            -V_reset (reset voltage)[-70 mV]
            -tau_m (time constant for leakage of the neuron)[10.0 ms]
            -C_m (membrane capacity)[250.0 pF]
            -I_e (external current)[0.0 pA]
            -V_init (inital membrane voltage)[-70.0 mV]
            -E_L (resting membrane potential)[-70.0 mV]
            -t_ref (absolute refractory period)[2.0 ms]
            -tau_in (time constant for decay of inhibitory postsynaptic current)[2.0 ms]
            -tau_ex (time constant for decay of excitatory postsynaptic current)[2.0 ms]
        """

        # call super constructor
        super().__init__(network, "lif_psc_exp_euler", params, default_params={'V_th': -55.0, 'V_reset': -70.0, 'tau_m': 10.0, 'C_m': 250.0, 'I_e': 0., 'V_init': -70.0,
                                                                               'E_L': -70.0, 't_ref': 2.0, 'tau_in': 2.0, 'tau_ex': 2.0})

        # set all necessarey parameters
        self.V_th = self.get_param("V_th")
        self.V_reset = self.get_param("V_reset")
        self.tau_m = self.get_param("tau_m")
        self.C_m = self.get_param("C_m")
        self.V_init = self.get_param("V_init")
        self.E_L = self.get_param("E_L")
        self.I_E = self.get_param("I_e")
        self.tau_in = self.get_param("tau_in")
        self.tau_ex = self.get_param("tau_ex")

        # add initial voltage to voltage trace
        self.V_m[0] = self.V_init

        # initialize synaptic current and spike buffer
        self.spike_current_in = np.zeros(len(self.V_m))
        self.spike_current_ex = np.zeros(len(self.V_m))
        self.I_syn_in = 0.
        self.I_syn_ex = 0.

    def handle_incoming_spike(self, weight, delay):
        """ Neuron handles incoming spike and adjusts postsynaptic current depending on the weight.
        weight: Current weight of the synapse the action potential comes from
        delay: Delay of the synapse
        """
        # calculate index in buffer
        index = 1 + self.network.get_timestep() + int(round(delay / self.dt, 0))
        # ignore spike if spike time is after simulation duration
        if index < len(self.spike_current_in):
            # save spike occurence in concerning buffer
            if weight > 0:
                self.spike_current_ex[index] += weight
            else:
                self.spike_current_in[index] += weight

    def update_step(self):
        """ Update the neuron for one timestep. """
        # get incoming spike currents
        spikes_ex = self.spike_current_ex[self.network.get_timestep()]
        spikes_in = self.spike_current_in[self.network.get_timestep()]

        # update synaptic currents correpsonding to the euler method
        self.I_syn_in += spikes_in + self.dt * (- self.I_syn_in / self.tau_in)
        self.I_syn_ex += spikes_ex + self.dt * (- self.I_syn_ex / self.tau_ex)

        # add current value to current trace
        cur_current = self.I_E + self.I_syn_in + self.I_syn_ex
        self.input_current[self.network.get_timestep()] = cur_current

        # check if neuron is refractory
        if self.refractory_steps > 0:
            # refractory ==> save reset voltage to V_m in order to
            # plot it later and decrease number of refractory timesteps
            self.V_m[self.network.get_timestep()] = self.V_reset
            self.refractory_steps -= 1
        else:
            # not refractory ==> evolve membrane voltage
            dv = self.dt * (self.input_current[self.network.get_timestep() - 1] /
                            self.C_m - (self.V_m[self.network.get_timestep() - 1] - self.E_L)/self.tau_m)
            # lot more exact: something between explicit and implicit euler method
            # dv = self.dt * (cur_current /
            #                self.C_m - (self.V_m[self.network.get_timestep() - 1] - self.E_L)/self.tau_m)
            self.V_m[self.network.get_timestep(
            )] = self.V_m[self.network.get_timestep()-1] + dv

        # check if membrane voltage has reached threshold
        if self.V_m[self.network.get_timestep()] > self.V_th:
            self.refractory_steps = int(round(self.t_ref / self.dt, 0))
            self.V_m[self.network.get_timestep()] = self.V_reset
            self.spike()


class lif_neuron_matrix(Neuron):
    """implementation of an integrate and fire neuron
    with exponentially shaped postsynaptic current"""

    def __init__(self, network,  params=None):
        """ Initialize lif_psc_exp_exact neuron. 
        network: Network instance the neuron belongs to
        params: Dictionary specifying the following paramters of the neuron: 
            -V_th (threshold voltage)[-55.0 mV]
            -V_reset (reset voltage)[-70 mV]
            -tau_m (time constant for leakage of the neuron)[10.0 ms]
            -C_m (membrane capacity)[250.0 pF]
            -I_e (external current)[0.0 pA]
            -V_init (inital membrane voltage)[-70.0 mV]
            -E_L (resting membrane potential)[-70.0 mV]
            -t_ref (absolute refractory period)[2.0 ms]
            -tau_in (time constant for decay of inhibitory postsynaptic current)[2.0 ms]
            -tau_ex (time constant for decay of excitatory postsynaptic current)[2.0 ms]
        """

        # call super constructor
        super().__init__(network, "lif_psc_exp_exact", params, default_params={'V_th': -55.0, 'V_reset': -70.0, 'tau_m': 10.0, 'C_m': 250.0, 'tau_ex': 2.0, 'I_e': 0.,
                                                                               'tau_in': 2.0, 'V_init': -70.0, 'E_L': -70.0, 't_ref': 2.0})

        # set all necessarey parameters
        self.V_th = self.get_param("V_th")
        self.V_reset = self.get_param("V_reset")
        self.tau_m = self.get_param("tau_m")
        self.tau_ex = self.get_param("tau_ex")
        self.tau_in = self.get_param("tau_in")
        self.V_init = self.get_param("V_init")
        self.E_L = self.get_param("E_L")
        self.I_e = self.get_param("I_e")
        self.C_m = self.get_param("C_m")

        # add initial voltage to voltage trace
        self.V_m[0] = self.V_init
        self.V_m_rel_to_E_L = self.V_init - self.E_L

        # initialize synaptic current and spike buffer
        self.I_syn_ex = 0.
        self.I_syn_in = 0.
        self.spike_current_in = np.zeros(len(self.V_m))
        self.spike_current_ex = np.zeros(len(self.V_m))

        # init values for matrix
        self.P_11_ex = np.exp(-self.dt/self.tau_ex)
        self.P_11_in = np.exp(-self.dt/self.tau_in)
        self.P_22 = np.exp(-self.dt / self.tau_m)
        self.P_20 = self.tau_m / self.C_m * (1. - self.P_22)
        self.P_21_ex = self.tau_m*self.tau_ex / \
            (self.C_m*(self.tau_ex-self.tau_m)) * (self.P_11_ex-self.P_22)
        self.P_21_in = self.tau_m*self.tau_in / \
            (self.C_m*(self.tau_in-self.tau_m)) * (self.P_11_in-self.P_22)

    def handle_incoming_spike(self, weight, delay):
        """ Neuron handles incoming spike and adjusts postsynaptic current depending on the weight.
        weight: Current weight of the synapse the action potential comes from
        delay: Delay of the synapse
        """
        index = 1 + self.network.get_timestep() + int(round(delay / self.dt, 0))
        # check if spike arrival is during simulation duration
        if index < len(self.spike_current_ex):
            if weight > 0:
                self.spike_current_ex[index] += weight
            else:
                self.spike_current_in[index] += weight

    def update_step(self):
        """ Update the neuron for one timestep. """
        # get incoming spike currents
        spikes_ex = self.spike_current_ex[self.network.get_timestep()]
        spikes_in = self.spike_current_in[self.network.get_timestep()]
        self.I_syn_ex += spikes_ex
        self.I_syn_in += spikes_in

        # check if neuron is refractory
        if self.refractory_steps == 0:
            # not refractory ==> evolve V_m
            self.V_m_rel_to_E_L = self.V_m_rel_to_E_L * self.P_22 + self.P_21_ex * \
                self.I_syn_ex + self.P_21_in * \
                self.I_syn_in + self.P_20 * self.I_e
        else:
            # refractory ==> set V_m to V_reset
            self.V_m_rel_to_E_L = self.V_reset - self.E_L
            self.refractory_steps -= 1

        # save membrane voltage to voltage history array in order to plot later
        self.V_m[self.network.get_timestep()] = self.V_m_rel_to_E_L + self.E_L

        if self.V_m[self.network.get_timestep()] >= self.V_th:
            # spike ==> start refractory period
            self.refractory_steps = int(round(self.t_ref/self.dt, 0))
            self.V_m[self.network.get_timestep()] = self.V_reset
            self.spike()

        # evolve synaptic currents
        self.I_syn_ex *= self.P_11_ex
        self.I_syn_in *= self.P_11_in

        # append current to current trace
        self.input_current[self.network.get_timestep(
        )] = self.I_syn_ex + self.I_syn_in + self.I_e
