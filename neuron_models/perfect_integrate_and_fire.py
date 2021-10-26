from neuron_models.neuron import Neuron
import numpy as np


# implementation of a perfect integrate and fire neuron with
# exponentially shaped postsynaptical current
class pif_neuron(Neuron):

    def __init__(self, network, params=None):

        # call super constructor
        super().__init__(network, "pif_psc_exp", params, default_params={'V_th': -55.0, 'V_reset': -70.0, 'tau_m': 10.0, 'C_m': 250.0, 'I_e': 0., 'V_init': -70.0,
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
            # refractory ==> set membrane voltage to reset voltage and decrease number of refractory timesteps
            self.V_m[self.network.get_timestep()] = self.V_reset
            self.refractory_steps -= 1
        else:
            # not refractory ==> evolve membrane voltage
            self.V_m[self.network.get_timestep()] = self.V_m[self.network.get_timestep()-1] + self.dt * cur_current / self.C_m

        # check if membrane voltage has reached threshold
        if self.V_m[self.network.get_timestep()] > self.V_th:
            self.refractory_steps = int(round(self.t_ref / self.dt, 0))
            self.V_m[self.network.get_timestep()] = self.V_reset
            self.spike()
