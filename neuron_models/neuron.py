import numpy as np
import matplotlib.pyplot as plt
from abc import ABC as AbstractBaseClass, abstractmethod


class Neuron(AbstractBaseClass):
    """ Abstract base class for neuron models. """

    def get_param(self, key):
        """ Return parameter with given key either from params
        or from default_params if not specified in params. """
        if self.params is not None and key in self.params:
            return self.params[key]
        elif self.default_params is not None and key in self.default_params:
            return self.default_params[key]
        return None

    def __init__(self, network, model_name, params, default_params=None):
        """ Initialize common parameters of neurons.
        network: Network instance the neuron belongs to
        model_name: Ideally unique model name
        params: Parameters specified for the neuron model
        default_params: Parameters the neuron uses if no parameters are specified in params
        """
        self.params = params
        self.default_params = default_params
        self.t_ref = self.get_param("t_ref")

        self.network = network
        network.register_neuron(self)

        self.dt = self.network.get_resolution()
        self.t_sim = self.network.get_simulation_duration()

        self.refractory_steps = 0

        t_len = int(self.t_sim/self.dt)+1
        self.V_m = np.zeros(t_len)
        self.input_current = np.zeros(t_len)

        self.model_name = model_name

    def plot_results(self, plot_input=True, title=None):
        """ Plot membrane voltage ofhe neuron. Neuron must have been simulated already!
        plot_input: Boolean to specify whether the input current should be plotted in the same plot
        title: Title of the plot; If None: Title will be Simulation of <model_name> 
        """
        fig, ax = plt.subplots()
        if title is None:
            fig.suptitle("Simulation of "+self.model_name)
        else:
            fig.suptitle(title)

        ax.set_xlabel("t [ms]", fontsize=14)
        if plot_input:
            ax2 = ax.twinx()
            ax2.plot(np.arange(self.dt, self.t_sim + (self.dt/2.), self.dt),
                     self.input_current[1:], color="blue", linewidth=0.6, linestyle=":")
            ax2.set_ylabel("I [pA]", color="blue", fontsize=14)

        ax.plot(np.arange(0, self.t_sim+(self.dt/2.), self.dt),
                self.V_m, color="red", linewidth=1)
        ax.set_ylabel("V_m [mV]", color="red", fontsize=14)
        V_th = self.get_param("V_th")
        V_reset = self.get_param("V_reset")
        if V_th is not None and V_reset is not None:
            ax.set_ylim((V_reset-1, V_th+1))

        plt.show()

    def spike(self):
        """ Method to be called by subclasses in case of an action potential. """
        self.network.handle_spike(self)

    @abstractmethod
    def handle_incoming_spike(self, weight, delay):
        """ Abstract method to handle an incoming spike from a connected neuron. 
        Needs to be implemented by subclasses. """
        pass

    @abstractmethod
    def update_step(self):
        """ Abstract method to update the neuron for one time step.
        Needs to be implemented by subclasses. """
        pass
