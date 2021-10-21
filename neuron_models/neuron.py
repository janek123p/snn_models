import numpy as np
import matplotlib.pyplot as plt

eps = 1e-6


class Neuron:
    
    def get_param(self, key):
        if self.params is not None and key in self.params:
            return self.params[key]
        elif self.default_params is not None and key in self.default_params:
            return self.default_params[key]
        raise ValueError("For this neuron "+key+" is a necessary parameter!")
    
    def __init__(self, network, model_name, params, default_params = None):
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
            
    def plot_results(self, plot_input = True, title = None):
        fig,ax = plt.subplots()
        if title is None:
            fig.suptitle("Simulation of "+self.model_name)
        else:
            fig.suptitle(title)
        
        ax.set_xlabel("t [ms]",fontsize=14)
        if plot_input:
            ax2=ax.twinx()
            ax2.plot(np.arange(0, self.t_sim+eps, self.dt), self.input_current,color="blue", linewidth = 0.6, linestyle = ":")
            ax2.set_ylabel("I [pA]",color="blue",fontsize=14)
            
        ax.plot(np.arange(0, self.t_sim+eps, self.dt), self.V_m, color="red", linewidth = 1)
        ax.set_ylabel("V_m [mV]",color="red",fontsize=14)
            
        plt.show()
    
    def handle_incoming_spike(self, weight, delay):
        raise NotImplementedError("Baseclass of Neuron must implement handle_incoming_spike method!")
    
    def spike(self):
        self.network.handle_spike(self)
    
    def update_step(self):
        raise NotImplementedError("Baseclass of Neuron must implement update_step method!")