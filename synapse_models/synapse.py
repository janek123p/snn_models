from abc import ABC as AbstractBaseClass, abstractmethod
import matplotlib.pyplot as plt
import numpy as np


class Synapse(AbstractBaseClass):
    """ Abstract base class for synapses. """
    
    def __init__(self, network, source, target, weight, delay):
        """ Initialize commpon properties of synapses.
        network: Network instance the synapse belongs to
        source: Source neuron or source neuron id
        target: Target neuron or target neuron id
        weight: Inital weight of the synapse
        delay: Delay of the synapse
        """
        self.network = network
        
        if isinstance(source, int):
            self.source_id = source
        else:
            self.source_id = source.id
            
        if isinstance(target, int):
            self.target_id = target
        else:
            self.target_id = target.id
            
        self.weight = weight
        self.delay = delay
        self.delay_steps = int(round(self.delay / self.network.get_resolution()))

        self.weight_changes = [self.weight]
        self.weight_change_times = [0]
        
        self.network.register_synapse(self)
        
    def get_weight(self):
        """ Return weight of the synapse. """
        return self.weight
    
    def get_delay(self):
        """ Return the delay of the synapse. """
        return self.delay
    
    def get_source_id(self):
        """ Return id of source neuron. """
        return self.source_id
    
    def note_weight_change(self):
        """ Save new weight to be ablte ot plot weight history """
        self.weight_changes.append(self.weight)
        self.weight_change_times.append(self.network.get_timestep())

    def plot_weight_history(self, title = None):
        """ Plot weight history """
        if title is None:
            title = "Synaptic weight history"

        self.weight_change_times.append(int(round(self.network.get_simulation_duration() / self.network.get_resolution(), 0)))

        weight_lists = [[self.weight_changes[i]]*(self.weight_change_times[i+1]-self.weight_change_times[i]) for i in range(len(self.weight_changes))]
        weights = []
        for l in weight_lists:
            weights += l

        fig, ax = plt.subplots()
        fig.suptitle(title)
        ax.set_xlabel("t [ms]", fontsize=14)
        ax.plot(np.arange(0, self.network.get_simulation_duration(), self.network.get_resolution()), weights, linewidth=1)
        ax.set_ylabel("w", fontsize=14)

        plt.show()

    def get_target_id(self):
        """ Return id of target neuron. """
        return self.target_id

    @abstractmethod
    def handle_presynaptic_spike(self):
        """ Abstract method to handle action potential of the presynaptic neuron.
        Needs to be imlpemented by the subclasses. """
        pass
    
    @abstractmethod
    def handle_postsynaptic_spike(self):
        """ Abstract method to handle action potential of the postsynaptic neuron.
        Needs to be imlpemented by the subclasses. """
        pass