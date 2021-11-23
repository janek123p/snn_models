from abc import ABC as AbstractBaseClass, abstractmethod


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