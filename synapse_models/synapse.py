from abc import ABC as AbstractBaseClass, abstractmethod


class Synapse(AbstractBaseClass):
    
    def __init__(self, network, source, target, weight, delay):
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
        return self.weight
    
    def get_delay(self):
        return self.delay
    
    def get_source_id(self):
        return self.source_id
    
    def get_target_id(self):
        return self.target_id

    @abstractmethod
    def handle_presynaptic_spike(self):
        pass
    
    @abstractmethod
    def handle_postsynaptic_spike(self):
        pass