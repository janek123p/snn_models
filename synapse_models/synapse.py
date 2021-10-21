class Synapse:
    
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
        
        self.network.register_synapse(self)
        
    def get_weight(self):
        return self.weight
    
    def get_delay(self):
        return self.delay
    
    def get_source_id(self):
        return self.source_id
    
    def get_target_id(self):
        return self.target_id
    
    def handle_spike(self):
        self.network.get_neuron_by_id(self.target_id).handle_incoming_spike(self.weight, self.delay)