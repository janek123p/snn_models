from synapse_models.synapse import Synapse


class StaticSynapse(Synapse):
    
    def __init__(self, network, source, target, weight, delay):
        super().__init__(network, source, target, weight, delay)
        
    def handle_presynaptic_spike(self):
        self.network.get_neuron_by_id(self.target_id).handle_incoming_spike(self.weight, self.delay)

    def handle_postsynaptic_spike(self):
        pass