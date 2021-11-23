from synapse_models.synapse import Synapse


class StaticSynapse(Synapse):
    """ Implementiation of a static synapse. """
    
    def __init__(self, network, source, target, weight, delay):
        """ Initialize static synapse. """
        super().__init__(network, source, target, weight, delay)
        
    def handle_presynaptic_spike(self):
        """ Handling of the presynaptic spike. """
        self.network.get_neuron_by_id(self.target_id).handle_incoming_spike(self.weight, self.delay)

    def handle_postsynaptic_spike(self):
        """ Postsynaptic spike is ignored. """
        pass