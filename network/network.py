class Network:
    """Network class to manage all neurons and synapses of a spiking neural network."""

    def __init__(self, sim_params=None):
        """ Initialize network with specific network parameters
        sim_params: Dict with network specific parameters t_sim [1000 ms] and dt [0.1 ms].       
        """
        params = {"t_sim": 1000., "dt": 0.1}
        if params is not None:
            params.update(sim_params)
        
        self.dt = params["dt"]
        self.t_sim = params["t_sim"]

        self.neuron_dict = {}
        self.synapse_dict_by_sources = {}
        self.synapse_dict_by_targets = {}

        self.cur_time_step = 1
        
    def get_resolution(self):
        """ Returns simulation resolution of this network. """
        return self.dt
    
    def get_simulation_duration(self):
        """ Returns duration of simulation. """
        return self.t_sim

    def get_next_neuron_id(self):
        """ Returns next available neuron id. """
        return len(self.neuron_dict)

    def register_neuron(self, neuron):
        """ Register a neuron in the network and distribute an ID to it. """
        neuron.id = self.get_next_neuron_id()
        self.neuron_dict[neuron.id] = neuron

    def register_synapse(self, synapse):
        """ Register a synapse in the network """
        if synapse.get_source_id() not in self.synapse_dict_by_sources:
            self.synapse_dict_by_sources[synapse.get_source_id()] = []
        self.synapse_dict_by_sources[synapse.get_source_id()].append(synapse)
        if synapse.get_target_id() not in self.synapse_dict_by_targets:
            self.synapse_dict_by_targets[synapse.get_target_id()] = []
        self.synapse_dict_by_targets[synapse.get_target_id()].append(synapse)
    
    def handle_spike(self, neuron):
        """ Handles an action potential of the given neuron. The paramter neuron can be either a neuron object or a neuron id."""
        if isinstance(neuron, int):
            neuron = self.neuron_dict[neuron]
        if neuron.id in self.synapse_dict_by_sources:
            synapses = self.synapse_dict_by_sources[neuron.id]
            for syn in synapses:
                syn.handle_presynaptic_spike()
        if neuron.id in self.synapse_dict_by_targets:
            synapses = self.synapse_dict_by_targets[neuron.id]
            for syn in synapses:
                syn.handle_postsynaptic_spike()

    def get_timestep(self):
        """ Return current timestep of the simulation. """
        return self.cur_time_step

    def simulate(self):
        """ Start simulation of the network with all its neurons and synapses. """
        num_time_steps = int(round(self.t_sim/self.dt,0))
        
        for i in range(num_time_steps):
            for neuron in self.neuron_dict.values():
                neuron.update_step()
            self.cur_time_step += 1

    def get_neuron_by_id(self, id):
        """ Return neuron object corresponding to given neuron id. """
        return self.neuron_dict[id]