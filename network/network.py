class Network:
    def __init__(self, sim_params=None):
        params = {"t_sim": 1000., "dt": 0.1}
        if params is not None:
            params.update(sim_params)
        
        self.dt = params["dt"]
        self.t_sim = params["t_sim"]

        self.neuron_dict = {}
        self.synapse_dict = {}

        self.cur_time_step = 1
        
    def get_resolution(self):
        return self.dt
    
    def get_simulation_duration(self):
        return self.t_sim

    def get_next_neuron_id(self):
        return len(self.neuron_dict)

    def register_neuron(self, neuron):
        neuron.id = self.get_next_neuron_id()
        self.neuron_dict[neuron.id] = neuron

    def register_synapse(self, synapse):
        if synapse.get_source_id() not in self.synapse_dict:
            self.synapse_dict[synapse.get_source_id()] = []
        self.synapse_dict[synapse.get_source_id()].append(synapse)
    
    def handle_spike(self, source_neuron):
        if isinstance(source_neuron, int):
            source_neuron = self.neuron_dict[source_neuron]
        if source_neuron.id in self.synapse_dict:
            synapses = self.synapse_dict[source_neuron.id]
            for syn in synapses:
                syn.handle_spike()

    def get_timestep(self):
        return self.cur_time_step

    def simulate(self):
        num_time_steps = int(round(self.t_sim/self.dt,0))
        
        for i in range(num_time_steps):
            for neuron in self.neuron_dict.values():
                neuron.update_step()
            self.cur_time_step += 1

    def get_neuron_by_id(self, id):
        return self.neuron_dict[id]