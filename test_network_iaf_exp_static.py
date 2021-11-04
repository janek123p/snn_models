from network.network import Network
from synapse_models.static_synapse import StaticSynapse 
from neuron_models.leaky_integrate_and_fire import lif_neuron_matrix

net = Network(sim_params={"t_sim": 100.})

input_neurons = []
for i in range(10):
    input_neurons.append(lif_neuron_matrix(net, {"I_e": 600. - (i % 5) * 100}))

output_neuron = lif_neuron_matrix(net)


for i in range(10):
    syn = StaticSynapse(net, input_neurons[i], output_neuron, 700., 2.5)

net.simulate()

V_m = output_neuron.V_m

for i in range(1, len(V_m)):
    print(i*0.1, V_m[i])

output_neuron.plot_results()
