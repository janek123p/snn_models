from network.network import Network
from neuron_models.leaky_integrate_and_fire import lif_neuron_matrix
from synapse_models.stdp_tahp_synapse import STDP_TAHP_Synapse

net = Network(sim_params={"t_sim": 1000.})

input_neuron = lif_neuron_matrix(net, {"I_e": 400.})
input_neuron2 = lif_neuron_matrix(net, {"I_e": 700.})
input_neuron3 = lif_neuron_matrix(net, {"I_e": 600.})
input_neuron4 = lif_neuron_matrix(net, {"I_e": 800.})

output_neuron = lif_neuron_matrix(net, {"I_e": 350.})

syn = STDP_TAHP_Synapse(net, input_neuron, output_neuron, init_weight = 700., delay = 1.5, params={"w_max":1400})
syn = STDP_TAHP_Synapse(net, input_neuron2, output_neuron, init_weight = 300., delay = 2.5, params={"w_max":1400})
syn = STDP_TAHP_Synapse(net, input_neuron3, output_neuron, init_weight = 400., delay = 2, params={"w_max":1400})
syn = STDP_TAHP_Synapse(net, input_neuron4, output_neuron, init_weight = 800., delay = 0.5, params={"w_max":1400})

net.simulate()

V_m = output_neuron.V_m

for i in range(0,10001,1):
  print(i*0.1, V_m[i])
# output_neuron.plot_results()
