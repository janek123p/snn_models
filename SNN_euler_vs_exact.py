from network.network import Network
from synapse_models.synapse import Synapse
from neuron_models.leaky_integrate_and_fire import lif_neuron_euler, lif_neuron_matrix

net = Network(sim_params = {"t_sim":100.})

input_neuron = lif_neuron_matrix(net, {"I_e":900.})
output_neuron = lif_neuron_matrix(net, {"I_e":300.})

input_neuron_euler = lif_neuron_euler(net, {"I_e":900.})
output_neuron_euler = lif_neuron_euler(net, {"I_e":300.})

syn = Synapse(net, input_neuron, output_neuron, 700., 1.5)
syn = Synapse(net, input_neuron_euler, output_neuron_euler, 700., 1.5)

net.simulate()

V_m = input_neuron.V_m
V_m_euler = input_neuron_euler.V_m

diff_sum = 0.
for i in range(0, len(V_m)):
    diff =  abs(V_m[i] - V_m_euler[i])
    print(i*0.1, diff )
    diff_sum += diff

print("RESULT: ", diff_sum / len(V_m))