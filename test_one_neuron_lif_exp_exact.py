from network.network import Network
from neuron_models.leaky_integrate_and_fire import lif_neuron_matrix

net = Network(sim_params={"t_sim": 20.})

neuron = lif_neuron_matrix(net, {"I_e": 250.})

net.simulate()

V_m = neuron.V_m

for i in range(1, len(V_m)):
    print(i*0.1, V_m[i])

neuron.plot_results()