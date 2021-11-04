from network.network import Network
from neuron_models.perfect_integrate_and_fire import pif_neuron

net = Network(sim_params={"t_sim": 20.})

neuron = pif_neuron(net, {"I_e": 250.})

net.simulate()

V_m = neuron.V_m

for i in range(1, len(V_m)):
    print(i*0.1, V_m[i])

neuron.plot_results()
