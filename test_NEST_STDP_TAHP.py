import nest

nest.set_verbosity("M_WARNING")
nest.ResetKernel()

input_neuron_nest = nest.Create("iaf_psc_exp")
nest.SetStatus(input_neuron_nest, {"V_m": -70.})
nest.SetStatus(input_neuron_nest, {"I_e":400.})


input_neuron_nest2 = nest.Create("iaf_psc_exp")
nest.SetStatus(input_neuron_nest2, {"V_m": -70.})
nest.SetStatus(input_neuron_nest2, {"I_e":700.})


input_neuron_nest3 = nest.Create("iaf_psc_exp")
nest.SetStatus(input_neuron_nest3, {"V_m": -70.})
nest.SetStatus(input_neuron_nest3, {"I_e":600.})


input_neuron_nest4 = nest.Create("iaf_psc_exp")
nest.SetStatus(input_neuron_nest4, {"V_m": -70.})
nest.SetStatus(input_neuron_nest4, {"I_e":800.})
    
output_neuron = nest.Create("iaf_psc_exp")
nest.SetStatus(output_neuron, {"V_m": -70.})
nest.SetStatus(output_neuron, {"I_e": 350.0})

wr = nest.Create("weight_recorder")
nest.CopyModel("stdp_synapse", "stdp_synapse_wr", params = {"weight_recorder":wr[0]})

nest.Connect(input_neuron_nest, output_neuron, syn_spec = {"weight": 700., "delay": 1.5, "model":"stdp_synapse_wr", "Wmax":1400.})
nest.Connect(input_neuron_nest2, output_neuron, syn_spec = {"weight": 300., "delay": 2.5, "model":"stdp_synapse_wr", "Wmax":1400.})
nest.Connect(input_neuron_nest3, output_neuron, syn_spec = {"weight": 400., "delay": 2, "model":"stdp_synapse_wr", "Wmax":1400.})
nest.Connect(input_neuron_nest4, output_neuron, syn_spec = {"weight": 800., "delay": 0.5, "model":"stdp_synapse_wr", "Wmax":1400.})


nest.SetStatus(output_neuron, {"tau_minus": 20.0})


voltmeter = nest.Create("voltmeter")
nest.SetStatus(voltmeter, {"interval":0.1})
nest.Connect(voltmeter, output_neuron)

nest.Simulate(1001.0)

wr_events = nest.GetStatus(wr)[0]["events"]
weights = wr_events["weights"]
times = wr_events["times"]

for i in range(len(times)):
    print(times[i],weights[i])
for i in range(20):
    print(100* '#')

times = nest.GetStatus(voltmeter)[0]["events"]["times"]
V_m = nest.GetStatus(voltmeter)[0]["events"]["V_m"]
for i in range(0, 10000, 1):
    print(times[i],V_m[i])