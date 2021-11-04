import nest
import matplotlib.pyplot as plt

nest.set_verbosity("M_WARNING")
nest.ResetKernel()

input_neurons = []
for i in range(10):
    input_neurons.append(nest.Create("iaf_psc_exp"))
    nest.SetStatus(input_neurons[i], {"V_m": -70.})
    nest.SetStatus(input_neurons[i], {"I_e":600. - i%5*100})
    
output_neuron = nest.Create("iaf_psc_exp")
nest.SetStatus(output_neuron, {"V_m": -70.})
nest.SetStatus(output_neuron, {"I_e":0.0})

for i in range(10):
    nest.Connect(input_neurons[i], output_neuron, syn_spec = {"weight": 700., "delay": 2.5})

voltmeter = nest.Create("voltmeter")
nest.SetStatus(voltmeter, {"interval":0.1})
nest.Connect(voltmeter, output_neuron)

nest.Simulate(101.0)

times = nest.GetStatus(voltmeter)[0]["events"]["times"]
V_m = nest.GetStatus(voltmeter)[0]["events"]["V_m"]
for i in range(len(times)):
    print(times[i],V_m[i])