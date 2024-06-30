""" 
Simulation of a single neuron using the NEST simulator.

Source: https://nest-simulator.readthedocs.io/en/stable/auto_examples/one_neuron_with_noise.html

author: Fabrizio Musacchio
date: May 30, 2024
"""
#%% IMPORTS
# additional packages need to be imported before importing NEST:
import matplotlib.pyplot as plt
import numpy as np
import nest
#import nest.voltage_trace
from pprint import pprint
# set the verbosity of the NEST simulator:
nest.set_verbosity("M_WARNING")
# set global font size for plots:
plt.rcParams.update({'font.size': 12})
# create a folder "figures" to save the plots (if it does not exist):
import os
if not os.path.exists('figures'):
    os.makedirs('figures')
#%% IAF PSC ALPHA NEURON MODEL
# reset the kernel for safety:
nest.ResetKernel()

# create the neuron, a spike recorder and a multimeter (all called "nodes"):
neuron     = nest.Create("iaf_psc_alpha")
multimeter = nest.Create("multimeter")
multimeter.set(record_from=["V_m"]) # record the membrane potential of the neuron (to which the multimeter will be connected)
spikerecorder = nest.Create("spike_recorder")
""" 
'iaf_psc_alpha' means: neuron is a leaky integrate-and-fire neuron with alpha-shaped postsynaptic currents.
The noise is used to simulate the background activity of the brain. It's in principle the current that
the neuron receives from other neurons. The voltmeter is used to record the membrane potential of the neuron.
"""
pprint(neuron.get())
pprint(f"I_e: {neuron.get('I_e')}")
pprint(f"V_reset: {neuron.get('V_reset')}")
pprint(f"{neuron.get(['V_m', 'V_th'])}")

neuron.set({"V_reset": -70.0})
pprint(f"{neuron.get('V_reset')}")

# set a constant input current for the neuron:
I_e = 376.0 # [pA]
neuron.I_e = I_e # [pA]
pprint(f"{neuron.get('I_e')}")

# list all available models:
pprint(nest.Models())

# list all recordable quantities
pprint(f"recordables of {neuron.model}: {nest.GetDefaults(neuron.model)['recordables']}")

# now, connect the nodes:
nest.Connect(multimeter, neuron)
nest.Connect(neuron, spikerecorder)
""" 
The order of the connections is important and reflects the flow of the events. If the neuron fires a spike,
it sends a signal to the spike recorder. The multimeter periodically sends requests to the neuron to get the
membrane potential at that point in time. This can be regarded as a perfect electrode stuck into the neuron.
"""

# run a simulation for 1000 ms:
nest.Simulate(1000.0)

# extract recorded data from the multimeter and plot it:
recorded_events = multimeter.get()
recorded_V = recorded_events["events"]["V_m"]
time = recorded_events["events"]["times"]
spikes = spikerecorder.get("events")
senders = spikes["senders"]

plt.figure(figsize=(8, 4))
plt.plot(time, recorded_V, label="membrane potential")
plt.plot(spikes["times"], spikes["senders"]+np.max(recorded_V), "r.", markersize=10,
         label="spike events")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title(f"Membrane potential of a {neuron.get('model')} neuron ($I_e$={I_e} pA)")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(f"figures/single_neuron_{neuron.get('model')}_Ie_{I_e}.png", dpi=300)
plt.show()
# %% IAF PSC ALPHA NEURON MODEL WITH NOISY INPUT CURRENT
# reset the kernel for safety:
nest.ResetKernel()

# create the neuron, a poisson (current) generator, a spike recorder and a multimeter (all four called "nodes"):
neuron     = nest.Create("iaf_psc_alpha")
noise      = nest.Create("poisson_generator")
multimeter = nest.Create("multimeter")
multimeter.set(record_from=["V_m", "I_syn_ex", "I_syn_in"])
spikerecorder = nest.Create("spike_recorder")

# ensure that the neuron's input current is zero:
neuron.I_e = 0.0
pprint(f"I_e: {neuron.get('I_e')}")

# change the membrane time constant:
nest.SetStatus(neuron, {"tau_m": 11}) # [ms], default is 10 ms
"""
When you increase the membrane time constant, the neuron will integrate the input current over a longer time
period. This means that the neuron will be more sensitive to the input current and will fire more easily.
"""

# change the spike threshold:
nest.SetStatus(neuron, {"V_th": -55.0}) # [mV], default is -55 mV
"""
When you decrease the spike threshold, the neuron will fire more easily, i.e., the neuron will
fire more often and at lower input currents.
"""

# set the parameters of injected Poisson-noisy current:
noise.rate = 68000.0 # [Hz]

# now, connect the nodes:
nest.Connect(multimeter, neuron)
nest.Connect(noise, neuron)
nest.Connect(neuron, spikerecorder)

# now we run a simulation for 1000 ms:
nest.Simulate(1000.0)

# extract recorded data from the multimeter and plot it:
recorded_events = multimeter.get()
recorded_V = recorded_events["events"]["V_m"]
time = recorded_events["events"]["times"]
spikes = spikerecorder.get("events")
senders = spikes["senders"]
recorded_current_ex = recorded_events["events"]["I_syn_ex"] # Excitatory synaptic input current
recorded_current_in = recorded_events["events"]["I_syn_in"] # Inhibitory synaptic input current

plt.figure(figsize=(8, 4))
plt.plot(time, recorded_V, label="membrane potential")
plt.plot(spikes["times"], spikes["senders"]+np.max(recorded_V), "r.", markersize=10,
         label="spike events")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title(f"Membrane potential of a {neuron.get('model')} neuron\nwith noisy (Poisson) input current")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(f"figures/single_neuron_{neuron.get('model')}_noisy_input.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(time, recorded_current_ex, label="excitatory synaptic input current")
plt.plot(spikes["times"], spikes["senders"]+np.max(recorded_current_ex), "r.", markersize=10,
         label="spike events")
plt.xlabel("Time (ms)")
plt.ylabel("Synaptic input current [pA]")
plt.title("Corresponding synaptic input current")
plt.ylim(300, 450)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(f"figures/single_neuron_{neuron.get('model')}_noisy_input_current.png", dpi=300)
plt.show()
#%% HODGKIN-HUXLEY NEURON MODEL
# reset the kernel for safety:
nest.ResetKernel()

# list all available neuron models:
pprint(nest.Models())

# create the neuron, a poisson (current) generator, a spike recorder and a multimeter (all four called "nodes"):
neuron     = nest.Create("hh_psc_alpha")
multimeter = nest.Create("multimeter")
multimeter.set(record_from=["V_m"]) # record the membrane potential of the neuron (to which the multimeter will be connected)
spikerecorder = nest.Create("spike_recorder")

# list all parameters of the model:
pprint(neuron.get())

# list all recordable quantities
pprint(f"recordables of {neuron.model}: {nest.GetDefaults(neuron.model)['recordables']}")

# set a constant input current for the neuron:
I_e = 650.0 # [pA] # 630.0: spike train; 620.0: a few spikes; 360.0: single spike
neuron.I_e = I_e # [pA]
pprint(f"{neuron.get('I_e')}")

# now, connect the nodes:
nest.Connect(multimeter, neuron)
nest.Connect(neuron, spikerecorder)

# now we run a simulation for 1000 ms:
nest.Simulate(200.0)

# extract recorded data from the multimeter and plot it:
recorded_events = multimeter.get()
recorded_V = recorded_events["events"]["V_m"]
time = recorded_events["events"]["times"]
spikes = spikerecorder.get("events")
senders = spikes["senders"]

plt.figure(figsize=(8, 4))
plt.plot(time, recorded_V, label="membrane potential")
plt.plot(spikes["times"], spikes["senders"]+np.max(recorded_V), "r.", markersize=10,
         label="spike events")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title(f"Membrane potential of a {neuron.get('model')} neuron ($I_e$={I_e} pA)")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.legend(loc="center right")
plt.tight_layout()
plt.savefig(f"figures/single_neuron_{neuron.get('model')}_Ie_{I_e}.png", dpi=300)
plt.show()
#%% IZHIKEVICH NEURON MODEL
# reset the kernel for safety:
nest.ResetKernel()

# list all available neuron models:
pprint(nest.Models())

# create the neuron, a poisson (current) generator, a spike recorder and a multimeter (all four called "nodes"):
neuron     = nest.Create("izhikevich")
multimeter = nest.Create("multimeter")
multimeter.set(record_from=["V_m"]) # record the membrane potential of the neuron (to which the multimeter will be connected)
spikerecorder = nest.Create("spike_recorder")

# list all parameters of the model:
pprint(neuron.get())

# list all recordable quantities
pprint(f"recordables of {neuron.model}: {nest.GetDefaults(neuron.model)['recordables']}")

# set a constant input current for the neuron:
I_e = 10.0 # [pA]
neuron.I_e = I_e # [pA]
pprint(f"{neuron.get('I_e')}")

# print the Izhi-parameters:
pprint(f"{neuron.get(['a', 'b', 'c', 'd'])}")

# define sets of typical parameters of the Izhikevich neuron model:
p_RS  = [0.02, 0.2, -65, 8, "regular spiking (RS)"] # regular spiking settings for excitatory neurons (RS)
p_IB  = [0.02, 0.2, -55, 4, "intrinsically bursting (IB)"] # intrinsically bursting (IB)
p_CH  = [0.02, 0.2, -51, 2, "chattering (CH)"] # chattering (CH)
p_FS  = [0.1, 0.2, -65, 2, "fast spiking (FS)"] # fast spiking (FS)
p_TC  = [0.02, 0.25, -65, 0.05, "thalamic-cortical (TC)"] # thalamic-cortical (TC) (doesn't work well)
p_LTS = [0.02, 0.25, -65, 2, "low-threshold spiking (LTS)"] # low-threshold spiking (LTS)
p_RZ  = [0.1, 0.26, -65, 2, "resonator (RZ)"] # resonator (RZ)
# change the parameters of the neuron to p_RS:
current_p = p_CH
neuron.set({"a": current_p[0], "b": current_p[1], "c": current_p[2], "d": current_p[3]})
pprint(f"{neuron.get(['a', 'b', 'c', 'd'])}")

# now, connect the nodes:
nest.Connect(multimeter, neuron)
nest.Connect(neuron, spikerecorder)

# now we run a simulation for 1000 ms:
nest.Simulate(1000.0)

# extract recorded data from the multimeter and plot it:
recorded_events = multimeter.get()
recorded_V = recorded_events["events"]["V_m"]
time = recorded_events["events"]["times"]
spikes = spikerecorder.get("events")
senders = spikes["senders"]

plt.figure(figsize=(8, 4))
plt.plot(time, recorded_V, label="membrane potential")
plt.plot(spikes["times"], spikes["senders"]+np.max(recorded_V), "r.", markersize=10,
         label="spike events")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title(f"Membrane potential of a {current_p[4]} {neuron.get('model')} neuron ($I_e$={I_e} pA)")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.legend(loc="center right")
plt.tight_layout()
plt.savefig(f"figures/single_neuron_{neuron.get('model')}_{current_p[4]}_Ie_{I_e}.png", dpi=300)
plt.show()
#%% IZHIKEVICH NEURON MODEL STREAMLINED
# reset the kernel for safety:
nest.ResetKernel()

# define sets of typical parameters of the Izhikevich neuron model:
p_RS  = [0.02, 0.2, -65, 8, "regular spiking (RS)"] # regular spiking settings for excitatory neurons (RS)
p_IB  = [0.02, 0.2, -55, 4, "intrinsically bursting (IB)"] # intrinsically bursting (IB)
p_CH  = [0.02, 0.2, -51, 2, "chattering (CH)"] # chattering (CH)
p_FS  = [0.1, 0.2, -65, 2, "fast spiking (FS)"] # fast spiking (FS)
p_TC  = [0.02, 0.25, -65, 0.05, "thalamic-cortical (TC)"] # thalamic-cortical (TC) (doesn't work well)
p_LTS = [0.02, 0.25, -65, 2, "low-threshold spiking (LTS)"] # low-threshold spiking (LTS)
p_RZ  = [0.1, 0.26, -65, 2, "resonator (RZ)"] # resonator (RZ)

# list all available neuron models:
pprint(nest.Models())

nest.CopyModel("izhikevich", "izhikevich_RS", {"a": p_RS[0], "b": p_RS[1], "c": p_RS[2], "d": p_RS[3]})
nest.CopyModel("izhikevich", "izhikevich_IB", {"a": p_IB[0], "b": p_IB[1], "c": p_IB[2], "d": p_IB[3]})
nest.CopyModel("izhikevich", "izhikevich_CH", {"a": p_CH[0], "b": p_CH[1], "c": p_CH[2], "d": p_CH[3]})
nest.CopyModel("izhikevich", "izhikevich_FS", {"a": p_FS[0], "b": p_FS[1], "c": p_FS[2], "d": p_FS[3]})
nest.CopyModel("izhikevich", "izhikevich_TC", {"a": p_TC[0], "b": p_TC[1], "c": p_TC[2], "d": p_TC[3]})
nest.CopyModel("izhikevich", "izhikevich_LTS", {"a": p_LTS[0], "b": p_LTS[1], "c": p_LTS[2], "d": p_LTS[3]})
nest.CopyModel("izhikevich", "izhikevich_RZ", {"a": p_RZ[0], "b": p_RZ[1], "c": p_RZ[2], "d": p_RZ[3]})

# list all available neuron models:
pprint(nest.Models())

model_loop_list = ["izhikevich_RS", "izhikevich_IB", "izhikevich_CH", "izhikevich_FS", "izhikevich_TC", "izhikevich_LTS", "izhikevich_RZ"]

for model in model_loop_list:
    # create the neuron, a spike recorder and a multimeter:
    neuron     = nest.Create(model)
    multimeter = nest.Create("multimeter")
    multimeter.set(record_from=["V_m"])
    spikerecorder = nest.Create("spike_recorder")
    
    # set a constant input current for the neuron:
    I_e = 10.0 # [pA]
    neuron.I_e = I_e # [pA]

    # now, connect the nodes:
    nest.Connect(multimeter, neuron)
    nest.Connect(neuron, spikerecorder)

    # now we run a simulation for 1000 ms:
    nest.Simulate(1000.0)

    # extract recorded data from the multimeter and plot it:
    recorded_events = multimeter.get()
    recorded_V = recorded_events["events"]["V_m"]
    time = recorded_events["events"]["times"]
    spikes = spikerecorder.get("events")
    senders = spikes["senders"]

    plt.figure(figsize=(8, 4))
    plt.plot(time, recorded_V, label="membrane potential")
    plt.plot(spikes["times"], spikes["senders"]+np.max(recorded_V), "r.", markersize=10,
            label="spike events")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.title(f"Membrane potential of a {neuron.get('model')} neuron ($I_e$={I_e} pA)")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(f"figures/single_neuron_{model}_Ie_{I_e}.png", dpi=300)
    plt.show()
#%% END