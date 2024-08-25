""" 
NEST simulation of an AEIF neuron with multiple synaptic rise and decay time constants.

https://nest-simulator.readthedocs.io/en/stable/auto_examples/aeif_cond_beta_multisynapse.html

modified by: Fabrizio Musacchio
date: Jun 26, 2024
"""
# %% IMPORTS
import os
import matplotlib.pyplot as plt
import numpy as np
import nest
import nest.raster_plot
# set the verbosity of the NEST simulator:
nest.set_verbosity("M_WARNING")
# set global properties for all plots:
plt.rcParams.update({'font.size': 12})
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.left"]   = False
plt.rcParams["axes.spines.right"]  = False
# create a folder "figures" to save the plots (if it does not exist):
if not os.path.exists('figures'):
    os.makedirs('figures')
# %% MAIN
nest.ResetKernel()
# define simulation time:
T = 1000.0 # ms

# define neuron parameters:
aeif_neuron_params = {
    "V_peak": 0.0,  # spike detection threshold in mV
    "a": 4.0,       # subthreshold adaptation in nS
    "b": 80.5,      # spike-triggered adaptation in pA
    "E_rev": [0.0, 0.0, 0.0, -85.0],        # reversal potentials in mV
    "tau_decay": [50.0, 20.0, 20.0, 20.0],  # synaptic decay time in ms
    "tau_rise": [10.0, 10.0, 1.0, 1.0]}     # synaptic rise time in ms

# create an AEIF neuron with multiple synapses:
neuron = nest.Create("aeif_cond_beta_multisynapse")
nest.SetStatus(neuron, params=aeif_neuron_params)

# create a spike generator:
spikerecorder = nest.Create("spike_generator", params={"spike_times": np.array([10.0])})

# create a voltmeter to record the membrane potential of the neuron:
voltmeter = nest.Create("voltmeter")

# connect the spike generator to the neuron:
delays = [1.0, 300.0, 500.0, 700.0]
w = [1.0, 1.0, 1.0, 1.0]
for syn in range(4):
    nest.Connect(
        spikerecorder,
        neuron,
        syn_spec={"synapse_model": "static_synapse", 
                  "receptor_type": 1 + syn, 
                  "weight": w[syn], 
                  "delay": delays[syn]},
    )

# connect the voltmeter to the neuron:
nest.Connect(voltmeter, neuron)

# simulate the network:
nest.Simulate(T)

# extract the data from the voltmeter:
Vms = voltmeter.get("events", "V_m")
ts = voltmeter.get("events", "times")

# plot the membrane potential:
plt.figure(figsize=(5.5, 4))
plt.plot(ts, Vms)
plt.xlabel("time [ms]")
plt.ylabel("membrane potential [mV]")
plt.title(f"AdEx neuron with multiple synapses")
plt.tight_layout()
plt.savefig(f"figures/aeif_neuron_with_multple_synapse_rices_and_decays.png", dpi=200)
plt.show()
# %% END