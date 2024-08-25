""" 
NEST simulation of an AEIF (or AdEx) neuron with multiple DC inputs

source: https://nest-simulator.readthedocs.io/en/stable/auto_examples/brette_gerstner_fig_2c.html

modified by: Fabrizio Musacchio
date: Jun 26, 2024
"""
# %% IMPORTS
import os
import matplotlib.pyplot as plt
import nest
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

# set the simulation and the resolution of the simulation:
T = 1000.0 # ms
nest.resolution = 0.1 # ms

# create an AEIF neuron with multiple synapses:
neuron = nest.Create("aeif_cond_alpha")

# set the parameters of the AEIF neuron:
neuron.set(a=4.0, b=80.5)
""" 
a: subthreshold adaptation in nS
b: spike-triggered adaptation in pA
"""

# create two DC generators:
dc = nest.Create("dc_generator", 2)
dc.set(amplitude=[500.0, 800.0], start=[0.0, 500.0], stop=[200.0, 1000.0])

# connect the DC generators to the neuron:
nest.Connect(dc, neuron, "all_to_all")

# create a voltmeter to record the membrane potential of the neuron:
voltmeter = nest.Create("voltmeter", params={"interval": 0.1})
nest.Connect(voltmeter, neuron)

# simulate the network:
nest.Simulate(T)

# extract the data from the voltmeter:
Vms  = voltmeter.get("events", "V_m")
time = voltmeter.get("events", "times")

# plot the membrane potential:
plt.figure(figsize=(5.5, 4))
plt.plot(time, Vms)
plt.xlabel("time (ms)")
plt.ylabel("membrane potential (mV)")
plt.title(f"AdEx neuron with multiple DC inputs")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"figures/aeif_neuron.png", dpi=200)
plt.show()
# %% END