""" 
NEST simulation of a single neuron with balanced excitatory and inhibitory inputs

source: https://nest-simulator.readthedocs.io/en/stable/auto_examples/balancedneuron.html

modified by: Fabrizio Musacchio
date: Jun 5, 2024
"""
# %% IMPORTS
import os
import matplotlib.pyplot as plt
import numpy as np
import nest
import nest.voltage_trace
from scipy.optimize import bisect
# set the verbosity of the NEST simulator:
nest.set_verbosity("M_WARNING")
# set global font size for plots:
plt.rcParams.update({'font.size': 12})
# create a folder "figures" to save the plots (if it does not exist):
if not os.path.exists('figures'):
    os.makedirs('figures')
# %% MAIN
nest.ResetKernel()

t_sim = 25000.0  # simulation time [ms]
n_ex = 16000  # size of the excitatory population
n_in = 4000   # size of the inhibitory population
r_ex = 5.0    # mean rate of the excitatory population
r_in = 20.5   # initial rate of the inhibitory population
epsc = 45.0   # peak amplitude of excitatory synaptic currents
ipsc = -45.0  # peak amplitude of inhibitory synaptic currents
d = 1.0       # synaptic delay
lower = 15.0  # lower bound of the search interval using bisect
upper = 25.0  # upper bound of the search interval using bisect
prec = 0.01   # precision of the bisect search

# create nodes:
neuron    = nest.Create("iaf_psc_alpha") # single neuron with alpha-shaped postsynaptic currents
noise     = nest.Create("poisson_generator", 2) # two Poisson generators for the excitatory and inhibitory populations
voltmeter     = nest.Create("voltmeter")
spikerecorder = nest.Create("spike_recorder")
multimeter    = nest.Create("multimeter")
multimeter.set(record_from=["V_m"]) # record the membrane potential of the neuron to which the multimeter will be connected

# define the noise rates:
noise.rate = [n_ex * r_ex, n_in * r_in]

# make connections:
nest.Connect(neuron, spikerecorder)
nest.Connect(multimeter, neuron)
nest.Connect(voltmeter, neuron)
nest.Connect(noise, neuron, syn_spec={"weight": [[epsc, ipsc]], "delay": 1.0})

def output_rate(guess):
    print("Inhibitory rate estimate: %5.2f Hz" % guess)
    rate = float(abs(n_in * guess))
    noise[1].rate = rate # update the Poisson firing rate of the inhibitory population
    spikerecorder.n_events = 0
    nest.Simulate(t_sim)
    out = spikerecorder.n_events * 1000.0 / t_sim
    print(f"  -> Neuron rate: {out} Hz (goal: {r_ex} Hz)")
    return out

in_rate = bisect(lambda x: output_rate(x) - r_ex, lower, upper, xtol=prec)
print(f"Optimal rate for the inhibitory population: {in_rate} Hz")

""" nest.voltage_trace.from_device(voltmeter)
plt.show() """

# extract recorded data from the multimeter and plot it:
recorded_events = multimeter.get()
recorded_V = recorded_events["events"]["V_m"]
time = recorded_events["events"]["times"]
spikes = spikerecorder.get("events")
senders = spikes["senders"]

plt.figure(figsize=(7, 5))
plt.plot(time, recorded_V, label="membrane potential")
""" plt.plot(spikes["times"], spikes["senders"]+np.max(recorded_V), "r.", markersize=10,
         label="spike events") """
plt.xlabel("time (ms)")
plt.ylabel("membrane potential (mV)")
plt.title(f"Membrane potential of a {neuron.get('model')} neuron")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(f"figures/single_neuron_{neuron.get('model')}.png", dpi=300)
plt.show()

# %% END