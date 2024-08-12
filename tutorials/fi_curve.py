""" 
NEST simulation of Hodgkin-Huxley neuron for calculating a FI-curve (firing rate vs. 
input current).

source: https://nest-simulator.readthedocs.io/en/stable/auto_examples/hh_psc_alpha.html

modified by: Fabrizio Musacchio
date: Jun 26, 2024
"""
# %% IMPORTS
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
time = 1000

# amplitude range (in pA):
I_start = 0
I_stopt = 2000
I_step  = 20

# define simulation step size (in mS):
h = 0.1

# create a Hodgkin-Huxley neuron and a spikerecorder node:
neuron = nest.Create("hh_psc_alpha")
spikerecorder = nest.Create("spike_recorder")
spikerecorder.record_to = "memory"

nest.Connect(neuron, spikerecorder, syn_spec={"weight": 1.0, "delay": h})

# simulation loop:
n_data = int(I_stopt / float(I_step))
amplitudes  = np.zeros(n_data)
event_freqs = np.zeros(n_data)
for i, amp in enumerate(range(I_start, I_stopt, I_step)):
    neuron.I_e = float(amp)
    
    nest.Simulate(1000)         # one second warm-up time for equilibrium state
    spikerecorder.n_events = 0  # then reset spike counts
    nest.Simulate(time)         # another simulation call to record firing rate

    n_events = spikerecorder.n_events
    amplitudes[i]  = amp
    event_freqs[i] = n_events / (time / 1000.0)
    print(f"Simulating with current I={amp} pA -> {n_events} spikes in {time} ms ({event_freqs[i]} Hz)")

# plot the results:
plt.figure(figsize=(5, 4))
plt.plot(amplitudes, event_freqs, lw=2.0)
plt.xlabel("input current (pA)")
plt.ylabel("firing rate (Hz)")
plt.title("Firing rate vs. input current\nof a Hodgkin-Huxley neuron")
plt.tight_layout()
plt.grid(True)
plt.savefig("figures/hh_fi_curve.png", dpi=200)
plt.show()
# %% END