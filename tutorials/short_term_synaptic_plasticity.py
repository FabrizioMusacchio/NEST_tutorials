""" 
Simple example of short-term synaptic plasticity following the model of Tsodyks and Markram (1998).

source: https://nest-simulator.readthedocs.io/en/stable/auto_examples/iaf_tum_2000_short_term_facilitation.html

modified by: Fabrizio Musacchio
date: Jul 28, 2024
"""
# %% IMPORTS
import numpy as np
import os
import matplotlib.pyplot as plt

# Set global properties for all plots:
plt.rcParams.update({'font.size': 12})
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.left"]   = False
plt.rcParams["axes.spines.right"]  = False

# create a folder "figures" to save the plots (if it does not exist):
if not os.path.exists('figures'):
    os.makedirs('figures')
# %% MAIN
U_initial = 0.0 # set initial release probability
x_initial = 1.0 # set initial fraction of synaptic vesicles in the readily releasable pool

u = U_initial
x = x_initial
I = 0           # initial synaptic current in arbitrary units

# set time constants and parameters:
tau_f = 50   # facilitation time constant
tau_d = 750  # depression time constant
tau_s = 5    # synaptic current decay time constant
U = 0.45     # utilization factor
A = 1        # synaptic response amplitude
""" 
STD-dominating synapse:
tau_f = 50
tau_d = 750
tau_s = 20
U = 0.45
A = 1

STF-dominating synapse:
tau_f = 750
tau_d = 50
U=0.15
"""

# set time step for numerical integration:
dt = 0.1  # ms
simulation_duration = 100 # ms

# set some example spike times:
spike_times = [10, 20, 30, 50, 70]  # in ms
# set a neural firing rate of 15 Hz:
#spike_times = np.arange(0, simulation_duration, 1000 / 15)

I_trace = []  # list to store the synaptic current over time
u_trace = []  # list to store the release probability over time
x_trace = []  # list to store the fraction of synaptic vesicles in the readily releasable pool over time

# simulation loop:
for t in np.arange(0, simulation_duration, dt):
    # check for spikes:
    if t in spike_times:
        # spike event:
        u = u + U * (1 - u)
        I = I + A * u * x
        x = x - u * x
    
    # update u, x, and I between spikes:
    u = u * np.exp(-dt/tau_f)    # exponential decay of facilitation
    x = x + (1 - x) * (dt/tau_d) # Euler step for recovery from depression
    I = I * np.exp(-dt/tau_s)    # exponential decay of synaptic current
    
    I_trace.append(I)
    u_trace.append(u)
    x_trace.append(x)

time = np.arange(0, simulation_duration, dt)

# plot the results:
plt.figure(figsize=(6, 4))
plt.plot(time, u_trace, label="release probability $u$", lw=2.0)
plt.plot(time, x_trace, label="fraction of synaptic vesicles $x$", lw=2.0, alpha=0.5)
plt.plot(time, I_trace, label="synaptic current $I$", lw=2.0, c="black")
plt.plot(spike_times, np.zeros_like(spike_times), 'ro',label="spike timepoint")
plt.xlabel("time [ms]")
plt.ylabel(f"arbitrary units")
plt.legend()
plt.tight_layout()
#plt.savefig(f"figures/short_term_synaptic_plasticity_STF.png", dpi=200)
plt.savefig(f"figures/short_term_synaptic_plasticity_STD.png", dpi=200)
plt.show()
# %% END