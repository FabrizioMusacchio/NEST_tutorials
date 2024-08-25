""" 
NEST simulation of spike synchronization through subthreshold oscillation

source: https://nest-simulator.readthedocs.io/en/stable/auto_examples/BrodyHopfield.html
based on: Brody CD and Hopfield JJ (2003). Simple networks for spike-timing-based computation, 
          with application to olfactory processing. Neuron 37, 843-852.

modified by: Fabrizio Musacchio
date: Jun 26, 2024
"""
# %% IMPORTS
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
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

N = 1000  # number of neurons
bias_begin = 140.0  # minimal value for the bias current injection [pA]
bias_end = 200.0  # maximal value for the bias current injection [pA]
T = 600  # simulation time (ms)

# parameters for the alternating-current generator
driveparams = {"amplitude": 50.0, "frequency": 35.0}
# parameters for the noise generator
noiseparams = {"mean": 0.0, "std": 25.0}
neuronparams = {
    "tau_m": 20.0,  # membrane time constant
    "V_th":  20.0,  # threshold potential
    "E_L":   10.0,  # membrane resting potential
    "t_ref":  2.0,  # refractory period
    "V_reset":0.0,  # reset potential
    "C_m":  200.0,  # membrane capacitance
    "V_m":    0.0,  # initial membrane potential
}

neurons = nest.Create("iaf_psc_alpha", N)
spikerecorder = nest.Create("spike_recorder")
noise = nest.Create("noise_generator")
drive = nest.Create("ac_generator")
multimeter = nest.Create("multimeter", params={"record_from": ["I_syn_ex", "I_syn_in", "V_m"]})

drive.set(driveparams)
noise.set(noiseparams)

neurons.set(neuronparams)
neurons.I_e = [(n * (bias_end - bias_begin) / N + bias_begin) for n in range(1, len(neurons) + 1)]

nest.Connect(drive, neurons) # we use default connection rule: all-to-all
nest.Connect(noise, neurons)
nest.Connect(neurons, spikerecorder)
nest.Connect(multimeter, neurons)

nest.Simulate(T)


# extract spike times and neuron IDs from the spike recorder for plotting:
spike_events = nest.GetStatus(spikerecorder, "events")[0]
spike_times = spike_events["times"]
neuron_ids = spike_events["senders"]

# combine the spike times and neuron IDs into a single array and sort by time:
spike_data = np.vstack((spike_times, neuron_ids)).T
spike_data_sorted = spike_data[spike_data[:, 0].argsort()]

# extract sorted spike times and neuron IDs:
sorted_spike_times = spike_data_sorted[:, 0]
sorted_neuron_ids = spike_data_sorted[:, 1]

# extract recorded data from the multimeter:
multimeter_events = nest.GetStatus(multimeter, "events")[0]
""" times = multimeter_data["times"]
Isyn_ex = multimeter_data["I_syn_ex"]
Isyn_in = multimeter_data["I_syn_in"]
V_m = multimeter_data["V_m"] """

# %% PLOTTING
""" nest.raster_plot.from_device(spikerecorder, hist=True)
plt.show() """


# spike raster plot and histogram of spiking rate:
fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(5, 1)

# create the first subplot (3/4 of the figure)
ax1 = plt.subplot(gs[0:4, :])
ax1.scatter(sorted_spike_times, sorted_neuron_ids, s=9.0, color='mediumaquamarine', alpha=1.0)
ax1.set_title("spike synchronization through subthreshold oscillation:\nspike times (top) and rate (bottom)")
#ax1.set_xlabel("time [ms]")
ax1.set_xticks([])
ax1.set_ylabel("neuron ID")
ax1.set_xlim([0, T])
ax1.set_ylim([0, N])
ax1.set_yticks(np.arange(0, N+1, 100))

# create the second subplot (1/4 of the figure)
ax2 = plt.subplot(gs[4, :])
hist_binwidth = 5.0
t_bins = np.arange(np.amin(sorted_spike_times), np.amax(sorted_spike_times), hist_binwidth)
n, bins = np.histogram(sorted_spike_times, bins=t_bins)
heights = 1000 * n / (hist_binwidth * (N))
ax2.bar(t_bins[:-1], heights, width=hist_binwidth, color='violet')
#ax2.set_title(f"histogram of spiking rate vs. time")
ax2.set_ylabel("firing rate\n[Hz]")
ax2.set_xlabel("time [ms]")
ax2.set_xlim([0, T])

plt.tight_layout()
plt.savefig("figures/spike_synchronization_through_oscillation.png", dpi=200)
plt.show()

# plot the membrane potential and synaptic currents for 3 exemplary neurons:
fig = plt.figure(figsize=(6, 2))
sender = 100
idc_sender = multimeter_events["senders"] == sender
plt.plot(multimeter_events["times"][idc_sender], multimeter_events["V_m"][idc_sender], 
         label=f"neuron ID {sender}: ", alpha=1.0, c="k", lw=1.75, zorder=3)
sender = 200
idc_sender = multimeter_events["senders"] == sender
plt.plot(multimeter_events["times"][idc_sender], multimeter_events["V_m"][idc_sender], 
         label=f"neuron ID {sender}: ", alpha=0.8)
sender = 800
idc_sender = multimeter_events["senders"] == sender
plt.plot(multimeter_events["times"][idc_sender], multimeter_events["V_m"][idc_sender], 
         label=f"neuron ID {sender}: ", alpha=0.8)
plt.ylabel("membrane\npotential\n[mV]")
plt.xlabel("time [ms]")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/spike_synchronization_through_oscillation_membrane_potential_examples.png", dpi=200)
plt.show()


""" fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 6), dpi=100)
axes = ax.flat
sender = 100
idc_sender = multimeter_events["senders"] == sender
axes[0].plot(multimeter_events["times"][idc_sender], multimeter_events["V_m"][idc_sender], label="membrane potential") """

# loop over all senders and collect all senders' V_m traces in a 2D array:
V_m_traces = np.zeros((N, T-1))
for sender_i, sender in enumerate(set(multimeter_events["senders"])):
    idc_sender = multimeter_events["senders"] == sender
    curr_V_m_trace = multimeter_events["times"][idc_sender]
    V_m_traces[sender_i, :] = multimeter_events["V_m"][idc_sender]

# plot neuron averages and std of membrane potential for different groups of neurons:
fig, ax = plt.subplots(5, 1, sharex=True, figsize=(6, 8), gridspec_kw={"hspace": 0.3})
axes = ax.flat
for i, (start, end) in enumerate([(800, 1000), (600, 800), (400, 600), (200, 400),(0, 200) ]):
    V_m_mean = np.mean(V_m_traces[start:end, :], axis=0)
    V_m_std = np.std(V_m_traces[start:end, :], axis=0)
    axes[i].plot(multimeter_events["times"][idc_sender], V_m_mean, label="mean membrane potential", c="k")
    axes[i].fill_between(multimeter_events["times"][idc_sender], V_m_mean - V_m_std, V_m_mean + V_m_std, color='gray', alpha=0.5)
    axes[i].set_ylabel(f"membrane\npotential $V_m$\n[mV]")
    axes[i].set_title(f"average and std of neurons {start} to {end}")
axes[-1].set_xlabel("time [ms]")
plt.tight_layout()
plt.savefig("figures/spike_synchronization_through_oscillation_membrane_potential_average_and_std.png", dpi=200)
plt.show()

# %% END