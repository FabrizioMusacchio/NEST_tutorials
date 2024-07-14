""" 
NEST simulation of a population of GIF neurons with oscillatory dynamics.

source: https://nest-simulator.readthedocs.io/en/stable/auto_examples/gif_population.html

author: Fabrizio Musacchio
date: Jun 2, 2024
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
# set global font size for plots:
plt.rcParams.update({'font.size': 12})
# create a folder "figures" to save the plots (if it does not exist):
if not os.path.exists('figures'):
    os.makedirs('figures')
# %% MAIN
nest.ResetKernel()

dt = 0.1 # simulation resolution [ms]
T = 2000.0 # simulation time [ms]

# set the resolution of the simulation:
nest.resolution = dt

# define the parameters of the GIF neuron model:
neuron_params = {
    "C_m": 83.1,
    "g_L": 3.7,
    "E_L": -67.0,
    "Delta_V": 1.4,
    "V_T_star": -39.6,
    "t_ref": 4.0,
    "V_reset": -36.7,
    "lambda_0": 1.0,
    "q_stc": [56.7, -6.9],
    "tau_stc": [57.8, 218.2],
    "q_sfa": [11.7, 1.8],
    "tau_sfa": [53.8, 640.0],
    "tau_syn_ex": 10.0,
}

# define the parameters of the population and noise:
N_ex = 100  # size of the population
p_ex = 0.3  # connection probability inside the population
w_ex = 30.0  # synaptic weights inside the population (pA)

N_noise = 67  # size of Poisson group, default: 50
rate_noise = 12.0  # firing rate of Poisson neurons (Hz), default: 12
w_noise = 20.0  # synaptic weights from Poisson to population neurons (pA)

# create nodes:
population = nest.Create("gif_psc_exp", N_ex, params=neuron_params)
noise = nest.Create("poisson_generator", N_noise, params={"rate": rate_noise})
spikerecorder = nest.Create("spike_recorder")

""" the model:
gif_psc_exp: Generalized Integrate-and-Fire (GIF) neuron model with exponential post-synaptic currents
Due to spike-frequency adaptation, the GIF neurons tend to show oscillatory behavior on the time scale 
comparable with the time constant of adaptation elements (stc and sfa).

https://nest-simulator.readthedocs.io/en/stable/models/gif_psc_exp.html
Mensi et al. (2012) and Pozzorini et al. (2015)
"""

# establish connections:
nest.Connect(population, population, {"rule": "pairwise_bernoulli", "p": p_ex}, syn_spec={"weight": w_ex})
nest.Connect(noise, population, "all_to_all", syn_spec={"weight": w_noise})
nest.Connect(population, spikerecorder)

# simulate the network:
nest.Simulate(T)

""" # plots:
nest.raster_plot.from_device(spikerecorder, hist=True,  hist_binwidth=5.0)
plt.title("Population dynamics")
plt.show() """

spike_events = nest.GetStatus(spikerecorder, "events")[0]
spike_times = spike_events["times"]
neuron_ids = spike_events["senders"]

# combine the spike times and neuron IDs into a single array and sort by time:
spike_data = np.vstack((spike_times, neuron_ids)).T
spike_data_sorted = spike_data[spike_data[:, 0].argsort()]

# extract sorted spike times and neuron IDs:
sorted_spike_times = spike_data_sorted[:, 0]
sorted_neuron_ids = spike_data_sorted[:, 1]

# %% PLOTTING
# spike raster plot and histogram of spiking rate:
fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(5, 1)

# create the first subplot (3/4 of the figure)
ax1 = plt.subplot(gs[0:4, :])
ax1.scatter(sorted_spike_times, sorted_neuron_ids, s=9.0, color='mediumaquamarine', alpha=1.0)
ax1.set_title("Oscillatory population dynamics of GIF neurons:\nspike times (top) and rate (bottom)")
#ax1.set_xlabel("time [ms]")
ax1.set_xticks([])
ax1.set_ylabel("neuron ID")
ax1.set_xlim([0, T])
ax1.set_ylim([0, N_ex])
ax1.spines["top"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_yticks(np.arange(0, N_ex+1, 10))

# create the second subplot (1/4 of the figure)
ax2 = plt.subplot(gs[4, :])
hist_binwidth = 5.0
t_bins = np.arange(np.amin(sorted_spike_times), np.amax(sorted_spike_times), hist_binwidth)
n, bins = np.histogram(sorted_spike_times, bins=t_bins)
heights = 1000 * n / (hist_binwidth * (N_ex))
ax2.bar(t_bins[:-1], heights, width=hist_binwidth, color='violet')
ax2.spines["top"].set_visible(False)
ax2.spines["bottom"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.spines["right"].set_visible(False)
#ax2.set_title(f"histogram of spiking rate vs. time")
ax2.set_ylabel("firing rate\n[Hz]")
ax2.set_xlabel("time [ms]")
ax2.set_xlim([0, T])

plt.tight_layout()
plt.savefig(f"figures/oscillatory_gif_population_combined_plot.png", dpi=200)
plt.show()
# %% END