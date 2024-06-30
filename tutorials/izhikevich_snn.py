"""
NEST simulation of a SNN with Izhikevich neurons and noise input.

author: Fabrizio Musacchio
date: Jun 02, 2024
"""
#%% IMPORTS
import os
import matplotlib.pyplot as plt
import numpy as np
import nest
# set the verbosity of the NEST simulator:
nest.set_verbosity("M_WARNING")
# set global font size for plots:
plt.rcParams.update({'font.size': 12})
# create a folder "figures" to save the plots (if it does not exist):
if not os.path.exists('figures'):
    os.makedirs('figures')
# %% MAIN
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

# copy the Izhikevich neuron model and set the parameters for the different neuron types:
nest.CopyModel("izhikevich", "izhikevich_RS", {"a": p_RS[0], "b": p_RS[1], "c": p_RS[2], "d": p_RS[3]})
nest.CopyModel("izhikevich", "izhikevich_IB", {"a": p_IB[0], "b": p_IB[1], "c": p_IB[2], "d": p_IB[3]})
nest.CopyModel("izhikevich", "izhikevich_CH", {"a": p_CH[0], "b": p_CH[1], "c": p_CH[2], "d": p_CH[3]})
nest.CopyModel("izhikevich", "izhikevich_FS", {"a": p_FS[0], "b": p_FS[1], "c": p_FS[2], "d": p_FS[3]})
nest.CopyModel("izhikevich", "izhikevich_TC", {"a": p_TC[0], "b": p_TC[1], "c": p_TC[2], "d": p_TC[3]})
nest.CopyModel("izhikevich", "izhikevich_LTS", {"a": p_LTS[0], "b": p_LTS[1], "c": p_LTS[2], "d": p_LTS[3]})
nest.CopyModel("izhikevich", "izhikevich_RZ", {"a": p_RZ[0], "b": p_RZ[1], "c": p_RZ[2], "d": p_RZ[3]})

# set up a two-neuron-type network according to Izhikevich's original paper:
Ne = 800  # Number of excitatory neurons
Ni = 200  # Number of inhibitory neurons
T = 1000.0 # Simulation time (ms)
population_e     = nest.Create("izhikevich_RS", n=Ne)
population_i     = nest.Create("izhikevich_CH", n=Ni)
multimeter = nest.Create("multimeter")
multimeter.set(record_from=["V_m"])
spikerecorder = nest.Create("spike_recorder")

# set a constant input current for the neuron:
I_e = 0.0 # [pA]
population_e.I_e = I_e
population_i.I_e = I_e

# set up some Poisson-noisy current input:
""" noise = nest.Create("poisson_generator")
noise.rate = 5000.0 # [Hz] """

# set up some Gaussian-noisy current input:
noise = nest.Create("noise_generator")
noise.mean = 10.0 # mean value of the noise current [pA]
noise.std = 2.0 # standard deviation of the noise current [pA]
noise.std_mod = 0.0 # modulation of the standard deviation of the noise current (pA)
noise.phase=0 # phase of sine modulation (0–360 deg)
#noise.get()

# define connectivity based on a percentage:
conn_prob_ex = 0.10  # connectivity probability of population E
conn_prob_in = 0.60  # connectivity probability of population I
# compute the number of connections based on the probabilities:
num_conn_ex_to_ex = int(Ne * conn_prob_ex)
num_conn_in_to_ex = 70#int(Ne * conn_prob_in)
num_conn_in_to_in = int(Ni * conn_prob_in)
num_conn_ex_to_in = 70#int(Ni * conn_prob_ex)
# create connection dictionaries for fixed indegree:
conn_dict_ex_to_ex = {"rule": "fixed_indegree", "indegree": num_conn_ex_to_ex}
conn_dict_ex_to_in = {"rule": "fixed_indegree", "indegree": num_conn_ex_to_in}
conn_dict_in_to_ex = {"rule": "fixed_indegree", "indegree": num_conn_in_to_ex}
conn_dict_in_to_in = {"rule": "fixed_indegree", "indegree": num_conn_in_to_in}

# synaptic weights and delays:
d = 1.0 # synaptic delay [ms]
syn_dict_ex = {"delay": d, "weight": 0.5}
syn_dict_in = {"delay": d, "weight": -1.0}

# connect neurons:
nest.Connect(population_e, population_e, conn_dict_ex_to_ex, syn_dict_ex)  # E to E
nest.Connect(population_e, population_i, conn_dict_ex_to_in, syn_dict_ex)  # E to I
nest.Connect(population_i, population_i, conn_dict_in_to_in, syn_dict_in)  # I to I
nest.Connect(population_i, population_e, conn_dict_in_to_ex, syn_dict_in)  # I to E


""" N_e = 30 # number of excitatory inputs
N_i = 20 # number of inhibitory inputs
conn_dict_ex = {"rule": "fixed_indegree", "indegree": N_e}
conn_dict_in = {"rule": "fixed_indegree", "indegree": N_i}
syn_dict_ex = {"delay": d, "weight":   0.5}
syn_dict_in = {"delay": d, "weight": -1.0}
nest.Connect(population_e, population_e + population_i, conn_dict_ex, syn_dict_ex)
nest.Connect(population_i, population_e + population_i, conn_dict_in, syn_dict_in) """

# connect noise to the populations:
nest.Connect(noise, population_e, syn_spec={'weight': 1.0})
nest.Connect(noise, population_i, syn_spec={'weight': 1.0})

# connect the multimeter to the excitatory population and to the inhibitory population:
nest.Connect(multimeter, population_e + population_i)
nest.Connect(population_e + population_i, spikerecorder)


# now we run a simulation for 1000 ms:
nest.Simulate(T)

spike_events = nest.GetStatus(spikerecorder, "events")[0]
spike_times = spike_events["times"]
neuron_ids = spike_events["senders"]

# Combine the spike times and neuron IDs into a single array and sort by time:
spike_data = np.vstack((spike_times, neuron_ids)).T
spike_data_sorted = spike_data[spike_data[:, 0].argsort()]

# Extract sorted spike times and neuron IDs:
sorted_spike_times = spike_data_sorted[:, 0]
sorted_neuron_ids = spike_data_sorted[:, 1]

# plotting spike times:
plt.figure(figsize=(6, 6))
plt.scatter(sorted_spike_times, sorted_neuron_ids, s=0.5, color='black')
plt.title("Spike times")
plt.xlabel("Time (ms)")
plt.ylabel("Neuron ID")
plt.axhline(y=Ne, color='k', linestyle='-', linewidth=1)
plt.text(0.7, 0.76, population_e.get('model')[0], color='k', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=1)) 
plt.text(0.7, 0.84, population_i.get('model')[0], color='k', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=1))
plt.xlim([0, T])
plt.ylim([0, Ne+Ni])
plt.yticks(np.arange(0, Ne+Ni+1, 200))
plt.tight_layout()
plt.savefig(f"figures/spike_raster_plot_{population_i.get('model')[0]}_{population_e.get('model')[0]}.png", dpi=200)
plt.show()


""" 
nest.raster_plot.from_device(spikerecorder, hist=True,  hist_binwidth=5.0)
plt.show() 
"""
""" 
Source code of NEST's histogram plot:
https://www.nest-simulator.org/pynest-api/_modules/nest/raster_plot.html
"""

# plot histogram of spiking rate [Hz] vs. time [ms]:
hist_binwidth = 5.0
t_bins = np.arange(np.amin(sorted_spike_times), np.amax(sorted_spike_times), hist_binwidth)
n, bins = np.histogram(sorted_spike_times, bins=t_bins)
heights = 1000 * n / (hist_binwidth * (Ne+Ni))
"""
calculates the average firing rate of the neurons in the network in units of spikes per second (Hz).
The factor of 1000 is used to convert milliseconds to seconds, as the firing rate is typically 
expressed in Hz (spikes per second), but the time bins and spike times are in milliseconds.
"""
plt.figure(figsize=(6, 2.0))
plt.bar(t_bins[:-1], heights, width=hist_binwidth, color='blue')
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.title(f"histogram of spiking rate vs. time")
plt.ylabel("firing rate [Hz]")
plt.xlabel("time [ms]")
plt.xlim([0, T])
plt.tight_layout()
plt.savefig(f"figures/spiking_rate_histogram_{population_i.get('model')[0]}_{population_e.get('model')[0]}.png", dpi=200)
plt.show()


# %% END