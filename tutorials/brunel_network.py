""" 
NEST simulation of a random balanced network (alpha synapses) 

source: https://nest-simulator.readthedocs.io/en/stable/auto_examples/brunel_alpha_nest.html

modified by: Fabrizio Musacchio
date: Jun 26, 2024
"""
# %% IMPORTS
import os
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import nest
import nest.raster_plot
import scipy.special as sp
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
# %% FUNCTIONS
def LambertWm1(x):
    """ 
    Here we define the Lambert W function for x < 0. The function is usually used
    in the context of the Brunel network to compute the maximum of the PSP (postsynaptic
    potential) for a current of unit amplitude.     
    """
    # using scipy to mimic the gsl_sf_lambert_Wm1 function:
    return sp.lambertw(x, k=-1 if x < 0 else 0).real

def ComputePSPnorm(tauMem, CMem, tauSyn):
    """ 
    This function computes the maximum of the PSP for a current of unit amplitude.
    """
    a = tauMem / tauSyn
    b = 1.0 / tauSyn - 1.0 / tauMem

    # time of maximum
    t_max = 1.0 / b * (-LambertWm1(-np.exp(-1.0 / a) / a) - 1.0 / a)

    # maximum of PSP for current of unit amplitude:
    return (
        np.exp(1.0)
        / (tauSyn * CMem * b)
        * ((np.exp(-t_max / tauMem) - np.exp(-t_max / tauSyn)) / b - t_max * np.exp(-t_max / tauSyn))
    )
# %% MAIN
startbuild = time.time()

# set the simulation resolution and time:
dt = 0.1          # the resolution in ms
simtime = 1000.0  # Simulation time in ms
delay = 1.5         # synaptic delay in ms; also controls the dynamics of the network

nest.ResetKernel()
nest.resolution = dt
nest.print_time = False
nest.overwrite_files = True

# set the parameters of the network:
g = 7.5          # ratio inhibitory weight/excitatory weight
eta = 0.9        # defines the ration between external rate and threshold rate
epsilon = 0.1    # connection probability
"""
Parameter study (reproducing Fig. 8 in accordance to Fig. 7 from Brunel, 2000):

g=3.5, eta=0.9 -> synchronized network, neurons fire synchronously at high rates, 
                  global activity is oscillatory but irregular (varying frequency)
g=4.0, eta=0.9 -> synchronized network, neurons fire synchronously at low rates, 
                  global activity is oscillatory but irregular (varying frequency)
                  
g=5.0, eta=0.9 -> entering asynchronous irregular state (network is not synchronized),
                  neurons fire irregularly at low rates, no global oscillations
                  
g=5.0, eta=1.0 -> asynchronized network, neurons fire irregularly at slightly higher rates
g=5.0, eta=2.0 -> asynchronized network, neurons fire irregularly at high rates
g=7.0, eta=2.0 -> asynchronized network, neurons fire irregularly at slightly lower rates
g=7.0, eta=3.5 -> asynchronized network, neurons fire irregularly at high rates

g=3.0, eta=2.0 -> fully synchronized network, neurons fire synchronously at high rates,
                  no global oscillations, just constant high activity
g=3.0, eta=0.9 ->  -"-

* Synchronous regular (SR) states, where neurons are almost fully synchronized in a few clusters and behave as oscillators when excitation dominates inhibition and synaptic time distributions are sharply peaked;
* Asynchronous regular (AR) states, with stationary global activity and quasi-regular individual neuron firing when excitation dominates inhibition and synaptic time distributions are broadly peaked;
* Asynchronous irregular (AI) states, with stationary global activity but strongly irregular individual firing at low rates when inhibition dominates excitation in an intermediate range of external frequencies;
* Synchronous irregular (SI) states, with oscillatory global activity but strongly irregular individual firing at low (compared to the global oscillation frequency) firing rates, when inhibition dominates excitation and either low external frequencies (slow oscillations) or high external frequencies (fast oscillations). When the average synaptic time constant is high enough, these two regions merge together.
"""

order = 2500    # scaling factor for number of neurons
NE = 4 * order  # number of excitatory neurons
NI = 1 * order  # number of inhibitory neurons
N_neurons = NE + NI  # number of neurons in total
N_rec = 50      # record from 50 neurons instead of all (for visualization in the plots later
                # and similar to Brunel's original publication)

CE = int(epsilon * NE)  # number of excitatory synapses per neuron
CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
C_tot = int(CI + CE)    # total number of synapses per neuron

tauSyn = 0.5  # synaptic time constant in ms
tauMem = 20.0 # time constant of membrane potential in ms
CMem = 250.0  # capacitance of membrane in in pF
theta = 20.0  # membrane threshold potential in mV
neuron_params = {
    "C_m": CMem,
    "tau_m": tauMem,
    "tau_syn_ex": tauSyn,
    "tau_syn_in": tauSyn,
    "t_ref": 2.0,
    "E_L": 0.0,
    "V_reset": 0.0,
    "V_m": 0.0,
    "V_th": theta}
J = 0.1  # postsynaptic amplitude in mV
J_unit = ComputePSPnorm(tauMem, CMem, tauSyn)
J_ex = J / J_unit # amplitude of excitatory postsynaptic current
J_in = -g * J_ex  # amplitude of inhibitory postsynaptic current

# define the threshold rate, which is the external rate needed to fix the membrane 
# potential around its threshold, as well as the external firing rate and the rate 
# of the poisson generator which is multiplied by the in-degree CE and converted to 
# Hz by multiplication by 1000:
nu_th = (theta * CMem) / (J_ex * CE * np.exp(1) * tauMem * tauSyn) # threshold rate
nu_ex = eta * nu_th # external rate
p_rate = 1000.0 * nu_ex * CE # rate of the Poisson generator

# create the nodes:
print("Building network...")
nodes_ex = nest.Create("iaf_psc_alpha", NE, params=neuron_params)
nodes_in = nest.Create("iaf_psc_alpha", NI, params=neuron_params)
noise    = nest.Create("poisson_generator", params={"rate": p_rate})
exspikes = nest.Create("spike_recorder") # record excitatory spikes
inspikes = nest.Create("spike_recorder") # record inhibitory spikes

exspikes.set(label="brunel-py-ex", record_to="memory")
inspikes.set(label="brunel-py-in", record_to="ascii") # record to ascii=to file (for memory reasons)

# make the connections:
print("Connecting devices...")
# create a copy of the static_synapse model and set its default the weight and delay parameters:
nest.CopyModel("static_synapse", "excitatory", {"weight": J_ex, "delay": delay})
nest.CopyModel("static_synapse", "inhibitory", {"weight": J_in, "delay": delay})

# connect the noise to the excitatory and inhibitory nodes  (using default all-to-all rule):
nest.Connect(noise, nodes_ex, syn_spec="excitatory")
nest.Connect(noise, nodes_in, syn_spec="excitatory")

# connect the spike recorders to the excitatory and inhibitory nodes:
nest.Connect(nodes_ex[:N_rec], exspikes, syn_spec="excitatory")
nest.Connect(nodes_in[:N_rec], inspikes, syn_spec="excitatory")

print("Connecting network...")

# connect the excitatory nodes:
print("Excitatory connections...")
conn_params_ex = {"rule": "fixed_indegree", "indegree": CE}
nest.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_ex, "excitatory")

# connect the inhibitory nodes:
print("Inhibitory connections")
conn_params_in = {"rule": "fixed_indegree", "indegree": CI}
nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_in, "inhibitory")

endbuild = time.time()

# start the simulation:
print("Simulating")
nest.Simulate(simtime)

endsimulate = time.time()

# extract the number of events and the firing rate:
events_ex = exspikes.n_events
events_in = inspikes.n_events

rate_ex = events_ex / simtime * 1000.0 / N_rec
rate_in = events_in / simtime * 1000.0 / N_rec

num_synapses_ex = nest.GetDefaults("excitatory")["num_connections"]
num_synapses_in = nest.GetDefaults("inhibitory")["num_connections"]
num_synapses = num_synapses_ex + num_synapses_in

# calculate the time it took to build the network and simulate it:
build_time = endbuild - startbuild
sim_time   = endsimulate - endbuild

# print some information about the simulation:
print("Brunel network simulation (Python)")
print(f"Number of neurons : {N_neurons}")
print(f"Number of synapses: {num_synapses}")
print(f"       Excitatory : {num_synapses_ex}")
print(f"       Inhibitory : {num_synapses_in}")
print(f"Excitatory rate   : {rate_ex:.2f} Hz")
print(f"Inhibitory rate   : {rate_in:.2f} Hz")
print(f"Building time     : {build_time:.2f} s")
print(f"Simulation time   : {sim_time:.2f} s")
## %% PLOTTING

# plot the raster plot of the excitatory neurons:
nest.raster_plot.from_device(exspikes, hist=True)
plt.title(f"50 example excitatory neurons (out of {NE})\ng: {g}, nu_ex/nu_th: {nu_ex/nu_th}, delay: {delay}, eta: {eta}, epsilon: {epsilon}")
plt.tight_layout()
plt.savefig(f"figures/brunel_network_raster_plot_g{g}_nu_ex_nu_th{nu_ex/nu_th}_delay{delay}_2.png", dpi=200)
plt.show()

# extract the spike times and neuron IDs from the excitatory spike recorder:
spike_events = nest.GetStatus(exspikes, "events")[0]
spike_times = spike_events["times"]
neuron_ids = spike_events["senders"]

# combine the spike times and neuron IDs into a single array and sort by time:
spike_data = np.vstack((spike_times, neuron_ids)).T
spike_data_sorted = spike_data[spike_data[:, 0].argsort()]

# extract sorted spike times and neuron IDs:
sorted_spike_times = spike_data_sorted[:, 0]
sorted_neuron_ids = spike_data_sorted[:, 1]

# spike raster plot and histogram of spiking rate:
fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(5, 1)

# create the first subplot (3/4 of the figure)
ax1 = plt.subplot(gs[0:4, :])
ax1.scatter(sorted_spike_times, sorted_neuron_ids, s=9.0, color='mediumaquamarine', alpha=1.0)
plt.title(f"50 example excitatory neurons (out of {NE})\ng: {g}, $\\nu_{{ex}}/\\nu_{{th}}$: {nu_ex/nu_th}, delay: {delay}, $\eta$: {eta}, $\epsilon$: {epsilon}")
#ax1.set_xlabel("time [ms]")
ax1.set_xticks([])
ax1.set_ylabel("neuron ID")
ax1.set_xlim([0, simtime+5])
ax1.set_ylim([0, N_rec+1])
ax1.set_yticks(np.arange(0, N_rec+1, 10))

# create the second subplot (1/4 of the figure)
ax2 = plt.subplot(gs[4, :])
hist_binwidth = 5.0
t_bins = np.arange(np.amin(sorted_spike_times), np.amax(sorted_spike_times), hist_binwidth)
n, bins = np.histogram(sorted_spike_times, bins=t_bins)
heights = 1000 * n / (hist_binwidth * (N_rec))
ax2.bar(t_bins[:-1], heights, width=hist_binwidth, color='violet')
#ax2.set_title(f"histogram of spiking rate vs. time")
ax2.set_ylabel("firing rate\n[Hz]")
ax2.set_xlabel("time [ms]")
ax2.set_xlim([0, simtime+5])
ax2.set_ylim([0, 200])

plt.tight_layout()
plt.savefig(f"figures/brunel_network_raster_plot_g{g}_nu_ex_nu_th{nu_ex/nu_th}_delay{delay}.png", dpi=200)
plt.show()
# %% END