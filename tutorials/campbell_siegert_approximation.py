""" 
NEST simulation of the Campbell & Siegert approximation.

source: https://nest-simulator.readthedocs.io/en/stable/auto_examples/CampbellSiegert.html

modified by: Fabrizio Musacchio
date: Jun 26, 2024
"""
# %% IMPORTS
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.optimize import fmin
from scipy.special import erf
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
# %% APPROXIMATION OF THE MEAN FIRING RATE

# set the simulation time:
simtime = 20000  # (ms) duration of simulation

# define some units:
pF = 1e-12
ms = 1e-3
pA = 1e-12
mV = 1e-3

# set the parameters of the neurons and noise sources:
n_neurons = 10  # number of simulated neurons

weights = [0.1]  # (mV) psp amplitudes
rates = [10000.0]  # (1/s) rate of Poisson sources
# weights = [0.1, 0.1]    # (mV) psp amplitudes
# rates = [5000., 5000.]  # (1/s) rate of Poisson sources

C_m     = 250.0  # (pF) capacitance
E_L     = -70.0  # (mV) resting potential
I_e     = 0.0  # (nA) external current
V_reset = -70.0  # (mV) reset potential
V_th    = -55.0  # (mV) firing threshold
t_ref   = 2.0  # (ms) refractory period
tau_m   = 10.0  # (ms) membrane time constant
tau_syn_ex = 0.5  # (ms) excitatory synaptic time constant
tau_syn_in = 2.0  # (ms) inhibitory synaptic time constant

# estimate the mean and variance of the input current using the Campbell & Siegert approximation:
mu = 0.0
sigma2 = 0.0
J = []
assert len(weights) == len(rates)
for rate, weight in zip(rates, weights):
    if weight > 0:
        tau_syn = tau_syn_ex
    else:
        tau_syn = tau_syn_in

    # we define the form of a single PSP (post-synaptic potential), which allows us to match the
    # maximal value to or chosen weight:
    def psp(x):
        return -(
            (C_m * pF)
            / (tau_syn * ms)
            * (1 / (C_m * pF))
            * (np.exp(1) / (tau_syn * ms))
            * (
                ((-x * np.exp(-x / (tau_syn * ms))) / (1 / (tau_syn * ms) - 1 / (tau_m * ms)))
                + (np.exp(-x / (tau_m * ms)) - np.exp(-x / (tau_syn * ms)))
                / ((1 / (tau_syn * ms) - 1 / (tau_m * ms)) ** 2)
            )
        )

    min_result = fmin(psp, [0], full_output=1, disp=0)

    # we need to calculate the PSC amplitude (i.e., the weight we set in NEST)
    # from the PSP amplitude, that we have specified above:
    fudge = -1.0 / min_result[1]
    J.append(C_m * weight / (tau_syn) * fudge)

    # we now use Campbell's theorem to calculate mean and variance of the input 
    # due to the Poisson sources. The mean and variance add up for each Poisson source:
    mu += rate * (J[-1] * pA) * (tau_syn * ms) * np.exp(1) * (tau_m * ms) / (C_m * pF)
    sigma2 += (
        rate
        * (2 * tau_m * ms + tau_syn * ms)
        * (J[-1] * pA * tau_syn * ms * np.exp(1) * tau_m * ms / (2 * (C_m * pF) * (tau_m * ms + tau_syn * ms))) ** 2
    )

mu += E_L * mV          # add the resting potential and convert to mV
sigma = np.sqrt(sigma2) # convert the variance to standard deviation

# after calculating the mean and variance of the input current, we can now calculate the firing rate
# using Siegert's approximation:
num_iterations = 100
upper = (V_th * mV - mu) / (sigma * np.sqrt(2))
lower = (E_L * mV - mu) / (sigma * np.sqrt(2))
interval = (upper - lower) / num_iterations
tmpsum = 0.0
for cu in range(0, num_iterations + 1):
    u = lower + cu * interval
    f = np.exp(u**2) * (1 + erf(u))
    tmpsum += interval * np.sqrt(np.pi) * f
r = 1.0 / (t_ref * ms + tau_m * ms * tmpsum) # firing rate
# %% SIMULATING THE NEURONS AND EMPIRICAL VALIDATION
""" 
We now simulate neurons receiving Poisson spike trains as input, and compare the 
theoretical result to the empirical value.
"""
nest.ResetKernel()

# define the parameters of the neurons for NEST:
neurondict = {
    "V_th": V_th,
    "tau_m": tau_m,
    "tau_syn_ex": tau_syn_ex,
    "tau_syn_in": tau_syn_in,
    "C_m": C_m,
    "E_L": E_L,
    "t_ref": t_ref,
    "V_m": E_L,
    "V_reset": E_L,
}

# create the neurons, Poisson generators, voltmeter, and spike recorder:
neurons     = nest.Create("iaf_psc_alpha", n_neurons, params=neurondict) 
neuron_free = nest.Create("iaf_psc_alpha", params=dict(neurondict, V_th=1e12))
""" 
The neuron_free represents a neuron that is set up with a very high threshold potential, 
effectively making it a 'silent' neuron. This neuron is used primarily for recording the 
membrane potential without it spiking. 
"""
poissongen    = nest.Create("poisson_generator", len(rates), {"rate": rates})
voltmeter     = nest.Create("voltmeter", params={"interval": 0.1})
spikerecorder = nest.Create("spike_recorder")

# connect the nodes:
poissongen_n_synspec = {"weight": np.tile(J, ((n_neurons), 1)), "delay": 0.1}
nest.Connect(poissongen, neurons, syn_spec=poissongen_n_synspec)
nest.Connect(poissongen, neuron_free, syn_spec={"weight": [J]})
nest.Connect(voltmeter, neuron_free)
nest.Connect(neurons, spikerecorder)

# simulate the network:
nest.Simulate(simtime)

# extract the membrane potential of the silent neuron:
v_free = voltmeter.events["V_m"]
Nskip = 500 # we skip the first 500 ms of the simulation

# compare the theoretical and empirical values of the mean membrane potential, variance, and firing rate:
print(f"mean membrane potential (actual / calculated): {np.mean(v_free[Nskip:])} / {mu * 1000}")
print(f"variance (actual / calculated): {np.var(v_free[Nskip:])} / {sigma2 * 1e6}")
print(f"firing rate (actual / calculated): {spikerecorder.n_events / (n_neurons * simtime * ms)} / {r}")

# extract the spike times and neuron IDs:
spike_events = nest.GetStatus(spikerecorder, "events")[0]
spike_times = spike_events["times"]
neuron_ids = spike_events["senders"]

# combine the spike times and neuron IDs into a single array and sort by time:
spike_data = np.vstack((spike_times, neuron_ids)).T
spike_data_sorted = spike_data[spike_data[:, 0].argsort()]

# extract sorted spike times and neuron IDs:
sorted_spike_times = spike_data_sorted[:, 0]
sorted_neuron_ids = spike_data_sorted[:, 1]
# %% SOME PLOTS

# plot the membrane potential of the silent neuron:
plt.figure(figsize=(6, 4))
plt.plot(voltmeter.events["times"][Nskip:], v_free[Nskip:], label=f"membrane potential $V_m$")
plt.axhline(y=V_th, color='r', linestyle='--', label=f"threshold potential $V_{{th}}$")
plt.axhline(y=V_reset, color='g', linestyle='--', label=f"reset potential $V_{{reset}}$")
plt.xlabel("time [ms]")
plt.ylabel("membrane potential [mV]")
plt.ylim([-71, -51])
plt.legend()
plt.title("Membrane potential of silent neuron")
plt.tight_layout()
plt.savefig("figures/campbell_siegert_approximation_membrane_potential.png", dpi=300)
plt.show()


# plot the spike raster plot using NEST's plot function:
nest.raster_plot.from_device(spikerecorder, hist=True)

# spike raster plot and histogram of spiking rate ("manually" plotted):
fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(5, 1)

# create the first subplot (3/4 of the figure)
ax1 = plt.subplot(gs[0:4, :])
ax1.scatter(sorted_spike_times, sorted_neuron_ids, s=9.0, color='mediumaquamarine', alpha=1.0)
ax1.set_title(f"Spike raster plot and histogram of spiking rate")
#ax1.set_xlabel("time [ms]")
ax1.set_xticks([])
ax1.set_ylabel("neuron ID")
ax1.set_xlim([0, simtime])
ax1.set_ylim([0, n_neurons+1])
ax1.set_yticks(np.arange(0, n_neurons+1, 10))

# create the second subplot (1/4 of the figure)
ax2 = plt.subplot(gs[4, :])
hist_binwidth = 55.0
t_bins = np.arange(np.amin(sorted_spike_times), np.amax(sorted_spike_times), hist_binwidth)
n, bins = np.histogram(sorted_spike_times, bins=t_bins)
heights = 10000 * n / (hist_binwidth * (n_neurons))
ax2.bar(t_bins[:-1], heights, width=hist_binwidth, color='violet')
#ax2.set_title(f"histogram of spiking rate vs. time")
ax2.text(0.05, 0.95, f"calculated firing rate: {np.round(r,2)} Hz",
            color='black', fontsize=12, ha='left', va='center',
            transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='white', alpha=0.5))
ax2.text(0.05, 0.7, f"actual firing rate:        {np.round(spikerecorder.n_events/(n_neurons * simtime * ms),2)} Hz", 
         color='black', fontsize=12, ha='left', va='center', 
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='white', alpha=0.5))
ax2.set_ylabel("firing rate\n[Hz]")
ax2.set_xlabel("time [ms]")
ax2.set_xlim([0, simtime])

plt.tight_layout()
plt.savefig("figures/campbell_siegert_approximation_spike_raster_and_histogram.png", dpi=300)
plt.show()

print(f"Total number of spikes (NEST): {nest.GetStatus(spikerecorder, 'n_events')[0]}")
print(f"Total number of spikes (Custom): {len(sorted_spike_times)}")
# %% END