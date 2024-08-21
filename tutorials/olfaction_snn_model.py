"""
NEST simulation of a SNN simulating the olfactory signalling pathway from
ORNs (olfactory receptor neurons) to glumeruli in the olfactory bulb. Glumeruli
are biological structures in the olfactory bulb where the axons of ORNs 
synapse onto the dendrites of mitral and tufted cells. The ORNs are stimulated
by odorants and send their signals to the glumeruli, where the mitral and tufted
cells are activated. The mitral and tufted cells then send their signals to
higher brain areas for further processing.

Here, we simulated ORNs receiving distinct odorant signals and sending them to
the glumeruli. The glumeroli are modeled as populations of excitatory (glumeroli cells) 
and inhibitory neurons (peri-glumerular cells) that receive the signals from the ORNs. 
We implement learning in the system by using the spike-timing-dependent plasticity (STDP),
so that the connections between the ORNs and the glumeruli are modified based on the
stimulation, thus, timing of the spikes.

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
nest.ResetKernel()
n_ORNs = 100  # total number of olfactory receptor neurons (ORNs)
n_glomeruli = 10  # total number of (excitatory) glomeruli
n_periglomerular = 20  # total number of peri-glomerular (inhibitory) cells

# create neuron populations:
ORN_population = nest.Create("iaf_psc_alpha", n_ORNs) 
""" alternatively, use a "poisson_generator" for the ORN or a simpler parrot_neuron model """
glomeruli_population = nest.Create("gif_psc_exp", n_glomeruli)
periglomerular_population = nest.Create("gif_psc_exp", n_periglomerular)
""" the "gif_psc_exp" model is a generic integrate-and-fire neuron with exponential postsynaptic currents """

# set up odorant signals for the ORNs:
odor1 = nest.Create("dc_generator")
odor2 = nest.Create("dc_generator")
odor3 = nest.Create("dc_generator")

# define the start and stop times for multiple trials:
num_trials = 5
trial_duration = 100.0 # duration of each trial in ms
inter_trial_interval = 400.0 # interval between trials in ms

# calculate odorant stimulation times:
odor1_times = [(i * (trial_duration + inter_trial_interval), i * (trial_duration + inter_trial_interval) + trial_duration) for i in range(num_trials)]
odor2_times = [(i * (trial_duration + inter_trial_interval) + 50, i * (trial_duration + inter_trial_interval) + 50 + trial_duration) for i in range(num_trials)]
odor3_times = [(i * (trial_duration + inter_trial_interval) + 100, i * (trial_duration + inter_trial_interval) + 100 + trial_duration) for i in range(num_trials)]

# flatten the list of times:
odor1_start_stop = [time for times in odor1_times for time in times]
odor2_start_stop = [time for times in odor2_times for time in times]
odor3_start_stop = [time for times in odor3_times for time in times]

# and set the status for the generators:
nest.SetStatus(odor1, {"amplitude": 500.0, "start": odor1_start_stop[0], "stop": odor1_start_stop[-1]})
nest.SetStatus(odor2, {"amplitude": 500.0, "start": odor2_start_stop[0], "stop": odor2_start_stop[-1]})
nest.SetStatus(odor3, {"amplitude": 500.0, "start": odor3_start_stop[0], "stop": odor3_start_stop[-1]})

# select subsets of ORNs to connect to each odor:
odor1_ORNs = ORN_population[:n_ORNs // 3]
odor2_ORNs = ORN_population[n_ORNs // 3: 2 * n_ORNs // 3]
odor3_ORNs = ORN_population[2 * n_ORNs // 3:]
""" 
We do this, so that each odorant is connected to a subset of ORNs to simulate 
specific groups of ORNs being activated by different odors. This segmentation 
reflects the biological reality where different ORNs respond to different sets 
of odor molecules.
"""

# connect the stimuli to the ORN subsets:
nest.Connect(odor1, odor1_ORNs)
nest.Connect(odor2, odor2_ORNs)
nest.Connect(odor3, odor3_ORNs)


# define the synaptic plasticity model (STDP):
stdp_synapse_dict = {
    "synapse_model": "stdp_synapse",
    "weight":    1.0, # initial weight of the synapse; positive for excitation
    #"delay":     1.0, # synaptic delay in ms
    "tau_plus": 20.0, # time constant of the STDP window for LTP (long-term potentiation)
    "Wmax":     10.0, # maximum weight of the synapse
    "mu_plus":   0.5, # learning rate for LTP
    "mu_minus":  0.5 # learning rate for LTD (long-term depression)
}

# connect all ORNs to all glomeruli with STDP:
nest.Connect(ORN_population, glomeruli_population, syn_spec="stdp_synapse")


# define inhibitory synapse for periglomerular cells for inhibition:
inhibitory_synapse = {
    "synapse_model": "static_synapse",
    "weight": -2.0,  # negative weight for inhibition
    "delay":   1.0   # synaptic delay in ms
}

# connect periglomerular cells to glomeruli for lateral inhibition
nest.Connect(periglomerular_population, glomeruli_population, syn_spec=inhibitory_synapse)

# connect ORNs to periglomerular cells to provide direct excitatory input:
ORN_to_periglomerular_syn_spec = {"weight": 1.0, "delay": 1.0}
nest.Connect(ORN_population, periglomerular_population, syn_spec=ORN_to_periglomerular_syn_spec)

# create spike detectors:
ORN_spike_detector = nest.Create("spike_recorder")
glomeruli_spike_detector = nest.Create("spike_recorder")
periglomerular_spike_detector = nest.Create("spike_recorder")

# Connect spike detectors to the neuron populations
nest.Connect(ORN_population, ORN_spike_detector)
nest.Connect(glomeruli_population, glomeruli_spike_detector)
nest.Connect(periglomerular_population, periglomerular_spike_detector)

# simulate for the total duration covering all trials:
total_simulation_time = num_trials * (trial_duration + inter_trial_interval)
nest.Simulate(total_simulation_time)

def plot_spikes(spike_recorder, title):
    events = nest.GetStatus(spike_recorder, "events")[0]
    plt.figure(figsize=(10, 4))
    plt.plot(events["times"], events["senders"], 'k.', markersize=2)
    plt.title(title)
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron ID")
    plt.show()

# Plot spike rasters
plot_spikes(ORN_spike_detector, "ORN Population Spike Raster")
plot_spikes(glomeruli_spike_detector, "Glomeruli Population Spike Raster")
plot_spikes(periglomerular_spike_detector, "Periglomerular Population Spike Raster")


# %% END
