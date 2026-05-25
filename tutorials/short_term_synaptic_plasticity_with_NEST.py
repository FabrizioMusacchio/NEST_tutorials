""" 
NEST simulation of short-term depression and facilitation

source: https://nest-simulator.readthedocs.io/en/stable/auto_examples/iaf_tum_2000_short_term_depression.html
        https://nest-simulator.readthedocs.io/en/stable/auto_examples/iaf_tum_2000_short_term_facilitation.html

modified by: Fabrizio Musacchio
date: Jun 26, 2024
"""
# %% IMPORTS
import os
import matplotlib.pyplot as plt
import numpy as np
import nest

# set the verbosity of the NEST simulator:
nest.set_verbosity("M_WARNING")

# Set global properties for all plots:
plt.rcParams.update({'font.size': 12})
plt.rcParams["axes.spines.top"]    = False
plt.rcParams["axes.spines.bottom"] = False
plt.rcParams["axes.spines.left"]   = False
plt.rcParams["axes.spines.right"]  = False

# create a folder "figures" to save the plots (if it does not exist):
if not os.path.exists('figures'):
    os.makedirs('figures')
# %% MODEL SHORT-TERM DEPRESSION (STD)
nest.ResetKernel()
nest.resolution = 0.1  # simulation step size [ms]

T_sim = 1200.0  # simulation time [ms]

nest.ResetKernel()
nest.resolution = 0.1  # simulation step size [ms]

T_sim = 1200.0  # simulation time [ms]

tau_m = 40.0  # membrane time constant [ms]
R_m = 0.1  # membrane input resistance [GΩ]
C_m = tau_m / R_m  # membrane capacitance [pF]
V_th = 15.0  # threshold potential [mV]
V_reset = 0.0  # reset potential [mV]
t_ref = 2.0  # refractory period [ms]

stim_start = 50.0  # start time of DC input [ms]
stim_end = 1050.0  # end time of DC input [ms]
f = 20.0 / 1000.0  # frequency used in [2] [mHz]
dc_amp = V_th * C_m / tau_m / (1 - np.exp(-(1 / f - t_ref) / tau_m))  # DC amplitude [pA]

dc_gen = nest.Create("dc_generator", 1,
                     params={"amplitude": dc_amp, 
                             "start": stim_start, 
                             "stop": stim_end})

x = 1.0  # initial fraction of synaptic vesicles in the readily releasable pool
u = 0.0  # initial release probability of synaptic vesicles
U = 0.5  # fraction determining the increase in u with each spike
tau_psc = 3.0  # decay constant of PSCs (tau_inact in [2]) [ms]
tau_rec = 800.0  # recovery time from synaptic depression [ms]
tau_fac = 0.0  # time constant for facilitation (off) [ms]

neurons = nest.Create("iaf_tum_2000", 2, params={
                    "C_m": C_m,
                    "tau_m": tau_m,
                    "tau_syn_ex": tau_psc,
                    "tau_syn_in": tau_psc,
                    "V_th": V_th,
                    "V_reset": V_reset,
                    "E_L": V_reset,
                    "V_m": V_reset,
                    "t_ref": t_ref,
                    "U": U,
                    "tau_psc": tau_psc,
                    "tau_rec": tau_rec,
                    "tau_fac": tau_fac,
                    "x": x,
                    "u": u})

nest.Connect(dc_gen, neurons[0])

weight = 250.0  # synaptic weight [pA]
delay = 0.1  # synaptic delay [ms]

nest.Connect(neurons[0], neurons[1], syn_spec={
                "synapse_model": "static_synapse",
                "weight": weight,
                "delay": delay,
                "receptor_type": 1})

multimeter_std = nest.Create("multimeter", 
                             params={"interval": 1.0, 
                                     "record_from": ["V_m", 'I_syn_ex', 'I_syn_in']})
nest.Connect(multimeter_std, neurons[1])

nest.Simulate(T_sim)

# extract recordings from the multimeter:
times_STD = multimeter_std.get("events")["times"]
I_syn_ex_STD = multimeter_std.get("events")["I_syn_ex"]
I_syn_in_STD = multimeter_std.get("events")["I_syn_in"]
V_m_STD = multimeter_std.get("events")["V_m"]
# %% MODEL SHORT-TERM FACILITATION (STF)
nest.ResetKernel()
nest.resolution = 0.1  # simulation step size [ms]

dc_gen = nest.Create("dc_generator", 1,
                     params={"amplitude": dc_amp, 
                             "start": stim_start, 
                             "stop": stim_end})

x = 1.0  # initial fraction of synaptic vesicles in the readily releasable pool
u = 0.0  # initial release probability of synaptic vesicles
U = 0.03  # fraction determining the increase in u with each spike
tau_psc = 1.5  # decay constant of PSCs (tau_inact in [2]) [ms]
tau_rec = 130.0  # recovery time from synaptic depression [ms]
tau_fac = 530.0  # time constant for facilitation [ms]

neurons = nest.Create("iaf_tum_2000", 2,params={
                    "C_m": C_m,
                    "tau_m": tau_m,
                    "tau_syn_ex": tau_psc,
                    "tau_syn_in": tau_psc,
                    "V_th": V_th,
                    "V_reset": V_reset,
                    "E_L": V_reset,
                    "V_m": V_reset,
                    "t_ref": t_ref,
                    "U": U,
                    "tau_psc": tau_psc,
                    "tau_rec": tau_rec,
                    "tau_fac": tau_fac,
                    "x": x,
                    "u": u})

nest.Connect(dc_gen, neurons[0])

weight = 1540.0  # synaptic weight [pA]
delay = 0.1  # synaptic delay [ms]

nest.Connect(neurons[0], neurons[1], syn_spec={
                "synapse_model": "static_synapse",
                "weight": weight,
                "delay": delay,
                "receptor_type": 1})

multimeter_stf = nest.Create("multimeter", 
                             params={"interval": 1.0, 
                                     "record_from": ["V_m", 'I_syn_ex', 'I_syn_in']})
nest.Connect(multimeter_stf, neurons[1])

nest.Simulate(T_sim)

# extract recordings from the multimeter:
times_STF = multimeter_stf.get("events")["times"]
I_syn_ex_STF = multimeter_stf.get("events")["I_syn_ex"]
I_syn_in_STF = multimeter_stf.get("events")["I_syn_in"]
V_m_STF = multimeter_stf.get("events")["V_m"]


fig1, ax = plt.subplots(2, 1, sharex=True, figsize=(6.5, 5))

ax[0].plot(times_STD, V_m_STD, label="V_m (STD)")
ax[0].plot(times_STF, V_m_STF, label="V_m (STF)", alpha=0.5)
ax[0].legend()
ax[0].set_ylabel(f"membrane potential\n[mV]")

ax[1].plot(times_STD, I_syn_ex_STD, label="I_syn_ex (STD)")
#ax[1].plot(times_STD, I_syn_in_STD, label="I_syn_in (STD)")
ax[1].plot(times_STF, I_syn_ex_STF, label="I_syn_ex (STF)", alpha=0.5)
#ax[1].plot(times_STF, I_syn_in_STF, label="I_syn_in (STF)")
ax[1].legend()
ax[1].set_xlabel("time [ms]")
ax[1].set_ylabel(f"synaptic current\n[pA]")

plt.tight_layout()
plt.savefig("figures/short_term_depression_facilitation.png", dpi=200)
plt.show()

# %% END