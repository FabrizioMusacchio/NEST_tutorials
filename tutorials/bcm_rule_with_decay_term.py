""" 
A simple example of the BCM learning rule.

author: Fabrizio Musacchio
date: Jul 15, 2024
"""
# %% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
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
# for reproducibility:
np.random.seed(1)

# define parameters:
eta = 0.01  # learning rate
tau = 100.0  # time constant for averaging postsynaptic activity
epsilon = 0.001  # decay rate
simulation_time = 500  # total simulation time in ms
time_step = 1  # time step for the simulation in ms
p = 2  # exponent for the sliding threshold function

# initialize synaptic weights and inputs:
w = np.array([0.5, 0.5])  # Initial synaptic weights
x1 = np.random.rand(simulation_time)  # Presynaptic input 1
x2 = np.random.rand(simulation_time)  # Presynaptic input 2
inputs = np.vstack((x1, x2))

# initialize variables for storing results:
y = np.zeros(simulation_time)
theta_M = np.zeros(simulation_time)
avg_y = 0  # initial average postsynaptic activity
w_history = np.zeros((simulation_time, 2))  # to store synaptic weights over time

# simulation loop:
for t in range(simulation_time):
    # compute postsynaptic activity:
    y[t] = np.dot(w, inputs[:, t])
    
    # update average postsynaptic activity:
    avg_y = avg_y + (y[t] - avg_y) / tau
    
    # update the sliding threshold:
    theta_M[t] = avg_y ** p
    
    # update synaptic weights according to the BCM rule:
    delta_w = eta * y[t] * (y[t] - theta_M[t]) * inputs[:, t] - epsilon * w
    w += delta_w
    
    # ensure weights remain within a reasonable range:
    w = np.clip(w, 0, 1)
    
    # store synaptic weights:
    w_history[t] = w

# plotting the results:
plt.figure(figsize=(6, 7))

# Plot synaptic weights
plt.subplot(2, 1, 1)
plt.plot(w_history[:, 0], label='weight 1')
plt.plot(w_history[:, 1], label='weight 2')
plt.xlabel('time [ms]')
plt.ylabel('synaptic weight')
plt.title('Evolution of synaptic weights')
plt.legend()

# Plot postsynaptic activity and sliding threshold
plt.subplot(2, 1, 2)
plt.plot(y, label='postsynaptic activity')
plt.plot(theta_M, label='sliding threshold', linestyle='--')
plt.xlabel('time [ms]')
plt.ylabel('activity / threshold')
plt.title('Postsynaptic activity and sliding threshold')
plt.legend()

plt.tight_layout()
plt.savefig('figures/bcm_rule_example_with_decay_term.png', dpi=300)
plt.show()
# %% END