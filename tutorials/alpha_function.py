""" 
A simple script to visualize the alpha function for synaptic current.

author: Fabrizio Musacchio
date: Jul 12, 2024
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
def alpha_function(t, tau):
    return (t / tau) * np.exp(1 - t / tau) * (t > 0)

t = np.linspace(0, 10, 1000)
tau = 2
alpha = alpha_function(t, tau)

fig=plt.figure(figsize=(4.5,3.5))
plt.plot(t, alpha, lw=2)
plt.title('Alpha function for synaptic current')
plt.xlabel('time (ms)')
plt.ylabel('synaptic current (normalized)')
plt.tight_layout()
plt.savefig('figures/alpha_function.png', dpi=200)
plt.show()
# %% END