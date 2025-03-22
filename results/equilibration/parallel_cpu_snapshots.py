#!/usr/bin/env python
# coding: utf-8

# # Metropolis Algorithm ($h=0$)
# ## Algorithm
# 1. Choose an initial state $S(0) = (S_1,\ldots,S_N)$
# 2. Choose an $i$ (randomly or sequentially) and calculate $\Delta E=-2S_ih_i$
# 3. If $\Delta E\geq 0$, then flip the spin, $S_i\to -S_i$. If $\Delta E< 0$, draw a uniformly distributed random number $r \in [0,1]$. If $r < e^{\Delta E/kbT}$, flip the spin, $S_i\to -S_i$, otherwise take the old configuration into account once more.
# 4. Iterate 2 and 3.
# 
# 

# $$H(S)=-JS_i\sum_{j\in N(i)}S_j-hS_i+remainder$$
# $$\Delta E=-2S_i\left(J\sum_{j\in N(i)}S_j+h\right)=-2S_ih_i$$

# # Equilibration Plot

# Temperatures
# - 1.2
# - 2.269
# - 2.8

# In[94]:


T = 2.269

inv_temp = 1/(T)

set_seed()

lattice_black = generate_lattice(lattice_n, lattice_m//2) # black
lattice_white = generate_lattice(lattice_n, lattice_m//2) # white

# INITIAL LATTICE
initial_lattice_1 = combine_lattice(lattice_black, lattice_white).copy()
print('Starting equlibration...')

for mc_step in range(mc_steps):
    randvals = generate_array(lattice_n, lattice_m//2)
    mc_move(lattice_black, lattice_white, randvals, True, inv_temp)

    randvals = generate_array(lattice_n, lattice_m//2)
    mc_move(lattice_white, lattice_black, randvals, True, inv_temp)

    
print('Starting simulation...')
for mc_step in range(mc_steps):
    randvals = generate_array(lattice_n, lattice_m//2)
    mc_move(lattice_black, lattice_white, randvals, True, inv_temp)

    randvals = generate_array(lattice_n, lattice_m//2)
    mc_move(lattice_white, lattice_black, randvals, True, inv_temp)

    
# FINAL LATTICE:
lattice_2 = combine_lattice(lattice_black, lattice_white).copy()
    
# plot_ising(initial_lattice)
plot_ising(initial_lattice_1)
plot_ising(lattice_2)

# In[102]:


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

im1 = axes[0, 0].imshow(initial_lattice_1, vmin=-1, vmax=1, cmap='gray')
axes[0, 0].axis('tight')
axes[0, 0].title.set_text('(a)')

im2 = axes[0, 1].imshow(lattice_1, vmin=-1, vmax=1, cmap='gray')
axes[0, 1].axis('tight')
axes[0, 1].title.set_text('(b)')

im3 = axes[1, 0].imshow(lattice_2, vmin=-1, vmax=1, cmap='gray')
axes[1, 0].axis('tight')
axes[1, 0].title.set_text('(c)')

im4 = axes[1, 1].imshow(lattice_3, vmin=-1, vmax=1, cmap='gray')
axes[1, 1].axis('tight')
axes[1, 1].title.set_text('(d)')

fig.colorbar(im1, ticks=[-1, 1])
fig.colorbar(im2, ticks=[-1, 1])
fig.colorbar(im3, ticks=[-1, 1])
fig.colorbar(im4, ticks=[-1, 1])

fig.savefig('images/snapshots', dpi=300, bbox_inches='tight')

# # Modules

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from time import time
import math

# # Functions & Variables

# In[89]:


# Set constants
TCRIT = 2.26918531421 # critical temperature

############ 
alpha = 0.1
lattice_n = 256
lattice_m = lattice_n
seed = 2
common_seed = 1234
use_common_seed = True
eq_steps = lattice_n*lattice_m
mc_steps = 100_000
J=1
h=0
kb=1.0

# Equilibration variables
M_equi = np.zeros(mc_steps)
E_equi = np.zeros(mc_steps)

def set_seed():
    np.random.seed(seed)

def generate_lattice(N, M):
    return np.random.choice(np.array([-1,1], dtype=np.int8),size=(N,M))

def generate_array(N, M):
    return np.random.rand(N, M)

@njit(parallel=True)
def mc_move(lattice, op_lattice, randvals, is_black, inv_temp):    
    '''Monte Carlo move using Metropolis algorithm '''
    n,m = lattice.shape
    for i in prange(n):
        for j in prange(m):
            # Set stencil indices with periodicity
            ipp = (i + 1) if (i + 1) < n else 0
            jpp = (j + 1) if (j + 1) < m else 0
            inn = (i - 1) if (i - 1) >= 0 else (n - 1)
            jnn = (j - 1) if (j - 1) >= 0 else (m - 1)  
           
            # Select off-column index based on color and row index parity
            if (is_black):
                joff = jpp if (i % 2) else jnn
            else:
                joff = jnn if (i % 2) else jpp
                
            # Compute sum of nearest neighbor spins
            nn_sum = op_lattice[inn, j] + op_lattice[i, j] + op_lattice[ipp, j] + op_lattice[i, joff]
            
            # Compute sum of nearest neighbor spins (taking values from neighboring
            spin_i = lattice[i, j]
            deltaE = -2* spin_i*(J*nn_sum+h)
            if deltaE >= 0:
                lattice[i, j] = -spin_i
            elif randvals[i, j] < math.exp(deltaE*inv_temp):
                lattice[i, j] = -spin_i
    return lattice

@njit(parallel=True)
def calc_energy(lattice):
    energy = 0
    n = lattice.shape[0]
    m = lattice.shape[1]
    for i in prange(n):
        for j in prange(m):
            # Periodicity
            ipp = (i + 1) if (i + 1) < n else 0
            jpp = (j + 1) if (j + 1) < m else 0
            inn = (i - 1) if (i - 1) >= 0 else (n - 1)
            jnn = (j - 1) if (j - 1) >= 0 else (m - 1)  

            spin = lattice[i,j]
            nb = lattice[ipp, j] + lattice[i,jpp] + lattice[inn, j] + lattice[i,jnn]
            energy += (-2*spin*(J*nb+h))
    return energy/4.

@njit(parallel=True)
def calc_mag(lattice):
    '''Magnetization of a given configuration'''
    return np.sum(lattice)


@njit(parallel=True)
def combine_lattice(lattice_black, lattice_white):
    lattice = np.zeros((lattice_n, lattice_m), dtype=np.int8)
    for i in prange(lattice_n):
        for j in prange(lattice_m // 2):
            if (i % 2):
                lattice[i, 2*j+1] = lattice_black[i, j]
                lattice[i, 2*j] = lattice_white[i, j]
            else:
                lattice[i, 2*j] = lattice_black[i, j]
                lattice[i, 2*j+1] = lattice_white[i, j]
    return lattice
        
def plot_ising(lattice):
    plt.imshow(lattice, cmap='gray', vmin=-1, vmax=1)
    plt.colorbar()
    plt.show()
