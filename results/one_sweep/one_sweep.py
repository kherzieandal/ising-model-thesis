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

# ## Modules

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import time
import csv

# ## Functions

# In[17]:


# Set constants
TCRIT = 2.26918531421 # critical temperature

############ 
alpha = 0.1
lattice_n = 256
lattice_m = lattice_n
seed = 1234
common_seed = 1234
use_common_seed = True
eq_steps = 100
mc_steps = 1000
J=1
h=0

#----------------------------------------------------------------------
##  BLOCK OF FUNCTIONS USED IN THE MAIN CODE
#----------------------------------------------------------------------
def initial_state(N, M):   
    state = np.random.choice(np.array([-1,1]),size=(N,M))
    return state


def mcmove(lattice, inv_T):    
    '''Monte Carlo move using Metropolis algorithm '''
    n = lattice.shape[0]
    m = lattice.shape[1]
    for i in range(n):
        for j in range(m):
            # Periodicity for neighbors out of index
            ipp = (i + 1) if (i + 1) < n else 0
            jpp = (j + 1) if (j + 1) < m else 0
            inn = (i - 1) if (i - 1) >= 0 else (n - 1)
            jnn = (j - 1) if (j - 1) >= 0 else (m - 1)  
            
            # Calculate neighbors
            nb = lattice[ipp,j] + lattice[i,jpp] + lattice[inn,j] + lattice[i,jnn]
            
            # Compute energy difference
            spin =  lattice[i, j]
            deltaE = -2*spin*(J*nb + h)
            if deltaE >= 0:
                lattice[i, j] = -spin
            elif np.random.rand() < np.exp(deltaE*inv_T):
                lattice[i, j] = -spin
    return lattice


def calcEnergy(lattice, J=1, h=0):
    '''Energy of a given configuration'''
    energy = 0
    for i in range(L):
        for j in range(L):
            S = lattice[i,j]
            nb = lattice[(i+1)%L, j] + lattice[i,(j+1)%L] + lattice[(i-1)%L, j] + lattice[i,(j-1)%L]
            energy += ((-J*nb*S)-(h*nb))
    return energy/2.


def calcMag(lattice):
    '''Magnetization of a given configuration'''
    mag = np.sum(lattice)/(lattice.shape[0]*lattice.shape[1])
    return mag

def plot_ising(lattice, colorbar=False):
    # plt.figure(figsize=(12, 9))
    plt.imshow(lattice, cmap='gray', vmin=-1, vmax=1) 
    plt.colorbar(ticks=[-1, 1])

# ## Simulation

# In[20]:


temp = 0.1

# Set seed
if use_common_seed:
    np.random.seed(common_seed)
else:
    np.random.seed(seed)

inv_T = 1/(temp)


lattice = initial_state(lattice_n, lattice_m)
initial_lattice = lattice.copy()

# Start
t0 = time.time()
for i in range(1):
    mcmove(lattice, inv_T)

t1 = time.time()
t = t1 - t0

m_global = calcMag(lattice)


# In[24]:


plot_ising(initial_lattice)
plt.savefig('initial', dpi=300, bbox_inches='tight')
plt.show()
plot_ising(lattice)
plt.savefig('one_sweep', dpi=300, bbox_inches='tight')
