#!/usr/bin/env python
# coding: utf-8

# # Formulas

# # Hamiltonian
# $$H = - J \sum_{\langle i j\rangle}s_i s_j - h \sum_i s_i$$
# # Mean Internal Energy
# $$\langle U \rangle = \frac{H}{4N}$$
# $$\langle U\rangle = \frac{1}{4N}\left(- J \sum_{\langle i j\rangle}^N s_i s_j - h \sum_i^N s_i\right)$$
# # Mean Magnetization
# $$\langle M \rangle = \frac{1}{N}\sum_i^N s_i$$
# # Absolute Mean Magnetization
# $$\langle |M| \rangle = \bigg|\frac{1}{N}\sum_i^N s_i\bigg|$$
# # Specific Heat per spin
# $$\langle C\rangle = \frac{1}{k_B T^2} (\langle U^2\rangle - \langle U\rangle^2)$$
# # Magnetic Susceptibility per spin
# $$\langle \chi\rangle = \frac{1}{k_B T}(\langle M^2\rangle - \langle M^2\rangle)$$

# # Code

# In[9]:


set_seed()

lattice_black = generate_lattice(lattice_n, lattice_m//2) # black
lattice_white = generate_lattice(lattice_n, lattice_m//2) # white
initial_lattice = combine_lattice(lattice_black, lattice_white).copy()
plot_ising(initial_lattice)

print(f"Lattice Size: {lattice_n}")
print(f"Eq steps: {eq_steps}")
for tt, temp in enumerate(T_desc):
    print(f"Starting at {tt}, temp: {temp}")
    inv_temp = 1/temp
    iT2=inv_temp*inv_temp
    
    # Equilibration
    for i in range(eq_steps):
        randvals = generate_array(lattice_n, lattice_m//2)
        mc_move(lattice_black, lattice_white, randvals, True, inv_temp)

        randvals = generate_array(lattice_n, lattice_m//2)
        mc_move(lattice_white, lattice_black, randvals, True, inv_temp)

    E1 = M1 = E2 = M2 = 0
    # MC Sweeps
    for i in range(mc_steps):
        randvals = generate_array(lattice_n, lattice_m//2)
        mc_move(lattice_black, lattice_white, randvals, True, inv_temp)

        randvals = generate_array(lattice_n, lattice_m//2)
        mc_move(lattice_white, lattice_black, randvals, False, inv_temp)

        Ene = calc_energy(lattice_black, lattice_white, True) + calc_energy(lattice_white, 
                                                                            lattice_black, False)
        Mag = calc_mag(lattice_black) + calc_mag(lattice_white)

        E1 += Ene
        M1 += Mag
        M2 += Mag*Mag 
        E2 += Ene*Ene

    E[tt] = n1*E1
    M[tt] = n1*M1
    C[tt] = (n1*E2 - n2*E1*E1)*iT2
    X[tt] = (n1*M2 - n2*M1*M1)*inv_temp

lattice = combine_lattice(lattice_black, lattice_white)
plot_ising(lattice)

# # Packages

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import math

# # Functions and Variables

# In[8]:


# Lattice Size
lattice_n = 256
lattice_m = lattice_n

# Random Seeds
seed = 1234
common_seed = 1234
use_common_seed = True

#Steps
eq_steps = lattice_n*lattice_m
mc_steps = 100_000

# Temperature
nt_ = 18
T_asc = np.append(np.flip(3.469 - np.geomspace(1.2, 2.269, nt_, endpoint=False)), np.geomspace(2.269, 2.8, nt_//2))
T_desc = np.flip(T_asc)
nt = T_desc.shape
#Observables
E,M,C = np.zeros(nt), np.zeros(nt), np.zeros(nt)
X = np.zeros(nt, dtype=np.float128)
n1, n2  = 1.0/(mc_steps*lattice_n*lattice_m), 1.0/(mc_steps*mc_steps*lattice_n*lattice_m)

# Parameters
J=1
h=0
kb=1.0

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
            spin = lattice[i, j]
            delta_E = -2*spin*(J*nn_sum+h)
            if delta_E >= 0:
                lattice[i, j] = -spin
            elif randvals[i, j] < math.exp(delta_E*inv_temp):
                lattice[i, j] = -spin
    return lattice

@njit(parallel=True)
def calc_energy(lattice, op_lattice, is_black):
    n,m = lattice.shape
    energy = 0
    for i in prange(n):
        for j in prange(m):
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
            
            spin = lattice[i,j]
            
            energy += ((-J*nn_sum*spin)-(h*spin))
    return energy/2.

def calc_mag(lattice):
    return np.sum(lattice, dtype=np.int64)


def set_seed():
    if use_common_seed:
        np.random.seed(common_seed)
    else:
        np.random.seed(seed)

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

# # Plot

# In[11]:


np.savetxt(f'E_{lattice_n}', E)
np.savetxt(f'M_{lattice_n}', M)
np.savetxt(f'C_{lattice_n}', C)
np.savetxt(f'X_{lattice_n}', X)

# In[10]:


f = plt.figure(figsize=(18, 10)); # plot the calculated values    

sp =  f.add_subplot(2, 2, 1 );
plt.scatter(T_desc, E, marker='o', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20);
plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');

sp =  f.add_subplot(2, 2, 2 );
plt.scatter(T_desc, abs(M), marker='o', color='RoyalBlue')
plt.xlabel("Temperature (T)", fontsize=20); 
plt.ylabel("Magnetization ", fontsize=20);   plt.axis('tight');

sp =  f.add_subplot(2, 2, 3 );
plt.scatter(T_desc, C, marker='o', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20);  
plt.ylabel("Specific Heat ", fontsize=20);   plt.axis('tight');   

sp =  f.add_subplot(2, 2, 4 );
plt.scatter(T_desc, X, marker='o', color='RoyalBlue')
plt.xlabel("Temperature (T)", fontsize=20); 
plt.ylabel("Susceptibility", fontsize=20);   plt.axis('tight');
