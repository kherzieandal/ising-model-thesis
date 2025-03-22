#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# In[2]:


nt_ = 18
T_asc = np.append(np.flip(3.469 - np.geomspace(1.2, 2.269, nt_, endpoint=False)), np.geomspace(2.269, 2.8, nt_//2))
T_desc = np.flip(T_asc)
nt = T_desc.shape[0]

size_list = [64, 128, 256, 512]

E, M, C, X = dict(), dict(), dict(), dict()
for size in size_list:
    E[size] = np.loadtxt(f'data/E_{size}')
    M[size] = np.loadtxt(f'data/M_{size}')
    C[size] = np.loadtxt(f'data/C_{size}')
    X[size] = np.loadtxt(f'data/X_{size}')

# In[3]:


df_ene = pd.read_csv('data/energy_exact.csv', header=None)
x_ene = df_ene[0].loc[(df_ene[0] > (min(T_desc)-0.1)) & (df_ene[0] < max(T_desc)+0.1)]
y_ene = df_ene[1][x_ene.index]

# In[13]:


plt.rcParams.update({'font.size': 12})
plt.plot(x_ene, y_ene, label='Exact', zorder= 10, color='slategray')
for size in size_list:
    plt.scatter(T_desc, E[size], marker='o', label=size)

plt.legend()
plt.xlabel('Temperature (T)')
plt.ylabel('Internal Energy (U)')
plt.title('Internal Energy vs Temperature')
plt.savefig('energy', bbox_inches='tight', dpi=300)
plt.show()

# In[9]:


T_C = 2/(np.log(1 + np.sqrt(2)))
nt = 1000000
# # Analyical Solution
T_con = np.linspace(1.269, 2.869, nt) 
T_2 = T_con[np.where(T_con < T_C)]
M_analytic = np.zeros(nt)
M_analytic[np.where(T_con < T_C)] = (1-(np.sinh(2/T_2))**(-4))**(1/8)

# plt.plot(T, M)
plt.plot(T_con, M_analytic, label='Exact', color='slategray')

for size in size_list:
    plt.scatter(T_desc, abs(M[size]), label=size)
plt.legend()
plt.xlabel('Temperature (T)')
plt.ylabel('Magnetization (|M|)')
plt.title('Magnetization vs Temperature')
plt.savefig('magnetization', bbox_inches='tight', dpi=300)
plt.show()

# In[6]:


T_con_left = T_con[T_con < T_C]
T_con_right = T_con[T_con > T_C]

# In[14]:


y_heat_left = -(2/np.pi)*(2/T_C)**2 * (np.log(1-(T_con_left/T_C))+0.66)
y_heat_right = -(2/np.pi)*(2/T_C)**2 * (np.log(-1+(T_con_right/T_C))+0.53)
plt.plot(T_con_left, y_heat_left, label='Exact', zorder= 10, color='slategray')
plt.plot(T_con_right, y_heat_right, zorder= 10, color='slategray')
for size in size_list:
    plt.scatter(T_desc, C[size], marker='o', label=size)
plt.xlabel('Temperature (T)')
plt.ylabel('Specific Heat (C)')
plt.title('Specific Heat vs Temperature')
plt.legend()
plt.savefig('heat', bbox_inches='tight', dpi=300)
plt.show()

# In[12]:


for size in size_list:
    plt.scatter(T_desc, X[size], marker='o', label=size)
plt.xlabel('Temperature (T)')
plt.ylabel('Magnetic Susceptibility (X)')
plt.legend()
plt.title('Magnetic Susceptibility vs Temperature')
plt.savefig('susceptibility', bbox_inches='tight', dpi=300)
plt.show()
