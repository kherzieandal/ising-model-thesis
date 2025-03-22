#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# In[80]:


nt = 10
Tn = np.linspace(1, 3.4, nt)
T = Tn[::-1]
T_C = 2/(np.log(1 + np.sqrt(2)))

M_16 = np.loadtxt("M_16.txt")
M_64 = np.loadtxt("M_64.txt")
M_128 = np.loadtxt("M_128.txt")
M_256 = np.loadtxt("M_256.txt")
M_512 = np.loadtxt("M_512.txt")

# In[6]:




# In[22]:


T_con = np.linspace(1, 3.4, 1000) 
T_2 = T_con[np.where(T_con < T_C)]

# In[84]:


# # Analyical Solution
T_con = np.linspace(1, 3.4, 1000) 
T_2 = T_con[np.where(T_con < T_C)]
M_analytic = np.zeros(1000)
M_analytic[np.where(T_con < T_C)] = (1-(np.sinh(2/T_2))**(-4))**(1/8)

# plt.plot(T, M)
plt.plot(T_con, M_analytic, label='Onsager Solution')

# Lattices:
# N = 16^2, MCS = 1K
plt.plot(T, M_16,label='$N=16^2$', marker='o', linewidth=0, fillstyle='none')
plt.xlabel("Temperature", fontsize=20); 
plt.ylabel("Magnetization", fontsize=20);   plt.axis('tight');
plt.legend()

# N = 64^2, MCS = 16K
plt.plot(T, M_64,label='$N=64^2$', marker='^', linewidth=0, fillstyle='none')
plt.xlabel("Temperature", fontsize=20); 
plt.ylabel("Magnetization", fontsize=20);   plt.axis('tight');
plt.legend()

# N = 128^2, MCS=64K
plt.plot(T, M_128,label='$N=128^2$', marker='s', linewidth=0, fillstyle='none')
plt.xlabel("Temperature", fontsize=20); 
plt.ylabel("Magnetization", fontsize=20);   plt.axis('tight');
plt.legend()

# N = 256^2, MCS=262K
plt.plot(T, M_256,label='$N=256^2$', marker='D', linewidth=0, fillstyle='none')
plt.xlabel("Temperature", fontsize=20); 
plt.ylabel("Magnetization", fontsize=20);   plt.axis('tight');
plt.legend()


# N = 512^2, MCS = 1M
plt.plot(T, M_512,label='$N=512^2$', marker='x', linewidth=0, fillstyle='none')
plt.xlabel("Temperature", fontsize=20); 
plt.ylabel("Magnetization", fontsize=20);   plt.axis('tight');
plt.legend()
plt.savefig('Magnetization', dpi=300, bbox_inches='tight')

