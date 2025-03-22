#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# In[2]:


nt = 10
Tn = np.linspace(1, 3.4, nt)
T = Tn[::-1]
T_C = 2/(np.log(1 + np.sqrt(2)))


# In[18]:


# # Analyical Solution
T_con = np.linspace(1, 3.4, 1000) 
T_2 = T_con[np.where(T_con < T_C)]
M_analytic = np.zeros(1000)
M_analytic[np.where(T_con < T_C)] = (1-(np.sinh(2/T_2))**(-4))**(1/8)

# plt.plot(T, M)
plt.plot(T_con, M_analytic, label='Onsager Solution')
plt.axvline(x = T_C, label = '$T_c= 2.269$', linestyle='--', color='orange')
plt.xlabel("Temperature (T)", fontsize=20); 
plt.ylabel("Magnetization (|M|)", fontsize=20);   plt.axis('tight');
plt.legend()
plt.savefig('magnetization_analytic', dpi=300, bbox_inches='tight')
