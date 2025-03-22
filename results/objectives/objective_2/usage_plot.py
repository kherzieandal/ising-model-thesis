#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# In[3]:


serial_cpu_util = np.loadtxt('serial_cpu_util.txt')
parallel_cpu_util = np.loadtxt('parallel_cpu_util.txt')
gpu_cpu_util = np.loadtxt('gpu_cpu_util.txt')
gpu_util = np.loadtxt('gpu_util_percent.txt')

# In[4]:


plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#1abc9c", "#2ecc71", "#e84393", "#80ff00", "#2ecc71", "#f1c40f",
                                                   "#f1c40f", "#e67e22", "#e74c3c", "#95a5a6", "#d35400", "#8e44ad",
                                                   "#8e44ad"]) 
plt.figure(figsize=[6.4*2, 4.8])
plt.plot(range(1, 101), serial_cpu_util, label=['CPU ' + str(i) for i in range(12)])
plt.ylim(0, 102)
plt.xlim(1, 100)
plt.ylabel('CPU Usage [\%]', fontsize=14)
plt.xlabel('No. of MC sweep', fontsize=14)
plt.legend(loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.18))
plt.grid()
plt.savefig('serial_cpu_util', dpi=300, bbox_inches='tight')
plt.show()

# In[83]:


plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#1abc9c", "#2ecc71", "#e84393", "#80ff00", "#2ecc71", "#f1c40f",
                                                   "#f1c40f", "#e67e22", "#e74c3c", "#95a5a6", "#d35400", "#8e44ad",
                                                   "#8e44ad"]) 
plt.figure(figsize=[6.4*2, 4.8])
plt.plot(range(1, 101), parallel_cpu_util, label=['CPU ' + str(i) for i in range(12)])
plt.ylim(0, 102)
plt.xlim(1, 100)
plt.ylabel('CPU Usage [\%]', fontsize=14)
plt.xlabel('No. of MC sweep', fontsize=14)
plt.legend(loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.18))
plt.grid()
plt.savefig('parallel_cpu_util', dpi=300, bbox_inches='tight')
plt.show()

# In[ ]:




# In[28]:


gpu_cpu_util

# In[85]:


plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#1abc9c", "#2ecc71", "#e84393", "#80ff00", "#2ecc71", "#f1c40f",
                                                   "#f1c40f", "#e67e22", "#e74c3c", "#95a5a6", "#d35400", "#8e44ad",
                                                   "#8e44ad"]) 
plt.figure(figsize=[6.4*2, 4.8])
plt.plot(range(1, 101), gpu_cpu_util, label=['CPU ' + str(i) for i in range(12)])
plt.ylim(0, 105)
plt.xlim(1, 100)
plt.ylabel('CPU Usage [\%]', fontsize=14)
plt.xlabel('No. of MC sweep', fontsize=14)
plt.legend(loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.18))
plt.grid()
plt.savefig('gpu_cpu_util', dpi=300, bbox_inches='tight')
plt.show()

# In[93]:


plt.figure(figsize=[6.4*1.5, 4.8])
plt.plot(range(1, 101), gpu_util)
plt.ylim(0, 105)
plt.xlim(1, 100)
plt.ylabel('GPU Usage [\%]', fontsize=14)
plt.xlabel('No. of MC sweep', fontsize=14)
plt.grid()
plt.savefig('gpu_util', dpi=300, bbox_inches='tight')
plt.show()
