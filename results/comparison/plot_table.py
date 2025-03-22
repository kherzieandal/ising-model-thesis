#!/usr/bin/env python
# coding: utf-8

# # Combined

# In[11]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 8))

plt.rc('axes', titlesize=28)        # Controls Axes Title
plt.rc('axes', labelsize=28)        # Controls Axes Labels
plt.rc('xtick', labelsize=24)       # Controls x Tick Labels
plt.rc('ytick', labelsize=24)       # Controls y Tick Labels
plt.rc('legend', fontsize=24)       # Controls Legend Font
plt.rc('figure', titlesize=24)

ax1.errorbar(python_interpreted.keys(), python_interpreted.values(), yerr=list(python_interpreted_err.values()),
            marker='o', label='Python (Interpreted)', markersize=12, elinewidth=4)
ax1.errorbar(python_compiled.keys(), python_compiled.values(), yerr=list(python_compiled_err.values()),
            marker='o', label='Python (Compiled)', markersize=12, elinewidth=4)
ax1.errorbar(parallel_cpu_low.keys(), parallel_cpu_low.values(), yerr=list(parallel_cpu_low_err.values()),
            marker='o', label='Parallel-CPU', markersize=12, elinewidth=4)
ax1.errorbar(gpu_low.keys(), gpu_low.values(), yerr=list(gpu_low_err.values()),
            marker='o', label='RTX 3050 Laptop GPU', markersize=12, elinewidth=4)
ax1.set_title('(a) Performance Comparison for Low Lattice Sizes')
ax1.legend(loc='upper left')
ax1.set_ylim(top=0.25)
ax1.set_yticks(ticks=[0.0004136, 0.05, 0.10, 0.15, 0.20], labels=[4e-4, 0.05, 0.10, 0.15, 0.20])
ax1.set_ylabel('Flips/ns')
ax1.set_xlabel('Lattice Sizes', labelpad=38.0)
ax1.set_xticks(ticks=[20, 40, 80, 160, 320, 640], labels=['20*128', '40*128', '80*128', '160*128', '320*128', 
                                                      '640*128'], rotation=90)

ax2.errorbar(parallel_cpu_high.keys(), parallel_cpu_high.values(), yerr=list(parallel_cpu_high_err.values()),
            marker='o', label='Parallel-CPU', markersize=12, elinewidth=4)
ax2.errorbar(gpu_high.keys(), gpu_high.values(), yerr=list(gpu_high_err.values()),
            marker='o', label='RTX 3050 Laptop GPU', markersize=12, elinewidth=4)
ax2.errorbar(pro_gpu_high.keys(), pro_gpu_high.values(), marker='o', label='Tesla V100-SXM', 
             markersize=12, elinewidth=4)
ax2.set_title('(b) Performance Comparison for High Lattice Sizes')
ax2.set_xlabel('Lattice Sizes')
ax2.set_yticks(ticks=[0.1965, 10, 20, 30, 40], labels=[0.2, 10, 20, 30, 40])
ax2.set_xticks(ticks=[20, 40, 80, 160, 320, 640], labels=['$(20*128)^2$', '$(40*128)^2$', '$(80*128)^2$', 
                                                      '$(160*128)^2$', '$(320*128)^2$', 
                                                      '$(640*128)^2$'], rotation=90)
ax2.legend()

plt.savefig('figure/performance_high_and_low_comparison', bbox_inches='tight', dpi=300)
plt.show()

# # Low Lattice

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

# In[2]:


python_interpreted = {20: 4.214e-4, 40: 4.197e-4, 80: 4.182e-4, 160: 4.518e-4, 320: 4.176e-4, 640: 4.136e-4}
python_interpreted_err = {20: 2.704e-6, 40: 1.858e-6, 80: 1.814e-6, 160: 1.832e-6, 320: 2.045e-6, 
                          640: 1.045e-6}

python_compiled = {20: 6.238e-2, 40: 6.130e-2, 80: 6.189e-2, 160: 6.274e-2, 320: 6.254e-2, 640: 6.192e-2}
python_compiled_err = {20: 6.645e-4, 40: 2.005e-3, 80: 1.437e-3, 160: 5.533e-4, 320: 6.352e-4, 640: 9.423e-4}

parallel_cpu_low = {20: 4.472e-2, 40: 6.327e-2, 80:8.487e-2, 160: 1.013e-1, 320: 1.336e-1, 640: 1.615e-1}
parallel_cpu_low_err = {20: 1.63e-3, 40: 5.379e-4, 80: 2.169e-3, 160: 2.647e-3, 320: 2.325e-3, 640: 2.66e-3}

gpu_low = {20: 5.810e-3, 40: 1.162e-2, 80: 2.344e-2, 160: 4.684e-2, 320: 9.208e-2, 640: 1.800e-1}
gpu_low_err = {20: 1.779e-4, 40: 2.393e-4, 80: 5.262e-4, 160: 1.483e-3, 320: 2.920e-3, 640: 4.839e-3}

# In[3]:


plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=16) 

plt.errorbar(python_interpreted.keys(), python_interpreted.values(), yerr=list(python_interpreted_err.values()),
            marker='o', label='Python (Interpreted)', markersize=5, elinewidth=2)
plt.errorbar(python_compiled.keys(), python_compiled.values(), yerr=list(python_compiled_err.values()),
            marker='o', label='Python (Compiled)', markersize=5, elinewidth=2)
plt.errorbar(parallel_cpu_low.keys(), parallel_cpu_low.values(), yerr=list(parallel_cpu_low_err.values()),
            marker='o', label='Parallel-CPU', markersize=5, elinewidth=2)
plt.errorbar(gpu_low.keys(), gpu_low.values(), yerr=list(gpu_low_err.values()),
            marker='o', label='RTX 3050 Laptop GPU', markersize=5, elinewidth=2)

plt.yticks(ticks=[0.0004136, 0.05, 0.10, 0.15, 0.20], labels=[4e-4, 0.05, 0.10, 0.15, 0.20])
plt.xticks(ticks=[20, 40, 80, 160, 320, 640], labels=['20*128', '40*128', '80*128', '160*128', '320*128', 
                                                      '640*128'], rotation=90)
plt.legend()
plt.title('Performance Comparison for Low Lattice Sizes')
plt.ylabel('Flips/ns')
plt.xlabel('Lattice Sizes')

plt.savefig('figure/performance_comparison_low.png', bbox_inches='tight', dpi=300)

# # High Lattice Sizes

# In[10]:


min(parallel_cpu_high.values())

# In[3]:


parallel_cpu_high = {20: 2.201e-1, 40: 1.965e-1, 80: 1.971e-1, 160: 1.971e-1, 310: 1.977e-1, 320: np.nan,
                    640: np.nan}
parallel_cpu_high_err = {20: 1.643e-3, 40: 2.486e-3, 80: 1.233e-3, 160: 4.629e-4, 310: 5.088e-4, 320: np.nan,
                    640: np.nan}

gpu_high = {20: 2.076, 40: 2.304, 80: 2.356, 160: 2.367, 210: 2.360, 320: np.nan, 640: np.nan}
gpu_high_err = {20: 5.474e-3, 40: 8.630e-3, 80: 8.611e-3, 160: 7.636e-3, 210: 2.293e-3, 320: np.nan, 640: np.nan}

pro_gpu_high = {20: 15.179, 40: 40.984, 80: 42.887, 160: 43.594, 320: 43.768, 640: 43.535}

# In[4]:


plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=16) 

plt.errorbar(parallel_cpu_high.keys(), parallel_cpu_high.values(), yerr=list(parallel_cpu_high_err.values()),
            marker='o', label='Parallel-CPU', markersize=5, elinewidth=2)
plt.errorbar(gpu_high.keys(), gpu_high.values(), yerr=list(gpu_high_err.values()),
            marker='o', label='RTX 3050 Laptop GPU', markersize=5, elinewidth=2)
plt.errorbar(pro_gpu_high.keys(), pro_gpu_high.values(), marker='o', label='Tesla V100-SXM', 
             markersize=5, elinewidth=2)

plt.yticks(ticks=[0.1965, 10, 20, 30, 40], labels=[0.2, 10, 20, 30, 40])
plt.xticks(ticks=[20, 40, 80, 160, 320, 640], labels=['$(20*128)^2$', '$(40*128)^2$', '$(80*128)^2$', 
                                                      '$(160*128)^2$', '$(320*128)^2$', 
                                                      '$(640*128)^2$'], rotation=90)
plt.legend()
plt.title('Performance Comparison for High Lattice Sizes')
plt.ylabel('Flips/ns')
plt.xlabel('Lattice Sizes')
plt.savefig('figure/performance_comparison_high.png', bbox_inches='tight', dpi=300)
