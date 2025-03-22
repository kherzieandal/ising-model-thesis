#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np

# In[19]:


barwidth = 0.25

sixteen = [38.6254155874252, 3.56052637100219, 26.2276685237884]
y_err1 = [1.74685784355401, 0.0953718778625799, 0.369726979640298]
sixty_four = [597.677275896072, 4.7124171257019, 27.0313105583191]
y_err2 = [3.18738216055888, 0.196042193764432, 0.355612486101739]

bar1 = np.arange(len(sixteen))
bar2 = [x + barwidth for x in bar1]

plt.bar(bar1, sixteen, label = '16', width=barwidth, yerr=y_err1)
plt.bar(bar2, sixty_four, label = '64', width=barwidth, yerr=y_err2)
plt.legend()
plt.ylabel('Time (s)')
plt.xticks([r + barwidth for r in range(len(sixteen))], ['Serial', 'CPU Parallel', 'GPU Parallel'])
plt.savefig('figure/fig1', dpi=200)
plt.show()


# In[37]:


cpu_parallel = [3.56052637100219, 4.7124171257019, 12.3859839439392, 28.4219918251037, 76.4901569684346, 128.383195082346, 281.022122542063]
y_err1 = [0.0953718778625799, 0.196042193764432, 0.269064759499212, 0.220011903251373, 0.747261872587483, 5.30202629355832, 6.48178412915866]
gpu = [26.2276685237884, 27.0313105583191, 34.9002047379811, 36.7034606138865, 62.6508736610412, 60.3600758711496, 95.3019227186838]
y_err2 = [0.369726979640298, 0.355612486101739, 0.539932692108871, 0.519776268917007, 0.250144804975641, 0.479413408203341, 1.32677784958185]

plt.errorbar(range(len(cpu_parallel)), cpu_parallel, yerr=y_err1, label='Parallel CPU')
plt.errorbar(range(len(gpu)), gpu, yerr=y_err2, label='GPU')
plt.legend()
plt.xticks(np.arange(len(gpu)),['16', '64', '256', '512', '768', '1024', '1280'])
plt.xlabel('Lattice Size $(L^2)$')
plt.ylabel('Time $(s)$')
plt.savefig('figure/fig2', dpi=300)
plt.show()
