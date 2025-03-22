#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt

# In[14]:


parallelCompiling = 11.33333333
parallelCompiled = 6.266666667
serial = 196.6666667
height = [serial, parallelCompiling, parallelCompiled]
x = ["Serial", "Parallel - Compiling", "Parallel - Compiled"]

# In[28]:


plt.bar(x, height)
plt.title("Parallel and Serial Computing Speed Comparison\nL = 20, Monte Carlo Steps = 400")
plt.ylabel("Second (s)")
plt.show()
