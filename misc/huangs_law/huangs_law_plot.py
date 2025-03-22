import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from scipy.interpolate import interp1d
import matplotlib.patches as mpatches

### Interpolation

# Log Interpolation
def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind, fill_value='extrapolate')
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

def quadratic_log_interp1d(xx, yy, deg=1):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = np.poly1d(np.polyfit(logx, logy, deg=deg))
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

### Start of Interpolation
df_gpu_scatter = pd.read_csv('scatter_plot_gpu.csv', header=None)
df_cpu_scatter = pd.read_csv('specint.csv')

### GPU Interpolation:
x_years_interp = np.linspace(2005, 2025, 100)
gpu_y_interp = log_interp1d(df_gpu_scatter[0], df_gpu_scatter[1])
gpu_y_data = gpu_y_interp(x_years_interp)

### CPU Interpolation
model = quadratic_log_interp1d(df_cpu_scatter['Year'], df_cpu_scatter['Performance'], 2)
cpu_x_years_interp = np.linspace(df_cpu_scatter['Year'].min(), 2015, 250)

model_2 = np.poly1d(np.polyfit(df_cpu_scatter['Year'], df_cpu_scatter['Performance'], 2))
cpu_x_years_interp_2 = np.linspace(2015, 2025, 50)

plt.figure(figsize=(8,4.5), dpi=300)
plt.rcParams['font.size'] = 14

## GPU Plot
plt.scatter(df_gpu_scatter[0], df_gpu_scatter[1], c='#73b730')
plt.plot(x_years_interp[np.where(x_years_interp < 2015)], gpu_y_data[np.where(x_years_interp < 2015)], c='#73b730')
plt.plot(x_years_interp[np.where(x_years_interp >= 2015)], gpu_y_data[np.where(x_years_interp >= 2015)], c='#73b730', linestyle='dashed')

### CPU Plot

plt.plot(cpu_x_years_interp, model(cpu_x_years_interp), c='#1774b9')
plt.plot(cpu_x_years_interp_2, model_2(cpu_x_years_interp_2), c='#1774b9', linestyle='dashed')
plt.scatter(df_cpu_scatter['Year'], df_cpu_scatter['Performance'], c='#1774b9')

green_patch = mpatches.Patch(color='#73b730', label='GPU-Computing Performance')
blue_patch = mpatches.Patch(color='#1774b9', label='Single-threaded Performance')
plt.legend(handles=[green_patch, blue_patch], loc='best')
plt.yscale('log')
plt.grid(axis='y')
plt.xlabel('Year')
plt.ylabel('Performance (SpecINT $x 10^3$ )')
plt.savefig('huangs_law', dpi=300, bbox_inches='tight')
plt.show()