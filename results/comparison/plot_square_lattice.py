import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 8), sharex=True)

plt.rc('axes', titlesize=24)        # Controls Axes Title
plt.rc('axes', labelsize=24)        # Controls Axes Labels
plt.rc('xtick', labelsize=20)       # Controls x Tick Labels
plt.rc('ytick', labelsize=24)       # Controls y Tick Labels
plt.rc('legend', fontsize=24)       # Controls Legend Font
plt.rc('figure', titlesize=24)

parallel_cpu_flip = {128: 0.131, 256: 0.150, 512: 0.152, 1024: 0.157, 2048: 0.157, 4096: 0.181}
parallel_cpu_flip_err = {128: 3.59e-3, 256: 5.49e-4, 512: 2.62e-4, 1024:1.59e-4, 2048: 7.77e-5, 4096: 1.47e-4}
gpu_flip = {128: 3.790e-2, 256: 1.459e-1, 512: 4.980e-1, 1024: 1.211, 2048: 1.928, 4096: 2.256}
gpu_flip_err = {128: 1.064e-3, 256: 3.379e-3, 512: 1.162e-2, 1024: 1.054e-2, 2048: 2.071e-3, 4096: 3.280e-3}

parallel_cpu_time = {128: 0.126, 256: 0.437, 512: 1.729, 1024: 6.665, 2048: 26.68, 4096: 92.64}
parallel_cpu_time_err = {128: 1.18e-2, 256: 5.10e-3, 512: 9.52e-2, 1024: 2.14e-2, 2048: 4.18e-2, 4096: 2.39e-1}
gpu_time = {128: 4.326e-1, 256: 4.494e-1, 512: 5.266e-1, 1024: 8.659e-1, 2048: 2.175e0, 4096: 7.437e0}
gpu_time_err = {128: 1.263e-2, 256: 1.093e-2, 512: 1.255e-2, 1024: 7.66e-3, 2048: 2.341e-3, 4096: 1.082e-2}

ax1.errorbar(parallel_cpu_flip.keys(), parallel_cpu_flip.values(), yerr=list(parallel_cpu_flip_err.values()),
            marker='o', label='Parallel-CPU', markersize=16, elinewidth=4)
ax1.errorbar(gpu_flip.keys(), gpu_flip.values(), yerr=list(gpu_flip_err.values()), elinewidth=4,
             marker='o', markersize=16, label='Mobile GPU')
ax1.set_xticks(ticks=[128, 256, 512, 1024, 2048, 4096], labels=[128, 256, 512, 1024, 2048, 4096], rotation=90)
ax1.legend()
ax1.set_title('(a) Parallel-CPU vs GPU Performance (Flips/ns)')
ax1.set_ylabel('Flips/ns')
ax1.set_xlabel('Square Lattice Sizes')

ax2.errorbar(parallel_cpu_time.keys(), parallel_cpu_time.values(), yerr=list(parallel_cpu_time_err.values()), 
             marker='o', markersize=16, label='Parallel-CPU', elinewidth=4)
ax2.errorbar(gpu_time.keys(), gpu_time.values(), marker='o', yerr=list(gpu_time_err.values()), elinewidth=4,
             markersize=16, label='Mobile GPU')
ax2.set_xticks(ticks=[128, 256, 512, 1024, 2048, 4096], labels=[128, 256, 512, 1024, 2048, 4096], rotation=90)
ax2.legend()
ax2.set_title('(b) Parallel-CPU vs GPU Performance (Time [s])')
ax2.set_ylabel('Time [s]')
ax2.set_xlabel('Square Lattice Sizes')

plt.savefig('figure/cpu-gpu_comparison_combined.png', bbox_inches='tight', dpi=300)


fig = plt.figure(figsize=(15, 10))

parallel_cpu_flip = {128: 0.131, 256: 0.150, 512: 0.152, 1024: 0.157, 2048: 0.157, 4096: 0.181}
parallel_cpu_err = {128: 3.59e-3, 256: 5.49e-4, 512: 2.62e-4, 1024:1.59e-4, 2048: 7.77e-5, 4096: 1.47e-4}
gpu = {128: 3.78e-2, 256: 1.45e-1, 512: 5e-1, 1024: 1.21, 2048: 1.93, 4096: 2.26}

plt.plot(parallel_cpu.keys(), parallel_cpu.values(), marker='o', markersize=16, label='Parallel-CPU')
plt.plot(gpu.keys(), gpu.values(), marker='o', markersize=16, label='GPU')
plt.xticks([256, 512, 1024, 2048, 4096], fontsize=16, rotation=90)
plt.yticks(fontsize=16)
plt.legend(fontsize=30)
plt.title('(a) Parallel-CPU vs GPU Performance (Flips/ns)', fontsize=32)
plt.xlabel('Square Lattice Sizes', fontsize=24)
plt.ylabel('Flips/ns', fontsize=24)
plt.savefig('figure/cpu-gpu_comparison_flips_per_ns.png', bbox_inches='tight', dpi=300)
plt.show()

parallel_cpu = {128: 1.75e-2, 256: 4.41e-1, 512: 1.53, 1024: 5.40, 2048: 2.01e1, 4096: 8.85e1, 8192: 3.54e2}
gpu = {128: 4.34e-1, 256: 4.51e-1, 512: 5.25e-1, 1024: 8.64e-1, 2048: 2.18e0, 4096: 7.42e0, 8192: 2.85e1}

fig = plt.figure(figsize=(15, 10))

parallel_cpu = {128: 1.75e-2, 256: 4.41e-1, 512: 1.53, 1024: 5.40, 2048: 2.01e1, 4096: 8.85e1}
gpu = {128: 4.34e-1, 256: 4.51e-1, 512: 5.25e-1, 1024: 8.64e-1, 2048: 2.18e0, 4096: 7.42e0}

plt.plot(parallel_cpu.keys(), parallel_cpu.values(), marker='o', markersize=16, label='Parallel-CPU')
plt.plot(gpu.keys(), gpu.values(), marker='o', markersize=16, label='GPU')
plt.xticks([256, 512, 1024, 2048, 4096], fontsize=16, rotation=90)
plt.yticks(fontsize=16)
plt.legend(fontsize=30)
plt.xlabel('Square Lattice Sizes', fontsize=24)
plt.title('Parallel-CPU vs GPU Performance (Time [s])', fontsize=32)
plt.ylabel('Time [s]', fontsize=24)
plt.savefig('figure/cpu-gpu_comparison_time.png', bbox_inches='tight', dpi=300)
plt.show()