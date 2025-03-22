import matplotlib.pyplot as plt
import numpy as np



flips_ns_ideal = dict()
flips_ns = {1:9.71e-2, 2: 1.43e-1, 3: 1.68e-1, 4:1.83e-1, 5: 1.91e-1, 6:1.98e-1, 7:1.83e-1, 8:1.90e-1,
            9:1.93e-1, 10:1.99e-1, 11:2.05e-1, 12:2.09e-1}
for i in range(12):
    i += 1
    flips_ns_ideal[i] = flips_ns[1] * i
print(flips_ns_ideal)



plt.plot(flips_ns.keys(), flips_ns_ideal.values(), label='Ideal speedup', linestyle='--')
plt.plot(flips_ns.keys(), flips_ns.values(), label='Real Speedup')
plt.ylabel('Flips/ns')
plt.xlabel('No. of threads')
plt.legend()
plt.grid()
plt.savefig('strong_scaling', dpi=300, bbox_inches='tight')
plt.show()