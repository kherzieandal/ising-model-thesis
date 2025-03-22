import matplotlib.pyplot as plt
from numpy import mean
import pandas as pd

plt.figure(figsize=(8, 5))
plt.rcParams['font.size'] = 14
moores_law = pd.read_csv('data_points/moores_law_1975.csv', header=None, index_col=0)
density = pd.read_csv('data_points/density.csv', header=None, index_col=0)

plt.plot(moores_law, marker='o')
plt.yscale('log')
plt.grid()
plt.title("Moore's law")
plt.xlabel('Year')
plt.ylabel('Transistors per chip')
plt.savefig('moores_law_only', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 5))
plt.rcParams['font.size'] = 14
moores_law = pd.read_csv('data_points/moores_law_1975.csv', header=None, index_col=0)
density = pd.read_csv('data_points/density.csv', header=None, index_col=0)

plt.plot(moores_law, marker='o', label="Moore's Law")
plt.plot(density, marker='o', label='Density')
plt.yscale('log')
plt.grid()
plt.legend()
plt.xlabel('Year')
plt.ylabel('Transistors per chip')
plt.savefig('moores_law', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(9, 5))
plt.rcParams['font.size'] = 14

y = [1, 4.81, 6309.57, 35854.65, 53366.99, 54495.88]
x = [1978.18, 1986.12, 2002.86, 2010.91, 2014.86, 2017.93]

y1 = y[0:2]
x1 = x[0:2]
plt.fill_between(x1, y1, 1, color='#ab0f13', label='CISC (22%/yr.)', zorder=3)
plt.annotate('CISC (22%/yr.)', xy=(mean(x1), 1), xytext=(mean(x1), 1e2), 
             arrowprops ={'width': 2, 'color':'#ab0f13'}, 
             fontsize=12, color='white', fontweight='bold', horizontalalignment='center', backgroundcolor='#ab0f13')

y2 = y[1:3]
x2 = x[1:3]
plt.fill_between(x2, y2, 1, color='#01ac4e', label='RISC (52%/yr.)', zorder=3)
plt.annotate('RISC (52%/yr.)', xy=(mean(x2), 1), xytext=(mean(x2)-5, 8e2), 
             arrowprops ={'width': 2, 'color':'#01ac4e'}, 
             fontsize=12, color='white', fontweight='bold', horizontalalignment='center', backgroundcolor='#01ac4e')

y3 = y[2:4]
x3 = x[2:4]
plt.fill_between(x3, y3, 1, color='#307fc2', label='End of Dennard Scaling (23%/yr.)', zorder=3)
plt.annotate('End of Dennard Scaling (23%/yr.)', xy=(mean(x3), 6e3), xytext=(mean(x3)-10, 5e3), 
             arrowprops ={'color':'#307fc2', 'width':2}, 
             fontsize=12, color='white', fontweight='bold', horizontalalignment='right', backgroundcolor='#307fc2',
            zorder= 5)

y4 = y[3:5]
x4 = x[3:5]
plt.fill_between(x4, y4, 1, color='#636466', label="Amdahl's Law (12%/yr.)", zorder=3)
plt.annotate("Amdahl's Law (12%/yr.)", xy=(mean(x4), 3e4), xytext=(mean(x3), 3e4), 
             arrowprops ={'color':'#636466', 'width':2}, 
             fontsize=12, color='white', fontweight='bold', horizontalalignment='right', backgroundcolor='#636466')


y5 = y[4:]
x5 = x[4:]
plt.fill_between(x5, y5, 1, color='#ef483f', label="End of the Line (3%/yr.)", zorder=3)
plt.annotate("End of the Line (3%/yr.)", xy=(mean(x5), 1), xytext=(mean(x5), 1e5), 
             arrowprops ={'color':'#ef483f', 'width':2}, 
             fontsize=12, color='white', fontweight='bold', horizontalalignment='right', backgroundcolor='#ef483f')


plt.yscale('log')
plt.ylim([1, 100_000])

# plt.legend(loc='upper left')
plt.xlabel('Year')
plt.ylabel('Performance vs. VAX11-780')
plt.grid(zorder=0)
plt.savefig("computer_performance_growth", dpi=300, bbox_inches='tight')