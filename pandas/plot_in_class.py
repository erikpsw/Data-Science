import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# plt.rc("font",family='MicroSoft YaHei')
# plt.rcParams['text.usetex'] = True
plt.rc("font",family='Times New Roman')
annealing_time = [1, 2, 5, 10, 20, 50, 100, 1000, 10000]
grain_size = [18.18186636, 27.58551749, 47.65676451, 66.60168293, 95.4851181, 101.7902391, 108.7814709, 104.556016, 101.781815]
oxide_layer_thickness = [12, 15, 19, 30, 40, 60, 100, 130, 140]

fig, ax1 = plt.subplots(figsize=(7, 5))
ax1.set_xlim(0.9, 11000)  # Adjusted limits to provide more space
ax1.set_xscale('log')
# Set x-axis formatter to avoid scientific notation
ax1.xaxis.set_major_formatter(ScalarFormatter())
line1, = ax1.plot(annealing_time, grain_size, '^-', color='blue', label='SEM')
ax1.set_xlabel('Annealing time (s)', size=12)
ax1.set_ylabel('Grain size (μm)', color='blue', size=12)
ax1.set_ylim(0, 120)
ax1.tick_params('y', colors='blue')

ax2 = ax1.twinx()
line2, = ax2.plot(annealing_time, oxide_layer_thickness, 's-', color='red', label='TEM')
ax2.set_ylabel('Thickness of oxidation (nm)', color='red', size=12)
ax2.tick_params('y', colors='red')
ax2.set_ylim(0, 160)

ax2.spines['left'].set_color('blue')
ax2.spines['right'].set_color('red')

# Combine legends
lines = [line1, line2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, bbox_to_anchor=(0.9, 0.2), fontsize=12)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.suptitle('Fig. 1. Evolution of grain size and thickness of oxidation with annealing time', y=0.05, size=12)
plt.show()