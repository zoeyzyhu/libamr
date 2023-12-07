#! /usr/bin/env python
import matplotlib
#matplotlib.use('TkAgg')
from pylab import *

data = genfromtxt('data_with_refine.txt')
print(data.shape)

figure(1, figsize=(8, 6))
ax = axes()

ax.plot(data[:,1], data[:,2] / (data[:,1] - data[:,0]), 'ko-', linewidth = 2)
#ax.grid(True)
ax.grid(True, which='both', color='0.65', linestyle='-')
#plt.grid(True, which='both', color='0.65', linestyle='-')
ax.set_xlabel('Number of final meshblocks', fontsize = 15)
ax.set_ylabel('Time to refine a meshblock (s)', fontsize = 15)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks([4, 8, 16, 32, 64])
ax.set_xticklabels([4, 8, 16, 32, 64])

savefig('time_to_refine.png')
