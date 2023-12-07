#! /usr/bin/env python
import matplotlib
#matplotlib.use('TkAgg')
from pylab import *

data = genfromtxt('data_without_refine.txt')

figure(1, figsize=(8, 6))
ax = axes()

ax.plot(data[:,0], data[:,1], 'ko-', linewidth = 2)
#ax.grid(True)
ax.grid(True, which='both', color='0.65', linestyle='-')
#plt.grid(True, which='both', color='0.65', linestyle='-')
ax.set_xlabel('Number of Cores', fontsize = 15)
ax.set_ylabel('Time to completion (s)', fontsize = 15)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks([1, 4, 8, 16, 32])
ax.set_xticklabels([1, 4, 8, 16, 32])

savefig('time_to_completion_norefine.png')
