
import matplotlib.pyplot as plt 
import json
import numpy as np 
import os

verts = [7, 14, 20, 26, 32, 40]
times = []
stds = []

for vert in verts:
    filename = 'ensemble_N'+str(vert)+'.json'
    with open(filename, 'r') as infile:
        data = json.load(infile)
    
        ensemble_times = data['ensemble_times']
        times.append(np.mean(ensemble_times))
        stds.append(np.std(ensemble_times))

#fit 
p = np.polyfit(verts, np.log10(times), 1)
plt.plot(verts, (p[0] * np.array(verts) + p[1]), 'k--', linewidth=2)

# plot
plt.plot(verts, np.log10(times), 'ro',markersize=8)
plt.errorbar(verts, np.log10(times), ecolor='k', elinewidth=2, fmt = None, yerr=[np.log10(np.array(times)-np.maximum(1.0e-7, np.array(times)-np.array(stds))), np.log10(2*np.array(stds))])
print(times, stds, np.log10(np.array(times)-np.maximum(1.0e-7, np.array(times)-np.array(stds))))
plt.axis([5,42, -8, 5])
plt.xlabel('|V|', fontsize=30)
plt.ylabel('TTS (classical) [s]', fontsize=30)
for item in ([plt.title, plt.xaxis.label, plt.yaxis.label] +
             plt.get_xticklabels() + plt.get_yticklabels()):
    item.set_fontsize(30)

plt.show()
