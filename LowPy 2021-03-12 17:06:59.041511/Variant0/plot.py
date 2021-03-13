import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import numpy as np
# from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.cm import ScalarMappable
from matplotlib import gridspec

#matplotlib.rcParams['backend'] = "WXAgg" #"Qt4Agg"
plt.rcParams['font.family']     = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.linewidth']  = 0
plt.rcParams['axes.edgecolor']  = 'grey'




fig = plt.figure(figsize=(6, 2.5)) 
fig.suptitle('SLP Weight and Bias Updates after 5 Epochs of MNIST', fontweight='bold')
gs = gridspec.GridSpec(1, 2, width_ratios=[7, 1]) 

ax0 = plt.subplot(gs[0])
fileLocation = os.path.dirname(os.path.realpath(__file__))
weights = pd.read_csv(fileLocation+"/Weights0.csv",index_col=0).to_numpy()
maxWeight = weights.max()
minWeight = weights.min()
weights = np.transpose(weights)
plt.imshow(weights, cmap='coolwarm', interpolation='nearest', aspect='auto')  # aspect of 10 is 10 pixels of height per 1 pixel of width
plt.ylabel('Output Weights')
plt.xlabel('Input Weights')




ax1 = plt.subplot(gs[1])
weights = pd.read_csv(fileLocation+"/Weights1.csv",index_col=0).to_numpy()
plt.imshow(weights, cmap='coolwarm', interpolation='nearest', aspect='auto', vmax=maxWeight, vmin=minWeight)  # aspect of 10 is 10 pixels of height per 1 pixel of width
plt.ylabel('Output Biases')
plt.xlabel('Input Bias')
plt.tight_layout()



cmap = plt.get_cmap("coolwarm")
norm = plt.Normalize(minWeight, maxWeight)
sm =  ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax1)
# cbar.ax.set_title("Writes")

plt.savefig(fileLocation + "/plot.png",dpi=1200,bbox_inches='tight')

plt.show()




