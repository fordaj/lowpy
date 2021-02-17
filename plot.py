import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import numpy as np

#matplotlib.rcParams['backend'] = "WXAgg" #"Qt4Agg"
plt.rcParams['font.family']     = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.linewidth']  = 2
plt.rcParams['axes.edgecolor']  = 'grey'

fileLocation = os.path.dirname(os.path.realpath(__file__))
metrics = pd.read_csv(fileLocation+"/Test/Accuracy.csv",index_col=0)
# for c in range(len(metrics.columns)):
#     metrics = metrics.rename(columns={metrics.columns[c]:'%0.3f'%float(metrics.columns[c])})
# architecture = pd.read_csv(fileLocation+"/Architecture.csv",index_col=0)
# labels = architecture['Beta'][0]
# labels = np.fromstring(labels[1:len(labels)-1],sep=' ')


ax = metrics.plot(linewidth = 3,figsize=(5,3),cmap='coolwarm')
plt.legend(title="Num States",loc="center left", bbox_to_anchor=(1, 0.5))
plt.title("CNN-MNIST - Varied Precision",fontweight='bold')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
#ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
plt.ylim(0,1)
plt.grid()
plt.savefig(fileLocation + "/plot.png",dpi=1200,bbox_inches='tight')