import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes as ax

#matplotlib.rcParams['backend'] = "WXAgg" #"Qt4Agg"
plt.rcParams['font.family']     = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.linewidth']  = 2
plt.rcParams['axes.edgecolor']  = 'grey'

fileLocation = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(fileLocation+"/Test/Accuracy.csv",index_col=0)

ax = data.plot(linewidth = 3,figsize=(5,3),cmap='coolwarm')
plt.legend(title=" σ ",loc="center left", bbox_to_anchor=(1, 0.5))
plt.title("1LP with Initialization Variability",fontweight='bold')
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
plt.ylim(0,1)
plt.grid()
plt.savefig(fileLocation + "/plot.png",dpi=1200,bbox_inches='tight')