#!/usr/bin/env python3

import glob
import pickle
import numpy as np

import cmaps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator,ScalarFormatter

"""test!!!test"""
clear=[list(),list()]
cloudy=[list(),list()]
fog=[list(),list()]
rain=[list(),list()]
for filename in glob.glob("/root/Data/Output/Matrix/?????.pickle"):
    print(filename)
    trans=pickle.load(open(filename,"rb"))
    for cell_key in trans["Raob"].keys():
        if cell_key in ["Clear","Fair"]:
            clear[0]+=[cell[0] for cell in trans["Raob"][cell_key]]
            clear[1]+=[cell[0] for cell in trans["GMWR"][cell_key]]
        elif cell_key in ["Cloudy","Overcast"]:
            cloudy[0]+=[cell[0] for cell in trans["Raob"][cell_key]]
            cloudy[1]+=[cell[0] for cell in trans["GMWR"][cell_key]]
        elif cell_key in ["Fog"]:
            fog[0]+=[cell[0] for cell in trans["Raob"][cell_key]]
            fog[1]+=[cell[0] for cell in trans["GMWR"][cell_key]]
        elif cell_key in ["LightRain","Rain","HeavyRain","RainShower","HeavyRainShower","Thunderstorm"]:
            rain[0]+=[cell[0] for cell in trans["Raob"][cell_key]]
            rain[1]+=[cell[0] for cell in trans["GMWR"][cell_key]]

clear=np.array(clear)
nobs=np.shape(clear)[1]
departure_clear=clear[0,:,:]-clear[1,:,:]
departure_clear-=np.mean(departure_clear,axis=0)
rmatrix_clear=np.dot(departure_clear.T,departure_clear)**0.5/nobs

cloudy=np.array(cloudy)
nobs=np.shape(cloudy)[1]
departure_cloudy=cloudy[0,:,:]-cloudy[1,:,:]
departure_cloudy-=np.mean(departure_cloudy,axis=0)
rmatrix_cloudy=np.dot(departure_cloudy.T,departure_cloudy)**0.5/nobs

fog=np.array(fog)
nobs=np.shape(fog)[1]
departure_fog=fog[0,:,:]-fog[1,:,:]
departure_fog-=np.mean(departure_fog,axis=0)
rmatrix_fog=np.dot(departure_fog.T,departure_fog)**0.5/nobs

rain=np.array(rain)
nobs=np.shape(rain)[1]
departure_rain=rain[0,:,:]-rain[1,:,:]
departure_rain-=np.mean(departure_rain,axis=0)
rmatrix_rain=np.dot(departure_rain.T,departure_rain)**0.5/nobs

plt.figure(figsize=(15.8*0.3937,15.3*0.3937))
axis1=plt.subplot(2,2,1)
plt.imshow(rmatrix_clear,vmin=0.0,vmax=4.0,cmap=cmaps.MPL_RdYlBu_r,origin="lower")
plt.xticks(np.arange(0,14,1),["","","","","","","","","","","","","",""])
plt.yticks(np.arange(0,14,1),["1","","3","","5","","7","","9","","11","","13",""])
plt.ylabel("Channel ID")
plt.title("Condition: Clear")
plt.text(0.05,0.960,"(a)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold',color="white")

axis2=plt.subplot(2,2,2)
plt.imshow(rmatrix_cloudy,vmin=0.0,vmax=4.0,cmap=cmaps.MPL_RdYlBu_r,origin="lower")
plt.xticks(np.arange(0,14,1),["","","","","","","","","","","","","",""])
plt.yticks(np.arange(0,14,1),["","","","","","","","","","","","","",""])
plt.title("Condition: Cloudy")
plt.text(0.05,0.960,"(b)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold',color="white")

axis3=plt.subplot(2,2,3)
plt.imshow(rmatrix_fog,vmin=0.0,vmax=4.0,cmap=cmaps.MPL_RdYlBu_r,origin="lower")
plt.xticks(np.arange(0,14,1),["1","","3","","5","","7","","9","","11","","13",""])
plt.yticks(np.arange(0,14,1),["1","","3","","5","","7","","9","","11","","13",""])
plt.xlabel("Channel ID")
plt.ylabel("Channel ID")
plt.title("Condition: Fog")
plt.text(0.05,0.960,"(c)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold',color="white")

plt.subplot(2,2,4)
plt.imshow(rmatrix_rain,vmin=0.0,vmax=4.0,cmap=cmaps.MPL_RdYlBu_r,origin="lower")
plt.xticks(np.arange(0,14,1),["1","","3","","5","","7","","9","","11","","13",""])
plt.yticks(np.arange(0,14,1),["","","","","","","","","","","","","",""])
plt.xlabel("Channel ID")
plt.title("Condition: Rain")
plt.text(0.05,0.960,"(d)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold',color="white")

cax=plt.axes((0.913,0.08,0.015,0.88))
cbar=plt.colorbar(cax=cax,ticks=np.arange(0,4.01,0.2),orientation='vertical',fraction=.1)
cbar.ax.tick_params(labelsize=8)
cbar.set_label("Brightness Temperature (K)",fontsize=8)

plt.subplots_adjust(left=0.02,right=0.96,top=0.96,bottom=0.08,wspace=-0.2,hspace=0.13)
plt.savefig("./Figures/TestMWRRMatrix.png",dpi=600)
plt.close()

flag_save=open("./Matrix/GMWRRMatrixJJJ.pickle","wb")
pickle.dump({"clear":rmatrix_clear,\
             "cloudy":rmatrix_cloudy,\
             "fog":rmatrix_fog,\
             "rain":rmatrix_rain},flag_save)
flag_save.close()