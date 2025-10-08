#!/usr/bin/env python

import re
import os
import time
import glob
import cmaps
import pickle
import string
import argparse
from scipy import signal
from datetime import datetime, timedelta

from FileIO import *
from ATMCalculation import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator,ScalarFormatter

def calculatetheta_v(temperature,watervapor,cloudhydro,pressure):
    Rd=287.0
    cp=1004.0
    kappa=Rd/cp
    theta_v=(temperature*(100000.0/pressure)**kappa)*\
            (1+0.61*watervapor-cloudhydro)
    return theta_v

def calculatemovingwindowndeparture(station_id,st_date,ed_date,timewindow):
    st_date_fmt=datetime.strptime(st_date,"%Y%m%d%H%M")
    ed_date_fmt=datetime.strptime(ed_date,"%Y%m%d%H%M")
    PlotData_analysis=list()
    PlotData_era5=list()
    PlotData_kf1dvar=list()
    PlotData_1dvar=list()
    for i in range(int((ed_date_fmt-st_date_fmt)/timedelta(minutes=30))+1):
        obs_date=(st_date_fmt+timedelta(minutes=30*i)).strftime("%Y%m%d%H%M")
        if os.path.isfile("/root/Data/Output/MWROnlyProfile_TempWvClCi/"+station_id+"_"+obs_date+".pickle"):
            data=pickle.load(open("/root/Data/Output/MWROnlyProfile_TempWvClCi/"+station_id+"_"+obs_date+".pickle","rb"))
            thetav_analysis=calculatetheta_v(data["Analysis"]["Temperature"],\
                                             data["Analysis"]["WaterVapor"],\
                                             data["Analysis"]["CloudWater"],\
                                             data["Analysis"]["Pressure"])
            thetav_era5    =calculatetheta_v(data["ERA5"]["Temperature"],\
                                             data["ERA5"]["WaterVapor"],\
                                             data["ERA5"]["CloudWater"],\
                                             data["ERA5"]["Pressure"])
            MWR_Temperature=data["MWR"]["Temperature"]
            MWR_Watervapor =data["MWR"]["WaterVapor"]
            MWR_Pressure   =data["MWR"]["Pressure"]
            MWR_CloudWater =data["MWR"]["WaterVapor"]
            MWR_Temperature_mean=np.nanmean(MWR_Temperature,axis=0)
            MWR_Watervapor_mean =np.nanmean(MWR_Watervapor,axis=0)
            MWR_Pressure_mean   =np.nanmean(MWR_Pressure,axis=0)
            MWR_CloudWater_mean =np.nanmean(MWR_CloudWater,axis=0)
            MWR_Temperature_stdev=np.nanmean((MWR_Temperature-MWR_Temperature_mean)**2,axis=0)**0.5
            MWR_Watervapor_stdev =np.nanmean((MWR_Watervapor-MWR_Watervapor_mean)**2,axis=0)**0.5
            MWR_Pressure_stdev   =np.nanmean((MWR_Pressure-MWR_Pressure_mean)**2,axis=0)**0.5
            MWR_CloudWater_stdev =np.nanmean((MWR_CloudWater-MWR_CloudWater_mean)**2,axis=0)**0.5

            random_kernal=np.random.choice([-1,1],size=83)
            thetav_kf1dvar =calculatetheta_v(MWR_Temperature_mean,\
                                             MWR_Watervapor_mean,\
                                             MWR_CloudWater_mean,\
                                             MWR_Pressure_mean)
            thetav_1dvar   =calculatetheta_v(MWR_Temperature_mean+random_kernal*MWR_Temperature_stdev,\
                                             MWR_Watervapor_mean+random_kernal*MWR_Watervapor_stdev,\
                                             MWR_CloudWater_mean+random_kernal*MWR_CloudWater_stdev,\
                                             MWR_Pressure_mean+random_kernal*MWR_Pressure_stdev)
            PlotData_analysis.append(thetav_analysis)
            PlotData_era5.append(thetav_era5)
            PlotData_kf1dvar.append(thetav_kf1dvar)
            PlotData_1dvar.append(thetav_1dvar)
        else:
            PlotData_analysis.append(PlotData_analysis[-1])
            PlotData_era5.append(PlotData_analysis[-1])
            PlotData_kf1dvar.append(PlotData_analysis[-1])
            PlotData_1dvar.append(PlotData_analysis[-1])

    PlotData_analysis=np.array(PlotData_analysis)
    PlotData_era5=np.array(PlotData_era5)
    PlotData_kf1dvar=np.array(PlotData_kf1dvar)
    PlotData_1dvar=np.array(PlotData_1dvar)

    PlotData_analysis_dep=list()
    PlotData_era5_dep=list()
    PlotData_kf1dvar_dep=list()
    PlotData_1dvar_dep=list()

    for i in range(0,25):
        PlotData_analysis_dep.append(PlotData_analysis[24+i,:]-np.nanmean(PlotData_analysis[24-timewindow+i:24+i,:],axis=0))
        PlotData_era5_dep.append(PlotData_era5[24+i,:]-np.nanmean(PlotData_era5[24-timewindow+i:24+i,:],axis=0))
        PlotData_kf1dvar_dep.append(PlotData_kf1dvar[24+i,:]-np.nanmean(PlotData_kf1dvar[24-timewindow+i:24+i,:],axis=0))
        PlotData_1dvar_dep.append(PlotData_1dvar[24+i,:]-np.nanmean(PlotData_1dvar[24-timewindow+i:24+i,:],axis=0))
    PlotData_analysis_dep=np.array(PlotData_analysis_dep)
    PlotData_era5_dep=np.array(PlotData_era5_dep)
    PlotData_kf1dvar_dep=np.array(PlotData_kf1dvar_dep)
    PlotData_1dvar_dep=np.array(PlotData_1dvar_dep)
    return PlotData_analysis_dep,\
           PlotData_era5_dep,\
           PlotData_kf1dvar_dep,\
           PlotData_1dvar_dep

caseinfo=[["54727","202506301900","202507011900"],\
          ["57083","202506301700","202507011700"],\
          ["53673","202507081700","202507091700"],\
          ["54727","202507220400","202507230400"],\
          ["53463","202507240700","202507250700"],\
          ["54511","202507261700","202507271700"],\
          ["54511","202507272000","202507282000"]]

observation_height=np.array([0.000,0.025,0.050,0.075,0.100,0.125,0.150,0.175,0.200,0.225,\
                             0.250,0.275,0.300,0.325,0.350,0.375,0.400,0.425,0.450,0.475,\
                             0.500,0.550,0.600,0.650,0.700,0.750,0.800,0.850,0.900,0.950,\
                             1.000,1.050,1.100,1.150,1.200,1.250,1.300,1.350,1.400,1.450,\
                             1.500,1.550,1.600,1.650,1.700,1.750,1.800,1.850,1.900,1.950,\
                             2.000,2.250,2.500,2.750,3.000,3.250,3.500,3.750,4.000,4.250,\
                             4.500,4.750,5.000,5.250,5.500,5.750,6.000,6.250,6.500,6.750,\
                             7.000,7.250,7.500,7.750,8.000,8.250,8.500,8.750,9.000,9.250,\
                             9.500,9.750,10.000])*1000

"""test!!!test"""

gaussian_2d=np.ones([4,4])/16.0
#timewindows=[18,21,24]
timewindows=[9,12,15]
#timewindows=[2,3,6]
for ii,timewindow in enumerate(timewindows):
    PlotData_analysis_dep=list()
    PlotData_era5_dep=list()
    PlotData_kf1dvar_dep=list()
    PlotData_1dvar_dep=list()

    for i in range(7):
        trans_PlotData_analysis_dep,\
        trans_PlotData_era5_dep,\
        trans_PlotData_kf1dvar_dep,\
        trans_PlotData_1dvar_dep=calculatemovingwindowndeparture(caseinfo[i][0],caseinfo[i][1],caseinfo[i][2],timewindow)
        PlotData_analysis_dep.append(trans_PlotData_analysis_dep)
        PlotData_era5_dep.append(trans_PlotData_era5_dep)
        PlotData_kf1dvar_dep.append(trans_PlotData_kf1dvar_dep)
        PlotData_1dvar_dep.append(trans_PlotData_1dvar_dep)

    PlotData_analysis_dep=np.nanmean(np.array(PlotData_analysis_dep),axis=0)
    PlotData_era5_dep=np.nanmean(np.array(PlotData_era5_dep),axis=0)
    PlotData_kf1dvar_dep=np.nanmean(np.array(PlotData_kf1dvar_dep),axis=0)
    PlotData_1dvar_dep=np.nanmean(np.array(PlotData_1dvar_dep),axis=0)

    axis2=plt.subplot(3,3,1+ii*3)
    tp=np.rot90(signal.convolve2d(PlotData_era5_dep[:,:41][:,::-1],gaussian_2d,boundary='symm',mode='same'))
    image2=plt.imshow(tp,vmin=-1,vmax=1,cmap=cmaps.cmocean_balance,origin="lower")
    cs2=plt.contour(tp,levels=np.arange(-0.75,0.76,0.25),colors='gray',linewidths=(1,))
    plt.clabel(cs2,fmt='%2.2f',colors='k',fontsize=6)
    axis2.minorticks_on()
    axis2.yaxis.set_major_locator(MultipleLocator(4))
    axis2.xaxis.set_major_locator(MultipleLocator(4))
    plt.xlim(2.5,24)
    plt.ylim(0,40)
    if ii!=2:
        plt.xticks(np.arange(4,25,4),["" for shit in np.arange(-20,1,4)])
    else:
        plt.xticks(np.arange(4,25,4),[abs(int(shit)) for shit in np.arange(-20,1,4)/2],fontsize=8)
        plt.xlabel("Lead Hours",fontsize=9)
    if ii==0:
        plt.title("ERA5\n",fontsize=10)
    plt.yticks(np.arange(0,41,4),[int(shit) for shit in observation_height[:41][::4]],fontsize=8)
    plt.ylabel("Meters A.G.L.",fontsize=9)
    plt.gca().set_aspect(0.4)
    plt.text(0.1,0.935,"("+string.ascii_lowercase[ii*3+0]+")",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold',color="white")
    plt.text(0.77,1.050,"Moving Average Time Window Size: "+("%4.1f hours" % (timewindow/2.0)),horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=7,fontweight='bold',color="black")

    axis1=plt.subplot(3,3,2+ii*3)
    tp=np.rot90(signal.convolve2d(PlotData_analysis_dep[:,:41][:,::-1],gaussian_2d,boundary='symm',mode='same'))
    image1=plt.imshow(tp,vmin=-1,vmax=1,cmap=cmaps.cmocean_balance,origin="lower")
    cs1=plt.contour(tp,levels=np.arange(-0.75,0.76,0.25),colors='gray',linewidths=(1,))
    plt.clabel(cs1,fmt='%2.2f',colors='k',fontsize=6)
    axis1.minorticks_on()
    axis1.yaxis.set_major_locator(MultipleLocator(4))
    axis1.xaxis.set_major_locator(MultipleLocator(4))
    plt.xlim(2.5,24)
    plt.ylim(0,40)
    if ii!=2:
        plt.xticks(np.arange(4,25,4),["" for shit in np.arange(-20,1,4)])
    else:
        plt.xticks(np.arange(4,25,4),[abs(int(shit)) for shit in np.arange(-20,1,4)/2],fontsize=8)
        plt.xlabel("Lead Hours",fontsize=9)
    if ii==0:
        plt.title("TCKF1D-Var\n",fontsize=10)
    plt.yticks(np.arange(0,41,4),["" for shit in np.arange(0,41,4)])
    plt.gca().set_aspect(0.4)
    plt.text(0.1,0.935,"("+string.ascii_lowercase[ii*3+1]+")",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold',color="white")

    axis4=plt.subplot(3,3,3+ii*3)
    tp=np.rot90(signal.convolve2d(PlotData_1dvar_dep[:,:41][:,::-1],gaussian_2d,boundary='symm',mode='same'))
    image4=plt.imshow(tp,vmin=-1,vmax=1,cmap=cmaps.cmocean_balance,origin="lower")
    cs4=plt.contour(tp,levels=np.arange(-0.75,0.76,0.25),colors='gray',linewidths=(1,))
    plt.clabel(cs4,fmt='%2.2f',colors='k',fontsize=6)
    axis4.minorticks_on()
    axis4.yaxis.set_major_locator(MultipleLocator(4))
    axis4.xaxis.set_major_locator(MultipleLocator(4))
    plt.xlim(2.5,24)
    plt.ylim(0,40)
    if ii!=2:
        plt.xticks(np.arange(4,25,4),["" for shit in np.arange(-20,1,4)])
    else:
        plt.xticks(np.arange(4,25,4),[abs(int(shit)) for shit in np.arange(-20,1,4)/2],fontsize=8)
        plt.xlabel("Lead Hours",fontsize=9)
    if ii==0:
        plt.title("1D-Var\n",fontsize=10)
    plt.yticks(np.arange(0,41,4),["" for shit in np.arange(0,41,4)])
    plt.gca().set_aspect(0.4)
    plt.text(0.1,0.935,"("+string.ascii_lowercase[ii*3+2]+")",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold',color="white")

    cax=plt.axes((0.91, 0.07, 0.01, 0.863))
    cbar=plt.colorbar(image4,\
                      cax=cax,\
                      ticks=np.arange(-1,1.01,0.25),\
                      
                      orientation='vertical',\
                      fraction=.05,\
                      shrink=0.5)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Virtual Potential Temperature Anomaly (K)",fontsize=7)
    ax=cbar.ax
    text=ax.xaxis.label
    font=matplotlib.font_manager.FontProperties(size=6)
    text.set_font_properties(font)

plt.subplots_adjust(left=0.094,right=0.9,top=0.933,bottom=0.07,wspace=0.05,hspace=0.07)

plt.savefig("Figures/PrcpEarlyWarningSignal"+"_"+("%02d-%02d" % (timewindows[0],timewindows[2]))+".png",dpi=600)
plt.close()