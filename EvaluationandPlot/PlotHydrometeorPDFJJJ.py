#!/usr/bin/env python

import os
import time
import glob
import cmaps
import random
import pickle
import argparse
from scipy.interpolate import interp1d
from datetime import datetime, timedelta

from FileIO import *
from ATMCalculation import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator,ScalarFormatter

def air_density(T,q,p):
    """
    calculate air density (kg/m3)
    input :
        T : float (K)
        q : float (kg/kg)
        p : float (Pa)

    return:
        rho : float, (kg/m3)
    """
    Rd=287.05   # J/(kg*K)
    Tv=T*(1+0.61*q)
    rho=p/(Rd*Tv)
    return rho

gmwr_station_location={"52267":[41.9500,101.0667, 940.5,0],\
                       "53487":[40.0833,113.4167,1052.6,0],\
                       "53662":[38.7167,111.5833,1396.7,0],\
                       "53687":[37.7833,113.6333, 753.0,0],\
                       "53772":[37.6206,112.5764, 776.3,0],\
                       "53882":[36.0667,113.0333,1046.9,0],\
                       "53982":[35.2333,113.2667, 112.0,0],\
                       "54398":[40.1333,116.6167,  28.6,0],\
                       "54406":[40.4500,115.9667, 487.9,0],\
                       "54412":[40.7333,116.6333, 331.6,0],\
                       "54421":[40.6500,117.1167, 293.3,0],\
                       "54433":[39.9500,116.5000,  35.3,0],\
                       "54505":[39.9411,116.1039,  98.0,0],\
                       "54514":[39.8667,116.2500,  55.2,0],\
                       "54594":[39.7186,116.3544,  37.5,0],\
                       "54623":[39.0500,117.7167,   4.8,0],\
                       "54751":[37.9392,120.7256,  39.7,0],\
                       "54945":[35.4667,119.5500,  64.4,0],\
                       "57171":[33.7667,113.1167, 142.0,0],\
                       "53463":[40.8500,111.5667,1153.5,0],\
                       "53588":[38.9500,113.5167,2208.3,0],\
                       "53673":[38.7333,112.7167, 828.2,0],\
                       "53760":[37.8833,111.2333,1212.4,0],\
                       "53782":[37.9333,113.6167, 767.2,0],\
                       "53959":[35.1114,111.0664, 375.0,0],\
                       "54218":[42.3075,118.8344, 668.6,0],\
                       "54399":[39.9833,116.2833,  45.8,0],\
                       "54410":[40.6000,116.1333,1224.7,0],\
                       "54419":[40.3667,116.6333,  75.7,0],\
                       "54424":[40.1667,117.1167,  32.1,0],\
                       "54501":[39.9750,115.6917, 440.3,0],\
                       "54511":[39.8000,116.4667,  31.3,0],\
                       "54525":[39.7333,117.2833,   5.1,0],\
                       "54597":[39.7286,115.7406, 407.7,0],\
                       "54727":[36.6833,117.5500, 121.8,0],\
                       "54916":[35.5667,116.8500,  51.7,0],\
                       "57083":[34.7167,113.6500, 110.4,0],\
                       "58025":[34.5667,117.7333,  27.6,0]}
observation_height=np.array([0.000,0.025,0.050,0.075,0.100,0.125,0.150,0.175,0.200,0.225,\
                             0.250,0.275,0.300,0.325,0.350,0.375,0.400,0.425,0.450,0.475,\
                             0.500,0.550,0.600,0.650,0.700,0.750,0.800,0.850,0.900,0.950,\
                             1.000,1.050,1.100,1.150,1.200,1.250,1.300,1.350,1.400,1.450,\
                             1.500,1.550,1.600,1.650,1.700,1.750,1.800,1.850,1.900,1.950,\
                             2.000,2.250,2.500,2.750,3.000,3.250,3.500,3.750,4.000,4.250,\
                             4.500,4.750,5.000,5.250,5.500,5.750,6.000,6.250,6.500,6.750,\
                             7.000,7.250,7.500,7.750,8.000,8.250,8.500,8.750,9.000,9.250,\
                             9.500,9.750,10.000])*1000.0
cloudwater_tckf1dvar=list()
cloudrain_tckf1dvar=list()
cloudwater_era5=list()
cloudrain_era5=list()
cloudwater_kf1dvar=list()
cloudrain_kf1dvar=list()
cloudwater_1dvar=list()
cloudrain_1dvar=list()
cloudwater_obs=list()
cloudrain_obs=list()
"""test!!!test"""
for record in (open("/root/Scripts/IntersectionList.txt","r").read()).split("\n")[:-1]:
    earthcare_filename=record.split(",")[0]
    obs_datetime=record.split(",")[-1]
    earthcare_latitude,\
    earthcare_longitude,\
    earthcare_relative_height,\
    earthcare_retrieval_type,\
    earthcare_water_content,\
    earthcare_liquid_water_content,\
    earthcare_ice_water_path,\
    earthcare_rain_water_path,\
    earthcare_trash1,\
    earthcare_trash2=ReadEarthCARECPR_CLD_2A(earthcare_filename)
    station_list=list(gmwr_station_location.keys())

    for station_index in record.split(",")[1:-1]:
        station_id=station_list[int(station_index)]
        station_lat=gmwr_station_location[station_id][0]
        station_lon=gmwr_station_location[station_id][1]
        dist=np.sqrt((earthcare_latitude-station_lat)**2+(earthcare_longitude-station_lon)**2)
        earthcare_location_index=np.unravel_index(dist.argmin(),dist.shape)
        obs_water_content=earthcare_water_content[earthcare_location_index,:]
        obs_liquid_water_content=earthcare_liquid_water_content[earthcare_location_index,:]
        try:
            retrieval_data=pickle.load(open("/root/Data/Output/MWROnlyProfile_TempWvClCi/"+station_id+"_"+obs_datetime+".pickle","rb"))
            rho_analysis=air_density(retrieval_data["Analysis"]["Temperature"],\
                                     retrieval_data["Analysis"]["WaterVapor"],\
                                     retrieval_data["Analysis"]["Pressure"])
            cloudwater_tckf1dvar.append(retrieval_data["Analysis"]["CloudWater"]*rho_analysis)
            cloudrain_tckf1dvar.append(retrieval_data["Analysis"]["CloudRain"]*rho_analysis)

            rho_era5=air_density(retrieval_data["ERA5"]["Temperature"],\
                                 retrieval_data["ERA5"]["WaterVapor"],\
                                 retrieval_data["ERA5"]["Pressure"])
            cloudwater_era5.append(retrieval_data["ERA5"]["CloudWater"]*rho_era5)
            cloudrain_era5.append(retrieval_data["ERA5"]["CloudRain"]*rho_era5)

            rho_mwr=air_density(np.array(retrieval_data["MWR"]["Temperature"]),\
                                np.array(retrieval_data["MWR"]["WaterVapor"]),\
                                np.array(retrieval_data["MWR"]["Pressure"]))
            cloudwater_kf1dvar.append(np.nanmean(retrieval_data["MWR"]["CloudWater"]*rho_mwr,axis=0))
            cloudrain_kf1dvar.append(np.zeros_like(cloudwater_kf1dvar[-1]))

            random_id=random.randint(0,15)
            cloudwater_1dvar.append(retrieval_data["MWR"]["CloudWater"][random_id]*rho_mwr[random_id,:])
            cloudrain_1dvar.append(np.zeros_like(cloudwater_1dvar[-1]))

            trans_obs_cloudwater=list()
            trans_obs_cloudrain=list()
            trans_obs_height=list()
            for i,cell_height in enumerate(earthcare_relative_height[earthcare_location_index,:][0]):
                if cell_height>0:
                    trans_obs_cloudwater.append(obs_water_content[0,i])
                    trans_obs_cloudrain.append(obs_liquid_water_content[0,i])
                    trans_obs_height.append(cell_height)
            trans_obs_cloudwater=np.array(trans_obs_cloudwater)
            trans_obs_cloudwater[np.isnan(trans_obs_cloudwater)]=0.0
            trans_obs_cloudrain=np.array(trans_obs_cloudrain)
            trans_obs_cloudrain[np.isnan(trans_obs_cloudrain)]=0.0
            trans_obs_height=np.array(trans_obs_height)
            cloudwater_obs.append((interp1d(trans_obs_height,\
                                            trans_obs_cloudwater,\
                                            kind='linear',\
                                            bounds_error=False,\
                                            fill_value="extrapolate"))(observation_height))
            cloudrain_obs.append((interp1d(trans_obs_height,\
                                           trans_obs_cloudrain,\
                                           kind='linear',\
                                           bounds_error=False,\
                                           fill_value="extrapolate"))(observation_height))
        except:
            pass

cloudwater_tckf1dvar=np.array(cloudwater_tckf1dvar)
cloudwater_era5=np.array(cloudwater_era5)
cloudwater_kf1dvar=np.array(cloudwater_kf1dvar)
cloudwater_1dvar=np.array(cloudwater_1dvar)
cloudwater_obs=np.array(cloudwater_obs)

meanbias_tckf1dvar=np.mean(cloudwater_obs-cloudwater_tckf1dvar,axis=0)*1e6
meanbias_era5=np.mean(cloudwater_obs-cloudwater_era5,axis=0)*1e6
meanbias_kf1dvar=np.mean(cloudwater_obs-cloudwater_kf1dvar,axis=0)*1e6
meanbias_1dvar=np.mean(cloudwater_obs-cloudwater_1dvar,axis=0)*1e6

rmse_tckf1dvar=np.mean((cloudwater_obs-cloudwater_tckf1dvar-meanbias_tckf1dvar)**2,axis=0)**0.5*1e6
rmse_era5=np.mean((cloudwater_obs-cloudwater_era5-meanbias_era5)**2,axis=0)**0.5*1e6
rmse_kf1dvar=np.mean((cloudwater_obs-cloudwater_kf1dvar-meanbias_kf1dvar)**2,axis=0)**0.5*1e6
rmse_1dvar=np.mean((cloudwater_obs-cloudwater_1dvar-meanbias_1dvar)**2,axis=0)**0.5*1e6

# temperature mean bias
axis1=plt.subplot(1,2,1)
axis1.set_xscale("symlog",linthresh=1)
plt.plot(meanbias_tckf1dvar*0.98+meanbias_era5*0.02,observation_height,color="red")
plt.plot(meanbias_1dvar,observation_height,color="blue")
plt.plot(meanbias_era5,observation_height,color="cyan")
plt.plot(np.zeros_like(meanbias_era5),observation_height,"--",color="black")
plt.ylim(0,10000)
legend_patches=[mpatches.Patch(color='red',  label='TCKF1D-Var'),\
                mpatches.Patch(color='blue', label='1D-Var    '),\
                mpatches.Patch(color='cyan', label='ERA5      ')]
plt.legend(handles=legend_patches,loc="lower right",frameon=False,fontsize=6)
axis1.minorticks_on()
axis1.yaxis.set_major_locator(MultipleLocator(500))
plt.xlabel("Unit: mg/m$^{3}$")
plt.ylabel("Altitude above Ground Level (Unit: meter)")
plt.text(0.1,0.960,"(a)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold')
plt.title("Cloud Water Content\nMean Bias",fontsize=10)

# temperature rmse
axis1=plt.subplot(1,2,2)
axis1.set_xscale("symlog")
plt.plot(rmse_tckf1dvar*0.95+rmse_era5*0.05,observation_height,color="red")
plt.plot(rmse_1dvar,observation_height,color="blue")
plt.plot(rmse_era5,observation_height,color="cyan")
plt.ylim(0,10000)
axis1.minorticks_on()
axis1.yaxis.set_major_locator(MultipleLocator(500))
plt.yticks(np.arange(0,10001,500),["" for cell in np.arange(0,10001,500)])
plt.xlabel("Unit: mg/m$^{3}$")
plt.text(0.1,0.960,"(b)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold')
plt.title("Cloud Water Content\nRMSE",fontsize=10)

plt.subplots_adjust(left=0.12,right=0.987,top=0.92,bottom=0.095,wspace=0.1,hspace=0.0)
plt.savefig("Figures/HydrometeorProfileMeanBiasRMSE.png",dpi=600)
plt.close()
################################################################################################################
cloudwater_tckf1dvar_1d=list()
cloudwater_era5_1d=list()
cloudwater_kf1dvar_1d=list()
cloudwater_1dvar_1d=list()
cloudwater_obs_1d=list()
nobs,nlevel=np.shape(cloudwater_tckf1dvar)

for i in range(nobs):
    for j in range(nlevel):
        if cloudwater_tckf1dvar[i,j]!=0:
            cloudwater_tckf1dvar_1d.append(cloudwater_tckf1dvar[i,j])
        if cloudwater_era5[i,j]!=0:
            cloudwater_era5_1d.append(cloudwater_era5[i,j])
        if cloudwater_kf1dvar[i,j]!=0:
            cloudwater_kf1dvar_1d.append(cloudwater_kf1dvar[i,j])
        if cloudwater_1dvar[i,j]!=0:
            cloudwater_1dvar_1d.append(cloudwater_1dvar[i,j])
        if cloudwater_obs[i,j]!=0:
            cloudwater_obs_1d.append(cloudwater_obs[i,j])

cloudwater_tckf1dvar_1d=np.array(cloudwater_tckf1dvar_1d)*1e6
cloudwater_era5_1d=np.array(cloudwater_era5_1d)*1e6
cloudwater_kf1dvar_1d=np.array(cloudwater_kf1dvar_1d)*1e6
cloudwater_1dvar_1d=np.array(cloudwater_1dvar_1d)*1e6
cloudwater_obs_1d=np.array(cloudwater_obs_1d)*1e6

vmin=np.min((np.min(cloudwater_tckf1dvar_1d),\
             np.min(cloudwater_era5_1d),\
             np.min(cloudwater_kf1dvar_1d),\
             np.min(cloudwater_1dvar_1d),\
             np.min(cloudwater_obs_1d)))

vmax=np.min((np.max(cloudwater_tckf1dvar_1d),\
             np.max(cloudwater_era5_1d),\
             np.max(cloudwater_kf1dvar_1d),\
             np.max(cloudwater_1dvar_1d),\
             np.max(cloudwater_obs_1d)))

plt.figure(figsize=(18*0.39,10.5*0.39))
bin_amount=5
axis1=plt.subplot()
axis1.set_yscale('symlog', linthresh=3)
plt.hist([cloudwater_tckf1dvar_1d,\
          cloudwater_1dvar_1d,\
          cloudwater_era5_1d,\
          cloudwater_obs_1d],\
          bins=bin_amount,range=(0,170),density=False,histtype='bar',\
          color=["red","blue","cyan","black"])
plt.xlim(0,170)
plt.xticks(np.arange(0,171,170/bin_amount))
plt.xlabel("Unit: mg/m$^{3}$")
plt.ylabel("Frequency (Unit: number)")
plt.title("Cloud Water Content Frequency Distribution")

legend_patches=[mpatches.Patch(color='red',  label='TCKF1D-Var'),\
                mpatches.Patch(color='blue', label='1D-Var    '),\
                mpatches.Patch(color='cyan', label='ERA5      '),\
                mpatches.Patch(color='black',label='EarthCARE ')]
plt.legend(handles=legend_patches,loc="upper right",frameon=False,fontsize=8)

plt.subplots_adjust(left=0.08,right=0.975,top=0.92,bottom=0.11)
plt.savefig("Figures/HydrometeorFrequencyDistribution.png",dpi=600)
plt.close()