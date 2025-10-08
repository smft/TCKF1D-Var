#!/usr/bin/env python

import os
import time
import glob
import cmaps
import pickle
import argparse
from meteostat import Hourly
from datetime import datetime, timedelta


from FileIO import *
from ATMCalculation import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator,ScalarFormatter

"""test!!!test"""
stations=["53463","53772","54218","54511","54727","57083"]
weather_conditions=["Clear",\
                    "Fair",\
                    "Cloudy",\
                    "Overcast",\
                    "Fog",\
                    "FreezingFog",\
                    "LightRain",\
                    "Rain",\
                    "HeavyRain",\
                    "FreezingRain",\
                    "HeavyFreezingRain",\
                    "Sleet",\
                    "HeavySleet",\
                    "LightSnowfall",\
                    "Snowfall",\
                    "HeavySnowfall",\
                    "RainShower",\
                    "HeavyRainShower",\
                    "SleetShower",\
                    "HeavySleetShower",\
                    "SnowShower",\
                    "HeavySnowShower",\
                    "Lightning",\
                    "Hail",\
                    "Thunderstorm",\
                    "HeavyThunderstorm",\
                    "Storm"]
observation_height=np.array([0.000,0.025,0.050,0.075,0.100,0.125,0.150,0.175,0.200,0.225,\
                             0.250,0.275,0.300,0.325,0.350,0.375,0.400,0.425,0.450,0.475,\
                             0.500,0.550,0.600,0.650,0.700,0.750,0.800,0.850,0.900,0.950,\
                             1.000,1.050,1.100,1.150,1.200,1.250,1.300,1.350,1.400,1.450,\
                             1.500,1.550,1.600,1.650,1.700,1.750,1.800,1.850,1.900,1.950,\
                             2.000,2.250,2.500,2.750,3.000,3.250,3.500,3.750,4.000,4.250,\
                             4.500,4.750,5.000,5.250,5.500,5.750,6.000,6.250,6.500,6.750,\
                             7.000,7.250,7.500,7.750,8.000,8.250,8.500,8.750,9.000,9.250,\
                             9.500,9.750,10.000])*1000.0
start_date_datetime=datetime(year=2025,month=7,day=1,hour=00,minute=0)

Bias_ERA5       ={"clear" :{"temperature":list(),"watervapor":list()},\
                  "cloudy":{"temperature":list(),"watervapor":list()},\
                  "fog"   :{"temperature":list(),"watervapor":list()},\
                  "rain"  :{"temperature":list(),"watervapor":list()}}
Bias_TCKF1Dvar  ={"clear" :{"temperature":list(),"watervapor":list()},\
                  "cloudy":{"temperature":list(),"watervapor":list()},\
                  "fog"   :{"temperature":list(),"watervapor":list()},\
                  "rain"  :{"temperature":list(),"watervapor":list()}}
Bias_KF1DVar    ={"clear" :{"temperature":list(),"watervapor":list()},\
                  "cloudy":{"temperature":list(),"watervapor":list()},\
                  "fog"   :{"temperature":list(),"watervapor":list()},\
                  "rain"  :{"temperature":list(),"watervapor":list()}}
Bias_1DVar      ={"clear" :{"temperature":list(),"watervapor":list()},\
                  "cloudy":{"temperature":list(),"watervapor":list()},\
                  "fog"   :{"temperature":list(),"watervapor":list()},\
                  "rain"  :{"temperature":list(),"watervapor":list()}}

for station in stations:
    # load radiosonde observation
    radiosonde_all=ParserIGRARadiosondeRecord("/root/Data/Radiosonde/CHM000"+station+"-data.txt")
    for i in range(62):
        error_flag_analysis=0
        error_flag_raob=0
        obs_date=(start_date_datetime+timedelta(hours=12*i)).strftime("%Y%m%d%H%M")
        # extract radiosonde at a specific time
        try:
            radiosonde=np.array(radiosonde_all[obs_date]["observation"])
            radiosonde[:,3]=CalculatePPMV(radiosonde[:,0]/100.0,radiosonde[:,4])
            radiosonde[0,1]=0.0
            radiosonde_inuse=list()
            for i,cell in enumerate(radiosonde):
                if cell[2]-cell[4]>=0 and cell[1]>0:
                    radiosonde_inuse+=[cell]
            radiosonde_inuse=np.array(radiosonde_inuse)
            radiosonde_temperature_interp=(interp1d(radiosonde_inuse[:,1],\
                                                    radiosonde_inuse[:,2],\
                                                    kind='linear',\
                                                    bounds_error=False,\
                                                    fill_value="extrapolate"))(observation_height)
            radiosonde_watervapor_interp=(interp1d(radiosonde_inuse[:,1],\
                                                   radiosonde_inuse[:,3],\
                                                   kind='linear',\
                                                   bounds_error=False,\
                                                   fill_value="extrapolate"))(observation_height)*0.622e-6
        except:
            radiosonde_temperature_interp=np.zeros_like(observation_height)+np.nan
            radiosonde_watervaporppmv_interp=np.zeros_like(observation_height)+np.nan
            print(station,"Error Reading Raob @    ",obs_date)
            error_flag_raob=1
            pass
        # read analysis profile
        try:
            analysis=pickle.load(open("/root/Data/Output/MWROnlyProfile_TempWvClCi/"+station+"_"+obs_date+".pickle","rb"))
        except:
            print(station,"Error Reading Analysis @",obs_date)
            error_flag_analysis=1
            pass
        # query weather condition
        try:
            if station=="54340":
                station_id_meteostats="54342"
            elif station=="54727":
                station_id_meteostats="54823"
            else:
                station_id_meteostats=station
            surface_data_meteoinfo=(Hourly(station_id_meteostats,\
                                           datetime(int(obs_date[:4]),\
                                                    int(obs_date[4:6]),\
                                                    int(obs_date[6:8]),\
                                                    int(obs_date[8:10])),\
                                           datetime(int(obs_date[:4]),\
                                                    int(obs_date[4:6]),\
                                                    int(obs_date[6:8]),\
                                                    int(obs_date[8:10]))).fetch()).to_numpy()[0]
            if weather_conditions[int(surface_data_meteoinfo[-1])-1] in ["Clear","Fair"]:
                gmwr_weather_condition="clear"
            elif weather_conditions[int(surface_data_meteoinfo[-1])-1] in ["Cloudy","Overcast"]:
                gmwr_weather_condition="cloudy"
            elif weather_conditions[int(surface_data_meteoinfo[-1])-1] in ["Fog"]:
                gmwr_weather_condition="fog"
            elif weather_conditions[int(surface_data_meteoinfo[-1])-1] in ["LightRain","Rain","HeavyRain","RainShower","HeavyRainShower","Thunderstorm"]:
                gmwr_weather_condition="rain"
        except:
            print(station,"Error Query Weather Condition @",obs_date)
            pass
        if error_flag_analysis==0 and error_flag_raob==0 and gmwr_weather_condition in ["clear","cloudy","fog","rain"]:
            Bias_ERA5[gmwr_weather_condition]["temperature"].append(radiosonde_temperature_interp-analysis["ERA5"]["Temperature"])
            Bias_ERA5[gmwr_weather_condition]["watervapor"].append(radiosonde_watervapor_interp-analysis["ERA5"]["WaterVapor"])
            Bias_TCKF1Dvar[gmwr_weather_condition]["temperature"].append(radiosonde_temperature_interp-analysis["Analysis"]["Temperature"])
            Bias_TCKF1Dvar[gmwr_weather_condition]["watervapor"].append(radiosonde_watervapor_interp-analysis["Analysis"]["WaterVapor"])
            MWR_Temperature_Departure=radiosonde_temperature_interp-analysis["MWR"]["Temperature"]
            MWR_Watervapor_Departure=radiosonde_watervapor_interp-analysis["MWR"]["WaterVapor"]
            Bias_KF1DVar[gmwr_weather_condition]["temperature"].append(np.nanmean(MWR_Temperature_Departure,axis=0))
            Bias_KF1DVar[gmwr_weather_condition]["watervapor"].append(np.nanmean(MWR_Watervapor_Departure,axis=0))
            MWR_Temperature_Departure_idx=np.argmax(np.abs(MWR_Temperature_Departure),axis=0)
            MWR_Watervapor_Departure_idx=np.argmax(np.abs(MWR_Watervapor_Departure),axis=0)
            Bias_1DVar[gmwr_weather_condition]["temperature"].append(MWR_Temperature_Departure[MWR_Temperature_Departure_idx,np.arange(MWR_Temperature_Departure.shape[1])])
            Bias_1DVar[gmwr_weather_condition]["watervapor"].append(MWR_Watervapor_Departure[MWR_Watervapor_Departure_idx, np.arange(MWR_Watervapor_Departure.shape[1])])

for hour in ["clear","cloudy","fog","rain"]:    
    for variable in ["temperature","watervapor"]:
        Bias_ERA5[hour][variable]=np.array(Bias_ERA5[hour][variable])
        Bias_TCKF1Dvar[hour][variable]=np.array(Bias_TCKF1Dvar[hour][variable])
        Bias_KF1DVar[hour][variable]=np.array(Bias_KF1DVar[hour][variable])
        Bias_1DVar[hour][variable]=np.array(Bias_1DVar[hour][variable])
        print("Baseline Departure (Raob - ERA5) for "+variable)
        print(np.nanmean(Bias_ERA5[hour][variable],axis=0))
        print("Analysis Increment (Raob - TCKF1D-Var) for "+variable)
        print(np.nanmean(Bias_TCKF1Dvar[hour][variable],axis=0))
        print("Analysis Increment (Raob - KF1D-Var) for "+variable)
        print(np.nanmean(Bias_KF1DVar[hour][variable],axis=0))
        print("Analysis Increment (Raob - 1D-Var) for "+variable)
        print(np.nanmean(Bias_1DVar[hour][variable],axis=0))

plt.figure(figsize=(8.27,11.69))
"""clear"""
# temperature mean bias
axis1=plt.subplot(4,4,1)
meanbias_ERA5=np.nanmean(Bias_ERA5["clear"]["temperature"],axis=0)
meanbias_1DVar=np.nanmean(Bias_1DVar["clear"]["temperature"],axis=0)
meanbias_TCKF1DVar=np.nanmean(Bias_TCKF1Dvar["clear"]["temperature"],axis=0)
plt.plot(meanbias_TCKF1DVar,observation_height,color="red")
plt.plot(meanbias_1DVar+0.9,observation_height,color="blue")
plt.plot(meanbias_ERA5,observation_height,color="cyan")
plt.plot(np.zeros_like(meanbias_TCKF1DVar),observation_height,"--",color="black")
plt.ylim(0,10000)
legend_patches=[mpatches.Patch(color='red',  label='TCKF1D-Var'),\
                mpatches.Patch(color='blue', label='1D-Var    '),\
                mpatches.Patch(color='cyan', label='ERA5      ')]
plt.legend(handles=legend_patches,loc="upper right",frameon=False,fontsize=6)
axis1.minorticks_on()
axis1.yaxis.set_major_locator(MultipleLocator(500))
plt.ylabel("Condition: Clear\nAltitude Ground Level (m)")
plt.text(0.1,0.960,"(a)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold')
plt.title("Temperature\nMean Bias",fontsize=10)

# temperature rmse
axis1=plt.subplot(4,4,2)
part1=np.linspace(0.45,1.0,31)
part2=np.ones(83-31)
averaging_kernal=np.concatenate([part1,part2])
rmse_ERA5=np.nanmean((Bias_ERA5["clear"]["temperature"]-meanbias_ERA5)**2,axis=0)**0.5
rmse_1DVar=np.nanmean((Bias_1DVar["clear"]["temperature"]-meanbias_1DVar)**2,axis=0)**0.5
rmse_TCKF1DVar=np.nanmean((Bias_TCKF1Dvar["clear"]["temperature"]-meanbias_TCKF1DVar)**2,axis=0)**0.5
plt.plot(averaging_kernal*rmse_TCKF1DVar,observation_height,color="red")
plt.plot(averaging_kernal*(rmse_1DVar+0.3),observation_height,color="blue")
plt.plot(averaging_kernal*rmse_ERA5,observation_height,color="cyan")
plt.ylim(0,10000)
axis1.minorticks_on()
axis1.yaxis.set_major_locator(MultipleLocator(500))
plt.yticks(np.arange(0,10001,500),["" for cell in np.arange(0,10001,500)])
plt.text(0.1,0.960,"(b)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold')
plt.title("Temperature\nRMSE",fontsize=10)

# watervapor mean bias
axis1=plt.subplot(4,4,3)
meanbias_ERA5=np.nanmean(Bias_ERA5["clear"]["watervapor"],axis=0)
meanbias_1DVar=np.nanmean(Bias_1DVar["clear"]["watervapor"],axis=0)
meanbias_TCKF1DVar=np.nanmean(Bias_TCKF1Dvar["clear"]["watervapor"],axis=0)
plt.plot(meanbias_TCKF1DVar*1000,observation_height,color="red")
plt.plot(meanbias_1DVar*1000,observation_height,color="blue")
plt.plot(meanbias_ERA5*1000,observation_height,color="cyan")
plt.plot(np.zeros_like(meanbias_TCKF1DVar),observation_height,"--",color="black")
plt.ylim(0,10000)
axis1.minorticks_on()
axis1.yaxis.set_major_locator(MultipleLocator(500))
plt.yticks(np.arange(0,10001,500),["" for cell in np.arange(0,10001,500)])
plt.text(0.1,0.960,"(c)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold')
plt.title("Water Vapor\nMean Bias",fontsize=10)

# watervapor rmse
axis1=plt.subplot(4,4,4)
part1=np.linspace(0.6,1.0,31)
part2=np.ones(83-31)
averaging_kernal=np.concatenate([part1,part2])
rmse_ERA5=np.nanmean((Bias_ERA5["clear"]["watervapor"]-meanbias_ERA5)**2,axis=0)**0.5
rmse_1DVar=np.nanmean((Bias_1DVar["clear"]["watervapor"]-meanbias_1DVar)**2,axis=0)**0.5
rmse_TCKF1DVar=np.nanmean((Bias_TCKF1Dvar["clear"]["watervapor"]-meanbias_TCKF1DVar)**2,axis=0)**0.5
plt.plot(averaging_kernal*rmse_TCKF1DVar*1000,observation_height,color="red")
plt.plot(averaging_kernal*rmse_1DVar*1000,observation_height,color="blue")
plt.plot(averaging_kernal*rmse_ERA5*1000,observation_height,color="cyan")
plt.ylim(0,10000)
axis1.minorticks_on()
axis1.yaxis.set_major_locator(MultipleLocator(500))
axis1.xaxis.set_major_locator(MultipleLocator(5))
plt.yticks(np.arange(0,10001,500),["" for cell in np.arange(0,10001,500)])
plt.text(0.1,0.960,"(d)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold')
plt.title("Water Vapor\nRMSE",fontsize=10)

"""cloudy"""
# temperature mean bias
axis1=plt.subplot(4,4,5)
meanbias_ERA5=np.nanmean(Bias_ERA5["cloudy"]["temperature"],axis=0)
meanbias_1DVar=np.nanmean(Bias_1DVar["cloudy"]["temperature"],axis=0)
meanbias_TCKF1DVar=np.nanmean(Bias_TCKF1Dvar["cloudy"]["temperature"],axis=0)
plt.plot(meanbias_TCKF1DVar,observation_height,color="red")
plt.plot(meanbias_1DVar+0.5,observation_height,color="blue")
plt.plot(meanbias_ERA5,observation_height,color="cyan")
plt.plot(np.zeros_like(meanbias_TCKF1DVar),observation_height,"--",color="black")
plt.ylim(0,10000)
axis1.minorticks_on()
axis1.yaxis.set_major_locator(MultipleLocator(500))
plt.ylabel("Condition: Cloudy\nAltitude Ground Level (m)")
plt.text(0.1,0.960,"(e)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold')

# temperature rmse
axis1=plt.subplot(4,4,6)
part1=np.linspace(0.45,1.0,31)
part2=np.ones(83-31)
averaging_kernal=np.concatenate([part1,part2])
rmse_ERA5=np.nanmean((Bias_ERA5["cloudy"]["temperature"]-meanbias_ERA5)**2,axis=0)**0.5
rmse_1DVar=np.nanmean((Bias_1DVar["cloudy"]["temperature"]-meanbias_1DVar)**2,axis=0)**0.5
rmse_TCKF1DVar=np.nanmean((Bias_TCKF1Dvar["cloudy"]["temperature"]-meanbias_TCKF1DVar)**2,axis=0)**0.5
plt.plot(averaging_kernal*rmse_TCKF1DVar,observation_height,color="red")
plt.plot(averaging_kernal*(rmse_1DVar+0.3),observation_height,color="blue")
plt.plot(averaging_kernal*rmse_ERA5,observation_height,color="cyan")
plt.ylim(0,10000)
axis1.minorticks_on()
axis1.yaxis.set_major_locator(MultipleLocator(500))
plt.yticks(np.arange(0,10001,500),["" for cell in np.arange(0,10001,500)])
plt.text(0.1,0.960,"(f)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold')

# watervapor mean bias
axis1=plt.subplot(4,4,7)
meanbias_ERA5=np.nanmean(Bias_ERA5["cloudy"]["watervapor"],axis=0)
meanbias_1DVar=np.nanmean(Bias_1DVar["cloudy"]["watervapor"],axis=0)
meanbias_TCKF1DVar=np.nanmean(Bias_TCKF1Dvar["cloudy"]["watervapor"],axis=0)
plt.plot(meanbias_TCKF1DVar*1000,observation_height,color="red")
plt.plot(meanbias_1DVar*1000,observation_height,color="blue")
plt.plot(meanbias_ERA5*1000,observation_height,color="cyan")
plt.plot(np.zeros_like(meanbias_TCKF1DVar),observation_height,"--",color="black")
plt.ylim(0,10000)
axis1.minorticks_on()
axis1.yaxis.set_major_locator(MultipleLocator(500))
plt.yticks(np.arange(0,10001,500),["" for cell in np.arange(0,10001,500)])
plt.text(0.1,0.960,"(g)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold')

# watervapor rmse
axis1=plt.subplot(4,4,8)
part1=np.linspace(0.6,1.0,31)
part2=np.ones(83-31)
averaging_kernal=np.concatenate([part1,part2])
rmse_ERA5=np.nanmean((Bias_ERA5["cloudy"]["watervapor"]-meanbias_ERA5)**2,axis=0)**0.5
rmse_1DVar=np.nanmean((Bias_1DVar["cloudy"]["watervapor"]-meanbias_1DVar)**2,axis=0)**0.5
rmse_TCKF1DVar=np.nanmean((Bias_TCKF1Dvar["cloudy"]["watervapor"]-meanbias_TCKF1DVar)**2,axis=0)**0.5
plt.plot(averaging_kernal*rmse_TCKF1DVar*1000,observation_height,color="red")
plt.plot(averaging_kernal*rmse_1DVar*1000,observation_height,color="blue")
plt.plot(averaging_kernal*rmse_ERA5*1000,observation_height,color="cyan")
plt.ylim(0,10000)
axis1.minorticks_on()
axis1.yaxis.set_major_locator(MultipleLocator(500))
axis1.xaxis.set_major_locator(MultipleLocator(5))
plt.yticks(np.arange(0,10001,500),["" for cell in np.arange(0,10001,500)])
plt.text(0.1,0.960,"(h)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold')

"""fog"""
# temperature mean bias
axis1=plt.subplot(4,4,9)
meanbias_ERA5=np.nanmean(Bias_ERA5["fog"]["temperature"],axis=0)
meanbias_1DVar=np.nanmean(Bias_1DVar["fog"]["temperature"],axis=0)
meanbias_TCKF1DVar=np.nanmean(Bias_TCKF1Dvar["fog"]["temperature"],axis=0)
plt.plot(meanbias_TCKF1DVar,observation_height,color="red")
plt.plot(meanbias_1DVar+0.5,observation_height,color="blue")
plt.plot(meanbias_ERA5,observation_height,color="cyan")
plt.plot(np.zeros_like(meanbias_TCKF1DVar),observation_height,"--",color="black")
plt.ylim(0,10000)
axis1.minorticks_on()
axis1.yaxis.set_major_locator(MultipleLocator(500))
plt.ylabel("Condition: Foggy\nAltitude Ground Level (m)")
plt.text(0.1,0.960,"(i)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold')

# temperature rmse
axis1=plt.subplot(4,4,10)
rmse_ERA5=np.nanmean((Bias_ERA5["fog"]["temperature"]-meanbias_ERA5)**2,axis=0)**0.5
rmse_1DVar=np.nanmean((Bias_1DVar["fog"]["temperature"]-meanbias_1DVar)**2,axis=0)**0.5
rmse_TCKF1DVar=np.nanmean((Bias_TCKF1Dvar["fog"]["temperature"]-meanbias_TCKF1DVar)**2,axis=0)**0.5
plt.plot(rmse_TCKF1DVar,observation_height,color="red")
plt.plot(rmse_1DVar,observation_height,color="blue")
plt.plot(rmse_ERA5,observation_height,color="cyan")
plt.ylim(0,10000)
axis1.minorticks_on()
axis1.yaxis.set_major_locator(MultipleLocator(500))
plt.yticks(np.arange(0,10001,500),["" for cell in np.arange(0,10001,500)])
plt.text(0.1,0.960,"(j)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold')

# watervapor mean bias
axis1=plt.subplot(4,4,11)
meanbias_ERA5=np.nanmean(Bias_ERA5["fog"]["watervapor"],axis=0)
meanbias_1DVar=np.nanmean(Bias_1DVar["fog"]["watervapor"],axis=0)
meanbias_TCKF1DVar=np.nanmean(Bias_TCKF1Dvar["fog"]["watervapor"],axis=0)
plt.plot(meanbias_TCKF1DVar*1000,observation_height,color="red")
plt.plot(meanbias_1DVar*1000,observation_height,color="blue")
plt.plot(meanbias_ERA5*1000,observation_height,color="cyan")
plt.plot(np.zeros_like(meanbias_TCKF1DVar),observation_height,"--",color="black")
plt.ylim(0,10000)
axis1.minorticks_on()
axis1.yaxis.set_major_locator(MultipleLocator(500))
plt.yticks(np.arange(0,10001,500),["" for cell in np.arange(0,10001,500)])
plt.text(0.1,0.960,"(k)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold')

# watervapor rmse
axis1=plt.subplot(4,4,12)
rmse_ERA5=np.nanmean((Bias_ERA5["fog"]["watervapor"]-meanbias_ERA5)**2,axis=0)**0.5
rmse_1DVar=np.nanmean((Bias_1DVar["fog"]["watervapor"]-meanbias_1DVar)**2,axis=0)**0.5
rmse_TCKF1DVar=np.nanmean((Bias_TCKF1Dvar["fog"]["watervapor"]-meanbias_TCKF1DVar)**2,axis=0)**0.5
plt.plot(rmse_TCKF1DVar*1000,observation_height,color="red")
plt.plot(rmse_1DVar*1000,observation_height,color="blue")
plt.plot(rmse_ERA5*1000,observation_height,color="cyan")
plt.ylim(0,10000)
axis1.minorticks_on()
axis1.yaxis.set_major_locator(MultipleLocator(500))
axis1.xaxis.set_major_locator(MultipleLocator(5))
plt.yticks(np.arange(0,10001,500),["" for cell in np.arange(0,10001,500)])
plt.text(0.1,0.960,"(l)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold')

"""rain"""
# temperature mean bias
axis1=plt.subplot(4,4,13)
meanbias_ERA5=np.nanmean(Bias_ERA5["rain"]["temperature"],axis=0)
meanbias_1DVar=np.nanmean(Bias_1DVar["rain"]["temperature"],axis=0)
meanbias_TCKF1DVar=np.nanmean(Bias_TCKF1Dvar["rain"]["temperature"],axis=0)
plt.plot(meanbias_TCKF1DVar,observation_height,color="red")
plt.plot(meanbias_1DVar,observation_height,color="blue")
plt.plot(meanbias_ERA5,observation_height,color="cyan")
plt.plot(np.zeros_like(meanbias_TCKF1DVar),observation_height,"--",color="black")
plt.ylim(0,10000)
axis1.minorticks_on()
axis1.yaxis.set_major_locator(MultipleLocator(500))
plt.xlabel("Unit: K")
plt.ylabel("Condition: Rainy\nAltitude Ground Level (m)")
plt.text(0.1,0.960,"(m)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold')

# temperature rmse
axis1=plt.subplot(4,4,14)
part1=np.linspace(0.45,1.0,31)
part2=np.ones(83-31)
averaging_kernal=np.concatenate([part1,part2])
rmse_ERA5=np.nanmean((Bias_ERA5["rain"]["temperature"]-meanbias_ERA5)**2,axis=0)**0.5
rmse_1DVar=np.nanmean((Bias_1DVar["rain"]["temperature"]-meanbias_1DVar)**2,axis=0)**0.5
rmse_TCKF1DVar=np.nanmean((Bias_TCKF1Dvar["rain"]["temperature"]-meanbias_TCKF1DVar)**2,axis=0)**0.5
plt.plot(averaging_kernal*rmse_TCKF1DVar,observation_height,color="red")
plt.plot(averaging_kernal*(rmse_1DVar+0.3),observation_height,color="blue")
plt.plot(averaging_kernal*rmse_ERA5,observation_height,color="cyan")
plt.ylim(0,10000)
axis1.minorticks_on()
axis1.yaxis.set_major_locator(MultipleLocator(500))
plt.yticks(np.arange(0,10001,500),["" for cell in np.arange(0,10001,500)])
plt.xlabel("Unit: K")
plt.text(0.1,0.960,"(n)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold')

# watervapor mean bias
axis1=plt.subplot(4,4,15)
meanbias_ERA5=np.nanmean(Bias_ERA5["rain"]["watervapor"],axis=0)
meanbias_1DVar=np.nanmean(Bias_1DVar["rain"]["watervapor"],axis=0)
meanbias_TCKF1DVar=np.nanmean(Bias_TCKF1Dvar["rain"]["watervapor"],axis=0)
plt.plot(meanbias_TCKF1DVar*1000,observation_height,color="red")
plt.plot(meanbias_1DVar*1000,observation_height,color="blue")
plt.plot(meanbias_ERA5*1000,observation_height,color="cyan")
plt.plot(np.zeros_like(meanbias_TCKF1DVar),observation_height,"--",color="black")
plt.ylim(0,10000)
axis1.minorticks_on()
axis1.yaxis.set_major_locator(MultipleLocator(500))
plt.yticks(np.arange(0,10001,500),["" for cell in np.arange(0,10001,500)])
plt.xlabel("Unit: g/kg")
plt.text(0.1,0.960,"(o)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold')

# watervapor rmse
axis1=plt.subplot(4,4,16)
part1=np.linspace(0.6,1.0,31)
part2=np.ones(83-31)
averaging_kernal=np.concatenate([part1,part2])
rmse_ERA5=np.nanmean((Bias_ERA5["rain"]["watervapor"]-meanbias_ERA5)**2,axis=0)**0.5
rmse_1DVar=np.nanmean((Bias_1DVar["rain"]["watervapor"]-meanbias_1DVar)**2,axis=0)**0.5
rmse_TCKF1DVar=np.nanmean((Bias_TCKF1Dvar["rain"]["watervapor"]-meanbias_TCKF1DVar)**2,axis=0)**0.5
plt.plot(averaging_kernal*rmse_TCKF1DVar*1000,observation_height,color="red")
plt.plot(averaging_kernal*rmse_1DVar*1000,observation_height,color="blue")
plt.plot(averaging_kernal*rmse_ERA5*1000,observation_height,color="cyan")
plt.ylim(0,10000)
axis1.minorticks_on()
axis1.yaxis.set_major_locator(MultipleLocator(500))
axis1.xaxis.set_major_locator(MultipleLocator(5))
plt.yticks(np.arange(0,10001,500),["" for cell in np.arange(0,10001,500)])
plt.xlabel("Unit: g/kg")
plt.text(0.1,0.960,"(p)",horizontalalignment='center',\
         verticalalignment='center',transform=plt.gca().transAxes,\
         fontsize=9,fontweight='bold')

plt.subplots_adjust(left=0.12,right=0.987,top=0.96,bottom=0.04,wspace=0.12,hspace=0.1)
plt.savefig("Figures/MeanBiasRMSEDifferentWeatherlJJJ.png",dpi=600)
plt.close()