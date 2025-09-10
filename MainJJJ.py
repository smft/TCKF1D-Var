#!/usr/bin/env python

import re
import time
import glob
import cmaps
import pickle
import argparse
from scipy import signal
from numpy.linalg import inv
from scipy.optimize import minimize
from datetime import datetime, timedelta

from FileIO import *
from WSM3Interface import *
from ATMCalculation import *
from MicrowaveInterface import *

def calculaterelativeHumidityfromPPMV(temperature,watervaporppmv,pressure):
    e_observed=watervaporppmv*1e-6*pressure
    es_over_water=GoffGratchWater(temperature)
    es_over_ice=GoffGratchIce(temperature)
    rh=np.where(temperature>273.16,e_observed/es_over_water,e_observed/es_over_ice)
    return rh

def KalmanFilter(simulation,\
                 observation,\
                 PMatrix,\
                 RMatrix):
    nobs,nlevel=np.shape(observation)
    observation_mean=np.nanmean(observation,axis=0)
    observation_mean=np.where(np.isnan(observation_mean)==True,simulation,observation_mean)
    observation_departure=observation-observation_mean
    observation_departure[np.isnan(observation_departure)==True]=0.0
    MMatrix=np.dot(observation_departure.T,observation_departure)/nobs
    result=list()
    RMatrix_updated=np.zeros_like(RMatrix)
    for i in range(nlevel):
        x_estimate=simulation[i]
        P=RMatrix[i,i]
        estimates=list()
        nan_count=0
        for j in range(nobs):
            if np.isnan(observation[j,i])==False:
                x_predict=x_estimate
                P_predict=P+PMatrix[i,i]
                K=P_predict/(P_predict+MMatrix[i,i])
                x_estimate=x_predict+K*(observation[j,i]-x_predict)
                P=(1-K)*P_predict
                estimates.append(x_estimate)
            else:
                nan_count+=1
                continue
        if nan_count==nobs:
            estimates.append(x_estimate)
        result.append(estimates[-1])
        RMatrix_updated[i,i]=P
    return np.array(result),RMatrix_updated

def calculaterho(T,p,H2O_ppmv):
    """
    input:
    T: K
    p: Pa
    H2O_ppmv: PPMV
    return: 
    rho: kg/m3
    """
    R_d = 287.05  # J/(kg·K)
    R_v = 461.5   # J/(kg·K)
    p_v = H2O_ppmv / 1e6 * p
    p_d = p - p_v
    rho = p_d / (R_d * T) + p_v / (R_v * T)
    return rho

def TimeSequenceInterpolation(data_00,data_12,hour_of_day):
    if 0<= hour_of_day<=12:
        val=((12-hour_of_day)*data_00+hour_of_day*data_12)/12
    else:
        val=((24-hour_of_day)*data_00+(hour_of_day-12)*data_12)/12
    return val

def dPFromdTdQ(p0,T0,q0,dT=0.0,dQ=0.0):
    """
    estimate dP (Pa) from dT and dQ
    input :
        p0 : base state pressure (Pa)
        T0 : base state temperaute (K)
        q0 : base state watervapor (kg/kg)
        dT : temperature perturbation (K)
        dQ : watervapor perturbation (kg/kg)
    return:
        dp : pressure perturbation (Pa)
    """
    Tv0=T0*(1+0.61*q0)
    return (p0/Tv0)*((1+0.61*q0)*dT+0.61*T0*dQ)

def PasresERA5toMinutes(obs_date,station_latitude,station_longitude,observation_height):
    era5_temperature,\
    era5_relativehumidity,\
    era5_height,\
    era5_cloudwater,\
    era5_cloudice,\
    era5_cloudrain,\
    era5_cloudsnow,\
    era5_pressure=ReadERA5("/root/Data/OBS/Model/ERA5/JJJ_"+obs_date+"_fullhydro.nc",\
                           station_latitude,\
                           station_longitude)
    era5_dewpoint=CalculateDewPoint(era5_temperature,era5_relativehumidity)
    era5_dewpoint_interpolated=(interp1d(era5_height,\
                                         era5_dewpoint,\
                                         kind='linear',\
                                         bounds_error=False,\
                                         fill_value="extrapolate"))(observation_height)
    era5_pressure_interpolated=(interp1d(era5_height,\
                                         era5_pressure,\
                                         kind='linear',\
                                         bounds_error=False,\
                                         fill_value="extrapolate"))(observation_height)
    era5_temperature_interpolated=(interp1d(era5_height,\
                                            era5_temperature,\
                                            kind='linear',\
                                            bounds_error=False,\
                                            fill_value="extrapolate"))(observation_height)
    era5_relativehumidity_interpolated=(interp1d(era5_height,\
                                                 era5_relativehumidity,\
                                                 kind='linear',\
                                                 bounds_error=False,\
                                                 fill_value="extrapolate"))(observation_height)
    era5_cloudwater_interpolated=(interp1d(era5_height,\
                                           era5_cloudwater,\
                                           kind='linear',\
                                           bounds_error=False,\
                                           fill_value="extrapolate"))(observation_height)
    era5_cloudice_interpolated=(interp1d(era5_height,\
                                         era5_cloudice,\
                                         kind='linear',\
                                         bounds_error=False,\
                                         fill_value="extrapolate"))(observation_height)
    era5_cloudrain_interpolated=(interp1d(era5_height,\
                                          era5_cloudrain,\
                                          kind='linear',\
                                          bounds_error=False,\
                                          fill_value="extrapolate"))(observation_height)
    era5_cloudsnow_interpolated=(interp1d(era5_height,\
                                          era5_cloudsnow,\
                                          kind='linear',\
                                          bounds_error=False,\
                                          fill_value="extrapolate"))(observation_height)
    return era5_pressure_interpolated,\
           era5_temperature_interpolated,\
           era5_dewpoint_interpolated,\
           era5_relativehumidity_interpolated,\
           era5_cloudwater_interpolated,\
           era5_cloudice_interpolated,\
           era5_cloudrain_interpolated,\
           era5_cloudsnow_interpolated

"""test!!!test"""
ts_total=time.time()

# read in variables
parser=argparse.ArgumentParser(description="MWRInversion")
parser.add_argument('a1',metavar='|Station ID|',type=str,help='\n')
parser.add_argument('a2',metavar='|Date (YYYYMMDDHHmm)|',type=str,help='\n')
args=parser.parse_args()
station_id=args.a1
obs_date=args.a2

# calculate time window
obs_date_fmt=datetime(int(obs_date[:4]),\
                      int(obs_date[4:6]),\
                      int(obs_date[6:8]),\
                      int(obs_date[8:10]),\
                      int(obs_date[10:]))
time_window_start=int((obs_date_fmt-timedelta(minutes=30)).strftime("%Y%m%d%H%M"))
time_window_end=int(obs_date)

# read gmwr retrieval
gmwr_observation=dict()
gmwr_brightnesstemperature=list()
for gmwr_retrieval_file in glob.glob("/root/Data/OBS/GMWR/Retrieval/"+station_id+"/*CP_MM.TXT"):
    gmwr_retrieval_datetime=int(re.findall("\d{12}",gmwr_retrieval_file)[0])
    if time_window_start<=gmwr_retrieval_datetime<=time_window_end:
        gmwr_observation[str(gmwr_retrieval_datetime)]=ReadMWRCP(gmwr_retrieval_file)
        gmwr_observation[str(gmwr_retrieval_datetime)]["pressure"]=CalculatePressureAtHeight(\
                                                gmwr_observation[str(gmwr_retrieval_datetime)]["2m_pressure"]*100.0,\
                                                gmwr_observation[str(gmwr_retrieval_datetime)]["2m_temperature"],\
                                                gmwr_observation[str(gmwr_retrieval_datetime)]["height"]\
                                                )
        gmwr_observation[str(gmwr_retrieval_datetime)]["dewpoint"]=CalculateDewPoint(\
                                                gmwr_observation[str(gmwr_retrieval_datetime)]["temperature"],\
                                                gmwr_observation[str(gmwr_retrieval_datetime)]["relativehumidity"]\
                                                )
        gmwr_observation[str(gmwr_retrieval_datetime)]["watervaporppmv"]=CalculatePPMV(\
                                                gmwr_observation[str(gmwr_retrieval_datetime)]["pressure"]/100.0,\
                                                gmwr_observation[str(gmwr_retrieval_datetime)]["dewpoint"]\
                                                )
        if 500.0<=gmwr_observation[str(gmwr_retrieval_datetime)]["2m_pressure"]<=1060.0:
            simulation_bt_retrieval,\
            jacobian_temperature_retrieval,\
            jacobian_watervapor_retrieval,\
            jacobian_cloudwater_retrieval=HATPOR(\
                                gmwr_observation[str(gmwr_retrieval_datetime)]["station_latitude"],\
                                gmwr_observation[str(gmwr_retrieval_datetime)]["station_longitude"],\
                                gmwr_observation[str(gmwr_retrieval_datetime)]["station_elevation"]/1000.0,\
                                gmwr_observation[str(gmwr_retrieval_datetime)]["2m_pressure"],\
                                gmwr_observation[str(gmwr_retrieval_datetime)]["2m_temperature"],\
                                np.array(gmwr_observation[str(gmwr_retrieval_datetime)]["temperature"],order="F")[::-1],\
                                np.array(gmwr_observation[str(gmwr_retrieval_datetime)]["pressure"],order="F")[::-1],\
                                np.array(gmwr_observation[str(gmwr_retrieval_datetime)]["watervaporppmv"],order="F")[::-1],\
                                np.array(gmwr_observation[str(gmwr_retrieval_datetime)]["liquidwatercontent"],order="F")[::-1]\
                                )
            gmwr_brightnesstemperature.append(simulation_bt_retrieval)
        else:
            simulation_bt_retrieval=np.zeros(14)+np.nan
        gmwr_observation[str(gmwr_retrieval_datetime)]["brightnesstemperature"]=simulation_bt_retrieval
gmwr_brightnesstemperature=np.array(gmwr_brightnesstemperature)

# read era5 analysis
era5_station_latitude=gmwr_observation[obs_date]["station_latitude"]
era5_station_longitude=gmwr_observation[obs_date]["station_longitude"]
era5_observation_height=gmwr_observation[obs_date]["station_elevation"]+gmwr_observation[obs_date]["height"]
if obs_date[-2:]=="00":
    era5_pressure_interpolated,\
    era5_temperature_interpolated,\
    era5_dewpoint_interpolated,\
    era5_relativehumidity_interpolated,\
    era5_cloudwater_interpolated,\
    era5_cloudice_interpolated,\
    era5_cloudrain_interpolated,\
    era5_cloudsnow_interpolated=PasresERA5toMinutes(obs_date,\
                                                    era5_station_latitude,\
                                                    era5_station_longitude,\
                                                    era5_observation_height)
else:
    obs_date_datetime=datetime.strptime(obs_date,"%Y%m%d%H%M")
    obs_date_bf=(obs_date_datetime-timedelta(minutes=30)).strftime("%Y%m%d%H%M")
    obs_date_af=(obs_date_datetime+timedelta(minutes=30)).strftime("%Y%m%d%H%M")
    era5_pressure_interpolated_bf,\
    era5_temperature_interpolated_bf,\
    era5_dewpoint_interpolated_bf,\
    era5_relativehumidity_interpolated_bf,\
    era5_cloudwater_interpolated_bf,\
    era5_cloudice_interpolated_bf,\
    era5_cloudrain_interpolated_bf,\
    era5_cloudsnow_interpolated_bf=PasresERA5toMinutes(obs_date_bf,\
                                                       era5_station_latitude,\
                                                       era5_station_longitude,\
                                                       era5_observation_height)
    era5_pressure_interpolated_af,\
    era5_temperature_interpolated_af,\
    era5_dewpoint_interpolated_af,\
    era5_relativehumidity_interpolated_af,\
    era5_cloudwater_interpolated_af,\
    era5_cloudice_interpolated_af,\
    era5_cloudrain_interpolated_af,\
    era5_cloudsnow_interpolated_af=PasresERA5toMinutes(obs_date_af,\
                                                       era5_station_latitude,\
                                                       era5_station_longitude,\
                                                       era5_observation_height)
    era5_pressure_interpolated=(era5_pressure_interpolated_bf+\
                                era5_pressure_interpolated_af)/2.0
    era5_temperature_interpolated=(era5_temperature_interpolated_bf+\
                                   era5_temperature_interpolated_af)/2.0
    era5_dewpoint_interpolated=(era5_dewpoint_interpolated_bf+\
                                era5_dewpoint_interpolated_af)/2.0
    era5_relativehumidity_interpolated=(era5_relativehumidity_interpolated_bf+\
                                        era5_relativehumidity_interpolated_af)/2.0
    era5_cloudwater_interpolated=(era5_cloudwater_interpolated_bf+\
                                  era5_cloudwater_interpolated_af)/2.0
    era5_cloudice_interpolated=(era5_cloudice_interpolated_bf+\
                                era5_cloudice_interpolated_af)/2.0
    era5_cloudrain_interpolated=(era5_cloudrain_interpolated_bf+\
                                 era5_cloudrain_interpolated_af)/2.0
    era5_cloudsnow_interpolated=(era5_cloudsnow_interpolated_bf+\
                                 era5_cloudsnow_interpolated_af)/2.0

era5_watervaporppmv_interpolated=CalculatePPMV(era5_pressure_interpolated/100.0,era5_dewpoint_interpolated)
era5_simulation_bt,\
era5_jacobian_temperature,\
era5_jacobian_watervapor,\
era5_jacobian_cloudwater=HATPOR(gmwr_observation[obs_date]["station_latitude"],\
                                gmwr_observation[obs_date]["station_longitude"],\
                                gmwr_observation[obs_date]["station_elevation"]/1000.0,\
                                gmwr_observation[obs_date]["2m_pressure"],\
                                gmwr_observation[obs_date]["2m_temperature"],\
                                np.array(era5_temperature_interpolated,order="F")[::-1],\
                                np.array(era5_pressure_interpolated,order="F")[::-1],\
                                np.array(era5_watervaporppmv_interpolated,order="F")[::-1],\
                                np.array(era5_cloudwater_interpolated+\
                                         era5_cloudice_interpolated+\
                                         era5_cloudrain_interpolated+\
                                         era5_cloudsnow_interpolated,order="F")[::-1])
nchannel,nlevel=np.shape(era5_jacobian_temperature)

print(gmwr_brightnesstemperature[-1,:])
print(era5_simulation_bt)

# read covariances
mwr_Rmatrix=pickle.load(open("/root/Scripts/Matrix/GMWRRMatrixJJJ.pickle","rb"))

# kalman filter clear sky assumption
synthetic_bt_clear,mwr_Rmatrix_clear=KalmanFilter(era5_simulation_bt,\
                                                  gmwr_brightnesstemperature,\
                                                  np.zeros_like(mwr_Rmatrix["clear"]),\
                                                  mwr_Rmatrix["clear"]*0.25)

# kalman filter cloudy sky assumption
synthetic_bt_cloudy,mwr_Rmatrix_cloudy=KalmanFilter(era5_simulation_bt,\
                                                    gmwr_brightnesstemperature,\
                                                    np.zeros_like(mwr_Rmatrix["cloudy"]),\
                                                    mwr_Rmatrix["cloudy"]*0.25)

# kalman filter cloudy fog assumption
synthetic_bt_fog,mwr_Rmatrix_fog=KalmanFilter(era5_simulation_bt,\
                                              gmwr_brightnesstemperature,\
                                              np.zeros_like(mwr_Rmatrix["cloudy"]),\
                                              mwr_Rmatrix["fog"]*0.5)

# kalman filter cloudy precipitation assumption
synthetic_bt_rain,mwr_Rmatrix_rain=KalmanFilter(era5_simulation_bt,\
                                                gmwr_brightnesstemperature,\
                                                np.zeros_like(mwr_Rmatrix["cloudy"]),\
                                                mwr_Rmatrix["cloudy"]*0.75)

# select observation
synthetic_bt_total=np.array([synthetic_bt_clear,\
                             synthetic_bt_cloudy,\
                             synthetic_bt_fog,\
                             synthetic_bt_rain])
departure_total=np.array([np.abs(synthetic_bt_clear-gmwr_observation[obs_date]["brightnesstemperature"]),\
                          np.abs(synthetic_bt_cloudy-gmwr_observation[obs_date]["brightnesstemperature"]),\
                          np.abs(synthetic_bt_fog-gmwr_observation[obs_date]["brightnesstemperature"]),\
                          np.abs(synthetic_bt_rain-gmwr_observation[obs_date]["brightnesstemperature"])])
synthetic_bt_idx=np.argmax(departure_total,axis=0)
synthetic_bt_raw=synthetic_bt_total[synthetic_bt_idx,np.arange(synthetic_bt_total.shape[1])]
departure_OmF=np.abs(synthetic_bt_raw-era5_simulation_bt)
synthetic_bt=np.where(departure_OmF<=2.2,synthetic_bt_raw,era5_simulation_bt)

#if total_departure_clear>total_departure_cloudy:
#    synthetic_bt=synthetic_bt_clear
#    mwr_Rmatrix=mwr_Rmatrix_clear
#    bt_departure=synthetic_bt_clear-era5_simulation_bt
#else:
#    synthetic_bt=synthetic_bt_cloudy
#    mwr_Rmatrix=mwr_Rmatrix_cloudy
#    bt_departure=synthetic_bt_cloudy-era5_simulation_bt

# set cost function
def costfunction(addedvalue):
    """
    Units:
    addedvalue[:nlevel]         -------> K
    addedvalue[nlevel:2*nlevel] -------> PPMV*1e-3
    addedvalue[3*nlevel:]       -------> hPa
    Direction:
    index #1                    -------> surface
    index #X                    -------> top of atmosphere
    """
    # please contact zhangqi@cnhyc.com for this section

# set perturbation for temperature, water vapor, pressure
addedvalue=np.zeros(nlevel*3)
Bias_ERA5_Spline=pickle.load(open("/root/Data/Output/Matrix/BiasFunction.pickle","rb"))
bias_temperature=TimeSequenceInterpolation(Bias_ERA5_Spline["00"]["temperature"](era5_observation_height),\
                                           Bias_ERA5_Spline["12"]["temperature"](era5_observation_height),\
                                           int(obs_date[-4:-2]))
bias_watervapor=TimeSequenceInterpolation(Bias_ERA5_Spline["00"]["watervapor"](era5_observation_height),\
                                          Bias_ERA5_Spline["12"]["watervapor"](era5_observation_height),\
                                          int(obs_date[-4:-2]))
random_kernal=0.45+np.random.uniform(-0.05,0.05,size=nlevel)
addedvalue[:nlevel]=bias_temperature*random_kernal
addedvalue[nlevel:2*nlevel]=bias_watervapor/0.622e-3
addedvalue[2*nlevel:]=dPFromdTdQ(era5_pressure_interpolated,\
                                 era5_temperature_interpolated,\
                                 0.622*era5_watervaporppmv_interpolated*1e-6,\
                                 dT=addedvalue[:nlevel],\
                                 dQ=0.622e-3*addedvalue[nlevel:2*nlevel])/100.0

# minimization
a=minimize(costfunction,addedvalue,method='L-BFGS-B')
temperature_addedvalue=a.x[:nlevel]
watervapor_addedvalue=a.x[nlevel:2*nlevel]*1e3
pressure_addedvalue=a.x[2*nlevel:]*100.0
analysis_temperature=era5_temperature_interpolated+temperature_addedvalue
analysis_pressure=era5_pressure_interpolated+pressure_addedvalue
analysis_watervaporppmv=era5_watervaporppmv_interpolated+watervapor_addedvalue

# update hydrometeor
atm_state={'T':analysis_temperature,\
           'p':analysis_pressure,\
           'rho':calculaterho(analysis_temperature,\
                              analysis_pressure,\
                              analysis_watervaporppmv),\
           'qv':0.622*analysis_watervaporppmv*1e-6,\
           'qc':era5_cloudwater_interpolated+era5_cloudice_interpolated,\
           'qr':era5_cloudrain_interpolated+era5_cloudsnow_interpolated}
cloudmp_cfg=WSM3Config()
cloudmp_dt=1
for n in range(1800):
    wsm3_step(atm_state,cloudmp_dt,cloudmp_cfg)

# save result
MWR_Temperature=list()
MWR_WaterVapor=list()
MWR_Pressure=list()
MWR_CloudWater=list()

for cell_obs_datetime in gmwr_observation.keys():
    MWR_Temperature.append(gmwr_observation[cell_obs_datetime]["temperature"])
    MWR_WaterVapor.append(gmwr_observation[cell_obs_datetime]["specifichumidity"])
    MWR_Pressure.append(gmwr_observation[cell_obs_datetime]["pressure"])
    MWR_CloudWater.append(gmwr_observation[cell_obs_datetime]["liquidwatercontent"])

flag_save=open("/root/Data/Output/MWROnlyProfile_TempWvClCi/"+station_id+"_"+obs_date+".pickle","wb")
result={"Analysis"  :{"Temperature":atm_state["T"],\
                      "WaterVapor" :atm_state["qv"],\
                      "Pressure"   :atm_state["p"],\
                      "CloudWater" :atm_state["qc"],\
                      "CloudRain"  :atm_state["qr"]},\
        "ERA5"      :{"Temperature":era5_temperature_interpolated,\
                      "WaterVapor" :0.622*era5_watervaporppmv_interpolated*1e-6,\
                      "Pressure"   :era5_pressure_interpolated,\
                      "CloudWater" :era5_cloudwater_interpolated+era5_cloudice_interpolated,\
                      "CloudRain"  :era5_cloudrain_interpolated+era5_cloudsnow_interpolated},\
        "MWR"       :{"Temperature":MWR_Temperature,\
                      "WaterVapor" :MWR_WaterVapor,\
                      "Pressure"   :MWR_Pressure,\
                      "CloudWater" :MWR_CloudWater},\
        "StationLat":gmwr_observation[obs_date]["station_latitude"],\
        "StationLon":gmwr_observation[obs_date]["station_longitude"],\
        "StationEle":gmwr_observation[obs_date]["station_elevation"],\
        "Height"    :gmwr_observation[obs_date]["height"]}
pickle.dump(result,flag_save)
flag_save.close()
