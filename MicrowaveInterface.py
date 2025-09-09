#!/usr/bin/env python

import PyRTTOVGB
import numpy as np

def HATPOR(station_latitude,\
           station_longitude,\
           station_elevation,\
           station_2m_pressure,\
           station_2m_temperature,\
           temperature_profile,\
           pressure_profile,\
           watervapor_profile,\
           cloudwater_profile):
    nlevels=np.shape(temperature_profile)[0]
    output_bt=np.array(np.zeros(14),order="F")
    temperature_jacobian=np.array(np.zeros([14,nlevels]),order="F")
    watervapor_jacobian=np.array(np.zeros([14,nlevels]),order="F")
    cloudwater_jacobian=np.array(np.zeros([14,nlevels]),order="F")
    PyRTTOVGB.hatprointerface(nlevels=nlevels,\
                              station_latitude=station_latitude,\
                              station_longitude=station_longitude,\
                              station_elevation=station_elevation,\
                              input_temperature_profile=np.array(temperature_profile,dtype=np.float32,order="F"),\
                              input_pressure_profile=np.array(pressure_profile/100.0,dtype=np.float32,order="F"),\
                              input_watervapor_profile=np.array(watervapor_profile,dtype=np.float32,order="F"),\
                              input_cloudwater_profile=np.array(cloudwater_profile,dtype=np.float32,order="F"),\
                              input_2m_pressure=station_2m_pressure,\
                              input_2m_temperature=station_2m_temperature,\
                              output_brightness_temperature=output_bt,\
                              output_jacobian_temperature=temperature_jacobian,\
                              output_jacobian_watervapor=watervapor_jacobian,\
                              output_jacobian_cloudwater=cloudwater_jacobian)
    return output_bt,temperature_jacobian,watervapor_jacobian,cloudwater_jacobian

def GMWRObservationQualityControl(observation,\
                                  simulation,\
                                  tir,\
                                  temperature_2m,\
                                  prcp_false_thresh,\
                                  prcp_false_bt_rmse):
    if tir-temperature_2m<=prcp_false_thresh:
       channel_inuse=list()
       departure=np.abs(observation-simulation)
       for channel in [0,1,2,3,4]:
           if departure[channel]<=prcp_false_bt_rmse[channel]:
                  channel_inuse.append(channel)
       return channel_inuse+[11,12,13],"Sunny"
    else:
       return [11,12,13],"Rainy"
