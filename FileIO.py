#!/usr/bin/env python3

import re
import pygrib
import warnings
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
warnings.simplefilter("ignore")

# Constants Section
PLANCK_CONSTANT              = 6.62607015e-34  # Planck constant (J·s)
SPEED_OF_LIGHT               = 3e8             # Speed of light (m/s)
BOLTZMANN_CONSTANT           = 1.380649e-23    # Boltzmann constant (J/K)
WAVELENGTH                   = 387e-9          # Raman wavelength (m)
AVOGADRO_CONSTANT            = 6.02214076e23   # Avogadro constant (1/mol)
R                            = 8.314462618     # Ideal gas constant (J/(mol·K))
N_S                          = 2.546899e25     # Molecule number density under standard state(m^-3)
NITROGEN_DEPOLARIZATION_RATIO= 0.0279          # Nitrogen Depolarization Ratio
ATOMSPHERE_REFRACTIVE_INDEX  = 1.000293        # Atmosphere refractive index
# End Constants Section 

def ReadKaCloudRadarLevel1(filename):
    generic_header=[("magic_number","i4"),                        #OK
                    ("major_version","i2"),                       #OK
                    ("minor_version","i2"),                       #OK
                    ("generic_type","i4"),                        #OK
                    ("res1","20c")]                               #OK
    generic_header_size=32

    site_config=[("site_code","S8"),                              #OK
                 ("site_name","S24"),                             #OK
                 ("Latitude","f4"),                               #OK
                 ("Longitude","f4"),                              #OK
                 ("antenna_height","f4"),                         #OK
                 ("ground_height","f4"),                          #OK
                 ("amend_north", "f4"),                           #OK
                 ("version", "i2"),                               #OK
                 ("radar_type", "i2"),                            #OK
                 ("manufacturer", "S6"),                          #OK
                 ("res2", "10c")]                                 #OK
    site_config_size=72

    radar_config=[("frequency","f4"),                             #OK
                  ("wavelength ","f4"),                           #OK
                  ("beam_width_hori","f4"),                       #OK
                  ("beam_width_vert","f4"),                       #OK
                  ("transmitter_peak_power","f4"),                #OK
                  ("antenna_gain","f4"),                          #OK
                  ("total_loss","f4"),                            #OK
                  ("receiver_gain","f4"),                         #OK
                  ("first_side","f4"),                            #OK
                  ("receiver_dynamic_range","f4"),                #OK
                  ("receiver_sensitivity","f4"),                  #OK
                  ("band_width","f4"),                            #OK
                  ("max_explore_range","i4"),                     #OK
                  ("distance_solution","i2"),                     #OK
                  ("polarization_type","i2"),                     #OK
                  ("res3","96c")]                                 #OK
    radar_config_size=152

    task_config=[("task_name","S16"),                             #OK
                 ("task_description","S96"),                      #OK
                 ("polarization_way","i2"),                       #OK
                 ("scan_type","i2"),                              #OK
                 ("pulse_width_1","i4"),                          #OK
                 ("pulse_width_2","i4"),                          #OK
                 ("pulse_width_3","i4"),                          #OK
                 ("pulse_width_4","i4"),                          #OK
                 ("scan_start_time","f8"),                        #OK
                 ("cut_number","i4"),                             #OK
                 ("horizontal_noise","f4"),                       #OK
                 ("vertical_noise","f4"),                         #OK
                 ("horizontal_alibration_1","f4"),                #OK
                 ("horizontal_alibration_2","f4"),                #OK
                 ("horizontal_alibration_3","f4"),                #OK
                 ("horizontal_alibration_4","f4"),                #OK
                 ("vertical_valibration_1","f4"),                 #OK
                 ("vertical_valibration_2","f4"),                 #OK
                 ("vertical_valibration_3","f4"),                 #OK
                 ("vertical_valibration_4","f4"),                 #OK
                 ("horizontal_noise_temperature","f4"),           #OK
                 ("vertical_noise_temperature","f4"),             #OK
                 ("zdr_calibration ","f4"),                       #OK
                 ("phidp_calibration","f4"),                      #OK
                 ("ldr_calibration","f4"),                        #OK
                 ("number_of_coherent_accumulation_1","S1"),      #OK
                 ("number_of_coherent_accumulation_2","S1"),      #OK
                 ("number_of_coherent_accumulation_3","S1"),      #OK
                 ("number_of_coherent_accumulation_4","S1"),      #OK
                 ("fft_count_1","i2"),                            #OK
                 ("fft_count_2","i2"),                            #OK
                 ("fft_count_3","i2"),                            #OK
                 ("fft_count_4","i2"),                            #OK
                 ("accumulation_of_power_spectrum_1","S1"),       #OK
                 ("accumulation_of_power_spectrum_2","S1"),       #OK
                 ("accumulation_of_power_spectrum_3","S1"),       #OK
                 ("accumulation_of_power_spectrum_4","S1"),       #OK
                 ("pulse_width_1_starting_position","i4"),        #OK
                 ("pulse_width_2_starting_position","i4"),        #OK
                 ("pulse_width_3_starting_position","i4"),        #OK
                 ("pulse_width_4_starting_position","i4"),        #OK
                 ("res4","20c")]                                  #OK
    task_config_size=256

    cut_config=[("process_mode","i2"),                            #OK
                ("wave_form","i2"),                               #OK
                ("prf_1","f4"),                                   #OK
                ("prf_2","f4"),                                   #OK
                ("prf_3","f4"),                                   #OK
                ("prf_4","f4"),                                   #OK
                ("prf_mode","i2"),                                #OK
                ("pulse_width_combination_mode","i2"),            #OK
                ("azimuth","f4"),                                 #OK
                ("elevation","f4"),                               #OK
                ("start_angle ","f4"),                            #OK
                ("end_angle","f4"),                               #OK
                ("angular_resolution","f4"),                      #OK
                ("scan_speed","f4"),                              #OK
                ("log_resolution","i4"),                          #OK
                ("doppler_resolution","i4"),                      #OK
                ("start_range","i4"),                             #OK
                ("phase_mode","i4"),                              #OK
                ("atmospheric_loss","f4"),                        #OK
                ("nyquist_speed","f4"),                           #OK
                ("misc_filter_mask","i4"),                        #OK
                ("sqi_threshold","f4"),                           #OK
                ("sig_threshold","f4"),                           #OK
                ("csr_threshold","f4"),                           #OK
                ("log_threshold ","f4"),                          #OK
                ("cpa_threshold","f4"),                           #OK
                ("pmi_threshold","f4"),                           #OK
                ("dplog_threshold","f4"),                         #OK
                ("thresholds_r","S12"),                           #OK
                ("dbt_mask","i4"),                                #OK
                ("dbz_mask","i4"),                                #OK
                ("velocity_mask","i4"),                           #OK
                ("spectrum_width_mask","i4"),                     #OK
                ("dp_mask","i4"),                                 #OK
                ("mask_reserved","12c"),                          #OK
                ("scan_sync","i4"),                               #OK
                ("direction","i4"),                               #OK
                ("ground_clutter_classifier_type","i2"),          #OK
                ("ground_clutter_filter_type","i2"),              #OK
                ("ground_clutter_filter_notch_width","i2"),       #OK
                ("ground_clutter_filter_window","i2"),            #OK
                ("res5","92c")]                                   #OK
    cut_config_size=256

    radial_header=[("radial_state","i2"),                         #OK
                   ("spot_blank","i2"),                           #OK
                   ("sequence_number","i2"),                      #OK
                   ("radial_number","i2"),                        #OK
                   ("moment_number","i2"),                        #OK
                   ("elevation_number","i2"),                     #OK
                   ("azimuth_radial","f4"),                       #OK
                   ("elevation_radial","f4"),                     #OK
                   ("seconds_radial","f8"),                       #OK
                   ("microseconds","i4"),                         #OK
                   ("length_of_data","i4"),                       #OK
                   ("seconds_radial_1","i2"),                     #OK
                   ("max_fft_count","i2"),                        #OK
                   ("res6","24c")]                                #OK
    radial_header_size=64

    moment_header=[("data_type","i2"),                            #OK
                   ("scale","i2"),                                #OK
                   ("offset","i2"),                               #OK
                   ("bin_bytes","i2"),                            #OK
                   ("bin_number","i2"),                           #OK
                   ("flags","i2"),                                #OK
                   ("data_length","i4"),                          #OK
                   ("res7","16c")]                                #OK
    moment_header_size=32

    # read step 1
    data_types=generic_header+site_config+radar_config+task_config
    read_out=np.fromfile(filename,dtype=np.dtype(data_types))[0]
    data_info=dict()
    for i,cell in enumerate(read_out):
        data_info[data_types[i][0]]=cell

    # read step 2
    data_types=list()
    for i in range(data_info["cut_number"]):
        data_types+=cut_config
    offset=generic_header_size+site_config_size+radar_config_size+task_config_size
    read_out=np.fromfile(filename,dtype=np.dtype(data_types),offset=offset)[0]
    for i,cell in enumerate(read_out):
        data_info[data_types[i][0]]=cell

    # read step 3
    offset+=cut_config_size*int(data_info["cut_number"])
    read_out=np.fromfile(filename,dtype=np.dtype(radial_header),offset=offset)[0]
    for i,cell in enumerate(read_out):
        data_info[radial_header[i][0]]=cell

    # read step 4
    obs_info_all=list()
    obs_data_all=list()
    offset+=radial_header_size
    for i in range(data_info["moment_number"]):
        # read moment header info
        read_out=np.fromfile(filename,dtype=np.dtype(moment_header),offset=offset)[0]
        obs_info=dict()
        for i,cell in enumerate(read_out):
            obs_info[moment_header[i][0]]=cell
        obs_info_all+=[obs_info]
        offset+=moment_header_size
        # read moment data
        if int(obs_info["bin_bytes"])==1:
            moment_data=[(("radarobs%05d" % i),"i1") for i in range(obs_info["bin_number"])]
        if int(obs_info["bin_bytes"])==2:
            moment_data=[(("radarobs%05d" % i),"i2") for i in range(obs_info["bin_number"])]
        moment_data_size=int(obs_info["bin_bytes"])*int(obs_info["bin_number"])
        read_out=np.fromfile(filename,dtype=np.dtype(moment_data),offset=offset)[0]
        offset+=moment_data_size
        obs_data=list()
        for cell in read_out:
            obs_data+=[int(cell)]
        obs_data_all+=[obs_data]
    return data_info,obs_info_all,obs_data_all

def ReadLidarL0(filename):
    file_header=[("magic_number","14c"),                       #reserved
                 ("data_level","i2"),                          #observed data level
                 ("record_version","i2"),                      #record version number
                 ("instrument_id","i4"),                       #instrument id
                 ("site_longitude","i4"),                      #site latitude
                 ("site_latitude","i4"),                       #site longitude
                 ("site_elevation","i4"),                      #site elevation
                 ("reserved_1","2c"),                          #first reserved location
                 ("scan_mode","i2"),                           #scan mode
                 ("record_start_second","i4"),                 #record start time in seconds (UTC)
                 ("record_end_second","i4"),                   #record start time in seconds (UTC)
                 ("record_julian_date","i2"),                  #record julian date
                 ("elevation_angle","i2"),                     #elevation angle
                 ("reserved_2","2c"),                          #second reserved location
                 ("emitted_laser_wavelength_1","i2"),          #first emitted laser wavelength
                 ("emitted_laser_wavelength_2","i2"),          #second emitted laser wavelength
                 ("emitted_laser_wavelength_3","i2"),          #third emitted laser wavelength
                 ("receiver_channel_amount","i2")]             #receiver channel amount
    file_header_chunk_size=60
    
    channel_header=[("channel_id","i2"),                       #channel id
                    ("recevier_type","i2"),                    #receiver type (first digit indicates receiver type, rest digits indicate wavelength)
                    ("polarization_type","i2"),                #polarization type
                    ("distance_resolution","i2"),              #distance resolution
                    ("blind_range_height","i2"),               #observation blind range height
                    ("observation_start_offset","i4"),         #channel observation start offset
                    ("observation_amount","i2")]               #channel observation amount
    channel_header_chunk_size=16
    
    # read file header
    file_info=dict()
    data_record=np.fromfile(filename,dtype=np.dtype(file_header))[0]
    for i,cell in enumerate(file_header):
        file_info[cell[0]]=data_record[i]
    # read channel header
    for i in range(file_info["receiver_channel_amount"]):
        file_info[("Channel%03dInfo" % (i+1))]=dict()
        read_start_offset=file_header_chunk_size+i*channel_header_chunk_size
        data_record=np.fromfile(filename,dtype=np.dtype(channel_header),offset=read_start_offset)[0]
        for j,cell in enumerate(channel_header):
            file_info[("Channel%03dInfo" % (i+1))][cell[0]]=data_record[j]
    # read channel observation
    for i in range(file_info["receiver_channel_amount"]):
        file_info[("Channel%03dData" % (i+1))]=None
        if i==0:
            read_start_offset=file_header_chunk_size+channel_header_chunk_size*file_info["receiver_channel_amount"]
        else:
            read_start_offset=file_header_chunk_size+channel_header_chunk_size*file_info["receiver_channel_amount"]
            for j in range(1,i+1):
                read_start_offset+=file_info[("Channel%03dInfo" % (i+1))]["observation_amount"]*4
        data_header=[(("obs%06d" % ii),"f4") for ii in range(file_info[("Channel%03dInfo" % (i+1))]["observation_amount"])]
        data_record=np.fromfile(filename,dtype=np.dtype(data_header),offset=read_start_offset)[0]
        file_info[("Channel%03dData" % (i+1))]=np.array(list(data_record))
    return file_info

def ReadLidarL1(filename):
    product_names=["355nm Mie Extinction Coefficient",         #01
                   "355nm Mie Backscatter Coefficient",        #02
                   "355nm Mie Depolarization Ratio",           #03
                   "386nm Raman Extinction Coefficient",       #04
                   "386nm Raman Backscatter Coefficient",      #05
                   "532nm Mie Extinction Coefficient",         #06
                   "532nm Mie Backscatter Coefficient",        #07
                   "532nm Mie Depolarization Ratio",           #08
                   "607nm Raman Extinction Coefficient",       #09
                   "607nm Raman Backscatter Coefficient",      #10
                   "1064nm Mie Extinction Coefficient",        #11
                   "1064nm Mie Backscatter Coefficient",       #12
                   "Water Vapor Mixing Ratio"]                 #13
    file_header=[("magic_number","14c"),                       #reserved
                 ("data_level","i2"),                          #observed data level
                 ("record_version","i2"),                      #record version number
                 ("instrument_id","i4"),                       #instrument id
                 ("site_longitude","i4"),                      #site latitude
                 ("site_latitude","i4"),                       #site longitude
                 ("site_elevation","i4"),                      #site elevation
                 ("distance_resolution","i2"),                 #first reserved location
                 ("scan_mode","i2"),                           #scan mode
                 ("record_start_second","i4"),                 #record start time in seconds (UTC)
                 ("record_end_second","i4"),                   #record start time in seconds (UTC)
                 ("record_julian_date","i2"),                  #record julian date
                 ("elevation_angle","i2")]                     #elevation angle
    file_header_chunk_size=50
    channel_header=[("product_id","i2"),                       #second reserved location
                    ("observation_amount","i2")]               #first emitted laser wavelength
    channel_header_chunk_size=4
    file_info=dict()
    # read file header
    data_record=np.fromfile(filename,dtype=np.dtype(file_header))[0]
    for i,cell in enumerate(file_header):
        file_info[cell[0]]=data_record[i]
    # read data
    for i in range(13):
        if i==0:
            record_start_offset=file_header_chunk_size
        else:
            record_start_offset+=file_info[product_names[i-1]]["observation_amount"]*4
        file_info[product_names[i]]=dict()
        data_record=np.fromfile(filename,dtype=np.dtype(channel_header),offset=record_start_offset)[0]
        file_info[product_names[i]]["product_id"]=data_record[0]
        file_info[product_names[i]]["observation_amount"]=data_record[1]
        record_start_offset+=channel_header_chunk_size
        data_header=[(("obs%06d" % j),"f4") for j in range(data_record[1])]
        data_record=np.fromfile(filename,dtype=np.dtype(data_header),offset=record_start_offset)[0]
        file_info[product_names[i]]["observation_data"]=np.array(list(data_record))
    return file_info

def ParserIGRARadiosondeRecord(filename):
    result=dict()
    with open(filename,'r') as file:
        lines=file.readlines()
        for line in lines:
            if line[0]=="#":
                station_id=line[1:12]
                station_lat=line[55:62]
                station_lon=line[63:71]
                observation_year=line[13:17]
                observation_month=line[18:20]
                observation_day=line[21:23]
                observation_validate_time=line[24:26]
                observation_start_time=line[27:31]
                observation_valid_datetime=observation_year+\
                                           observation_month+\
                                           observation_day+\
                                           observation_validate_time+"00"
                result[observation_valid_datetime]={"station_id":station_id,\
                                                    "latitude":float(station_lat)*1e-4,\
                                                    "longitude":float(station_lon)*1e-4,\
                                                    "observation_start_datetime":observation_year+\
                                                                                 observation_month+\
                                                                                 observation_day+\
                                                                                 observation_start_time,\
                                                    "observation":list()}
            else:
                elasticminutes=float((line[9:15])[:-2])
                elasticseconds=float((line[9:15])[-2:])
                pressure=float(line[9:15])
                geopotential=float(line[16:21])
                temperature=float(line[22:27])*1e-1+273.15
                relativehumidity=float(line[28:33])*1e-1
                dewpoint=-float(line[34:39])*1e-1+temperature
                winddirection=float(line[40:45])
                windspeed=float(line[46:51])*1e-1
                result[observation_valid_datetime]["observation"]+=[[pressure,\
                                                                     geopotential,\
                                                                     temperature,\
                                                                     relativehumidity,\
                                                                     dewpoint,\
                                                                     winddirection,\
                                                                     windspeed,\
                                                                     elasticminutes*60+elasticseconds]]
    return result

def ParserGFSForecast(filename,latitude,longitude):
    flag=pygrib.open(filename)

    # read latitude longitude
    cell=flag.select(name="Temperature",typeOfLevel='surface')[0]
    lat,lon=cell.latlons()
    dist=np.sqrt((lat-latitude)**2+(lon-longitude)**2)
    idx_y,idx_x=np.unravel_index(dist.argmin(),dist.shape)

    # read surface pressure
    cell=flag.select(name="Surface pressure",typeOfLevel='surface')[0]
    surface_pressure=cell.values[idx_y,idx_x]

    # read surface height
    cell=flag.select(name="Orography",typeOfLevel='surface')[0]
    station_height=cell.values[idx_y,idx_x]

    # read temperature
    temperature=list()
    pressure_temperature=list()
    for cell in flag.select(name="Temperature",typeOfLevel='isobaricInPa'):
        if cell.level<surface_pressure:
            temperature+=[cell.values[idx_y,idx_x]]
            pressure_temperature+=[cell.level]
    for cell in flag.select(name="Temperature",typeOfLevel='isobaricInhPa'):
        if cell.level*100<surface_pressure:
            temperature+=[cell.values[idx_y,idx_x]]
            pressure_temperature+=[cell.level*100]
    cell=flag.select(name="2 metre temperature",typeOfLevel='heightAboveGround')[0]
    temperature+=[cell.values[idx_y,idx_x]]
    pressure_temperature+=[surface_pressure]
    temperature=np.array(temperature)
    pressure_temperature=np.array(pressure_temperature)

    # read relative humidity
    relativehumidity=list()
    pressure_relativehumidity=list()
    for cell in flag.select(name="Relative humidity",typeOfLevel='isobaricInPa'):
        if cell.level<surface_pressure:
            relativehumidity+=[cell.values[idx_y,idx_x]]
            pressure_relativehumidity+=[cell.level]
    for cell in flag.select(name="Relative humidity",typeOfLevel='isobaricInhPa'):
        if cell.level*100<surface_pressure:
            relativehumidity+=[cell.values[idx_y,idx_x]]
            pressure_relativehumidity+=[cell.level*100]
    cell=flag.select(name="2 metre relative humidity",typeOfLevel='heightAboveGround')[0]
    relativehumidity+=[cell.values[idx_y,idx_x]]
    pressure_relativehumidity+=[surface_pressure]
    relativehumidity=np.array(relativehumidity)
    pressure_relativehumidity=np.array(pressure_relativehumidity)
    
    # read geopotientail
    geopotential=list()
    pressure_geopotential=list()
    for cell in flag.select(name="Geopotential height",typeOfLevel='isobaricInPa'):
        if cell.level<surface_pressure:
            geopotential+=[cell.values[idx_y,idx_x]]
            pressure_geopotential+=[cell.level]
    for cell in flag.select(name="Geopotential height",typeOfLevel='isobaricInhPa'):
        if cell.level*100<surface_pressure:
            geopotential+=[cell.values[idx_y,idx_x]]
            pressure_geopotential+=[cell.level*100]
    geopotential+=[station_height]
    pressure_geopotential+=[surface_pressure]
    geopotential=np.array(geopotential)-station_height
    pressure_geopotential=np.array(pressure_geopotential)

    # read u wind
    uwind=list()
    pressure_uwind=list()
    for cell in flag.select(name="U component of wind",typeOfLevel='isobaricInPa'):
        if cell.level<surface_pressure:
            uwind+=[cell.values[idx_y,idx_x]]
            pressure_uwind+=[cell.level]
    for cell in flag.select(name="U component of wind",typeOfLevel='isobaricInhPa'):
        if cell.level*100<surface_pressure:
            uwind+=[cell.values[idx_y,idx_x]]
            pressure_uwind+=[cell.level*100]
    cell=flag.select(name="10 metre U wind component",typeOfLevel='heightAboveGround')[0]
    uwind+=[cell.values[idx_y,idx_x]]
    pressure_uwind+=[surface_pressure]
    uwind=np.array(uwind)
    pressure_uwind=np.array(pressure_uwind)

    # read v wind
    vwind=list()
    pressure_vwind=list()
    for cell in flag.select(name="V component of wind",typeOfLevel='isobaricInPa'):
        if cell.level<surface_pressure:
            vwind+=[cell.values[idx_y,idx_x]]
            pressure_vwind+=[cell.level]
    for cell in flag.select(name="V component of wind",typeOfLevel='isobaricInhPa'):
        if cell.level*100<surface_pressure:
            vwind+=[cell.values[idx_y,idx_x]]
            pressure_vwind+=[cell.level*100]
    cell=flag.select(name="10 metre V wind component",typeOfLevel='heightAboveGround')[0]
    vwind+=[cell.values[idx_y,idx_x]]
    pressure_vwind+=[surface_pressure]
    vwind=np.array(vwind)
    pressure_vwind=np.array(pressure_vwind)
    
    flag.close()
    return temperature[::-1],\
           relativehumidity[::-1],\
           uwind[::-1],\
           vwind[::-1],\
           geopotential[::-1],\
           pressure_geopotential[::-1]

def ReadMWRRaw(filename):
    raw_data=open(filename,"r").read().split("\n")[1:-1]
    output={"station_id":None,\
            "station_latitude":None,\
            "station_longitude":None,\
            "station_elevation":None,\
            "channel_information":None,\
            "2m_temperature":None,\
            "2m_pressure":None,\
            "2m_relativehumidity":None,\
            "sky_tir":None,\
            "channel_observation":None}
    output["channel_observation"]=list()
    output["channel_observation"]=list()
    output["station_id"]=raw_data[0].split(",")[0]
    output["station_longitude"]=float(raw_data[0].split(",")[1])
    output["station_latitude"]=float(raw_data[0].split(",")[2])
    output["station_elevation"]=float(raw_data[0].split(",")[3])
    output["2m_temperature"]=float(raw_data[2].split(",")[2])
    output["2m_relativehumidity"]=float(raw_data[2].split(",")[3])
    output["2m_pressure"]=float(raw_data[2].split(",")[4])
    output["sky_tir"]=float(raw_data[2].split(",")[5])
    output["channel_information"]=list()
    output["channel_observation"]=list()
    for i,cell in enumerate(raw_data[1].split(",")):
        if "Ch " in cell and cell not in ["Ch 30.000","Ch 55.500"]:
            output["channel_information"].append(float(cell[3:]))
            output["channel_observation"].append(float(raw_data[2].split(",")[i]))
    return output

def ReadMWRCP(filename):
    raw_data=open(filename,"r").read().split("\n")[1:-1]
    output={"station_id":None,\
            "station_latitude":None,\
            "station_longitude":None,\
            "station_elevation":None,\
            "nlevels":None,\
            "2m_temperature":None,\
            "2m_pressure":None,\
            "2m_relativehumidity":None,\
            "sky_tir":None,\
            "precipitation":None,\
            "height":None,\
            "temperature":None,\
            "specifichumidity":None,\
            "relativehumidity":None,\
            "liquidwatercontent":None}
    line_0=raw_data[0].split(",")
    line_1=raw_data[1].split(",")
    line_2=raw_data[2].split(",")
    line_3=raw_data[3].split(",")
    line_4=raw_data[4].split(",")
    line_5=raw_data[5].split(",")
    output["station_id"]         =line_0[0]
    output["station_longitude"]  =float(line_0[1])
    output["station_latitude"]   =float(line_0[2])
    output["station_elevation"]  =float(line_0[3])
    output["nlevels"]            =int(line_0[-1])
    output["2m_temperature"]     =float(line_2[3])+273.15
    output["2m_pressure"]        =float(line_2[5])
    output["2m_relativehumidity"]=float(line_2[4])
    output["sky_tir"]            =float(line_2[6])+273.15
    output["precipitation"]      =int(line_2[7])
    output["height"]             =np.array([float(cell[:-4])*1000.0 for cell in line_1[11:11+output["nlevels"]]])
    output["temperature"]        =np.array([float(cell)+273.15 for cell in line_2[11:11+output["nlevels"]]])
    output["specifichumidity"]   =np.array([float(cell)/1000.0 for cell in line_3[11:11+output["nlevels"]]])
    output["relativehumidity"]   =np.array([float(cell) for cell in line_4[11:11+output["nlevels"]]])
    output["liquidwatercontent"] =np.array([float(cell)/1000.0 for cell in line_5[11:11+output["nlevels"]]])
    return output

def ReadERA5(filename,sitelat,sitelon):
    flag=Dataset(filename)
    temperature=flag.variables["t"][0,:,:,:]
    qvapor=flag.variables["r"][0,:,:,:]
    geopotential=flag.variables["z"][0,:,:,:]
    cloudwater=flag.variables["clwc"][0,:,:,:]
    cloudice=flag.variables["ciwc"][0,:,:,:]
    try:
        cloudrain=flag.variables["crwc"][0,:,:,:]
    except:
        cloudrain=np.zeros_like(cloudwater)
        pass
    try:
        cloudsnow=flag.variables["cswc"][0,:,:,:]
    except:
        cloudsnow=np.zeros_like(cloudwater)
        pass
    pressure=np.array([1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 
                        700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225,
                        200, 175, 150, 125, 100, 70,  50])*100.0
    latitude=flag.variables["latitude"][:]
    longitude=flag.variables["longitude"][:]
    idx_y=np.unravel_index(np.abs(latitude-sitelat).argmin(),latitude.shape)
    idx_x=np.unravel_index(np.abs(longitude-sitelon).argmin(),latitude.shape)
    return temperature[:,idx_y,idx_x][:,0],\
           qvapor[:,idx_y,idx_x][:,0],\
           geopotential[:,idx_y,idx_x][:,0]/9.80665,\
           cloudwater[:,idx_y,idx_x][:,0],\
           cloudice[:,idx_y,idx_x][:,0],\
           cloudrain[:,idx_y,idx_x][:,0],\
           cloudsnow[:,idx_y,idx_x][:,0],\
           pressure

def ReadAWSS3GRASIProfile(filename):
    flag=Dataset(filename)
    latitude=flag.variables["refLatitude"][:]
    longitude=flag.variables["refLongitude"][:]
    pressure=flag.variables["pressure"][:]
    temperature=flag.variables["temperature"][:]
    watervapor=flag.variables["waterVaporPressure"][:]
    height=flag.variables["altitude"][:]
    return latitude,longitude,pressure,temperature,watervapor,height

def ReadNSMCGNOSIIProfile(filename):
    flag=Dataset(filename)
    latitude=flag.getncattr("lat")
    longitude=flag.getncattr("lon")
    pressure=flag.variables["Pres"][:]*100.0
    temperature=flag.variables["Temp"][:]
    watervapor=flag.variables["Shum"][:]/1000.00
    height=flag.variables["Geop"][:]*1000.00
    return latitude,longitude,pressure,temperature,watervapor,height

def ReadGNSSMET(filename):
    raw_data=open(filename).read().split("\n")[1]
    observation=list()
    for cell in raw_data.split(" "):
        if cell!="":
            observation.append(cell)
    return float(observation[2]),\
           float(observation[3]),\
           float(observation[4]),\
           float(observation[11]),\
           float(observation[17])

def ReadEarthCARECPR_CLD_2A(filename):
    flag=Dataset(filename)
    latitude=flag["ScienceData"]["latitude"][:].data
    longitude=flag["ScienceData"]["longitude"][:].data
    relative_height=flag["ScienceData"]["height"][:].data-\
                    (flag["ScienceData"]["geoid_offset"][:].data)[:,np.newaxis]-\
                    (flag["ScienceData"]["surface_elevation"][:].data)[:,np.newaxis]
    relative_height=np.where(relative_height>=9e+36,np.nan,relative_height)
    retrieval_type=flag["ScienceData"]["retrieval_classification"][:].data
    water_content=flag["ScienceData"]["water_content"][:].data
    water_content=np.where(water_content>=9e+36,np.nan,water_content)
    liquid_water_content=flag["ScienceData"]["liquid_water_content"][:].data
    liquid_water_content=np.where(liquid_water_content>=9e+36,np.nan,liquid_water_content)
    ice_water_path=flag["ScienceData"]["ice_water_path"][:].data
    ice_water_path=np.where(ice_water_path>=9e+36,np.nan,ice_water_path)
    rain_water_path=flag["ScienceData"]["rain_water_path"][:].data
    rain_water_path=np.where(rain_water_path>=9e+36,np.nan,rain_water_path)
    obs_start,obs_end=re.findall("\d{8}T\d{6}Z",filename.split("/")[-1])
    flag.close()
    return latitude,\
           longitude,\
           relative_height,\
           retrieval_type,\
           water_content,\
           liquid_water_content,\
           ice_water_path,\
           rain_water_path,\
           datetime.strptime(obs_start,"%Y%m%dT%H%M%SZ"),\
           datetime.strptime(obs_end,"%Y%m%dT%H%M%SZ")