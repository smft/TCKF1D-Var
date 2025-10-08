#!/usr/bin/env python

import cdsapi
from datetime import datetime, timedelta

client=cdsapi.Client()
dataset = "reanalysis-era5-pressure-levels"
startdate=datetime(year=2025,month=6,day=30,hour=00)
for i in range(1*24):
    nowdate=startdate+timedelta(hours=i)
    nowdate_fmt=nowdate.strftime("%Y%m%d%H")
    request={"product_type": ["reanalysis"],
             "variable": ["geopotential",
                          "relative_humidity",
                          "specific_cloud_ice_water_content",
                          "specific_cloud_liquid_water_content",
                          "specific_rain_water_content",
                          "specific_snow_water_content",
                          "specific_humidity",
                          "temperature",
                          "u_component_of_wind",
                          "v_component_of_wind"],
             "year": [nowdate_fmt[:4]],
             "month": [nowdate_fmt[4:6]],
             "day": [nowdate_fmt[6:8]],
             "time": [nowdate_fmt[8:]+":00"],
             "pressure_level": ["50" , "70" , "100", "125",
                                "150", "175", "200", "225",
                                "250", "300", "350", "400",
                                "450", "500", "550", "600",
                                "650", "700", "750", "775",
                                "800", "825", "850", "875",
                                "900", "925", "950", "975",
                                "1000"],
             "data_format": "netcdf",
             "download_format": "unarchived",
             "area": [43.5, 110, 33, 125]}
    client.retrieve(dataset,request,"/root/Data/OBS/Model/ERA5/JJJ_"+nowdate_fmt+"00_fullhydro.nc")