#!/usr/bin/env python

import os
import time
import glob
import cmaps
import pickle
import argparse
from scipy import signal
from astral.sun import sun
from numpy.linalg import inv
from astral import LocationInfo
from scipy.interpolate import UnivariateSpline
from datetime import datetime, timedelta

from FileIO import *
from ATMCalculation import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator,ScalarFormatter
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

"""test!!!test"""
for line in open("/root/Scripts/IntersectionList.txt","r").read().split("\n"):
    intesection_info=line.split(",")
    latitude,\
    longitude,\
    relative_height,\
    retrieval_type,\
    water_content,\
    liquid_water_content,\
    ice_water_path,\
    rain_water_path,\
    obs_start_datetime,\
    obs_end_datetime=ReadEarthCARECPR_CLD_2A(intesection_info[0])
    for idx in intesection_info[1:-1]:
        station_id=list(gmwr_station_location.keys())[int(idx)]
        gmwr_filenames="/root/Data/Output/MWROnlyProfile_TempWvClCi/"+station_id+"_"+intesection_info[-1]+".pickle"
        print(gmwr_filenames,os.path.isfile(gmwr_filenames))