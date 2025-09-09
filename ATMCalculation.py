#!/usr/bin/env python3

import warnings
import numpy as np
from scipy.interpolate import interp1d
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

def CalculateRelativeHumidity(temperature,dewpoint):
    """
    Calculate Relative Humidity from radiosonde observation using Goff-Gratch function
    Parameters:
        temperature (numpy array): temperature (K)
        dewpoint (numpy array)   : dewpoint temperature (K)
    Returns:
        numpy array              : relative humidity (0 - 1)
    """
    es_over_water=GoffGratchWater(temperature)
    e_over_water=GoffGratchWater(dewpoint)
    es_over_ice=GoffGratchIce(temperature)
    e_over_ice=GoffGratchIce(dewpoint)
    rh=np.where(temperature>273.16,e_over_water/es_over_water,e_over_ice/es_over_ice)
    rh[rh>1]=np.nan
    rh[rh<0]=np.nan
    return rh

def GoffGratchWater(temp_k):
    """
    Goff-Gratch function over water
    Parameters:
        temp_k (float): temperature (K)
    Returns:
        float: saturated water vapor pressure (Pa)
    """
    t_ref=373.15
    es_ref=1013.25
    ln_es=(-7.90298*(t_ref/temp_k-1)+5.02808*np.log10(t_ref/temp_k)-\
            1.3816e-7*(10**(11.344*(1-temp_k/t_ref))-1)+\
            8.1328e-3*(10**(-3.49149*(t_ref/temp_k-1))-1))+np.log10(es_ref)
    es=10**ln_es
    return es*100

def GoffGratchWaterInv(es):
    """
    Goff-Gratch function over water
    Parameters:
        temp_k (float): temperature (K)
    Returns:
        float: saturated water vapor pressure (Pa)
    """
    t_ref=373.15
    es_ref=1013.25
    ln_es=(-7.90298*(t_ref/temp_k-1)+5.02808*np.log10(t_ref/temp_k)-\
            1.3816e-7*(10**(11.344*(1-temp_k/t_ref))-1)+\
            8.1328e-3*(10**(-3.49149*(t_ref/temp_k-1))-1))+np.log10(es_ref)
    es=10**ln_es
    return es*100

def GoffGratchIce(temp_k):
    """
    Goff-Gratch function over ice
    Parameters:
        temp_k (float): temperature (K)
    Returns:
        float: saturated water vapor pressure (Pa)
    """
    t_ref=273.16
    es_ref=6.112
    ln_es=(-9.09718*(t_ref/temp_k-1)-\
            3.56654*np.log10(t_ref/temp_k)+\
            0.876793 * (1 - temp_k / t_ref)+np.log10(es_ref))
    es=10**ln_es
    return es*100

def CalculateDewPoint(temp_k,rh):
    """
    calculate dew point
    Parameters:
        temp_k (float): temperature (K)
        rh (float): relative humidity (%)
    Returns:
        float: dew point (K)
    """
    # Constants for Goff-Gratch formula
    a = 17.27
    b = 237.7
    c = 237.7
    # Constants for Ice surface (slightly modified values compared to water)
    ice_a = 22.46
    ice_b = 272.62
    ice_c = 272.62
    # Calculate the gamma function (γ) for both water and ice surfaces
    gamma=np.log(rh/100.0)+(a*(temp_k-273.15))/(b+(temp_k-273.15))
    gamma_ice=np.log(rh/100.0)+(ice_a*(temp_k-273.15))/(ice_b+(temp_k-273.15))
    # Calculate the dew point using the Goff-Gratch formula
    dew_point=np.where(temp_k>=273.15,(b*gamma)/(a-gamma),(b*gamma_ice)/(a-gamma_ice))
    return dew_point+273.15

def CalculatePPMV(pressure,dewpoint):
    """
    calculate dew point
    Parameters:
        pressure (float): temperature (hPa)
        dewpoint (float): relative humidity (K)
    Returns:
        float: PPMV
    """
    dewpoint_c=dewpoint-273.15
    e=6.112*np.exp((17.67*dewpoint_c)/(dewpoint_c+243.5))
    ppmv=(e/pressure)*1e6
    return ppmv

def CalculateRelativeHumidityFromPPMV(temperature,watervaporppmv,pressure):
    e_observed=watervaporppmv*1e-6*pressure
    es_over_water=GoffGratchWater(temperature)
    es_over_ice=GoffGratchIce(temperature)
    rh=np.where(temperature>273.16,e_observed/es_over_water,e_observed/es_over_ice)
    return rh

def CalculateLWCfromKaBanddBZ(dBZ):
    """
    calculate cloud liquid water content from Ka-band Radar
    Parameters:
        dBZ (float): reflectivity (-35 ~ 20 dBZ)
    Returns:
        float: liquid water content (g/m3)
    """
    dBZ_inuse=np.where(dBZ<-35,np.nan,dBZ)
    dBZ_inuse=np.where(dBZ>=20,20,dBZ)
    a=0.1
    b=0.038
    return a*10**(b*dBZ_inuse)

def CalculateIWCfromKaBanddBZ(dBZ):
    """
    calculate cloud ice water content from Ka-band Radar
    Parameters:
        dBZ (float): reflectivity (-35 ~ 20 dBZ)
    Returns:
        float: ice water content (g/m3)
    """
    dBZ_inuse=np.where(dBZ<-35,np.nan,dBZ)
    dBZ_inuse=np.where(dBZ>=20,20,dBZ)
    a=0.02
    b=0.058
    return a*10**(b*dBZ_inuse)

def CalculateRWCfromKaBanddBZ(dBZ):
    """
    calculate cloud rain water content from Ka-band Radar
    Parameters:
        dBZ (float): reflectivity (20 ~ 35 dBZ)
    Returns:
        float: liquid water content (g/m3)
    """
    dBZ_inuse=np.where(dBZ>=35,35,dBZ)
    a_r=0.036
    b_r=0.062
    return a_r*10**(b_r*dBZ_inuse)

def CalculateSWCfromKaBanddBZ(dBZ):
    """
    calculate cloud snow water content from Ka-band Radar
    Parameters:
        dBZ (float): reflectivity (20 ~ 35 dBZ)
    Returns:
        float: liquid water content (g/m3)
    """
    dBZ_inuse=np.where(dBZ>=35,35,dBZ)
    a_s=0.0038
    b_s=0.092
    return a_s*10**(b_s*dBZ_inuse)

def CalculateHydroemteorRetrieval(dBZ,temperature):
    """
    calculate ice water content from Ka-band Radar
    Parameters:
        dBZ (float): reflectivity (-35 ~ 20 dBZ)
        temperature (float): temperature (K)
    Returns:
        float: Hydroemteor Retrieval (g/m3)
    """
    lwc=CalculateLWCfromKaBanddBZ(dBZ)
    iwc=CalculateIWCfromKaBanddBZ(dBZ)
    rwc=CalculateRWCfromKaBanddBZ(dBZ)
    swc=CalculateSWCfromKaBanddBZ(dBZ)
    hydro_profile_water=np.zeros_like(temperature)
    hydro_profile_ice=np.zeros_like(temperature)
    hydro_profile_rain=np.zeros_like(temperature)
    hydro_profile_snow=np.zeros_like(temperature)

    for i,cell in enumerate(temperature):
        if cell-273.15>0:
            hydro_profile_water[i]=lwc[i]
            if dBZ[i]>=10:
                hydro_profile_rain[i]=rwc[i]
        elif cell-273.15<=-10:
            hydro_profile_ice[i]=iwc[i]
            if dBZ[i]>=10:
                hydro_profile_snow[i]=swc[i]
        else:
            frac_ice=(-temperature)/10
            frac_water=1-frac_ice
            hydro_profile_water[i]=lwc[i]*frac_water[i]
            hydro_profile_ice[i]=iwc[i]*frac_ice[i]
            if dBZ[i]>=10:
                hydro_profile_snow[i]=swc[i]*frac_ice[i]
                hydro_profile_rain[i]=rwc[i]*frac_water[i]
    return hydro_profile_water,hydro_profile_ice,hydro_profile_rain,hydro_profile_snow

def CalculatePressureAtHeight(P0,T0,z):
    """
    Input:
        P0: surface pressure (Pa)
        T0: surface temperature (K)
        z: height (m)

    return:
        P: pressure (Pa)
    """
    g = 9.80665      # m/s²
    Rd = 287.05      # J/(kg·K)
    P = P0 * np.exp(-g * z / (Rd * T0))
    return P