"""
WSM3 (WRF Single-Moment 3-class microphysics)
------------------------------------------------------------------
- variables and units
  T: temperature: K
  p: air pressure : Pa
  qv: water vapor mixing ratio: kg/kg
  qc: cloud water mixing ratio: kg/kg
  qr: cloud rain mixing ratio: kg/kg
  rho: air densitiy: kg/m^3
- phase chage:
  T0 = 273.15 K
  temperature-weighted average for cloud water and cloud ice
"""
from __future__ import annotations
import numpy as np

# ----------------------------- constants ---------------------------------
T0 = 273.15    # melting point [K]
Rv = 461.5     # water vapor constant [J/(kg·K)]
Cp = 1004.0    # dry air specific heat at constant pressure  [J/(kg·K)]
Lv0 = 2.5e6    # 0°C latent heat of vaporization [J/kg]
Ls0 = 2.834e6  # 0°C latent heat of sublimation [J/kg]

g = 9.80665

def latent_heats(T: np.ndarray):
    """latent heat adjustment for temperature change"""
    Lv = Lv0 - (T - T0) * 2.3e3  # J/kg
    Ls = Ls0 - (T - T0) * 1.6e3  # J/kg
    Lf = Ls - Lv
    return Lv, Ls, Lf

# ------------------------ water vapor -----------------------------

def esat_water(T: np.ndarray):
    """saturate water vapor pressure water surface (Pa)"""
    return 610.94 * np.exp(17.625 * (T - T0) / (T - 30.11))

def esat_ice(T: np.ndarray):
    """saturate water vapor pressure water surface (Pa)"""
    return 610.94 * np.exp(22.587 * (T - T0) / (T - 0.7))

def qsat(p: np.ndarray, T: np.ndarray, over_ice_w: np.ndarray):
    """saturate water vapor mixing ratio (kg/kg)。over_ice_w (0, water→1, ice)。"""
    e_w = esat_water(T)
    e_i = esat_ice(T)
    e = (1.0 - over_ice_w) * e_w + over_ice_w * e_i
    eps = 0.622
    return eps * e / (p - (1 - eps) * e)

# ------------------------ cloud hydrometeors -------------------------

def ice_weight(T: np.ndarray, dT: float = 5.0):
    """hydrometeor phase based on temperature. T<=T0-dT all ice，T>=T0+dT all water"""
    x = np.clip((T0 + dT - T) / (2 * dT), 0.0, 1.0)
    return x  # 1→冰过程，0→水过程

# ----------------------------- scheme configuration ---------------------------------
class WSM3Config:
    def __init__(self):
        # Kessler auto convert
        self.qc_autoth = 1.0e-3   # cloud water / cloud ice
        self.c_autow = 1.0e-3     # warm rain auto convert [1/s]
        self.c_autoi = 2.0e-4     # ice auto convert [1/s]
        # cloud collection
        self.c_accrw = 2.2        # [1/s]
        self.c_accri = 1.2        # [1/s]
        # evaporation
        self.c_evap = 1.0e-3      # [1/s]
        # minimum positive number
        self.eps = 1.0e-12

# ----------------------------- 主更新核 ---------------------------------

def saturation_adjust(state, cfg: WSM3Config):
    """saturation adjustment: qv↔qc water vapor exchange with cloud water, update letent heat"""
    T = state['T']; p = state['p']
    qv = state['qv']; qc = state['qc']

    wi = ice_weight(T)
    qs = qsat(p, T, wi)
    Lv, Ls, Lf = latent_heats(T)
    L = (1.0 - wi) * Lv + wi * Ls

    # over saturation: qv→qc
    dqv = qv - qs
    cond = np.where(dqv > 0, dqv, 0.0)
    qv = qv - cond
    qc = qc + cond
    dT1 = (L / Cp) * cond
    T = T + dT1

    # subsaturation: qc→qv
    wi = ice_weight(T)
    qs = qsat(p, T, wi)
    L = (1.0 - wi) * Lv + wi * Ls

    dqv = qs - qv
    evap = np.minimum(np.maximum(dqv, 0.0), np.maximum(qc, 0.0))
    qv = qv + evap
    qc = qc - evap
    dT2 = -(L / Cp) * evap
    T = T + dT2

    state['T'] = T
    state['qv'] = np.clip(qv, 0.0, None)
    state['qc'] = np.clip(qc, 0.0, None)


def autoconversion_and_accretion(state, dt: float, cfg: WSM3Config):
    """Autoconversion and accretion from cloud to precipitation, with temperature-dependent coefficients."""
    T = state['T']
    qc = state['qc']; qr = state['qr']
    wi = ice_weight(T)

    c_auto = (1.0 - wi) * cfg.c_autow + wi * cfg.c_autoi
    c_accr = (1.0 - wi) * cfg.c_accrw + wi * cfg.c_accri

    aut = np.where(qc > cfg.qc_autoth, c_auto * (qc - cfg.qc_autoth), 0.0)
    accr = c_accr * qr * qc

    tend = (aut + accr) * dt
    tend = np.minimum(tend, qc)

    state['qc'] = np.clip(qc - tend, 0.0, None)
    state['qr'] = np.clip(qr + tend, 0.0, None)


def precipitation_evaporation(state, dt: float, cfg: WSM3Config):
    """Precipitation evaporation/sublimation, including latent heat effects."""
    T = state['T']; p = state['p']
    qv = state['qv']; qr = state['qr']

    wi = ice_weight(T)
    qs = qsat(p, T, wi)
    Lv, Ls, Lf = latent_heats(T)
    L = (1.0 - wi) * Lv + wi * Ls

    subsat = np.maximum(qs - qv, 0.0)
    evap_rate = cfg.c_evap * subsat * np.sqrt(np.maximum(qr, 0.0) + cfg.eps)
    evap = np.minimum(evap_rate * dt, qr)

    qv = qv + evap
    qr = qr - evap
    T = T - (L / Cp) * evap

    state['T'] = T
    state['qv'] = np.clip(qv, 0.0, None)
    state['qr'] = np.clip(qr, 0.0, None)

def freezing_melting_exchange(state, dt: float, cfg: WSM3Config):
    """In this simplified implementation, the cloud phase is merged into qc;
       phase-change latent heat is represented only through temperature balance,
       without explicitly separating qi/qs variables. 
       If explicit ice/snow categories are required, 
       the scheme can be extended in the style of WSM5/6."""
    # to be added in the future, may be never.
    return

def wsm3_step(state: dict[str, np.ndarray], dt: float, cfg: WSM3Config | None = None):
    """
    input:
        state: T[K], p[Pa], rho, qv, qc, qr
        dt: time step: [s]
    return:
        state
    """
    if cfg is None:
        cfg = WSM3Config()

    # 1) saturation ajustment (qv↔qc，update T)
    saturation_adjust(state, cfg)
    # 2) Cloud to precipitation: autoconversion + accretion
    autoconversion_and_accretion(state, dt, cfg)
    # 3) Precipitation evaporation/sublimation (including temperature feedback)
    precipitation_evaporation(state, dt, cfg)
    # 4) Optional: freezing/melting (already reflected in the weighting and latent heat)
    freezing_melting_exchange(state, dt, cfg)

    for k in ('qv', 'qc', 'qr'):
        state[k] = np.clip(state[k], 0.0, None)
    return state

# ---------------------------- test ---------------------------------
"""
if __name__ == "__main__":
    nz, ny, nx = 1, 1, 1
    shape = (nz, ny, nx)
    state = {
        'T':  (T0 + 5.0) * np.ones(shape),  # 5℃
        'p':  90000.0 * np.ones(shape),     # 900 hPa
        'rho': 1.1 * np.ones(shape),
        'qv':  0.012 * np.ones(shape),
        'qc':  0.0015 * np.ones(shape),
        'qr':  0.0 * np.ones(shape),
    }
    cfg = WSM3Config()
    dt = 1.0
    for n in range(300):
        wsm3_step(state, dt, cfg)
    print({k: float(v) for k, v in state.items() if k in ('T','qv','qc','qr')})
"""
