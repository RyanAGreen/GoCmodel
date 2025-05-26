import numpy as np

# Constants
TEMP_CELSIUS = 19
# TEMP_CELSIUS = 24
TEMP_KELVIN = TEMP_CELSIUS + 273.15
SALINITY = 35
# SALINITY = 34.6
# BORON = 1.179e-5 * SALINITY  # Total Boron mol/kg as a fraction of salinity
"""Total borate in mol/kg-sw following U74."""
# === CO2SYS.m comments: =======
# Uppstrom, L., Deep-Sea Research 21:161-162, 1974:
# this is .000416*Sali/35. = .0000119*Sali
# total_borate[FF] = (0.000232/10.811)*(Sal[FF]/1.80655); in mol/kg-SW.
BORON = 0.0004157 * SALINITY / 35
# MIXED_LAYER_DEPTH = 1  # meters
MIXED_LAYER_DEPTH = 5  # meters
# SURFACE_AREA = 1
SURFACE_AREA = 5e6  # m2 (5km2)
SWD = 1029  # sw density kg/m3
PISTON_VELOCITY = 0.067  # mol m-2 yr-1 atm-1 (from Stocker 1994 and Broecker 1985)

# Carbonate and boric acid equilibrium constants as functions of temp and S
# I use the same default equilibrium constants as PyCO2sys

KW = np.exp(
    148.9802
    - 13847.26 / TEMP_KELVIN
    - 23.6521 * np.log(TEMP_KELVIN)
    + (-79.2447 + 3298.72 / TEMP_KELVIN + 12.0408 * np.log(TEMP_KELVIN))
    * np.sqrt(SALINITY)
    - 0.019813 * SALINITY
)

"""Henry's constant for CO2 solubility in mol/kg-sw/atm following W74."""
# Weiss, R. F., Marine Chemistry 2:203-215, 1974.
# This is in mol/kg-SW/atm.
TempK100 = TEMP_KELVIN / 100
lnK0 = (
    -60.2409
    + 93.4517 / TempK100
    + 23.3585 * np.log(TempK100)
    + SALINITY * (0.023517 - 0.023656 * TempK100 + 0.0047036 * TempK100**2)
)
K0 = np.exp(lnK0)

"""Carbonic acid dissociation constants following SLH20."""
# Coefficients and their 95% confidence intervals from SLH20 Table 1.
# doi:10.5194/os-2020-19
pK1 = (
    8510.63 / TEMP_KELVIN  # ±1139.8
    - 172.4493  # ±26.131
    + 26.32996 * np.log(TEMP_KELVIN)  # ±3.9161
    - 0.011555 * SALINITY
    + 0.0001152 * SALINITY**2
)
K1 = 10.0**-pK1  # this is on the Total pH scale in mol/kg-SW
pK2 = (
    4226.23 / TEMP_KELVIN  # ±1050.8
    - 59.4636  # ±24.016
    + 9.60817 * np.log(TEMP_KELVIN)  # ±3.5966
    - 0.01781 * SALINITY
    + 0.0001122 * SALINITY**2
)
K2 = 10.0**-pK2  # this is on the Total pH scale in mol/kg-SW


"""Boric acid dissociation constant following D90b."""
# Dickson, A. G., Deep-Sea Research 37:755-766, 1990.
# lnKB is on Total pH scale
sqrSal = np.sqrt(SALINITY)
lnKBtop = (
    -8966.9
    - 2890.53 * sqrSal
    - 77.942 * SALINITY
    + 1.728 * sqrSal * SALINITY
    - 0.0996 * SALINITY**2
)
lnKB = (
    lnKBtop / TEMP_KELVIN
    + 148.0248
    + 137.1942 * sqrSal
    + 1.62142 * SALINITY
    + (-24.4344 - 25.085 * sqrSal - 0.2474 * SALINITY) * np.log(TEMP_KELVIN)
    + 0.053105 * sqrSal * TEMP_KELVIN
)
Kb = np.exp(lnKB)


# """Phosphate dissociation constants following KP67."""
# # === CO2SYS.m comments: =======
# # Peng et al don't include the contribution from the KP1 term,
# # but it is so small it doesn't contribute. It needs to be
# # kept so that the routines work ok.
# # KP2, KP3 from Kester, D. R., and Pytkowicz, R. M.,
# # Limnology and Oceanography 12:243-252, 1967:
# # these are only for sals 33 to 36 and are on the NBS scale.
# KP1 = 0.02  # This is already on the seawater scale!
# KP2 = np.exp(-9.039 - 1450 / TEMP_KELVIN)
# KP3 = np.exp(4.466 - 7276 / TEMP_KELVIN)

"""Phosphate dissociation constants following YM95."""
# === CO2SYS.m comments: =======
# Yao and Millero, Aquatic Geochemistry 1:53-88, 1995
# KP1, KP2, KP3 are on the SWS pH scale in mol/kg-SW.
lnKP1 = (
    -4576.752 / TEMP_KELVIN
    + 115.54
    - 18.453 * np.log(TEMP_KELVIN)
    + (-106.736 / TEMP_KELVIN + 0.69171) * np.sqrt(SALINITY)
    + (-0.65643 / TEMP_KELVIN - 0.01844) * SALINITY
)
KP1 = np.exp(lnKP1)
lnKP2 = (
    -8814.715 / TEMP_KELVIN
    + 172.1033
    - 27.927 * np.log(TEMP_KELVIN)
    + (-160.34 / TEMP_KELVIN + 1.3566) * np.sqrt(SALINITY)
    + (0.37335 / TEMP_KELVIN - 0.05778) * SALINITY
)
KP2 = np.exp(lnKP2)
lnKP3 = (
    -3070.75 / TEMP_KELVIN
    - 18.126
    + (17.27039 / TEMP_KELVIN + 2.81197) * np.sqrt(SALINITY)
    + (-44.99486 / TEMP_KELVIN - 0.09984) * SALINITY
)
KP3 = np.exp(lnKP3)


def ionic_strength_DOE94(salinity):
    """Ionic strength following DOE94."""
    # === CO2SYS.m comments: =======
    # This is from the DOE handbook, Chapter 5, p. 13/22, eq. 7.2.4.
    return 19.924 * salinity / (1000 - 1.005 * salinity)


"""Silicate dissociation constant following YM95."""
# === CO2SYS.m comments: =======
# Yao and Millero, Aquatic Geochemistry 1:53-88, 1995
# KSi was given on the SWS pH scale in mol/kg-H2O, but is converted here
# to mol/kg-sw.
IonS = ionic_strength_DOE94(SALINITY)
lnKSi = (
    -8904.2 / TEMP_KELVIN
    + 117.4
    - 19.334 * np.log(TEMP_KELVIN)
    + (-458.79 / TEMP_KELVIN + 3.5913) * np.sqrt(IonS)
    + (188.74 / TEMP_KELVIN - 1.5998) * IonS
    + (-12.1652 / TEMP_KELVIN + 0.07871) * IonS**2
)
Ksi = np.exp(lnKSi) * (1 - 0.001005 * SALINITY)

"""Bisulfate (hydrogen sulfate) dissociation constant following D90a."""
# === CO2SYS.m comments: =======
# Dickson, A. G., J. Chemical Thermodynamics, 22:113-127, 1990
# The goodness of fit is .021.
# It was given in mol/kg-H2O. I convert it to mol/kg-SW.
# TYPO on p. 121: the constant e9 should be e8.
# Output KS is on the free pH scale in mol/kg-sw.
# This is from eqs 22 and 23 on p. 123, and Table 4 on p 121:
logTempK = np.log(TEMP_KELVIN)
IonS = ionic_strength_DOE94(SALINITY)
lnKSO4 = (
    -4276.1 / TEMP_KELVIN
    + 141.328
    - 23.093 * logTempK
    + (-13856 / TEMP_KELVIN + 324.57 - 47.986 * logTempK) * np.sqrt(IonS)
    + (35474 / TEMP_KELVIN - 771.54 + 114.723 * logTempK) * IonS
    + (-2698 / TEMP_KELVIN) * np.sqrt(IonS) * IonS
    + (1776 / TEMP_KELVIN) * IonS**2
)
Ks = np.exp(lnKSO4) * (1 - 0.001005 * SALINITY)


"""Hydrogen fluoride dissociation constant following DR79."""
# === CO2SYS.m comments: =======
# Dickson, A. G. and Riley, J. P., Marine Chemistry 7:89-99, 1979:
# this is on the free pH scale in mol/kg-sw
IonS = ionic_strength_DOE94(SALINITY)
lnKF = 1590.2 / TEMP_KELVIN - 12.641 + 1.525 * IonS**0.5
Kf = np.exp(lnKF) * (1 - 0.001005 * SALINITY)
