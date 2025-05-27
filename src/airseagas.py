import numpy as np
import PyCO2SYS as pyco2

temp_kelvin = 20 + 273.15
salinity = 34.78


def alpha_to_epsilon(alpha):
    epsilon = (alpha - 1) * 1000
    return epsilon

def pco2_calc_old(current_state):
    ############## Calculate pCO2 of surface ocean ###############

    surface_dic = current_state[0, 2] * 1e-6
    surface_alk = current_state[1, 2] * 1e-6

    K0 = np.exp(
        -60.2409
        + 9345.17 / temp_kelvin
        + 23.3585 * np.log(temp_kelvin / 100)
        + salinity
        * (0.023517 - 0.00023656 * temp_kelvin + 0.0047036 * (temp_kelvin / 100) ** 2)
    )

    K1 = np.exp(
        2.18867
        - 2275.036 / temp_kelvin
        - 1.468591 * np.log(temp_kelvin)
        + (-0.138681 - 9.33291 / temp_kelvin) * np.sqrt(salinity)
        + 0.0726483 * salinity
        - 0.00574938 * salinity ** 1.5
    )

    K2 = np.exp(
        -0.84226
        - 3741.1288 / temp_kelvin
        - 1.437139 * np.log(temp_kelvin)
        + (-0.128417 - 24.41239 / temp_kelvin) * np.sqrt(salinity)
        + 0.1195308 * salinity
        - 0.0091284 * salinity ** 1.5
    )

    Kb = np.exp(
        (
            -8966.90
            - 2890.51 * np.sqrt(salinity)
            - 77.942 * salinity
            + 1.726 * salinity ** 1.5
            - 0.0993 * salinity ** 2
        )
        / temp_kelvin
        + (148.0248 + 137.194 * np.sqrt(salinity) + 1.62247 * salinity)
        + (-24.4344 - 25.085 * np.sqrt(salinity) - 0.2474 * salinity)
        * np.log(temp_kelvin)
        + 0.053105 * np.sqrt(salinity) * temp_kelvin
    )

    H = 10 ** (-8)

    diff_H = H
    tiny_diff_H = 1e-15

    # iter = 0

    while diff_H > tiny_diff_H:

        H_old = H

        CA = surface_alk  # umol/kg

        a = CA
        b = K1 * (a - surface_dic)
        c = K1 * K2 * ((a - 2) * surface_dic)

        H = ((-1 * b) + np.sqrt((b) ** 2 - 4 * a * c)) / (2 * a)

        diff_H = abs(H - H_old)
        # iter = iter + 1

    aq_CO2 = a / ((K1 / H) + 2 * K1 * (K2 / (H ** 2)))

    pco2_ocean = (aq_CO2 / K0) * 1e6  # ppm
    return pco2_ocean

def h_total(
    H,
    K1,
    K1p,
    K12,
    K12p,
    K123p,
    Kf,
    Ks,
    Kw,
    invKb,
    invKs,
    invKsi,
    alk,
    borate,
    dic,
    fluoride,
    phos,
    sili,
    sulfate,
):
    """

    Evaluate the root, f([H+])

    Eqn (18), the expression for AT, from Dickson 2007, in terms of total concentrations and [H+]

    (RG fixed below equation from ROMS)
    fn=HCO3+CO3+borate+OH+HPO4+2*PO4+silicate+NH3-Hfree-HSO4-HF-H3PO4-TALK

    """
    H3 = H * H * H
    invH = 1.0 / H
    A = H * (K12p + H * (K1p + H)) + K123p
    B = H * (K1 + H) + K12
    C = 1.0 / (1.0 + sulfate * invKs)

    res = (
        dic * (K1 * H + 2.0 * K12) / B
        + borate / (1.0 + H * invKb)
        + Kw * invH
        + phos * (K12p * H + 2.0 * K123p - H3) / A
        + sili / (1.0 + H * invKsi)
        - H * C
        - sulfate / (1.0 + Ks * invH * C)
        - fluoride / (1.0 + Kf * invH)
        - alk
    )
    return res

def pco2_calc(
    current_state,
    temp=20.0,
    salt=35.0,
    po4=0.0,
    sio3=0.0,
):
    """
    !***********************************************************************
    !                                                                      !
    !  This routine computes equilibrium partial pressure of CO2 (pCO2)    !
    !  in the surface seawater.                                            !
    !                                                                      !
    !  On Input:                                                           !
    !                                                                      !
    !     tic      Total inorganic carbon (millimol/m3).                   !
    !     talk     Total alkalinity (milli-equivalents/m3).                !
    !     temp     Surface temperature (Celsius).                          !
    !     salt     Surface salinity (PSS).                                 !
    !     tic      Total inorganic carbon (millimol/m3).                   !
    !     talk     Total alkalinity (milli-equivalents/m3).                !
    !     po4      Inorganic phosphate (millimol/m3).                      !
    !     sio3     Inorganic silicate (millimol/m3).                       !
    !                                                                      !
    !  On Output:                                                          !
    !                                                                      !
    !     ph       pH.                                                     !
    !     omega_arag    CaCO3 saturation state.                            !
    !     pco2     partial pressure of CO2 (ppmv).                         !
    !     co2star  H2CO3 + CO2(aq).                                        !
    !                                                                      !
    !  Check Value:  (tempik=24, saltik=36.6, ticik=2040, talkik=2390,     !
    !                 po4ik=0, sio3ik=0)                                   !
    !                                                                      !
    !                pcO2=0.35074945E+03 ppmv  (DoNewton=0)                !
    !                pCO2=0.35073560E+03 ppmv  (DoNewton=1)                !
    !                                                                      !
    !  This subroutine was adapted by Mick Follows (Oct 1999) from OCMIP2  !
    !  code CO2CALC. Modified for ROMS by Hernan Arango (Nov 2003).        !
    !  Performance improvements and code streamlining by Jann Paul Mattern !
    !  (Jul 2015).                                                         !
    !                                                                      !
    !  USING DEFAULT CONSTANTS FROM ROMS                                   !                                                                                            !
    !**********************************************************************!
    """
    # local variable definitions
    tic = current_state[0, 2]
    talk = current_state[1, 2]
    # determine coefficients and constants for surface ocean chemistry
    Tk = temp + 273.15
    centiTk = 0.01 * Tk
    invTk = 1.0 / Tk
    logTk = np.log(Tk)
    sqrtS = np.sqrt(salt)
    SO4 = 19.924 * salt / (1000 - 1.005 * salt)
    sqrtSO4 = np.sqrt(SO4)
    scl = salt / 1.80655

    # converting from umol/kg or mmol/m3 to mol/kg
    alk = talk * 0.000001
    dic = tic * 0.000001
    phos = po4 * 0.000001
    sili = sio3 * 0.000001

    # Correction term for non-ideality (ff) ff=k0*(1-pH2O). Equation 13 with
    # table 6 values from Weiss and Price (1980, Mar. Chem., 8, 347-359).

    ff = np.exp(
        -162.8301
        + 218.2968 / centiTk
        + np.log(centiTk) * 90.9241
        - centiTk**2 * 1.47696
        + salt * (0.025695 - centiTk * (0.025225 - centiTk * 0.0049867))
    )

    # Dissociation constants of carbonic acid
    K1 = 10.0 ** (
        62.008 - invTk * 3670.7 - logTk * 9.7944 + salt * (0.0118 - salt * 0.000116)
    )
    K2 = 10.0 ** (-4.777 - invTk * 1394.7 + salt * (0.0184 - salt * 0.000118))

    # Dissociation constant of boric acid
    Kb = np.exp(
        -invTk
        * (
            8966.90
            + sqrtS * (2890.53 + sqrtS * (77.942 - sqrtS * (1.728 - sqrtS * 0.0996)))
        )
        - logTk * (24.4344 + sqrtS * (25.085 + sqrtS * 0.2474))
        + Tk * (sqrtS * 0.053105)
        + 148.0248
        + sqrtS * (137.1942 + sqrtS * 1.62142)
    )

    # Dissociation constants of phosphoric acid
    K1p = np.exp(
        115.525
        - invTk * 4576.752
        - logTk * 18.453
        + sqrtS * (0.69171 - invTk * 106.736)
        - salt * (0.01844 + invTk * 0.65643)
    )
    K2p = np.exp(
        172.0883
        - invTk * 8814.715
        - logTk * 27.927
        + sqrtS * (1.3566 - invTk * 160.340)
        - salt * (0.05778 - invTk * 0.37335)
    )
    K3p = np.exp(
        -18.141
        - invTk * 3070.75
        + sqrtS * (2.81197 + invTk * 17.27039)
        - salt * (0.09984 + invTk * 44.99486)
    )

    # Dissociation constant of silica
    Ksi = np.exp(
        117.385
        - invTk * 8904.2
        - logTk * 19.334
        + sqrtSO4 * (3.5913 - invTk * 458.79)
        - SO4 * (1.5998 - invTk * 188.74 - SO4 * (0.07871 - invTk * 12.1652))
        + np.log(1.0 - 0.001005 * salt)
    )

    # Ion product of water
    Kw = np.exp(
        148.9652
        - invTk * 13847.26
        - logTk * 23.6521
        - sqrtS * (5.977 - invTk * 118.67 - logTk * 1.0495)
        - salt * 0.01615
    )

    # Salinity constant of hydrogen sulfate
    Ks = np.exp(
        141.328
        - invTk * 4276.1
        - logTk * 23.093
        + sqrtSO4 * (324.57 - invTk * 13856.0 - logTk * 47.986 - SO4 * invTk * 2698.0)
        - SO4 * (771.54 - invTk * 35474.0 - logTk * 114.723 - SO4 * invTk * 1776.0)
        + np.log(1.0 - 0.001005 * salt)
    )

    # Stability constant of hydrogen fluoride
    Kf = np.exp(
        -12.641
        + invTk * 1590.2
        + sqrtSO4 * 1.525
        + np.log(1.0 - 0.001005 * salt)
        + np.log(1.0 + 0.1400 * scl / (96.062 * Ks))
    )

    # Concentrations for borate, sulfate, and fluoride
    borate = 0.000232 * scl / 10.811
    sulfate = 0.14 * scl / 96.062
    fluoride = 0.000067 * scl / 18.9984

    K12 = K1 * K2
    K12p = K1p * K2p
    K123p = K12p * K3p
    invKb = 1.0 / Kb
    invKs = 1.0 / Ks
    invKsi = 1.0 / Ksi

    # =======================================================================
    #  Iterative solver for computing hydrogen ions [H+] using:
    #
    #       bracket and bisection
    # =======================================================================
    #
    # -----------------------------------------------------------------------
    #  Bracket and bisection method.
    # -----------------------------------------------------------------------
    #

    IbrackMax = 30
    # Set Brackets for [H+] solvers.
    # H_lo=10.0_r8**(-pH_hi) with high bracket pH_hi=10.0
    H_lo = 10.0 ** (-10.0)
    # H_hi=10.0_r8**(-pH_lo) with low bracket pH_lo=5.0
    H_hi = 10.0 ** (-5.0)

    f_hi = h_total(
        H_hi,
        K1,
        K1p,
        K12,
        K12p,
        K123p,
        Kf,
        Ks,
        Kw,
        invKb,
        invKs,
        invKsi,
        alk,
        borate,
        dic,
        fluoride,
        phos,
        sili,
        sulfate,
    )
    #   f_lo=h_total(H_lo, K1, K1p, K12, K12p, K123p,
    #  Kf, Ks, Kw, invKb, invKs, invKsi,
    #  alk, borate, dic, fluoride, phos, sili, sulfate)
    H_mid = 0.5 * (H_lo + H_hi)

    for Ibrack in range(1, IbrackMax + 1):
        # Evaluate f([H+]) for bracketing and mid-value cases.

        H_mid = 0.5 * (H_lo + H_hi)
        f_mid = h_total(
            H_mid,
            K1,
            K1p,
            K12,
            K12p,
            K123p,
            Kf,
            Ks,
            Kw,
            invKb,
            invKs,
            invKsi,
            alk,
            borate,
            dic,
            fluoride,
            phos,
            sili,
            sulfate,
        )

        # Now, bracket solution within two of the three.
        if f_mid == 0:
            break
        else:
            ftest = f_hi / f_mid
            if ftest > 0:
                H_hi = H_mid
                f_hi = f_mid
            else:
                H_lo = H_mid
                # f_lo = f_mid
            H_mid = 0.5 * (H_lo + H_hi)

    # Last iteration gives value.
    # pylance: disable-reportUnboundVariable
    H = H_mid

    # Determine pCO2.
    # Total Hydrogen ion concentration, H = [H+].
    H2 = H * H

    # Calculate [CO2*] (mole/m3) as defined in DOE Methods Handbook 1994
    # Version 2, ORNL/CDIAC-74, Dickson and Goyet, Eds. (Chapter 2,
    # page 10, Eq A.49).
    co2starik = dic * H2 / (H2 + K1 * H + K1 * K2)

    # -------------------------RG added---------------------- #
    # Calculate [CO32-] (mole/m3) as defined in DOE Methods Handbook 1994
    # Version 2, ORNL/CDIAC-74, Dickson and Goyet, Eds. (Chapter 2,
    # page 10, Eq A.51).
    co3_insitu = dic * K1 * K2 / (H2 + K1 * H + K1 * K2)

    #  Convert to mol kg-1 (double check units here!!! TODO RG)
    #  mol / m3  / kg/m3 = mol / kg.  ?
    sw_density = 1027  # kg/m3
    # co3_insitu = co3_insitu / sw_density
    co3_insitu = co3_insitu

    # Calculate the solubility product constant (Ksp) of aragonite in sw.
    #  Units are mol2 kg-2. Equation originally from Mucci (1983).
    # Not corrected for pressure-only use values at surface TODO RG

    ksp = 10.0 ** (
        -171.945
        - 0.077993 * Tk
        + 2903.293 / Tk
        + 71.595 * np.log10(Tk)
        + (-0.068393 + 0.0017276 * Tk + 88.135 / Tk) * salt**0.5
        - 0.10018 * salt
        + 0.0059415 * salt**1.5
    )

    #  Calculates the carbonate ion concentration at saturation. Full eqn is
    # ksp (mol2 kg-2) = [CO32-]sat (mol kg-1) * [Ca2+]sat (mol kg-1)
    # Due to small changes in dissolved Ca2+, we assume a constant value of
    # 10.28 mmol kg-1 based on the ocean's mean concentration.
    ca_sat = 10.28
    # convert to mol kg-1
    ca_sat = ca_sat / 1e3
    co3_sat = ksp / ca_sat

    # Approximatation to calculate the saturation state of seawater (Ω).
    # omega_arag = [CO32-]insitu / [CO32-]sat
    # We can use this approximation because [Ca2+] variations are very small
    # relative to those of CO32-. eqn 9.3.3 from Chapter 9 in Ocean
    # Biogeochemical Dynamics Sarimento and Gruber (2006)
    omega_aragik = co3_insitu / co3_sat

    # -------------------------RG added---------------------- #

    # Save pH is used again outside this routine.
    phik = -np.log10(H)

    # Add two output arguments for storing pCO2surf.

    pco2ik = co2starik * 1000000.0 / ff

    # can also return pH, omega_aragik,co2starik
    return pco2ik

def gas_exchange(
    current_state,
    num_tracer,
    num_box,
    CO2_atm,
    d13C_atm,
    D14C_atm,
    surface_area,
    surface_mass,
):

    k = 0.067  # mol m-2 yr-1 atm-1 # stocker 1994 and Broecker 1985 (just going to use this for now)

    mol_to_umolkg = 1e6 / surface_mass

    ######### Calculate fluxes ##########

    pco2_ocean = pco2_calc(current_state)
    # pco2_ocean = pco2_calc_pyco2sys(current_state)

    # print("pco2 is ", pco2_ocean)

    d13C_ocean = current_state[3, 2] / current_state[0, 2]  # ppmil
    D14C_ocean = current_state[4, 2] / current_state[0, 2]
    pco2_atm = CO2_atm

    d13C_atm = d13C_atm
    D14C_atm = D14C_atm

    # Kinetic Fractionation factor for CO2 gas transfer aross the air-sea interface
    kinetic_frac_c13 = 0.9995
    kinetic_frac_c13_permil = alpha_to_epsilon(kinetic_frac_c13)  # converted to per mil

    # temperature dependent equilibrium fractionation factors
    eq_frac_sa_c13 = -9.866 / (temp_kelvin) + 1.02412
    eq_frac_sa_c13_permil = alpha_to_epsilon(eq_frac_sa_c13)

    eq_frac_as_c13 = -0.373 / (temp_kelvin) + 1.00019
    eq_frac_as_c13_permil = alpha_to_epsilon(eq_frac_as_c13)

    SeatoAir = k * pco2_ocean * surface_area  # mol yr-1
    AirtoSea = k * pco2_atm * surface_area  # mol yr-1

    fract_sa_c13 = eq_frac_sa_c13_permil + kinetic_frac_c13_permil
    fract_as_c13 = eq_frac_as_c13_permil + kinetic_frac_c13_permil

    dC13_SeatoAir = SeatoAir * (d13C_ocean + fract_sa_c13)
    dC13_AirtoSea = AirtoSea * (d13C_atm + fract_as_c13)
    DC14_SeatoAir = SeatoAir * (D14C_ocean + (2 * fract_sa_c13))
    DC14_AirtoSea = AirtoSea * (D14C_atm + (2 * fract_as_c13))

    d_dt = np.zeros((num_tracer, num_box))
    d_dt[0, 2] += (AirtoSea - SeatoAir) * mol_to_umolkg
    d_dt[3, 2] += (dC13_AirtoSea - dC13_SeatoAir) * mol_to_umolkg
    d_dt[4, 2] += (DC14_AirtoSea - DC14_SeatoAir) * mol_to_umolkg

    return d_dt
