import numpy as np

temp_kelvin = 20 + 273.15
salinity = 34.78


def alpha_to_epsilon(alpha):
    epsilon = (alpha - 1) * 1000
    return epsilon


def pco2_calc(current_state):
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
    print("pco2 is ", pco2_ocean)

    d13C_ocean = current_state[3, 2] / current_state[0, 2]  # ppmil
    D14C_ocean = current_state[4, 2] / current_state[0, 2]
    pco2_atm = CO2_atm
    # print("pco2 misfit is ", pco2_ocean - pco2_atm)
    # pco2_atm = 180
    d13C_atm = d13C_atm
    # d13C_atm = -6.5
    # print("d13C of the atmosphere is ", d13C_atm)
    D14C_atm = D14C_atm
    # D14C_atm = 100
    # print("D14C of the atmosphere is ", D14C_atm)

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
