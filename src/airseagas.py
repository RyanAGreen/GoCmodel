import numpy as np


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

    # --------------------------------------------------------------------------------------------------- constants & variables

    Temp = 298.15  # Kelvin
    Sal = 35  # partperthousand
    SWD = 1029  # kg/m^3
    surf_gas_flux = 0.00003472  # m/s

    A1_C = -60.3409  # CO2 Solubility parameters from Weiss (1974), in mol/ (kg atm)
    A2_C = 93.4517
    A3_C = 23.3585
    B1_C = 0.023517
    B2_C = -0.023656
    B3_C = 0.0047036

    FK = 0.9995  # The thermodynamic fractionation factor for carbon isotopes in air-sea exchange (no units)
    FKR = 0.9990

    # air-sea fractionation factors dC13
    FSA = -9.866 / (Temp) + 1.02412
    FAS = -0.373 / (Temp) + 1.00019

    # air-sea fractionation factors dC14
    FSAR = 0.92182
    FASR = 0.99786

    K0 = np.exp(
        (
            A1_C
            + A2_C * (100.0 / (Temp))  # umol/ kg atm
            + A3_C * np.log(Temp / 100.0)
            + (Sal) * (B1_C + B2_C * ((Temp) / 100.0) + B3_C * (((Temp) / 100.0) ** 2))
        )
    )

    # --------------------------------------------------------------------------------------------------- pco2 solver method controls

    pco2_method = "Mathis"

    if pco2_method == "Mathis":
        surface_dic = current_state[0, 2]
        surface_alk = current_state[1, 2]

        pco2 = ((2 * surface_dic - surface_alk) ** 2) / surface_alk - surface_dic

    if pco2_method == "carbcalc":
        surface_dic = current_state[0, 2] * 1e-6
        surface_alk = current_state[1, 2] * 1e-6
        Boron = 1.179e-5 * Sal

        K_0 = np.exp(
            -60.2409
            + 9345.17 / Temp
            + 23.3585 * np.log(Temp / 100)
            + Sal * (0.023517 - 0.00023656 * Temp + 0.0047036 * (Temp / 100) ** 2)
        )

        K1 = np.exp(
            2.18867
            - 2275.036 / Temp
            - 1.468591 * np.log(Temp)
            + (-0.138681 - 9.33291 / Temp) * np.sqrt(Sal)
            + 0.0726483 * Sal
            - 0.00574938 * Sal ** 1.5
        )

        K2 = np.exp(
            -0.84226
            - 3741.1288 / Temp
            - 1.437139 * np.log(Temp)
            + (-0.128417 - 24.41239 / Temp) * np.sqrt(Sal)
            + 0.1195308 * Sal
            - 0.0091284 * Sal ** 1.5
        )

        Kb = np.exp(
            (
                -8966.90
                - 2890.51 * np.sqrt(Sal)
                - 77.942 * Sal
                + 1.726 * Sal ** 1.5
                - 0.0993 * Sal ** 2
            )
            / Temp
            + (148.0248 + 137.194 * np.sqrt(Sal) + 1.62247 * Sal)
            + (-24.4344 - 25.085 * np.sqrt(Sal) - 0.2474 * Sal) * np.log(Temp)
            + 0.053105 * np.sqrt(Sal) * Temp
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
        # self.aq_CO2 = surface_dic / (1 + (self.K1 / self.H) + ((self.K1*self.K2) / (self.H)**2))

        pco2 = (aq_CO2 / K0) * 1e6  # ppm

        pH = -np.log10(H)

    # --------------------------------------------------------------------------------------------------- carbon flux

    CO2flux_seatoair = SWD * K0 * surf_gas_flux * (CO2_atm - pco2)  # umol/(s m^2)

    # --------------------------------------------------------------------------------------------------- d13C flux

    kinetic_frac = SWD * K0 * surf_gas_flux * FK  # umol / (atm s)
    del_13_c_ppmil = current_state[3, 2] / current_state[0, 2]  # ppmil

    d13Cflux_seatoair = kinetic_frac * (
        ((FAS * (d13C_atm / CO2_atm)) * CO2_atm)
        - (FSA * (del_13_c_ppmil / pco2) * pco2)
    )  # umol / m^2 s

    # --------------------------------------------------------------------------------------------------- d14C flux

    radio_kinetic_frac = SWD * K0 * surf_gas_flux * FKR  # umol / (atm s)
    del_14_c_ppmil = current_state[4, 2] / current_state[0, 2]  # ppmil

    D14Cflux_seatoair = radio_kinetic_frac * (
        ((FASR * (D14C_atm / CO2_atm)) * CO2_atm)
        - (FSAR * (del_14_c_ppmil / pco2) * pco2)
    )  # umol / m^2 s

    # convert from umol/m^2s to umol/kg
    CO2flux_seatoair = CO2flux_seatoair * 3.1536e7 * surface_area / surface_mass
    d13Cflux_seatoair = d13Cflux_seatoair * 3.1536e7 * surface_area / surface_mass
    D14Cflux_seatoair = D14Cflux_seatoair * 3.1536e7 * surface_area / surface_mass

    d_dt = np.zeros((num_tracer, num_box))
    d_dt[0, 2] += CO2flux_seatoair
    d_dt[3, 2] += d13Cflux_seatoair
    d_dt[4, 2] += D14Cflux_seatoair

    return d_dt
