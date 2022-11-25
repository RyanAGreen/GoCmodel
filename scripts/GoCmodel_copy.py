"""Gulf of California
Regional Model
Going to move things to modules after they work in OOP first
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PyCO2SYS as pyco2
from scipy.integrate import solve_ivp
from scipy import optimize
from scipy.interpolate import interp1d
import src.geologic as geologic
import src.airseagas as airsea
import src.inputoutput as io
import src.circulation as circulation
import src.product as product
import time


class GoCModel:
    """three box ocean model with circulation, biological pump, air sea gas exchange
    and DIC,ALK,P,N,d13C,D14C tracers. Box order is Baja California,
    Gulf of California-Deep, Gulf of California-Surface, North Pacific-
    Intermediate depth and North Pacific Surface"""

    def __init__(self, reminBoolean):

        self.num_box = 3
        self.num_bc = 2
        self.num_tracer = 6

        # kg,calculated from GoC volume
        self.goc_mass = 1.38e14 * 1026.8
        self.goc_surface_mass = 1.65e13 * 1026.8  # kg
        self.goc_subsurface_mass = 3.3e13 * 1026.8  # kg
        self.goc_source_mass = self.goc_mass * 10  # kg
        self.np_surf_mass = self.goc_source_mass * 20  # kg
        self.np_mid_mass = self.np_surf_mass * 2  # kg

        self.mass = np.array(
            [
                self.goc_source_mass,
                self.goc_subsurface_mass,
                self.goc_surface_mass,
                self.np_mid_mass,
                self.np_surf_mass,
            ]
        )

        # surface area of GoC and surface volume

        self.surf_volume = 1.65e13  # m^3
        self.surf_area = self.surf_volume / 200  # m^2

        self.CO2_atm = io.read_co2_data("data/observations/CO2data.txt")

        self.D14C_atm = io.read_14C_atm_data("data/observations/D14Cdata.txt")

        self.d13C_atm = io.read_d13C_atm_data(
            "data/observations/d13Cdata_500yearsnotadded.txt"
        )

        # Setting up inital values
        self.carbon = np.array([2350, 2300, 2100])  # umol/kg
        self.alkalinity = np.array([2420, 2420, 2410])  # umol/kg
        self.phosphorus = np.array([30, 30, 30])  # umol/kg
        self.del_13_c = (
            np.array([0.1, 0.1, 0.1]) * self.carbon
        )  # delta [permil] * concentration
        self.del_14_c = (
            np.array([0.1, 0.1, 0.1]) * self.carbon
        )  # delta [permil] * concentration

        self.cum_geologic_carbon_to_marchitto = 0
        self.cum_geologic_carbon_to_goc_sub = 0
        self.cum_geologic_carbon_to_goc_surf = 0
        self.cum_geologic_carbon = np.array([0, 0, 0])

        self.CaRatio = 0.4

        # Packing initial conditions in matrixes
        # flat array (18,)
        self.state_v0 = np.hstack(
            (
                self.carbon,
                self.alkalinity,
                self.phosphorus,
                self.del_13_c,
                self.del_14_c,
                self.cum_geologic_carbon,
            )
        )
        # self.boundary_condition = io.read_bc(
        #     "data/ISchange/2Dinversion/Powell2Dinversion.txt", 0
        # )
        self.boundary_condition = io.read_bc(
            "data/NoISchange/ForwardRun/control.txt", 0
        )

        svedrup_matrix = circulation.circ(
            self.num_box, self.num_bc, 0.45, 0.03, 0.03, 0.03, 0.03,
        )
        self.transport_matrix = circulation.make_transport_matrix(
            self.num_box, self.num_bc, svedrup_matrix, self.mass
        )
        self.export_matrix = np.array(
            [
                [0, 0, 0, 0, 0],  # GoC surface --> GoC subsurface
                [0, 0, 0, 0, 0],  # NP surface --> Marchitto
                [0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1],
            ]
        )

        self.remin_matrix = np.array(
            [
                [0, 0, 0, 0, 0.8],  # 0.75 for Marchitto
                [0, 0, 0.25, 0, 0],  # 0.25 for GoC Subsurface
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        self.result = None
        self.carbonate_chemistry = None
        self.time = None
        self.output = None

        self.obs_d14c = 0

        self.f_surf, self.f_sub, self.f_mar = self.read_files()

        self.geologic_carbon_initial_guess = np.array([0])

        self.state_copy = np.zeros((self.num_tracer, self.num_box))
        self.time_copy = 0
        self.optimized_timesteps_mar = np.zeros((20000))
        self.optimized_timesteps_sub = np.zeros((20000))
        self.optimized_timesteps_surf = np.zeros((20000))
        self.reminBoolean = reminBoolean
        self.marchitto_rate = np.array([0])
        self.subsurface_rate = np.array([0])
        self.surface_rate = np.array([0])
        self.box_idx = 0
        self.optimize_step = True
        self.counter = 0
        #
        self.marchitto_rate_array = []
        self.subsurface_rate_array = []
        self.surface_rate_array = []

        self.set = set()
        self.prev_set_len = len(self.set)

    def make_state_a(self, state_v, time, bc):
        """Gets called every year and makes new state in matrix format. Boxes are in columns and tracers are in
        rows.
        example:
        all tracers for box 3 === stateA[:,3]
        tracer 2 for all boxes === stateA[2,:]
        we feed in time evolving boundary condition from CYCLOPS every 100 years
        """
        time_rounded = int(time)

        spinuptime = 1000

        if bc == "control":
            if (time_rounded % 100 == 0) and (time_rounded >= spinuptime):
                self.boundary_condition = io.read_bc(
                    "data/NoISchange/ForwardRun/control.txt",
                    ((time_rounded - spinuptime) / 100),
                )
            elif time_rounded < spinuptime:
                self.boundary_condition = io.read_bc(
                    "data/NoISchange/ForwardRun/control.txt", 0
                )

        if bc == "2dinversion":
            if (time_rounded % 100 == 0) and (time_rounded >= spinuptime):
                self.boundary_condition = io.read_bc(
                    "data/ISchange/2Dinversion/Powell2Dinversion.txt",
                    ((time_rounded - spinuptime) / 100),
                )
            elif time_rounded < spinuptime:
                self.boundary_condition = io.read_bc(
                    "data/ISchange/2Dinversion/Powell2Dinversion.txt", 0
                )

        if (time_rounded % 100 == 0) and (time_rounded >= spinuptime):
            idx = int((time_rounded - spinuptime) / 100)
            self.CO2_atm_currentyr = self.CO2_atm[idx, 1]  # ppm
            self.D14C_atm_currentyr = self.D14C_atm[idx, 1]
            self.d13C_atm_currentyr = self.d13C_atm[idx, 1]
        elif time_rounded < spinuptime:
            idx = 0
            self.CO2_atm_currentyr = self.CO2_atm[idx, 1]  # ppm
            self.D14C_atm_currentyr = self.D14C_atm[idx, 1]
            self.d13C_atm_currentyr = self.d13C_atm[idx, 1]

        # reshape flat array to rows as tracers and columns as boxes
        # (18,) -> (6,3)
        state_v_reshaped = state_v.T.reshape(self.num_tracer, self.num_box)
        # print('state_v is', state_v)

        # adds the boundary condition (column index of [:,4] and [:,5])
        # -> (6,5)
        state_a = np.hstack((state_v_reshaped, self.boundary_condition,))
        # clearing cumulative carbon tracer so that it is not affected
        # by circulation matrix mutiplication
        state_a[5, 0] = 0
        state_a[5, 1] = 0
        state_a[5, 2] = 0

        return state_a

    def read_files(self):
        obspath = "data/observations/"

        Rafter_surface = pd.read_csv(obspath + "Rafter_2019.tab", sep="\t", header=24)
        Rafter_surface = Rafter_surface.loc[(Rafter_surface["Habitat"] == "planktic")]
        Rafter_surface["Cal age [ka BP]"] = 1000 * Rafter_surface["Cal age [ka BP]"]
        Rafter_surface = Rafter_surface.sort_values(by=["Cal age [ka BP]"])

        Rafter_subsurface = pd.read_excel(
            obspath + "prafter-2019-Gulf-CA-Data-for-Ryan.xls"
        )
        Rafter_subsurface = Rafter_subsurface.loc[
            (Rafter_subsurface["species"] == "U. peregrina")
            | (Rafter_subsurface["species"] == "Planulina ariminensis")
            | (Rafter_subsurface["species"] == "U. peregrina ")
        ]
        Rafter_subsurface = Rafter_subsurface.sort_values(by=["calendar age [kyr BP]"])
        Rafter_subsurface = Rafter_subsurface[["calendar age [kyr BP]", "D14C"]]
        Rafter_subsurface = Rafter_subsurface.dropna(subset=["D14C"])
        Rafter_subsurface = (
            Rafter_subsurface.groupby("calendar age [kyr BP]").mean().reset_index()
        )
        Rafter_subsurface["calendar age [kyr BP]"] *= 1000

        Mar = pd.read_csv(obspath + "Marchitto.txt", sep="\s+")
        Mar["Cal.Age"] = 1000 * Mar["Cal.Age"]

        self.surf_min = Rafter_surface["Cal age [ka BP]"].min()
        self.sub_min = Rafter_subsurface["calendar age [kyr BP]"].min()
        self.mar_min = Mar["Cal.Age"].min()

        # The following are functions that return the D14C value at an inputted timestep
        # Linear interpolation
        # fMar is different because there was a duplicated timestep,
        # but it was outside the time range we care about so I just removed it
        fSurf = interp1d(
            Rafter_surface["Cal age [ka BP]"], Rafter_surface["Δ14C [‰]"], kind="linear"
        )
        fSub = interp1d(
            Rafter_subsurface["calendar age [kyr BP]"], Rafter_subsurface["D14C"]
        )
        fMar = interp1d(
            np.delete(np.array([Mar["Cal.Age"]]), [-3, -4]),
            np.delete(np.array([Mar["D14C"]]), [-3, -4]),
        )
        return fSurf, fSub, fMar

    def obj_func(self, geologic_carbon_rate):  # units of PgC​
        """
        Rules:
        1. Algorithm to stop running when misfit < tolerance (0.1)
        2. No removal of carbon (geologic_carbon_rate =! 0)
        """

        # shouldnt we just be calculating a single number not an array for the carbon_flux?

        # convert PgC to model concentration units (umol/kg)
        # carbon_flux = geologic_carbon_rate * 1e15 / 12 * 1e6 / self.mass[: self.num_box]
        carbon_flux_conc = (
            geologic_carbon_rate * 1e15 / 12 * 1e6 / self.mass[self.box_idx]
        )

        # grab delta 14C value for the specific year and the specified box
        # get_d14c is is either f_mar, f_surf, or f_sub and is changed in box_model
        # depending on the current box being optimized
        # del_14_c_obs_permil = np.array([self.obs_d14c])

        # calculate D14C state with geologic carbon
        # all geologic carbon has a per mil value of -1000
        # carbon_flux is the additional change in concentration after carbon addition

        del_14_c_model = self.state_copy[4, self.box_idx] + (-1000 * carbon_flux_conc)

        # convert model state to per mil units
        del_14_c_model_permil = del_14_c_model / (
            self.state_copy[0, self.box_idx] + carbon_flux_conc
        )

        misfit = abs((self.obs_d14c - del_14_c_model_permil))

        return misfit  # per mil

    def box_model(self, time, statev):
        # pylint: disable=unused-argument
        """
        box_model takes in current model state and organizes the data into the
        correct matrix notation. Then box_model finds the change in each tracer
        for a given time step (d_dt). d_dt is returned to the ODE solver.
        """

        state_a = self.make_state_a(statev, time, "control")

        time_bp = 21000 - time

        current_state = state_a[:, : self.num_box]
        self.state_copy = state_a
        self.time_copy = round(time_bp)

        d_dt_geologic = np.zeros((self.num_tracer, self.num_box))

        # d_dt_circ = (self.transport_matrix @ self.state_copy.T).T[:, : self.num_box]
        # self.state_copy[:, : self.num_box] = current_state + d_dt_circ
        # needed to extract interpolated d14c values

        # self.prev_set_len = len(self.set)
        # if self.time_copy % 100 == 0 and self.time_copy <= 20000:

        # if a new 100 year period, we add to list
        if time_bp < 19000 and time_bp > 7500 and round(time_bp) % 100 == 0:
            # if obs < model, we optimize
            # Marchitto
            # d_dt_circ = (self.transport_matrix @ self.state_copy.T).T[:, : self.num_box]
            # self.state_copy[:, : self.num_box] = current_state + d_dt_circ
            self.box_idx = 0
            if (
                self.f_mar(self.time_copy)
                < current_state[4, self.box_idx] / current_state[0, self.box_idx]
            ):
                self.obs_d14c = self.f_mar(self.time_copy)
                self.marchitto_rate = optimize.minimize(self.obj_func, 0.1, tol=1,).x
            # if obs > model, rate = 0
            else:
                self.marchitto_rate = 0

            # current_misfit = (
            #         self.f_sub(self.time_copy)
            #         - (current_state[4, 1] / current_state[0, 1])
            #     )
            # subsurface
            # self.box_idx = 1
            # if (
            #     self.f_sub(self.time_copy)
            #     < current_state[4, self.box_idx] / current_state[0, self.box_idx]
            # ):
            #     self.obs_d14c = self.f_sub(self.time_copy)
            #     self.subsurface_rate = optimize.minimize(self.obj_func, 0.1, tol=1,).x
            # # if obs > model, rate = 0
            # else:
            #     self.subsurface_rate = 0
            # # Surface
            # self.box_idx = 2
            # if time_bp > 12791:
            #     if (
            #         self.f_surf(self.time_copy)
            #         < current_state[4, self.box_idx] / current_state[0, self.box_idx]
            #     ):
            #         self.obs_d14c = self.f_surf(self.time_copy)
            #         self.surface_rate = optimize.minimize(self.obj_func, 0.1, tol=1,).x
            #     else:
            #         self.surface_rate = 0
        else:
            self.marchitto_rate = 0
            self.subsurface_rate = 0
            self.surface_rate = 0

        # marchitto rate should stay as whatever it was above, unless outside of those years.
        # if time_bp > 19000 and time_bp < 7500:

        # if it is a new 100 year period, we optimize
        # if len(self.set) > self.prev_set_len:

        print("The rounded time is ", self.time_copy)

        # rate is either 0 or optimized. It should stay what it was for the remaining years until the optimization is called again..
        d_dt_geologic += geologic.manual_carbon_add(
            self.num_tracer, self.num_box, self.marchitto_rate, "marchitto", self.mass,
        )
        d_dt_geologic += geologic.manual_carbon_add(
            self.num_tracer,
            self.num_box,
            self.subsurface_rate,
            "subsurface",
            self.mass,
        )
        d_dt_geologic += geologic.manual_carbon_add(
            self.num_tracer, self.num_box, self.surface_rate, "surface", self.mass,
        )

        # self.prev_set_len = len(self.set)
        # # no more obs data after 505
        # if self.time_copy % 100 == 0 and self.time_copy <= 20000:
        #     self.set.add(self.time_copy)

        # if len(self.set) > self.prev_set_len:

        #     # optimize
        #     if time_bp > 505 and False:
        #         d_dt_circ = (self.transport_matrix @ self.state_copy.T).T[
        #             :, : self.num_box
        #         ]
        #         self.state_copy[:, : self.num_box] = current_state + d_dt_circ
        #         self.get_d14c = self.f_mar  # needed to extract interpolated d14c values
        #         self.box_idx = 0
        #         self.marchitto_rate = optimize.minimize(
        #             self.obj_func, self.geologic_carbon_initial_guess, tol=0.1,
        #         ).x
        #         if self.marchitto_rate < 0 and False:
        #             print("Marchitto rate is negative")
        #             self.marchitto_rate = 0
        #         d_dt_geologic += geologic.manual_carbon_add(
        #             self.num_tracer,
        #             self.num_box,
        #             self.marchitto_rate,
        #             "marchitto",
        #             self.mass,
        #         )
        #         self.state_copy[:, : self.num_box] += current_state + d_dt_geologic

        #         # Subsurface
        #         d_dt_circ = (self.transport_matrix @ self.state_copy.T).T[
        #             :, : self.num_box
        #         ]
        #         self.state_copy[:, : self.num_box] = current_state + d_dt_circ
        #         self.get_d14c = self.f_sub  # needed to extract interpolated d14c values
        #         self.box_idx = 1
        #         self.subsurface_rate = 0
        #         #     0.0
        #         #     * optimize.minimize(
        #         #         self.obj_func,
        #         #         self.geologic_carbon_initial_guess,
        #         #         tol=0.1,
        #         #         method="Nelder-Mead",
        #         #     ).x
        #         # )
        #         d_dt_geologic += geologic.manual_carbon_add(
        #             self.num_tracer,
        #             self.num_box,
        #             self.subsurface_rate,
        #             "subsurface",
        #             self.mass,
        #         )
        #         self.state_copy[:, : self.num_box] += current_state + d_dt_geologic
        #     else:
        #         self.marchitto_rate = 0
        #         self.subsurface_rate = 0

        #     # Surface
        #     if time_bp > 12791 and False:
        #         d_dt_circ = (self.transport_matrix @ self.state_copy.T).T[
        #             :, : self.num_box
        #         ]
        #         self.state_copy[:, : self.num_box] = current_state + d_dt_circ
        #         self.get_d14c = (
        #             self.f_surf
        #         )  # needed to extract interpolated d14c values
        #         self.box_idx = 2
        #         self.surface_rate = (
        #             0.05
        #             * optimize.minimize(
        #                 self.obj_func,
        #                 self.geologic_carbon_initial_guess,
        #                 tol=0.1,
        #                 method="Nelder-Mead",
        #             ).x
        #         )
        #         d_dt_geologic += geologic.manual_carbon_add(
        #             self.num_tracer,
        #             self.num_box,
        #             self.surface_rate,
        #             "surface",
        #             self.mass,
        #         )
        #     else:
        #         self.surface_rate = 0
        #     self.counter += 1
        #     self.marchitto_rate_array.append(self.marchitto_rate)
        #     self.subsurface_rate_array.append(self.subsurface_rate)
        #     self.surface_rate_array.append(self.surface_rate)

        # else:
        #     d_dt_geologic += geologic.manual_carbon_add(
        #         self.num_tracer,
        #         self.num_box,
        #         self.marchitto_rate,
        #         "marchitto",
        #         self.mass,
        #     )
        #     d_dt_geologic += geologic.manual_carbon_add(
        #         self.num_tracer,
        #         self.num_box,
        #         self.subsurface_rate,
        #         "subsurface",
        #         self.mass,
        #     )
        #     d_dt_geologic += geologic.manual_carbon_add(
        #         self.num_tracer, self.num_box, self.surface_rate, "surface", self.mass,
        #     )
        # if self.time_copy % 50 == 0 and self.time_copy <= 20000:
        #     print("Time is ", time_bp, " and Marchitto rate is ", self.marchitto_rate)
        # (Below) For every integer timestep, recalculate the geologic add, and use
        # the same geologic carbon addition for all increments within an
        # integer year

        # trying to think of clever way to run every integer time step

        """
        Method of Optimization
        1. calculate flux due to new circulation
        2. add flux from circulation into state
        3. optimize geologic carbon addition for current box
        4. add optimized carbon to state
        """
        # geologic_add_initial_guess = np.array([0])

        ### Geologic Carbon Addition ###
        # if time_bp <= 20000 and time_bp > 505:
        #     # Marchitto
        #     d_dt_circ = (self.transport_matrix @ self.state_copy.T).T[:, : self.num_box]
        #     self.state_copy[:, : self.num_box] = current_state + d_dt_circ
        #     self.get_d14c = self.f_mar  # needed to extract interpolated d14c values
        #     self.box_idx = 0
        #     if self.optimized_timesteps_mar[self.time_copy - 1001] == 0:
        #         # print("Time Marchitto: " + str(time_bp))
        #         self.optimized_timesteps_mar[self.time_copy - 1001] = 1
        #         self.marchitto_rate = (
        #             0.1
        #             * optimize.minimize(
        #                 self.obj_func, self.geologic_carbon_initial_guess, tol=0.1,
        #             ).x
        #         )
        #         if self.marchitto_rate < 0:
        #             print("Marchitto rate is negative")
        #             self.marchitto_rate = 0
        #     d_dt_geologic += geologic.manual_carbon_add(
        #         self.num_tracer,
        #         self.num_box,
        #         self.marchitto_rate,
        #         "marchitto",
        #         self.mass,
        #     )
        #     self.state_copy[:, : self.num_box] += current_state + d_dt_geologic

        #     # Subsurface
        #     d_dt_circ = (self.transport_matrix @ self.state_copy.T).T[:, : self.num_box]
        #     self.state_copy[:, : self.num_box] = current_state + d_dt_circ
        #     self.get_d14c = self.f_sub  # needed to extract interpolated d14c values
        #     self.box_idx = 1
        #     if self.optimized_timesteps_sub[self.time_copy - 1001] == 0:
        #         # print("Time Subsurface: " + str(time_bp))
        #         self.optimized_timesteps_sub[self.time_copy - 1001] = 1
        #         self.subsurface_rate = (
        #             0.1
        #             * optimize.minimize(
        #                 self.obj_func, self.geologic_carbon_initial_guess, tol=0.1,
        #             ).x
        #         )
        #     d_dt_geologic += geologic.manual_carbon_add(
        #         self.num_tracer,
        #         self.num_box,
        #         self.subsurface_rate,
        #         "subsurface",
        #         self.mass,
        #     )
        #     self.state_copy[:, : self.num_box] += current_state + d_dt_geologic

        #     # Surface
        #     if time_bp > 12791:
        #         d_dt_circ = (self.transport_matrix @ self.state_copy.T).T[
        #             :, : self.num_box
        #         ]
        #         self.state_copy[:, : self.num_box] = current_state + d_dt_circ
        #         self.get_d14c = (
        #             self.f_surf
        #         )  # needed to extract interpolated d14c values
        #         self.box_idx = 2
        #         if self.optimized_timesteps_surf[self.time_copy - 1001] == 0:
        #             # print("Time Surface: " + str(time_bp) + "\n\n\n")
        #             self.optimized_timesteps_surf[self.time_copy - 1001] = 1
        #             self.surface_rate = (
        #                 0.1
        #                 * optimize.minimize(
        #                     self.obj_func, self.geologic_carbon_initial_guess, tol=0.1,
        #                 ).x
        #             )
        #         d_dt_geologic += geologic.manual_carbon_add(
        #             self.num_tracer,
        #             self.num_box,
        #             self.surface_rate,
        #             "surface",
        #             self.mass,
        #         )
        #     else:
        #         self.surface_rate = 0

        # print("Geologic Add at Time = " + str(self.time_copy) + ":", end=" ")
        # print(
        #     str(self.marchitto_rate)
        #     + "\t"
        #     + str(self.subsurface_rate)
        #     + "\t"
        #     + str(self.surface_rate)
        # )

        # ### Biological Productivity (Soft Tissue + Carbonate) ###
        # d_dt_export, exportP, del_13_c_org, del_14_c_org = product.productivity(
        #     state_a[:, : self.num_box],
        #     self.boundary_condition,
        #     self.num_tracer,
        #     self.num_box,
        #     self.num_bc,
        #     self.CaRatio,
        #     self.export_matrix,
        # )

        # ### Remineralization ###
        # d_dt_remin = np.zeros((self.num_tracer, self.num_box))
        # if self.reminBoolean:
        #     d_dt_remin = product.remin(
        #         exportP,
        #         del_13_c_org,
        #         del_14_c_org,
        #         self.num_tracer,
        #         self.num_box,
        #         self.remin_matrix,
        #     )

        ### Circulation ###
        d_dt_circ = (self.transport_matrix @ state_a.T).T[:, : self.num_box]
        # [5 x 5] x [5 x 6] = [5 x 6] --> [6 x 5] --> [6 x 3]

        ### Air-Sea Gas Exchange ###
        d_dt_gasexchange = airsea.gas_exchange(
            state_a[:, : self.num_box],
            self.num_tracer,
            self.num_box,
            self.CO2_atm_currentyr,
            self.d13C_atm_currentyr,
            self.D14C_atm_currentyr,
            self.surf_area,
            self.mass[2],
        )

        d_dt = np.zeros((self.num_tracer, self.num_box))

        # where you can turn on or off any processes
        d_dt += d_dt_circ
        d_dt += d_dt_geologic
        # d_dt += d_dt_export
        # d_dt += d_dt_remin
        # d_dt += d_dt_gasexchange

        return d_dt.flatten()

    def run_box_model(self, tmax, num_steps):
        """runs the box model with ODE solver giving stateV0 as initial condition"""
        start = time.time()

        # we don't want to store the
        self.time = np.linspace(
            0, tmax, num_steps
        )  # sets at what times to store the computed solution
        self.result = solve_ivp(
            self.box_model,
            [0, tmax],
            self.state_v0,
            method="RK45",
            t_eval=self.time,
            vectorized=True,
            rtol=1e-6,
            atol=1e-6,
            # jac = None,
            # min_step = 0.00000001,
        )
        # self.carbonate_chemistry = self.carb_chem1()  # shape = [tracer,box,year]
        # print("The shape of carbonate_chemistry is ", np.shape(self.carbonate_chemistry))
        end = time.time()

        self.time = np.flipud(self.result.t)  # plot from past to present
        self.time = self.time[10:]  # we dont care about the spin up
        self.output = self.result.y[:, 10:]  # we dont care about the spin up
        # print("length of ouput is ", len(self.output))
        # print("length of time is ", len(self.time))
        # print("The counter is ", self.counter)
        print("this solver took ", end - start, " seconds.")
        # print("the length of marchitto array is ", len(self.marchitto_rate_array))

        io.make_plot(self.time, self.output, self.carbonate_chemistry, self.mass)
        io.save_file(
            self.time, self.output, self.carbonate_chemistry,
        )


if __name__ == "__main__":
    ModelInstance = GoCModel(reminBoolean=True)
    ModelInstance.run_box_model(21000, 211)
    # time = np.linspace(10000, 17500)
    # plt.plot(time, ModelInstance.f_sub(time))
    # plt.show()
