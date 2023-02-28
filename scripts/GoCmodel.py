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
import src.carbchem as cc
import time


class GoCModel:
    """three box ocean model with circulation, biological pump, air sea gas exchange
    and DIC,ALK,P,N,d13C,D14C tracers. Box order is Baja California,
    Gulf of California-Deep, Gulf of California-Surface, North Pacific-
    Intermediate depth and North Pacific Surface"""

    def __init__(self, geologic_d13c_source):

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

        # data we will compare model to
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

        if geologic_d13c_source == "AOM":
            self.geologic_d13c = -12  # same d13C as organic matter
            self.filename = "AOM_source"
        elif geologic_d13c_source == "CO2_dissolving_carbonates":
            self.geologic_d13c = -2.5  # CO2 is -5, CaCO3 is SW 0
            self.filename = "CO2carbonate_source"
        elif geologic_d13c_source == "biogenic_methane":
            self.geologic_d13c = -50
            self.filename = "methane_source"
        else:
            print(
                "This is not a correct geochemical pathway of geologic carbon addition."
            )
            exit()

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

        self.f_surf, self.f_sub, self.f_mar = io.read_files()

        self.state_copy = np.zeros((self.num_tracer, self.num_box))
        self.time_copy = 0
        self.optimized_timesteps = np.zeros((20000))
        self.geologic_add = np.array([0, 0, 0])
        self.counter = 1
        self.marchitto_idx = 0
        self.subsurface_idx = 1
        self.surface_idx = 2

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

    def obj_func(self, geologic_carbon_rate, box_idx):  # units of PgC
        """
        Rules:
        1. Algorithm to stop running when misfit < tolerance (0.1)
        2. No removal of carbon (geologic_carbon_rate =! 0)
        """

        # convert PgC to model concentration units (umol/kg)
        carbon_flux_conc = geologic_carbon_rate * 1e15 / 12 * 1e6 / self.mass[box_idx]

        # calculate D14C state with geologic carbon
        # all geologic carbon has a per mil value of -1000
        # carbon_flux is the additional change in concentration after carbon addition
        # new dc / new DIC, delta units
        del_14_c_model = (self.state_copy[4, box_idx] + (-1000 * carbon_flux_conc)) / (
            self.state_copy[0, box_idx] + carbon_flux_conc
        )
        # del_14_c_model = (modelstate[4, box_idx] + (-1000 * carbon_flux_conc)) / (modelstate[0, box_idx] + carbon_flux_conc)

        # print("MODEL D14C is ", del_14_c_model, " and the rate to get me here was ", geologic_carbon_rate)
        # misfit = abs((self.obs_d14c - del_14_c_model))
        misfit = abs((self.obs_d14c - del_14_c_model))
        return misfit  # per mil

    def box_model(self, time, statev):
        # pylint: disable=unused-argument
        """
        box_model takes in current model state and organizes the data into the
        correct matrix notation. Then box_model finds the change in each tracer
        for a given time step (d_dt). d_dt is returned to the ODE solver.
        """

        state_a = self.make_state_a(statev, time, "control")

        time_bp = round(21000 - time)

        current_state = state_a[:, : self.num_box]
        self.state_copy = state_a

        if self.time_copy == time_bp:
            self.counter += 1
            if self.counter > 200:
                print("current count is ", self.counter)
        else:
            # print("This year took ", self.counter, " steps.")
            self.counter = 1

        self.time_copy = round(time_bp)
        # print(self.time_copy)

        d_dt_geologic = np.zeros((self.num_tracer, self.num_box))

        optimization = "true"
        if optimization == "true":
            # Marchitto
            try:
                self.obs_d14c = self.f_mar(self.time_copy)
                self.marchitto_rate = optimize.minimize(
                    fun=self.obj_func,
                    x0=0.1,
                    method="TNC",
                    args=(self.marchitto_idx),
                    bounds=[(0, None)],
                    tol=0.1,
                ).x
            except:
                self.marchitto_rate = 0

            d_dt_geologic += geologic.carbon_add(
                self.num_tracer,
                self.num_box,
                self.marchitto_rate,
                "marchitto",
                self.mass,
                self.geologic_d13c,
            )

            # Subsurface
            try:
                self.obs_d14c = self.f_sub(self.time_copy)
                # multiple by 8 gets closer to line but takes longer
                self.subsurface_rate = (
                    3
                    * optimize.minimize(
                        fun=self.obj_func,
                        x0=0.1,
                        method="TNC",
                        args=(self.subsurface_idx),
                        bounds=[(0, None)],
                        tol=0.1,
                    ).x
                )
            except:
                self.subsurface_rate = 0
            d_dt_geologic += geologic.carbon_add(
                self.num_tracer,
                self.num_box,
                self.subsurface_rate,
                "subsurface",
                self.mass,
                self.geologic_d13c,
            )

            # Surface
            try:
                self.obs_d14c = self.f_surf(self.time_copy)
                # multiply by 7 gets closer to line but takes longer
                self.surface_rate = (
                    3
                    * optimize.minimize(
                        fun=self.obj_func,
                        x0=0.1,
                        method="TNC",
                        args=(self.surface_idx),
                        bounds=[(0, None)],
                        tol=0.1,
                    ).x
                )
            except:
                self.surface_rate = 0
        d_dt_geologic += geologic.carbon_add(
            self.num_tracer,
            self.num_box,
            self.surface_rate,
            "surface",
            self.mass,
            self.geologic_d13c,
        )

        ### Biological Productivity (Soft Tissue + Carbonate) ###
        d_dt_export, exportP, del_13_c_org, del_14_c_org = product.productivity(
            state_a[:, : self.num_box],
            self.boundary_condition,
            self.num_tracer,
            self.num_box,
            self.num_bc,
            self.CaRatio,
            self.export_matrix,
        )

        ### Remineralization ###
        d_dt_remin = np.zeros((self.num_tracer, self.num_box))

        d_dt_remin = product.remin(
            exportP,
            del_13_c_org,
            del_14_c_org,
            self.num_tracer,
            self.num_box,
            self.remin_matrix,
        )

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
        d_dt += d_dt_export
        d_dt += d_dt_remin
        d_dt += d_dt_gasexchange

        return d_dt.flatten()

    def run_box_model(self, tmax, num_steps):
        """runs the box model with ODE solver giving stateV0 as initial condition"""
        start = time.time()

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
            rtol=1e-2,
            atol=1e-2,
        )

        end = time.time()

        self.time = np.flipud(self.result.t)  # plot from past to present
        self.time = self.time[10:]  # we dont care about the spin up
        self.output = self.result.y[:, 10:]  # we dont care about the spin up

        carb_chem = cc.carb_chem(self.output)  # shape = [tracer,box,year]
        pH = carb_chem[3, :, :]

        print(
            "This solver took {:.2f} seconds for a ".format(end - start),
            tmax,
            " year simulation.",
        )
        io.make_plot(self.time, self.output, carb_chem, self.filename)
        # io.make_plot_interp(self.time, self.output)
        # io.save_rates_GoC_file(self.time, self.output, self.filename)
        io.save_file(self.time, self.output, self.filename)
        # io.save_file(self.time, self.output, "control_run")

    def make_AGU_plots(self):
        io.make_carbon_rate_plot(self.filename)
        io.save_file(self.time, self.result.y, self.carbonate_chemistry)
        io.save_rates_GoC_file(self.time, self.result.y, self.carbonate_chemistry)


if __name__ == "__main__":
    # AOM = GoCModel("AOM")
    # AOM.make_AGU_plots()
    # AOM.run_box_model(21000, 211)
    CO2 = GoCModel("CO2_dissolving_carbonates")
    CO2.run_box_model(21000, 211)
    # methane = GoCModel("biogenic_methane")
    # methane.run_box_model(21000, 211)

