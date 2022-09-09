"""Gulf of California
Regional Model
Going to move things to modules after they work in OOP first
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PyCO2SYS as pyco2
from scipy.integrate import solve_ivp
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

    def __init__(self):

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
        self.carbon = np.array([2400, 2400, 2300])  # umol/kg
        self.alkalinity = np.array([2450, 2450, 2400])  # umol/kg
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

        self.carbon_add_scenario = io.read_cadd_scenario(
            "data/ISchange/2Dinversion/Powell2Dinversion.txt"
        )
        self.carbon_add = self.carbon_add_scenario[0, 0]
        self.alk_dic_ratio = self.carbon_add_scenario[0, 1]
        svedrup_matrix = circulation.circ(
            self.num_box, self.num_bc, 0.45, 0.03, 0.03, 0.03, 0.03,
        )
        self.transport_matrix = circulation.make_transport_matrix(
            self.num_box, self.num_bc, svedrup_matrix, self.mass
        )
        self.export_matrix = np.array(
            [
                [0, 0, 0, 0, 1],  # GoC surface --> GoC subsurface
                [0, 0, 1, 0, 0],  # NP surface --> Marchitto
                [0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1],
            ]
        )

        self.remin_matrix = np.array(
            [
                [0, 0, 0, 0, 0.75],  # 0.75 for Marchitto
                [0, 0, 0.25, 0, 0],  # 0.25 for GoC Subsurface
                [0, 0, -0.25, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, -0.75],
            ]
        )

        self.result = None
        self.carbonate_chemistry = None
        self.time = None
        self.output = None

    def make_state_a(self, state_v, time, bc):
        """Gets called every year and makes new state in matrix format. Boxes are in columns and tracers are in
        rows.
        example:
        all tracers for box 3 === stateA[:,3]
        tracer 2 for all boxes === stateA[2,:]
        we feed in time evolving boundary condition from CYCLOPS every 100 years
        """
        time_rounded = int(time)

        if bc == "control":
            if time_rounded % 100 == 0:
                self.boundary_condition = io.read_bc(
                    "data/NoISchange/ForwardRun/control.txt", (time_rounded / 100)
                )

        if bc == "2dinversion":
            if time_rounded % 100 == 0:
                self.boundary_condition = io.read_bc(
                    "data/ISchange/2Dinversion/Powell2Dinversion.txt",
                    (time_rounded / 100),
                )

        if time_rounded % 100 == 0:
            idx = int(time_rounded / 100)
            self.carbon_add = self.carbon_add_scenario[idx, 0]
            self.alk_dic_ratio = self.carbon_add_scenario[idx, 1]
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

        """geologic carbon addition"""

        if box == "marchitto":
            i = 0
        elif box == "subsurface":
            i = 1
        elif box == "surface":
            i = 2

        carbon_flux = rate * 1e15 / 12 * 1e6 / self.mass[i]

        d_dt = np.zeros((self.num_tracer, self.num_box))
        # DIC
        d_dt[0, i] = carbon_flux
        # ALK
        d_dt[1, i] = 1 * carbon_flux
        # d13C
        d_dt[3, i] = -9 * carbon_flux
        # D14C
        d_dt[4, i] = -1000 * carbon_flux
        # Cum Carbon
        d_dt[5, i] = rate

        return d_dt

    def box_model(self, time, statev):
        # pylint: disable=unused-argument
        """
        box_model takes in current model state and organizes the data into the
        correct matrix notation. Then box_model finds the change in each tracer
        for a given time step (d_dt). d_dt is returned to the ODE solver.
        """

        state_a = self.make_state_a(statev, time, "control")
        time_bp = 20000 - time

        current_state = state_a[:, : self.num_box]

        ### Geologic Carbon Addition ###
        d_dt_geologic = np.zeros((self.num_tracer, self.num_box))

        # Marchitto box additions #
        if (time_bp < 16500) and (time_bp > 14500):
            d_dt_geologic += geologic.manual_carbon_add(
                self.num_tracer, self.num_box, 0.06, "marchitto", self.mass
            )
        if (time_bp < 12750) and (time_bp > 12000):
            d_dt_geologic += geologic.manual_carbon_add(
                self.num_tracer, self.num_box, 0.06, "marchitto", self.mass
            )

        # subsurface addition #
        if (time_bp < 18000) and (time_bp >= 16500):
            d_dt_geologic += geologic.manual_carbon_add(
                self.num_tracer, self.num_box, 0.05, "subsurface", self.mass
            )
            d_dt_geologic += geologic.manual_carbon_add(
                self.num_tracer, self.num_box, 0.06, "marchitto", self.mass
            )

        if (time_bp < 15500) and (time_bp >= 14500):
            d_dt_geologic += geologic.manual_carbon_add(
                self.num_tracer, self.num_box, 0.07, "subsurface", self.mass
            )

        if (time_bp <= 14500) and (time_bp >= 13500):
            d_dt_geologic += geologic.manual_carbon_add(
                self.num_tracer, self.num_box, 0.08, "subsurface", self.mass
            )

        if (time_bp < 13500) and (time_bp >= 12000):
            d_dt_geologic += geologic.manual_carbon_add(
                self.num_tracer, self.num_box, 0.08, "subsurface", self.mass
            )
            d_dt_geologic += geologic.manual_carbon_add(
                self.num_tracer, self.num_box, 0.1, "surface", self.mass
            )

        # surface addition #
        if (time_bp < 15500) and (time_bp > 14500):
            d_dt_geologic += geologic.manual_carbon_add(
                self.num_tracer, self.num_box, 0.075, "surface", self.mass
            )

        if (time_bp < 13500) and (time_bp > 12000):
            d_dt_geologic += geologic.manual_carbon_add(
                self.num_tracer, self.num_box, 0.1, "surface", self.mass
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
            self.remin_matrix,
        )

        ### Remineralization ###
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
            current_state,
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
            rtol=1e-6,
            atol=1e-6,
            # jac = None,
            # min_step = 0.00000001,
        )
        # self.carbonate_chemistry = self.carb_chem1()  # shape = [tracer,box,year]
        # print("The shape of carbonate_chemistry is ", np.shape(self.carbonate_chemistry))
        end = time.time()

        self.time = np.flipud(self.result.t)  # plot from past to present
        self.output = self.result.y

        # calculate manual cumulative carbon values based on lines 254-288
        self.cum_geologic_carbon_to_marchitto = 0.06 * 749 + 0.06 * 2000 + 0.06 * 1500
        self.cum_geologic_carbon_to_goc_sub = (
            0.05 * 1500 + 0.07 * 1000 + 0.08 * 1001 + 0.08 * 1500
        )
        self.cum_geologic_carbon_to_goc_surf = 0.1 * 1500 + 0.075 * 999 + 0.1 * 1499

        # print(
        #     "Manual cumulative carbon to the Marchitto box is ",
        #     self.cum_geologic_carbon_to_marchitto,
        #     "[PgC]",
        # )
        # print(
        #     "Manual cumulative carbon to the GoC subsurface is ",
        #     self.cum_geologic_carbon_to_goc_sub,
        #     "[PgC]",
        # )
        # print(
        #     "Manual cumulative carbon to the GoC surface is ",
        #     self.cum_geologic_carbon_to_goc_surf,
        #     "[PgC]",
        # )

        # print(
        #     "ODE solved tracer cumulative carbon to the Marchitto box is ",
        #     self.output[15, -1],
        #     "[PgC]",
        # )
        # print(
        #     "ODE solved tracer cumulative carbon to the GoC subsurface is ",
        #     self.output[16, -1],
        #     "[PgC]",
        # )
        # print(
        #     "ODE solved tracer cumulative carbon to the GoC surface is ",
        #     self.output[17, -1],
        #     "[PgC]",
        # )

        print("this solver took ", end - start, " seconds.")

        # print("Max for flattened DIC is ",self.result.y[0,-1].max()," ",self.result.y[1,-2].max()," ",self.result.y[2,-1])

        io.make_plot(self.time, self.result.y, self.carbonate_chemistry, self.mass)
        io.save_file(self.time, self.result.y, self.carbonate_chemistry)


if __name__ == "__main__":
    ModelInstance = GoCModel()
    ModelInstance.run_box_model(20000, 2001)
