"""Gulf of California
Regional Model
Going to move things to modules after they work in OOP first
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PyCO2SYS as pyco2
from scipy.integrate import solve_ivp
import src.inputoutput as io
import src.circulation as circulation

# import src.conversions as conversions


class GoCModel:
    """three box ocean model with circulation, biological pump, air sea gas exchange
    and DIC,ALK,P,N,d13C,D14C tracers. Box order is Baja California,
    Gulf of California-Deep, Gulf of California-Surface, North Pacific-
    Intermediate depth and North Pacific Surface"""

    def __init__(self):

        self.num_box = 3
        self.num_bc = 2
        self.num_tracer = 5
        self.boxlabel = [
            "Shadow zone source box",
            "Gulf of California-Subsurface",
            "Gulf of California-Surface",
        ]

        # kg,calculated from GoC volume
        # (1.45e+14 m3) * density of sw (kg/m3)
        self.goc_mass = 1.45e14 * 1026.8
        self.goc_model_mass = self.goc_mass * 0.2
        self.goc_surface_mass = self.goc_model_mass * 0.33  # kg
        self.goc_subsurface_mass = self.goc_model_mass * 0.67  # kg
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

        # Setting up inital values
        self.carbon = np.array([2400, 2400, 2300])  # umol/kg
        self.alkalinity = np.array([2450, 2450, 2450])  # umol/kg
        self.nitrate = np.array([30, 30, 30])  # umol/kg
        self.del_13_c = np.array([0.1, 0.1, 0.1]) * self.carbon  # permil
        self.del_14_c = np.array([0.1, 0.1, 0.1]) * self.carbon  # permil

        # Packing initial conditions in matrixes
        self.state_v0 = np.hstack(
            (self.carbon, self.alkalinity, self.nitrate, self.del_13_c, self.del_14_c)
        )
        self.boundary_condition = io.read_bc(
            "data/ISchange/2Dinversion/Powell2Dinversion.txt", 0
        )
        # self.boundary_condition = io.read_bc(
        #     "data/NoISchange/ForwardRun/control.txt", 0
        # )

        self.carbon_add_scenario = io.read_cadd_scenario(
            "data/ISchange/2Dinversion/Powell2Dinversion.txt"
        )
        self.carbon_add = self.carbon_add_scenario[0, 0]
        self.alk_dic_ratio = self.carbon_add_scenario[0, 1]
        svedrup_matrix = circulation.circ(
            self.num_box, self.num_bc, 0.45, 0.05, 0.05, 0.05, 0.05
        )
        self.transport_matrix = circulation.make_transport_matrix(
            self.num_box, self.num_bc, svedrup_matrix, self.mass
        )

        self.result = None
        self.carbonate_chemistry = None
        self.time = None
        self.output = None
        self.tracers_arr = np.zeros((5, 5, 1))

    def make_state_a(self, state_v, time):
        """Gets called every year and makes new state in matrix format. Boxes are in columns and tracers are in
        rows.
        example:
        all tracers for box 3 === stateA[:,3]
        tracer 2 for all boxes === stateA[2,:]
        we feed in time evolving boundary condition from CYCLOPS every 100 years
        """

        time_rounded = int(time)

        if time_rounded % 100 == 0:
            self.boundary_condition = io.read_bc(
                "data/ISchange/2Dinversion/Powell2Dinversion.txt", (time_rounded / 100)
            )

        # if time_rounded % 100 == 0:
        #     self.boundary_condition = io.read_bc(
        #         "data/NoISchange/ForwardRun/control.txt",
        #         (time_rounded / 100)

        #     )

        if time_rounded % 100 == 0:
            idx = int(time_rounded / 100)
            self.carbon_add = self.carbon_add_scenario[idx, 0]
            self.alk_dic_ratio = self.carbon_add_scenario[idx, 1]

        state_a = np.hstack(
            (state_v.T.reshape(self.num_tracer, self.num_box), self.boundary_condition)
        )

        return state_a

    def geologic_carbon_add(self):
        """geologic carbon addition"""

        carbon_flux_goc_source = (self.carbon_add * 0.95) / self.mass[0]
        carbon_flux_goc_subsurface = (self.carbon_add * 0.05) / self.mass[1]

        d_dt = np.zeros((self.num_tracer, self.num_box))
        # DIC to shadow source box
        d_dt[0, 0] = carbon_flux_goc_source
        # ALK to shadow source box
        d_dt[1, 0] = self.alk_dic_ratio * carbon_flux_goc_source
        # d13C to shadow source box
        d_dt[3, 0] = -9 * carbon_flux_goc_source
        # D14C to shadow source box
        d_dt[4, 0] = -1000 * carbon_flux_goc_source

        # DIC to GoC subsurface
        d_dt[0, 1] = carbon_flux_goc_subsurface
        # ALK to GoC subsurface
        d_dt[1, 1] = self.alk_dic_ratio * carbon_flux_goc_subsurface
        # d13C to GoC subsurface
        d_dt[3, 1] = -9 * carbon_flux_goc_subsurface
        # D14C to GoC subsurface
        d_dt[4, 1] = -1000 * carbon_flux_goc_subsurface

        return d_dt

    def carb_chem(self):
        """
        Converts DIC and ALK to microles/kg then
        uses pyCO2sys to solve carbonate chemistry
        returns DIC speciation, pH, omega and pCO2
        """
        dic = []
        alk = []
        for i in range(3):
            dic.append(self.result.y[i, :])
            alk.append(self.result.y[i + 3, :])

        carbon_chemistry = pyco2.sys(par1=alk, par2=dic, par1_type=1, par2_type=2)
        values = ["HCO3", "CO3", "CO2", "pH", "saturation_calcite", "pCO2"]
        carbonate_results = []
        for term in values:
            carbonate_results.append(carbon_chemistry[term])
        return np.array(carbonate_results)

    def box_model(self, time, statev):
        # pylint: disable=unused-argument
        """
        makes matrix with tracers in rows and boxes in columns from initial conditions.
        Then multiplies matrix by transport matrix to find the change in each.
        This is the derivative of dy/dt
        [:,: self.num_box] grabs all rows (tracers) and all columns up to the number of boxes
        from the box model (excluding boundary condition boxes)
        """

        state_a = self.make_state_a(statev, time)
        # io.make_text(state_a, self.tracers_arr)

        # multiplying tracers by fluxes
        d_dt = (self.transport_matrix @ state_a.T).T[:, : self.num_box]

        # d_dt += self.geologic_carbon_add()
        # d_dt += self.prod(stateA)
        # d_dt += self.Fix(stateA)

        return d_dt.flatten()

    def run_box_model(self, tmax, num_steps):
        """runs the box model with ODE solver giving stateV0 as initial condition"""
        self.time = np.linspace(0, tmax, num_steps)

        self.result = solve_ivp(
            self.box_model,
            [0, tmax],
            self.state_v0,
            method="RK45",
            t_eval=self.time,
            vectorized=True,
        )
        self.carbonate_chemistry = self.carb_chem()  # shape = [tracer,box,year]

        # plot from past to present
        self.time = np.flipud(self.result.t)
        self.output = self.result.y

        io.make_plot(self.time, self.result.y, self.carbonate_chemistry, self.mass)


if __name__ == "__main__":

    ModelInstance = GoCModel()
    ModelInstance.run_box_model(20000, 201)

