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
import time

# import src.conversions as conversions


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

        # Setting up inital values
        self.carbon = np.array([2400, 2400, 2300])  # umol/kg
        self.alkalinity = np.array([2450, 2450, 2450])  # umol/kg
        self.nitrate = np.array([30, 30, 30])  # umol/kg
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

        # Packing initial conditions in matrixes
        # flat array (18,)
        self.state_v0 = np.hstack(
            (
                self.carbon,
                self.alkalinity,
                self.nitrate,
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

        # reshape flat array to rows as tracers and columns as boxes
        # (18,) -> (6,3)
        state_a = state_v.T.reshape(self.num_tracer, self.num_box)

        # adds the boundary condition (column index of [:,4] and [:,5])
        # -> (6,5)
        state_a = np.hstack((state_a, self.boundary_condition,))
        # clearing cumulative carbon tracer so that it is not affected
        # by circulation matrix mutiplication
        state_a[5, 0] = 0
        state_a[5, 1] = 0
        state_a[5, 2] = 0

        return state_a

    def geologic_carbon_add(self, rate, box):
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
        values = ["HCO3", "CO3", "CO2", "pH", "saturation_calcite", "pCO2", "k_CO2"]
        carbonate_results = []
        for term in values:
            carbonate_results.append(carbon_chemistry[term])
        return np.array(carbonate_results)

    def gas_exchange(
        self,
        carbonic_acid,
        k0,
        surface_area,
        carbon_13_atm_moles,
        carbon_14_atm_moles,
        carbon_13_surface_umol,
        carbon_14_surface_umol,
        atm_co2=280,
        temp=25,
    ):
        """calculates air to sea and sea to air gas exchange"""
        N = 15
        for dt in range(1, N + 1):
            # 1536000 / 16 / 32
            carbon_ingassed = (
                k0 * atm_co2 * surface_area * (1536000 / ((N + 1) / 2 * N)) * dt
            )
            # * (1500* 1/45) * 1024;
            # reconsider *1024 ... µM= 1e-3mol/m3 ... nothing about kg ...
            # I don't know where 153600 comes from? - Ryan

            carbon_outgassed = (
                carbonic_acid * surface_area * (1536000 / ((N + 1) / 2 * N)) * dt
            )
            # * (1500* 1/45) * 1024;

            d13_outgassed = (
                (carbon_13_surface_umol / self.state_v0[2])
                + (0.107 * temp - 10.53 - 0.875)
            ) * carbon_outgassed
            d13_ingassed = (carbon_13_atm_moles / atm_co2 - 0.875) * carbon_ingassed
            d14_outgassed = (
                (carbon_14_surface_umol / self.state_v0[2])
                + 2 * (0.107 * temp - 10.53 - 0.875)
            ) * carbon_outgassed
            d14_ingassed = (carbon_14_atm_moles / atm_co2 - 2 * 0.875) * carbon_ingassed

            # difference in carbon change converted to concentration
            # not sure if state_v0 is the way to adjust the carbon inventory during the ODE solver
            self.state_v0[2] += (carbon_ingassed - carbon_outgassed) / self.mass[
                2
            ]  # mol / time step
        # This will be used when I change the atmospheric CO2 value in CYCLOPS
        # atm.ppm -= (carbon_ingassed.sum() - carbon_outgassed.sum()) / (1.773E+20)

        # after this I will need to rerun the carbonate solver for the next time step
        return (
            d13_ingassed,
            d14_ingassed,
            d13_outgassed,
            d14_outgassed,
            carbon_ingassed,
            carbon_outgassed,
        )

    # Biological Productivity
    def ComputeExportN(self, state):
        idxN = 2  # index of nitrate in the tracers array
        boxesN = [2, 4]  # GoC Surface and NP Surface

        N = state.reshape(self.num_tracer, self.num_box)  # [6 x 3]
        N = np.hstack((N, self.boundary_condition))  # [6 x 5]
        N = N[idxN, :] / self.mass  # [1 x 5] converting to concentration

        ExportN = np.zeros(self.num_box + self.num_bc)
        SetN = np.array([0, 0, 1e-6, 0, 1e-7])

        timescale = 1  # year
        for box in boxesN:
            if N[box] - SetN[box] > 0:
                ExportN[box] = (
                    (N[box] - SetN[box]) / timescale * self.mass[box]
                )  # umol surface N/year
                self.boundary_condition[idxN, 4] -= ExportN[4]
                self.boundary_condition[0] -= ExportN[4] * 106 / 16  # Redfield ratio
            else:
                pass  # not enough nutrients to sustain productivity

        product = self.export_matrix @ ExportN  # [5 x 5] x [5,] = [5,]
        # Let X be the amount from GoC surface to GoC subsurface
        # Let Y be the amount from NP surface to Marchitto
        # Then, ExportN will equal a column vector [0,0,X,0,Y]
        # Finally, EM @ ExportN will equal a column vector [Y,X,-X,0,Y],
        # which is correct and can be added to d_dt

        product = product[
            : self.num_box
        ]  # the boundary conditions don't receive biological productivity

        return product

    def box_model(self, time, statev):
        # pylint: disable=unused-argument
        """
        makes matrix with tracers in rows and boxes in columns from initial conditions.
        Then multiplies matrix by transport matrix to find the change in each.
        This is the derivative of dy/dt
        [:,: self.num_box] grabs all rows (tracers) and all columns up to the number of boxes
        from the box model (excluding boundary condition boxes)
        """

        state_a = self.make_state_a(statev, time, "control")

        # multiplying tracers by fluxes
        d_dt = (self.transport_matrix @ state_a.T).T[
            :, : self.num_box
        ]  # [5 x 5] x [5 x 6] = [5 x 6]
        # [6 x 3][all tracers, all boxes (excluding boundary conditions)]

        time_bp = 20000 - time

        # Marchitto box additions #
        if (time_bp <= 16500) and (time_bp > 14500):
            d_dt += self.geologic_carbon_add(0.06, "marchitto")
        if (time_bp < 12750) and (time_bp > 12000):
            d_dt += self.geologic_carbon_add(0.06, "marchitto")

        # subsurface addition #
        if (time_bp < 18000) and (time_bp >= 16500):
            d_dt += self.geologic_carbon_add(0.05, "subsurface")
            d_dt += self.geologic_carbon_add(0.06, "marchitto")

        if (time_bp < 15500) and (time_bp >= 14500):
            d_dt += self.geologic_carbon_add(0.07, "subsurface")

        if (time_bp <= 14500) and (time_bp >= 13500):
            d_dt += self.geologic_carbon_add(0.08, "subsurface")

        if (time_bp < 13500) and (time_bp >= 12000):
            d_dt += self.geologic_carbon_add(0.08, "subsurface")
            d_dt += self.geologic_carbon_add(0.1, "surface")

        # surface addition #
        if (time_bp < 15500) and (time_bp > 14500):
            d_dt += self.geologic_carbon_add(0.075, "surface")

        if (time_bp < 13500) and (time_bp > 12000):
            d_dt += self.geologic_carbon_add(0.1, "surface")

        # Biological Productivity
        product = self.ComputeExportN(statev)
        d_dt[2] += product
        d_dt[0] += product * 106 / 16  # Redfield ratio

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
            rtol=1e-10,
            atol=1e-7,
        )
        self.carbonate_chemistry = self.carb_chem()  # shape = [tracer,box,year]
        end = time.time()

        self.time = np.flipud(self.result.t)  # plot from past to present
        self.output = self.result.y

        # calculate manual cumulative carbon values based on lines 254-288
        self.cum_geologic_carbon_to_marchitto = 0.06 * 749 + 0.06 * 2000 + 0.06 * 1500
        self.cum_geologic_carbon_to_goc_sub = (
            0.05 * 1500 + 0.07 * 1000 + 0.08 * 1001 + 0.08 * 1500
        )
        self.cum_geologic_carbon_to_goc_surf = 0.1 * 1500 + 0.075 * 999 + 0.1 * 1499

        print(
            "Manual cumulative carbon to the Marchitto box is ",
            self.cum_geologic_carbon_to_marchitto,
            "[PgC]",
        )
        print(
            "Manual cumulative carbon to the GoC subsurface is ",
            self.cum_geologic_carbon_to_goc_sub,
            "[PgC]",
        )
        print(
            "Manual cumulative carbon to the GoC surface is ",
            self.cum_geologic_carbon_to_goc_surf,
            "[PgC]",
        )

        print(
            "ODE solved tracer cumulative carbon to the Marchitto box is ",
            self.output[15, -1],
            "[PgC]",
        )
        print(
            "ODE solved tracer cumulative carbon to the GoC subsurface is ",
            self.output[16, -1],
            "[PgC]",
        )
        print(
            "ODE solved tracer cumulative carbon to the GoC surface is ",
            self.output[17, -1],
            "[PgC]",
        )

        print("this solver took ", end - start, " seconds.")

        # io.make_plot(self.time, self.result.y, self.carbonate_chemistry, self.mass)
        # io.save_file(self.time, self.result.y, self.carbonate_chemistry)


if __name__ == "__main__":
    ModelInstance = GoCModel()
    ModelInstance.run_box_model(20000, 2001)
