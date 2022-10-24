"""Gulf of California
Regional Model
Going to move things to modules after they work in OOP first
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PyCO2SYS as pyco2
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
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

        # mass of the atmopsere from NASA and from SCPM_parameters.py

        self.mass_of_atm = 5.1e18  # kg
        self.mol_of_atm = 28.97  # mean molecular weight ??units??

        # volume of atm (mols)

        self.atm_volume = (self.mass_of_atm * 1e3) / self.mol_of_atm

        # CO2 Solubility parameters from Weiss (1974), in mol/ (kg atm)
        self.A1_C = -60.3409
        self.A2_C = 93.4517
        self.A3_C = 23.3585
        self.B1_C = 0.023517
        self.B2_C = -0.023656
        self.B3_C = 0.0047036

        # surface area of GoC and surface volume

        self.surf_volume = 1.65e13  # m^3
        self.surf_area = self.surf_volume / 200  # m^2

        # piston velocity for air-sea gas exchange (from SCPM_paramters.py)

        self.PV0 = 3.0  # m/day

        # The "thermodynamic fractionation factor" for carbon isotopes in air-sea exchange
        self.FK = 0.9995  # no units...0.99915 Stabe carbon as per Schmittner et al(2013) and SCPM_parameters.py
        self.FKR = 0.9990  # no units...0.9990 Radiocarbon as per Toggweiler and Sarmiento (1985) SCPM_paramters.py

        # Setting up inital values
        self.carbon = np.array([2400, 2400, 2300])  # umol/kg
        self.alkalinity = np.array([2450, 2450, 2450])  # umol/kg
        self.phosphorus = np.array([0.001, 0.001, 0.001])  # umol/kg
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

        self.surf, self.sub, self.mar = self.read_files()

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

    def air_sea_gas_exchange(
        self, temp=25, Kelv=273.15, SWD=1029
    ):  # every string is a variable we do not have data for yet

        # K0 from GLODAP_processing.py and Wiess 1974...need to know the salinity (Sal) and its units
        self.K0 = np.exp(
            (
                self.A1_C
                + self.A2_C * (100.0 / ((temp + Kelv)))
                + self.A3_C * np.log((temp + Kelv) / 100.0)
                + ("Sal")
                * (
                    self.B1_C
                    + self.B2_C * ((temp + Kelv) / 100.0)
                    + self.B3_C * (((temp + Kelv) / 100.0) ** 2)
                )
            )
        )  # mol/ kg atm

        # air-sea surface gas transfer
        def makeFXarr(PV0, secsday=86400):
            makeFXarr = np.zeros([3, 1])  # [3x1] so it can multiply
            makeFXarr[2, 0] = self.PV0 / secsday  # but only doing surface box
            return makeFXarr  # m/s

        self.surf_gas_flux = makeFXarr(
            self.PV0, secsday=86400
        )  # * self.surface_area  # m^3 / s

        cflux1 = (
            SWD * self.K0 * 1e6 * self.surf_gas_flux * ("AtCO2" - "pCO2")
        )  # umol/(s m^2)
        cflux = cflux1 / self.surf_volume  # WHY DIVIDING BY VOLUME??
        Carbon_flux = -sum(cflux1)

        # d13C air-sea fractionation factors
        FSA = np.zeros([3, 1])
        FSA = -9.866 / (temp + Kelv) + 1.02412  # Mook (1974)  unitless
        FAS = np.zeros([3, 1])
        FAS = -0.373 / (temp + Kelv) + 1.00019  # Mook (1974)  unitless

        # radiocarbon air-sea fractination factors
        FSAR = 0.92182
        FASR = 0.99786

        # air-sea flux 13C
        kinetic_frac = (
            SWD * self.K0 * self.surf_gas_flux * self.FK * 1e6
        )  # umol / (atm s)
        del_13_c_ppmil = self.del_13_c[2, 0] / self.carbon  # ppmil

        SCPCO2 = kinetic_frac * (
            ((FAS * ("del_13_c_atm_ppmil" / "AtCO2")) * "AtCO2")
            - (FSA * (del_13_c_ppmil / "pCO2") * "pCO2")
        )  # umol / m^2 s
        Scflux = SCPCO2 / self.surf_volume  # Ocean boxes ...umol/ (s m^3) ??
        # AtSCflux=-sum(SCPCO2)/Varrat # Atmosphere

        # air-sea flux 14C
        radio_kinetic_frac = (
            SWD * self.K0 * self.surf_gas_fluc * self.FKR * 1e6
        )  # umol / (atm s)
        del_14_c_ppmil = self.del_14_c[2, 0] / self.carbon  # ppmil

        RCPCO2 = radio_kinetic_frac * (
            ((FASR * ("del_14_c_atm_ppmil" / "AtCO2")) * "AtCO2")
            - (FSAR * (del_14_c_ppmil / "pCO2") * "pCO2")
        )  # umol / m^2 s
        Rcflux = RCPCO2 / self.surf_volume
        # AtRCflux=-sum(RCPCO2)/Varrat # Atmosphere

    def read_files(self):
        obspath = "data/observations/"

        Rafter_surface = pd.read_csv(obspath + "Rafter_2019.tab", sep="\t", header=24)
        Rafter_surface = Rafter_surface.loc[(Rafter_surface["Habitat"] == "planktic")]
        Rafter_surface["Cal age [ka BP]"] = 1000 * Rafter_surface["Cal age [ka BP]"]
        Rafter_surface = Rafter_surface.sort_values(by=["Cal age [ka BP]"])

        Rafter_subsurface = pd.read_excel(obspath + "prafter-2019-Gulf-CA-Data-for-Ryan.xls")
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

        return Rafter_surface, Rafter_subsurface, Mar

    def get_del_14_c_values(self, time, surf, sub, mar):
        arr_surf = surf["Cal age [ka BP]"] - time
        arr_sub = sub["calendar age [kyr BP]"] - time
        arr_mar = mar["Cal.Age"] - time
        idx_surf = np.where(arr_surf < 0, arr_surf, -np.inf).argmax()
        idx_sub = np.where(arr_sub < 0, arr_sub, -np.inf).argmax()
        idx_mar = np.where(arr_mar < 0, arr_mar, -np.inf).argmax()
        del_14_c_values = np.array([mar["D14C"][idx_mar],
                                    sub["D14C"][idx_sub],
                                    surf["Δ14C [‰]"][idx_surf]])
        return del_14_c_values

    def obj_func(self, state, time):
        del_14_c_values = self.get_del_14_c_values(time, self.surf, self.sub, self.mar) # [1 x 3]
        # weights = np.array([1,1,1]) # [3 x 1]
        # new_del_14_c = (state[4] + del_14_c_change) / state[0]
        # error = ((new_del_14_c - del_14_c_values) @ weights) ** 2
        # return error
        del_14_c_change = del_14_c_values * state[0] - state[4]
        # print(del_14_c_change)
        return del_14_c_change

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

        time_bp = 20000 - time

        time_rounded = int(time_bp)
        print(time_rounded)

        # if time_rounded % 100 == 0:
        # self.state = state_a[:,:self.num_box]
        # self.time_bp = time_bp
            # guess = np.array([0,0,0])
            # xmin = minimize(self.obj_func, x0=guess, method="Powell").x
            # print(xmin)
            # print(self.obj_func(xmin))
            # self.marchitto_add = xmin[0]
            # self.subsurface_add = xmin[1]
            # self.surface_add = xmin[2]

        geologic_add = -1 * self.obj_func(state_a[:,:self.num_box], time_bp) / 0.5e7
        print(geologic_add)
        self.marchitto_add = geologic_add[0]
        self.subsurface_add = geologic_add[1]
        self.surface_add = geologic_add[2]
        # print(geologic_add)
        d_dt_geologic = np.zeros((self.num_tracer, self.num_box))
        d_dt_geologic += self.geologic_carbon_add(self.marchitto_add, "marchitto")
        d_dt_geologic += self.geologic_carbon_add(self.subsurface_add, "subsurface")
        d_dt_geologic += self.geologic_carbon_add(self.surface_add, "surface")

        # # Marchitto box additions #
        # if (time_bp < 16500) and (time_bp > 14500):
        #     d_dt_geologic += self.geologic_carbon_add(0.06, "marchitto")
        # if (time_bp < 12750) and (time_bp > 12000):
        #     d_dt_geologic += self.geologic_carbon_add(0.06, "marchitto")
        #
        # # subsurface addition #
        # if (time_bp < 18000) and (time_bp >= 16500):
        #     d_dt_geologic += self.geologic_carbon_add(0.05, "subsurface")
        #     d_dt_geologic += self.geologic_carbon_add(0.06, "marchitto")
        #
        # if (time_bp < 15500) and (time_bp >= 14500):
        #     d_dt_geologic += self.geologic_carbon_add(0.07, "subsurface")
        #
        # if (time_bp <= 14500) and (time_bp >= 13500):
        #     d_dt_geologic += self.geologic_carbon_add(0.08, "subsurface")
        #
        # if (time_bp < 13500) and (time_bp >= 12000):
        #     d_dt_geologic += self.geologic_carbon_add(0.08, "subsurface")
        #     d_dt_geologic += self.geologic_carbon_add(0.1, "surface")
        #
        # # surface addition #
        # if (time_bp < 15500) and (time_bp > 14500):
        #     d_dt_geologic += self.geologic_carbon_add(0.075, "surface")
        #
        # if (time_bp < 13500) and (time_bp > 12000):
        #     d_dt_geologic += self.geologic_carbon_add(0.1, "surface")

        # Biological Productivity (Soft Tissue + Carbonate)
        d_dt_export, exportP, del_13_c_org, del_14_c_org = product.productivity(
            state_a[:, : self.num_box], self.boundary_condition,
            self.num_tracer, self.num_box, self.num_bc, self.CaRatio,
            self.export_matrix, self.remin_matrix
        )
        d_dt_remin = product.remin(
            exportP, del_13_c_org, del_14_c_org,
            self.num_tracer, self.num_box, self.remin_matrix)
        # multiplying tracers by fluxes
        d_dt = (self.transport_matrix @ state_a.T).T[:, : self.num_box]  # [5 x 5] x [5 x 6] = [5 x 6] --> [6 x 5] --> [6 x 3]

        d_dt += d_dt_geologic
        d_dt += d_dt_export
        d_dt += d_dt_remin

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

        io.make_plot(self.time, self.result.y, self.carbonate_chemistry, self.mass)
        # io.save_file(self.time, self.result.y, self.carbonate_chemistry)

    def plot_rate(self):
        rate_geologic_carbon_to_marchitto = np.zeros((20000))
        rate_geologic_carbon_to_goc_sub = np.zeros((20000))
        rate_geologic_carbon_to_goc_surf = np.zeros((20000))

        rate_geologic_carbon_to_goc_sub[12000:13500] = 0.08
        rate_geologic_carbon_to_goc_sub[13500:14500] = 0.08
        rate_geologic_carbon_to_goc_sub[14500:15500] = 0.07
        rate_geologic_carbon_to_goc_sub[16500:18000] = 0.05

        rate_geologic_carbon_to_goc_surf[12000:13500] = 0.01
        rate_geologic_carbon_to_goc_surf[14500:15500] = 0.075
        rate_geologic_carbon_to_goc_surf[14500:15500] = 0.075

        rate_geologic_carbon_to_marchitto[12000:12750] = 0.06
        rate_geologic_carbon_to_marchitto[14500:16500] = 0.06
        rate_geologic_carbon_to_marchitto[16500:18000] = 0.08

        plt.plot(rate_geologic_carbon_to_goc_sub, label="GoC sub")
        plt.plot(rate_geologic_carbon_to_goc_surf, label="GoC surf")
        plt.plot(rate_geologic_carbon_to_marchitto, label="Marchitto")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    ModelInstance = GoCModel()
    ModelInstance.run_box_model(20000, 2001)
    # ModelInstance.plot_rate()
