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
                [0, 0, 0, 0, 0],  # GoC surface --> GoC subsurface
                [0, 0, 0, 0, 0],  # NP surface --> Marchitto
                [0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1],
            ]
        )

        self.remin_matrix = np.array(
            [
                [0, 0, 0, 0, 0.75],  # 0.75 for Marchitto
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

        self.surf, self.sub, self.mar = self.read_files()

        self.state_copy = np.zeros((self.num_tracer, self.num_box))
        self.time_copy = 0
        self.optimized_timesteps = np.zeros((20000))
        self.reminBoolean = reminBoolean
        self.geologic_add = np.array([0,0,0])

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

    # def obj_func(self, state, time):
    #     del_14_c_values = self.get_del_14_c_values(time, self.surf, self.sub, self.mar) # [1 x 3]
    #     del_14_c_change = del_14_c_values * state[0] - state[4]
    #     return del_14_c_change

    def obj_func(self, geologic_carbon_rate): # units of PgC​
        '''
        Rules:
        1. Algorithm to stop running when misfit < tolerance (0.1)
        2. No removal of carbon (geologic_carbon_rate =! 0)
        '''

        # convert PgC to model concentration units (umol/kg)
        carbon_flux = geologic_carbon_rate * 1e15 / 12 * 1e6 / self.mass[:self.num_box]

        # grab delta 14C value for the specific year
        del_14_c_obs_permil = self.get_del_14_c_values(self.time_copy, self.surf, self.sub, self.mar) # [1 x 3]
        # calculate D14C state with geologic carbon
        # all geologic carbon has a per mil value of -1000
        # carbon_flux is the additional change in concentration after carbon addition
        del_14_c_model = self.state_copy[4] + (-1000*carbon_flux)

        # convert model state to per mil units
        del_14_c_model_permil = del_14_c_model / (self.state_copy[0] + carbon_flux)

        misfit = np.sum((del_14_c_obs_permil - del_14_c_model_permil) ** 2)

        return misfit # per mil

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
        self.state_copy = current_state
        self.time_copy = int(time_bp)

        d_dt_geologic = np.zeros((self.num_tracer, self.num_box))

        # (Below) For every integer timestep, recalculate the geologic add, and use
        # the same geologic carbon addition for all increments within an
        # integer year

        '''
        if self.optimized_timesteps[self.time_copy-1] == 0:
            self.optimized_timesteps[self.time_copy-1] = 1;
            geologic_add_initial_guess = np.array([0,0,0])
            self.geologic_add = optimize.minimize(self.obj_func, geologic_add_initial_guess, tol=0.1, method="Powell").x
        print("Geologic Add at Time = " + str(self.time_copy) + ":", end=" ")
        print(self.geologic_add)

        ### Geologic Carbon Addition ###
        d_dt_geologic += geologic.manual_carbon_add(self.num_tracer, self.num_box, self.geologic_add[0], "marchitto", self.mass)
        d_dt_geologic += geologic.manual_carbon_add(self.num_tracer, self.num_box, self.geologic_add[1], "subsurface", self.mass)
        d_dt_geologic += geologic.manual_carbon_add(self.num_tracer, self.num_box, self.geologic_add[2], "surface", self.mass)
        '''

        """ Commenting out manual geologic carbon additions
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
        """

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
        if self.reminBoolean:
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
        # d_dt += d_dt_geologic
        d_dt += d_dt_export
        # d_dt += d_dt_remin
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

        print("this solver took ", end - start, " seconds.")

        io.make_plot(self.time, self.result.y, self.carbonate_chemistry, self.mass)
        io.save_file(self.time, self.result.y, self.carbonate_chemistry)


if __name__ == "__main__":
    ModelInstance = GoCModel(reminBoolean=True)
    ModelInstance.run_box_model(20000, 2001)
