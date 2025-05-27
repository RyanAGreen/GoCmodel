"""Gulf of California
Regional Model
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
import sys
from multiprocessing import Pool, cpu_count
import csv

# Check if at least one argument is provided
# if len(sys.argv) > 1:
#     print(f"Arguments: {sys.argv}")
#     argument = sys.argv[4] 
#     print(f"The provided boundary condition is: {argument}")
# else:
#     print(f"Arguments: {sys.argv}")
#     print("No boundary condition provided. Using 'coupled' by default.")
#     argument = "coupled"

class GoCModel:
    """three box ocean model with circulation, biological pump, air sea gas exchange
    and DIC,ALK,P,N,d13C,D14C tracers. Box order is Baja California,
    Gulf of California-Deep, Gulf of California-Surface, North Pacific-
    Intermediate depth and North Pacific Surface"""

    def __init__(self, geologic_d13c, ALK_DIC_ratio, experiment, boundary_condition, marchitto_mass, mixing_rate):
        self.num_box = 3
        self.num_bc = 2
        self.num_tracer = 6

        # kg,calculated from GoC volume
        # 1.45e14 m3 from Rebekah K. Nix. "The Gulf of California: A Physical, Geological, and Biological Study" (PDF). University of Texas at Dallas. Retrieved April 10, 2010.
        self.goc_mass = 1.45e14 * 1026.8
        # 550 km length * 150 m width * 200 m depth
        self.goc_surface_mass = 1.65e13 * 1026.8  # kg
        # 550 km length * 150 m width * 400 m depth
        self.goc_subsurface_mass = 3.3e13 * 1026.8  # kg

        self.goc_source_mass = marchitto_mass
        self.np_surf_mass = self.goc_source_mass * 20  # kg
        self.np_mid_mass = self.np_surf_mass * 2  # kg

        # array of water masses for each box
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

        # reconstructed data we compare to simulated data
        self.CO2_atm = io.read_co2_data("data/observations/CO2data.txt")
        self.D14C_atm = io.read_14C_atm_data("data/observations/D14Cdata.txt")
        self.d13C_atm = io.read_d13C_atm_data(
            "data/observations/d13Cdata_500yearsnotadded.txt"
        )

        # Setting up inital values for model
        self.carbon = np.array([2350, 2300, 2100])  # umol/kg
        self.alkalinity = np.array([2420, 2420, 2410])  # umol/kg
        self.phosphorus = np.array([30, 30, 30])  # umol/kg
        self.del_13_c = (
            np.array([0.1, 0.1, 0.1]) * self.carbon
        )  # delta [permil] * concentration
        self.del_14_c = (
            np.array([0.1, 0.1, 0.1]) * self.carbon
        )  # delta [permil] * concentration

        self.geologic_d13c = geologic_d13c
        self.ALK_DIC_ratio = ALK_DIC_ratio
        self.filename = "d13c-" + str(geologic_d13c)
        self.filename += "_ALK_DIC-" + str(self.ALK_DIC_ratio) + "_"
        self.experiment = experiment
        self.filename += experiment
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
        # self.boundary_condition = io.read_bc(
        #     "data/NoISchange/ForwardRun/control.txt", 0
        # )
        # normal boundary condition
        self.filename += "_" + boundary_condition
        # ALL CYCLOPS BOUNDARY CONDITIONS ARE BASED ON NP+LC+PF instead of 
        if boundary_condition == "coupled":
            """carbon added to CYCLOPS based on GoC optimization"""
            self.boundary_condition = io.read_bc("data/model/CoupledRun.txt", 0)
            self.bc = "coupled"

        elif boundary_condition == "CYCLOPS_control":
            """No carbon to CYCLOPS added and it uses the NP+LC+PF scenario
            """
            self.boundary_condition = io.read_bc("data/model/Control_noheaders.txt", 0)
            self.bc = "CYCLOPS_control"

        # boundary condition for discussion figure
        # self.boundary_condition = io.read_bc(
        #     "data/model/NP_LC_PF_forward_NP_CO2.txt", 0
        # )
        # this is for increased mixing from NP to Marchitto
        svedrup_matrix = circulation.circ(
            self.num_box,
            self.num_bc,
            0.45,
            mixing_rate,
            0.03,
            0.03,
            0.03,
        )
        # svedrup_matrix = circulation.circ(
        #     self.num_box,
        #     self.num_bc,
        #     0.45,
        #     0.03,
        #     0.03,
        #     0.03,
        #     0.03,
        # )
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
        if mixing_rate > 1:
            self.filename = "low_isolation_control_HCO3_d13c-" + str(geologic_d13c)
        else:
            self.filename = "high_isolation_control_HCO3_d13c-" + str(geologic_d13c)

    def make_state_a(self, state_v, time):
        """Gets called every year and makes new state in matrix format. Boxes are in columns and tracers are in
        rows.
        example:
        all tracers for box 3 === stateA[:,3]
        tracer 2 for all boxes === stateA[2,:]
        we feed in time evolving boundary condition from CYCLOPS every 100 years
        """
        time_rounded = int(time)
        spinuptime = 1000

        # this is not actually GoC control, this is no carbon added in CYCLOPS
        if self.bc == "CYCLOPS_control":
            # after spin up, update every 100 years
            if (time_rounded % 100 == 0) and (time_rounded >= spinuptime):
                self.boundary_condition = io.read_bc(
                    "data/model/Control_noheaders.txt",
                    ((time_rounded - spinuptime) / 100),
                )
            # during spin up time
            elif time_rounded < spinuptime:
                self.boundary_condition = io.read_bc(
                    "data/model/Control_noheaders.txt", 0
                )
        elif self.bc == "coupled":
            # after spin up, update every 100 years
            if (time_rounded % 100 == 0) and (time_rounded >= spinuptime):
                self.boundary_condition = io.read_bc(
                    "data/model/CoupledRun.txt", ((time_rounded - spinuptime) / 100)
                )
            # during spin up time
            elif time_rounded < spinuptime:
                self.boundary_condition = io.read_bc("data/model/CoupledRun.txt", 0)

        if self.bc == "discussionfig":
            if (time_rounded % 100 == 0) and (time_rounded >= spinuptime):
                self.boundary_condition = io.read_bc(
                    "data/model/NP_LC_PF_forward_NP_CO2.txt",
                    ((time_rounded - spinuptime) / 100),
                )
            elif time_rounded < spinuptime:
                self.boundary_condition = io.read_bc(
                    "data/model/NP_LC_PF_forward_NP_CO2.txt", 0
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
        state_a = np.hstack(
            (
                state_v_reshaped,
                self.boundary_condition,
            )
        )
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
        2. No removal of carbon (geologic_carbon_rate =!  0)
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

        state_a = self.make_state_a(statev, time)

        current_state = state_a[:, : self.num_box]
        self.state_copy = state_a

        # round the time step to whole number and
        time_bp = round(21000 - time)
        if self.time_copy == time_bp:
            # counting how many steps per year
            self.counter += 1
            if self.counter > 500:
                print("current count is ", self.counter)
                # stop the code here
                exit()
        else:
            # print("This year took ", self.counter, " steps.")
            self.counter = 1
        self.time_copy = round(time_bp)

        # might not need this? Don't think I need to initialize each d_dt matrix
        d_dt_geologic = np.zeros((self.num_tracer, self.num_box))

        if self.experiment == "optimization":  # inverse run
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
                self.ALK_DIC_ratio,
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
                self.ALK_DIC_ratio,
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
                self.ALK_DIC_ratio,
            )

        elif self.experiment == "control":
            d_dt_geologic = 0
        elif self.experiment == "forward":  # forward run
            if int(time) > 1000:  # accounting for spin up time
                geologic_rates = io.read_all_geologic_rates(
                    "results/simulations/GoC_rates.txt"
                )
                # geologic_rates_total = io.read_all_geologic_rates(
                #     "results/total_GoC_rates.txt"
                # )

                # adding all to marchitto for the experiments in disucssion figure 6
                # d_dt_geologic += geologic.carbon_add(
                #     self.num_tracer,
                #     self.num_box,
                #     geologic_rates_total[int((time - 1000) / 100), 0],
                #     "subsurface",
                #     self.mass,
                #     self.geologic_d13c,
                #     self.ALK_DIC_ratio,
                # )

                d_dt_geologic += geologic.carbon_add(
                    self.num_tracer,
                    self.num_box,
                    geologic_rates[int((time - 1000) / 100), 0],
                    "marchitto",
                    self.mass,
                    self.geologic_d13c,
                    self.ALK_DIC_ratio,
                )

                d_dt_geologic += geologic.carbon_add(
                    self.num_tracer,
                    self.num_box,
                    geologic_rates[int((time - 1000) / 100), 1],
                    "subsurface",
                    self.mass,
                    self.geologic_d13c,
                    self.ALK_DIC_ratio,
                )
                d_dt_geologic += geologic.carbon_add(
                    self.num_tracer,
                    self.num_box,
                    geologic_rates[int((time - 1000) / 100), 2],
                    "surface",
                    self.mass,
                    self.geologic_d13c,
                    self.ALK_DIC_ratio,
                )
        else:
            print("please provide a correct experiment argument.")
            exit()

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
        # [5 x 5] * [5 x 6] = [5 x 6] --> [6 x 5] --> [6 x 3]

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
        # if time_bp % 500 == 0:
        #     print("the year is ", time_bp)
        # where you can turn on or off any processes
        d_dt += d_dt_circ
        d_dt += d_dt_geologic
        d_dt += d_dt_export
        d_dt += d_dt_remin
        d_dt += d_dt_gasexchange

        return d_dt.flatten()
    
    def CheckRate(self,filename):
        df = pd.read_table(
            "~/GoCmodel/results/simulations/" + filename,
            sep="\s+",
            header=None,
        )
        total_carbon_added = df[16][200] + df[17][200] + df[18][200]
        print("TOTAL CARBON IS: ", total_carbon_added)
        return total_carbon_added

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
            "This solver (d13C = "
            + str(self.geologic_d13c)
            + ", ALK/DIC = "
            + str(self.ALK_DIC_ratio)
            + ") took {:.2f} seconds for a".format(end - start),
            tmax,
            "year simulation.",
        )

        io.save_file(self.time, self.output, pH, self.filename)
        # new_total_carbon = self.CheckRate("d13c--1_ALK_DIC-1_optimization_coupled.txt")
        # Calculate total carbon rate directly from self.output
        # Extract cumulative carbon values and calculate total carbon adfded from the last time step
        mar_cum_carbon = self.output[15, -1]
        goc_sub_cum_carbon = self.output[16, -1]
        goc_surf_cum_carbon = self.output[17, -1]

        # Calculate total carbon added over the 20,000-year period
        total_carbon_added = mar_cum_carbon + goc_sub_cum_carbon + goc_surf_cum_carbon


        # print("total rate is ", total_carbon_added)
        # io.save_file(self.time, self.output, pH, "NoRegionalIsolation")
        # io.save_file(self.time, self.output, pH, "NoRegionalIsolation_optimized_test")
        return total_carbon_added
def convert_per_sec_to_per_year(per_sec):
    per_year = per_sec * 60 * 60 * 24 * 365
    return per_year

def calc_tau(volume, inflow):
    # all units are m3 and years
    # fluxes are is in m3/year
    tau = volume / inflow
    print("tau is ", tau)
    return tau

    
def run_simulation(params):
    volume_factor, mixing_rate = params
    # volume_factor = 5
    # mixing_rate = 5
    goc_volume = 1.45e14  # in m^3
    marchitto_mass = volume_factor * goc_volume * 1026.8  # Convert to kg
    # geologic_d13c = -2.5
    geologic_d13c = -8.9

    ALK_DIC_ratio = 1
    # ALK_DIC_ratio = 0
    experiment = "optimization"
    boundary_condition = "coupled"
    
    model = GoCModel(geologic_d13c, ALK_DIC_ratio, experiment, boundary_condition, marchitto_mass, mixing_rate)
    new_total_carbon = model.run_box_model(21000, 211)
    marchitto_volume = marchitto_mass / 1026.8
    inflow = 0.48e6 + mixing_rate * 1e6  # m3/s # double check why its 0.48e6 instead of 0.45e6
    inflow = convert_per_sec_to_per_year(inflow)
    tau = calc_tau(marchitto_volume, inflow)
    # print("simulation complete")
    return (volume_factor, mixing_rate, new_total_carbon,tau)

if __name__ == "__main__":
    ###### to run a single simulation
    # Example parameters
    # volume_factor = 10.86
    # mixing_rate = 4.29
    # # volume_factor = 15
    # # mixing_rate = 0.01

    # # Run the simulation with specific parameters
    # result = run_simulation((volume_factor, mixing_rate))       

    ###### to run 2 simulations at once
    # params = [
    # (10.86, 4.29),         # volume_factor = 10.86, mixing_rate = 4.29 m^3/s
    # (15, 0.01)       # volume_factor = 15, mixing_rate = 0.01 Sv converted to m^3/s
    # ]
    params = [
    (5.6, 4.7),         # volume_factor = 10.86, mixing_rate = 4.29 m^3/s
    (9.65, 0.5)       # volume_factor = 15, mixing_rate = 0.01 Sv converted to m^3/s
    ]
    with Pool(32) as pool:
        results = pool.map(run_simulation, params)
    print("All simulations complete")

    # to run for many parameters
    # num_samples = 10  # For example
    # volume_factors = np.linspace(0.5, 15, num_samples)  # From 1x to 1000x
    # mixing_rates = np.linspace(0.01, 10, num_samples)  # From 0.1 Sv to 500 Sv

    # params = [(v, m) for v in volume_factors for m in mixing_rates]

    # # Run simulations in parallel
    # # num_cores = max(1, cpu_count() - 5)  # Leave 5 cores open
    # # print("cores: ", cpu_count())
    # with Pool(32) as pool:
    #     results = pool.map(run_simulation, params)
    # print("All simulations complete")

    # # Save results to a CSV file
    # with open('results.csv', 'w', newline='') as csvfile:
    #     fieldnames = ['volume_factor', 'mixing_rate', 'total_carbon_added', 'tau']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for res in results:
    #         writer.writerow({'volume_factor': res[0], 'mixing_rate': res[1], 'total_carbon_added': res[2], 'tau': res[3]})

    # print("results saved to results.csv")
    # Extract results
    # volume_factors = np.array([res[0] for res in results])
    # mixing_rates = np.array([res[1] for res in results])
    # total_carbons = np.array([res[2] for res in results])

    # # Reshape results for contour plotting
    # volume_factors = volume_factors.reshape((num_samples, num_samples))
    # mixing_rates = mixing_rates.reshape((num_samples, num_samples))
    # total_carbons = total_carbons.reshape((num_samples, num_samples))

    # # Plot the results
    # plt.contourf(volume_factors, mixing_rates, total_carbons, cmap='viridis')
    # plt.colorbar(label='Total Carbon Added')
    # plt.xlabel('Volume Factor')
    # plt.ylabel('Mixing Rate (Sv)')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title('Total Carbon Added vs. Volume Factor and Mixing Rate')
    # plt.show()

#     def run_simulation(self, params):
#         volume_factor, mixing_rate = params
#         marchitto_mass = volume_factor * self.goc_mass * 1026.8  # Convert to kg
#         model = GoCModel(-2.5, 1, "optimization", "coupled", marchitto_mass, mixing_rate)
#         model.run_box_model(21000, 211) # 21000 years, 211 steps
#         return (volume_factor, mixing_rate, model.cum_geologic_carbon)


