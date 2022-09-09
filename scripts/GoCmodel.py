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
import src.product as product
import time

# import fluxengine as flx


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

        # mass of the atmopsere from NASA and from SCPM_parameters.py

        self.mass_of_atm = 5.1e18  # kg
        self.mol_of_atm = 28.97  # mean molecular weight ??units??

        # volume of atm (mols)

        self.atm_volume = (self.mass_of_atm * 1e3) / self.mol_of_atm

        # surface area of GoC and surface volume

        self.surf_volume = 1.65e13  # m^3
        self.surf_area = self.surf_volume / 200  # m^2       


        self.CO2_data = io.read_co2_data("data/observations/CO2data.txt")
        self.CO2_data_int = self.CO2_data[0, 1]

        self.c14_atm_data = io.read_14C_atm_data("data/observations/D14Cdata.txt")
        self.c14_atm_data_int = self.c14_atm_data[0, 1]

        self.d13C_atm_data = io.read_d13C_atm_data(
            "data/observations/d13Cdata_500yearsnotadded.txt"
        )
        self.d13C_atm_data_int = self.d13C_atm_data[0, 1]

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
        self.pH = None
        self.pco2 = None 

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

        if time_rounded % 100 == 0:
            idx = int(time_rounded / 100)
            self.CO2_data_int = self.CO2_data[idx, 1] #ppm

        if time_rounded % 100 == 0:
            idx = int(time_rounded / 100)
            self.c14_atm_data_int = self.c14_atm_data[idx, 1]

        if time_rounded % 100 == 0:
            idx = int(time_rounded / 100)
            self.d13C_atm_data_int = self.d13C_atm_data[idx, 1]

        # reshape flat array to rows as tracers and columns as boxes
        # (18,) -> (6,3)
        state_a = state_v.T.reshape(self.num_tracer, self.num_box)
        # print('state_v is', state_v)

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

    def carb_chem(self, surface_alk, surface_dic):
        """
        Converts DIC and ALK to microles/kg then
        uses pyCO2sys to solve carbonate chemistry
        returns DIC speciation, pH, omega and pCO2
        """
        # dic = []
        # alk = []
        # for i in range(3):
        #     dic.append(self.result.y[i, :])
        #     alk.append(self.result.y[i + 3, :])

        carbon_chemistry = pyco2.sys(
            par1=surface_alk, par2=surface_dic, par1_type=1, par2_type=2
        )
        values = ["HCO3", "CO3", "CO2", "pH", "saturation_calcite", "pCO2", "k_CO2"]
        carbonate_results = []
        for term in values:
            carbonate_results.append(carbon_chemistry[term])
        return np.array(carbonate_results)

    def air_sea_gas_exchange(self, current_state):
    
        #--------------------------------------------------------------------------------------------------- constants & variables
        

        Temp = 298.15 #Kelvin
        Sal = 35 #partperthousand
        SWD = 1029 #kg/m^3
        surf_gas_flux = 0.00003472 #m/s
        
        A1_C = -60.3409 # CO2 Solubility parameters from Weiss (1974), in mol/ (kg atm)
        A2_C = 93.4517
        A3_C = 23.3585
        B1_C = 0.023517
        B2_C = -0.023656
        B3_C = 0.0047036

        FK = 0.9995  # The thermodynamic fractionation factor for carbon isotopes in air-sea exchange (no units)
        FKR = 0.9990  

        #air-sea fractionation factors dC13
        FSA=(-9.866/(Temp) +1.02412)          
        FAS=(-0.373/(Temp) +1.00019)

        #air-sea fractionation factors dC14
        FSAR = 0.92182 
        FASR = 0.99786

        K0 = (np.exp((A1_C + A2_C * (100.0/(Temp)) # umol/ kg atm
                + A3_C*np.log(Temp/100.0) + (Sal) * (B1_C + B2_C 
                *((Temp)/100.0) + B3_C * (((Temp)/100.0)**2)))))
        
    
        #--------------------------------------------------------------------------------------------------- pco2 solver method controls

        pco2_method = 'Mathis'
        
        if pco2_method == "Mathis":
            surface_dic = current_state[0,2] 
            surface_alk = current_state[1,2]

            self.pco2 = (((2*surface_dic - surface_alk)**2) / surface_alk - surface_dic)
        
        
        if pco2_method == "carbcalc":
            surface_dic = current_state[0,2]  *1e-6
            surface_alk = current_state[1,2]  *1e-6
            Boron = 1.179e-5 * Sal

            K_0 = np.exp(-60.2409 + 9345.17/Temp + 23.3585*np.log(Temp/100) 
            + Sal * (0.023517 - 0.00023656*Temp +0.0047036*(Temp/100)**2) )

            K1 = np.exp(2.18867 - 2275.036/Temp - 1.468591 * np.log(Temp) 
            + (-0.138681 - 9.33291/Temp) * np.sqrt(Sal) + 0.0726483*Sal    
            - 0.00574938 * Sal **1.5)

            K2 = np.exp(-0.84226 - 3741.1288/Temp -1.437139 * np.log(Temp)
            + (-0.128417 - 24.41239/Temp)*np.sqrt(Sal) + 0.1195308 * Sal   
            - 0.0091284 * Sal **1.5 )

            Kb = np.exp( (-8966.90 - 2890.51*np.sqrt(Sal) - 77.942*Sal 
                + 1.726 * Sal **1.5 - 0.0993*Sal**2) / Temp                
                + (148.0248 + 137.194 * np.sqrt(Sal) + 1.62247 * Sal)       
                + (-24.4344 - 25.085 * np.sqrt(Sal) - 0.2474 * Sal) * np.log(Temp)
                + 0.053105 * np.sqrt(Sal) * Temp)


            H = 10**(-8)
                                 
            diff_H = H     
            tiny_diff_H = 1e-15 

            #iter = 0

            while diff_H > tiny_diff_H:

                H_old = H

                CA = surface_alk #umol/kg

                a = CA
                b = K1*(a - surface_dic)
                c = K1 * K2 * ((a - 2) * surface_dic)

                H = ((-1*b) + np.sqrt((b)**2 - 4*a*c)) / (2*a)

                diff_H = abs(H - H_old)
                #iter = iter + 1

            aq_CO2 = a / ((K1 / H) + 2*K1*(K2 / (H**2)))  
            #self.aq_CO2 = surface_dic / (1 + (self.K1 / self.H) + ((self.K1*self.K2) / (self.H)**2))
   
            
            self.pco2 = ((aq_CO2 / K0) * 1e6) #ppm
            
            self.pH = -np.log10(H)
           
       
        #--------------------------------------------------------------------------------------------------- carbon flux
    

        cflux1 = (SWD * K0 * surf_gas_flux * (self.CO2_data_int - self.pco2)) #umol/(s m^2)
        

        #--------------------------------------------------------------------------------------------------- d13C flux


        kinetic_frac = SWD * K0 * surf_gas_flux * FK # umol / (atm s)
        del_13_c_ppmil = current_state[3,2] / current_state[0,2] #ppmil

        SCPCO2 = kinetic_frac*(((FAS*(self.d13C_atm_data_int / self.CO2_data_int)) * self.CO2_data_int) - (FSA*(del_13_c_ppmil / self.pco2)*self.pco2)) # umol / m^2 s
       

        #--------------------------------------------------------------------------------------------------- d14C flux


        radio_kinetic_frac = SWD * K0 * surf_gas_flux * FKR # umol / (atm s)
        del_14_c_ppmil = current_state[4,2] / current_state[0,2] #ppmil

        RCPCO2 = radio_kinetic_frac*(((FASR*(self.c14_atm_data_int / self.CO2_data_int)) * self.CO2_data_int) - (FSAR*(del_14_c_ppmil / self.pco2) * self.pco2)) # umol / m^2 s
      
        
        return (cflux1, SCPCO2, RCPCO2)

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

        d_dt_geologic = np.zeros((self.num_tracer, self.num_box))

        #current_state = state_a[:, : self.num_box] + d_dt

        # Marchitto box additions #
        if (time_bp < 16500) and (time_bp > 14500):
            d_dt_geologic += self.geologic_carbon_add(0.06, "marchitto")
        if (time_bp < 12750) and (time_bp > 12000):
            d_dt_geologic += self.geologic_carbon_add(0.06, "marchitto")

        # subsurface addition #
        if (time_bp < 18000) and (time_bp >= 16500):
            d_dt_geologic += self.geologic_carbon_add(0.05, "subsurface")
            d_dt_geologic += self.geologic_carbon_add(0.06, "marchitto")

        if (time_bp < 15500) and (time_bp >= 14500):
            d_dt_geologic += self.geologic_carbon_add(0.07, "subsurface")

        if (time_bp <= 14500) and (time_bp >= 13500):
            d_dt_geologic += self.geologic_carbon_add(0.08, "subsurface")

        if (time_bp < 13500) and (time_bp >= 12000):
            d_dt_geologic += self.geologic_carbon_add(0.08, "subsurface")
            d_dt_geologic += self.geologic_carbon_add(0.1, "surface")

        # surface addition #
        if (time_bp < 15500) and (time_bp > 14500):
            d_dt_geologic += self.geologic_carbon_add(0.075, "surface")

        if (time_bp < 13500) and (time_bp > 12000):
            d_dt_geologic += self.geologic_carbon_add(0.1, "surface")

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
           
        # Air-Sea Gas Exchange
        
        current_state = state_a[:, : self.num_box] + d_dt
        cflux1, SCPCO2, RCPCO2 = self.air_sea_gas_exchange(current_state)
    
        d_dt[0,2] += cflux1*3.1536e7*self.surf_area / self.mass[2] #converting from umol/m^2s to umol/kg
        d_dt[3,2] += SCPCO2*3.1536e7*self.surf_area / self.mass[2]
        d_dt[4,2] += RCPCO2*3.1536e7*self.surf_area / self.mass[2]

        new_state = state_a[:, : self.num_box] + d_dt

        verbose = "True"

        if verbose == "True":
            
            print('d_dt 0,2 is', d_dt[0,2])
            #print("K0 is", K0)
            #print('K_0 is', self.K_0)
            print('pH is', self.pH)
            #print('min pH is', self.min_ph)
            #print('surface_dic is', self.surface_dic)
            #print('aqCO2', self.aq_CO2)
            print('cflux1 is', cflux1)
            #print('cflux_out is', self.cflux_out)
            #print('clfux_in', self.cflux_in)
            print('pco2 is', self.pco2)
            print('atm_co2 is', self.CO2_data_int)
            print('CO2 gradient', (self.CO2_data_int - self.pco2))
            #print('d13C flux is', SCPCO2)
            #print('d13C is', current_state[3,2] )
            #print('stateV0 is', self.state_v0)
            print("DIC surf is ", new_state[0,2]),#'DIC deep is', current_state[0,1],"DIC Marc is", current_state[0,0]," and ALK is ", current_state[1,2])
            print("Current year is", time_bp)
        
        if verbose == "False":
            pass
        
        
        
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
