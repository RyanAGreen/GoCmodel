# from IPython import get_ipython
# get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate

def RtoF(R): # conversion of isotope ratios to fractional abundances
    # convert isotope ratio to fractional abundance of isotope
    return R/(1+R)
def FtoR(F): # conversion of isotope fractional abundances to ratios
    # convert fractional abundance of isotope to isotope ratio
    return F/(1-F)

class ODEBoxModel(): # three box ocean model with circulation, bio pump, and Sr tracers

    def __init__(self):
        # set up transport matrix for 3 box ocean
        self.ocean_mass = 1.4*10**21 # kg
        self.m0 = 0.4*0.15*self.ocean_mass # kg, high lat surface box
        self.m1 = 0.4*0.85*self.ocean_mass # kg, low lat surface box
        self.m2 = (1-0.04)*self.ocean_mass # kg, deep ocean box
        self.m = np.array([self.m0,self.m1,self.m2]) # mass vector

        self.SvM = np.array([[ 0,20,10],
                            [  0, 0,20],
                            [ 30, 0, 0]]) # Sv matrix

        self.TM4concentrations, self.TM4inventories = self.makeTM()

        # initialize biological pump export
        self.Nsurf = 2
        self.Ninterior  = 1
        self.EM = np.array([[-1,0,0],[0,-1,0],[1,1,0]]) # Export matrix; fraction of export from surface (column) to interior (row)

        self.time = None

        # construct initial state
        # isotopes use concentration (or inventory) * delta or concentration (or inventory) * fractional abundance
        # rows = # of boxes, columns = # of tracers
        self.state0 = np.zeros((3,5))
        self.state0[:,0] = np.array([1e-6,1e-7,2.3e-6])*self.m # dissolved phosphorous, inventory (mol)
        self.state0[:,1] = self.m*(45000e15/12.01)/self.m.sum()  # DIC, inventory (mol)
        self.state0[:,2] = self.m*(120e15)/self.m.sum() # alkalinity, inventory (mol)
        self.state0[:,3] = np.array([0.2, 0.2, 0.2])*self.state0[:,1] # d13C, permil * DIC inventory
        self.state0[:,4] = np.array([0.05, 0.05, 0.05])*self.state0[:,1] # d14C, permil * DIC inventory
        # shape: [3,5]
        # state must be 1-D for integrate function (x,) where x = rows*columns
        self.state0 = self.state0.reshape(15,)


    def makeTM(self):
        # makeTM() returns a NxN matrix defining the fractional mixing system of equations, representing 1 year of ocean circulation
        #
        # Function inputs:
        # 1. m: ocean box mass vector (kg) e.g. [m1, m2, m3] for 3 box ocean
        # 2. SvM: Sverdrup matrix of fluxes (Sv) e.g [[0, f1-0, f2-0], [f0-1, 0, f2-1], [f0-2, f1-2, 0]] where fx-y is flux from box x to box y
        #
        # This function converts SvM to mass (kg) fluxes per timestep (units = yrs) and the mass lost from each box is calculated
        # as the sum of each column (sum along rows). The fraction of each ocean box's mass retained after moving fluxes is given
        # by the diagonal of the transport matrix. Unique transport matrices are needed for concentration and inventory fluxes
        # (TM_ForConcentrations, TM_ForInventories). The difference in the transport matrices is the definition of "fractional fluxes"
        # which describe the transport from one box to another with respect to the size (mass) of the receiving box (for concentration)
        # or with respect to the giving box (for inventory). The new concentration of a given box is equal to the sum of the fractions
        # of contributing boxes multiplied by their respective concentrations (i.e. mixing equation where the new concentration of
        # box0 = fraction of box0 remaining * concentration of box0 + fraction of box0 contributed by box1 * concentration of box1).
        # The new inventory of a given box is equal to the sum of the contributions from all boxes (i.e. the new inventory of box0 =
        # fraction of box0 remaining * box0 inventory + fraction of box1 given to box0 * box1 inventory)

        dt = 1 # timestep (yr)
        flux = self.SvM*(10**9*3.154*10**7)*dt  # kg moved in 1 timestep
        m_lost = np.sum(flux, axis=0) # sum of all mass fluxes out of each box
        fraction_retained = (self.m-m_lost)/self.m # fraction of mass retained in each box

        fractional_fluxes = flux/self.m.reshape((len(self.m),1)) # divide flux array rows by mass for concentration
        fractional_fluxes_inv = flux/self.m.T # divide flux array columns by mass for inventory
        TM_ForConcentrations = fractional_fluxes + np.diag(fraction_retained) # TM_ForConcentrations is NxN matrix defining the fractional mixing system of equations for concentration units, representing 1 year of ocean circulation
        TM_ForInventories = fractional_fluxes_inv + np.diag(fraction_retained) # TM_ForInventories is NxN matrix defining the fractional mixing system of equations for inventory units, representing 1 year of ocean circulation
        return TM_ForConcentrations, TM_ForInventories


    def TestTM(self):
        UnitTracer = np.array([1,1,1])
        print("before 1k steps the UnitTracer is:", UnitTracer)
        print("before 1k steps the UnitTracer inventory is:", (UnitTracer*self.m).sum())

        for year in range(1,1001):
            UnitTracer = self.TM4concentrations@UnitTracer
        print("after 1k steps the UnitTracer is:", UnitTracer)
        print("after 1k steps the UnitTracer inventory is:", (UnitTracer*self.m).sum())


        MassTracer = self.m
        print("before 1k steps the MassTracer is:", MassTracer)
        print("before 1k steps the MassTracer inventory is:", (MassTracer).sum())
        for year in range(1,1001):
            MassTracer = self.TM4inventories@MassTracer

        print("after 1k steps the MassTracer is:", MassTracer)
        print("after 1k steps the MassTracer inventory is:", (MassTracer).sum())


    def TestModel(self):
        TestState = np.copy(self.state0).reshape(3,9)
        dTSdt = self.BoxModel(0,TestState).reshape(3,9)
        print("TestState:", TestState)
        print("dTSdt:", dTSdt)
        # shape: [3,9]


    def ComputeExportP(self, state):
        ExportP = np.zeros(3).T

        P = state.reshape(3,5)[:,0]/self.m[:] # mol/kg P
        SetP = np.array([1e-6,1e-7])
        for s in range(0,self.Nsurf):
            timescale = 20 # year
            if P[s]-SetP[s] >0:
                ExportP[s] = (P[s]-SetP[s])/timescale*self.m[s] # mol surfacePO4/year

            else:
                #print(P[s],SetP[s],P[s]-SetP[s])
                pass # not enough nutrients to sustain productivity
        return self.EM @ ExportP


    def CarbonCycle(self,t,state):
        dCdt = np.zeros((3,3))
        DIC = state.reshape(3,5)[:,1]
        d13C = state.reshape(3,5)[:,3]
        d14C = state.reshape(3,5)[:,4]

        d13C_deep = d13C[2]/DIC[2]
        d14C_deep = d14C[2]/DIC[2]

        # modern (interglacial) fluxes
        F_in = 56e9 # mol/yr
        F_in_LS = (self.m[1]/(self.m[0]+self.m[1]))*F_in # mol/yr
        F_in_HS = (self.m[0]/(self.m[0]+self.m[1]))*F_in # mol/yr
        d13C_in = 0.32 # permil
        d14C_in = RtoF(0.7116)
        epsilon = -0.18 # permil, fractionation between seawater & carbonate
        F_out = 150e9 # mol/yr

        dCdt[:,0] = [F_in_HS, F_in_LS, -F_out]
        dCdt[:,1] = [d13C_in*F_in_HS, (d13C_in*F_in_LS), -(d13C_deep+epsilon)*F_out]
        dCdt[:,2] = [d14C_in*F_in_HS, (d14C_in*F_in_LS), -(d14C_deep)*F_out]

        return dCdt

    def BoxModel(self,t,state):
        # when the length of the state vector length is not equal to the TM matrix width (# of col),
        # .reshape() must be used on the state array and flux arrays so that rows equal TM matrix columns

        EP = self.ComputeExportP(state) # shape: [3,3]
        SR = self.CarbonCycle(t, state) # shape: [3,3]

        d_dt = (self.TM4inventories@state.reshape(3,5))-state.reshape(3,5) # CIRCULATION 3 boxes with 8 tracers
        # print(d_dt)
        d_dt[:,0] += 1 * EP # P export
        d_dt[:,1] += 106 * EP # DIC export

        d_dt[:,2] += SR[:,0] # ALK inventory
        d_dt[:,3] += SR[:,1] # d13C * DIC inventory
        d_dt[:,4] += SR[:,2] # d14C * DIC inventory

        return d_dt



    def RunBoxModel(self,tmax):
        # run box model with ODE solver
        t = np.linspace(0, tmax, 1000) #t0, tmax, nsteps

        self.result = scipy.integrate.solve_ivp(self.BoxModel, [0,tmax], self.state0, method ='RK45', t_eval=t, vectorized = True) # should we allow user to specific nsteps for this function?
        self.time = self.result.t


    def MakePlot(self):
        fig, ax = plt.subplots(5, figsize = (16,20))

        ax[0].plot(self.time, self.result.y[0,:]/self.m[0], label='H surface P')
        ax[1].plot(self.time, self.result.y[1,:]/self.m[0], label='H surface C')
        ax[2].plot(self.time, self.result.y[2,:], label='H surface ALK')
        ax[3].plot(self.time, self.result.y[3,:]/self.m[0], label='H surface d13c')
        ax[4].plot(self.time, self.result.y[4,:]/self.m[0], label='H surface d14c')

        ax[0].plot(self.time, self.result.y[5,:]/self.m[1], linestyle='dotted', label='L surface P')
        ax[1].plot(self.time, self.result.y[6,:]/self.m[1], linestyle='dotted', label='L surface C')
        ax[2].plot(self.time, self.result.y[7,:], linestyle='dotted', label='L surface ALK')
        ax[3].plot(self.time, self.result.y[8,:]/self.m[1], linestyle='dotted', label='L surface d13c')
        ax[4].plot(self.time, self.result.y[9,:]/self.m[1], linestyle='dotted', label='L surface d14c')

        ax[0].plot(self.time, self.result.y[10,:]/self.m[2], linestyle='dashed', label='deep P')
        ax[1].plot(self.time, self.result.y[11,:]/self.m[2], linestyle='dashed', label='deep C')
        ax[2].plot(self.time, self.result.y[12,:], linestyle='dashed', label='deep ALK')
        ax[3].plot(self.time, self.result.y[13,:]/self.m[2], linestyle='dashed', label='deep d13c')
        ax[4].plot(self.time, self.result.y[14,:]/self.m[2], linestyle='dashed', label='deep d14c')

        ax[0].legend(loc = 1)
        ax[1].legend(loc = 1)
        ax[2].legend(loc = 1)
        ax[3].legend(loc = 1)
        ax[4].legend(loc = 1)

        ax[0].set_xlabel('t:years')
        ax[0].set_ylabel('P mol/kg')
        ax[0].set_title('Dissolved P')
        ax[1].set_xlabel('t:years')
        ax[1].set_ylabel('DIC mol/kg')
        ax[1].set_title('DIC')
        ax[2].set_xlabel('t:years')
        ax[2].set_ylabel('ALK (units)')
        ax[2].set_title('ALK')
        ax[3].set_xlabel('t:years')
        ax[3].set_ylabel('d13c (units)')
        ax[3].set_title('d13c')
        ax[4].set_xlabel('t:years')
        ax[4].set_ylabel('d14c (units)')
        ax[4].set_title('d14c')

        plt.tight_layout()
        fig.savefig("../results/SummaryPlot.pdf")

if __name__ == "__main__":

    ModelInstance = ODEBoxModel()
    # ModelInstance.TestTM()
    # ModelInstance.TestModel()
    ModelInstance.RunBoxModel(100000)
    ModelInstance.MakePlot()
