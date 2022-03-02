# from IPython import get_ipython
# get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp

def RtoF(R): # conversion of isotope ratios to fractional abundances
    # convert isotope ratio to fractional abundance of isotope
    return R/(1+R)
def FtoR(F): # conversion of isotope fractional abundances to ratios
    # convert fractional abundance of isotope to isotope ratio
    return F/(1-F)



class ODEBoxModel(): # three box ocean model with circulation, bio pump, and Sr tracers

    def __init__(self):
        self.Nbox = 3
        self.Nbc = 2
        self.Ntracer = 6

        self.boxlabel = ["Baja California","Gulf of California-Deep","Gulf of California-Surface"]

        self.ocean_mass = 9.48024*10**17 # kg -> calculated from GoC volume (1.45e+14 m3) + Baja volume (1.45e+14*5 m3) * density of sw (kg/m3)
        self.m0 = 0.833*0.15*self.ocean_mass # kg, Baja intermediate depth box -> 5/6 of oceanmass
        self.m1 = 0.167*0.5*self.ocean_mass # kg, Gulf of California deep box -> 1/2 of 1/6 of oceanmass
        self.m2 = 0.167*0.33*self.ocean_mass # kg, Gulf of California surface box -> 1/3 of 1/6 of oceanmass
        self.m3 = 1e200*self.ocean_mass # kg, NP intermediate
        self.m4 = 1e200*self.ocean_mass # kg, NP surface (very large box to be essentially infinite)
        self.m = np.array([self.m0,self.m1,self.m2,self.m3,self.m4]) # mass vector

        # set up tracers
        self.DIC = np.array([2000e-6,2000e-6,2000e-6])*self.m0 # umol kg-1 -> mol
        self.ALK = np.array([2200e-6,2200e-6,2200e-6])*self.m0 # umol kg-1 -> mol
        self.P = np.array([2e-6,2e-6,2e-6])*self.m0 # mol
        self.N  = np.array([30e-6, 30e-6, 30e-6])*self.m0 # mol
        self.d13C = np.array([-0.5, -0.5, -0.5]) # *self.DIC # permil
        self.D14C = np.array([100, 100, 100]) # *self.DIC # permil

        self.stateV0 = np.hstack((self.DIC,self.ALK,self.P,self.N,self.d13C,self.D14C))
        self.BC = np.array([[3000e-6*self.m0,3000e-6*self.m0],[3400e-6*self.m0,3400e-6*self.m0],[3e-6*self.m0,3e-6*self.m0],[35e-6*self.m0,35e-6*self.m0],[-0.1,-0.1],[200,200]]) # (tracers,boxes) for boundary condition

        self.SvM = self.circ(0.75,0.45,.005,0.0005) # Sv 0.45,0.05
        self.TM = self.makeTM(self.SvM)

        # Key
        # First Row: self.state0[0,:] - Baja California Box
        # Second Row: self.state0[1,:] - Gulf of California Deep Box
        # Third Row: self.state0[2,:] - Gulf of California Surface Box
        stateA = self.MakeStateA(self.stateV0)
        d_dt = (self.TM@stateA.T).T[:,:self.Nbox]

        # initialize biological pump export
        self.Nsurf = 1
        self.Ninterior  = 2
        self.EM = np.array([[1,0,0],[0,0,-1],[0,1,0]]) # Export matrix; fraction of export from surface (column) to interior (row)




    def circ(self,inflow,outflow,advection,mixing):

        AD = np.zeros((self.Nbox+self.Nbc,self.Nbox+self.Nbc))
        AD[0,1] = advection
        AD[1,2] = advection
        AD[2,4] = advection
        AD[3,0] = advection

        # AD[1,2] = 1

        M = np.zeros((self.Nbox+self.Nbc,self.Nbox+self.Nbc))
        M[0,1] = mixing
        M[1,2] = mixing
        M[2,4] = mixing
        M[3,0] = mixing
        M[0,3] = mixing
        M[1,0] = mixing
        M[2,1] = mixing
        M[4,2] = mixing

        O = np.zeros((self.Nbox+self.Nbc,self.Nbox+self.Nbc))
        O[2,4] = outflow

        I = np.zeros((self.Nbox+self.Nbc,self.Nbox+self.Nbc))
        I[3,0] = inflow

        return AD+M+I+O

    def makeTM(self,SvM):
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
        flux = SvM*(1e6*1026*3.154e7)*dt  # conversion from Sv (10e6 m3/s) to kg/yr moved in 1 timestep
        m_lost = np.sum(flux, axis=0) # sum of all mass fluxes out of each box
        fraction_retained = (self.m-m_lost)/self.m # fraction of mass retained in each box

        # wouldnt this be kg / kg ??
        fractional_fluxes = flux/self.m.reshape((len(self.m),1)) # divide flux array rows by mass for concentration
        fractional_fluxes_inv = flux/self.m.T # divide flux array columns by mass for inventory
        TM_ForConcentrations = fractional_fluxes + np.diag(fraction_retained) # TM_ForConcentrations is NxN matrix defining the fractional mixing system of equations for concentration units, representing 1 year of ocean circulation
        TM_ForInventories = fractional_fluxes_inv + np.diag(fraction_retained) # TM_ForInventories is NxN matrix defining the fractional mixing system of equations for inventory units, representing 1 year of ocean circulation
        return TM_ForConcentrations-np.identity(self.Nbox+self.Nbc) #, TM_ForInventories

    def MakeStateA(self,stateV):
        stateA = np.hstack((stateV.T.reshape(self.Ntracer,self.Nbox),self.BC))
        # tracers for box 3 === stateA[:,3]
        # tracer 2 for all boxes === stateA[2,:]
        return stateA

    def ComputeExportP(self, state):
        ExportP = np.zeros(3).T

        P = state.reshape(3,6)[:,0]/self.m[:] # mol/kg P
        SetP = np.array([1e-6,1e-7])
        for s in range(0,self.Nsurf):
            timescale = 20 # year
            if P[s]-SetP[s] >0:
                ExportP[s] = (P[s]-SetP[s])/timescale*self.m[s] # mol surfacePO4/year

            else:
                #print(P[s],SetP[s],P[s]-SetP[s])
                pass # not enough nutrients to sustain productivity
        return self.EM @ ExportP

    def BoxModel(self,t,stateV):
        stateA = self.MakeStateA(stateV)
        d_dt = (self.TM@stateA.T).T[:,:self.Nbox]
        # d_dt += self.Prod(stateA)
        # d_dt += self.Fix(stateA)
        return d_dt.flatten()

    def RunBoxModel(self,tmax):
        # run box model with ODE solver
        t = np.linspace(0,tmax, 200) #t0, tmax, nsteps

        self.result = solve_ivp(self.BoxModel, [0,tmax], self.stateV0, method ='RK45', t_eval=t, vectorized = True) # should we allow user to specific nsteps for this function?
        self.time = np.flipud(self.result.t) # plot from past to present
        self.output = self.result.y
        print(self.output.shape)


    def MakePlot(self):
        fig, ax = plt.subplots(5, figsize = (16,20))

        ax[1].plot(self.time, self.result.y[0,:]/self.m[0], label='Baja California C')
        ax[2].plot(self.time, self.result.y[1,:]/self.m[0], label='Baja California ALK')
        ax[0].plot(self.time, self.result.y[3,:]/self.m[0], label='Baja California N')
        ax[3].plot(self.time, self.result.y[4,:]/self.m[0], label='Baja California δ$^{13}$C')
        ax[4].plot(self.time, self.result.y[5,:]/self.m[0], label='Baja California ∆$^{14}$C')

        ax[1].plot(self.time, self.result.y[6,:]/self.m[1], linestyle='dotted', label='GoC deep C')
        ax[2].plot(self.time, self.result.y[7,:]/self.m[1], linestyle='dotted', label='GoC deep ALK')
        ax[0].plot(self.time, self.result.y[9,:]/self.m[1], linestyle='dotted', label='GoC deep N')
        ax[3].plot(self.time, self.result.y[10,:]/self.m[1], linestyle='dotted', label='GoC deep δ$^{13}$C')
        ax[4].plot(self.time, self.result.y[11,:]/self.m[1], linestyle='dotted', label='GoC deep ∆$^{14}$C')

        ax[1].plot(self.time, self.result.y[12,:]/self.m[2], linestyle='dashed', label='GoC surface C')
        ax[2].plot(self.time, self.result.y[13,:]/self.m[2], linestyle='dashed', label='GoC surface ALK')
        ax[0].plot(self.time, self.result.y[15,:]/self.m[2], linestyle='dashed', label='GoC surface N')
        ax[3].plot(self.time, self.result.y[16,:]/self.m[2], linestyle='dashed', label='GoC surface δ$^{13}$C')
        ax[4].plot(self.time, self.result.y[17,:]/self.m[2], linestyle='dashed', label='GoC surface ∆$^{14}$C')

        ax[0].legend(loc = 1)
        ax[1].legend(loc = 1)
        ax[2].legend(loc = 1)
        ax[3].legend(loc = 1)
        ax[4].legend(loc = 1)

        ax[0].set_xlabel('t:years')
        ax[0].set_ylabel('N mol/kg')
        ax[0].set_title('Dissolved NO$_3$$^-$')
        ax[1].set_xlabel('t:years')
        ax[1].set_ylabel('DIC µmol/kg')
        ax[1].set_title('DIC')
        ax[2].set_xlabel('t:years')
        ax[2].set_ylabel('ALK (µmol/kg)')
        ax[2].set_title('ALK')
        ax[3].set_xlabel('t:years')
        ax[3].set_ylabel('δ$^{13}$C (permil)')
        ax[3].set_title('δ$^{13}$C')
        ax[4].set_xlabel('t:years')
        ax[4].set_ylabel('∆$^{14}$C (permil)')
        ax[4].set_title('∆$^{14}$C')

        plt.tight_layout()
        fig.savefig("../results/SummaryPlot.pdf")

if __name__ == "__main__":

    ModelInstance = ODEBoxModel()
    ModelInstance.RunBoxModel(20000)
    ModelInstance.MakePlot()
