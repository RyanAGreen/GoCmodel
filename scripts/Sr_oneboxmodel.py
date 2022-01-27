import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate

def RtoF(R):
    # convert isotope ratio to fractional abundance of isotope
    return R/(1+R)

def FtoR(F):
    # convert fractional abundance of isotope to isotope ratio
    return F/(1-F)


class SrBoxModel():
    
    def __init__(self):
        # set base fluxes (modern values based on Pearce et al. 2015): [F_in, F_out, d88_in, 87_86_out]
        self.fluxes = np.array([46e9, 174e9, 0.324, 0.71161])
        # set initial state: [Sr inventory, d88Sr*Sr inventory, f87Sr*Sr inventory]
        self.state0 = np.array([120e15, 0.39*120e15, RtoF(0.70918)*120e15])
        
    
    def StrontiumCycle(self,t,state): 
            dSrdt = np.zeros(3)
            F_in = self.fluxes[0] 
            F_out = self.fluxes[1]
            d88_in = self.fluxes[2]
            f87_in = RtoF(self.fluxes[3]) 
            epsilon = -0.18 # permil, fractionation between seawater & carbonate
            d88_sw = state[1]/state[0]  
            f87_sw = state[2]/state[0]
        
            dSrdt[0] = F_in-F_out
            dSrdt[1] = (d88_in*F_in)-((d88_sw+epsilon)*F_out)
            dSrdt[2] = (f87_in*F_in)-(f87_sw*F_out)
            
            return dSrdt

    def BoxModel(self, t, state):  # redundant for now but useful later if adding tracers
        SR = self.StrontiumCycle(t, state)
        d_dt = np.zeros(3)
        d_dt[0] += SR[0] # Sr inventory
        d_dt[1] += SR[1] # d88Sr * Sr inventory
        d_dt[2] += SR[2] # f87Sr * Sr inventory

        return   d_dt 

    def RunBoxModel(self, tmax):
        t = np.linspace(0, tmax, 10) #t0, tmax, nsteps
        self.result = scipy.integrate.solve_ivp(self.BoxModel, [0,tmax], self.state0, method ='RK45', t_eval=t, vectorized = True)
        self.time = self.result.t
        self.result1 = self.result.y
        

    # def Experiment1(self, tmax):
    #     self.fluxes[0] = 174e9 # ~3x modern input (mol/yr)
    #     t = np.linspace(0, tmax, 10) #t0, tmax, nsteps
    #     self.result = scipy.integrate.solve_ivp(self.BoxModel, [0,tmax], self.state0, method ='RK45', t_eval=t, vectorized = True)
    #     self.time = self.result.t
    #     self.result2 = self.result.y
    #
    # def Experiment2(self, tmax):
    #     self.fluxes[0] = 174e9 # ~3x modern input (mol/yr)
    #     self.fluxes[2] = 0.21 # lower d88_in for shelf carbonate weathering
    #     t = np.linspace(0, tmax, 10) #t0, tmax, nsteps
    #     self.result = scipy.integrate.solve_ivp(self.BoxModel, [0,tmax], self.state0, method ='RK45', t_eval=t, vectorized = True)
    #     self.time = self.result.t
    #     self.result3 = self.result.y
        
            
    def MakePlot(self, plotname):
        ocean_volume = 1.4*10**21/1.025 # kg/(kg/L)
        inventory2conc_factor = (1/ocean_volume)*(1e6)
        fig,ax = plt.subplots(2,1)
        ax[0].plot(self.time, self.result1[0,:]*inventory2conc_factor, label= 'modern fluxes \n' r'($\delta^\mathregular{88/86}$Sr=0.32‰)')
        ax[1].plot(self.time, self.result1[1,:]/self.result1[0,:], label='modern fluxes \n' r'($\delta^\mathregular{88/86}$Sr=0.32‰)')
        #ax[2].plot(self.time, FtoR(self.result1[2,:]/self.result1[0,:]), label='modern')
        
        #ax[0].plot(self.time, self.result2[0,:]*inventory2conc_factor, alpha=0.3, color = 'green',  label='increased glacial input \n' r'($\delta^\mathregular{88/86}$Sr=0.31‰)')
        #ax[1].plot(self.time, self.result2[1,:]/self.result2[0,:], alpha=0.3, color = 'green', label='increased glacial input \n' r'($\delta^\mathregular{88/86}$Sr=0.31‰)')
        #ax[2].plot(self.time, FtoR(self.result2[2,:]/self.result2[0,:]), label='increased glacial input')
        
        #ax[0].plot(self.time, self.result3[0,:]*inventory2conc_factor, color = 'green', linestyle = 'dashed' , label='increased glacial input \n' r'($\delta^\mathregular{88/86}$Sr=0.21‰)')
        #ax[1].plot(self.time, self.result3[1,:]/self.result3[0,:], color = 'green', linestyle = 'dashed' , label='increased glacial input \n' r'($\delta^\mathregular{88/86}$Sr=0.21‰)')
        #ax[2].plot(self.time, FtoR(self.result3[2,:]/self.result3[0,:]), label='increased glacial input + light Sr')
        
        ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #ax[2].legend(loc = 1)
           
        ax[0].set_xlabel('years')
        ax[0].set_ylabel('Sr (uM)')
        ax[1].set_xlabel('years')
        ax[1].set_ylabel(r'$\delta^\mathregular{88/86}$Sr (‰)')
        #ax[2].set_xlabel('years')
        #ax[2].set_ylabel('87Sr_86Sr')
        
        plt.tight_layout()
        fig.savefig(plotname)
        
        
if __name__ == "__main__":
    ModelInstance = SrBoxModel()
    ModelInstance.RunBoxModel(100000)
#    ModelInstance.Experiment1(100000)
#    ModelInstance.Experiment2(100000)
    ModelInstance.MakePlot("../results/CombinedPlot1.pdf")
        
