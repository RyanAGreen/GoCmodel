# this is a model of N isotope processing within the upper Atlantic ocean
#from IPython import get_ipython
#get_ipython().magic('reset -sf')

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

def RtoF(R):
    # convert isotope ratios to fractions
    return R/(1+R)

def FtoR(F):
    # convert fractions to ratios
    return F/(1-F)



class ANIM():

    def __init__(self):
        self.Nbox = 2
        self.Nbc = 4
        self.Ntracer = 4

        self.epsi_assim = 5
        self.delta_fix = -1

        self.boxlabel = ["surface","deep"]
        self.m = 0.15*1.4e21*np.array( [0.05*0.4,1e200]) # kg
        self.DIC =np.ones(self.Nbox)
        self.ALK =np.ones(self.Nbox)/16
        self.d13C = 5*self.DIC
        self.D14C = 5*self.DIC
        self.stateV0 = np.hstack((self.DIC,self.ALK,self.d13C,self.D14C))
        self.BC = np.array([[1,20,32],[0.1,1.5,2.2],[10*1,8*20,5*32]])

        #self.SvM_nadw = self.NADW(2,0,10,2,2,5) # 10 boxes and 3 BC
        self.SvM_nadw = self.NADW_DMS(5,5,5, 1) # 10 boxes and 3 BC
        self.TM_nadw = self.makeTM(self.SvM_nadw)
        self.TM = self.TM_nadw
        self.EPM = self.EP_TM_DMS()

        stateA = self.MakeStateA(self.stateV0)
        d_dt = (self.TM@stateA.T).T[:,:self.Nbox]


    def NADW(self, u,l,e,sm,dm,mm):

        U = np.zeros((self.Nbox-5+self.Nbc,self.Nbox-5+self.Nbc))
        U[0,10] = u
        U[1,0] = u
        U[2,1] = u
        U[3,2] = u
        U[4,3] = u
        U[8,4] = u
        U[9,8] = u
        U[12,9] = u

        L = np.zeros((self.Nbox-5+self.Nbc,self.Nbox-5+self.Nbc))
        L[5,11] = l
        L[6,5] = l
        L[7,6] = l
        L[8,7] = l
        L[9,8] = l
        L[12,9] = l

        E = np.zeros((self.Nbox-5+self.Nbc,self.Nbox-5+self.Nbc))
        E[5,11] = e
        E[6,5] = e
        E[1,6] = e
        E[2,1] = e
        E[3,2] = e
        E[4,3] = e
        E[8,4] = e
        E[9,8] = e
        E[12,9] = e

        SM = np.zeros((self.Nbox-5+self.Nbc,self.Nbox-5+self.Nbc))
        SM[5,0] = sm
        SM[0,5] = sm
        SM[6,1] = sm
        SM[1,6] = sm
        SM[7,3] = sm
        SM[3,7] = sm
        SM[8,4] = sm
        SM[4,8] = sm

        DM = np.zeros((self.Nbox-5+self.Nbc,self.Nbox-5+self.Nbc))
        DM[5,9] = dm
        DM[9,5] = dm
        DM[6,9] = dm
        DM[9,6] = dm
        DM[7,9] = dm
        DM[9,7] = dm
        DM[9,8] = dm
        DM[8,9] = dm

        MM = np.zeros((self.Nbox-5+self.Nbc,self.Nbox-5+self.Nbc))
        MM[5,6] = mm
        MM[6,5] = mm
        MM[7,6] = mm
        MM[6,7] = mm
        MM[7,8] = mm
        MM[8,7] = mm


        InteriorCirculation = U+L+E+SM+DM+MM
        CompleteCirculation = np.pad(InteriorCirculation, [(5, 0), (5, 0)], mode='constant')

        SMLM = [2,5,0.5,2,10]
        for SML in range(0,5):
            CompleteCirculation[SML,SML+5] = SMLM[SML]
            CompleteCirculation[SML+5,SML] = SMLM[SML]

        return CompleteCirculation

    def NADW_DMS(self,eq, nag, re, m):

        EQ = np.zeros((self.Nbox+self.Nbc,self.Nbox+self.Nbc))
        EQ[10,16] = eq
        EQ[11,10] = eq
        EQ[6,11] = eq
        EQ[1,6] = eq
        EQ[2,1] = eq
        EQ[3,2] = eq
        EQ[4,3] = eq
        EQ[9,4] = eq
        EQ[13,9] = eq
        EQ[14,13] = eq
        EQ[17,14] = eq

        NAG = np.zeros((self.Nbox+self.Nbc,self.Nbox+self.Nbc))
        NAG[10,16] = nag
        NAG[11,10] = nag
        NAG[12,11] = nag
        NAG[8,12] = nag
        NAG[3,8] = nag
        NAG[4,3] = nag
        NAG[9,4] = nag
        NAG[13,9] = nag
        NAG[14,13] = nag
        NAG[17,14] = nag

        RE = np.zeros((self.Nbox+self.Nbc,self.Nbox+self.Nbc))
        RE[10,16] = re
        RE[11,10] = re
        RE[12,11] = re
        RE[13,12] = re
        RE[14,13] = re
        RE[17,14] = re

        M = np.zeros((self.Nbox+self.Nbc,self.Nbox+self.Nbc))
        M[11,10] = M[10,11] = 2*m
        M[12,11] = M[11,12] = 2*m
        M[13,12] = M[12,13] = 2*m
        M[0,5] = M[5,0] = 2*m
        M[1,6] = M[6,1] = 2*m
        M[2,7] = M[7,2] = 2*m
        M[3,8] = M[8,3] = 2*m
        M[4,9] = M[9,4] = 5*m
        M[10,5] = M[5,10] = 2*m
        M[11,6] = M[6,11] = 2*m
        M[12,7] = M[7,12] = 2*m
        M[12,8] = M[8,12] = 2*m
        M[13,9] = M[9,13] = 5*m


        CompleteCirculation = EQ+NAG+RE+M
        #print(CompleteCirculation)
        return CompleteCirculation


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
            flux = SvM*(10**9*3.154*10**7)*dt  # kg moved in 1 timestep
            m_lost = np.sum(flux, axis=0) # sum of all mass fluxes out of each box
            fraction_retained = (self.m-m_lost)/self.m # fraction of mass retained in each box

            fractional_fluxes = flux/self.m.reshape((self.Nbox+self.Nbc,1)) # divide flux array rows by mass for concentration
            fractional_fluxes_inv = flux/self.m.T # divide flux array columns by mass for inventory
            TM_ForConcentrations = fractional_fluxes + np.diag(fraction_retained) # TM_ForConcentrations is NxN matrix defining the fractional mixing system of equations for concentration units, representing 1 year of ocean circulation
            TM_ForInventories = fractional_fluxes_inv + np.diag(fraction_retained) # TM_ForInventories is NxN matrix defining the fractional mixing system of equations for inventory units, representing 1 year of ocean circulation
            return TM_ForConcentrations-np.identity(self.Nbox+self.Nbc)#, TM_ForInventories

    def EP_TM(self):
        EPM = np.zeros((self.Nbox,5)) # Rows are counting each of the boxes in order, columns are the surface boxes
        trgt_shallow = [5,6,7,8,9] # target boxes that are shallow (second row)
        trgt_md = [10,11,12,12,13] # target boxes that are mid-depth (third row)
        for s in range(0,5):
            EPM[s,s] = -1/self.m[s]
            EPM[trgt_shallow[s],s] = 0.9/self.m[trgt_shallow[s]] # getting the one below
            EPM[trgt_md[s],s] = 0.84*0.1/self.m[trgt_md[s]] # getting the one below
            EPM[14,s] = 0.16/self.m[14]

        return EPM

    def EP_TM_DMS(self):
        EPM = np.zeros((self.Nbox,5))
        trgt_shallow = [5,6,7,8,9]
        trgt_md = [10,11,12,12,13]
        for s in range(0,5):
            EPM[s,s] = -1/self.m[s]
            EPM[trgt_shallow[s],s] = 0.45/self.m[trgt_shallow[s]]
            EPM[trgt_md[s],s] = 0.45/self.m[trgt_md[s]]
            EPM[14,s] = 0.10/self.m[14]

        return EPM

    def MakeStateA(self,stateV):
        #print(stateV.shape, self.BC.shape, stateV.T.reshape(self.Ntracer,self.Nbox).shape)
        stateA = np.hstack((stateV.T.reshape(self.Ntracer,self.Nbox),self.BC))
        # tracers for box 3 === stateA[:,3]
        # tracer 2 for all boxes === stateA[2,:]
        return stateA

    def Prod(self,stateA):
        NPP = 4*stateA[0,0:5]*self.m[0:5] # 1/yr * µmol/kg * kg = µmol/yr
        d_dt = np.zeros((self.Ntracer,self.Nbox))
        d_dt[0,:] = self.EPM@NPP
        d_dt[1,:] = self.EPM@NPP/16
        d_dt[2,:] = self.EPM@ (NPP*(stateA[2,0:5]/stateA[0,0:5]-self.epsi_assim))
        return d_dt , NPP*1e-6*1e-12*14, (stateA[2,0:5]/stateA[0,0:5]-self.epsi_assim)

    def Fix(self,stateA):
        Ndef = 16*stateA[1,0:5]-stateA[0,0:5]
        FIX = np.zeros(5)
        for s in [0,2,3,4]:
            if Ndef[s]/stateA[0,0:5][s]>1:
                FIX[s] = 1 * Ndef[s]
            else:
                FIX[s] = 1 * Ndef[s] * np.exp(Ndef[s]/stateA[0,0:5][s])

        d_dt = np.zeros((self.Ntracer,self.Nbox))
        d_dt[0,0:5] = FIX
        d_dt[2,0:5] = FIX * self.delta_fix
        return d_dt,FIX*self.m[0:5]*1e-6*1e-12*14



    def BoxModel(self,t,stateV):
        stateA = self.MakeStateA(stateV)
        d_dt = (self.TM@stateA.T).T[:,:self.Nbox]
        d_dt += self.Prod(stateA)[0]
        d_dt += self.Fix(stateA)[0]
        return d_dt.flatten()

    def RunBoxModel(self,tmax):
            # run box model with ODE solver
            t = np.linspace(0, tmax, 100) #t0, tmax, nsteps

            self.result = solve_ivp(self.BoxModel, [0,tmax], self.stateV0, method ='RK45', t_eval=t, vectorized = True) # should we allow user to specific nsteps for this function?
            self.time = self.result.t
            self.output = self.result.y
            self.PostCompute()
            print(self.output.shape)

    def PostCompute(self):
        stateA = self.MakeStateA(self.output[:,-1])
        self.output_npp,self.output_d15N  = self.Prod(stateA)[1:]
        self.output_fix = self.Fix(stateA)[1]
        print(self.output_npp)
        print(self.output_d15N)
        print(self.output_fix)


    def Plot1T(self, idT, idB):
        for idb in idB:
            plt.plot(self.time,self.output[idT*self.Nbox+idb,:],label="T{} B{}".format(idT,idb))
        plt.legend()
        plt.xlabel("time")
        plt.savefig("Plot1T.pdf")

    def PlotXT(self, idT, idB):
        fig, axs = plt.subplots(len(idT), 1)
        for idt in idT:
            for idb in idB:
                axs[idt].plot(self.time,self.output[idt*self.Nbox+idb,:],label="T{} B{}".format(idt,idb))
            axs[idt].legend()
            axs[idt].grid(True,linestyle="--")

        axs[-1].set_xlabel("time")
        plt.savefig("PlotXT.pdf")

    def PlotXTx(self, idT, idB):
        fig, axs = plt.subplots(len(idT)+2, 1)
        fig.set_size_inches(6, 8)
        for idt in idT:
            for idb in idB:
                axs[idt].plot(self.time,self.output[idt*self.Nbox+idb,:],label=self.boxlabel[idb])
            axs[idt].legend()
            axs[idt].grid(True,linestyle="--")
        for idb in idB:
            axs[-2].plot(self.time, 16*self.output[self.Nbox+idb,:]-self.output[idb,:], label=self.boxlabel[idb])
            axs[-2].legend()
            axs[-2].grid(True,linestyle="--")
        for idb in idB:
            axs[-1].plot(self.time, self.output[2*self.Nbox+idb,:]/self.output[idb,:], label=self.boxlabel[idb])
            axs[-1].legend()
            axs[-1].grid(True,linestyle="--")



        #axs[-1].plot(self.time,self.output[idB,:]-16*self.output[self.Nbox+idB,:],label="T{} B{}".format(idt,idb))

        axs[0].set_ylabel("N [µmol/kg]")
        axs[1].set_ylabel("P [µmol/kg]")
        axs[2].set_ylabel("N deficit [µmol/kg]")
        axs[3].set_ylabel("d15N [o/oo]")
        axs[-1].set_xlabel("time")
        plt.savefig("PlotXTx.pdf")

    def PlotOneField(self):

        import pylab as pl
        stateA = self.MakeStateA(self.output[:,-1])
        c_vals = stateA[2,:]/stateA[0,:]
        normal = pl.Normalize(-1, 15)
        colors = pl.cm.jet(normal(c_vals))
        import matplotlib.patches as patches
        import matplotlib.colorbar as cbar

        fig, ax = plt.subplots(1)
        fig.set_size_inches(8, 4.5)

        boxes = []
        boxes.append(patches.Rectangle((0, 4), 3, 0.5, linewidth=1, edgecolor='k', facecolor=colors[0]))
        boxes.append(patches.Rectangle((3, 4), 1, 0.5, linewidth=1, edgecolor='k', facecolor=colors[1]))
        boxes.append(patches.Rectangle((4, 4), 1, 0.5, linewidth=1, edgecolor='k', facecolor=colors[2]))
        boxes.append(patches.Rectangle((5, 4), 2, 0.5, linewidth=1, edgecolor='k', facecolor=colors[3]))
        boxes.append(patches.Rectangle((7, 4), 1, 0.5, linewidth=1, edgecolor='k', facecolor=colors[4]))

        boxes.append(patches.Rectangle((0, 3), 3, 1, linewidth=1, edgecolor='k', facecolor=colors[5]))
        boxes.append(patches.Rectangle((3, 3), 1, 1, linewidth=1, edgecolor='k', facecolor=colors[6]))
        boxes.append(patches.Rectangle((4, 3), 1, 1, linewidth=1, edgecolor='k', facecolor=colors[7]))
        boxes.append(patches.Rectangle((5, 3), 2, 1, linewidth=1, edgecolor='k', facecolor=colors[8]))
        boxes.append(patches.Rectangle((7, 3), 1, 1, linewidth=1, edgecolor='k', facecolor=colors[9]))
        boxes.append(patches.Rectangle((0, 1), 3, 2, linewidth=1, edgecolor='k', facecolor=colors[10]))
        boxes.append(patches.Rectangle((3, 1), 1, 2, linewidth=1, edgecolor='k', facecolor=colors[11]))
        boxes.append(patches.Rectangle((4, 1), 3, 2, linewidth=1, edgecolor='k', facecolor=colors[12]))
        boxes.append(patches.Rectangle((7, 1), 1, 2, linewidth=1, edgecolor='k', facecolor=colors[13]))
        boxes.append(patches.Rectangle((0, 0), 8, 1, linewidth=1, edgecolor='k', facecolor=colors[14]))

        boxes.append(patches.Rectangle((-1.5, 3), 1, 1, linewidth=1, edgecolor='k', facecolor=colors[15]))
        boxes.append(patches.Rectangle((-1.5, 1), 1, 2, linewidth=1, edgecolor='k', facecolor=colors[16]))
        boxes.append(patches.Rectangle((-1.5, 0), 1, 1, linewidth=1, edgecolor='k', facecolor=colors[17]))

        ax.set_xlim([-1.5,8])
        ax.set_ylim([0,4.5])
        cax, _ = cbar.make_axes(ax)
        cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.jet,norm=normal)

        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        for eachbox in boxes:
            ax.add_patch(eachbox)
        fig.savefig("PlotOneField_wML.pdf")

    def PlotAllField(self):

        import pylab as pl
        import matplotlib.patches as patches
        import matplotlib.colorbar as cbar
        import matplotlib.colors as mplc
        stateA = self.MakeStateA(self.output[:,-1])

        fig, axs = plt.subplots(3)
        fig.set_size_inches(8, 12)

        for panel in [0,1,2]:
            ax = axs[panel]
            if panel==0:
                c_vals = stateA[0,:]
                t_vals = self.BC[0,:]
                normal = mplc.Normalize(0.1, 36)
                text = self.output_npp
                label = "N [µmol/kg]"
            elif panel==1:
                c_vals = 16*stateA[1,:]-stateA[0,:]
                t_vals = 16*self.BC[1,:] -self.BC[0,:]
                normal = mplc.Normalize(0, 4)
                text = self.output_fix
                label = "N deficit [µmol/kg]"
            elif panel==2:
                c_vals = stateA[2,:]/stateA[0,:]
                t_vals = self.BC[2,:]/self.BC[0,:]
                normal = mplc.Normalize(c_vals.min(),c_vals.max())
                text = self.output_d15N
                label = "d15N [o/oo]"
            colors = pl.cm.jet(normal(c_vals))


            boxes = []
            boxes.append(patches.Rectangle((0, 4), 3, 0.5, linewidth=1, edgecolor='k', facecolor=colors[0]))
            boxes.append(patches.Rectangle((3, 4), 1, 0.5, linewidth=1, edgecolor='k', facecolor=colors[1]))
            boxes.append(patches.Rectangle((4, 4), 1, 0.5, linewidth=1, edgecolor='k', facecolor=colors[2]))
            boxes.append(patches.Rectangle((5, 4), 2, 0.5, linewidth=1, edgecolor='k', facecolor=colors[3]))
            boxes.append(patches.Rectangle((7, 4), 1, 0.5, linewidth=1, edgecolor='k', facecolor=colors[4]))

            boxes.append(patches.Rectangle((0, 3), 3, 1, linewidth=1, edgecolor='k', facecolor=colors[5]))
            boxes.append(patches.Rectangle((3, 3), 1, 1, linewidth=1, edgecolor='k', facecolor=colors[6]))
            boxes.append(patches.Rectangle((4, 3), 1, 1, linewidth=1, edgecolor='k', facecolor=colors[7]))
            boxes.append(patches.Rectangle((5, 3), 2, 1, linewidth=1, edgecolor='k', facecolor=colors[8]))
            boxes.append(patches.Rectangle((7, 3), 1, 1, linewidth=1, edgecolor='k', facecolor=colors[9]))
            boxes.append(patches.Rectangle((0, 1), 3, 2, linewidth=1, edgecolor='k', facecolor=colors[10]))
            boxes.append(patches.Rectangle((3, 1), 1, 2, linewidth=1, edgecolor='k', facecolor=colors[11]))
            boxes.append(patches.Rectangle((4, 1), 3, 2, linewidth=1, edgecolor='k', facecolor=colors[12]))
            boxes.append(patches.Rectangle((7, 1), 1, 2, linewidth=1, edgecolor='k', facecolor=colors[13]))
            boxes.append(patches.Rectangle((0, 0), 8, 1, linewidth=1, edgecolor='k', facecolor=colors[14]))

            boxes.append(patches.Rectangle((-1.5, 3), 1, 1, linewidth=1, edgecolor='k', facecolor=colors[15]))
            boxes.append(patches.Rectangle((-1.5, 1), 1, 2, linewidth=1, edgecolor='k', facecolor=colors[16]))
            boxes.append(patches.Rectangle((-1.5, 0), 1, 1, linewidth=1, edgecolor='k', facecolor=colors[17]))

            ax.set_xlim([-1.5,8])
            ax.set_ylim([0,4.5])
            cax, _ = cbar.make_axes(ax)
            cb2 = cbar.ColorbarBase(cax, cmap=pl.cm.jet,norm=normal,label=label)

            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            for eachbox in boxes:
                ax.add_patch(eachbox)

            ax.text(0.1, 4.1,"{:.1f}".format(text[0]),color="white")
            ax.text(3.1, 4.1,"{:.1f}".format(text[1]),color="white")
            ax.text(4.1, 4.1,"{:.1f}".format(text[2]),color="white")
            ax.text(5.1, 4.1,"{:.1f}".format(text[3]),color="white")
            ax.text(7.1, 4.1,"{:.1f}".format(text[4]),color="white")

            ax.text(0.1, 3.1,"{:.1f}".format(c_vals[5]))
            ax.text(3.1, 3.1,"{:.1f}".format(c_vals[6]))
            ax.text(4.1, 3.1,"{:.1f}".format(c_vals[7]))
            ax.text(5.1, 3.1,"{:.1f}".format(c_vals[8]))
            ax.text(7.1, 3.1,"{:.1f}".format(c_vals[9]))

            ax.text(0.1, 1.1,"{:.1f}".format(c_vals[10]))
            ax.text(3.1, 1.1,"{:.1f}".format(c_vals[11]))
            ax.text(4.1, 1.1,"{:.1f}".format(c_vals[12]))
            ax.text(7.1, 1.1,"{:.1f}".format(c_vals[13]))
            ax.text(0.1, 0.1,"{:.1f}".format(c_vals[14]))

            ax.text(-1.4, 3.1,"{:.1f}".format(t_vals[0]))
            ax.text(-1.4, 1.1,"{:.1f}".format(t_vals[1]))
            ax.text(-1.4, 0.1,"{:.1f}".format(t_vals[2]))


        fig.savefig("PlotField_wML.pdf")


ModelInstance = ANIM()
ModelInstance.RunBoxModel(3000)
ModelInstance.PlotXTx([0,1],range(0,5))
ModelInstance.PlotXTx([0,1],range(0,5))
ModelInstance.PlotOneField()
ModelInstance.PlotAllField()
