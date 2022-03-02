"""Gulf of California
Regional Model
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def ratio_to_frac(ratio):
    """# convert isotope ratio to fractional abundance of isotope"""
    return ratio / (1 + ratio)


def frac_to_ratio(fraction):
    """convert fractional abundance of isotope to isotope ratio"""
    return fraction / (1 - fraction)


class GoCModel:
    """three box ocean model with circulation, biological pump,
    and DIC,ALK,P,N,d13C,D14C tracers"""

    def __init__(self):
        self.num_box = 3
        self.num_bc = 2
        self.num_tracer = 6
        self.boxlabel = [
            "Baja California",
            "Gulf of California-Deep",
            "Gulf of California-Surface",
        ]

        self.ocean_mass = 9.48024 * 10 ** 17
        """kg,calculated from GoC volume (1.45e+14 mass_3) + Baja volume (1.45e+14*5 mass_3)
        * density of sw (kg/mass_3)
        """

        self.mass_0 = (
            0.833 * 0.15 * self.ocean_mass
        )  # kg, Baja intermediate depth box -> 5/6 of oceanmass
        self.mass_1 = (
            0.167 * 0.5 * self.ocean_mass
        )  # kg, Gulf of California deep box -> 1/2 of 1/6 of oceanmass
        self.mass_2 = (
            0.167 * 0.33 * self.ocean_mass
        )  # kg, Gulf of California surface box -> 1/3 of 1/6 of oceanmass
        self.mass_3 = 1e200 * self.ocean_mass  # kg, NP intermediate
        self.mass_4 = (
            1e200 * self.ocean_mass
        )  # kg, NP surface (very large box to be essentially infinite)
        self.mass = np.array(
            [self.mass_0, self.mass_1, self.mass_2, self.mass_3, self.mass_4]
        )  # mass vector

        # set up tracers
        self.carbon = (
            np.array([2000e-6, 2000e-6, 2000e-6]) * self.mass_0
        )  # umol kg-1 -> mol
        self.alkalinity = (
            np.array([2200e-6, 2200e-6, 2200e-6]) * self.mass_0
        )  # umol kg-1 -> mol
        self.phosphorus = np.array([2e-6, 2e-6, 2e-6]) * self.mass_0  # mol
        self.nitrate = np.array([30e-6, 30e-6, 30e-6]) * self.mass_0  # mol
        self.del_13_c = np.array([-0.5, -0.5, -0.5])  # *self._carbon # permil
        self.del_14_c = np.array([100, 100, 100])  # *self.carbon # permil

        self.state_v0 = np.hstack(
            (
                self.carbon,
                self.alkalinity,
                self.phosphorus,
                self.nitrate,
                self.del_13_c,
                self.del_14_c,
            )
        )
        self.boundary_condition = np.array(
            [
                [3000e-6 * self.mass_0, 3000e-6 * self.mass_0],
                [3400e-6 * self.mass_0, 3400e-6 * self.mass_0],
                [3e-6 * self.mass_0, 3e-6 * self.mass_0],
                [35e-6 * self.mass_0, 35e-6 * self.mass_0],
                [-0.1, -0.1],
                [200, 200],
            ]
        )  # (tracers,boxes) for boundary condition

        self.svedrup_matrix = self.circ(0.45, 0.005)  # Sv 0.45,0.05
        self.transport_matrix = self.make_transport_matrix(self.svedrup_matrix)

        # Key
        # First Row: self.state0[0,:] - Baja California Box
        # Second Row: self.state0[1,:] - Gulf of California Deep Box
        # Third Row: self.state0[2,:] - Gulf of California Surface Box
        # state_A = self.MakeStateA(self.stateV0)
        # d_dt = (self.TM @ stateA.T).T[:, : self.num_box]

        # initialize biological pump export
        self.num_surf = 1
        self.num_interior = 2
        self.export_matrix = np.array(
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        )  # Export matrix; fraction of export from surface (column) to interior (row)
        self.result = None
        self.time = None
        self.output = None

    def circ(self, advection, mixing):
        """function that takes in circulation (units Sv) and populates a circulation matrix"""

        advect = np.zeros((self.num_box + self.num_bc, self.num_box + self.num_bc))
        advect[1, 0] = advection
        advect[2, 1] = advection
        advect[4, 2] = advection
        advect[0, 3] = advection

        # AD[1,2] = 1

        mix = np.zeros((self.num_box + self.num_bc, self.num_box + self.num_bc))
        mix[1, 0] = mixing
        mix[3, 0] = mixing
        mix[0, 1] = mixing
        mix[2, 1] = mixing
        mix[1, 2] = mixing
        mix[4, 2] = mixing
        mix[0, 3] = mixing
        mix[2, 4] = mixing

        # O = np.zeros((self.num_box+self.num_bc,self.num_box+self.num_bc))
        # O[2,4] = outflow

        # I = np.zeros((self.num_box+self.num_bc,self.num_box+self.num_bc))
        # I[3,0] = inflow

        return advect + mix

    def make_transport_matrix(self, svedrup_matrix):
        """makeTM() returns a NxN matrix defining the fractional mixing system of equations,
        representing 1 year of ocean circulation
        Function inputs:
        1. m: ocean box mass vector (kg) e.g. [mass_1, mass_2, mass_3] for 3 box ocean
        2. SvM: Sverdrup matrix of fluxes (Sv) e.g [[0, f1-0, f2-0], [f0-1, 0, f2-1],
        [f0-2, f1-2, 0]] where fx-y is flux from box x to box y

        This function converts SvM to mass (kg) fluxes per timestep (units = yrs) and the mass lost
        from each box is calculated as the sum of each column (sum along rows). The fraction of each
        ocean box's mass retained after moving fluxes is given by the diagonal of the transport
        matrix. Unique transport matrices are needed for concentration and inventory fluxes
        (TM_ForConcentrations,TM_ForInventories). The difference in the transport matrices is the
        definition of "fractional fluxes" which describe the transport from one box to another with
        respect to the size (mass) of the receiving box (for concentration) or with respect to the
        giving box (for inventory). The new concentration of a given box is equal to the sum of the
        fractions of contributing boxes multiplied by their respective concentrations (i.e. mixing
        equation where the new concentration of box0 = fraction of box0 remaining * concentration
        of box0 + fraction of box0 contributed by box1 * concentration of box1). The new inventory
        of a given box is equal to the sum of the contributions from all boxes (i.e. the new
        inventory of box0 = fraction of box0 remaining * box0 inventory + fraction of box1 given
        to box0 * box1 inventory)
        TM_ForConcentrations is NxN matrix defining the fractional mixing system of equations for
        concentration units, representing 1 year of ocean circulation
        TM_ForInventories is NxN matrix defining the fractional mixing system of equations for
        inventory units, representing 1 year of ocean circulation
        """

        time_step = 1  # timestep (yr)
        flux = (
            svedrup_matrix * (1e6 * 1026 * 3.154e7) * time_step
        )  # conversion from Sv (10e6 mass_3/s) to kg/yr moved in 1 timestep
        mass_lost = np.sum(flux, axis=0)  # sum of all mass fluxes out of each box
        fraction_retained = (
            self.mass - mass_lost
        ) / self.mass  # fraction of mass retained in each box

        # wouldnt this be kg / kg ??
        fractional_fluxes = flux / self.mass.reshape(
            (len(self.mass), 1)
        )  # divide flux array rows by mass for concentration
        # fractional_fluxes_inv = (flux / self.m.T)# divide flux array columns by mass for inventory
        transport_matrix_concentrations = fractional_fluxes + np.diag(fraction_retained)
        # TM_ForInventories = fractional_fluxes_inv + np.diag(fraction_retained)
        return transport_matrix_concentrations - np.identity(
            self.num_box + self.num_bc
        )  # , TM_ForInventories

    def make_state_a(self, state_v):
        """makes new state in matrix format"""
        state_a = np.hstack(
            (state_v.T.reshape(self.num_tracer, self.num_box), self.boundary_condition)
        )
        # tracers for box 3 === stateA[:,3]
        # tracer 2 for all boxes === stateA[2,:]
        return state_a

    def export_phosphorus(self, state):
        """computes phosphorus export"""
        export_phos = np.zeros(3).T

        phos = state.reshape(3, 6)[:, 0] / self.mass[:]  # mol/kg P
        set_phos = np.array([1e-6, 1e-7])
        for surf_boxes in range(0, self.num_surf):
            timescale = 20  # year
            if phos[surf_boxes] - set_phos[surf_boxes] > 0:
                export_phos[surf_boxes] = (
                    (phos[surf_boxes] - set_phos[surf_boxes])
                    / timescale
                    * self.mass[surf_boxes]
                )  # mol surfacePO4/year

            else:
                # print(P[s],SetP[s],P[s]-SetP[s])
                pass  # not enough nutrients to sustain productivity
        return self.export_matrix @ export_phos

    def box_model(self, time, statev):
        # pylint: disable=unused-argument
        """makes new state and calculates the change in state with time"""
        state_a = self.make_state_a(statev)
        d_dt = (self.transport_matrix @ state_a.T).T[:, : self.num_box]
        # d_dt += self.phosphorusrod(stateA)
        # d_dt += self.Fix(stateA)
        return d_dt.flatten()

    def run_box_model(self, tmax):
        """runs the box model with ODE solver"""
        time = np.linspace(0, tmax, 200)  # t0, tmax, nsteps

        self.result = solve_ivp(
            self.box_model,
            [0, tmax],
            self.state_v0,
            method="RK45",
            t_eval=time,
            vectorized=True,
        )  # should we allow user to specific nsteps for this function?
        self.time = np.flipud(self.result.t)  # plot from past to present
        self.output = self.result.y
        print(self.output.shape)

    def make_plot(self):
        """makes all plots"""
        # Disable all the invalid-name violations in this function
        # pylint: disable=invalid-name
        fig, ax = plt.subplots(5, figsize=(16, 20))

        ax[1].plot(
            self.time, self.result.y[0, :] / self.mass[0], label="Baja California C"
        )
        ax[2].plot(
            self.time, self.result.y[1, :] / self.mass[0], label="Baja California ALK"
        )
        ax[0].plot(
            self.time, self.result.y[3, :] / self.mass[0], label="Baja California N"
        )
        ax[3].plot(
            self.time,
            self.result.y[4, :] / self.mass[0],
            label="Baja California δ$^{13}$C",
        )
        ax[4].plot(
            self.time,
            self.result.y[5, :] / self.mass[0],
            label="Baja California ∆$^{14}$C",
        )

        ax[1].plot(
            self.time,
            self.result.y[6, :] / self.mass[1],
            linestyle="dotted",
            label="GoC deep C",
        )
        ax[2].plot(
            self.time,
            self.result.y[7, :] / self.mass[1],
            linestyle="dotted",
            label="GoC deep ALK",
        )
        ax[0].plot(
            self.time,
            self.result.y[9, :] / self.mass[1],
            linestyle="dotted",
            label="GoC deep N",
        )
        ax[3].plot(
            self.time,
            self.result.y[10, :] / self.mass[1],
            linestyle="dotted",
            label="GoC deep δ$^{13}$C",
        )
        ax[4].plot(
            self.time,
            self.result.y[11, :] / self.mass[1],
            linestyle="dotted",
            label="GoC deep ∆$^{14}$C",
        )

        ax[1].plot(
            self.time,
            self.result.y[12, :] / self.mass[2],
            linestyle="dashed",
            label="GoC surface C",
        )
        ax[2].plot(
            self.time,
            self.result.y[13, :] / self.mass[2],
            linestyle="dashed",
            label="GoC surface ALK",
        )
        ax[0].plot(
            self.time,
            self.result.y[15, :] / self.mass[2],
            linestyle="dashed",
            label="GoC surface N",
        )
        ax[3].plot(
            self.time,
            self.result.y[16, :] / self.mass[2],
            linestyle="dashed",
            label="GoC surface δ$^{13}$C",
        )
        ax[4].plot(
            self.time,
            self.result.y[17, :] / self.mass[2],
            linestyle="dashed",
            label="GoC surface ∆$^{14}$C",
        )

        ax[0].legend(loc=1)
        ax[1].legend(loc=1)
        ax[2].legend(loc=1)
        ax[3].legend(loc=1)
        ax[4].legend(loc=1)

        ax[0].set_xlabel("t:years")
        ax[0].set_ylabel("N mol/kg")
        ax[0].set_title("Dissolved NO$_3$$^-$")
        ax[1].set_xlabel("t:years")
        ax[1].set_ylabel("DIC µmol/kg")
        ax[1].set_title("DIC")
        ax[2].set_xlabel("t:years")
        ax[2].set_ylabel("ALK (µmol/kg)")
        ax[2].set_title("ALK")
        ax[3].set_xlabel("t:years")
        ax[3].set_ylabel("δ$^{13}$C (permil)")
        ax[3].set_title("δ$^{13}$C")
        ax[4].set_xlabel("t:years")
        ax[4].set_ylabel("∆$^{14}$C (permil)")
        ax[4].set_title("∆$^{14}$C")

        plt.tight_layout()
        fig.savefig("../results/SummaryPlot.pdf")


if __name__ == "__main__":

    ModelInstance = GoCModel()
    ModelInstance.run_box_model(20000)
    ModelInstance.make_plot()
