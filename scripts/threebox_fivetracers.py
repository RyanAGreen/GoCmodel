"""Gulf of California
Regional Model
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PyCO2SYS as pyco2
from scipy.integrate import solve_ivp
from conversions import svedrup_to_kg_year


class GoCModel:
    """three box ocean model with circulation, biological pump,
    and DIC,ALK,P,N,d13C,D14C tracers. Box order is Baja California,
    Gulf of California-Deep, Gulf of California-Surface, North Pacific-
    Intermediate depth and North Pacific Surface"""

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
            5 / 6 * self.ocean_mass
        )  # kg, Baja intermediate depth box -> 5/6 of oceanmass
        self.mass_1 = (
            2 / 3 * 1 / 6 * self.ocean_mass
        )  # kg, Gulf of California deep box -> 2/3 of 1/6 of oceanmass
        self.mass_2 = (
            1 / 3 * 1 / 6 * self.ocean_mass
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

        self.svedrup_matrix = self.circ(0.45, 0.1, 0.1, 0.1, 0.1)  # Sv 0.45,0.05
        self.transport_matrix = self.make_transport_matrix(self.svedrup_matrix)

        # initialize biological pump export
        self.num_surf = 1
        self.num_interior = 2
        self.export_matrix = np.array(
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        )  # Export matrix; fraction of export from surface (column) to interior (row)
        self.result = None
        self.time = None
        self.output = None
        self.tracers_arr = np.zeros((6, 5, 1))

        self.cols = [
            "year",
            "ALKintNP",
            "ALKsurfNP",
            "DICintNP",
            "DICsurfNP",
            "NintNP",
            "NsurfNP",
            "D14CintNP",
            "D14CsurfNP",
            "d13CintNP",
            "d13CsurfNP",
        ]

    def circ(self, advection, mix_np_baja, mix_baja_gulf, mix_gulf_gulf, mix_gulf_np):
        """function that takes in circulation (units Sv) and populates a circulation matrix"""

        advect = np.zeros((self.num_box + self.num_bc, self.num_box + self.num_bc))
        advect[1, 0] = advection
        advect[2, 1] = advection
        advect[4, 2] = advection
        advect[0, 3] = advection

        mix_n_b = np.zeros((self.num_box + self.num_bc, self.num_box + self.num_bc))
        mix_n_b[0, 3] = mix_np_baja
        mix_n_b[3, 0] = mix_np_baja

        mix_b_g = np.zeros((self.num_box + self.num_bc, self.num_box + self.num_bc))
        mix_b_g[1, 0] = mix_baja_gulf
        mix_b_g[0, 1] = mix_baja_gulf

        mix_g_g = np.zeros((self.num_box + self.num_bc, self.num_box + self.num_bc))
        mix_g_g[2, 1] = mix_gulf_gulf
        mix_g_g[1, 2] = mix_gulf_gulf

        mix_g_n = np.zeros((self.num_box + self.num_bc, self.num_box + self.num_bc))
        mix_g_n[4, 2] = mix_gulf_np
        mix_g_n[2, 4] = mix_gulf_np

        print("Advect: " + str(advection))
        print("Mix between NP-I and Baja: " + str(mix_n_b))
        print("Mix between Baja and Gulf: " + str(mix_b_g))
        print("Mix between Gulf-D and Gulf-S: " + str(mix_g_g))
        print("Mix between Gulf-S and NP-S: " + str(mix_g_n))

        return advect + mix_n_b + mix_b_g + mix_g_g + mix_g_n

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
        flux = svedrup_to_kg_year(svedrup_matrix) * time_step
        mass_lost = np.sum(flux, axis=0)  # sum of all mass fluxes out of each box

        # fraction of mass retained in each box
        fraction_retained = 0.1 * (self.mass - mass_lost) / self.mass
        print("Fraction Retained: " + str(fraction_retained))
        # wouldnt this be kg / kg ??
        # divide flux array rows by mass for concentration
        fractional_fluxes = flux / self.mass.reshape((len(self.mass), 1))
        print("Fractional Fluxes: " + str(fractional_fluxes))
        # fractional_fluxes_inv = (flux / self.m.T)# divide flux array columns by mass for inventory
        transport_matrix_concentrations = fractional_fluxes + np.diag(fraction_retained)
        # TM_ForInventories = fractional_fluxes_inv + np.diag(fraction_retained)
        # print("Transport Matrix: " + str(transport_matrix_concentrations))
        # print("Returned: " + str(transport_matrix_concentrations - np.identity(self.num_box + self.num_bc)))
        print(
            "Transport Matrix: "
            + str(
                transport_matrix_concentrations
                - np.identity(self.num_box + self.num_bc)
            )
        )
        return transport_matrix_concentrations - np.identity(self.num_box + self.num_bc)

    def make_state_a(self, state_v):
        """makes new state in matrix format. Boxes are in columns and tracers are in
        rows.
        example:
        all tracers for box 3 === stateA[:,3]
        tracer 2 for all boxes === stateA[2,:]
        """
        state_a = np.hstack(
            (state_v.T.reshape(self.num_tracer, self.num_box), self.boundary_condition)
        )
        return state_a

    def export_phosphorus(self, state):
        """computes phosphorus export"""
        export_phos = np.zeros(3).T
        state = state[:, :-2]
        phos = state.reshape(3, 6)[:, 2] / self.mass[:]  # mol/kg P
        set_phos = np.array([1e-6, 1e-7])
        for surf_boxes in range(0, self.num_surf):
            timescale = 20  # year
            diff = phos[surf_boxes] - set_phos[surf_boxes]
            if diff > 0:
                export_phos[surf_boxes] = (
                    diff / timescale * self.mass[surf_boxes]
                )  # mol surfacePO4/year

            else:
                # print(P[s],SetP[s],P[s]-SetP[s])
                pass  # not enough nutrients to sustain productivity
        print(self.export_matrix)
        print(export_phos)
        return self.export_matrix @ export_phos

    def box_model(self, time, statev):
        # pylint: disable=unused-argument
        """makes new state and calculates the change in state with time"""
        state_a = self.make_state_a(statev)
        self.make_text(state_a)
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

    def make_text(self, state):
        state = state.reshape(state.shape[0], state.shape[1], 1)
        self.tracers_arr = np.concatenate((self.tracers_arr, state), axis=2)

    def save(self):
        np.save("../results/tracers.npy", self.tracers_arr)

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
        fig.savefig("../results/SummaryPlotLessRetained.pdf")

    def read_data(txt):
        df = pd.read_table(txt)
        df = organize_data(df)
        return df

    def organize_data(df):
        df = df.rename(
            columns={
                0: "year",
                1: "Crate",
                2: "ALKtoDIC",
                3: "Ccum",
                4: "CO2",
                5: "D14C",
                6: "D14Cerror",
                7: "CO2error",
                8: "totalerror",
                9: "DICintNP",
                10: "ALKintNP",
                11: "d13CintNP",
                12: "D14CintNP",
                13: "NintNP",
                14: "DICsurfNP",
                15: "ALKsurfNP",
                16: "d13CsurfNP",
                17: "D14CsurfNP",
                18: "NsurfNP",
                19: "AtlCSH",
                20: "IndCSH",
                21: "SPacCSH",
                22: "NPacCSH",
            }
        )
        return df[cols]


if __name__ == "__main__":

    ModelInstance = GoCModel()
    ModelInstance.run_box_model(20000)
    ModelInstance.save()
    ModelInstance.make_plot()
