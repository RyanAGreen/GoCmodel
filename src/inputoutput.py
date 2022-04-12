import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.conversions as conversions


def make_text(state, tracers_arr):
    state_reshaped = state.reshape(state.shape[0], state.shape[1], 1)
    tracers_arr = np.concatenate((tracers_arr, state_reshaped), axis=2)
    return tracers_arr


def save_tracers(tracers_arr):
    np.save("../results/tracers.npy", tracers_arr)


def make_plot(time, tracers, carbonate_chemistry, mass):
    """makes all plots"""
    fig, ax = plt.subplots(5, figsize=(16, 20), sharex=True)

    ax[0].plot(time, carbonate_chemistry[3, 0, :], label="Baja California pH")
    ax[1].plot(
        time, tracers[0, :], label="Baja California C",
    )
    # ax[2].plot(
    #     time,
    #     conversions.moles_to_micromoles_kg(tracers[3, :], mass[0]),
    #     label="Baja California ALK",
    # )

    ax[2].plot(
        time, tracers[3, :], label="Baja California ALK",
    )
    ax[3].plot(
        time, tracers[9, :] / tracers[0, :], label="Baja California δ$^{13}$C",
    )
    ax[4].plot(
        time, tracers[12, :] / tracers[0, :], label="Baja California ∆$^{14}$C",
    )

    ax[0].plot(
        time, carbonate_chemistry[3, 1, :], linestyle="dotted", label="GoC deep pH",
    )
    ax[1].plot(
        time, tracers[1, :], linestyle="dotted", label="GoC deep C",
    )
    ax[2].plot(
        time, tracers[4, :], linestyle="dotted", label="GoC deep ALK",
    )
    ax[3].plot(
        time,
        tracers[10, :] / tracers[1, :],
        linestyle="dotted",
        label="GoC deep δ$^{13}$C",
    )
    ax[4].plot(
        time,
        tracers[13, :] / tracers[1, :],
        linestyle="dotted",
        label="GoC deep ∆$^{14}$C",
    )

    ax[0].plot(
        time, carbonate_chemistry[3, 2, :], linestyle="dashed", label="GoC surface pH",
    )
    ax[1].plot(
        time, tracers[2, :], linestyle="dashed", label="GoC surface C",
    )
    ax[2].plot(
        time, tracers[5, :], linestyle="dashed", label="GoC surface ALK",
    )
    ax[3].plot(
        time,
        tracers[11, :] / tracers[2, :],
        linestyle="dashed",
        label="GoC surface δ$^{13}$C",
    )
    ax[4].plot(
        time,
        tracers[14, :] / tracers[2, :],
        linestyle="dashed",
        label="GoC surface ∆$^{14}$C",
    )

    ax[0].legend(loc=1)
    ax[1].legend(loc=1)
    ax[2].legend(loc=1)
    ax[3].legend(loc=1)
    ax[4].legend(loc=1)

    ax[0].set_ylabel("pH")
    ax[0].set_title("pH")
    ax[1].set_ylabel("DIC µmol/kg")
    ax[1].set_title("DIC")
    ax[2].set_ylabel("ALK (µmol/kg)")
    ax[2].set_title("ALK")
    ax[3].set_ylabel("δ$^{13}$C (permil)")
    ax[3].set_title("δ$^{13}$C")
    ax[4].set_xlabel("Years BP")
    ax[4].set_ylabel("∆$^{14}$C (permil)")
    ax[4].set_title("∆$^{14}$C")

    plt.tight_layout()
    try:
        fig.savefig("results/SummaryPlotLessRetained.pdf")
    except:
        fig.savefig("../results/SummaryPlotLessRetained.pdf")


def read_cadd_scenario(file):
    """returns numpy array with rate of carbon addition in
    column 0 and ALK:DIC ration in column 1. rows correspond
    to 100 year intervals (201,2)"""
    df = pd.read_table(str(file), sep="\s+", header=None)
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
    df1 = df[["Crate", "ALKtoDIC"]]

    scenario = df1.to_numpy()
    scenario[:, 0] = scenario[:, 0] * 1e15 / 12 * 1e6
    # Crate is in PgC
    # need to get in umol then divide by box added to (umol/kg)

    return scenario


def read_bc(file, row):

    """at some point, should try to read in as an array to make code run faster"""
    df = pd.read_table(str(file), sep="\s+", header=None)
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
    bc = np.array(
        [
            [df["DICintNP"][row], df["DICsurfNP"][row]],
            [df["ALKintNP"][row], df["ALKsurfNP"][row]],
            [df["NintNP"][row], df["NsurfNP"][row]],
            [
                df["d13CintNP"][row] * df["DICintNP"][row],
                df["d13CsurfNP"][row] * df["DICsurfNP"][row],
            ],
            [
                df["D14CintNP"][row] * df["DICintNP"][row],
                df["D14CsurfNP"][row] * df["DICsurfNP"][row],
            ],
        ]
    )
    return bc
