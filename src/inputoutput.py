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

    for i in range(0, 3):
        tracers[i, :] = conversions.moles_to_micromoles_kg(tracers[i, :], mass[i])
    for i in range(3, 6):
        for j in range(0, 3):
            tracers[i, :] = conversions.moles_to_micromoles_kg(tracers[i, :], mass[j])

    ax[0].plot(time, carbonate_chemistry[3, 0, :], label="Baja California pH")
    ax[1].plot(time, tracers[0, :], label="Baja California C")
    ax[2].plot(time, tracers[3, :], label="Baja California ALK")
    ax[3].plot(
        time, tracers[12, :], label="Baja California δ$^{13}$C",
    )
    ax[4].plot(
        time, tracers[15, :], label="Baja California ∆$^{14}$C",
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
        time, tracers[13, :], linestyle="dotted", label="GoC deep δ$^{13}$C",
    )
    ax[4].plot(
        time, tracers[16, :], linestyle="dotted", label="GoC deep ∆$^{14}$C",
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
        time, tracers[14, :], linestyle="dashed", label="GoC surface δ$^{13}$C",
    )
    ax[4].plot(
        time, tracers[17, :], linestyle="dashed", label="GoC surface ∆$^{14}$C",
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


def read_data(txt):
    df = pd.read_table(txt)
    df = organize_data(df)
    return df


def read_bc(file):
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


def organize_data(df):
    """Reading in CYCLOPS code for boundary condition"""
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
