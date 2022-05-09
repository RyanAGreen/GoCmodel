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


def save_file(time, tracers, carbonate_chemistry):
    df = pd.DataFrame(
        {
            "time": time / 1000,
            "mar_DIC": tracers[0, :],
            "mar_ALK": tracers[3, :],
            "mar_pH": carbonate_chemistry[3, 0, :],
            "mar_d13c": tracers[9, :] / tracers[0, :],
            "mar_D14c": tracers[12, :] / tracers[0, :],
            "goc_sub_DIC": tracers[1, :],
            "goc_sub_ALK": tracers[4, :],
            "goc_sub_pH": carbonate_chemistry[3, 1, :],
            "goc_sub_d13c": tracers[10, :] / tracers[1, :],
            "goc_sub_D14c": tracers[13, :] / tracers[1, :],
            "goc_surf_DIC": tracers[2, :],
            "goc_surf_ALK": tracers[5, :],
            "goc_surf_pH": carbonate_chemistry[3, 2, :],
            "goc_surf_d13c": tracers[11, :] / tracers[2, :],
            "goc_surf_D14c": tracers[14, :] / tracers[2, :],
        }
    )
    np.savetxt(
        r"results/last_simulation_tracers.txt", df.values, fmt="%.2f", delimiter="\t"
    )
    return


def make_plot(time, tracers, carbonate_chemistry, mass):
    """makes all plots"""
    fig, ax = plt.subplots(5, figsize=(16, 20), sharex=True)

    obspath = "data/observations/"

    Rafter_surface = pd.read_csv(obspath + "Rafter_2019.tab", sep="\t", header=24)
    Rafter_surface.loc[(Rafter_surface["Habitat"] == "planktic")]
    Rafter_surface["Cal age [ka BP]"] = 1000 * Rafter_surface["Cal age [ka BP]"]
    Rafter_surface = Rafter_surface.sort_values(by=["Cal age [ka BP]"])

    Rafter_subsurface = pd.read_excel(
        obspath + "prafter-2019-Gulf-CA-Data-for-Ryan.xls"
    )
    Rafter_subsurface = Rafter_subsurface.loc[
        (Rafter_subsurface["species"] == "U. peregrina")
        | (Rafter_subsurface["species"] == "Planulina ariminensis")
        | (Rafter_subsurface["species"] == "U. peregrina ")
    ]
    Rafter_subsurface = Rafter_subsurface.sort_values(by=["calendar age [kyr BP]"])
    Rafter_subsurface = Rafter_subsurface[["calendar age [kyr BP]", "D14C"]]
    Rafter_subsurface = Rafter_subsurface.dropna(subset=["D14C"])
    Rafter_subsurface = (
        Rafter_subsurface.groupby("calendar age [kyr BP]").mean().reset_index()
    )

    Mar = pd.read_csv(obspath + "Marchitto.txt", sep="\s+")
    Mar["Cal.Age"] = 1000 * Mar["Cal.Age"]

    ax[0].plot(time, carbonate_chemistry[3, 0, :], label="Baja California pH")
    ax[1].plot(
        time, tracers[0, :], color="#706513", label="Baja California C",
    )
    # ax[2].plot(
    #     time,
    #     conversions.moles_to_micromoles_kg(tracers[3, :], mass[0]),
    #     label="Baja California ALK",
    # )

    ax[2].plot(
        time, tracers[3, :], color="#706513", label="Baja California ALK",
    )
    ax[3].plot(
        time,
        tracers[9, :] / tracers[0, :],
        color="#706513",
        label="Baja California δ$^{13}$C",
    )
    ax[4].plot(
        time,
        tracers[12, :] / tracers[0, :],
        color="#706513",
        lw=4,
        label="Marchitto box ∆$^{14}$C",
    )

    ax[0].plot(
        time,
        carbonate_chemistry[3, 1, :],
        linestyle="dotted",
        color="#B57114",
        label="GoC deep pH",
    )
    ax[1].plot(
        time, tracers[1, :], linestyle="dotted", color="#B57114", label="GoC deep C",
    )
    ax[2].plot(
        time, tracers[4, :], linestyle="dotted", color="#B57114", label="GoC deep ALK",
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
        linestyle="solid",
        lw=4,
        color="#B57114",
        label="GoC subsurface ∆$^{14}$C",
    )

    ax[0].plot(
        time, carbonate_chemistry[3, 2, :], linestyle="dashed", label="GoC surface pH",
    )
    ax[1].plot(
        time, tracers[2, :], linestyle="dashed", color="#520120", label="GoC surface C",
    )
    ax[2].plot(
        time,
        tracers[5, :],
        linestyle="dashed",
        color="#520120",
        label="GoC surface ALK",
    )
    ax[3].plot(
        time,
        tracers[11, :] / tracers[2, :],
        linestyle="dashed",
        color="#520120",
        label="GoC surface δ$^{13}$C",
    )
    ax[4].plot(
        time,
        tracers[14, :] / tracers[2, :],
        linestyle="solid",
        lw=4,
        color="#520120",
        label="GoC surface ∆$^{14}$C",
    )

    ax[4].plot(
        Rafter_subsurface["calendar age [kyr BP]"] * 1000,
        Rafter_subsurface["D14C"],
        marker="s",
        markeredgecolor="k",
        markerfacecolor="white",
        linestyle="dashed",
        color="#B57114",
        label="Rafter et al. 2019-GoC subsurface",
        markersize=6,
        lw=2,
    )

    ax[4].plot(
        Rafter_surface["Cal age [ka BP]"],
        Rafter_surface["Δ14C [‰]"],
        marker="^",
        markeredgecolor="k",
        markerfacecolor="white",
        linestyle="dashed",
        color="#520120",
        label="Rafter et al. 2019-GoC surface",
        markersize=6,
        lw=2,
    )

    ax[4].plot(
        Mar["Cal.Age"],
        Mar["D14C"],
        marker="o",
        markeredgecolor="k",
        markerfacecolor="white",
        linestyle="dashed",
        color="#706513",
        label="Marchitto et al. 2007",
        markersize=6,
        lw=2,
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
    ax[4].set_ylim(-450, 350)
    ax[4].grid()
    for i in range(5):
        ax[i].set_xlim(0, 20000)

    plt.tight_layout()
    try:
        fig.savefig("results/Plot.pdf")
    except:
        fig.savefig("../results/Plot.pdf")


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
