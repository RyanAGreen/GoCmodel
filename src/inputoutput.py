import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.conversions as conversions
from scipy.interpolate import interp1d


def make_text(state, tracers_arr):
    state_reshaped = state.reshape(state.shape[0], state.shape[1], 1)
    tracers_arr = np.concatenate((tracers_arr, state_reshaped), axis=2)
    return tracers_arr


def save_tracers(tracers_arr):
    np.save("../results/tracers.npy", tracers_arr)


def save_file(time, tracers, filename):
    df = pd.DataFrame(
        {
            "time": time / 1000,
            "mar_DIC": tracers[0, :],
            "goc_sub_DIC": tracers[1, :],
            "goc_surf_DIC": tracers[2, :],
            "mar_ALK": tracers[3, :],
            "goc_sub_ALK": tracers[4, :],
            "goc_surf_ALK": tracers[5, :],
            # "mar_pH": carbonate_chemistry[3, 0, :],
            "mar_d13c": tracers[9, :] / tracers[0, :],
            "goc_sub_d13c": tracers[10, :] / tracers[1, :],
            "goc_surf_d13c": tracers[11, :] / tracers[2, :],
            "mar_D14c": tracers[12, :] / tracers[0, :],
            "goc_sub_D14c": tracers[13, :] / tracers[1, :],
            "goc_surf_D14c": tracers[14, :] / tracers[2, :],
            # "goc_sub_pH": carbonate_chemistry[3, 1, :],
            # "goc_surf_pH": carbonate_chemistry[3, 2, :],
            "mar_cum_carbon": tracers[15, :],
            "goc_sub_cum_carbon": tracers[16, :],
            "goc_surf_cum_carbon": tracers[17, :],
            # "mar_carbon_rate": marchitto_rate_array[:],
            # "goc_sub_carbon_rate": subsurface_rate_array[:],
            # "goc_surf_carbon_rate": surface_rate_array,
        }
    )

    rates = [[], [], []]
    for i in range(len(rates)):
        for j in range(df.shape[0] - 1):
            rates[i].append((df.iloc[j + 1, -(3 - i)] - df.iloc[j, -(3 - i)]) / 100)
        rates[i].append(0)
    df["mar_rate_carbon"] = rates[0]
    df["goc_sub_rate_carbon"] = rates[1]
    df["goc_surf_rate_carbon"] = rates[2]
    df["total_rate"] = (
        df["mar_rate_carbon"] + df["goc_sub_rate_carbon"] + df["goc_surf_rate_carbon"]
    )

    np.savetxt(
        r"results/total_GoC_rates.txt", df["total_rate"], fmt="%.2f", delimiter="\t"
    )
    np.savetxt(
        "results/optimizedrun_" + str(filename) + ".txt",
        df.values,
        fmt="%.2f",
        delimiter="\t",
    )
    return


def make_carbon_rate_plot(filename):
    df = pd.read_table(
        "results/optimizedrun_" + filename + ".txt", sep="\s+", header=None
    )
    df = df.rename(
        columns={
            0: "year",
            1: "DIC_mar",
            2: "DIC_sub",
            3: "DIC_surf",
            4: "ALK_mar",
            5: "ALK_sub",
            6: "ALK_surf",
            7: "d13C_mar",
            8: "d13C_sub",
            9: "d13C_surf",
            10: "D14C_mar",
            11: "D14C_sub",
            12: "D14C_surf",
            13: "Ccum_mar",
            14: "Ccum_sub",
            15: "Ccum_surf",
            16: "Crate_mar",
            17: "Crate_sub",
            18: "Crate_surf",
        }
    )

    obspath = "data/observations/"

    Rafter_surface = pd.read_csv(obspath + "Rafter_2019.tab", sep="\t", header=24)
    Rafter_surface = Rafter_surface.loc[(Rafter_surface["Habitat"] == "planktic")]
    # Rafter_surface["Cal age [ka BP]"] = 1000 * Rafter_surface["Cal age [ka BP]"]
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
    # Mar["Cal.Age"] = 1000 * Mar["Cal.Age"]
    # df["year"] = 1000 * df["year"]

    # Marchitto, subsurface, surface
    colors = ["#706513", "#B57114", "#520120"]
    D14C_model = [df.D14C_mar, df.D14C_sub, df.D14C_surf]
    D14C_obs = [Mar["D14C"], Rafter_subsurface["D14C"], Rafter_surface["Δ14C [‰]"]]
    year_obs = [
        Mar["Cal.Age"],
        Rafter_subsurface["calendar age [kyr BP]"],
        Rafter_surface["Cal age [ka BP]"],
    ]
    marker_obs = ["s", "^", "o"]
    Crate_model = [df.Crate_mar, df.Crate_sub, df.Crate_surf]

    fig, ax = plt.subplots(2, sharex=True)
    ax1 = ax[1].twinx()
    for i in range(3):
        ax[0].plot(df.year, D14C_model[i], color=colors[i], linestyle="solid")
        ax[0].plot(
            year_obs[i],
            D14C_obs[i],
            color=colors[i],
            linestyle="dashed",
            marker=marker_obs[i],
        )
        ax[1].plot(df.year, Crate_model[i], color=colors[i], linestyle="solid")
    ax1.plot(df.year, df.Ccum_mar + df.Ccum_sub + df.Ccum_surf, color="k")
    ax1.fill_between(
        df.year, df.Ccum_mar + df.Ccum_sub + df.Ccum_surf, 0, color="k", alpha=0.1
    )

    ax[1].set_xlim(0, 20)

    fig.savefig("results/Crate_plot_" + filename + ".pdf")
    return


def make_plot(time, tracers, pH, filename):
    """makes all plots"""
    fig, ax = plt.subplots(3, sharex=True, figsize=(4, 10))

    obspath = "data/observations/"

    Rafter_surface = pd.read_csv(obspath + "Rafter_2019.tab", sep="\t", header=24)
    Rafter_surface = Rafter_surface.loc[(Rafter_surface["Habitat"] == "planktic")]
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

    d13C_obs = pd.read_excel(obspath + "d13C_GoC_benthic_LPAZ21P.xlsx")

    d13C_obs = d13C_obs.sort_values(by=["cal.age"])

    d11B_obs = pd.read_excel("data/observations/prafter-2022-12-21-LPAZ21P-d11B.xlsx")

    # Constants
    dSW = 39.61  # delta of modern SW, per mille
    alpha = 1.0272  # from Hain et al 2018
    epsilon = 27.2  # from Hain et al 2018 per mille
    delta_pKb = 0  # from Hain et al 2018, assume no change in pkb
    d0 = d11B_obs["d11B"].iloc[10]

    def pH_change(d0, d1):
        return delta_pKb - np.log10(
            1
            + (d1 - d0)
            / (dSW - alpha * d1 - epsilon)
            * ((alpha - 1) * dSW - epsilon / (d0 - dSW))
        )

    pH_changes_obs = []
    for i in range(11):
        pH_changes_obs.append(pH_change(d0, d11B_obs["d11B"].iloc[i]))

    # ax[0].plot(time, pH[3, 0, :], label="Baja California pH")
    ax[0].plot(time, pH[3, 0, :] - pH[3, 0, 0], label="subsurface pH")
    ax[0].plot(
        d11B_obs["cal.age.kyr"].iloc[:11] * 1000,
        pH_changes_obs,
        "o--",
        color="black",
        label="subsurface pH",
    )

    # ax[3].plot(
    #     time,
    #     tracers[9, :] / tracers[0, :],
    #     color="#706513",
    #     label="Baja California δ$^{13}$C",
    # )
    ax[2].plot(
        time,
        tracers[12, :] / tracers[0, :],
        color="#706513",
        lw=4,
        label="Marchitto box ∆$^{14}$C",
    )

    # ax[0].plot(
    #     time, pH[3, 1, :], linestyle="dotted", color="#B57114", label="GoC deep pH",
    # )

    # ax[1].plot(
    #     time,
    #     (tracers[10, :] / tracers[1, :]) - (0.6884),
    #     linestyle="dotted",
    #     label="GoC deep δ$^{13}$C",
    # )
    ax[1].plot(
        time,
        (tracers[10, :] / tracers[1, :]),
        linestyle="dotted",
        label="GoC deep δ$^{13}$C",
    )

    # ax[1].plot(
    #     d13C_obs["cal.age"] * 1000,
    #     d13C_obs["δ¹³C (‰, VPDB)"] - (-0.17),
    #     marker="s",
    #     markeredgecolor="k",
    #     markerfacecolor="white",
    #     linestyle="dashed",
    #     color="#B57114",
    #     label="benthic d13C data",
    # )
    ax[1].plot(
        d13C_obs["cal.age"] * 1000,
        d13C_obs["δ¹³C (‰, VPDB)"],
        marker="s",
        markeredgecolor="k",
        markerfacecolor="white",
        linestyle="dashed",
        color="#B57114",
        label="benthic d13C data",
    )
    ax[2].plot(
        time,
        tracers[13, :] / tracers[1, :],
        linestyle="solid",
        lw=4,
        color="#B57114",
        label="GoC subsurface ∆$^{14}$C",
    )

    # ax[0].plot(
    #     time, pH[3, 2, :], linestyle="dashed", label="GoC surface pH",
    # )

    # ax[3].plot(
    #     time,
    #     tracers[11, :] / tracers[2, :],
    #     linestyle="dashed",
    #     color="#520120",
    #     label="GoC surface δ$^{13}$C",
    # )

    ax[2].plot(
        time,
        tracers[14, :] / tracers[2, :],
        linestyle="solid",
        lw=4,
        color="#520120",
        label="GoC surface ∆$^{14}$C",
        zorder=4,
    )

    ax[2].plot(
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

    ax[2].plot(
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

    ax[2].plot(
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

    ax[0].legend(loc="best")
    ax[1].legend(loc="best")
    ax[2].legend(loc="best")

    ax[0].set_ylabel("∆pH")
    ax[0].set_title("∆pH")
    # ax[1].set_ylabel("δ$^{13}$C (permil) \n anomaly from LGM")
    ax[1].set_ylabel("δ$^{13}$C (permil)")
    ax[1].set_title("δ$^{13}$C")
    ax[2].set_xlabel("Years BP")
    ax[2].set_ylabel("∆$^{14}$C (permil)")
    ax[2].set_title("∆$^{14}$C")
    # ax[4].set_ylim(-450, 350)
    ax[2].grid()
    for i in range(3):
        ax[i].set_xlim(0, 20000)

    plt.tight_layout()
    try:
        fig.savefig("results/Plot_" + filename + ".pdf")
    except:
        fig.savefig("../results/Plot_" + filename + ".pdf")


def make_plot_interp(time, tracers, filename):
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
    d13C_obs = pd.read_excel(obspath + "d13C_GoC_benthic_LPAZ21P.xlsx")
    d13C_obs = d13C_obs.sort_values(by=["cal.age"])

    f_surf, f_sub, f_mar = read_files()
    # ax[0].plot(time, carbonate_chemistry[3, 0, :], label="Baja California pH")
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
    # ax[3].plot(
    #     time,
    #     tracers[9, :] / tracers[0, :],
    #     color="#706513",
    #     label="Baja California δ$^{13}$C",
    # )
    ax[4].plot(
        time,
        tracers[12, :] / tracers[0, :],
        color="#706513",
        lw=4,
        label="Marchitto box ∆$^{14}$C",
    )

    # ax[0].plot(
    #     time,
    #     carbonate_chemistry[3, 1, :],
    #     linestyle="dotted",
    #     color="#B57114",
    #     label="GoC deep pH",
    # )
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
    ax[3].plot(
        d13C_obs["cal.age"],
        d13C_obs["δ¹³C (‰, VPDB)"],
        marker="s",
        markeredgecolor="k",
        markerfacecolor="white",
        linestyle="dashed",
        color="#B57114",
        label="benthic d13C data",
    )
    ax[4].plot(
        time,
        tracers[13, :] / tracers[1, :],
        linestyle="solid",
        lw=4,
        color="#B57114",
        label="GoC subsurface ∆$^{14}$C",
    )

    # ax[0].plot(
    #     time, carbonate_chemistry[3, 2, :], linestyle="dashed", label="GoC surface pH",
    # )
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
    # ax[3].plot(
    #     time,
    #     tracers[11, :] / tracers[2, :],
    #     linestyle="dashed",
    #     color="#520120",
    #     label="GoC surface δ$^{13}$C",
    # )

    ax[4].plot(
        time,
        tracers[14, :] / tracers[2, :],
        linestyle="solid",
        lw=4,
        color="#520120",
        label="GoC surface ∆$^{14}$C",
        zorder=4,
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

    mar_time = np.linspace(320, 21000)
    ax[4].plot(
        mar_time,
        f_mar(mar_time),
        linestyle=":",
        color="#706513",
        label="Marchitto interpolated",
        lw=2,
    )
    surf_time = np.linspace(12791, 21000)
    ax[4].plot(
        surf_time,
        f_surf(surf_time),
        linestyle=":",
        color="#520120",
        label="surface interpolated",
        lw=2,
    )
    sub_time = np.linspace(504, 21000)
    ax[4].plot(
        sub_time,
        f_sub(sub_time),
        linestyle=":",
        color="#B57114",
        label="subsurface interpolated ",
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
    # ax[4].set_ylim(-450, 350)
    ax[4].grid()
    for i in range(5):
        ax[i].set_xlim(0, 20000)

    plt.tight_layout()
    try:
        fig.savefig("results/Plot_" + filename + ".pdf")
    except:
        fig.savefig("../results/Plot_" + filename + ".pdf")


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
            13: "PintNP",
            14: "DICsurfNP",
            15: "ALKsurfNP",
            16: "d13CsurfNP",
            17: "D14CsurfNP",
            18: "PsurfNP",
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
            13: "PintNP",
            14: "DICsurfNP",
            15: "ALKsurfNP",
            16: "d13CsurfNP",
            17: "D14CsurfNP",
            18: "PsurfNP",
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
            [df["PintNP"][row], df["PsurfNP"][row]],
            [
                df["d13CintNP"][row] * df["DICintNP"][row],
                df["d13CsurfNP"][row] * df["DICsurfNP"][row],
            ],
            [
                df["D14CintNP"][row] * df["DICintNP"][row],
                df["D14CsurfNP"][row] * df["DICsurfNP"][row],
            ],
            [0, 0],
        ]
    )
    return bc


def read_co2_data(file):

    df = pd.read_table(str(file), sep="\t", header=None)
    df = df.rename(columns={0: "year", 1: "atm_co2"})

    co2_data = df.to_numpy()

    return co2_data


def read_14C_atm_data(file):

    df = pd.read_table(str(file), sep="\t", header=None)
    df = df.rename(columns={0: "year", 1: "14C_atm"})

    c14_atm_data = df.to_numpy()

    return c14_atm_data


def read_d13C_atm_data(file):

    df = pd.read_table(str(file), sep="\t", header=None)
    df.rename(columns={0: "year", 1: "d13C_atm", 2: "stnd_dev"})

    d13C_atm_data = df.to_numpy()

    return d13C_atm_data


def read_files():
    obspath = "data/observations/"

    Rafter_surface = pd.read_csv(obspath + "Rafter_2019.tab", sep="\t", header=24)
    Rafter_surface = Rafter_surface.loc[(Rafter_surface["Habitat"] == "planktic")]
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
    Rafter_subsurface["calendar age [kyr BP]"] *= 1000

    Mar = pd.read_csv(obspath + "Marchitto.txt", sep="\s+")
    Mar["Cal.Age"] = 1000 * Mar["Cal.Age"]

    # self.surf_min = Rafter_surface["Cal age [ka BP]"].min()
    # self.sub_min = Rafter_subsurface["calendar age [kyr BP]"].min()
    # self.mar_min = Mar["Cal.Age"].min()

    # The following are functions that return the D14C value at an inputted timestep
    # Linear interpolation
    # fMar is different because there was a duplicated timestep,
    # but it was outside the time range we care about so I just removed it
    fSurf = interp1d(
        Rafter_surface["Cal age [ka BP]"], Rafter_surface["Δ14C [‰]"], kind="linear"
    )
    fSub = interp1d(
        Rafter_subsurface["calendar age [kyr BP]"], Rafter_subsurface["D14C"]
    )
    fMar = interp1d(
        np.delete(np.array([Mar["Cal.Age"]]), [-3, -4]),
        np.delete(np.array([Mar["D14C"]]), [-3, -4]),
    )
    return fSurf, fSub, fMar


def plot_rate():
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
