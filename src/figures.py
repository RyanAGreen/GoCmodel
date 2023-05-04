import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.conversions as conversions
from scipy.interpolate import interp1d
import src.functions as f
import cartopy
import matplotlib.ticker as mticker
import cartopy.mpl.geoaxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import src.carbchem as cc
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import netCDF4 as nc4
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

obspath = "data/observations/"
modelpath = "data/model/"
figurepath = "results/figures/"

plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.family"] = "PT Sans Narrow"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["xtick.major.size"] = 5.5
plt.rcParams["ytick.major.size"] = 5.5
# plt.rcParams["lines.linewidth"] = 3
plt.rcParams["axes.linewidth"] = 3


# plt.style.use('/src/presentation.mplstyle')


def pH_change(d0, d1):
    return delta_pKb - np.log10(
        1
        + (d1 - d0)
        / (dSW - alpha * d1 - epsilon)
        * ((alpha - 1) * dSW - epsilon / (d0 - dSW))
    )


color_global_obs = "darkgray"
color_atlantic = "#0455BF"
color_indopac = "#88B6F2"
color_global_model = "black"
color_marchitto = "#F2A35E"
color_subsurface = "#01403A"
color_surface = "#F27E7E"
# 5 markers, none for compilation
markers = ["o", "v", "s", "<", "P"]

font_label_size = 12

if "reading in ∆14C, CO2, d13c,d11b, CO3, and OCIM data":
    # atmospheric data

    # Reconstructed CO2 observations, compiled in Bereiter et al. 2015
    bereiter2015_atmCO2 = pd.read_csv(obspath + "CO2data1.txt", sep="\t")
    bereiter2015_atmCO2["year"] = bereiter2015_atmCO2["year"] / 1000

    icecore_CO2 = pd.read_csv(obspath + "IceCoreCO2.txt", sep="\t", skiprows=137)

    # Reconstructed ∆14C observations, IntCal20 Reimer et al. 2020
    reimer2020_atmD14C = pd.read_csv(obspath + "IntCalSmoothed.txt", header=None)
    reimer2020_atmD14C = reimer2020_atmD14C.rename(columns={0: "year", 3: "D14C"})
    reimer2020_atmD14C["year"] = reimer2020_atmD14C["year"] / 1000

    # for tree Ring Data
    def calc_D14C(df):
        df["∆14C"] = (
            np.exp(-1 * df["r_date"] / 8033) * np.exp((1950.5 - df["t"]) / 8267) - 1
        ) * 1000
        return df

    # Read the CSV file
    df = pd.read_csv(
        obspath + "TreeRing_IntCal20.csv",
        header=None,
        skiprows=lambda x: x in range(1, x),
    )

    # Identify the row indices where the separators occur
    separator_rows = df.index[df.isnull().all(axis=1)].tolist()

    # Initialize an empty list to store the section dataframes
    section_dfs = []

    # Loop through the separator rows and create separate dataframes for each section
    for i in range(len(separator_rows)):
        if i == 0:
            start_idx = 0
            end_idx = separator_rows[i]
            section_df = df.iloc[start_idx:end_idx]
        else:
            start_idx = separator_rows[i - 1] + 1
            end_idx = separator_rows[i]
            section_df = df.loc[start_idx:end_idx]
        section_df.columns = section_df.iloc[0]
        section_df = section_df[1:]  # Remove first row from DataFrame
        section_df = section_df.dropna(axis=1, how="all")  # Remove empty columns
        section_df = section_df.loc[
            :, ["r_date", "t"]
        ]  # Extract columns "z" and "calage"
        section_df = section_df.apply(
            pd.to_numeric, errors="coerce"
        )  # Convert "calage" column to numeric
        section_df = section_df.reset_index(drop=True)  # Reset index
        section_dfs.append(section_df)

    # Concatenate all section dataframes into a single dataframe
    tree_ring = pd.concat(section_dfs, ignore_index=True)
    tree_ring = calc_D14C(tree_ring)
    tree_ring["kyrBP"] = (1950 - tree_ring["t"]) / 1000

    ### Hulu Cave observations
    Hulu = pd.read_csv(obspath + "HuluCaveD14C.tab", sep="\t", skiprows=15)
    Hulu = Hulu[(Hulu["Age [ka BP]"] >= 15) & (Hulu["Age [ka BP]"] <= 26)]

    intcal20 = pd.read_csv(obspath + "INTCAL20.txt", skiprows=10)
    intcal13 = pd.read_csv(obspath + "INTCAL13.txt", skiprows=10, encoding="latin-1")
    intcal09 = pd.read_csv(obspath + "INTCAL09.txt", skiprows=10)

    # oceanic data
    # Benthic foram data from the mouth of GoC. Rafter et al. 2018. anomalies observed
    rafter2018_benthic = pd.read_excel(
        obspath + "prafter-2019-Gulf-CA-Data-for-Ryan.xls"
    )
    # Based on Pats instructions -> we use P.ariminensis and U.peregrina, which appear to be the most consistent.
    # both benthic species
    rafter2018_benthic = rafter2018_benthic.loc[
        (rafter2018_benthic["species"] == "U. peregrina")
        | (rafter2018_benthic["species"] == "Planulina ariminensis")
        | (rafter2018_benthic["species"] == "U. peregrina ")
    ]
    rafter2018_benthic = rafter2018_benthic.sort_values(by=["calendar age [kyr BP]"])
    rafter2018_benthic = rafter2018_benthic[["calendar age [kyr BP]", "D14C"]]
    rafter2018_benthic = rafter2018_benthic.dropna(subset=["D14C"])
    rafter2018_benthic = (
        rafter2018_benthic.groupby("calendar age [kyr BP]").mean().reset_index()
    )
    rafter2018_benthic["calendar age [kyr BP]"] = (
        1000 * rafter2018_benthic["calendar age [kyr BP]"]
    )

    # Planktic foram data from the mouth of GoC. Rafter et al. 2018. anomalies observed
    rafter2018_planktic = pd.read_csv(obspath + "Rafter_2019.tab", sep="\t", header=24)
    rafter2018_planktic.loc[(rafter2018_planktic["Habitat"] == "planktic")]
    rafter2018_planktic["Cal age [ka BP]"] = (
        1000 * rafter2018_planktic["Cal age [ka BP]"]
    )
    rafter2018_planktic = rafter2018_planktic.sort_values(by=["Cal age [ka BP]"])

    # Deep sea coral ∆14C data from 627 depth near Galapagos, Chen et al. 2020. No anomalies observed
    chen = pd.read_csv(obspath + "Chen2020.txt", sep="\t", header=0, skiprows=110)
    chen2020_coral = chen[chen["water.depth"] == 627]

    # Benthic foram data from the coast of Baja California. Marchitto et al. 2007. Anomalies observed
    marchitto2007_benthic = pd.read_csv(obspath + "Marchitto.txt", sep="\s+")
    marchitto2007_benthic["Cal.Age"] = 1000 * marchitto2007_benthic["Cal.Age"]

    # Benthic foram ∆14C data from 617 depth near Galapagos. large anomalies observed
    stott2009_benthic = chen[chen["water.depth"] == 617]
    stott2009_benthic = stott2009_benthic[
        stott2009_benthic["ref."] == "Stott et al. (2009)"
    ]

    # ∆14C compilation from Rafter et al. 2022
    rafter2022_14Ccompilation = pd.read_csv(
        obspath + "prafter-2022-Global-D14C-Comp-FIN.csv"
    )
    rafter2022_deeppac = rafter2022_14Ccompilation.loc[
        (
            rafter2022_14Ccompilation[
                "Ocean basin and water mass (along density surfaces; see text)"
            ]
            == "PACIFIC BOTTOM"
        )
    ]

    rafter2022_midpac = rafter2022_14Ccompilation.loc[
        (
            rafter2022_14Ccompilation[
                "Ocean basin and water mass (along density surfaces; see text)"
            ]
            == "PACIFIC MID"
        )
    ]

    rafter2022_time = rafter2022_deeppac["calendar age bin (years BP)"] / 1000
    mean_deep = rafter2022_deeppac["loess_fit"]
    upr68_deep = rafter2022_deeppac["loess_upr68"]
    lwr68_deep = rafter2022_deeppac["loess_lwr68"]
    upr95_deep = rafter2022_deeppac["loess_upr95"]
    lwr95_deep = rafter2022_deeppac["loess_lwr95"]

    mean_mid = rafter2022_midpac["loess_fit"]
    upr68_mid = rafter2022_midpac["loess_upr68"]
    lwr68_mid = rafter2022_midpac["loess_lwr68"]
    upr95_mid = rafter2022_midpac["loess_upr95"]
    lwr95_mid = rafter2022_midpac["loess_lwr95"]

    # Order -> 4 anoamlies (don't behave), then 2 parallel records (behave)
    GoCobs = [
        marchitto2007_benthic,
        stott2009_benthic,
        rafter2018_planktic,
        rafter2018_benthic,
        chen2020_coral,
        rafter2022_midpac,
        rafter2022_deeppac,
    ]

    # make sure the columns are named the same thing
    GoCobs[0] = GoCobs[0].rename(
        columns={
            "Cal.Age": "year",
            "D14C": "D14CintNP",
        }
    )
    GoCobs[1] = GoCobs[1].rename(
        columns={"cal.age": "year", "benthic.D14C": "D14CintNP"}
    )
    GoCobs[2] = GoCobs[2].rename(
        columns={"Cal age [ka BP]": "year", "Δ14C [‰]": "D14CintNP"}
    )
    GoCobs[3] = GoCobs[3].rename(
        columns={
            "calendar age [kyr BP]": "year",
            "D14C": "D14CintNP",
        }
    )
    GoCobs[4] = GoCobs[4].rename(
        columns={"cal.age": "year", "benthic.D14C": "D14CintNP"}
    )
    GoCobs[5] = GoCobs[5].rename(
        columns={
            "calendar age bin (years BP)": "year",
            "loess_fit": "D14CintNP",
        }
    )
    GoCobs[6] = GoCobs[6].rename(
        columns={
            "calendar age bin (years BP)": "year",
            "loess_fit": "D14CintNP",
        }
    )

    for i in range(7):
        GoCobs[i] = GoCobs[i][GoCobs[i]["year"] < 20000]
        GoCobs[i]["year"] = GoCobs[i]["year"] / 1000

    Anomalies = [GoCobs[0], GoCobs[1], GoCobs[2], GoCobs[3]]

    # d13C data from benthic forams in the GoC. unpublished
    d13C_benthic_GoC = pd.read_excel(obspath + "d13C_GoC_benthic_LPAZ21P_averaged.xlsx")
    d13C_benthic_GoC = d13C_benthic_GoC.sort_values(by=["cal.age"])

    # d11B data from benthic forams in the GoC. unpublished
    d11B_benthic_GoC = pd.read_excel(
        "data/observations/prafter-2022-12-21-LPAZ21P-d11B.xlsx"
    )
    # Constants
    dSW = 39.61  # delta of modern SW, per mille
    alpha = 1.0272  # from Hain et al 2018
    epsilon = 27.2  # from Hain et al 2018 per mille
    delta_pKb = 0  # from Hain et al 2018, assume no change in pkb
    d0 = d11B_benthic_GoC["d11B"].iloc[10]  # starting around 20 kyr BP
    pH_changes_obs = []
    for i in range(11):
        pH_changes_obs.append(pH_change(d0, d11B_benthic_GoC["d11B"].iloc[i]))

    # read in CO3 observational data
    indian_obs = pd.read_table(
        obspath + "WIND28K_Yuetal2010.txt", sep="\t", header=None
    )
    indian_obs = indian_obs.drop(labels=0, axis=1)
    indian_obs = indian_obs.rename(columns={1: "time", 2: "CO3"})

    centralpac = pd.read_table(
        obspath + "TTNO13PC61_Yuetal2013.txt", sep="\t", skiprows=3
    )
    southatl = pd.read_table(obspath + "TNO57-21_Yuetal2014.txt", sep="\t", skiprows=2)

    atlantic_obs = pd.read_table(
        obspath + "BOFS8K_Yuetal2008.txt", sep="\t", header=None
    )
    atlantic_obs = atlantic_obs.rename(columns={0: "time", 1: "CO3"})
    pacific_obs = pd.read_fwf(obspath + "GGC48_Yuetal2010.txt", sep="\t", header=None)
    pacific_obs = pacific_obs.drop(
        labels=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12], axis=1
    )
    pacific_obs = pacific_obs.rename(columns={3: "time", 13: "CO3"})

if "reading in CYCLOPS and GoC output":
    # load in CYCLOPS data
    # need to fix these names when we run the simulation in a coupled run
    control = pd.read_table(modelpath + "Control.txt", sep="\s+")
    control_RC = pd.read_table(modelpath + "Control+RC.txt", sep="\s+")
    NP = pd.read_table(modelpath + "NP.txt", sep="\s+")
    NP_LC = pd.read_table(modelpath + "NP+LC.txt", sep="\s+")
    NP_LC_PF = pd.read_table(modelpath + "NP+LC+PF.txt", sep="\s+")
    NP_LC_PF_RC = pd.read_table(modelpath + "NP+LC+PF+RC.txt", sep="\s+")

    # read in GoC data
    GoC = pd.read_table(
        "results/optimizedrun_CO2carbonate_source.txt", header=None, sep="\s+"
    )
    GoC = f.organizedata_goc(GoC)
    GoC_d13C_neg5 = pd.read_table(
        "results/optimizedrun_d13c--5_ALK_DIC-1_forward_run.txt", header=None, sep="\s+"
    )
    GoC_d13C_neg5 = f.organizedata_goc(GoC_d13C_neg5)
    GoC_d13C_neg0 = pd.read_table(
        "results/optimizedrun_d13c-0_ALK_DIC-1_forward_run.txt", header=None, sep="\s+"
    )
    GoC_d13C_neg0 = f.organizedata_goc(GoC_d13C_neg0)

    # reading in model data for the discussion figure
    NP_LC_PF_RC_forward_NP_HCO3 = pd.read_table(
        modelpath + "NP+LC+PF+RC+forward+NP+HCO3.txt", sep="\s+"
    )
    NP_LC_PF_RC_forward_marchitto_HCO3 = pd.read_table(
        modelpath + "NP+LC+PF+RC+forward+marchitto+HCO3.txt", sep="\s+"
    )
    NP_LC_PF_RC_forward_GoCsub_HCO3 = pd.read_table(
        modelpath + "NP+LC+PF+RC+forward+gocsub+HCO3.txt", sep="\s+"
    )
    NP_LC_PF_RC_forward_NP_CO2 = pd.read_table(
        modelpath + "NP+LC+PF+RC+forward+NP+CO2.txt", sep="\s+"
    )
    NP_LC_PF_RC_forward_marchitto_CO2 = pd.read_table(
        modelpath + "NP+LC+PF+RC+forward+marchitto+CO2.txt", sep="\s+"
    )
    NP_LC_PF_RC_forward_GoCsub_CO2 = pd.read_table(
        modelpath + "NP+LC+PF+RC+forward+gocsub+CO2+CO2.txt", sep="\s+"
    )

    GoC_forward_NP_HCO3 = pd.read_table(
        "results/optimizedrun_CO2carbonate_source.txt", header=None, sep="\s+"
    )
    GoC_forward_NP_HCO3 = f.organizedata_goc(GoC_forward_NP_HCO3)
    GoC_forward_marchitto_HCO3 = pd.read_table(
        "results/optimizedrun_CO2carbonate_source.txt", header=None, sep="\s+"
    )
    GoC_forward_GoC_forward_marchitto_HCO3 = f.organizedata_goc(
        GoC_forward_marchitto_HCO3
    )
    GoC_forward_GoCsub_HCO3 = pd.read_table(
        "results/optimizedrun_CO2carbonate_source.txt", header=None, sep="\s+"
    )
    GoC_forward_GoCsub_HCO3 = f.organizedata_goc(GoC_forward_GoCsub_HCO3)

    GoC_forward_NP_CO2 = pd.read_table(
        "results/optimizedrun_CO2carbonate_source.txt", header=None, sep="\s+"
    )
    GoC_forward_NP_CO2 = f.organizedata_goc(GoC_forward_NP_CO2)
    GoC_forward_marchitto_CO2 = pd.read_table(
        "results/optimizedrun_CO2carbonate_source.txt", header=None, sep="\s+"
    )
    GoC_forward_GoC_forward_marchitto_CO2 = f.organizedata_goc(
        GoC_forward_marchitto_CO2
    )
    GoC_forward_GoCsub_CO2 = pd.read_table(
        "results/optimizedrun_CO2carbonate_source.txt", header=None, sep="\s+"
    )
    GoC_forward_GoCsub_CO2 = f.organizedata_goc(GoC_forward_GoCsub_CO2)

    # making indo pacific CO3
    NP_LC_PF_RC_forward_NP_HCO3[
        "deep_indo_pacific_CO3_umolkg"
    ] = NP_LC_PF_RC_forward_NP_HCO3[
        [
            "deep_north_pacific_CO3_umolkg",
            "deep_south_pacific_CO3_umolkg",
            "deep_indian_CO3_umolkg",
        ]
    ].mean(
        axis=1
    )
    NP_LC_PF_RC_forward_marchitto_HCO3[
        "deep_indo_pacific_CO3_umolkg"
    ] = NP_LC_PF_RC_forward_marchitto_HCO3[
        [
            "deep_north_pacific_CO3_umolkg",
            "deep_south_pacific_CO3_umolkg",
            "deep_indian_CO3_umolkg",
        ]
    ].mean(
        axis=1
    )
    NP_LC_PF_RC_forward_GoCsub_HCO3[
        "deep_indo_pacific_CO3_umolkg"
    ] = NP_LC_PF_RC_forward_GoCsub_HCO3[
        [
            "deep_north_pacific_CO3_umolkg",
            "deep_south_pacific_CO3_umolkg",
            "deep_indian_CO3_umolkg",
        ]
    ].mean(
        axis=1
    )

    NP_LC_PF_RC_forward_NP_CO2[
        "deep_indo_pacific_CO3_umolkg"
    ] = NP_LC_PF_RC_forward_NP_CO2[
        [
            "deep_north_pacific_CO3_umolkg",
            "deep_south_pacific_CO3_umolkg",
            "deep_indian_CO3_umolkg",
        ]
    ].mean(
        axis=1
    )
    NP_LC_PF_RC_forward_marchitto_CO2[
        "deep_indo_pacific_CO3_umolkg"
    ] = NP_LC_PF_RC_forward_marchitto_CO2[
        [
            "deep_north_pacific_CO3_umolkg",
            "deep_south_pacific_CO3_umolkg",
            "deep_indian_CO3_umolkg",
        ]
    ].mean(
        axis=1
    )
    NP_LC_PF_RC_forward_GoCsub_CO2[
        "deep_indo_pacific_CO3_umolkg"
    ] = NP_LC_PF_RC_forward_GoCsub_CO2[
        [
            "deep_north_pacific_CO3_umolkg",
            "deep_south_pacific_CO3_umolkg",
            "deep_indian_CO3_umolkg",
        ]
    ].mean(
        axis=1
    )


def Figure1_anomalies():
    """plotting ∆14C anomalies from ETNP"""

    color_anomaly = "orange"
    color_parallel = "teal"

    # 5 markers, none for compilation
    markers = ["o", "v", "s", "<", "P"]

    fig, ax = plt.subplots(1)

    # thickening axis
    for axis in ["top", "left", "right", "bottom"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(axis="y", labelsize="large")
    ax.tick_params(axis="x", labelsize="large")
    # ax[0].set_yticklabels([])

    ax.set_ylabel("∆$^{14}$C (‰)", fontsize=20, fontweight="bold")
    ax.set_xlabel("Calendar age (kyr BP)", fontsize=20, fontweight="bold")
    ax.set_xlim((0, 20))
    labels = [
        "Marchitto et al., 2007-Benthic",
        "Stott et al., 2009-Benthic",
        "Rafter et al., 2018-Planktic",
        "Rafter et al., 2018-Benthic",
        "Chen et al., 2020-Coral",
        "Rafter et al., 2022-$^{14}$C compilation",
    ]

    for i in range(len(Anomalies)):
        ax.plot(
            Anomalies[i].year,
            Anomalies[i].D14CintNP,
            marker=markers[i],
            markeredgecolor="k",
            markerfacecolor="orange",
            color="orange",
            ls="solid",
            label=labels[i],
            lw=1,
            markersize=8,
            zorder=2,
        )

    # Chen
    ax.plot(
        GoCobs[4].year,
        GoCobs[4].D14CintNP,
        marker=markers[4],
        markeredgecolor="k",
        markerfacecolor="teal",
        color="teal",
        ls="solid",
        lw=1,
        label=labels[4],
        markersize=8,
        zorder=2,
    )
    # Rafter Compilation
    ax.plot(
        GoCobs[5].year,
        GoCobs[5].D14CintNP,
        linestyle="solid",
        color="teal",
        label=labels[5],
        lw=4,
        zorder=0,
    )
    ax.fill_between(rafter2022_time, lwr95_mid, upr95_mid, color="teal", alpha=0.5)
    # atmospheric observations
    ax.plot(
        reimer2020_atmD14C.year,
        reimer2020_atmD14C.D14C,
        color="darkgray",
        ls="-",
        label="Atmospheric observations",
        lw=4,
        zorder=0,
    )
    ax.legend(
        loc="lower left", fontsize=10, ncol=1, frameon=False
    )  # , bbox_to_anchor=(1, 0.96)

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis="both", direction="in", length=7, width=3, color="black")

    ax.set_ylim(-700, 650)
    ax.set_xlim(0, 20)

    plt.savefig(
        "/home/rygreen/GoCmodel/results/Figures/Figure1.png", bbox_inches="tight"
    )
    return


def Figure3_modelresults():
    """This figure shows the model results (∆14C or carbon rate?) for each box"""

    fig, ax = plt.subplots(3, figsize=(8, 11), sharex=True)
    ax0 = ax[0].twinx()
    ax1 = ax[1].twinx()
    ax2 = ax[2].twinx()

    # designing plot
    for i in range(3):
        # for axis in ["top", "bottom", "left", "right"]:
        #     ax[i].spines[axis].set_linewidth(3.5)
        ax[i].tick_params(bottom=True, top=True, left=True, right=False)
        ax[i].tick_params(
            labelbottom=False, labeltop=False, labelleft=True, labelright=False
        )
        ax[i].tick_params(axis="both", direction="in", length=7, width=3, color="black")
        ax[i].set_ylim(-450, 350)
        # ax[i].grid()
    # ax[2].set_title("Gulf of California surface box", fontweight="bold", fontsize=10)
    # ax[1].set_title("Gulf of California subsurface box", fontweight="bold", fontsize=10)
    # ax[0].set_title("Marchitto box", fontweight="bold", fontsize=10)
    ax[2].set_xlabel("Calendar age [kyr BP]", fontweight="bold", fontsize=10)
    ax[2].set_xlim(0, 20)
    # plotting data
    # surface box
    ax[2].plot(
        GoC.year,
        GoC.D14C_surf,
        linestyle="solid",
        color=color_surface,
        lw=4,
    )
    ax[2].plot(
        rafter2018_planktic["Cal age [ka BP]"] / 1000,
        rafter2018_planktic["Δ14C [‰]"],
        marker="^",
        markeredgecolor="k",
        markerfacecolor="white",
        linestyle="dashed",
        color=color_surface,
        label="Rafter et al. 2019-GoC surface",
        markersize=6,
        lw=2,
    )

    # subsurface box
    ax[1].plot(
        GoC.year,
        GoC.D14C_sub,
        linestyle="solid",
        color=color_subsurface,
        lw=4,
    )
    ax[1].plot(
        rafter2018_benthic["calendar age [kyr BP]"] / 1000,
        rafter2018_benthic["D14C"],
        marker="s",
        markeredgecolor="k",
        markerfacecolor="white",
        linestyle="dashed",
        color=color_subsurface,
        label="Rafter et al. 2019-benthic",
        markersize=6,
        lw=2,
    )

    # Marchitto Box
    ax[0].plot(
        GoC.year,
        GoC.D14C_mar,
        linestyle="solid",
        color=color_marchitto,
        lw=4,
    )
    ax[0].plot(
        marchitto2007_benthic["Cal.Age"] / 1000,
        marchitto2007_benthic["D14C"],
        marker="o",
        markeredgecolor="k",
        markerfacecolor="white",
        linestyle="dashed",
        color=color_marchitto,
        label="Marchitto et al. 2007-benthic",
        markersize=6,
        lw=2,
    )

    # release rates
    ax0.plot(GoC.year, GoC.Crate_mar, color=color_marchitto, alpha=0.8, zorder=0)
    ax0.bar(
        GoC.year, GoC.Crate_mar, width=0.1, alpha=0.4, color=color_marchitto, zorder=0
    )
    ax1.plot(GoC.year, GoC.Crate_sub, color=color_subsurface, alpha=0.8, zorder=1)
    ax1.bar(
        GoC.year, GoC.Crate_sub, width=0.1, alpha=0.4, color=color_subsurface, zorder=1
    )
    ax2.plot(GoC.year, GoC.Crate_surf, color=color_surface, alpha=0.8, zorder=2)
    ax2.bar(
        GoC.year, GoC.Crate_surf, width=0.1, alpha=0.4, color=color_surface, zorder=2
    )

    carbon_added = [
        str(round(GoC.Ccum_mar[200])),
        str(round(GoC.Ccum_sub[200])),
        str(round(GoC.Ccum_surf[200])),
    ]
    colors = [color_marchitto, color_subsurface, color_surface]
    names = ["Marchitto box", "GoC subsurface", "GoC surface"]
    for i in range(3):
        ax[i].text(
            2.51,
            75,
            names[i] + "\n    " + carbon_added[i] + " [PgC]",
            fontsize=10,
            fontweight="bold",
            color=colors[i],
        )

    # set up common y label
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    plt.ylabel("∆$^{14}$C [‰]", fontweight="bold", fontsize=10, labelpad=15)
    yaxis = plt.twinx()
    twinaxes = [yaxis, ax0, ax1, ax2]
    for i in range(len(twinaxes)):
        twinaxes[i].tick_params(bottom=False, top=False, left=False, right=True)
        twinaxes[i].tick_params(
            labelbottom=False, labeltop=False, labelleft=False, labelright=True
        )
        twinaxes[i].tick_params(
            axis="both", direction="in", length=7, width=3, color="black"
        )

    yaxis.set_ylabel("Release rate", fontsize=10, fontweight="bold")
    # plt.show()
    # putting twin axis below original axis
    ax[0].set_zorder(ax0.get_zorder() + 1)
    ax[0].set_frame_on(False)
    # putting twin axis below original axis
    ax[1].set_zorder(ax1.get_zorder() + 1)
    ax[1].set_frame_on(False)
    # putting twin axis below original axis
    ax[2].set_zorder(ax2.get_zorder() + 1)
    ax[2].set_frame_on(False)
    plt.savefig(figurepath + "Figure3_modelresults.pdf", bbox_inches="tight")


def Figure4_modelvsobs():
    """
    plotting global and regional results against observations
    atmospheric CO2, ∆14C, deep ocean [CO3], regional pH, regional ∆14C, rate of carbon additon?
    """
    # color_anomaly = "#44AA99"
    # color_parallel = "#8A0C0D"
    color_global_obs = "darkgray"
    color_atlantic = "#0455BF"
    color_indopac = "#88B6F2"
    color_global_model = "black"
    color_marchitto = "#F2A35E"
    color_subsurface = "#01403A"
    color_surface = "#F27E7E"

    font_label_size = 12

    atlantic_color = "#1763F9"
    indo_pac_color = "#BF840E"
    markers = ["o", "v", "s", "<", "P"]

    fig, ax = plt.subplots(6, 1, figsize=(3.5, 15.5), sharex=True)

    for i in range(6):
        # thickening axis
        for axis in ["top", "left", "right", "bottom"]:
            ax[i].spines[axis].set_linewidth(3)

        ax[i].axvspan(11.6, 12.9, alpha=0.4, color="darkgray", zorder=0)
        ax[i].axvspan(14.5, 18, alpha=0.4, color="darkgray", zorder=0)
        ax[i].tick_params(bottom=True, top=True, left=True, right=True)
        ax[i].tick_params(axis="both", direction="in", length=7, width=2, color="black")

    ### atmospheric CO2 ###

    # observations
    ax[0].plot(
        bereiter2015_atmCO2.year,
        bereiter2015_atmCO2.CO2,
        color=color_global_obs,
        marker="o",
        markersize=1,
        ls=" ",
        lw=2,
        zorder=0,
    )
    ax[0].plot(
        bereiter2015_atmCO2.year,
        bereiter2015_atmCO2.CO2,
        color=color_global_obs,
        ls=":",
        lw=2,
        zorder=0,
    )
    # models
    ax[0].plot(
        NP_LC_PF_RC.year_kyrBP,
        NP_LC_PF_RC.atmospheric_CO2_ppm,
        color=color_global_model,
        lw=2,
    )
    # labels
    ax[0].set_ylabel(
        "CO$_{2}$$^{atm}$ (ppm)", fontsize=font_label_size, fontweight="bold"
    )

    ### atmospheric D14C ###

    # observations
    # ax[1].plot(
    #     D14C.year,
    #     D14C.D14C,
    #     color=color_global_obs,
    #     marker="o",
    #     markersize=1,
    #     ls=" ",
    #     lw=2,
    #     zorder=0,
    # )
    ax[1].plot(
        reimer2020_atmD14C.year,
        reimer2020_atmD14C.D14C,
        color=color_global_obs,
        ls=":",
        lw=2,
        zorder=0,
    )
    # models
    ax[1].plot(
        NP_LC_PF_RC.year_kyrBP,
        NP_LC_PF_RC["atmospheric_∆14C_permil"],
        color=color_global_model,
        lw=2,
    )
    # labels
    ax[1].set_ylabel(
        "∆$^{14}$C$^{atm}$ (‰)", fontsize=font_label_size, fontweight="bold"
    )

    ### Deep ocean CO3 ###

    # models
    ax[2].plot(
        NP_LC_PF_RC.year_kyrBP,
        NP_LC_PF_RC.deep_atlantic_CO3_umolkg,
        color=color_atlantic,
        alpha=1,
        linestyle="-",
        lw=2,
        label="Atlantic",
    )
    ax[2].plot(
        NP_LC_PF_RC.year_kyrBP,
        (
            NP_LC_PF_RC.deep_north_pacific_CO3_umolkg
            + NP_LC_PF_RC.deep_south_pacific_CO3_umolkg
            + NP_LC_PF_RC.deep_indian_CO3_umolkg
        )
        / 3,
        color=color_indopac,
        linestyle="-",
        lw=2,
        label="Indo-pacific",
    )
    # observations
    ax[2].plot(
        atlantic_obs["time"],
        atlantic_obs["CO3"],
        color=color_atlantic,
        alpha=0.5,
        lw=1,
        marker="o",
        markeredgecolor=color_atlantic,
        markerfacecolor=color_atlantic,
        ls="None",
        markersize=5,
        zorder=0,
        label="BOFS 8K N. Atlantic",
    )
    ax[2].plot(
        indian_obs["time"],
        indian_obs["CO3"],
        color=color_indopac,
        alpha=0.5,
        lw=1,
        marker="^",
        markeredgecolor=color_indopac,
        markerfacecolor=color_indopac,
        ls="None",
        markersize=5,
        zorder=0,
        label="WIND 28K Indian",
    )
    ax[2].plot(
        pacific_obs["time"],
        pacific_obs["CO3"],
        color=color_indopac,
        alpha=0.5,
        lw=1,
        marker="s",
        markeredgecolor=color_indopac,
        markerfacecolor=color_indopac,
        ls="None",
        markersize=5,
        zorder=0,
        label="GGC48 EQ Pacific",
    )
    # labels
    ax[2].set_ylabel("[CO$_{3}$$^{2-}$]", fontsize=font_label_size, fontweight="bold")

    ### pH ###
    ax[3].plot(
        GoC.year,
        GoC.pH_sub - GoC.pH_sub[0],
        lw=2,
        color=color_subsurface,
        label="subsurface pH",
    )
    ax[3].plot(
        d11B_benthic_GoC["cal.age.kyr"].iloc[:11],
        pH_changes_obs,
        ls=":",
        lw=1,
        marker="s",
        markersize=4,
        color=color_subsurface,
        label="subsurface pH",
    )
    ax[3].axhline(0, ls=":", color="k", alpha=0.4)
    ax[3].set_ylabel("∆pH", fontsize=font_label_size, fontweight="bold")

    ### ETNP ∆14C ###
    # Marchitto
    ax[4].plot(
        Anomalies[0].year,
        Anomalies[0].D14CintNP,
        marker=markers[0],
        markeredgecolor="k",
        markerfacecolor=color_marchitto,
        color=color_marchitto,
        ls=":",
        lw=1,
        markersize=4,
        zorder=2,
        alpha=0.8,
    )
    # subsurface
    ax[4].plot(
        Anomalies[2].year,
        Anomalies[2].D14CintNP,
        marker=markers[2],
        markeredgecolor="k",
        markerfacecolor=color_subsurface,
        color=color_subsurface,
        ls=":",
        lw=1,
        markersize=4,
        zorder=2,
        alpha=0.8,
    )

    # surface
    ax[4].plot(
        Anomalies[3].year,
        Anomalies[3].D14CintNP,
        marker=markers[3],
        markeredgecolor="k",
        markerfacecolor=color_surface,
        color=color_surface,
        ls=":",
        lw=1,
        markersize=4,
        zorder=2,
        alpha=0.8,
    )
    ax[4].plot(GoC.year, GoC.D14C_mar, lw=2, color=color_marchitto)
    ax[4].plot(GoC.year, GoC.D14C_sub, lw=2, color=color_subsurface)
    ax[4].plot(GoC.year, GoC.D14C_surf, lw=2, color=color_surface)
    ax[4].set_ylabel("ETNP ∆$^{14}$C (‰)", fontsize=font_label_size, fontweight="bold")

    ### Release Rate ###
    ax[5].plot(GoC.year, GoC.Crate_mar, color=color_marchitto, zorder=0)
    ax[5].bar(
        GoC.year, GoC.Crate_mar, width=0.1, alpha=0.8, color=color_marchitto, zorder=0
    )
    ax[5].plot(GoC.year, GoC.Crate_sub, color=color_subsurface, zorder=1)
    ax[5].bar(
        GoC.year, GoC.Crate_sub, width=0.1, alpha=0.8, color=color_subsurface, zorder=1
    )
    ax[5].plot(GoC.year, GoC.Crate_surf, color=color_surface, zorder=2)
    ax[5].bar(
        GoC.year, GoC.Crate_surf, width=0.1, alpha=0.8, color=color_surface, zorder=2
    )
    ax[5].set_ylabel("Release rate", fontsize=font_label_size, fontweight="bold")
    ax[5].set_xlabel(
        "Calendar age (kyr BP)", fontsize=font_label_size, fontweight="bold"
    )
    ax[5].set_xlim(0, 20)

    fig.text(
        1.04, 0.68, "Global constraints", va="center", rotation="vertical", fontsize=15
    )
    fig.text(
        1.04, 0.4, "Regional constraints", va="center", rotation="vertical", fontsize=15
    )

    if "making legends":
        # make legends
        legend_global_model = mlines.Line2D(
            [],
            [],
            color=color_global_model,
            linestyle="solid",
            lw=2.5,
            label="Model",
        )
        legend_global_obs = mlines.Line2D(
            [],
            [],
            color=color_global_obs,
            marker="o",
            linestyle=":",
            lw=2,
            label="Observations",
        )
        legend_atlantic_model = mlines.Line2D(
            [],
            [],
            color=color_atlantic,
            linestyle="solid",
            lw=2.5,
            label="Model-Atlantic",
        )
        legend_indopac_model = mlines.Line2D(
            [],
            [],
            color=color_indopac,
            linestyle="solid",
            lw=2.5,
            label="Model-Indo Pacific",
        )
        legend_atlantic_obs = mlines.Line2D(
            [],
            [],
            color=color_atlantic,
            alpha=0.5,
            marker="o",
            linestyle="None",
            markersize=5,
            label="Observations-Atlantic",
        )
        legend_ind_obs = mlines.Line2D(
            [],
            [],
            color=color_indopac,
            alpha=0.5,
            marker="^",
            linestyle="None",
            markersize=5,
            label="Observations-Indian",
        )
        legend_pac_obs = mlines.Line2D(
            [],
            [],
            color=color_indopac,
            alpha=0.5,
            marker="s",
            linestyle="None",
            markersize=5,
            label="Observations-Pacific",
        )

        legend_marchitto_model = mlines.Line2D(
            [],
            [],
            color=color_marchitto,
            linestyle="solid",
            lw=2.5,
            label="Simulated Marchitto box",
        )
        legend_subsurface_model = mlines.Line2D(
            [],
            [],
            color=color_subsurface,
            linestyle="solid",
            lw=2.5,
            label="Simulated GoC subsurface box",
        )
        legend_surface_model = mlines.Line2D(
            [],
            [],
            color=color_surface,
            linestyle="solid",
            lw=2.5,
            label="Simulated GoC surface box",
        )
        legend_marchitto_obs = mlines.Line2D(
            [],
            [],
            color=color_marchitto,
            marker=markers[0],
            linestyle="--",
            lw=2.5,
            label="Observations from Marchitto box",
        )
        legend_subsurface_obs = mlines.Line2D(
            [],
            [],
            color=color_subsurface,
            linestyle="--",
            marker=markers[2],
            lw=2.5,
            label="Observations from GoC subsurface box",
        )
        legend_surface_obs = mlines.Line2D(
            [],
            [],
            color=color_surface,
            linestyle="--",
            marker=markers[3],
            lw=2.5,
            label="Observations from GoC surface box",
        )

        ### simplified version ###

        legend_CO2_model = mlines.Line2D(
            [],
            [],
            color=color_global_model,
            linestyle="solid",
            lw=2.5,
            label="CO$_2^{model}$",
        )
        legend_CO2_obs = mlines.Line2D(
            [],
            [],
            color=color_global_obs,
            linestyle=":",
            lw=2,
            label="CO$_2^{obs}$",
        )
        legend_D14C_model = mlines.Line2D(
            [],
            [],
            color=color_global_model,
            linestyle="solid",
            lw=2.5,
            label="∆$^{14}$C$^{model}$",
        )
        legend_D14C_obs = mlines.Line2D(
            [],
            [],
            color=color_global_obs,
            linestyle=":",
            lw=2,
            label="∆$^{14}$C$^{obs}$",
        )
        legend_atlantic = mpatches.Patch(
            color=color_atlantic, linestyle="solid", lw=2.5, label="Atlantic"
        )
        legend_indopac = mpatches.Patch(
            color=color_indopac, linestyle="solid", lw=2.5, label="Indo Pacific"
        )
        legend_marchitto = mpatches.Patch(
            color=color_marchitto,
            linestyle="solid",
            lw=2.5,
            label="Marchitto box",
        )
        legend_subsurface = mpatches.Patch(
            color=color_subsurface,
            linestyle="solid",
            lw=2.5,
            label="GoC subsurface box",
        )
        legend_surface = mpatches.Patch(
            color=color_surface,
            linestyle="solid",
            lw=2.5,
            label="GoC surface box",
        )

        ax[0].legend(
            handles=[legend_CO2_model, legend_CO2_obs],
            ncol=1,
            loc="lower left",
            frameon=False,
        )
        ax[1].legend(
            handles=[legend_D14C_model, legend_D14C_obs],
            ncol=1,
            loc="upper left",
            frameon=False,
        )
        #     ax[2].legend(
        #     handles=[legend_atlantic_model,legend_indopac_model,legend_atlantic_obs,legend_ind_obs,legend_pac_obs],
        #     loc="upper left",
        #     frameon=False,
        # )
        ax[2].legend(
            handles=[legend_atlantic, legend_indopac],
            loc="upper left",
            frameon=True,
        )
        ax[5].legend(
            handles=[legend_surface, legend_subsurface, legend_marchitto],
            ncol=1,
            loc="upper left",
            frameon=False,
        )

    plt.savefig(figurepath + "Figure3.pdf", bbox_inches="tight")
    return


def Figure5_d13C():
    """plotting d13C model output vs observations"""
    fig, ax = plt.subplots(1)

    ax.set_ylabel("$\delta^{13}$C change since LGM (‰)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Calendar age (kyr BP)", fontsize=13, fontweight="bold")

    # ax.plot(
    #     GoC.year, GoC.d13C_mar - GoC.d13C_mar[0], ls="-", lw=4, zorder=0,
    # )
    ax.plot(
        GoC.year,
        GoC.d13C_sub - GoC.d13C_sub[0],
        ls="-",
        color="#e9c46a",
        lw=3,
        zorder=0,
    )
    ax.plot(
        GoC_d13C_neg5.year,
        GoC_d13C_neg5.d13C_sub - GoC_d13C_neg5.d13C_sub[0],
        ls="-",
        color="#2a9d8f",
        alpha=0.35,
        lw=3,
        zorder=0,
    )
    ax.plot(
        GoC_d13C_neg0.year,
        GoC_d13C_neg0.d13C_sub - GoC_d13C_neg0.d13C_sub[0],
        ls="-",
        color="#264653",
        alpha=0.35,
        lw=3,
        zorder=0,
    )

    # ax.plot(
    #     GoC.year, GoC.d13C_surf - GoC.d13C_surf[0], ls="-", lw=4, zorder=0,
    # )
    ax.plot(
        d13C_benthic_GoC["cal.age"],
        d13C_benthic_GoC["δ¹³C (‰, VPDB)"] - d13C_benthic_GoC["δ¹³C (‰, VPDB)"][14],
        marker="s",
        markersize=10,
        markeredgecolor="k",
        markerfacecolor="white",
        linestyle="dashed",
        lw=3,
        color="#f4a261",
        label="benthic d13C data",
        zorder=5,
    )

    # ax.legend(
    #     loc="lower left", fontsize=10, ncol=1, frameon=False
    # )  # , bbox_to_anchor=(1, 0.96)
    # ax.text(
    #     2.25,
    #     -0.8,
    #     "sedimentary carbonate \n dissolution δ¹³C = -2.5‰",
    #     fontsize=10,
    #     fontweight="bold",
    #     color="#e9c46a",
    # )

    # ax.text(
    #     15,
    #     -1.5,
    #     "δ¹³C = -5‰",
    #     fontsize=10,
    #     alpha=0.35,
    #     fontweight="bold",
    #     color=color_subsurface,
    # )
    # ax.text(
    #     7,
    #     -0.12,
    #     "δ¹³C = 0‰",
    #     fontsize=10,
    #     alpha=0.35,
    #     fontweight="bold",
    #     color=color_subsurface,
    # )

    ax.tick_params(bottom=True, top=True, left=True, right=True)

    ax.set_xlim(7.5, 20)

    plt.savefig(figurepath + "Figure5_d13C.pdf", bbox_inches="tight")
    return


def Figure6_discussion():
    """
    this figure will show two rows (one bicarbonate and one CO2 addition)
    and three columns (North Pacific intermediate box from CYCLOPS,
    Marchitto Box, GoC subsurface box). The columns show where the carbon
    was added and what the ∆14C is, but the row shows what type of carbon
    """
    fig, ax = plt.subplots(2, 3, sharex="col", sharey="row")
    ax = ax.flatten()
    for i in range(6):
        ax[i].tick_params(bottom=True, top=True, left=True, right=True)
    for i in range(3, 6):
        # ax[i].set_xlabel("Calendar age (kyr BP)")
        ax[i].set_xlim(10, 20)

    # plotting ∆14C

    # North Pacific ∆14C. HCO3 added to North Pacific
    ax[0].plot(
        NP_LC_PF_RC_forward_NP_HCO3["year_kyrBP"],
        NP_LC_PF_RC_forward_NP_HCO3["intNP_∆14C_permil"],
        lw=2,
        color=color_marchitto,
    )
    # Marchitto ∆14C. HCO3 added to Marchitto Box
    ax[1].plot(
        GoC_forward_marchitto_HCO3.year,
        GoC_forward_marchitto_HCO3.D14C_mar,
        lw=2,
        color=color_subsurface,
    )
    # GoC subsurface ∆14C. HCO3 added to GoC subsurface
    ax[2].plot(
        GoC_forward_GoCsub_HCO3.year,
        GoC_forward_GoCsub_HCO3.D14C_sub,
        lw=2,
        color=color_surface,
    )
    # North Pacific ∆14C. CO2 added to North Pacific
    ax[3].plot(
        NP_LC_PF_RC_forward_NP_CO2["year_kyrBP"],
        NP_LC_PF_RC_forward_NP_CO2["intNP_∆14C_permil"],
        lw=2,
        color=color_marchitto,
    )
    # Marchitto ∆14C. CO2 added to Marchitto Box
    ax[4].plot(
        GoC_forward_marchitto_CO2.year,
        GoC_forward_marchitto_CO2.D14C_mar,
        lw=2,
        color=color_subsurface,
    )
    # GoC subsurface ∆14C. CO2 added to GoC subsurface
    ax[5].plot(
        GoC_forward_GoCsub_CO2.year,
        GoC_forward_GoCsub_CO2.D14C_sub,
        lw=2,
        color=color_surface,
    )

    # plotting atmospheric CO2 (always going to come from CYCLOPS)

    # atmospheric CO2. HCO3 added to North Pacific
    ax[0].plot(
        NP_LC_PF_RC_forward_NP_HCO3.year_kyrBP,
        NP_LC_PF_RC_forward_NP_HCO3.atmospheric_CO2_ppm,
        lw=2,
    )
    # atmospheric CO2. HCO3 added to Marchitto Box
    ax[0].plot(
        NP_LC_PF_RC_forward_marchitto_HCO3.year_kyrBP,
        NP_LC_PF_RC_forward_GoCsub_HCO3.atmospheric_CO2_ppm,
        lw=2,
    )
    # atmospheric CO2. HCO3 added to GoC subsurface
    ax[0].plot(
        NP_LC_PF_RC_forward_GoCsub_HCO3.year_kyrBP,
        NP_LC_PF_RC_forward_GoCsub_HCO3.atmospheric_CO2_ppm,
        lw=2,
    )
    # atmospheric CO2. CO2 added to North Pacific
    ax[0].plot(
        NP_LC_PF_RC_forward_NP_CO2.year_kyrBP,
        NP_LC_PF_RC_forward_NP_CO2.atmospheric_CO2_ppm,
        lw=2,
    )
    # atmospheric CO2. CO2 added to Marchitto Box
    ax[0].plot(
        NP_LC_PF_RC_forward_marchitto_CO2.year_kyrBP,
        NP_LC_PF_RC_forward_marchitto_CO2.atmospheric_CO2_ppm,
        lw=2,
    )
    # atmospheric CO2. CO2 added to GoC subsurface
    ax[0].plot(
        NP_LC_PF_RC_forward_GoCsub_CO2.year_kyrBP,
        NP_LC_PF_RC_forward_GoCsub_CO2.atmospheric_CO2_ppm,
        lw=2,
    )

    # plotting CO3 concentration (Indo pacific?)(always going to come from CYCLOPS)

    # deep indo-pacific CO3. HCO3 added to North Pacific
    ax[0].plot(
        NP_LC_PF_RC_forward_NP_HCO3.year_kyrBP,
        NP_LC_PF_RC_forward_NP_HCO3["deep_indo_pacific_CO3_umolkg"],
        lw=2,
    )
    # deep indo-pacific CO3. HCO3 added to Marchitto Box
    ax[0].plot(
        NP_LC_PF_RC_forward_marchitto_HCO3.year_kyrBP,
        NP_LC_PF_RC_forward_GoCsub_HCO3["deep_indo_pacific_CO3_umolkg"],
        lw=2,
    )
    # deep indo-pacific CO3. HCO3 added to GoC subsurface
    ax[0].plot(
        NP_LC_PF_RC_forward_GoCsub_HCO3.year_kyrBP,
        NP_LC_PF_RC_forward_GoCsub_HCO3["deep_indo_pacific_CO3_umolkg"],
        lw=2,
    )
    # deep indo-pacific CO3. CO2 added to North Pacific
    ax[0].plot(
        NP_LC_PF_RC_forward_NP_CO2.year_kyrBP,
        NP_LC_PF_RC_forward_NP_CO2["deep_indo_pacific_CO3_umolkg"],
        lw=2,
    )
    # deep indo-pacific CO3. CO2 added to Marchitto Box
    ax[0].plot(
        NP_LC_PF_RC_forward_marchitto_CO2.year_kyrBP,
        NP_LC_PF_RC_forward_marchitto_CO2["deep_indo_pacific_CO3_umolkg"],
        lw=2,
    )
    # deep indo-pacific CO3. CO2 added to GoC subsurface
    ax[0].plot(
        NP_LC_PF_RC_forward_GoCsub_CO2.year_kyrBP,
        NP_LC_PF_RC_forward_GoCsub_CO2["deep_indo_pacific_CO3_umolkg"],
        lw=2,
    )


def presentation_fig_anomalies():
    """this introduces the anomalies for a presentation"""

    fig, ax = plt.subplots(1, figsize=(10, 6))

    colors = ["#B57114", "#B57114", "#B57114"]

    ax.set_ylabel("∆$^{14}$C (‰)")
    ax.set_xlabel("Calendar age (kyr BP)")
    ax.set_xlim((0, 20))
    labels = [
        "Marchitto et al., 2007-Benthic",
        "Stott et al., 2009-Benthic",
        "Rafter et al., 2018-Planktic",
        "Rafter et al., 2018-Benthic",
        "Chen et al., 2020-Coral",
        "Rafter et al., 2022-$^{14}$C compilation",
    ]
    for i in range(1):
        ax.plot(
            Anomalies[i].year,
            Anomalies[i].D14CintNP,
            marker=markers[i],
            markeredgecolor="k",
            markerfacecolor="orange",
            color="orange",
            ls="--",
            label=labels[i],
            lw=3,
            markersize=8,
            zorder=2,
        )

    # Rafter Compilation
    ax.plot(
        GoCobs[5].year,
        GoCobs[5].D14CintNP,
        linestyle="-",
        color="teal",
        label=labels[5],
        lw=4,
        zorder=0,
    )
    ax.fill_between(rafter2022_time, lwr95_mid, upr95_mid, color="teal", alpha=0.5)
    # atmospheric observations
    ax.plot(
        reimer2020_atmD14C.year,
        reimer2020_atmD14C.D14C,
        color="darkgray",
        ls="-",
        label="Atmospheric observations",
        lw=4,
        zorder=0,
    )

    ax.tick_params(axis="both", which="minor", labelsize=15)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis="both", direction="in", length=7, width=3, color="black")
    ax.tick_params(bottom=True, top=True, left=True, right=True)

    ax.set_ylim(-250, 450)
    plt.savefig(figurepath + "anomalies_presentation.pdf", bbox_inches="tight")
    return


def presentation_fig_atmrecords():
    """ "this figure shows atmospheric ∆14C and CO2"""
    fig, ax = plt.subplots(1)
    ax1 = ax.twinx()

    # atmospheric observations
    # ax.plot(
    #     reimer2020_atmD14C.year,
    #     reimer2020_atmD14C.D14C,
    #     color="darkgray",
    #     ls="-",
    #     label="Atmospheric observations",
    #     lw=4,
    #     zorder=0,
    # )
    # atmospheric observations
    ax1.plot(
        bereiter2015_atmCO2.year,
        bereiter2015_atmCO2.CO2,
        color="darkgray",
        ls="-",
        label="Atmospheric observations",
        lw=4,
        zorder=0,
    )

    ax.plot(
        Anomalies[0].year,
        Anomalies[0].D14CintNP,
        marker=markers[0],
        markeredgecolor="k",
        markerfacecolor="orange",
        color="orange",
        ls="--",
        lw=3,
        markersize=8,
        zorder=2,
    )

    ax.set_ylabel("∆$^{14}$C (‰)")
    ax.set_xlabel("Calendar age (kyr BP)")
    ax1.set_ylabel("CO$_2$ (ppm)")
    ax.set_xlim((0, 20))
    ax.set_ylim(-250, 450)

    # putting twin axis below original axis
    ax.set_zorder(ax1.get_zorder() + 1)
    ax.set_frame_on(False)

    plt.savefig(
        figurepath + "atmrecords_presentation_noatm14C.pdf", bbox_inches="tight"
    )

    return


def presentation_fig_atm_robust():
    """this shows that the atmospheric data is well constrained and why we use it"""
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))

    # atmospheric ∆14C observations
    ax[0].plot(
        reimer2020_atmD14C.year,
        reimer2020_atmD14C.D14C,
        color="darkgray",
        ls="-",
        lw=4,
        zorder=0,
    )

    ax[0].plot(
        Hulu["Age [ka BP]"],
        Hulu["Δ14C [‰]"],
        color="#BF840E",
        marker=".",
        ls=" ",
        lw=1,
        zorder=1,
    )

    ax[0].plot(
        tree_ring["kyrBP"],
        tree_ring["∆14C"],
        color="#627A10",
        marker=".",
        linestyle=" ",
        lw=1,
        zorder=2,
    )

    # atmospheric CO2 ice core data
    ax[1].plot(
        icecore_CO2.age_gas_calBP / 1000,
        icecore_CO2.co2_ppm,
        color="#627AD1",
        marker=".",
        linestyle=" ",
        lw=4,
        zorder=0,
    )

    for i in range(2):
        ax[i].tick_params(bottom=True, top=True, left=True, right=True)
        ax[i].set_xlabel("Calendar age (kyr BP)")
        ax[i].set_xlim(0, 20)
    ax[0].set_ylabel("∆$^{14}$C (‰)")

    ax[1].set_ylabel("CO$_2$ (ppm)")
    ax[0].set_ylim(-100, 450)
    ax[1].set_ylim(175, 300)
    plt.tight_layout()
    plt.savefig(figurepath + "robust_atm_data_presentation.pdf", bbox_inches="tight")
    return


def presentation_global_anomalies():
    """This figure compares ∆14C from CYCLOPS after adding 2400 PgC to the marchitto record"""
    fig, ax = plt.subplots(1, figsize=(10, 6))

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.set_ylabel("∆$^{14}$C (‰)")
    ax.set_xlabel("Calendar age (kyr BP)")
    ax.set_xlim((0, 20))

    ax.plot(
        Anomalies[0].year,
        Anomalies[0].D14CintNP,
        marker=markers[0],
        markeredgecolor="k",
        markerfacecolor="orange",
        color="orange",
        ls="--",
        lw=3,
        markersize=8,
        zorder=2,
    )
    ax.plot(
        NP_LC_PF_RC.year_kyrBP,
        NP_LC_PF_RC["intNP_∆14C_permil"],
        color="black",
        ls="-",
        lw=3,
        zorder=3,
    )
    plt.savefig(figurepath + "CYCLOPS_anomaly_presentation.pdf", bbox_inches="tight")


def presentation_fig_methods():
    """this figure progressively goes through the anomalies, how they are interpolated, and how the model simulates the anomalies"""
    return
