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
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import netCDF4 as nc4

obspath = "data/observations/"
ISpath = "data/ISchange/"
NoISpath = "data/NoISchange/"
figurepath = "results/Figures/"

plt.rcParams["font.weight"] = "bold"

if "reading in ∆14C, CO2, CO3, and OCIM data":
    Rafter_subsurface = pd.read_excel(
        obspath + "prafter-2019-Gulf-CA-Data-for-Ryan.xls"
    )
    # Based on Pats instructions -> we use P.ariminensis and U.peregrina, which appear to be the most consistent.
    # both benthic species
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
    Rafter_subsurface["calendar age [kyr BP]"] = (
        1000 * Rafter_subsurface["calendar age [kyr BP]"]
    )

    # Rafter planktic
    Rafter_surface = pd.read_csv(obspath + "Rafter_2019.tab", sep="\t", header=24)
    Rafter_surface.loc[(Rafter_surface["Habitat"] == "planktic")]
    Rafter_surface["Cal age [ka BP]"] = 1000 * Rafter_surface["Cal age [ka BP]"]
    Rafter_surface = Rafter_surface.sort_values(by=["Cal age [ka BP]"])

    # CO2 observations EPICA
    CO2obs = pd.read_csv(obspath + "CO2data1.txt", sep="\t")
    CO2obs["year"] = CO2obs["year"] / 1000

    # ∆14C observations IntCal
    d14C = pd.read_csv(obspath + "IntCalSmoothed.txt", header=None)
    D14C = d14C.rename(columns={0: "year", 3: "D14C"})
    D14C["year"] = D14C["year"] / 1000

    # Chen ∆14C
    chen = pd.read_csv(obspath + "Chen2020.txt", sep="\t", header=0, skiprows=110)
    Chen = chen[chen["water.depth"] == 627]

    # Marchitto ∆14C
    Mar = pd.read_csv(obspath + "Marchitto.txt", sep="\s+")
    Mar["Cal.Age"] = 1000 * Mar["Cal.Age"]

    # Stott ∆14C
    Stott = chen[chen["water.depth"] == 617]
    Stott = Stott[Stott["ref."] == "Stott et al. (2009)"]

    # Rafter ∆14C compilation
    Rafter_compilation = pd.read_csv(obspath + "csv-D14Cbin_500bin_Pacific_mid.csv")
    t = Rafter_compilation.int_age / 1000
    m = Rafter_compilation.loess_fit
    s = Rafter_compilation.loess_se

    # Order -> 4 anoamlies (don't behave), then 2 parallel records (behave)
    GoCobs = [Mar, Stott, Rafter_surface, Rafter_subsurface, Chen, Rafter_compilation]

    # make sure the columns are named the same thing
    GoCobs[0] = GoCobs[0].rename(columns={"Cal.Age": "year", "D14C": "D14CintNP",})
    GoCobs[1] = GoCobs[1].rename(
        columns={"cal.age": "year", "benthic.D14C": "D14CintNP"}
    )
    GoCobs[2] = GoCobs[2].rename(
        columns={"Cal age [ka BP]": "year", "Δ14C [‰]": "D14CintNP"}
    )
    GoCobs[3] = GoCobs[3].rename(
        columns={"calendar age [kyr BP]": "year", "D14C": "D14CintNP",}
    )
    GoCobs[4] = GoCobs[4].rename(
        columns={"cal.age": "year", "benthic.D14C": "D14CintNP"}
    )
    GoCobs[5] = GoCobs[5].rename(columns={"int_age": "year", "loess_fit": "D14CintNP"})

    for i in range(6):
        GoCobs[i] = GoCobs[i][GoCobs[i]["year"] < 20000]
        GoCobs[i]["year"] = GoCobs[i]["year"] / 1000

    Anomalies = [GoCobs[0], GoCobs[1], GoCobs[2], GoCobs[3]]

    # CO3 observational data
    indian_obs = pd.read_table(
        obspath + "WIND28K_Yuetal2010.txt", sep="\t", header=None
    )
    indian_obs = indian_obs.drop(labels=0, axis=1)
    indian_obs = indian_obs.rename(columns={1: "time", 2: "CO3"})
    atlantic_obs = pd.read_table(
        obspath + "BOFS8K_Yuetal2008.txt", sep="\t", header=None
    )
    atlantic_obs = atlantic_obs.rename(columns={0: "time", 1: "CO3"})
    pacific_obs = pd.read_fwf(obspath + "GGC48_Yuetal2010.txt", sep="\t", header=None)
    pacific_obs = pacific_obs.drop(
        labels=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12], axis=1
    )
    pacific_obs = pacific_obs.rename(columns={3: "time", 13: "CO3"})

    # OCIM
    OCIM = pd.read_csv(obspath + "14C-Pac-OCIM-North-and-ETNP.csv", skiprows=1)
    OCIM["cal.age"] = OCIM["cal.age"] / 1000

if "reading in CYCLOPS and GoC output":
    # read in model data
    control = pd.read_fwf(
        NoISpath + "ForwardRun/ControlRun.txt", header=None, infer_nrows=1000
    )
    control = f.organizedata(control)
    IScontrol = pd.read_fwf(
        ISpath + "ForwardRun/ControlRun.txt", header=None, infer_nrows=1000
    )
    IScontrol = f.organizedata(IScontrol)
    # ex1 = pd.read_table(
    #     NoISpath + "2Dinversion/Powell2Dinversion.txt", header=None, sep="\s+",
    # )
    # ex2 = pd.read_table(
    #     NoISpath + "2Dinversion/Experiment2/Powell2Dinversion.txt",
    #     header=None,
    #     sep="\s+",
    # )
    # ex3 = pd.read_table(
    #     NoISpath + "2Dinversion/Experiment3/Powell2Dinversion.txt",
    #     header=None,
    #     sep="\s+",
    # )
    ex4 = pd.read_table(
        ISpath + "2Dinversion/Powell2Dinversion.txt", header=None, sep="\s+"
    )

    # ex1 = f.organizedata(ex1)
    # ex2 = f.organizedata(ex2)
    # ex3 = f.organizedata(ex3)
    ex4 = f.organizedata(ex4)


def Figure1():

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
    ax.fill_between(t, m - s, m + s, color="teal", alpha=0.5)
    # atmospheric observations
    ax.plot(
        D14C.year,
        D14C.D14C,
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


def Figure3():
    """
    plotting global and regional results against observations
    atmospheric CO2, ∆14C, deep ocean [CO3], regional pH, regional ∆14C, rate of carbon additon?
    """
    color_anomaly = "#44AA99"
    color_parallel = "#8A0C0D"
    color_model = "#6699A2"

    atlantic_color = "#1763F9"
    indo_pac_color = "#BF840E"
    markers = ["o", "v", "s", "<", "P"]

    fig, ax = plt.subplots(6, 1, figsize=(3, 15), sharex=True)

    for i in range(6):
        # thickening axis
        for axis in ["top", "left", "right", "bottom"]:
            ax[i].spines[axis].set_linewidth(3)
        ax[i].tick_params(axis="y", labelsize="large")
        ax[i].tick_params(axis="x", labelsize="large")
        ax[i].tick_params(bottom=True, top=True, left=True, right=True)
        ax[i].tick_params(axis="both", direction="in", length=7, width=2, color="black")

    ### atmospheric CO2 ###

    # observations
    ax[0].plot(
        CO2obs.year, CO2obs.CO2, color="darkgray", ls="-", lw=4, zorder=0,
    )
    # models
    ax[0].plot(ex4.year, ex4.CO2)
    # labels
    ax[0].set_ylabel("CO$_{2}$$^{atm}$ (ppm)", fontsize=8, fontweight="bold")

    ### atmospheric D14C ###

    # observations
    ax[1].plot(
        D14C.year, D14C.D14C, color="darkgray", ls="-", lw=4, zorder=0,
    )
    # models
    ax[1].plot(ex4.year, ex4.D14C)
    # labels
    ax[1].set_ylabel("∆$^{14}$C$^{atm}$ (‰)", fontsize=8, fontweight="bold")

    ### Deep ocean CO3 ###

    # models
    ax[2].plot(
        ex4.year,
        ex4.AtlCO3,
        color=atlantic_color,
        alpha=1,
        linestyle="-",
        lw=2,
        label="Atlantic",
    )
    ax[2].plot(
        ex4.year,
        (ex4.NPacCO3 + ex4.SPacCO3 + ex4.IndCO3) / 3,
        color=indo_pac_color,
        linestyle="-",
        lw=2,
        label="Indo-pacific",
    )
    # observations
    # ax[2].plot(
    #     atlantic_obs["time"],
    #     atlantic_obs["CO3"],
    #     color=color_parallel,
    #     alpha=0.5,
    #     lw=1,
    #     marker="o",
    #     markeredgecolor=atlantic_color,
    #     markerfacecolor=atlantic_color,
    #     ls="None",
    #     markersize=5,
    #     zorder=0,
    #     label="BOFS 8K N. Atlantic",
    # )
    # ax[2].plot(
    #     indian_obs["time"],
    #     indian_obs["CO3"],
    #     color=color_parallel,
    #     alpha=0.5,
    #     lw=1,
    #     marker="^",
    #     markeredgecolor=indo_pac_color,
    #     markerfacecolor=indo_pac_color,
    #     ls="None",
    #     markersize=5,
    #     zorder=0,
    #     label="WIND 28K Indian",
    # )
    # ax[2].plot(
    #     pacific_obs["time"],
    #     pacific_obs["CO3"],
    #     color=indo_pac_color,
    #     alpha=0.5,
    #     lw=1,
    #     marker="s",
    #     markeredgecolor=indo_pac_color,
    #     markerfacecolor=indo_pac_color,
    #     ls="None",
    #     markersize=5,
    #     zorder=0,
    #     label="GGC48 EQ Pacific",
    # )
    # labels
    ax[2].set_ylabel("[CO$_{3}$$^{2-}$]", fontsize=8, fontweight="bold")

    ### pH ###
    # ax[3].plot()
    ax[3].set_ylabel("pH", fontsize=8, fontweight="bold")

    ### ETNP ∆14C ###
    ax[4].plot(
        ex4.year,
        ex4.D14CintNP,
        linewidth=2.5,
        ls="-",
        color=color_model,
        label="NP+LC+PF+RC",
    )
    ax[4].plot(
        t,
        m,
        linewidth=1,
        ls="solid",
        color=color_parallel,
        label="Mid-depth ∆$^{14}$C compilation",
    )
    ax[4].plot(
        OCIM["cal.age"],
        OCIM["D14C.Shallow.Pacific.North.East"],
        linewidth=2,
        ls="-",
        color="k",
        label="OCIM model",
    )
    for i in range(len(Anomalies)):
        ax[4].plot(
            Anomalies[i].year,
            Anomalies[i].D14CintNP,
            marker=markers[i],
            markeredgecolor="k",
            markerfacecolor=color_anomaly,
            color=color_anomaly,
            ls="solid",
            lw=1,
            markersize=8,
            zorder=2,
        )

    # Chen
    ax[4].plot(
        GoCobs[4].year,
        GoCobs[4].D14CintNP,
        marker=markers[4],
        markeredgecolor="k",
        markerfacecolor=color_parallel,
        color=color_parallel,
        ls="solid",
        lw=1,
        markersize=10,
        zorder=2,
    )
    # Rafter Compilation
    ax[4].plot(
        GoCobs[5].year,
        GoCobs[5].D14CintNP,
        linestyle="solid",
        color=color_parallel,
        lw=2.5,
        zorder=0,
    )
    ax[4].fill_between(t, m - s, m + s, color=color_parallel, alpha=0.5)
    ax[4].set_ylabel("ETNP ∆$^{14}$C (‰)", fontsize=8, fontweight="bold")

    # ax[5].plot()
    ax[5].set_ylabel("Carbon rate of release", fontsize=8, fontweight="bold")
    ax[5].set_xlabel("Calendar age (kyr BP)", fontsize=8, fontweight="bold")
    ax[5].set_xlim(0, 20)

    plt.savefig(figurepath + "Figure2.png", bbox_inches="tight")
    return

