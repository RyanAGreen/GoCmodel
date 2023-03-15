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
ISpath = "data/ISchange/"
NoISpath = "data/NoISchange/"
figurepath = "results/Figures/"

plt.rcParams["font.weight"] = "bold"


def pH_change(d0, d1):
    return delta_pKb - np.log10(
        1
        + (d1 - d0)
        / (dSW - alpha * d1 - epsilon)
        * ((alpha - 1) * dSW - epsilon / (d0 - dSW))
    )


if "reading in ∆14C, CO2, d13c,d11b, CO3, and OCIM data":
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

    # d13C observations
    d13C_obs = pd.read_excel(obspath + "d13C_GoC_benthic_LPAZ21P.xlsx")
    d13C_obs = d13C_obs.sort_values(by=["cal.age"])

    # d11B pH change data
    d11B_obs = pd.read_excel("data/observations/prafter-2022-12-21-LPAZ21P-d11B.xlsx")
    # Constants
    dSW = 39.61  # delta of modern SW, per mille
    alpha = 1.0272  # from Hain et al 2018
    epsilon = 27.2  # from Hain et al 2018 per mille
    delta_pKb = 0  # from Hain et al 2018, assume no change in pkb
    d0 = d11B_obs["d11B"].iloc[10]  # starting around 20 kyr BP
    pH_changes_obs = []
    for i in range(11):
        pH_changes_obs.append(pH_change(d0, d11B_obs["d11B"].iloc[i]))

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
    # read in CYCLOPS data
    control = pd.read_table(
        NoISpath + "ForwardRun/ControlRun.txt", header=None, sep="\s+"
    )
    control = f.organizedata(control)
    IScontrol = pd.read_table(
        ISpath + "ForwardRun/ControlRun.txt", header=None, sep="\s+"
    )
    IScontrol = f.organizedata(IScontrol)
    ex4 = pd.read_table(
        ISpath + "2Dinversion/Powell2Dinversion.txt", header=None, sep="\s+"
    )

    ex4 = f.organizedata(ex4)

    # read in GoC data
    GoC = pd.read_table(
        "results/optimizedrun_CO2carbonate_source.txt", header=None, sep="\s+"
    )
    GoC = f.organizedata_goc(GoC)


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
        ax[i].tick_params(axis="y", labelsize="large")
        ax[i].tick_params(axis="x", labelsize="large")
        ax[i].axvspan(11.6, 12.9, alpha=0.4, color="darkgray", zorder=0)
        ax[i].axvspan(14.5, 18, alpha=0.4, color="darkgray", zorder=0)
        ax[i].tick_params(bottom=True, top=True, left=True, right=True)
        ax[i].tick_params(axis="both", direction="in", length=7, width=2, color="black")

    ### atmospheric CO2 ###

    # observations
    ax[0].plot(
        CO2obs.year,
        CO2obs.CO2,
        color=color_global_obs,
        marker="o",
        markersize=1,
        ls=" ",
        lw=2,
        zorder=0,
    )
    ax[0].plot(
        CO2obs.year, CO2obs.CO2, color=color_global_obs, ls=":", lw=2, zorder=0,
    )
    # models
    ax[0].plot(ex4.year, ex4.CO2, color=color_global_model, lw=2)
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
        D14C.year, D14C.D14C, color=color_global_obs, ls=":", lw=2, zorder=0,
    )
    # models
    ax[1].plot(ex4.year, ex4.D14C, color=color_global_model, lw=2)
    # labels
    ax[1].set_ylabel(
        "∆$^{14}$C$^{atm}$ (‰)", fontsize=font_label_size, fontweight="bold"
    )

    ### Deep ocean CO3 ###

    # models
    ax[2].plot(
        ex4.year,
        ex4.AtlCO3,
        color=color_atlantic,
        alpha=1,
        linestyle="-",
        lw=2,
        label="Atlantic",
    )
    ax[2].plot(
        ex4.year,
        (ex4.NPacCO3 + ex4.SPacCO3 + ex4.IndCO3) / 3,
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
        d11B_obs["cal.age.kyr"].iloc[:11],
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
        1.02, 0.68, "Global constraints", va="center", rotation="vertical", fontsize=10
    )
    fig.text(
        1.02, 0.4, "Regional constraints", va="center", rotation="vertical", fontsize=10
    )

    if "making legends":

        # make legends
        legend_global_model = mlines.Line2D(
            [], [], color=color_global_model, linestyle="solid", lw=2.5, label="Model",
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
            [], [], color=color_global_obs, linestyle=":", lw=2, label="CO$_2^{obs}$",
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
            color=color_marchitto, linestyle="solid", lw=2.5, label="Marchitto box",
        )
        legend_subsurface = mpatches.Patch(
            color=color_subsurface,
            linestyle="solid",
            lw=2.5,
            label="GoC subsurface box",
        )
        legend_surface = mpatches.Patch(
            color=color_surface, linestyle="solid", lw=2.5, label="GoC surface box",
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
            handles=[legend_atlantic, legend_indopac], loc="upper left", frameon=True,
        )
        ax[5].legend(
            handles=[legend_surface, legend_subsurface, legend_marchitto],
            ncol=1,
            loc="upper left",
            frameon=False,
        )

    plt.savefig(figurepath + "Figure3.pdf", bbox_inches="tight")
    return


def Figure4():
    """plotting d13C model output vs observations"""
    fig, ax = plt.subplots(1)

    # thickening axis
    for axis in ["top", "left", "right", "bottom"]:
        ax.spines[axis].set_linewidth(3)
    ax.tick_params(axis="y", labelsize="large")
    ax.tick_params(axis="x", labelsize="large")
    # ax[0].set_yticklabels([])

    ax.set_ylabel("∆$\delta^{13}$C (‰)", fontsize=20, fontweight="bold")
    ax.set_xlabel("Calendar age (kyr BP)", fontsize=20, fontweight="bold")
    ax.set_xlim((0, 20))

    # ax.plot(
    #     GoC.year, GoC.d13C_mar - GoC.d13C_mar[0], ls="-", lw=4, zorder=0,
    # )
    ax.plot(
        GoC.year, GoC.d13C_sub - GoC.d13C_sub[0], ls="-", lw=4, zorder=0,
    )
    # ax.plot(
    #     GoC.year, GoC.d13C_surf - GoC.d13C_surf[0], ls="-", lw=4, zorder=0,
    # )
    ax.plot(
        d13C_obs["cal.age"],
        d13C_obs["δ¹³C (‰, VPDB)"] - d13C_obs["δ¹³C (‰, VPDB)"][14],
        marker="s",
        markeredgecolor="k",
        markerfacecolor="white",
        linestyle="dashed",
        color="#B57114",
        label="benthic d13C data",
    )

    # ax.legend(
    #     loc="lower left", fontsize=10, ncol=1, frameon=False
    # )  # , bbox_to_anchor=(1, 0.96)

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis="both", direction="in", length=7, width=3, color="black")

    ax.set_xlim(0, 20)

    plt.savefig(
        "/home/rygreen/GoCmodel/results/Figures/Figure4.png", bbox_inches="tight"
    )
    return
