"""

Generates a 3-panel plot comparing modeled and observed Δ14C values in the Gulf of California under
low and high isolation scenarios. Overlays carbon release rates on secondary axes.

"""

# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import src.functions as f

# Optional custom styling
try:
    import font_setup  # Sets up fonts
    import plot_style  # Applies RG custom plot style
except ImportError:
    pass  # Use default settings if modules aren't found

# === File Paths ===
obspath = "data/observations/"
GoC_modelpath = "results/simulations/"
figurepath = "results/figures/"

# === Load & Preprocess Model Data ===
GoC_high_iso = pd.read_table(GoC_modelpath + "high_isolation_optimization.txt", header=None, sep="\s+")
GoC_low_iso = pd.read_table(GoC_modelpath + "low_isolation_optimization.txt", header=None, sep="\s+")
GoC_high_iso = f.organizedata_goc(GoC_high_iso)
GoC_low_iso = f.organizedata_goc(GoC_low_iso)

for col in ['Crate_surf', 'Crate_sub', 'Crate_mar']:
    GoC_low_iso[col] = np.where(GoC_low_iso[col] == 0, np.nan, GoC_low_iso[col])
    GoC_high_iso[col] = np.where(GoC_high_iso[col] == 0, np.nan, GoC_high_iso[col])

# === Load Observational Data ===
rafter2024_benthic = pd.read_excel(obspath + "D14C_PP.xlsx")
rafter2024_benthic = rafter2024_benthic[rafter2024_benthic['mat.dated'] != 'terrestrial wood']
rafter2024_benthic = rafter2024_benthic.sort_values(by='cal.age')

def process_rafter_planktic_data(file_path):
    df = pd.read_csv(file_path, sep="\t", header=24)
    df = df[df["Habitat"] == "planktic"]
    df["Cal age [ka BP]"] *= 1000
    return df.sort_values(by="Cal age [ka BP]")

rafter2018_planktic = process_rafter_planktic_data(obspath + "Rafter_2019.tab")
marchitto2007_benthic = pd.read_csv(obspath + "Marchitto.txt", sep="\s+")
marchitto2007_benthic["Cal.Age"] = 1000 * marchitto2007_benthic["Cal.Age"]

# === Standardize Column Names ===
marchitto2007_benthic = marchitto2007_benthic.rename(columns={"Cal.Age": "year", "D14C": "D14CintNP"})
rafter2018_planktic = rafter2018_planktic.rename(columns={"Cal age [ka BP]": "year", "Δ14C [‰]": "D14CintNP"})
rafter2024_benthic = rafter2024_benthic.rename(columns={"cal.age": "year", "D14C": "D14CintNP"})

# Convert to kyr
GoCobs = [marchitto2007_benthic, rafter2018_planktic, rafter2024_benthic]
for df in GoCobs:
    df["year"] /= 1000

# === Plot Colors ===
color_low_isolation = "#FF2A00"
color_high_isolation = "#FFD500"

# === Plot Setup ===
fig, ax = plt.subplots(3, 1, figsize=(3.25, 7), sharex=True)
twin_axes = [ax[i].twinx() for i in range(3)]
ytick_loc = [0, 0.3, 0.6, 0.9]

# Layering and aesthetics
for i in range(3):
    ax[i].set_zorder(2)
    twin_axes[i].set_zorder(1)
    ax[i].patch.set_visible(False)
    ax[i].tick_params(bottom=True, top=True, left=True, right=False)
    ax[i].set_ylim(-450, 350)
    ax[i].axvspan(11.6, 12.9, alpha=0.1, color="darkgray", zorder=0)
    ax[i].axvspan(14.5, 18, alpha=0.1, color="darkgray", zorder=0)
    twin_axes[i].set_ylim(0, 0.15)
    twin_axes[i].set_yticks(ytick_loc)
    twin_axes[i].tick_params(bottom=False, top=False, left=False, right=True)
    twin_axes[i].set_yticklabels(ytick_loc)

# === Plot Data ===
# Panel 1: Surface
ax[0].plot(rafter2018_planktic["year"], rafter2018_planktic["D14CintNP"],
           marker="^", markeredgecolor="k", markerfacecolor="white", linestyle="dashed",
           color="black", label="Rafter et al. 2019-GoC surface", markersize=4, lw=2)
ax[0].plot(GoC_low_iso.year, GoC_low_iso.D14C_surf, linestyle="solid", color=color_low_isolation, lw=2)
ax[0].plot(GoC_high_iso.year, GoC_high_iso.D14C_surf, linestyle="solid", color=color_high_isolation, lw=2)

# Panel 2: Subsurface
ax[1].plot(rafter2024_benthic["year"], rafter2024_benthic["D14CintNP"],
           marker="s", markeredgecolor="k", markerfacecolor="white", linestyle="dashed",
           color="black", label="Rafter et al. 2024-benthic", markersize=4, lw=2)
ax[1].plot(GoC_low_iso.year, GoC_low_iso.D14C_sub, linestyle="solid", color=color_low_isolation, lw=2)
ax[1].plot(GoC_high_iso.year, GoC_high_iso.D14C_sub, linestyle="solid", color=color_high_isolation, lw=2)

# Panel 3: Marchitto box
ax[2].plot(marchitto2007_benthic["year"], marchitto2007_benthic["D14CintNP"],
           marker="o", markeredgecolor="k", markerfacecolor="white", linestyle="dashed",
           color="black", label="Marchitto et al. 2007-benthic", markersize=4, lw=2)
ax[2].plot(GoC_low_iso.year, GoC_low_iso.D14C_mar, linestyle="solid", color=color_low_isolation, lw=2)
ax[2].plot(GoC_high_iso.year, GoC_high_iso.D14C_mar, linestyle="solid", color=color_high_isolation, lw=2)

# === Plot Release Rates ===
twin_axes[0].plot(GoC_low_iso.year, GoC_low_iso.Crate_surf, color=color_low_isolation)
twin_axes[0].bar(GoC_low_iso.year, GoC_low_iso.Crate_surf, width=0.1, alpha=0.7, color=color_low_isolation)
twin_axes[0].plot(GoC_high_iso.year, GoC_high_iso.Crate_surf, color=color_high_isolation)
twin_axes[0].bar(GoC_high_iso.year, GoC_high_iso.Crate_surf, width=0.1, alpha=0.7, color=color_high_isolation)

twin_axes[1].plot(GoC_low_iso.year, GoC_low_iso.Crate_sub, color=color_low_isolation)
twin_axes[1].bar(GoC_low_iso.year, GoC_low_iso.Crate_sub, width=0.1, alpha=0.7, color=color_low_isolation)
twin_axes[1].plot(GoC_high_iso.year, GoC_high_iso.Crate_sub, color=color_high_isolation)
twin_axes[1].bar(GoC_high_iso.year, GoC_high_iso.Crate_sub, width=0.1, alpha=0.7, color=color_high_isolation)

twin_axes[2].plot(GoC_low_iso.year, GoC_low_iso.Crate_mar, color=color_low_isolation)
twin_axes[2].bar(GoC_low_iso.year, GoC_low_iso.Crate_mar, width=0.1, alpha=0.7, color=color_low_isolation)
twin_axes[2].plot(GoC_high_iso.year, GoC_high_iso.Crate_mar, color=color_high_isolation)
twin_axes[2].bar(GoC_high_iso.year, GoC_high_iso.Crate_mar, width=0.1, alpha=0.7, color=color_high_isolation)

# === Aesthetic Adjustments ===
ax[2].set_xlim(0, 20)
for i, axs in enumerate(twin_axes):
    ax[i].set_zorder(axs.get_zorder() + 1)
    ax[i].set_frame_on(False)

# === Legend ===
from matplotlib.lines import Line2D

observations_legend = Line2D([0], [0], color='black', lw=2, linestyle='--',
                             marker='s', markerfacecolor='white', markeredgecolor='black', markersize=5)
ax[2].legend(handles=[observations_legend], loc='lower left', fontsize=12, frameon=False)

# === Finalize Figure ===
plt.tight_layout()
# plt.savefig(figurepath + "Figure3_leftcolumn.pdf", bbox_inches="tight")
plt.show()