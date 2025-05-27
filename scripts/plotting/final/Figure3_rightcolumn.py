"""

Generates a 5-panel plot showing model vs. observational records for ∆14C, CO₂, ∆pH, and cumulative carbon addition
under high and low isolation scenarios. Includes δ11B-constrained ∆pH and IntCal/CO₂ observational overlays.

"""

# === Imports ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import src.functions as f

try:
    import font_setup  # Sets up fonts
    import plot_style  # Applies RG custom plot style
except ImportError:
    pass  # font_setup and/or plot_style modules not found

# === File Paths ===
GoC_modelpath = "results/simulations/"
obs_file_path = "data/observations/d11B_D14C_RAG_cleaned.xlsx"

# === Color Definitions ===
color_low_iso = "#FF2A00"
color_high_iso = "#FFD500"
model_geo_color = "blue"
model_control_color = "red"
observation_color = "black"
linewidths = 2

# === Load Observational Data ===
benthic_df = pd.read_excel(obs_file_path, sheet_name="Benthic", skiprows=1)
planktic_df = pd.read_excel(obs_file_path, sheet_name="Planktic", skiprows=1)

benthic_cal_age = benthic_df["cal.age"] / 1000
benthic_dpH = benthic_df["Delta pH"]
benthic_sigma_pH = benthic_df["1 Sigma pH"]

planktic_cal_age = planktic_df["cal.age"] / 1000
planktic_dpH = planktic_df["Delta pH"]
planktic_sigma_pH = planktic_df["1 Sigma pH"]

# === Load Model Data ===
CYCLOPS_high_iso = pd.read_table("data/model/CoupledRun_high_isolation.txt", sep="\s+", header=None)
CYCLOPS_low_iso = pd.read_table("data/model/CoupledRun_low_isolation_ratio1.txt", sep="\s+", header=None)
CYCLOPS_control = pd.read_table("data/model/ControlRun.txt", sep="\s+", header=None)

GoC_control_high_iso = f.organizedata_goc(pd.read_table(GoC_modelpath + "high_isolation_control.txt", sep="\s+", header=None))
GoC_control_low_iso = f.organizedata_goc(pd.read_table(GoC_modelpath + "low_isolation_control.txt", sep="\s+", header=None))

GoC_high_iso = f.organizedata_goc(pd.read_table(GoC_modelpath + "high_isolation_optimization.txt", sep="\s+", header=None))
GoC_low_iso = f.organizedata_goc(pd.read_table(GoC_modelpath + "low_isolation_optimization.txt", sep="\s+", header=None))

GoC_high_iso_CO2 = f.organizedata_goc(pd.read_table(GoC_modelpath + "high_isolation_optimization_CO2.txt", sep="\s+", header=None))
GoC_low_iso_CO2 = f.organizedata_goc(pd.read_table(GoC_modelpath + "low_isolation_optimization_CO2.txt", sep="\s+", header=None))

# === Compute pH Effects ===
pH_effect_high_iso_subsurface = GoC_high_iso["pH_sub"] - GoC_control_high_iso["pH_sub"]
pH_effect_low_iso_subsurface = GoC_low_iso["pH_sub"] - GoC_control_low_iso["pH_sub"]
pH_effect_high_iso_surface = GoC_high_iso["pH_surf"] - GoC_control_high_iso["pH_surf"]
pH_effect_low_iso_surface = GoC_low_iso["pH_surf"] - GoC_control_low_iso["pH_surf"]
pH_effect_high_iso_CO2_subsurface = GoC_high_iso_CO2["pH_sub"] - GoC_control_high_iso["pH_sub"]
pH_effect_low_iso_CO2_subsurface = GoC_low_iso_CO2["pH_sub"] - GoC_control_low_iso["pH_sub"]

# === Load Additional Observations ===
rafter2024_benthic = pd.read_excel("data/observations/prafter-2024-Gulf-CA-Data-for-Ryan-3.3.xlsx")
rafter2024_wood = rafter2024_benthic[rafter2024_benthic['mat.dated'] == 'terrestrial wood']
rafter2024_benthic = rafter2024_benthic[rafter2024_benthic['mat.dated'] != 'terrestrial wood'].sort_values(by='cal.age')
rafter2024_benthic["year"] = rafter2024_benthic["cal.age"] / 1000
rafter2024_wood["year"] = rafter2024_wood["cal.age"] / 1000

reimer2020_atmD14C = pd.read_csv("data/observations/IntCalSmoothed.txt", header=None)
reimer2020_atmD14C = reimer2020_atmD14C.rename(columns={0: "year", 3: "D14C"})
reimer2020_atmD14C["year"] /= 1000

bereiter2015_atmCO2 = pd.read_csv("data/observations/CO2data1.txt", sep="\t")
bereiter2015_atmCO2["year"] /= 1000

# === Set Up Figure ===
fig, axs = plt.subplots(5, 1, sharex=True, figsize=(3, 7), gridspec_kw={'hspace': 0.2})

# === Panel 1: Δ14C ===
ax0 = axs[0]
ax0.plot(CYCLOPS_high_iso[0]/1000, CYCLOPS_high_iso[5], lw=linewidths, color=color_high_iso)
ax0.plot(CYCLOPS_low_iso[0]/1000, CYCLOPS_low_iso[5], lw=linewidths, color=color_low_iso)
ax0.plot(CYCLOPS_control[0]/1000, CYCLOPS_control[5], lw=linewidths, color='black')
ax0.fill_between(CYCLOPS_high_iso[0]/1000, CYCLOPS_high_iso[5], CYCLOPS_control[5], color=color_high_iso, alpha=0.5)
ax0.fill_between(CYCLOPS_low_iso[0]/1000, CYCLOPS_low_iso[5], CYCLOPS_high_iso[5], color=color_low_iso, alpha=0.5)
ax0.plot(rafter2024_wood["year"], rafter2024_wood["D14C"], 'o', color='tab:brown', markeredgecolor='black', markersize=4, zorder=4)
ax0.plot(reimer2020_atmD14C["year"], reimer2020_atmD14C["D14C"], ls='dashed', color=observation_color, lw=linewidths)
ax0.yaxis.set_label_position("right")
ax0.spines['bottom'].set_visible(False)
ax0.tick_params(bottom=False, right=True, left=True, labelleft=False, labelright=True)

# === Panel 2: CO₂ ===
ax1 = axs[1]
ax1.plot(CYCLOPS_high_iso[0]/1000, CYCLOPS_high_iso[4], lw=linewidths, color=color_high_iso)
ax1.plot(CYCLOPS_low_iso[0]/1000, CYCLOPS_low_iso[4], lw=linewidths, color=color_low_iso)
ax1.plot(CYCLOPS_control[0]/1000, CYCLOPS_control[4], lw=linewidths, color='black')
ax1.fill_between(CYCLOPS_high_iso[0]/1000, CYCLOPS_high_iso[4], CYCLOPS_control[4], color=color_high_iso, alpha=0.5)
ax1.fill_between(CYCLOPS_low_iso[0]/1000, CYCLOPS_low_iso[4], CYCLOPS_high_iso[4], color=color_low_iso, alpha=0.5)
ax1.plot(bereiter2015_atmCO2["year"], bereiter2015_atmCO2["CO2"], ls='dashed', color=observation_color, lw=linewidths)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.tick_params(top=False, bottom=False, labelbottom=False, right=True, left=True)

# === Panel 3: ∆pH ===
ax2 = axs[2]
ax2.errorbar(benthic_cal_age, benthic_dpH, yerr=benthic_sigma_pH, fmt='s', color='#FF7F00', markeredgecolor='black', capsize=3, markersize=4, zorder=4)
ax2.errorbar(planktic_cal_age, planktic_dpH, yerr=planktic_sigma_pH, fmt='^', color='#FF7F00', markeredgecolor='black', capsize=3, markersize=4, zorder=4)
ax2.plot(GoC_high_iso["year"], pH_effect_high_iso_subsurface, lw=linewidths, color=color_high_iso)
ax2.plot(GoC_low_iso["year"], pH_effect_low_iso_subsurface, lw=linewidths, color=color_low_iso)
ax2.plot(GoC_low_iso_CO2["year"], pH_effect_low_iso_CO2_subsurface, lw=linewidths, ls=':', color=color_low_iso)
ax2.set_ylim(-0.075, 0.075)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.tick_params(top=False, bottom=False, labelbottom=False, right=True, left=False)
ax2.yaxis.set_label_position("right")

# === Panel 4: pH zoomed out ===
ax3 = axs[3]
ax3.plot(GoC_low_iso_CO2["year"], pH_effect_low_iso_CO2_subsurface, lw=linewidths, ls=':', color=color_low_iso)
ax3.set_ylim(-1.5, -0.25)
ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.tick_params(top=False, bottom=False, labelbottom=False, right=True, left=False)
ax3.yaxis.set_label_position("right")

# === Panel 5: Cumulative Carbon Addition ===
ax4 = axs[4]
time = GoC_high_iso["year"]
cumC_low = GoC_low_iso[["Ccum_surf", "Ccum_sub", "Ccum_mar"]].sum(axis=1)
cumC_high = GoC_high_iso[["Ccum_surf", "Ccum_sub", "Ccum_mar"]].sum(axis=1)
ax4.plot(time, cumC_low, label='Low isolation', color=color_low_iso, lw=linewidths)
ax4.plot(time, cumC_high, label='High isolation', color=color_high_iso, lw=linewidths)
ax4.axhline(y=850, color='black', ls=':', lw=2)
ax4.axhline(y=2500, color='black', ls=':', lw=2)
ax4.set_xlim(10, 20)
ax4.spines['top'].set_visible(False)
ax4.tick_params(top=False)

# === Shaded Regions ===
fig.patches.append(plt.Rectangle((0.25, 0.05), 0.1, 0.9, transform=fig.transFigure, color="darkgray", alpha=0.1, zorder=0))
fig.patches.append(plt.Rectangle((0.475, 0.05), 0.27, 0.9, transform=fig.transFigure, color="darkgray", alpha=0.1, zorder=0))

# === Legend ===
from matplotlib.lines import Line2D
obs_legend = Line2D([0], [0], color='black',ls='dashed', lw=2, label=' ')
control_legend = Line2D([0], [0], color='black', lw=2, ls='solid', label=' ')
high_iso_legend = Line2D([0], [0], color=color_high_iso, lw=2, label=' ')
low_iso_legend = Line2D([0], [0], color=color_low_iso, lw=2, label=' ')
wood_legend = Line2D([0], [0], color='tab:brown', marker='o', markersize=5, lw=0, label='Wood $\Delta^{14}$C')
d11B_legend = Line2D([0], [0], color='black', marker='s', markersize=5, lw=0, label='$\Delta$pH derived from $\delta$$^{11}$B')

ax4.legend(handles=[obs_legend, control_legend, high_iso_legend, low_iso_legend], loc='upper right', frameon=False, fontsize=12)

# === Final Touches ===
plt.subplots_adjust(top=0.95, bottom=0.05)
# plt.savefig("results/figures/Figure3_rightcolumn.pdf", bbox_inches='tight')
plt.show()