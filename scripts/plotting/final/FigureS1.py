# Import necessary libraries
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  # For creating proxy artists for the legend
import numpy as np
import pandas as pd
import src.functions as f

try:
    import font_setup  # Sets up fonts
    import plot_style  # Applies RG custom plot style
except ImportError:
    pass  # font_setup and/or plot_style modules not found. Using default settings.

# File paths
GoC_modelpath = "/home/rygreen/GoCmodel/results/simulations/"
obspath = "/home/rygreen/GoCmodel/data/observations/"

# Colors for plotting
color_CO2_addition = "#FF2A00"
color_HCO3_addition = "#FFD500"
color_obs = "#FF7F00"

marker_surf = "^"
markert_sub = "s"

# Load d11B dpH data
d11B_data = pd.read_excel(obspath + "d11B_dpH_benthic_planktic_cleaned_RAG.xlsx")
benthic_dpH = d11B_data['benthic_dpH']
benthic_age = d11B_data['benthic_cal_age'] / 1000  # Age in thousands of years
planktic_dpH = d11B_data['planktic_dpH']
planktic_age = d11B_data['planktic_cal_age'] / 1000  # Age in thousands of years

# Load d13C data for benthic forams in GoC
d13C_benthic_GoC = pd.read_excel(obspath + "prafter-2024-Gulf-CA-Data-for-Ryan-2.0.xls", skiprows=1)
d13C_benthic_GoC = d13C_benthic_GoC[d13C_benthic_GoC["P.ariminensis d13C"].notna()]
d13C_benthic_GoC = d13C_benthic_GoC.sort_values(by=["calendar age [yr BP]"])
LGM_value = d13C_benthic_GoC["P.ariminensis d13C"].iloc[-1]
d13C_benthic_GoC["P.ariminensis d13C diff"] = d13C_benthic_GoC["P.ariminensis d13C"] - LGM_value

# Load model data
GoC_low_iso_d13C_negative_89 = pd.read_table(GoC_modelpath + "low_isolation_control_CO2_d13c--8.9.txt", header=None, sep="\s+")
GoC_low_iso_d13C_negative_89 = f.organizedata_goc(GoC_low_iso_d13C_negative_89)

GoC_control_high_isolation = pd.read_table(GoC_modelpath + "high_isolation_control.txt", header=None, sep="\s+")
GoC_control_high_isolation = f.organizedata_goc(GoC_control_high_isolation)

GoC_control_low_isolation = pd.read_table(GoC_modelpath + "low_isolation_control.txt", header=None, sep="\s+")
GoC_control_low_isolation = f.organizedata_goc(GoC_control_low_isolation)

# Load Gulf of California data adding bicarbonate (these use d13C of -2.5)
GoC_high_iso = pd.read_table(GoC_modelpath + "high_isolation_optimization.txt", header=None, sep="\s+")
GoC_high_iso = f.organizedata_goc(GoC_high_iso)
GoC_low_iso = pd.read_table(GoC_modelpath + "low_isolation_optimization.txt", header=None, sep="\s+")
GoC_low_iso = f.organizedata_goc(GoC_low_iso)

pH_effect_high_iso_subsurface = (GoC_high_iso["pH_sub"] - GoC_control_high_isolation["pH_sub"])
pH_effect_low_iso_subsurface = (GoC_low_iso["pH_sub"] - GoC_control_low_isolation["pH_sub"])

pH_effect_high_iso_surface = (GoC_high_iso["pH_surf"] - GoC_control_high_isolation["pH_surf"])
pH_effect_low_iso_surface = (GoC_low_iso["pH_surf"] - GoC_control_low_isolation["pH_surf"])

# Load Gulf of California data adding CO2
GoC_high_iso_CO2 = pd.read_table(GoC_modelpath + "high_isolation_optimization_CO2.txt", header=None, sep="\s+")
GoC_high_iso_CO2 = f.organizedata_goc(GoC_high_iso_CO2)
GoC_low_iso_CO2 = pd.read_table(GoC_modelpath + "low_isolation_optimization_CO2.txt", header=None, sep="\s+")
GoC_low_iso_CO2 = f.organizedata_goc(GoC_low_iso_CO2)

pH_effect_high_iso_subsurface_CO2 = (GoC_high_iso_CO2["pH_sub"] - GoC_control_high_isolation["pH_sub"])
pH_effect_low_iso_subsurface_CO2 = (GoC_low_iso_CO2["pH_sub"] - GoC_control_low_isolation["pH_sub"])

pH_effect_high_iso_surface_CO2 = (GoC_high_iso_CO2["pH_surf"] - GoC_control_high_isolation["pH_surf"])
pH_effect_low_iso_surface_CO2 = (GoC_low_iso_CO2["pH_surf"] - GoC_control_low_isolation["pH_surf"])

# load CYCLOPS to show CO2 effects
CYCLOPS_high_iso_CO2 = pd.read_table("data/model/CoupledRun_high_isolation_CO2.txt", sep="\s+", header=None)
CYCLOPS_high_iso_HCO3 = pd.read_table("data/model/CoupledRun_high_isolation.txt", sep="\s+", header=None)
CYCLOPS_low_iso = pd.read_table("data/model/CoupledRun_low_isolation_ratio1.txt", sep="\s+", header=None)
CYCLOPS_control = pd.read_table("data/model/ControlRun.txt", sep="\s+", header=None)
bereiter2015_atmCO2 = pd.read_csv("data/observations/CO2data1.txt", sep="\t")
bereiter2015_atmCO2["year"] = bereiter2015_atmCO2["year"] / 1000


# Set up figure and axes
fig, axs = plt.subplots(3, 1, figsize=(3.5, 7), constrained_layout=True, sharex=True)

# Subplot 1: Atmospheric CO2 over time (HCO3- vs CO2 addition)
axs[0].plot(CYCLOPS_high_iso_HCO3[0]/1000, CYCLOPS_high_iso_HCO3[4], lw=2, color=color_HCO3_addition, label="High Isolation (HCO3-)")
axs[0].plot(CYCLOPS_high_iso_CO2[0]/1000, CYCLOPS_high_iso_CO2[4], lw=2, color=color_CO2_addition, label="High Isolation (CO2)")
# axs[0].plot(CYCLOPS_control[0]/1000, CYCLOPS_control[4], lw=2, color='black', label="Control")
# axs[0].legend(frameon=False)
# ax0.fill_between(CYCLOPS_high_iso[0]/1000, CYCLOPS_high_iso[5], CYCLOPS_control[5], color=color_high_iso, alpha=0.5)
# ax0.fill_between(CYCLOPS_low_iso[0]/1000, CYCLOPS_low_iso[5], CYCLOPS_high_iso[5], color=color_low_iso, alpha=0.5)
axs[0].plot(bereiter2015_atmCO2["year"], bereiter2015_atmCO2["CO2"],marker='o', markersize=5, markerfacecolor=color_obs, markeredgecolor='black', ls=' ')


# Plot d11B (dpH) data in the first subplot
axs[1].plot(benthic_age, benthic_dpH, marker=markert_sub, markersize=7, markerfacecolor=color_obs, markeredgecolor='black', ls=' ', label="Benthic dpH", zorder=4)
axs[1].plot(planktic_age, planktic_dpH, marker=marker_surf, markersize=7, markerfacecolor=color_obs, markeredgecolor='black', ls=' ', label="Planktic dpH", zorder=4)

# Plot model results
axs[1].plot(GoC_high_iso["year"], pH_effect_high_iso_surface, '--', lw=2, color=color_HCO3_addition)
axs[1].plot(GoC_high_iso["year"], pH_effect_high_iso_subsurface, '-', lw=2, color=color_HCO3_addition)

axs[1].plot(GoC_high_iso["year"], pH_effect_high_iso_surface_CO2, '--', lw=2, color=color_CO2_addition)
axs[1].plot(GoC_low_iso["year"], pH_effect_low_iso_subsurface_CO2, '-', lw=2, color=color_CO2_addition)

# Formatting first plot
# axs[1].set_ylabel("$\Delta$pH")
axs[1].set_ylim(-2, 2)

# Plot d13C data in the second subplot
d13C_model_subsurface = GoC_low_iso["d13C_sub"] - GoC_low_iso["d13C_sub"].iloc[0]
axs[2].plot(GoC_high_iso["year"], d13C_model_subsurface, '-', lw=2, color="black")

d13C_model_subsurface_neg89 = GoC_low_iso_d13C_negative_89["d13C_sub"] - GoC_low_iso_d13C_negative_89["d13C_sub"].iloc[0]
axs[2].plot(GoC_high_iso["year"], d13C_model_subsurface_neg89, '--', lw=2, color="black")

# d13C observational data (benthic forams)
axs[2].plot(d13C_benthic_GoC["calendar age [yr BP]"] / 1000, d13C_benthic_GoC["P.ariminensis d13C diff"],
            's', markersize=7, markeredgecolor="black", markerfacecolor=color_obs, linestyle=" ", lw=1)

# Formatting second plot
# axs[2].set_xlabel("Calendar age [kyr BP]")
# axs[2].set_ylabel("$\Delta$$\delta$$^{13}$C (‰)")
axs[2].set_ylim(-3, 3)
axs[2].set_xlim(10, 20)

# Create proxy artists for the legend
proxy_benthic_dpH = mlines.Line2D([], [], color='white', marker=markert_sub, markersize=7, markerfacecolor=color_obs, markeredgecolor='black', linestyle='None', label=' ')
proxy_planktic_dpH = mlines.Line2D([], [], color='white', marker=marker_surf, markersize=7, markerfacecolor=color_obs, markeredgecolor='black', linestyle='None', label=' ')
proxy_CO2 = mlines.Line2D([], [], color='white', marker='o', markersize=7, markerfacecolor=color_obs, markeredgecolor='black', linestyle='None', label=' ')


proxy_CO2_addition = mlines.Line2D([], [], color=color_CO2_addition, lw=2, linestyle='-', label=' ')
proxy_HCO3_addition = mlines.Line2D([], [], color=color_HCO3_addition, lw=2, linestyle='-', label=' ')

# Add the legend to the plot
axs[0].legend(handles=[proxy_CO2,proxy_benthic_dpH, proxy_planktic_dpH, proxy_CO2_addition, proxy_HCO3_addition], loc='best',frameon=False)

# Show the plot
# plt.savefig("results/figures/FigureS1.pdf", bbox_inches='tight')

plt.show()