import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import PyCO2SYS as pyco2
from matplotlib import cm, ticker
import cmocean

try:
    import font_setup  # Sets up fonts
    import plot_style  # Applies RG custom plot style
except ImportError:
    pass  # font_setup and/or plot_style modules not found. Using default settings.

# Load data for contour plot
df = pd.read_csv('results/simulations/results.csv')

# Extract columns for contour plot
volume_factor = df['volume_factor'].values
mixing_rate = df['mixing_rate'].values
total_carbon_added = df['total_carbon_added'].values
tau = df['tau'].values

# Read the Excel file
file_path = "data/observations/d11B_PP.xlsx"
benthic_df = pd.read_excel(file_path, sheet_name="Benthic",skiprows=1)
planktic_df = pd.read_excel(file_path, sheet_name="Planktic",skiprows=1)

# Extract relevant columns
benthic_cal_age = benthic_df["cal.age"]/1000
benthic_dpH = benthic_df["Delta pH"]
benthic_sigma_pH = benthic_df["1 Sigma pH"]
benthic_DD14C = benthic_df['DD14C']

planktic_cal_age = planktic_df["cal.age"]/1000
planktic_dpH = planktic_df["Delta pH"]
planktic_sigma_pH = planktic_df["1 Sigma pH"]
planktic_DD14C = planktic_df['DD14C']

color_low_iso = "#FF2A00"
color_high_iso = "#FFD500"

# Define grid for contour plot
num_samples = 100
grid_x, grid_y = np.meshgrid(
    np.linspace(volume_factor.min(), volume_factor.max(), num_samples),
    np.linspace(mixing_rate.min(), mixing_rate.max(), num_samples)
)

# Interpolate total carbon added onto the grid
grid_z = griddata((volume_factor, mixing_rate), total_carbon_added, (grid_x, grid_y), method='cubic')

# Convert inflow from per second to per year
def convert_per_sec_to_per_year(per_sec):
    per_year = per_sec * 60 * 60 * 24 * 365
    return per_year

# Calculate tau
def calc_tau(volume, inflow):
    tau = volume / inflow
    return tau

# Function to calculate tau on the grid
def compute_tau(volume_factor, mixing_rate):
    inflow = 0.48e6 + mixing_rate * 1e6  # m3/s
    inflow_per_year = convert_per_sec_to_per_year(inflow)
    volume = 1.45e14 * volume_factor
    tau = calc_tau(volume, inflow_per_year)
    return tau

class Csolve():
    def __init__(self,cats=np.array([2000,8.05,10,34.7,0])):
        # 2000 to 3500 umol/kg
        dic = cats[0] + np.arange(0, 2060, 10)
        # 7.80 to 8.20
        ph = cats[1] + np.arange(-0.2, 0.21, 0.01)
        self.pH, self.DIC = np.meshgrid(ph, dic)
        
        self.Tc = cats[2]
        self.Sal = cats[3]
        self.Zm = cats[4]
        self.sol = self.ComputeChem()
        self.ALK = self.sol["alkalinity"]

        #print(self.sol.keys())
                
        self.O = self.sol["saturation_calcite"]
                
    def ComputeChem(self,dZ=0):
        kwargs0 = dict(
            par1 = self.DIC,  # Value of the first parameter
            par2 = self.pH,  # Value of the second parameter, which is a long vector of different DIC's!
            par1_type = 2,  # The first parameter supplied is of type "1", which is "alkalinity"
            par2_type = 3,  # The second parameter supplied is of type "2", which is "DIC"
            salinity = self.Sal,  # Salinity of the sample
            temperature = self.Tc,  # Temperature at input conditions
            total_silicate = 50,  # Concentration of silicate  in the sample (in umol/kg)
            total_phosphate = 2,  # Concentration of phosphate in the sample (in umol/kg)
            opt_k_carbonic = 10,  # Choice of H2CO3 and HCO3- dissociation constants K1 and K2 ("4" means "Mehrbach refit")
            opt_k_bisulfate = 1,  # Choice of HSO4- dissociation constants KSO4 ("1" means "Dickson")
            pressure = 1.007*(self.Zm+dZ),
        )
        return pyco2.sys(**kwargs0)
        
def fmtOm(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"$\Omega$={s}" if plt.rcParams["text.usetex"] else f"$\Omega$={s}"

result = Csolve(cats=np.array([2150, 8.074850442092606, 10, 34.7, 0]))
# result = Csolve(cats=np.array([2355.04, 7.716797986382256, 10, 34.7, 0]))


reference_DIC = 2150
reference_TA = 2350

# reference_DIC = 2355.04
# reference_TA = 2419.59
def ComputeChem(DIC,TA,dZ=0):
        kwargs0 = dict(
            par1 = DIC,  # Value of the first parameter
            par2 = TA,  # Value of the second parameter, which is a long vector of different DIC's!
            par1_type = 2,  # The first parameter supplied is of type "1", which is "alkalinity"
            par2_type = 1,  # The second parameter supplied is of type "2", which is "DIC"
            salinity = 34.7,  # Salinity of the sample
            temperature = 10,  # Temperature at input conditions
            total_silicate = 50,  # Concentration of silicate  in the sample (in umol/kg)
            total_phosphate = 2,  # Concentration of phosphate in the sample (in umol/kg)
            opt_k_carbonic = 10,  # Choice of H2CO3 and HCO3- dissociation constants K1 and K2 ("4" means "Mehrbach refit")
            opt_k_bisulfate = 1,  # Choice of HSO4- dissociation constants KSO4 ("1" means "Dickson")
            pressure = 1.007*(0+dZ),
        )
        return pyco2.sys(**kwargs0)

sol = ComputeChem(reference_DIC,reference_TA)
reference_pH = sol["pH"]
# print(reference_pH)
reference_D14C = 100 # per mil

dALK = result.ALK - reference_TA
dDIC = result.DIC - reference_DIC

# Calculating D14C, assuming 14C-free carbon being added
D14C = ((reference_DIC * 1) / result.DIC - 1) * 1000
D14C_anomaly = D14C #- reference_D14C


delta_pH = result.pH - reference_pH

# Calculate tau on the grid
tau_grid = compute_tau(grid_x, grid_y)

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

# Left subplot: Contour plot
contour_levels = np.linspace(750, 3000, 11)  # Levels from 750 to 3000

physical_constraints_colormap = 'RdYlGn_r'
physical_constraints_colormap = cmocean.cm.amp


contourf = ax1.contourf(grid_x, grid_y, grid_z, levels=contour_levels, cmap=physical_constraints_colormap, vmin=750, vmax=3000)
contour = ax1.contour(grid_x, grid_y, grid_z, levels=contour_levels, linewidths=1, colors='black')

# Overlay constant tau contour lines
tau_values = [5, 50]
contours = ax1.contour(grid_x, grid_y, tau_grid, levels=tau_values, colors='black', linestyles='--',linewidths=1.5)

# Add contour line for 1000 PgC and 2500 PgC with hatching
contour_1000_pgC = ax1.contour(grid_x, grid_y, grid_z, levels=[contour_levels[0]], colors='black', linewidths=2.5)
contour_2500_pgC = ax1.contour(grid_x, grid_y, grid_z, levels=[contour_levels[-1]], colors='black', linewidths=2.5)

# Add stars with black outline and dark gray fill
ax1.plot(5.6, 4.7, marker='o', markeredgecolor='black', markerfacecolor=color_low_iso, markersize=12, label='Experiment High')
ax1.plot(9.65, 0.5, marker='o', markeredgecolor='black', markerfacecolor=color_high_iso, markersize=12, label='Experiment Low')

ax1.set_ylim(0, 5)
ax1.set_xlim(0.5, 10)

color_low_iso = "#FF2A00"
color_high_iso = "#FFD500"

# Right subplot: Scatter plot
chemical_constraints_colormap = cmocean.cm.ice

ax2t = ax2.twinx()
CSo = ax2.contourf(delta_pH, dDIC, result.O, levels=[1,2, 3, 4, 5, 6, 7, 8, 9], cmap='PuBu', extend='both')
CSo2 = ax2.contour(delta_pH, dDIC, result.O, levels=[1,2, 3, 4, 5, 6, 7, 8, 9], linewidths=1, colors='black')

# Load d11B data for scatter plot
d11B_data = pd.read_excel("data/observations/d11B_dpH_benthic_planktic_cleaned_RAG.xlsx")
benthic_D14C = d11B_data['benthic_D14C']

planktic_D14C = d11B_data['planktic_D14C']



benthic_anomaly = benthic_D14C - reference_D14C
planktic_anomaly = planktic_D14C - reference_D14C

color_low_iso = "#FF2A00"
color_high_iso = "#FFD500"

ax2t.errorbar(benthic_dpH, -benthic_DD14C, xerr=benthic_sigma_pH, fmt='s', 
              markersize=8, markerfacecolor=color_high_iso, markeredgecolor='black', 
              ecolor=(*plt.matplotlib.colors.to_rgba(color_high_iso)[:3], 0.6), capsize=3, label='Benthic ∆pH (1 Sigma)', zorder=4)

# print lengths of planktic_DD14C and planktic_dpH, and planktic_sigma_pH
# Plot planktic data with horizontal error bars
ax2t.errorbar(planktic_dpH, -planktic_DD14C, xerr=planktic_sigma_pH, fmt='^', 
              markersize=8, markerfacecolor=color_low_iso, markeredgecolor='black', 
              ecolor=(*plt.matplotlib.colors.to_rgba(color_low_iso)[:3], 0.6), capsize=3, label='Planktic ∆pH (1 Sigma)', zorder=4)
# flip the y-axis
ax2t.invert_yaxis()

# Define the range of dDIC you want to map to delta_delta14c using the regression equation
min_dDIC, max_dDIC = dDIC.min(), dDIC.max()

min_delta_delta14c = -0.31 * min_dDIC 
max_delta_delta14c = -0.31 * max_dDIC

# Set these as the y-axis limits for the twin axis
ax2t.set_ylim(min_delta_delta14c, max_delta_delta14c)

# Contours for ALK/DIC ratio (solid and dashed lines)
CS = ax2.contour(delta_pH, dDIC, dALK/dDIC, levels=[0, 1, 1.2, 2], colors='k', linewidths=1.5, linestyles="dashed")
fmt = {}
strs = ['0:1', '1:1', '1.2:1', '2:1']
for l, s in zip(CS.levels, strs):
    fmt[l] = s

ax2.set_xlim(-0.2, 0.2)


# Adjust layout and show figure
plt.tight_layout()
# plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1, wspace=0.3)

plt.show()
# plt.savefig("results/figures/Figure4.pdf", bbox_inches="tight")
