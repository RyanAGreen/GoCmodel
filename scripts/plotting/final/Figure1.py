import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import matplotlib.colors as mcolors
import cmocean
from pykrige.ok import OrdinaryKriging
from scipy.interpolate import griddata

# Setup custom plot style if available
try:
    import font_setup  # Sets up fonts
    import plot_style  # Applies RG custom plot style
except ImportError:
    print("Style files not found.")
    pass  # Use default settings if not found

# Paths setup
obspath = "data/observations/"
data_path = "data/preprocessed/"

# === Load and preprocess oxygen data ===
ds = xr.open_dataset(os.path.join(obspath, "woa23_all_o00_01.nc"), decode_times=False)
oxy_500 = ds.o_an.sel(depth=500, method='nearest').drop_vars(['depth', 'time'])
oxy_500_filled = oxy_500.interpolate_na(dim='lon', method='nearest')
lat = oxy_500.lat
oxy_500, lon = add_cyclic_point(oxy_500_filled, coord=ds.lon)

# === Load and preprocess 14C data ===
rafter_comp = pd.read_csv(os.path.join(obspath, 'rafter-2022-Global-14C-Compilation-FIN_editedGoA.csv'), skiprows=1)

# Filter relevant data within specified criteria
rafter_comp_filtered = rafter_comp[
    (rafter_comp['cal.age'] <= 18000) & (rafter_comp['cal.age'] >= 12000) &
    (rafter_comp['water.depth'] < 1000) & (rafter_comp['water.depth'] > 0) &
    (rafter_comp['proxy.14C.age'] != -999)
]

# Compute average Δ14C for each (lat, lon)
all_14C = rafter_comp_filtered.groupby(['latitude', 'longitude'])['DELTA14Cage.atmos.RAW'].mean().reset_index()

# Identify anomalous and non-anomalous sites
anomalous_sites = all_14C[all_14C['DELTA14Cage.atmos.RAW'] > 2500]
non_anomalous_sites = all_14C[all_14C['DELTA14Cage.atmos.RAW'] <= 2500]

# === Define key study locations ===
study_lat, study_lon = 22.91, -109.5
Stott_lat, Stott_lon = -1.2, -89.68333333

# Get Δ14C values for specific study locations
study_d14c = anomalous_sites.loc[
    (anomalous_sites['latitude'] == study_lat) & (anomalous_sites['longitude'] == study_lon),
    'DELTA14Cage.atmos.RAW'
].values[0]

Stott_D14C = anomalous_sites.loc[
    (anomalous_sites['latitude'] == Stott_lat) & (anomalous_sites['longitude'] == Stott_lon),
    'DELTA14Cage.atmos.RAW'
].values[0]

# Remove the primary study location from anomalous sites to avoid double-counting
anomalous_sites = anomalous_sites[(anomalous_sites['latitude'] != study_lat)]

# === Load hydrothermal vent locations ===
vents = pd.read_csv(os.path.join(obspath, 'vent_fields_all.csv'))

# Prepare figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), subplot_kw={'projection': ccrs.PlateCarree()})
extent = [-170, -60, -40, 55]

# Set up the colormap for anomalies
cmap_anomalies = plt.cm.autumn_r.copy()
cmap_anomalies.set_over('darkred')
cmap_anomalies.set_under('darkgray')
norm_anomalies = mcolors.BoundaryNorm(np.linspace(2500, 4000, 4).tolist() + [Stott_D14C], cmap_anomalies.N)

for ax in [ax1, ax2]:
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='black')
    ax.add_feature(cfeature.COASTLINE)
    ax.set_xticks(np.arange(-170, -59, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-40, 66, 20), crs=ccrs.PlateCarree())
    ax.tick_params(axis='both', length=5, width=2, labelleft=False, labelbottom=False)

    # Plot non-anomalous sites in gray
    ax.scatter(non_anomalous_sites['longitude'], non_anomalous_sites['latitude'], 
               color='darkgray', edgecolor='black', alpha=0.7, s=200, transform=ccrs.PlateCarree(), zorder=5)

    # Plot anomalous sites in color
    ax.scatter(anomalous_sites['longitude'], anomalous_sites['latitude'], 
               c=anomalous_sites['DELTA14Cage.atmos.RAW'], cmap=cmap_anomalies, 
               norm=norm_anomalies, edgecolor='black', s=250, alpha=1, transform=ccrs.PlateCarree(), zorder=6)

    # Highlight the study site with a star
    ax.scatter(study_lon, study_lat, c=[study_d14c], cmap=cmap_anomalies, norm=norm_anomalies, 
               edgecolor='black', s=250, marker='*', transform=ccrs.PlateCarree(), alpha=1, zorder=7)

    # Highlight Stott location
    ax.scatter(Stott_lon, Stott_lat, color='darkred', edgecolor='black', s=200, 
               transform=ccrs.PlateCarree(), zorder=7)

# === Oxygen data plot using contourf ===
oxy_levels = np.linspace(0, 20, 9)
cmap_oxy = cmocean.cm.amp_r

oxy = ax1.contourf(
    lon, lat, oxy_500[0, :, :], cmap=cmap_oxy, levels=oxy_levels, 
    extend='max', transform=ccrs.PlateCarree()
)
cbar = fig.colorbar(oxy, ax=ax1, orientation='vertical', extend='max', extendfrac=0.1, ticks=[0, 5, 10, 15, 20])

# Overlay contour line at 20 O2 level
contour = ax1.contour(lon, lat, oxy_500[0, :, :], transform=ccrs.PlateCarree(), colors='black', levels=[20], linewidths=3)
ax1.clabel(contour, inline=True, fmt=lambda x: f'O$_2$={int(x)}', fontsize=12)

# === CFC-11 data using contourf ===
lon_grid, lat_grid = np.meshgrid(np.linspace(-180, -60, 720), np.linspace(-40, 65, 360))
kriging_data_path = os.path.join(data_path, "cfc11_kriging.npz")
with np.load(kriging_data_path) as data:
    cfc11_grid = data['cfc11_grid']
cfc_levels = np.linspace(0, 2, 9)
cfc_plot = ax2.contourf(
    lon_grid, lat_grid, cfc11_grid, cmap=cmocean.cm.ice, levels=cfc_levels, 
    extend='max', transform=ccrs.PlateCarree()
)
cbar_cfc = fig.colorbar(cfc_plot, ax=ax2, orientation='vertical', extend='max', extendfrac=0.1, ticks=[0, 0.5, 1, 1.5, 2])

pv_data_path = os.path.join(data_path, "pv_values_interpolated.npz")
with np.load(pv_data_path) as data:
    pv_values_interpolated = data['pv_values_interpolated']
# Overlay potential vorticity contours
PVcontour = ax2.contour(lon_grid, lat_grid, pv_values_interpolated, colors='white', linewidths=3, transform=ccrs.PlateCarree())
ax2.clabel(PVcontour, inline=True, fontsize=13, fmt='%1.1f')

# === Plot hydrothermal vents ===
for ax in [ax1, ax2]:
    ax.plot(vents['Longitude'], vents['Latitude'], marker='^', markeredgecolor='black',
            markerfacecolor='white', markersize=6, linestyle=' ', transform=ccrs.PlateCarree())
    land = ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "110m", facecolor="black", zorder=4)
    )

plt.tight_layout()
plt.show()
# plt.savefig("results/figures/Figure1_new_correct.pdf", dpi=300, bbox_inches='tight')