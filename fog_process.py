import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="herbie")

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os, json, shutil
from scipy.interpolate import griddata
from datetime import datetime, timedelta, UTC
from herbie import Herbie
import matplotlib.colors as mcolors

# ================= CONFIGURATION =================
OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
EXTENT = [-83, -75, 31.5, 38.5] 

CITIES = [
    [-80.22, 36.13, 'KINT'], [-79.94, 36.10, 'KGSO'], 
    [-78.79, 35.88, 'KRDU'], [-78.88, 35.00, 'KFAY'],
    [-77.89, 35.85, 'KRWI']
]

MODEL_CONFIGS = [
    {'id': 'HRRR', 'model': 'hrrr', 'prod': 'sfc', 'search': ':(TMP):2 m|:(UGRD|VGRD):925 mb'},
    {'id': 'RAP',  'model': 'rap',  'prod': 'awp130pgrb', 'search': ':(TMP):2 m'},
    {'id': 'NBM_P25', 'model': 'nbm', 'prod': 'co', 'search': ':TMP:2 m:.*25%'},
    {'id': 'NBM_P50', 'model': 'nbm', 'prod': 'co', 'search': ':TMP:2 m:.*50%'},
    {'id': 'NBM_P75', 'model': 'nbm', 'prod': 'co', 'search': ':TMP:2 m:.*75%'}
]

def add_map_features(ax):
    ax.set_extent(EXTENT)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.STATES, linewidth=0.8)
    counties = cfeature.NaturalEarthFeature(category='cultural', name='admin_2_counties', scale='10m', facecolor='none')
    ax.add_feature(counties, edgecolor='black', linewidth=0.4, alpha=0.5)

# ================= 1. DYNAMIC CROSSOVER LOGIC =================
now = datetime.now(UTC).replace(tzinfo=None)
ref_time = now.replace(hour=21, minute=0, second=0, microsecond=0)
if ref_time > now: ref_time -= timedelta(days=1)

H_init = Herbie(ref_time, model='rtma', product='anl')
ds_init = H_init.xarray(":(TMP|DPT):2 m")
if isinstance(ds_init, list): ds_init = ds_init[0]

if 'nav_lon' in ds_init.coords: ds_init = ds_init.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
lons_rtma, lats_rtma = ds_init.longitude.values, ds_init.latitude.values
max_t_grid, xover_grid = np.full(lons_rtma.shape, -999.0), np.full(lons_rtma.shape, -999.0)

for i in range(12):
    t = ref_time - timedelta(hours=i)
    try:
        H = Herbie(t, model='rtma', product='anl', verbose=False)
        ds = H.xarray(":(TMP|DPT):2 m")
        if isinstance(ds, list): ds = ds[0]
        t_key = [k for k in ds.data_vars if 't2m' in k or 'tmp' in k.lower()][0]
        d_key = [k for k in ds.data_vars if 'd2m' in k or 'dpt' in k.lower()][0]
        t_f, d_f = (ds[t_key].values - 273.15) * 9/5 + 32, (ds[d_key].values - 273.15) * 9/5 + 32
        mask = t_f > max_t_grid
        max_t_grid[mask], xover_grid[mask] = t_f[mask], d_f[mask]
    except: continue

# Plot Crossover Map with RESTORED City Identifiers
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
add_map_features(ax)
levels = np.arange(20, 78, 2)
mesh = ax.pcolormesh(lons_rtma, lats_rtma, xover_grid, cmap='turbo', norm=mcolors.BoundaryNorm(levels, 256), transform=ccrs.PlateCarree())
plt.colorbar(mesh, ax=ax, shrink=0.8, label='°F')
for lon, lat, name in CITIES:
    ax.plot(lon, lat, 'ko', markersize=4, transform=ccrs.PlateCarree())
    ax.text(lon + 0.05, lat + 0.05, name, transform=ccrs.PlateCarree(), fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
plt.title(f"Analysis: Crossover Threshold (Max T Dewpoint)\nReference: {ref_time.strftime('%Y-%m-%d')}", loc='left', fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, "crossover_analysis.png"), bbox_inches='tight', dpi=130); plt.close()

# ================= 2. FORECAST LOOP =================
rtma_pts = np.array([lons_rtma.ravel(), lats_rtma.ravel()]).T
rtma_vals = xover_grid.ravel()
hrrr_init = (now - timedelta(hours=3)).replace(minute=0, second=0, microsecond=0)
run_id = hrrr_init.strftime("%Y%m%d_%Hz")

for cfg in MODEL_CONFIGS:
    print(f"--- Attempting {cfg['id']} ---")
    for fxx in range(1, 19):
        try:
            H_fcst = Herbie(hrrr_init, model=cfg['model'], product=cfg['prod'], fxx=fxx, verbose=False)
            
            # FIXED: Bypass Herbie subsetting for NBM to prevent subset path errors
            if cfg['model'] == 'nbm':
                local_file = H_fcst.download(cfg['search'])
                ds_data = H_fcst.xarray(local_file)
            else:
                ds_data = H_fcst.xarray(cfg['search'])
            
            ds = ds_data[0] if isinstance(ds_data, list) else ds_data
            if 'nav_lon' in ds.coords: ds = ds.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
            t_var = [v for v in ds.data_vars if 't' in v.lower() and 'height' not in v.lower()][0]
            f_temp = (ds[t_var].values - 273.15) * 9/5 + 32
            
            try:
                u, v = (ds['u925'].values, ds['v925'].values) if 'u925' in ds else (ds['u'].values, ds['v'].values)
                f_wind = np.sqrt(u**2 + v**2) * 1.94384
            except:
                f_wind = np.full(f_temp.shape, 5.0)

            f_thresh = griddata(rtma_pts, rtma_vals, (ds.longitude.values, ds.latitude.values), method='linear')
            fog = np.zeros_like(f_temp)
            fog[(f_temp <= f_thresh) & (f_wind <= 15.0)] = 1
            fog[(f_temp <= (f_thresh - 3.0)) & (f_wind <= 15.0)] = 2

            fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})
            add_map_features(ax)
            for lon, lat, name in CITIES:
                ax.plot(lon, lat, 'ko', markersize=5, transform=ccrs.PlateCarree())
                ax.text(lon + 0.06, lat + 0.05, name, transform=ccrs.PlateCarree(), fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, pad=1))

            ax.pcolormesh(ds.longitude, ds.latitude, np.ma.masked_where(fog == 0, fog), 
                          transform=ccrs.PlateCarree(), cmap=mcolors.ListedColormap(['none', 'gold', 'purple']), 
                          vmin=0, vmax=2, alpha=0.8)
            
            plt.title(f"{cfg['id']} Fog Forecast | Init: {hrrr_init.strftime('%H')}Z | Valid: {(hrrr_init + timedelta(hours=fxx)).strftime('%HZ')}", loc='left', fontweight='bold')
            plt.savefig(os.path.join(OUTPUT_DIR, f"fog_{cfg['id']}_{run_id}_f{fxx:02d}.png"), bbox_inches='tight', dpi=100); plt.close()
            print(f"  Processed {cfg['id']} F{fxx:02d}")
        except Exception as e:
            print(f"  Error on {cfg['id']} F{fxx}: {e}")
            continue

with open(os.path.join(OUTPUT_DIR, "current_status.json"), "w") as f:
    json.dump({"run_id": run_id, "model_init": f"{hrrr_init.strftime('%H')}Z", "generated_at": now.strftime("%Y-%m-%d %H:%M:%S UTC")}, f)
