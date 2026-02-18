import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="herbie")

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os, json, glob, imageio.v2 as imageio
from scipy.interpolate import griddata
from datetime import datetime, timedelta
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
    {'id': 'RAP',  'model': 'rap',  'prod': 'anl', 'search': ':(TMP):2 m|:(UGRD|VGRD):925 mb'},
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
now = datetime.utcnow()
ref_time = now.replace(hour=21, minute=0, second=0, microsecond=0)
if ref_time > now: ref_time -= timedelta(days=1)

H_init = Herbie(ref_time, model='rtma', product='anl')
ds_init = H_init.xarray(":(TMP|DPT):2 m")[0]
if 'nav_lon' in ds_init.coords: ds_init = ds_init.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
lons_rtma, lats_rtma = ds_init.longitude.values, ds_init.latitude.values
max_t_grid, xover_grid = np.full(lons_rtma.shape, -999.0), np.full(lons_rtma.shape, -999.0)

for i in range(12):
    t = ref_time - timedelta(hours=i)
    try:
        H = Herbie(t, model='rtma', product='anl', verbose=False)
        ds = H.xarray(":(TMP|DPT):2 m")[0]
        t_f, d_f = (ds['t2m'].values - 273.15) * 9/5 + 32, (ds['d2m'].values - 273.15) * 9/5 + 32
        mask = t_f > max_t_grid
        max_t_grid[mask], xover_grid[mask] = t_f[mask], d_f[mask]
    except: continue

# ================= 2. FORECAST LOOP =================
rtma_pts = np.array([lons_rtma.ravel(), lats_rtma.ravel()]).T
rtma_vals = xover_grid.ravel()
hrrr_init = (now - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)
run_id = hrrr_init.strftime("%Y%m%d_%Hz")

for cfg in MODEL_CONFIGS:
    print(f"Starting loop for {cfg['id']}...")
    for fxx in range(1, 2):
        try:
            H_fcst = Herbie(hrrr_init, model=cfg['model'], product=cfg['prod'], fxx=fxx, verbose=False)
            ds_list = H_fcst.xarray(cfg['search'])
            ds = ds_list[0] if isinstance(ds_list, list) else ds_list
            if 'nav_lon' in ds.coords: ds = ds.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
            
            # Surface Temp extraction
            t_var = 't2m' if 't2m' in ds else list(ds.data_vars)[0]
            f_temp = (ds[t_var].values - 273.15) * 9/5 + 32
            
            # Wind logic: NBM/RAP fallback to 5kts if 925mb wind not found
            f_wind = np.full(f_temp.shape, 5.0) 
            f_thresh = griddata(rtma_pts, rtma_vals, (ds.longitude.values, ds.latitude.values), method='linear')
            
            fog = np.zeros_like(f_temp)
            fog[(f_temp <= f_thresh)] = 1
            fog[(f_temp <= (f_thresh - 3.0))] = 2

            fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})
            add_map_features(ax)
            
            # RE-ADDED CITY DOTS AND LABELS
            for lon, lat, name in CITIES:
                ax.plot(lon, lat, 'ko', markersize=5, transform=ccrs.PlateCarree())
                x_off = -0.06 if name == 'KINT' else 0.06
                ax.text(lon + x_off, lat + 0.05, name, transform=ccrs.PlateCarree(), 
                        fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, pad=1))

            ax.pcolormesh(ds.longitude, ds.latitude, np.ma.masked_where(fog == 0, fog), 
                          transform=ccrs.PlateCarree(), cmap=mcolors.ListedColormap(['none', 'gold', 'purple']), 
                          vmin=0, vmax=2, alpha=0.8)
            
            valid_z = (hrrr_init + timedelta(hours=fxx)).strftime('%HZ')
            plt.title(f"{cfg['id']} Fog Forecast | Init: {hrrr_init.strftime('%H')}Z | Valid: {valid_z}", loc='left', fontweight='bold')
            plt.savefig(os.path.join(OUTPUT_DIR, f"fog_{cfg['id']}_{run_id}_f{fxx:02d}.png"), bbox_inches='tight', dpi=100)
            plt.close()
        except Exception as e:
            print(f"Skipping {cfg['id']} F{fxx}: {e}")
            continue

with open(os.path.join(OUTPUT_DIR, "current_status.json"), "w") as f:
    json.dump({"run_id": run_id, "model_init": f"{hrrr_init.strftime('%H')}Z", "generated_at": now.strftime("%Y-%m-%d %H:%M:%S UTC")}, f)
