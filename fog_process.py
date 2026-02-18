import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os, json, shutil, requests
import imageio.v2 as imageio
from scipy.interpolate import griddata
from datetime import datetime, timedelta, timezone
from herbie import Herbie
import matplotlib.colors as mcolors
import xarray as xr

# ================= CONFIGURATION =================
OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# UPDATED EXTENT: Tighter box around NC to reduce "elongated" look
# [West, East, South, North]
EXTENT = [-84.2, -75.3, 33.2, 37.0] 

CITIES = [
    [-80.22, 36.13, 'KINT'], 
    [-79.94, 36.10, 'KGSO'], 
    [-78.79, 35.88, 'KRDU'], 
    [-78.88, 35.00, 'KFAY'],
    [-77.89, 35.85, 'KRWI']
]

# CLEANED MODEL LIST: Only HRRR, RAP, and NDFD
MODEL_CONFIGS = [
    {'id': 'HRRR', 'source': 'herbie', 'model': 'hrrr', 'prod': 'sfc', 'search': ':(TMP):2 m|:(UGRD|VGRD):925 mb'},
    {'id': 'RAP',  'source': 'herbie', 'model': 'rap',  'prod': 'awp130pgrb', 'search': ':(TMP):2 m|:(UGRD|VGRD):925 mb'},
    {'id': 'NDFD', 'source': 'manual_ndfd'}
]

def add_map_features(ax):
    ax.set_extent(EXTENT)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax.add_feature(cfeature.STATES, linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linewidth=1.0)
    counties = cfeature.NaturalEarthFeature(category='cultural', name='admin_2_counties', scale='10m', facecolor='none')
    ax.add_feature(counties, edgecolor='gray', linewidth=0.3)

def plot_cities(ax):
    for lon, lat, name in CITIES:
        ax.plot(lon, lat, 'ko', markersize=5, transform=ccrs.PlateCarree(), zorder=10)
        t = ax.text(lon + 0.05, lat + 0.05, name, transform=ccrs.PlateCarree(), 
                fontsize=9, fontweight='bold', zorder=10)
        t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# ================= 1. RTMA ANALYSIS =================
print("\n--- Step 1: Generating Crossover Analysis ---")
now = datetime.now(timezone.utc).replace(tzinfo=None)
ref_time = now.replace(hour=21, minute=0, second=0, microsecond=0)
if ref_time > now: ref_time -= timedelta(days=1)

# Default Grid
lons_rtma, lats_rtma = np.meshgrid(np.linspace(EXTENT[0], EXTENT[1], 100), np.linspace(EXTENT[2], EXTENT[3], 100))
xover_grid = np.full(lons_rtma.shape, 50.0)
rtma_success = False

try:
    H_init = Herbie(ref_time, model='rtma', product='anl', verbose=False)
    ds_init = H_init.xarray(":(TMP|DPT):2 m")
    if isinstance(ds_init, list): ds_init = ds_init[0]
    
    if 'nav_lon' in ds_init.coords: ds_init = ds_init.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
    lons_rtma, lats_rtma = ds_init.longitude.values, ds_init.latitude.values
    max_t_grid = np.full(lons_rtma.shape, -999.0)
    xover_grid = np.full(lons_rtma.shape, -999.0)

    for i in range(12):
        t_check = ref_time - timedelta(hours=i)
        try:
            H = Herbie(t_check, model='rtma', product='anl', verbose=False)
            ds = H.xarray(":(TMP|DPT):2 m")
            if isinstance(ds, list): ds = ds[0]
            
            t_key = [k for k in ds.data_vars if 't2m' in k or 'tmp' in k.lower()][0]
            d_key = [k for k in ds.data_vars if 'd2m' in k or 'dpt' in k.lower()][0]
            t_f = (ds[t_key].values - 273.15) * 9/5 + 32
            d_f = (ds[d_key].values - 273.15) * 9/5 + 32
            mask = t_f > max_t_grid
            max_t_grid[mask] = t_f[mask]
            xover_grid[mask] = d_f[mask]
            rtma_success = True
        except: continue
except Exception as e:
    print(f"Warning: RTMA Analysis failed ({e}). Using dummy threshold.")

# Plot RTMA - Adjusted figsize to (12, 7) for better aspect ratio
fig, ax = plt.subplots(figsize=(12, 7), subplot_kw={'projection': ccrs.PlateCarree()})
add_map_features(ax)
levels = np.arange(20, 82, 2)
cmap = plt.cm.turbo
norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
mesh = ax.pcolormesh(lons_rtma, lats_rtma, xover_grid, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
plt.colorbar(mesh, ax=ax, shrink=0.8, ticks=levels[::2], label='Crossover Temp (°F)')
plot_cities(ax)
plt.title(f"Crossover Threshold Analysis | {ref_time.strftime('%Y-%m-%d %H')}Z", fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, "crossover_analysis.png"), bbox_inches='tight')
plt.close()

if rtma_success:
    rtma_pts = np.array([lons_rtma.ravel(), lats_rtma.ravel()]).T
    rtma_vals = xover_grid.ravel()
else:
    rtma_pts = np.array([lons_rtma.ravel(), lats_rtma.ravel()]).T
    rtma_vals = xover_grid.ravel()


# ================= 2. FORECAST GENERATION =================
print("\n--- Step 2: Running Models ---")

for cfg in MODEL_CONFIGS:
    gif_frames = []
    found_init = None
    
    # ------------------ HERBIE PATH (HRRR / RAP) ------------------
    if cfg['source'] == 'herbie':
        for h_back in range(0, 6):
            check_time = (now - timedelta(hours=h_back)).replace(minute=0, second=0, microsecond=0)
            try:
                H_test = Herbie(check_time, model=cfg['model'], product=cfg['prod'], verbose=False)
                if H_test.grib:
                    found_init = check_time
                    break
            except: continue
        
        if not found_init:
            print(f"Skipping {cfg['id']}: No data found.")
            continue

        print(f"Processing {cfg['id']} (Init: {found_init.strftime('%H')}Z)")

        for fxx in range(1, 19):
            try:
                H_fcst = Herbie(found_init, model=cfg['model'], product=cfg['prod'], fxx=fxx, verbose=False)
                ds = H_fcst.xarray(cfg['search'])[0]
                
                if 'nav_lon' in ds.coords: ds = ds.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
                
                t_var = [v for v in ds.data_vars if 't' in v.lower() and 'height' not in v.lower()][0]
                f_temp = (ds[t_var].values - 273.15) * 9/5 + 32

                try:
                    u = [v for v in ds.data_vars if 'u' in v.lower()][0]
                    v = [v for v in ds.data_vars if 'v' in v.lower()][0]
                    f_wind = np.sqrt(ds[u].values**2 + ds[v].values**2) * 1.94384
                except: f_wind = np.full(f_temp.shape, 5.0)

                f_thresh = griddata(rtma_pts, rtma_vals, (ds.longitude.values, ds.latitude.values), method='linear')
                fog = np.zeros_like(f_temp)
                fog[(f_temp <= f_thresh) & (f_wind <= 15.0)] = 1
                fog[(f_temp <= (f_thresh - 3.0)) & (f_wind <= 15.0)] = 2

                # Adjusted figsize here too
                fig, ax = plt.subplots(figsize=(12, 7), subplot_kw={'projection': ccrs.PlateCarree()})
                add_map_features(ax)
                ax.text(1.0, 1.05, 'Dense Fog (< 1/2 SM)', color='purple', fontsize=11, fontweight='bold', ha='right', transform=ax.transAxes)
                ax.text(1.0, 1.01, 'Mist (1-3 SM)', color='#E6AC00', fontsize=11, fontweight='bold', ha='right', transform=ax.transAxes)
                plot_cities(ax)
                ax.pcolormesh(ds.longitude, ds.latitude, np.ma.masked_where(fog == 0, fog), 
                              transform=ccrs.PlateCarree(), cmap=mcolors.ListedColormap(['#E6AC00', 'purple']), alpha=0.8)
                
                valid_str = (found_init + timedelta(hours=fxx)).strftime('%H')
                plt.title(f"{cfg['id']} Forecast | Init: {found_init.strftime('%H')}Z | Valid: {valid_str}Z", loc='left', fontweight='bold')
                f_name = os.path.join(OUTPUT_DIR, f"fog_{cfg['id']}_f{fxx:02d}.png")
                plt.savefig(f_name, bbox_inches='tight', dpi=100); plt.close()
                gif_frames.append(imageio.imread(f_name))

            except Exception as e: continue

    # ------------------ NDFD PATH (MANUAL) ------------------
    elif cfg['source'] == 'manual_ndfd':
        print(f"Processing {cfg['id']} (NDFD Operational)")
        temp_file = "temp_ndfd.grib2"
        urls = [
            "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.temp.bin",
            "https://nomads.ncep.noaa.gov/pub/data/nccf/com/ndfd/prod/ndfd.20240320/ds.temp.bin" 
        ]
        
        success = False
        for url in urls:
            try:
                r = requests.get(url, stream=True, timeout=10)
                if r.status_code == 200:
                    with open(temp_file, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                    success = True
                    break
            except: continue
        
        if not success:
            print("  > Failed to download NDFD.")
            continue

        try:
            ds_full = xr.open_dataset(temp_file, engine='cfgrib', 
                                      backend_kwargs={'filter_by_keys': {'shortName': '2t'}})
            
            steps = ds_full.step.values
            if len(steps) > 18: steps = steps[:18]

            for i, step_delta in enumerate(steps):
                try:
                    ds = ds_full.sel(step=step_delta)
                    if 'longitude' not in ds.coords and 'lon' in ds.coords:
                        ds = ds.rename({'lon': 'longitude', 'lat': 'latitude'})

                    f_temp = (ds.t2m.values - 273.15) * 9/5 + 32
                    f_wind = np.full(f_temp.shape, 5.0)

                    f_thresh = griddata(rtma_pts, rtma_vals, (ds.longitude.values, ds.latitude.values), method='linear')
                    fog = np.zeros_like(f_temp)
                    fog[(f_temp <= f_thresh)] = 1
                    fog[(f_temp <= (f_thresh - 3.0))] = 2

                    # Adjusted figsize here too
                    fig, ax = plt.subplots(figsize=(12, 7), subplot_kw={'projection': ccrs.PlateCarree()})
                    add_map_features(ax)
                    ax.text(1.0, 1.05, 'Dense Fog (< 1/2 SM)', color='purple', fontsize=11, fontweight='bold', ha='right', transform=ax.transAxes)
                    ax.text(1.0, 1.01, 'Mist (1-3 SM)', color='#E6AC00', fontsize=11, fontweight='bold', ha='right', transform=ax.transAxes)
                    plot_cities(ax)
                    ax.pcolormesh(ds.longitude, ds.latitude, np.ma.masked_where(fog == 0, fog), 
                                  transform=ccrs.PlateCarree(), cmap=mcolors.ListedColormap(['#E6AC00', 'purple']), alpha=0.8)
                    
                    valid_dt = now + timedelta(seconds=int(step_delta / np.timedelta64(1, 's')))
                    valid_str = valid_dt.strftime('%H')
                    f_name = os.path.join(OUTPUT_DIR, f"fog_{cfg['id']}_f{i+1:02d}.png")
                    plt.title(f"{cfg['id']} Forecast | Init: {now.strftime('%H')}Z | Valid: {valid_str}Z", loc='left', fontweight='bold')
                    plt.savefig(f_name, bbox_inches='tight', dpi=100); plt.close()
                    gif_frames.append(imageio.imread(f_name))

                except Exception as e: continue
            
            ds_full.close()
            if os.path.exists(temp_file): os.remove(temp_file)

        except Exception as e:
            print(f"NDFD Critical Failure: {e}")

    # --- SAVE GIF ---
    if gif_frames:
        gif_name = os.path.join(OUTPUT_DIR, f"fog_{cfg['id']}_loop.gif")
        imageio.mimsave(gif_name, gif_frames, fps=2, loop=0)
        print(f"  > Generated GIF: {gif_name}")

with open(os.path.join(OUTPUT_DIR, "current_status.json"), "w") as f:
    json.dump({"generated_at": now.strftime("%Y-%m-%d %H:%M:%S UTC")}, f)

print("\n--- Process Complete ---")
