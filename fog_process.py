import warnings
# Suppress the noisy warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os, json, shutil, requests
import imageio.v2 as imageio  # Explicit import to fix "not defined" error
from scipy.interpolate import griddata
from datetime import datetime, timedelta, timezone
from herbie import Herbie
import matplotlib.colors as mcolors
import xarray as xr

# ================= CONFIGURATION =================
OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Focused Extent for NC/SC
EXTENT = [-84.0, -75.0, 33.0, 37.5] 

CITIES = [
    [-80.22, 36.13, 'KINT'], 
    [-79.94, 36.10, 'KGSO'], 
    [-78.79, 35.88, 'KRDU'], 
    [-78.88, 35.00, 'KFAY'],
    [-77.89, 35.85, 'KRWI']
]

# HYBRID CONFIGURATION
# HRRR/RAP -> Use Herbie (Fastest for hourly data)
# HREF/NDFD -> Use Manual Download (Most reliable for ensembles/operational data)
MODEL_CONFIGS = [
    {'id': 'HRRR', 'source': 'herbie', 'model': 'hrrr', 'prod': 'sfc', 'search': ':(TMP):2 m|:(UGRD|VGRD):925 mb'},
    {'id': 'RAP',  'source': 'herbie', 'model': 'rap',  'prod': 'awp130pgrb', 'search': ':(TMP):2 m|:(UGRD|VGRD):925 mb'},
    {'id': 'HREF', 'source': 'manual_href'},
    {'id': 'NDFD', 'source': 'manual_ndfd'}
]

def add_map_features(ax):
    ax.set_extent(EXTENT)
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax.add_feature(cfeature.STATES, linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linewidth=1.0)
    # Add Counties
    counties = cfeature.NaturalEarthFeature(
        category='cultural', name='admin_2_counties', scale='10m', facecolor='none')
    ax.add_feature(counties, edgecolor='gray', linewidth=0.3)

def plot_cities(ax):
    """Helper to ensure cities always plot on top"""
    for lon, lat, name in CITIES:
        # Plot black dot
        ax.plot(lon, lat, 'ko', markersize=5, transform=ccrs.PlateCarree(), zorder=10)
        # Plot label with white halo for readability
        t = ax.text(lon + 0.05, lat + 0.05, name, transform=ccrs.PlateCarree(), 
                fontsize=9, fontweight='bold', zorder=10)
        t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

# ================= 1. RTMA ANALYSIS (Baseline) =================
print("\n--- Step 1: Generating Crossover Analysis ---")
now = datetime.now(timezone.utc).replace(tzinfo=None) # Ensure naive UTC
ref_time = now.replace(hour=21, minute=0, second=0, microsecond=0)
if ref_time > now: ref_time -= timedelta(days=1)

# Default grid if RTMA fails
lons_rtma, lats_rtma = np.meshgrid(np.linspace(EXTENT[0], EXTENT[1], 100), np.linspace(EXTENT[2], EXTENT[3], 100))
xover_grid = np.full(lons_rtma.shape, 50.0)
rtma_success = False

try:
    # Attempt to grab RTMA
    H_init = Herbie(ref_time, model='rtma', product='anl', verbose=False)
    ds_init = H_init.xarray(":(TMP|DPT):2 m")
    if isinstance(ds_init, list): ds_init = ds_init[0]
    
    if 'nav_lon' in ds_init.coords: ds_init = ds_init.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
    lons_rtma, lats_rtma = ds_init.longitude.values, ds_init.latitude.values
    
    # Reset grid to proper shape
    max_t_grid = np.full(lons_rtma.shape, -999.0)
    xover_grid = np.full(lons_rtma.shape, -999.0)

    # Loop back 12 hours to find Max T and Dewpoint at Max T
    for i in range(12):
        t_check = ref_time - timedelta(hours=i)
        try:
            H = Herbie(t_check, model='rtma', product='anl', verbose=False)
            ds = H.xarray(":(TMP|DPT):2 m")
            if isinstance(ds, list): ds = ds[0]
            
            # Extract variables safely
            t_key = [k for k in ds.data_vars if 't2m' in k or 'tmp' in k.lower()][0]
            d_key = [k for k in ds.data_vars if 'd2m' in k or 'dpt' in k.lower()][0]
            
            t_f = (ds[t_key].values - 273.15) * 9/5 + 32
            d_f = (ds[d_key].values - 273.15) * 9/5 + 32
            
            # Logic: Update Max T and capture Dewpoint at that time
            mask = t_f > max_t_grid
            max_t_grid[mask] = t_f[mask]
            xover_grid[mask] = d_f[mask]
            rtma_success = True
        except: continue
except Exception as e:
    print(f"Warning: RTMA Analysis failed ({e}). Using dummy threshold.")

# Plot Analysis
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
add_map_features(ax)
levels = np.arange(20, 82, 2)
cmap = plt.cm.turbo
norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
mesh = ax.pcolormesh(lons_rtma, lats_rtma, xover_grid, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
plt.colorbar(mesh, ax=ax, shrink=0.8, ticks=levels[::2], label='Crossover Temp (°F)')
plot_cities(ax) # FORCE PLOT CITIES
plt.title(f"Crossover Threshold Analysis | {ref_time.strftime('%Y-%m-%d %H')}Z", fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, "crossover_analysis.png"), bbox_inches='tight')
plt.close()

# Prepare interpolation points
if rtma_success:
    rtma_pts = np.array([lons_rtma.ravel(), lats_rtma.ravel()]).T
    rtma_vals = xover_grid.ravel()
else:
    # Fallback points if RTMA failed
    rtma_pts = np.array([lons_rtma.ravel(), lats_rtma.ravel()]).T
    rtma_vals = xover_grid.ravel()


# ================= 2. FORECAST GENERATION =================
print("\n--- Step 2: Running Models ---")

for cfg in MODEL_CONFIGS:
    gif_frames = []
    found_init = None
    
    # --- FIND INITIALIZATION TIME ---
    
    # PATH A: HERBIE (HRRR/RAP)
    if cfg['source'] == 'herbie':
        for h_back in range(0, 6):
            check_time = (now - timedelta(hours=h_back)).replace(minute=0, second=0, microsecond=0)
            try:
                # Quick check if exists
                H_test = Herbie(check_time, model=cfg['model'], product=cfg['prod'], verbose=False)
                if H_test.grib:
                    found_init = check_time
                    break
            except: continue
            
    # PATH B: MANUAL DOWNLOAD (HREF / NDFD)
    elif cfg['source'].startswith('manual'):
        # Search back 24 hours for synoptic runs
        for h_back in range(0, 24):
            check_time = (now - timedelta(hours=h_back)).replace(minute=0, second=0, microsecond=0)
            d_str = check_time.strftime('%Y%m%d')
            h_str = check_time.strftime('%H')
            
            url = ""
            if cfg['source'] == 'manual_href':
                # HREF usually available at 00, 06, 12, 18
                if int(h_str) % 6 != 0: continue 
                url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/href.{d_str}/ensmean/href.t{h_str}z.conus.mean.f01.grib2"
            
            elif cfg['source'] == 'manual_ndfd':
                # NDFD on TGFTP is reliable. It is stored as a rolling file "ds.temp.bin"
                # We assume the file exists and treat "check_time" as "Now" effectively
                url = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.temp.bin"
                found_init = now # NDFD is always "current"
                break
            
            # Check URL existence
            try:
                r = requests.head(url, timeout=5)
                if r.status_code == 200:
                    found_init = check_time
                    break
            except: continue

    if not found_init:
        print(f"Skipping {cfg['id']}: No data found.")
        continue

    print(f"Processing {cfg['id']} (Init: {found_init.strftime('%H')}Z)")

    # --- FORECAST LOOP (f01 - f18) ---
    for fxx in range(1, 2):
        try:
            ds = None
            f_wind = None
            temp_file = f"temp_{cfg['id']}.grib2"

            # 1. DOWNLOAD / LOAD DATA
            if cfg['source'] == 'herbie':
                H_fcst = Herbie(found_init, model=cfg['model'], product=cfg['prod'], fxx=fxx, verbose=False)
                ds = H_fcst.xarray(cfg['search'])[0]
                
                # Wind Extraction for Herbie models
                try:
                    u = [v for v in ds.data_vars if 'u' in v.lower()][0]
                    v = [v for v in ds.data_vars if 'v' in v.lower()][0]
                    f_wind = np.sqrt(ds[u].values**2 + ds[v].values**2) * 1.94384
                except: f_wind = None

            elif cfg['source'] == 'manual_href':
                d_str, h_str = found_init.strftime('%Y%m%d'), found_init.strftime('%H')
                url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/href.{d_str}/ensmean/href.t{h_str}z.conus.mean.f{fxx:02d}.grib2"
                
                r = requests.get(url, stream=True)
                with open(temp_file, 'wb') as f: shutil.copyfileobj(r.raw, f)
                
                # Open with xarray/cfgrib
                ds = xr.open_dataset(temp_file, engine='cfgrib', 
                                     backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
                # HREF Mean wind often in different file, use dummy wind (no mask)
                f_wind = np.full(ds.t2m.shape, 5.0) 

            elif cfg['source'] == 'manual_ndfd':
                # NDFD is a single file for multiple times, OR individual files. 
                # For simplicity/reliability on TGFTP, we use ds.temp.bin (current forecast)
                if fxx == 1: # Only download once
                    url = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.temp.bin"
                    r = requests.get(url, stream=True)
                    with open(temp_file, 'wb') as f: shutil.copyfileobj(r.raw, f)
                
                # NDFD contains multiple steps. We select by step.
                ds_full = xr.open_dataset(temp_file, engine='cfgrib')
                # Find the step closest to fxx hours
                valid_step = ds_full.step[ds_full.step >= timedelta(hours=fxx)].min()
                ds = ds_full.sel(step=valid_step)
                f_wind = np.full(ds.t2m.shape, 5.0)

            # 2. STANDARDIZE COORDINATES
            if 'nav_lon' in ds.coords: ds = ds.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
            
            # 3. GET TEMP
            t_var = [v for v in ds.data_vars if 't' in v.lower() and 'height' not in v.lower()][0]
            f_temp = (ds[t_var].values - 273.15) * 9/5 + 32
            
            # 4. FOG LOGIC
            if f_wind is None: f_wind = np.full(f_temp.shape, 5.0)

            f_thresh = griddata(rtma_pts, rtma_vals, (ds.longitude.values, ds.latitude.values), method='linear')
            fog = np.zeros_like(f_temp)
            
            # Mist Logic
            mask_mist = (f_temp <= f_thresh) & (f_wind <= 15.0)
            fog[mask_mist] = 1
            
            # Dense Fog Logic
            mask_dense = (f_temp <= (f_thresh - 3.0)) & (f_wind <= 15.0)
            fog[mask_dense] = 2

            # 5. PLOT
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            add_map_features(ax)
            
            # Text Legend (Upper Right)
            ax.text(1.0, 1.05, 'Dense Fog (< 1/2 SM)', color='purple', fontsize=11, fontweight='bold', ha='right', transform=ax.transAxes)
            ax.text(1.0, 1.01, 'Mist (1-3 SM)', color='#E6AC00', fontsize=11, fontweight='bold', ha='right', transform=ax.transAxes)

            plot_cities(ax) # FORCE PLOT CITIES

            # Plot Data
            ax.pcolormesh(ds.longitude, ds.latitude, np.ma.masked_where(fog == 0, fog), 
                          transform=ccrs.PlateCarree(), 
                          cmap=mcolors.ListedColormap(['#E6AC00', 'purple']), alpha=0.8)
            
            # Title
            valid_str = (found_init + timedelta(hours=fxx)).strftime('%H')
            plt.title(f"{cfg['id']} Forecast | Init: {found_init.strftime('%H')}Z | Valid: {valid_str}Z", 
                      loc='left', fontweight='bold')
            
            f_name = os.path.join(OUTPUT_DIR, f"fog_{cfg['id']}_f{fxx:02d}.png")
            plt.savefig(f_name, bbox_inches='tight', dpi=100)
            plt.close()
            
            # Append to GIF list
            gif_frames.append(imageio.imread(f_name))

        except Exception as e:
            # print(f"  Frame {fxx} skipped: {e}") 
            continue

    # 6. MAKE GIF
    if gif_frames:
        gif_name = os.path.join(OUTPUT_DIR, f"fog_{cfg['id']}_loop.gif")
        imageio.mimsave(gif_name, gif_frames, fps=2, loop=0)
        print(f"  > Generated GIF: {gif_name}")
    
    # Cleanup Temp File
    if 'manual' in cfg['source'] and os.path.exists(f"temp_{cfg['id']}.grib2"):
        os.remove(f"temp_{cfg['id']}.grib2")

# Final Status Update
with open(os.path.join(OUTPUT_DIR, "current_status.json"), "w") as f:
    json.dump({"generated_at": now.strftime("%Y-%m-%d %H:%M:%S UTC")}, f)

print("\n--- Process Complete ---")
