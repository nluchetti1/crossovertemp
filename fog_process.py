import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os, json, shutil, requests
from scipy.interpolate import griddata
from datetime import datetime, timedelta, UTC
from herbie import Herbie
import matplotlib.colors as mcolors
import xarray as xr

# ================= CONFIGURATION =================
OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
EXTENT = [-83, -75, 31.5, 38.5] 

CITIES = [
    [-80.22, 36.13, 'KINT'], [-79.94, 36.10, 'KGSO'], 
    [-78.79, 35.88, 'KRDU'], [-78.88, 35.00, 'KFAY'],
    [-77.89, 35.85, 'KRWI']
]

# HYBRID CONFIG: HRRR/RAP use Herbie (AWS); HREF/NDFD use manual NOMADS (HTTP)
MODEL_CONFIGS = [
    {'id': 'HRRR', 'source': 'herbie', 'model': 'hrrr', 'prod': 'sfc', 'search': ':(TMP):2 m|:(UGRD|VGRD):925 mb'},
    {'id': 'RAP',  'source': 'herbie', 'model': 'rap',  'prod': 'awp130pgrb', 'search': ':(TMP):2 m|:(UGRD|VGRD):925 mb'},
    {'id': 'HREF', 'source': 'nomads', 'type': 'href_mean'},
    {'id': 'NDFD', 'source': 'nomads', 'type': 'ndfd_temp'}
]

def add_map_features(ax):
    ax.set_extent(EXTENT)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.STATES, linewidth=0.8)
    counties = cfeature.NaturalEarthFeature(category='cultural', name='admin_2_counties', scale='10m', facecolor='none')
    ax.add_feature(counties, edgecolor='black', linewidth=0.4, alpha=0.5)

# ================= 1. RTMA ANALYSIS (Baseline) =================
print("--- Generating Crossover Analysis ---")
now = datetime.now(UTC).replace(tzinfo=None)
ref_time = now.replace(hour=21, minute=0, second=0, microsecond=0)
if ref_time > now: ref_time -= timedelta(days=1)

# Grab RTMA for Crossover Thresholds
try:
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
            t_f = (ds[t_key].values - 273.15) * 9/5 + 32
            d_f = (ds[d_key].values - 273.15) * 9/5 + 32
            mask = t_f > max_t_grid
            max_t_grid[mask], xover_grid[mask] = t_f[mask], d_f[mask]
        except: continue
except Exception as e:
    print(f"RTMA Failed: {e}")
    # Fallback to dummy data if RTMA fails entirely so script doesn't crash
    lons_rtma, lats_rtma = np.meshgrid(np.linspace(-85, -70, 100), np.linspace(30, 40, 100))
    xover_grid = np.full(lons_rtma.shape, 50.0)

# Plot Analysis
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
add_map_features(ax)
levels = np.arange(20, 82, 2)
cmap = plt.cm.turbo
norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
mesh = ax.pcolormesh(lons_rtma, lats_rtma, xover_grid, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
plt.colorbar(mesh, ax=ax, shrink=0.8, ticks=levels[::2], label='°F')
plt.title(f"Crossover Threshold Analysis | {ref_time.strftime('%Y-%m-%d %H')}Z", fontweight='bold')
plt.savefig(os.path.join(OUTPUT_DIR, "crossover_analysis.png"), bbox_inches='tight'); plt.close()

rtma_pts = np.array([lons_rtma.ravel(), lats_rtma.ravel()]).T
rtma_vals = xover_grid.ravel()

# ================= 2. FORECAST GENERATION =================
for cfg in MODEL_CONFIGS:
    gif_frames = []
    found_init = None
    
    # 1. HERBIE PATH (HRRR / RAP)
    if cfg['source'] == 'herbie':
        for h_back in range(0, 6):
            check_time = (now - timedelta(hours=h_back)).replace(minute=0, second=0, microsecond=0)
            try:
                H_test = Herbie(check_time, model=cfg['model'], product=cfg['prod'], verbose=False)
                if H_test.grib: 
                    found_init = check_time
                    break
            except: continue
            
    # 2. NOMADS PATH (HREF / NDFD)
    elif cfg['source'] == 'nomads':
        # Search back 24 hours for synoptic runs (00, 06, 12, 18Z)
        for h_back in range(0, 24):
            check_time = (now - timedelta(hours=h_back)).replace(minute=0, second=0, microsecond=0)
            d_str, h_str = check_time.strftime('%Y%m%d'), check_time.strftime('%H')
            
            if cfg['type'] == 'href_mean':
                # HREF only runs at 00 and 12Z often, sometimes 06/18.
                if int(h_str) % 6 != 0: continue
                url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/href.{d_str}/ensmean/href.t{h_str}z.conus.mean.f01.grib2"
            elif cfg['type'] == 'ndfd_temp':
                # NDFD is updated irregularly, but we check specific issuance times
                # Using the standard 'ds.temp.bin' directory structure
                url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/ndfd/prod/ndfd.{d_str}/ds.temp.bin" 
                # Note: NDFD scraping is complex; usually easier to grab the GRIB2 from tgftp
                # SWITCHING TO TGFTP for NDFD reliability:
                url = f"https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.temp.bin"
            
            try:
                r = requests.head(url, timeout=5)
                if r.status_code == 200:
                    found_init = check_time
                    break
            except: continue

    if not found_init: 
        print(f"Skipping {cfg['id']}: Data not found.")
        continue

    print(f"--- Processing {cfg['id']} (Init: {found_init.strftime('%H')}Z) ---")

    # LOOP THROUGH FORECAST HOURS
    for fxx in range(1, 2):
        try:
            ds = None
            f_wind = None

            # --- DATA RETRIEVAL ---
            if cfg['source'] == 'herbie':
                H_fcst = Herbie(found_init, model=cfg['model'], product=cfg['prod'], fxx=fxx, verbose=False)
                ds = H_fcst.xarray(cfg['search'])[0]
                
                # Wind Logic for HRRR/RAP
                try:
                    u = [v for v in ds.data_vars if 'u' in v.lower()][0]
                    v = [v for v in ds.data_vars if 'v' in v.lower()][0]
                    f_wind = np.sqrt(ds[u].values**2 + ds[v].values**2) * 1.94384
                except: f_wind = None

            elif cfg['source'] == 'nomads':
                d_str, h_str = found_init.strftime('%Y%m%d'), found_init.strftime('%H')
                temp_file = f"temp_{cfg['id']}.grib2"

                if cfg['type'] == 'href_mean':
                    url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/href.{d_str}/ensmean/href.t{h_str}z.conus.mean.f{fxx:02d}.grib2"
                    r = requests.get(url, stream=True)
                    with open(temp_file, 'wb') as f: shutil.copyfileobj(r.raw, f)
                    
                    # HREF Mean GRIB often has "unknown" params in xarray, need careful filtering
                    ds = xr.open_dataset(temp_file, engine='cfgrib', 
                                         filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2})
                    # Simple wind proxy (HREF mean often lacks wind components in same file)
                    f_wind = np.full(ds.t2m.shape, 5.0) 

                elif cfg['type'] == 'ndfd_temp':
                    # NDFD is tricky. We will use Herbie to grab NDFD because Herbie handles the complex NDFD directory best
                    # if manual fails. But let's try Herbie for NDFD specifically since it's cleaner.
                    H_ndfd = Herbie(found_init, model='ndfd', product='conus', fxx=fxx)
                    ds = H_ndfd.xarray(":(TMP):2 m")[0]
                    f_wind = np.full(ds.t2m.shape, 5.0) # NDFD Temp file has no wind

            # --- PLOTTING ---
            if 'nav_lon' in ds.coords: ds = ds.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
            t_var = [v for v in ds.data_vars if 't' in v.lower() and 'height' not in v.lower()][0]
            f_temp = (ds[t_var].values - 273.15) * 9/5 + 32
            
            if f_wind is None: f_wind = np.full(f_temp.shape, 5.0)

            f_thresh = griddata(rtma_pts, rtma_vals, (ds.longitude.values, ds.latitude.values), method='linear')
            fog = np.zeros_like(f_temp)
            fog[(f_temp <= f_thresh) & (f_wind <= 15.0)] = 1
            fog[(f_temp <= (f_thresh - 3.0)) & (f_wind <= 15.0)] = 2

            fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
            add_map_features(ax)
            ax.text(1.0, 1.05, 'Dense Fog (< 1/2 SM)', color='purple', fontsize=12, fontweight='bold', ha='right', transform=ax.transAxes)
            ax.text(1.0, 1.01, 'Mist (1-3 SM)', color='#E6AC00', fontsize=12, fontweight='bold', ha='right', transform=ax.transAxes)
            ax.pcolormesh(ds.longitude, ds.latitude, np.ma.masked_where(fog == 0, fog), transform=ccrs.PlateCarree(), cmap=mcolors.ListedColormap(['#E6AC00', 'purple']), alpha=0.8)
            plt.title(f"{cfg['id']} Forecast | Init: {found_init.strftime('%H')}Z | Valid: F{fxx:02d}", loc='left', fontweight='bold')
            f_name = os.path.join(OUTPUT_DIR, f"fog_{cfg['id']}_f{fxx:02d}.png")
            plt.savefig(f_name, bbox_inches='tight', dpi=100); plt.close()
            gif_frames.append(imageio.imread(f_name))
            
            if cfg['source'] == 'nomads' and os.path.exists(temp_file): os.remove(temp_file)

        except Exception as e: 
            print(f"Error {cfg['id']} F{fxx}: {e}")
            continue

    if gif_frames:
        imageio.mimsave(os.path.join(OUTPUT_DIR, f"fog_{cfg['id']}_loop.gif"), gif_frames, fps=2, loop=0)

with open(os.path.join(OUTPUT_DIR, "current_status.json"), "w") as f:
    json.dump({"generated_at": now.strftime("%Y-%m-%d %H:%M:%S UTC")}, f)
