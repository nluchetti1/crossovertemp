import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="herbie")

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os, json, shutil, imageio.v2 as imageio, requests
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

# HREF and NamNest require specific products and domains
MODEL_CONFIGS = [
    {'id': 'HRRR',    'model': 'hrrr', 'prod': 'sfc',        'search': ':(TMP):2 m|:(UGRD|VGRD):925 mb', 'freq': 'hourly'},
    {'id': 'RAP',     'model': 'rap',  'prod': 'awp130pgrb', 'search': ':(TMP):2 m|:(UGRD|VGRD):925 mb', 'freq': 'hourly'},
    {'id': 'NamNest', 'model': 'nam',  'prod': 'conusnest',  'search': ':(TMP):2 m|:(UGRD|VGRD):925 mb', 'freq': 'synoptic'},
    {'id': 'HREF',    'model': 'href', 'prod': 'mean',       'search': ':(TMP):2 m|:(UGRD|VGRD):925 mb', 'freq': 'synoptic', 'domain': 'conus'}
]

def add_map_features(ax):
    ax.set_extent(EXTENT)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.STATES, linewidth=0.8)
    counties = cfeature.NaturalEarthFeature(category='cultural', name='admin_2_counties', scale='10m', facecolor='none')
    ax.add_feature(counties, edgecolor='black', linewidth=0.4, alpha=0.5)

# ================= 1. RTMA ANALYSIS =================
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

# ================= 2. FORECAST GENERATION =================
rtma_pts = np.array([lons_rtma.ravel(), lats_rtma.ravel()]).T
rtma_vals = xover_grid.ravel()

for cfg in MODEL_CONFIGS:
    gif_frames = []
    found_init = None
    
    # Logic to prioritize the most recent run based on model frequency
    search_hours = 24 if cfg['freq'] == 'synoptic' else 6
    extra_kwargs = {k: v for k, v in cfg.items() if k == 'domain'}
    
    for h_back in range(0, search_hours + 1):
        check_time = (now - timedelta(hours=h_back)).replace(minute=0, second=0, microsecond=0)
        
        # Ensure synoptic models only attempt 00, 06, 12, or 18Z
        if cfg['freq'] == 'synoptic' and check_time.hour % 6 != 0:
            continue
            
        try:
            H_test = Herbie(check_time, model=cfg['model'], product=cfg['prod'], verbose=False, **extra_kwargs)
            if H_test.grib: 
                found_init = check_time
                break
        except: continue
    
    if not found_init: 
        print(f"Skipping {cfg['id']}: No recent data found.")
        continue
        
    print(f"--- Processing {cfg['id']} (Init: {found_init.strftime('%H')}Z) ---")
    
    for fxx in range(1, 19):
        try:
            H_fcst = Herbie(found_init, model=cfg['model'], product=cfg['prod'], fxx=fxx, verbose=False, **extra_kwargs)
            ds_data = H_fcst.xarray(cfg['search'])
            ds = ds_data[0] if isinstance(ds_data, list) else ds_data
            
            if 'nav_lon' in ds.coords: ds = ds.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
            t_var = [v for v in ds.data_vars if 't' in v.lower() and 'height' not in v.lower()][0]
            f_temp = (ds[t_var].values - 273.15) * 9/5 + 32
            
            # 925 mb Stability Mask Logic
            try:
                u_key = [v for v in ds.data_vars if 'u' in v.lower() and ('925' in str(v) or 'grd' in str(v).lower())][0]
                v_key = [v for v in ds.data_vars if 'v' in v.lower() and ('925' in str(v) or 'grd' in str(v).lower())][0]
                f_wind = np.sqrt(ds[u_key].values**2 + ds[v_key].values**2) * 1.94384
            except: f_wind = np.full(f_temp.shape, 5.0)

            f_thresh = griddata(rtma_pts, rtma_vals, (ds.longitude.values, ds.latitude.values), method='linear')
            fog = np.zeros_like(f_temp)
            fog[(f_temp <= f_thresh) & (f_wind <= 15.0)] = 1
            fog[(f_temp <= (f_thresh - 3.0)) & (f_wind <= 15.0)] = 2

            fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
            add_map_features(ax)
            
            # Legend positioning above the map
            ax.text(1.0, 1.05, 'Dense Fog (< 1/2 SM)', color='purple', fontsize=12, fontweight='bold', ha='right', transform=ax.transAxes)
            ax.text(1.0, 1.01, 'Mist (1-3 SM)', color='#E6AC00', fontsize=12, fontweight='bold', ha='right', transform=ax.transAxes)

            for lon, lat, name in CITIES:
                ax.plot(lon, lat, 'ko', markersize=5, transform=ccrs.PlateCarree())
                ax.text(lon + 0.05, lat + 0.05, name, transform=ccrs.PlateCarree(), fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            ax.pcolormesh(ds.longitude, ds.latitude, np.ma.masked_where(fog == 0, fog), transform=ccrs.PlateCarree(), cmap=mcolors.ListedColormap(['#E6AC00', 'purple']), alpha=0.8)
            
            plt.title(f"{cfg['id']} Fog Forecast | Init: {found_init.strftime('%H')}Z | Valid: {(found_init + timedelta(hours=fxx)).strftime('%HZ')}", loc='left', fontweight='bold', fontsize=14, pad=10)
            
            f_name = os.path.join(OUTPUT_DIR, f"fog_{cfg['id']}_f{fxx:02d}.png")
            plt.savefig(f_name, bbox_inches='tight', dpi=100); plt.close()
            gif_frames.append(imageio.imread(f_name))
        except Exception as e:
            continue
    
    if gif_frames:
        imageio.mimsave(os.path.join(OUTPUT_DIR, f"fog_{cfg['id']}_loop.gif"), gif_frames, fps=2, loop=0)

with open(os.path.join(OUTPUT_DIR, "current_status.json"), "w") as f:
    json.dump({"generated_at": now.strftime("%Y-%m-%d %H:%M:%S UTC")}, f)
