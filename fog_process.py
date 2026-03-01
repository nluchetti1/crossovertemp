import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os, json, shutil, requests, glob
import imageio.v2 as imageio
from scipy.interpolate import griddata
from datetime import datetime, timedelta, timezone
from herbie import Herbie
import matplotlib.colors as mcolors
import xarray as xr
import pandas as pd

# ================= CONFIGURATION =================
OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# EXTENT: [West, East, South, North]
EXTENT = [-84.2, -75.3, 33.2, 37.0] 

CITIES = [
    [-80.22, 36.13, 'KINT'], 
    [-79.94, 36.10, 'KGSO'], 
    [-78.79, 35.88, 'KRDU'], 
    [-78.88, 35.00, 'KFAY'],
    [-77.89, 35.85, 'KRWI']
]

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

# Helper to combine linear interpolation with nearest-neighbor fallback (prevents edge NaNs)
def interp_to_grid(pts, vals, lons, lats):
    grid_lin = griddata(pts, vals, (lons, lats), method='linear')
    grid_near = griddata(pts, vals, (lons, lats), method='nearest')
    return np.where(np.isnan(grid_lin), grid_near, grid_lin)

# ================= 1. GATHER OBSERVATIONS (NIGHT SHIFT) =================
print("\n--- Step 1: Gathering Observations ---")
now = datetime.now(timezone.utc).replace(tzinfo=None)
current_hour = now.hour

is_day_shift = 12 <= current_hour < 23
rtma_pts, rtma_vals = None, None
asos_pts, asos_vals = None, None

if not is_day_shift:
    print(f"  > Shift: Night (Pulling RTMA & ASOS)")
    
    # --- 1A. Pull RTMA ---
    ref_time = now.replace(hour=21, minute=0, second=0, microsecond=0)
    if ref_time > now: ref_time -= timedelta(days=1)
    try:
        H_init = Herbie(ref_time, model='rtma', product='anl', verbose=False)
        ds_init = H_init.xarray(":(TMP|DPT):2 m")
        if isinstance(ds_init, list): ds_init = ds_init[0]
        if 'nav_lon' in ds_init.coords: ds_init = ds_init.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
        lons_xover, lats_xover = ds_init.longitude.values, ds_init.latitude.values
        max_t_grid = np.full(lons_xover.shape, -999.0)
        rtma_grid = np.full(lons_xover.shape, -999.0)

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
                rtma_grid[mask] = d_f[mask]
            except: continue
        rtma_pts = np.array([lons_xover.ravel(), lats_xover.ravel()]).T
        rtma_vals = rtma_grid.ravel()
        print("  > RTMA Pull Successful.")
    except Exception as e:
        print(f"  > Warning: RTMA Pull failed ({e}).")

    # --- 1B. Pull ASOS/AWOS from IEM ---
    heat_date = now if current_hour >= 23 else now - timedelta(days=1)
    try:
        url = f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?network=NC_ASOS&network=SC_ASOS&network=VA_ASOS&data=tmpf&data=dwpf&year1={heat_date.year}&month1={heat_date.month}&day1={heat_date.day}&hour1=16&year2={heat_date.year}&month2={heat_date.month}&day2={heat_date.day}&hour2=23&tz=Etc/UTC&format=onlycomma&latlon=yes"
        df = pd.read_csv(url, na_values='M')
        df = df.dropna(subset=['tmpf', 'dwpf', 'lon', 'lat'])
        asos_data = []
        for stn, group in df.groupby('station'):
            max_row = group.loc[group['tmpf'].idxmax()]
            asos_data.append([max_row['lon'], max_row['lat'], max_row['dwpf']])
        
        asos_arr = np.array(asos_data)
        if len(asos_arr) > 0:
            asos_pts = asos_arr[:, :2]
            asos_vals = asos_arr[:, 2]
            print(f"  > ASOS Pull Successful ({len(asos_arr)} stations).")
    except Exception as e:
        print(f"  > Warning: ASOS Pull failed ({e}).")
else:
    print(f"  > Shift: Day (Skipping Observations. Using Forecasts.)")


# ================= 2. FORECAST GENERATION =================
print("\n--- Step 2: Running Models ---")

for cfg in MODEL_CONFIGS:
    # Cleanup old frames
    for f in glob.glob(os.path.join(OUTPUT_DIR, f"*_{cfg['id']}*.*")):
        try: os.remove(f)
        except OSError: pass
    
    # ------------------ HERBIE PATH ------------------
    if cfg['source'] == 'herbie':
        found_init = None
        for h_back in range(2, 6):
            check_time = (now - timedelta(hours=h_back)).replace(minute=0, second=0, microsecond=0)
            try:
                if Herbie(check_time, model=cfg['model'], product=cfg['prod'], verbose=False).grib:
                    found_init = check_time
                    break
            except: continue
        
        if not found_init: continue
        print(f"\nProcessing {cfg['id']} (Init: {found_init.strftime('%H')}Z)")

        # Generate base grids and threshold dictionaries
        H_base = Herbie(found_init, model=cfg['model'], product=cfg['prod'], fxx=1, verbose=False)
        ds_base = H_base.xarray(":(TMP|DPT):2 m")
        if isinstance(ds_base, list): ds_base = ds_base[0]
        if 'nav_lon' in ds_base.coords: ds_base = ds_base.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
        
        thresh_dict = {}
        gif_frames = {}

        if is_day_shift:
            # Native Forecast Calculation
            mod_max_t = np.full(ds_base.t2m.shape, -999.0)
            native_grid = np.full(ds_base.t2m.shape, 50.0)
            for fxx_check in range(0, 15):
                v_time = found_init + timedelta(hours=fxx_check)
                if 16 <= v_time.hour <= 23:
                    try:
                        H_c = Herbie(found_init, model=cfg['model'], product=cfg['prod'], fxx=fxx_check, verbose=False)
                        ds_c = H_c.xarray(":(TMP|DPT):2 m")
                        if isinstance(ds_c, list): ds_c = ds_c[0]
                        t_key = [k for k in ds_c.data_vars if 't2m' in k or 'tmp' in k.lower()][0]
                        d_key = [k for k in ds_c.data_vars if 'd2m' in k or 'dpt' in k.lower()][0]
                        t_f = (ds_c[t_key].values - 273.15) * 9/5 + 32
                        d_f = (ds_c[d_key].values - 273.15) * 9/5 + 32
                        mask = t_f > mod_max_t
                        mod_max_t[mask] = t_f[mask]
                        native_grid[mask] = d_f[mask]
                    except: continue
            thresh_dict[''] = {'grid': native_grid, 'title': f"Forecasted Crossover ({cfg['id']})"}
            gif_frames[''] = []
        else:
            if rtma_pts is not None:
                rtma_grid = interp_to_grid(rtma_pts, rtma_vals, ds_base.longitude.values, ds_base.latitude.values)
                thresh_dict['_RTMA'] = {'grid': rtma_grid, 'title': f"Observed RTMA Crossover"}
                gif_frames['_RTMA'] = []
            if asos_pts is not None:
                asos_grid = interp_to_grid(asos_pts, asos_vals, ds_base.longitude.values, ds_base.latitude.values)
                thresh_dict['_ASOS'] = {'grid': asos_grid, 'title': f"Observed ASOS/AWOS Crossover"}
                gif_frames['_ASOS'] = []

        # Plot the Threshold Analysis Maps
        for suffix, info in thresh_dict.items():
            fig, ax = plt.subplots(figsize=(12, 7), subplot_kw={'projection': ccrs.PlateCarree()})
            add_map_features(ax)
            levels = np.arange(20, 82, 2)
            cmap = plt.cm.turbo
            norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            mesh = ax.pcolormesh(ds_base.longitude, ds_base.latitude, info['grid'], cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
            plt.colorbar(mesh, ax=ax, shrink=0.8, ticks=levels[::2], label='Crossover Temp (°F)')
            if suffix == '_ASOS': ax.plot(asos_pts[:, 0], asos_pts[:, 1], 'k.', markersize=2, transform=ccrs.PlateCarree())
            plot_cities(ax)
            plt.title(f"{info['title']} | Init: {found_init.strftime('%H')}Z", fontweight='bold')
            plt.savefig(os.path.join(OUTPUT_DIR, f"crossover_{cfg['id']}{suffix}.png"), bbox_inches='tight')
            plt.close()

        # Generate Forecast Loops
        for fxx in range(1, 19):
            try:
                H_fcst = Herbie(found_init, model=cfg['model'], product=cfg['prod'], fxx=fxx, verbose=False)
                ds = H_fcst.xarray(cfg['search'])
                if isinstance(ds, list): ds = ds[0]
                if 'nav_lon' in ds.coords: ds = ds.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
                t_var = [v for v in ds.data_vars if 't' in v.lower() and 'height' not in v.lower()][0]
                f_temp = (ds[t_var].values - 273.15) * 9/5 + 32
                try:
                    u = [v for v in ds.data_vars if 'u' in v.lower()][0]
                    v = [v for v in ds.data_vars if 'v' in v.lower()][0]
                    f_wind = np.sqrt(ds[u].values**2 + ds[v].values**2) * 1.94384
                except: f_wind = np.full(f_temp.shape, 5.0)

                for suffix, info in thresh_dict.items():
                    fog = np.zeros_like(f_temp)
                    fog[(f_temp <= info['grid']) & (f_wind <= 15.0)] = 1
                    fog[(f_temp <= (info['grid'] - 3.0)) & (f_wind <= 15.0)] = 2

                    fig, ax = plt.subplots(figsize=(12, 7), subplot_kw={'projection': ccrs.PlateCarree()})
                    add_map_features(ax)
                    ax.text(1.0, 1.05, 'Dense Fog (< 1/2 SM)', color='purple', fontsize=11, fontweight='bold', ha='right', transform=ax.transAxes)
                    ax.text(1.0, 1.01, 'Mist (1-3 SM)', color='#E6AC00', fontsize=11, fontweight='bold', ha='right', transform=ax.transAxes)
                    plot_cities(ax)
                    ax.pcolormesh(ds.longitude, ds.latitude, np.ma.masked_where(fog == 0, fog), 
                                  transform=ccrs.PlateCarree(), cmap=mcolors.ListedColormap(['#E6AC00', 'purple']), alpha=0.8)
                    v_str = (found_init + timedelta(hours=fxx)).strftime('%H')
                    plt.title(f"{cfg['id']}{suffix.replace('_',' ')} Forecast | Init: {found_init.strftime('%H')}Z | Valid: {v_str}Z", loc='left', fontweight='bold')
                    f_name = os.path.join(OUTPUT_DIR, f"fog_{cfg['id']}{suffix}_f{fxx:02d}.png")
                    plt.savefig(f_name, bbox_inches='tight', dpi=100); plt.close()
                    gif_frames[suffix].append(imageio.imread(f_name))
            except: continue
        
        for suffix, frames in gif_frames.items():
            if frames: imageio.mimsave(os.path.join(OUTPUT_DIR, f"fog_{cfg['id']}{suffix}_loop.gif"), frames, fps=2, loop=0)

    # ------------------ NDFD PATH ------------------
    elif cfg['source'] == 'manual_ndfd':
        print(f"\nProcessing {cfg['id']} (NDFD Operational)")
        temp_file = "temp_ndfd.grib2"
        urls = ["https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.temp.bin",
                "https://nomads.ncep.noaa.gov/pub/data/nccf/com/ndfd/prod/ndfd.20240320/ds.temp.bin"]
        
        success = False
        for url in urls:
            try:
                r = requests.get(url, stream=True, timeout=10)
                if r.status_code == 200:
                    with open(temp_file, 'wb') as f: shutil.copyfileobj(r.raw, f)
                    success = True
                    break
            except: continue
        
        if not success: continue

        try:
            ds_t = xr.open_dataset(temp_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': '2t'}})
            ds_d = None
            try: ds_d = xr.open_dataset(temp_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': '2d'}})
            except: pass

            steps = ds_t.step.values
            if len(steps) > 18: steps = steps[:18]

            ds_base = ds_t.sel(step=steps[0])
            if 'longitude' not in ds_base.coords and 'lon' in ds_base.coords: ds_base = ds_base.rename({'lon': 'longitude', 'lat': 'latitude'})
            
            thresh_dict = {}
            gif_frames = {}

            if is_day_shift and ds_d is not None:
                ndfd_max_t = np.full(ds_base.t2m.shape, -999.0)
                native_grid = np.full(ds_base.t2m.shape, 50.0)
                for step_val in steps:
                    v_dt = now + timedelta(seconds=int(step_val / np.timedelta64(1, 's')))
                    if 16 <= v_dt.hour <= 23:
                        try:
                            t_val = (ds_t.sel(step=step_val).t2m.values - 273.15) * 9/5 + 32
                            d_val = (ds_d.sel(step=step_val).d2m.values - 273.15) * 9/5 + 32
                            mask = t_val > ndfd_max_t
                            ndfd_max_t[mask] = t_val[mask]
                            native_grid[mask] = d_val[mask]
                        except: continue
                thresh_dict[''] = {'grid': native_grid, 'title': "Forecasted Crossover (NDFD)"}
                gif_frames[''] = []
            elif not is_day_shift:
                if rtma_pts is not None:
                    rtma_grid = interp_to_grid(rtma_pts, rtma_vals, ds_base.longitude.values, ds_base.latitude.values)
                    thresh_dict['_RTMA'] = {'grid': rtma_grid, 'title': "Observed RTMA Crossover"}
                    gif_frames['_RTMA'] = []
                if asos_pts is not None:
                    asos_grid = interp_to_grid(asos_pts, asos_vals, ds_base.longitude.values, ds_base.latitude.values)
                    thresh_dict['_ASOS'] = {'grid': asos_grid, 'title': "Observed ASOS/AWOS Crossover"}
                    gif_frames['_ASOS'] = []

            for suffix, info in thresh_dict.items():
                fig, ax = plt.subplots(figsize=(12, 7), subplot_kw={'projection': ccrs.PlateCarree()})
                add_map_features(ax)
                levels = np.arange(20, 82, 2)
                cmap = plt.cm.turbo
                norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
                mesh = ax.pcolormesh(ds_base.longitude, ds_base.latitude, info['grid'], cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
                plt.colorbar(mesh, ax=ax, shrink=0.8, ticks=levels[::2], label='Crossover Temp (°F)')
                if suffix == '_ASOS': ax.plot(asos_pts[:, 0], asos_pts[:, 1], 'k.', markersize=2, transform=ccrs.PlateCarree())
                plot_cities(ax)
                plt.title(f"{info['title']} | Init: {now.strftime('%H')}Z", fontweight='bold')
                plt.savefig(os.path.join(OUTPUT_DIR, f"crossover_{cfg['id']}{suffix}.png"), bbox_inches='tight')
                plt.close()

            for i, step_delta in enumerate(steps):
                try:
                    ds_frame = ds_t.sel(step=step_delta)
                    if 'longitude' not in ds_frame.coords and 'lon' in ds_frame.coords: ds_frame = ds_frame.rename({'lon': 'longitude', 'lat': 'latitude'})
                    f_temp = (ds_frame.t2m.values - 273.15) * 9/5 + 32
                    
                    for suffix, info in thresh_dict.items():
                        fog = np.zeros_like(f_temp)
                        fog[(f_temp <= info['grid'])] = 1
                        fog[(f_temp <= (info['grid'] - 3.0))] = 2

                        fig, ax = plt.subplots(figsize=(12, 7), subplot_kw={'projection': ccrs.PlateCarree()})
                        add_map_features(ax)
                        ax.text(1.0, 1.05, 'Dense Fog (< 1/2 SM)', color='purple', fontsize=11, fontweight='bold', ha='right', transform=ax.transAxes)
                        ax.text(1.0, 1.01, 'Mist (1-3 SM)', color='#E6AC00', fontsize=11, fontweight='bold', ha='right', transform=ax.transAxes)
                        plot_cities(ax)
                        ax.pcolormesh(ds_frame.longitude, ds_frame.latitude, np.ma.masked_where(fog == 0, fog), transform=ccrs.PlateCarree(), cmap=mcolors.ListedColormap(['#E6AC00', 'purple']), alpha=0.8)
                        v_str = (now + timedelta(seconds=int(step_delta / np.timedelta64(1, 's')))).strftime('%H')
                        plt.title(f"{cfg['id']}{suffix.replace('_',' ')} Forecast | Init: {now.strftime('%H')}Z | Valid: {v_str}Z", loc='left', fontweight='bold')
                        f_name = os.path.join(OUTPUT_DIR, f"fog_{cfg['id']}{suffix}_f{i+1:02d}.png")
                        plt.savefig(f_name, bbox_inches='tight', dpi=100); plt.close()
                        gif_frames[suffix].append(imageio.imread(f_name))
                except: continue
            
            ds_t.close()
            if ds_d is not None: ds_d.close()
            if os.path.exists(temp_file): os.remove(temp_file)
            
            for suffix, frames in gif_frames.items():
                if frames: imageio.mimsave(os.path.join(OUTPUT_DIR, f"fog_{cfg['id']}{suffix}_loop.gif"), frames, fps=2, loop=0)

        except Exception as e:
            print(f"NDFD Critical Failure: {e}")

# Save state
shift_type = "day" if is_day_shift else "night"
with open(os.path.join(OUTPUT_DIR, "current_status.json"), "w") as f:
    json.dump({"generated_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"), "shift": shift_type}, f)

print("\n--- Process Complete ---")
