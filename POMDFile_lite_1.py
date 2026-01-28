import sys
import math
import numpy as np
import pandas as pd
import pyhdf as hdf4
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import stats
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

print("[START] Script execution started")

# ============================================================
# Read HDF4 file (filtered variables)
# ============================================================
def read_sds(filename, required_vars):
    print(f"[READ] Reading HDF4 file: {filename}")
    import pyhdf.SD as H

    out = {}
    a = H.SD(filename)

    for name in required_vars:
        print(f"[READ]  Loading dataset: {name}")
        sd = a.select(name)
        out[name] = sd.get()
        print(f"[READ]  Loaded: {name}")

    print(f"[READ] Finished reading {filename}")
    return out

# ============================================================
# Create histogram bins
# ============================================================
def bins_dfs(df):
    print("[BINS] Creating histogram bins")
    dict_bins = {}
    dict_log = {}

    for key in df.columns:
        vmin = df[key].values.min()
        vmax = df[key].values.max()
        nbins = 30

        if vmin <= 0:
            if key in ["LON", "LAT", "R_ORIENTATION"]:
                min1 = math.floor(vmin)
                max1 = math.ceil(vmax)
                dict_bins[key] = np.linspace(min1, max1, int((max1 - min1) / 2) + 1)
                dict_log[key] = False
            elif key in ["RAINAREA_5", "VOLRAIN_5"]:
                dict_bins[key] = np.logspace(0, math.ceil(math.log10(vmax)), nbins)
                dict_log[key] = True
            else:
                dict_bins[key] = np.linspace(vmin, vmax, nbins)
                dict_log[key] = False
        else:
            dict_bins[key] = np.logspace(
                math.floor(math.log10(vmin)),
                math.ceil(math.log10(vmax)),
                nbins,
            )
            dict_log[key] = True

    print("[BINS] Bin creation complete")
    return dict_bins, dict_log

# ============================================================
# 1D Histogram comparison
# ============================================================
def hist1d_dfs(df1, df2, name1, name2, bins, logs, var_list):
    print("[HIST1D] Starting 1D histogram analysis")

    for i, var in enumerate(var_list):
        if i % 4 == 0:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        ax = axes.flatten()[i % 4]
        ax.hist(
            (df1[var], df2[var]),
            bins=bins[var],
            log=logs[var],
            alpha=0.5,
            color=("red", "blue"),
            label=(name1, name2),
        )

        ax.set_title(var)
        ax.set_xlabel(var)
        ax.set_ylabel("PF Counts")
        ax.set_yscale("log")
        if logs[var]:
            ax.set_xscale("log")
        ax.legend()

        if i % 4 == 3 or i == len(var_list) - 1:
            fig.suptitle(f"{name1} vs {name2} – 1D Histogram")
            fig.tight_layout()
            plt.show()

    print("[HIST1D] Completed 1D histograms")

# ============================================================
# 2D Histogram comparison
# ============================================================
def hist2d_dfs(df1, df2, name1, name2, bins, logs, pairs):
    print("[HIST2D] Starting 2D histogram analysis")

    for i, (x, y) in enumerate(pairs):
        if i % 2 == 0:
            fig, axes = plt.subplots(3, 2, figsize=(10, 12))

        for j, (df, label) in enumerate([(df1, name1), (df2, name2)]):
            ax = axes[j * 2 + (i % 2)]
            h, xe, ye, im = ax.hist2d(
                df[x], df[y],
                bins=[bins[x], bins[y]],
                density=True,
                norm=colors.LogNorm(),
                cmap="hot",
            )

            if logs[x]:
                ax.set_xscale("log")
            if logs[y]:
                ax.set_yscale("log")

            ax.set_title(f"{label}: {x} vs {y}")
            ax.set_xlabel(x)
            ax.set_ylabel(y)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

        if i % 2 == 1 or i == len(pairs) - 1:
            fig.suptitle(f"{name1} vs {name2} – 2D Histogram")
            fig.tight_layout()
            plt.show()

    print("[HIST2D] Completed 2D histograms")

# ============================================================
# Geographic 2D analysis
# ============================================================
def geo2d_dfs(df1, df2, name1, name2, geo_list):
    print("[GEO2D] Starting geographic analysis")

    lonmin = min(df1["LON"].min(), df2["LON"].min())
    lonmax = max(df1["LON"].max(), df2["LON"].max())
    latmin = min(df1["LAT"].min(), df2["LAT"].min())
    latmax = max(df1["LAT"].max(), df2["LAT"].max())

    lonbins = np.linspace(lonmin, lonmax, int(lonmax - lonmin) + 1)
    latbins = np.linspace(latmin, latmax, int(latmax - latmin) + 1)

    for i, (var, stat) in enumerate(geo_list):
        if i % 2 == 0:
            fig, axes = plt.subplots(
                3, 2, figsize=(10, 8),
                subplot_kw={"projection": ccrs.PlateCarree()},
            )

        for j, (df, label) in enumerate([(df1, name1), (df2, name2)]):
            statbin, _, _, _ = stats.binned_statistic_2d(
                df["LON"], df["LAT"], df[var], stat,
                bins=[lonbins, latbins]
            )

            ax = axes[j * 2 + (i % 2)]
            cf = ax.contourf(
                lonbins[:-1], latbins[:-1], statbin.T,
                transform=ccrs.PlateCarree(), cmap="jet"
            )
            ax.coastlines()
            ax.set_title(f"{label}: {var} ({stat})")
            fig.colorbar(cf, ax=ax, shrink=0.7)

        if i % 2 == 1 or i == len(geo_list) - 1:
            fig.suptitle(f"{name1} vs {name2} – Geographic Analysis")
            fig.tight_layout()
            plt.show()

    print("[GEO2D] Geographic analysis completed")

# ============================================================
# Comparison wrapper (CONFIG DRIVEN)
# ============================================================
def compare_dfs(df1, df2, name1, name2, var1d, var2d, geo2d):
    print(f"[COMPARE] Comparing {name1} vs {name2}")
    bins, logs = bins_dfs(df1)

    hist1d_dfs(df1, df2, name1, name2, bins, logs, var1d)
    hist2d_dfs(df1, df2, name1, name2, bins, logs, var2d)
    geo2d_dfs(df1, df2, name1, name2, geo2d)

    print("[COMPARE] Dataset comparison complete")

# ============================================================
# MAIN
# ============================================================
REQUIRED_VARS = [
    "LON", "LAT",
    "RAINAREA", "RAINAREA_5",
    "VOLRAIN", "VOLRAIN_5",
    "MEANRAINRATE",
    "R_SOLID",
    "R_ORIENTATION",
]

VAR1D_LIST = ["RAINAREA"]
VAR2D_LIST = [("RAINAREA", "MEANRAINRATE")]
GEO2D_LIST = [("RAINAREA", "count")]

print("[LOAD] Reading IMERG dataset")
dict_IMERG = read_sds("/home/cc/SciunitTrainingFiles/IMERG_202002.HDF", REQUIRED_VARS)
df_IMERG = pd.DataFrame.from_dict(dict_IMERG)

print("[LOAD] Reading GEOS5 dataset")
dict_GEOS5 = read_sds("/home/cc/SciunitTrainingFiles/GEOS5_202002.HDF", REQUIRED_VARS)
df_GEOS5 = pd.DataFrame.from_dict(dict_GEOS5)

print("[SUBSET] Selecting South America large convective storms (IMERG)")

lon = df_IMERG["LON"].values
lat = df_IMERG["LAT"].values
ra  = df_IMERG["RAINAREA"].values
ra5 = df_IMERG["RAINAREA_5"].values

mask = (
    (lon > -85) & (lon < -30) &
    (lat > -60) & (lat < 15) &
    (ra > 2000) &
    (ra5 > 100)
)

df_IMERG_SA_ST = df_IMERG.iloc[np.where(mask)]

print("[df_IMERG steps completed]")

print("[df_GEOS5 steps started]")

lon = df_GEOS5["LON"].values
lat = df_GEOS5["LAT"].values
ra  = df_GEOS5["RAINAREA"].values
ra5 = df_GEOS5["RAINAREA_5"].values

mask = (
    (lon > -85) & (lon < -30) &
    (lat > -60) & (lat < 15) &
    (ra > 2000) &
    (ra5 > 100)
)

df_GEOS5_SA_ST = df_GEOS5.iloc[np.where(mask)]

print("[SUBSET] IMERG rows:", len(df_IMERG_SA_ST))
print("[SUBSET] GEOS5 rows:", len(df_GEOS5_SA_ST))

OUT_DIR = "./plots"
os.makedirs(OUT_DIR, exist_ok=True)

_fig_counter = {"i": 0}

def custom_show(*args, **kwargs):
    fig = plt.gcf()
    _fig_counter["i"] += 1
    path = f"{OUT_DIR}/IMERG_vs_GEOS5_fig{_fig_counter['i']}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] Figure saved: {path}")
    plt.close(fig)

plt.show = custom_show

compare_dfs(
    df_IMERG_SA_ST,
    df_GEOS5_SA_ST,
    "IMERG",
    "GEOS5",
    VAR1D_LIST,
    VAR2D_LIST,
    GEO2D_LIST,
)

print("[END] Script executed successfully")
