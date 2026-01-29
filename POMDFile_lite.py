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
# FAST MODE CONTROLS (lighter + faster figures)
# ============================================================
FAST_MODE = True
MAX_PLOT_POINTS_1D = 20000
MAX_PLOT_POINTS_2D = 15000
GEO_BIN_STEP_DEG = 2.0

# ============================================================
# Helpers
# ============================================================
def sample_array(arr, max_n):
    arr = np.asarray(arr)
    if arr.shape[0] <= max_n:
        return arr
    idx = np.random.choice(arr.shape[0], max_n, replace=False)
    return arr[idx]

# ============================================================
# Read HDF4 file
# ============================================================
def read_sds(filename, required_vars):
    print(f"[READ] Reading HDF4 file: {filename}")
    import pyhdf.SD as H

    out = {}
    a = H.SD(filename)

    for name in required_vars:
        print(f"[READ]  Loading dataset: {name}")
        out[name] = a.select(name).get()
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
        vmin = df[key].min()
        vmax = df[key].max()
        nbins = 20 if FAST_MODE else 30

        if vmin <= 0:
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
# 1D Histogram (FORCED OUTPUT)
# ============================================================
def hist1d_dfs(df1, df2, name1, name2, bins, logs, var_list):
    print("[HIST1D] Forced-output 1D histograms")

    for i, var in enumerate(var_list, start=1):
        print(f"[HIST1D] ({i}/{len(var_list)}) {var}")

        x1 = sample_array(df1[var].to_numpy(), MAX_PLOT_POINTS_1D)
        x2 = sample_array(df2[var].to_numpy(), MAX_PLOT_POINTS_1D)

        edges = bins[var]
        if len(edges) > 15:
            edges = np.linspace(edges[0], edges[-1], 15)

        h1, _ = np.histogram(x1, bins=edges)
        h2, _ = np.histogram(x2, bins=edges)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(edges[:-1], h1 + 1, label=name1, color="red")
        ax.plot(edges[:-1], h2 + 1, label=name2, color="blue")

        ax.set_title(var)
        ax.set_xlabel(var)
        ax.set_ylabel("Counts")
        ax.set_yscale("log")
        if logs[var]:
            ax.set_xscale("log")
        ax.legend()

        plt.show()

    print("[HIST1D] Done")

# ============================================================
# 2D Histogram (FORCED OUTPUT)
# ============================================================
def hist2d_dfs(df1, df2, name1, name2, bins, logs, pairs):
    print("[HIST2D] Forced-output 2D histograms")

    for i, (x, y) in enumerate(pairs, start=1):
        print(f"[HIST2D] ({i}/{len(pairs)}) {x} vs {y}")

        x1 = sample_array(df1[x].to_numpy(), MAX_PLOT_POINTS_2D)
        y1 = sample_array(df1[y].to_numpy(), MAX_PLOT_POINTS_2D)
        x2 = sample_array(df2[x].to_numpy(), MAX_PLOT_POINTS_2D)
        y2 = sample_array(df2[y].to_numpy(), MAX_PLOT_POINTS_2D)

        xb = np.linspace(x1.min(), x1.max(), 40)
        yb = np.linspace(y1.min(), y1.max(), 40)

        h1, _, _ = np.histogram2d(x1, y1, bins=[xb, yb])
        h2, _, _ = np.histogram2d(x2, y2, bins=[xb, yb])

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(h1.T, origin="lower", aspect="auto")
        axes[0].set_title(name1)
        axes[1].imshow(h2.T, origin="lower", aspect="auto")
        axes[1].set_title(name2)

        fig.suptitle(f"{x} vs {y}")
        plt.show()

    print("[HIST2D] Done")

# ============================================================
# Geographic 2D (FORCED OUTPUT)
# ============================================================
def geo2d_dfs(df1, df2, name1, name2, geo_list):
    print("[GEO2D] Forced-output geographic plots")

    lonbins = np.arange(-90, -20, GEO_BIN_STEP_DEG)
    latbins = np.arange(-60, 20, GEO_BIN_STEP_DEG)

    for i, (var, stat) in enumerate(geo_list, start=1):
        print(f"[GEO2D] ({i}/{len(geo_list)}) {var}")

        g1, _, _, _ = stats.binned_statistic_2d(
            df1["LON"], df1["LAT"], df1[var], stat,
            bins=[lonbins, latbins]
        )
        g2, _, _, _ = stats.binned_statistic_2d(
            df2["LON"], df2["LAT"], df2[var], stat,
            bins=[lonbins, latbins]
        )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(g1.T, origin="lower", aspect="auto")
        axes[0].set_title(name1)
        axes[1].imshow(g2.T, origin="lower", aspect="auto")
        axes[1].set_title(name2)
        axes[2].imshow((g2 - g1).T, origin="lower", aspect="auto")
        axes[2].set_title("Difference")

        fig.suptitle(f"Geo: {var} ({stat})")
        plt.show()

    print("[GEO2D] Done")

# ============================================================
# Comparison wrapper
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
df_IMERG = pd.DataFrame.from_dict(
    read_sds("/home/cc/SciunitTrainingFiles/IMERG_202002.HDF", REQUIRED_VARS)
)

print("[LOAD] Reading GEOS5 dataset")
df_GEOS5 = pd.DataFrame.from_dict(
    read_sds("/home/cc/SciunitTrainingFiles/GEOS5_202002.HDF", REQUIRED_VARS)
)
"""
df_IMERG = pd.DataFrame.from_dict(
    read_sds("./IMERG_202002.HDF", REQUIRED_VARS)
)

print("[LOAD] Reading GEOS5 dataset")
df_GEOS5 = pd.DataFrame.from_dict(
    read_sds("./GEOS5_202002.HDF", REQUIRED_VARS)
)
"""
print("[SUBSET] Selecting South America large convective storms")

def subset(df):
    lon, lat = df["LON"].values, df["LAT"].values
    ra, ra5 = df["RAINAREA"].values, df["RAINAREA_5"].values
    mask = (
        (lon > -85) & (lon < -30) &
        (lat > -60) & (lat < 15) &
        (ra > 2000) & (ra5 > 100)
    )
    return df.iloc[np.where(mask)]

df_IMERG_SA_ST = subset(df_IMERG)
df_GEOS5_SA_ST = subset(df_GEOS5)

print("[SUBSET] IMERG rows:", len(df_IMERG_SA_ST))
print("[SUBSET] GEOS5 rows:", len(df_GEOS5_SA_ST))

OUT_DIR = "./plots2"
os.makedirs(OUT_DIR, exist_ok=True)

_fig_counter = {"i": 0}
def custom_show(*args, **kwargs):
    fig = plt.gcf()
    _fig_counter["i"] += 1
    path = f"{OUT_DIR}/IMERG_vs_GEOS5_fig{_fig_counter['i']}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    #print(f"[SAVE] Figure saved: {path}")
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
