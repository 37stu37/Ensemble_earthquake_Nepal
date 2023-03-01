from pathlib import Path
import geopandas as gpd
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

from mods_geom_ops import join_Pcentroid_at_Pl
from mods_vulnerability import append_vulnerability, weighted_probability_of_collapse
from mods_qc import plot_qc_f1_auc_threshold, plot_AUCROC, plot_AUCF1

# load results from mh algorithm
datadir = Path("G:\My Drive\Projects\sajag-nepal\Workfolder\Hyperedge_mh_methodology\data")
R = gpd.read_file(datadir.parent / "results" / "IMPACTS_bldgs_preprocs_light_districts [NatBoundary].shp")
vuln = gpd.read_file(datadir / "npl-ic-exp-dist_UTM45.shp")
vuln = vuln[["size_dist", "cnt_dist", "geometry"]]

#############################################################################
# %% calculate damages from earthquake

nb, ty = append_vulnerability(R, vuln)
low, mid, high = weighted_probability_of_collapse(ty, nb, R.PGA_osm)

R['low'] = low
R['mid'] = mid
R['high'] = high

timestr = time.strftime("%Y%m%d")
R.to_file(datadir.parent / "results" / f"{timestr}_EQ_impacts.shp")