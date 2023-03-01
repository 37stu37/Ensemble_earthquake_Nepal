# %%
'''
load the multi-hazard model for buildings
and check the validation against Gorkha results
need packages from conda env update -n my_env --file qc.yaml
conda install --channel conda-forge geopandas
'''

import os
import rioxarray as rio
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.stats import lognorm
from numpy.random import default_rng

rng = default_rng()

from mods_geom_ops import Pcentroid_Rsampling, join_Pcentroid_at_Pl, mh_cascade_scenario

#############################################################################
# %% loading data
#############################################################################

datadir = Path(os.path.dirname(__file__)).parent / "data"

# %%

# load shapes
su = gpd.read_file(datadir / "shapes" / "slopeunits_preprocs [Natboundary].shp")
su = su[["su_id", "SI", "geometry"]]
osm = gpd.read_file(datadir / "shapes" / "bldgs_preprocs_light_districts [NatBoundary].shp")
print("data, paths and libraries loaded ...")

# Prepare hyperedge dataset
print("========== PGA and susceptibility PGA at slope units ==========")
raster = rio.open_rasterio(datadir / "rasters" / "gorkha2015_pga_Mosaiczero_Warp.tif")
su = Pcentroid_Rsampling(su, raster, "PGA_su")

# parameters from best fit with true positive from Gorkha
# from /Users/alexdunant/Documents/Github/SN_MH-methodology/QC.ipynb
μ = 0.5129102564102563
σ = 0.12724392947057922
su["pPGA_su"] = lognorm(μ, scale=σ).cdf(su.PGA_su)

print("========== slope attributes to building dataset ==========")
osm = join_Pcentroid_at_Pl(osm,
                           ["osm_id", "DISTRICT", "geometry"],
                           su,
                           ["su_id", "SI", "PGA_su", "pPGA_su", "geometry"])

print("========== add FlowR values to osm ==========")
fR_stats = pd.read_csv(datadir / "bldgs_fR_stats.csv")
fR_stats = fR_stats[["osm_id", "FlowR_mean", "FlowR_std"]]
osm = pd.merge(osm, fR_stats, on="osm_id", how="left")

print("========== add administrative units to osm ==========")
admin = gpd.read_file(datadir / "shapes" / "LocalbodiesWARD_753_UTM45N.shp")
osm = join_Pcentroid_at_Pl(osm, ['osm_id', 'DISTRICT', 'geometry', 'su_id', 'SI', 'PGA_su', 'pPGA_su', 'FlowR_mean',
                                 'FlowR_std'], admin, ['GaPa_NaPa', 'geometry'])

print("========== PGA at osm ==========")
osm = Pcentroid_Rsampling(osm, raster, "PGA_osm")
osm = osm.drop('index_right', axis=1)


#############################################################################
# %% cascade scenarios
#############################################################################

def iterate_N_mh_cascade_scenarios(N):
    import numpy as np
    EXP_IDS = osm.osm_id.values
    PPGA_SU = osm.pPGA_su.values
    SI_SU = osm.SI.values
    FLOWR_MEAN = osm.FlowR_mean.values
    FLOWR_STD = osm.FlowR_std.values
    r = np.zeros(EXP_IDS.shape[0], np.float64)
    for n in range(N):
        a = mh_cascade_scenario(EXP_IDS, PPGA_SU, SI_SU,
                                FLOWR_MEAN, FLOWR_STD)
        r = r + a

    return r


# number of cascade scenarios
T = 1e4

"""
Run algorithm on multiple CPUS
"""

from multiprocessing import cpu_count
import multiprocess as mp
import time

num_cores = cpu_count()
Total_scenarios_required = T
N_scenarios = np.rint(Total_scenarios_required / num_cores).astype(int)

# protect the entry point
if __name__ == '__main__':
    # mp.freeze_support()  # needed for Windows
    # create and configure the process pool
    p = mp.Pool(num_cores)
    start = time.time()
    # progress_bar = tqdm(total=N_scenarios)
    print("mapping ...")
    results = p.map(iterate_N_mh_cascade_scenarios, [N_scenarios] * num_cores)
    print("running ...")
    # concatenate the results
    concat = sum(results)
    print("done !!")
    p.terminate()
    p.close()
    print("Time Taken: ", str(time.time() - start))

# append results to exposure dataset
osm["impact"] = concat / T

# # extract results to disk
# osm.to_file(datadir / "results" / "IMPACTS_bldgs_preprocs_light_districts [NatBoundary].shp")
