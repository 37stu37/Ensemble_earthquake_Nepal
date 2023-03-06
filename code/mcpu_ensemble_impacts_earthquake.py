from pathlib import Path
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from numba import njit
import multiprocessing


def eq_impact(bldgs,
              bldgs_vuln,
              shake_ras,
              shake_dir):
    """
    Calculate the low, mid and high case for
    probability of collapse for every bldgs
    -> bldgs is a shapefile of all the buildings
    -> ras is a shaking footprint as a tif file
    """

    # Load libraries
    from mods_vulnerability import append_vulnerability, weighted_probability_of_collapse
    from mods_geom_ops import Pcentroid_Rsampling

    # append raster value to buildings
    b = Pcentroid_Rsampling(bldgs, shake_ras, shake_dir)
    # b = b.drop('index_right', axis=1)

    # append the meteor vulnerability to buildings
    n, t = append_vulnerability(b, bldgs_vuln)

    # extract the probabilities of collapse and attached to buildings
    b[f'low_{shake_ras}'], b[f'mid_{shake_ras}'], b[f'high_{shake_ras}'] = weighted_probability_of_collapse(t, n, b[
        shake_ras].values)

    bldgs.to_csv(f'./results/{shake_ras[8:-4]}__eqImpact.csv', index=False)

    return print(f'{shake_ras[8:-4]}__eqImpact.csv')


###################################################################
# Load datapath and datasets
GDrive = Path('G:\My Drive\Projects\sajag-nepal\Workfolder\Ensemble_earthquake_Nepal')
datadir = GDrive / 'shp'
rasdir = GDrive / 'tif'

# Load data
bldg = gpd.read_file(datadir / "bldgs_preprocs_light_districts [NatBoundary].shp")
vuln = gpd.read_file(datadir / "npl-ic-exp-dist_UTM45.shp")
vuln = vuln[["size_dist", "cnt_dist", "geometry"]]
slope = gpd.read_file(datadir / "slopeunits_preprocs [Natboundary].shp")
slope = slope[["su_id", "SI", "geometry"]]
fR_stats = pd.read_csv(datadir / "bldgs_fR_stats.csv")
fR_stats = fR_stats[["osm_id", "FlowR_mean", "FlowR_std"]]
list_rasters = [file for file in os.listdir(rasdir) if file.lower().endswith(".tif")]

print(list_rasters)

###################################################################
# run algo in parallel


if __name__ == '__main__':
    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=5)

    # Create a list of argument tuples
    argument_tuples = [(bldg, vuln, list_rasters[i], rasdir) for i in range(len(list_rasters[:5]))]

    # Apply the function to each tuple of arguments in parallel using the starmap method
    results = pool.starmap(eq_impact, argument_tuples)

    # Close the pool of worker processes
    pool.close()
    pool.join()

    print(results) # prints [2, 4, 6, 8, 10]
