import ray
from pathlib import Path
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from numba import njit


# Define the function to run in parallel
@ray.remote
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

    return b


@njit()
def mh_cascade_scenario(EXP_IDS, PPGA_SU, SI_SU, FLOWR_MEAN, FLOWR_STD):
    a = np.zeros(EXP_IDS.shape[0], np.float64)

    for i, _ in enumerate(EXP_IDS):
        rng = np.random.uniform(0, 1)
        if PPGA_SU[i] > rng:
            rng = np.random.uniform(0, 1)
            if np.random.normal(SI_SU[i], SI_SU[i] * 0.1) > rng:
                rng = np.random.uniform(0, 1)
                if np.random.normal(FLOWR_MEAN[i], FLOWR_STD[i]) > rng:
                    a[i] = 1
    return a


@ray.remote
def ls_impact(bldgs,
              slopes,
              shake_ras,
              shake_dir,
              flowr_stats,
              n_scenarios):

    from scipy.stats import lognorm
    import pandas as pd
    from tqdm import trange
    from mods_geom_ops import Pcentroid_Rsampling, join_Pcentroid_at_Pl

    # sample PGA values at slope unit
    slopes_pga = Pcentroid_Rsampling(slopes, shake_ras, shake_dir)

    # probability of slope triggering landslide from specific PGA
    # parameters from best fit with true positive from Gorkha
    # from /Users/alexdunant/Documents/Github/SN_MH-methodology/QC.ipynb
    μ = 0.5129102564102563
    σ = 0.12724392947057922
    slopes_pga["pPGA_su"] = lognorm(μ, scale=σ).cdf(slopes_pga[f'{shake_ras}'])

    # slope attributes to building dataset
    bldgs = join_Pcentroid_at_Pl(bldgs,
                                 ["osm_id", "DISTRICT", "geometry"],
                                 slopes_pga,
                                 ["su_id", "SI", "pPGA_su", "geometry"])

    # add FlowR values to osm
    bldgs = pd.merge(bldgs, flowr_stats, on="osm_id", how="left")

    # generate landslide scenarios for the earthquake
    scenarios_results = np.zeros(bldgs.osm_id.values.shape[0], np.float64)

    # Monte Carlo simulation of landslides for earthquake
    for n in trange(n_scenarios):
        scenario_impact = mh_cascade_scenario(EXP_IDS=bldgs.osm_id.values,
                                              PPGA_SU=bldgs.pPGA_su.values,
                                              SI_SU=bldgs.SI.values,
                                              FLOWR_MEAN=bldgs.FlowR_mean.values,
                                              FLOWR_STD=bldgs.FlowR_std.values)

        scenarios_results = scenarios_results + scenario_impact

    bldgs['impact'] = scenarios_results / n_scenarios

    return bldgs


###################################################################
# Load datapath and datasets
GDrive = Path('G:/My Drive/Projects/sajag-nepal/Workfolder/Ensemble_earthquake_Nepal')
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

###################################################################
# run algo in parallel

##########################
# Earthquake

# Initialize Ray
ray.init()

# Create a list to hold the output futures
output_futures = []

# Iterate through the inputs and run the function in parallel
for raster in list_rasters[:5]:
    # Call the function asynchronously and append the output future to the list
    output_futures.append(eq_impact.remote(bldg,
                                           vuln,
                                           raster,
                                           rasdir))


# Wait for all the output futures to complete
outputs = ray.get(output_futures)

# write building datasets to disk
for index, output in enumerate(outputs):
    # Write the DataFrame to a CSV file
    output.to_csv(GDrive / 'results' / f'{list_rasters[index][8:-4]}__eqImpact.csv', index=False)

# Shutdown Ray
ray.shutdown()


##########################
# Landslide

# Initialize Ray
ray.init()

# Create a list to hold the output futures
output_futures = []

# Iterate through the inputs and run the function in parallel
for raster in list_rasters[:5]:
    # Call the function asynchronously and append the output future to the list
    output_futures.append(ls_impact.remote(bldg,
                                              slope,
                                              raster,
                                              rasdir,
                                              fR_stats,
                                              10000))

# Wait for all the output futures to complete
outputs = ray.get(output_futures)

# write building datasets to disk
for index, output in enumerate(outputs):
    # Write the DataFrame to a CSV file
    output.to_csv(GDrive / 'results' / f'{list_rasters[index][8:-4]}__lsImpact.csv', index=False)

# Shutdown Ray
ray.shutdown()
