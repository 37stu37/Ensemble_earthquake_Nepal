import ray
from pathlib import Path
import os
import geopandas as gpd

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
    b[f'low_{shake_ras}'], b[f'mid_{shake_ras}'], b[f'high_{shake_ras}'] = weighted_probability_of_collapse(t, n, b[shake_ras].values)

    return b


###################################################################
# Load datapath and datasets
datadir = Path('/Users/alexdunant/Documents/Github/Ensemble_earthquake_Nepal/shp')
rasdir = Path('/Users/alexdunant/Documents/Github/Ensemble_earthquake_Nepal/tif')

# Load data
bldg = gpd.read_file(datadir / "bldgs_preprocs_light_districts [NatBoundary].shp")
vuln = gpd.read_file(datadir / "npl-ic-exp-dist_UTM45.shp")
vuln = vuln[["size_dist", "cnt_dist", "geometry"]]
list_rasters = [file for file in os.listdir(rasdir) if file.lower().endswith(".tif")]

###################################################################
# run algo in parallel

# Initialize Ray
ray.init()

# Create a list to hold the output futures
output_futures = []

# Iterate through the inputs and run the function in parallel
for raster in list_rasters[:2]:
    # Call the function asynchronously and append the output future to the list
    output_futures.append(eq_impact.remote(bldg.head(1000),
                                           vuln,
                                           raster,
                                           rasdir))

# Wait for all the output futures to complete
outputs = ray.get(output_futures)

# write building datasets to disk
for i, output in enumerate(outputs):
    # Write the DataFrame to a CSV file
    output.to_csv(f'results/output_{list_rasters[i]}.csv', index=False)

    # Shutdown Ray
    ray.shutdown()
