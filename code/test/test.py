import csv
import multiprocessing as mp
from pathlib import Path
import os
import geopandas as gpd

from ensmble_eq_impact import eq_impact


###################################################################
# Load datapath and datasets
datadir = Path("/Users/alexdunant/Documents/Github/Ensemble_earthquake_Nepal/shp")
rasdir = Path("/Users/alexdunant/Documents/Github/Ensemble_earthquake_Nepal/tif")

# Load data
bldg = gpd.read_file(datadir / "bldgs_preprocs_light_districts [NatBoundary].shp")
vuln = gpd.read_file(datadir / "npl-ic-exp-dist_UTM45.shp")
vuln = vuln[["size_dist", "cnt_dist", "geometry"]]
list_rasters = [file for file in os.listdir(rasdir) if file.lower().endswith(".tif")]

# test run
test = eq_impact(bldgs=bldg.head(500),
                 bldgs_vuln=vuln,
                 shake_ras=list_rasters[0],
                 shake_dir=rasdir)




import ray
import pandas as pd

# Define the function to run in parallel
@ray.remote
def my_function(input):
    # Do some computation on the input
    output = input * 2
    return output

# Define the list of inputs
inputs = [1, 2, 3, 4, 5]

# Initialize Ray
ray.init()

# Create a list to hold the output futures
output_futures = []

# Iterate through the inputs and run the function in parallel
for input in inputs:
    # Call the function asynchronously and append the output future to the list
    output_futures.append(my_function.remote(input))

# Wait for all the output futures to complete
outputs = ray.get(output_futures)

# Create a Pandas DataFrame from the outputs
df = pd.DataFrame(outputs, columns=['Output'])

# Write the DataFrame to a CSV file
df.to_csv('output.csv', index=False)

# Shutdown Ray
ray.shutdown()

