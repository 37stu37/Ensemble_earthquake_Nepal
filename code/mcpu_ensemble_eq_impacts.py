import csv
import multiprocessing as mp
from pathlib import Path
import os
import geopandas as gpd

from ensmble_eq_impact import eq_impact


def worker(input_bldgs,
           input_vuln,
           input_ras,
           input_ras_dir):

    # Call the function with the input
    output = eq_impact(input_bldgs,
                       input_vuln,
                       input_ras,
                       input_ras_dir)

    # Save the output as a CSV file
    with open(f"{input_ras}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(output)


def main(input_bldgs,
         input_vuln,
         inputs_ras):
    # Create a process pool with the number of available CPUs
    pool = mp.Pool(mp.cpu_count())

    # Iterate over the list of inputs and create a process for each input
    results = []
    for input_ras in inputs_ras:
        result = pool.apply_async(worker, args=(input_bldgs, input_vuln, input_ras,))
        results.append(result)

    # Wait for all processes to complete and gather the results
    for result in results:
        result.wait()

    # Close the process pool
    pool.close()


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

# run algo in parallel
main(bldg, vuln, list_rasters)
