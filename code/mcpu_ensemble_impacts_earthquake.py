from pathlib import Path
import os
import geopandas as gpd
import multiprocessing

from mods_vulnerability_ensemble import extract_vulnerability_parameters_from, probability_of_collapse_mean_parameters


def eq_impact(buildings, vulnerability, shake_ras, shake_dir):
    """
    Calculate the low, mid and high case for
    probability of collapse for every bldgs
    -> bldgs is a shapefile of all the buildings
    -> ras is a shaking footprint as a tif file
    """

    # Load libraries
    from mods_geom_ops import Pcentroid_Rsampling
    from mods_vulnerability_ensemble import join_buildings_and_vulnerabilities, calculate_probability_of_collapse

    
    print(f"Task {shake_ras} enter")

    # append raster value to buildings
    b = Pcentroid_Rsampling(buildings, shake_ras, shake_dir)
    
    # append the meteor vulnerability to buildings
    b = join_buildings_and_vulnerabilities(b, vulnerability)

    # extract the probabilities of collapse and attached to buildings
    b[f'high_{shake_ras}'], b[f'mid_{shake_ras}'], b[f'low_{shake_ras}'] = calculate_probability_of_collapse(b)

    b.to_csv(f'/Volumes/LaCie/workfolder/sajag_nepal/Ensemble_results/21_03_23/{shake_ras[8:-4]}__eqImpact.csv', index=False)

    print(f"Task {shake_ras} exit")
    
    return print(f'{shake_ras[8:-4]}__eqImpact.csv')


###################################################################
# Load datapath and datasets
GDrive = Path('/Users/alexdunant/Documents/Github/Ensemble_earthquake_Nepal')
# GDrive = Path('D:\Github\Ensemble_earthquake_Nepal')
datadir = GDrive / 'shp'
rasdir = GDrive / 'tif'

# Load data
bldg = gpd.read_file(datadir / "bldgs_preprocs_light_districts [NatBoundary].shp")
vuln = gpd.read_file(datadir / "npl-ic-exp-dist_UTM45.shp")
vuln = vuln[["size_dist", "cnt_dist", "geometry"]]

n, t = extract_vulnerability_parameters_from(vuln)
vuln = probability_of_collapse_mean_parameters(t, n, vuln)

list_rasters = [file for file in os.listdir(rasdir) if file.lower().endswith(".tif")]

print(list_rasters)

###################################################################
# run algo in parallel


if __name__ == '__main__':
    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=2)

    # Create a list of argument tuples
    argument_tuples = [(bldg, vuln, raster, rasdir) for raster in list_rasters]

    # Apply the function to each tuple of arguments in parallel using the starmap method
    results = pool.starmap(eq_impact, argument_tuples)

    # Close the pool of worker processes
    pool.close()
    pool.join()

    print(results)
