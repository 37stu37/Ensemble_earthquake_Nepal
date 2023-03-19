from pathlib import Path
import os
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import numpy as np

from mods_geom_ops import join_Pcentroid_at_Polygons_all_columns

# Load datapath and datasets
GDrive = Path('/Users/alexdunant/Documents/Github/Ensemble_earthquake_Nepal')
# GDrive = Path('D:\Github\Ensemble_earthquake_Nepal')
datadir = GDrive / 'shp'
resdir = Path('/Volumes/LaCie/tmp/results_full_ensemble')

# Load data
bldgs = gpd.read_file(datadir / "bldgs_preprocs_light_districts [NatBoundary].shp", usecol=["osm_id", "geometry"])

list_csv_eq = [file for file in os.listdir(resdir) if (file.endswith("eqImpact.csv") and file.startswith("UTM45"))]
list_csv_ls = [file for file in os.listdir(resdir) if (file.endswith("lsImpact.csv") and file.startswith("UTM45"))]

# Process earthquake impact results
impact_eq_results = []
impact_ls_results = []

for (eq, ls) in tqdm(zip(list_csv_eq, list_csv_ls)):
    # read csv as dataframe
    df_eq = pd.read_csv(resdir / eq,
                        usecols=lambda col: col.startswith('mid_') or col.startswith('osm') or col.startswith(
                            'geometry'))
    df_ls = pd.read_csv(resdir / ls,
                        usecols=lambda col: col.startswith('impact') or col.startswith('osm') or col.startswith(
                            'geometry'))
    impact_eq_results.append(df_eq.iloc[:, -1])
    impact_ls_results.append(df_ls.iloc[:, -1])

# calculate mean value of impact from earthquake and landslide
arrays = [np.array(x) for x in impact_eq_results]
bldgs["mean_eq_probability_complete"] = [np.mean(k) for k in zip(*arrays)]
arrays = [np.array(x) for x in impact_ls_results]
bldgs["mean_ls_probability"] = [np.mean(k) for k in zip(*arrays)]

# attach ward to buildings
geobldgs = gpd.GeoDataFrame(bldgs, geometry='geometry')
ward = gpd.read_file("/Users/alexdunant/Library/CloudStorage/GoogleDrive-37stu37@gmail.com/My Drive/Projects/sajag-nepal/Workfolder/Hyperedge_mh_methodology/data/shapes/LocalbodiesWARD_753_UTM45N.shp")

geobldgs_ward = join_Pcentroid_at_Polygons_all_columns(geobldgs, ward)

# write to disk
geobldgs_ward.to_file(GDrive / "code" / "results" / "mean_results_ensemble.gpkg", driver="GPKG")
