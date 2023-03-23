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
resdir = Path('/Volumes/LaCie/workfolder/sajag_nepal/Ensemble_results/21_03_23')

# Load data
bldgs = gpd.read_file(datadir / "bldgs_preprocs_light_districts [NatBoundary].shp", usecol=["osm_id", "geometry"])

list_csv_eq = [file for file in os.listdir(resdir) if (file.endswith("eqImpact.csv") and file.startswith("UTM45"))]
list_csv_ls = [file for file in os.listdir(resdir) if (file.endswith("lsImpact.csv") and file.startswith("UTM45"))]

# Process earthquake impact results
impact_eq_results_high = []
impact_eq_results_mid = []
impact_eq_results_low = []
impact_ls_results = []

for (eq, ls) in tqdm(zip(list_csv_eq, list_csv_ls)):
    # read csv as dataframe
    df_eq = pd.read_csv(resdir / eq,
                        usecols=lambda col: col.endswith('.tif') or col.startswith('osm') or col.startswith(
                            'geometry'))
    df_ls = pd.read_csv(resdir / ls, 
                        usecols=lambda col: col.startswith('impact') or col.startswith('osm') or col.startswith(
                            'geometry'))
    impact_eq_results_low.append(df_eq.iloc[:, -1])
    impact_eq_results_mid.append(df_eq.iloc[:, -2])
    impact_eq_results_high.append(df_eq.iloc[:, -3])
    impact_ls_results.append(df_ls.iloc[:, -1])

# calculate mean value of impact from earthquake and landslide
arrays = [np.array(x) for x in impact_eq_results_low]
bldgs["average_eq_probability_complete_low"] = [np.median(k) for k in zip(*arrays)]
arrays = [np.array(x) for x in impact_eq_results_mid]
bldgs["average_eq_probability_complete_mid"] = [np.median(k) for k in zip(*arrays)]
arrays = [np.array(x) for x in impact_eq_results_high]
bldgs["average_eq_probability_complete_high"] = [np.median(k) for k in zip(*arrays)]
arrays = [np.array(x) for x in impact_ls_results]
bldgs["average_ls_probability"] = [np.median(k) for k in zip(*arrays)]

# attach ward to buildings
geobldgs = gpd.GeoDataFrame(bldgs, geometry='geometry')
ward = gpd.read_file("/Users/alexdunant/Library/CloudStorage/GoogleDrive-37stu37@gmail.com/My Drive/Projects/sajag-nepal/Workfolder/Hyperedge_mh_methodology/data/shapes/LocalbodiesWARD_753_UTM45N.shp")
ward = ward[["DISTRICT", "GaPa_NaPa", "NEW_WARD_N", "X", "Y", "geometry"]]
ward["ward_id"] = np.arange(0, len(ward), 1)

geobldgs_ward = gpd.sjoin(ward, geobldgs, how="left")
res_ward = geobldgs_ward.groupby("ward_id").agg({"average_eq_probability_complete_low": ['mean', 'median', 'sum', 'max', 'std'],
                                                 "average_eq_probability_complete_mid": ['mean', 'median', 'sum', 'max', 'std'],
                                                 "average_eq_probability_complete_high": ['mean', 'median', 'sum', 'max', 'std'], 
                                                 "average_ls_probability": ['mean', 'median', 'sum', 'max', 'std'], 
                                                 "geometry": 'first'}).reset_index()
res_ward.columns=res_ward.columns.droplevel(0)
res_ward.columns = ['ward_id', 'eq_mean_low', 'eq_median_low', 'eq_sum_low', 'eq_max_low', 'eq_std_low', 
                    'eq_mean_mid', 'eq_median_mid', 'eq_sum_mid', 'eq_max_mid', 'eq_std_mid',
                    'eq_mean_high', 'eq_median_high', 'eq_sum_high', 'eq_max_high', 'eq_std_high',
                    'ls_mean', 'ls_median', 'ls_sum', 'ls_max', 'ls_std', 'geometry']

res_ward = gpd.GeoDataFrame(res_ward, geometry='geometry', crs=ward.crs)
res_ward['high_sum'] = res_ward.eq_sum_high + res_ward.ls_sum
res_ward['mid_sum'] = res_ward.eq_sum_mid + res_ward.ls_sum
res_ward['low_sum'] = res_ward.eq_sum_low + res_ward.ls_sum

# write to disk
# geobldgs.to_file(GDrive / "code" / "results" / "mean_results_ensemble.gpkg", driver="GPKG")
res_ward.to_file(GDrive / "code" / "results" / "ward_ensemble_stats_220323.gpkg", driver="GPKG")
