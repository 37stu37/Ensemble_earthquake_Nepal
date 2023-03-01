from pathlib import Path
import geopandas as gpd
import time

from mods_geom_ops import join_Pcentroid_at_Pl
from mods_qc import plot_qc_f1_auc_threshold

# load results from mh algorithm
datadir = Path("G:\My Drive\Projects\sajag-nepal\Workfolder\Hyperedge_mh_methodology\data")
R = gpd.read_file(datadir.parent / "results" / "IMPACTS_bldgs_preprocs_light_districts [NatBoundary].shp")
E4polygons = gpd.read_file(datadir / "shapes" / "Epoch04_FINAL_TopologyFixed.shp")

#############################################################################
# Postprocessing
#############################################################################

print("joining model with actual occurence ...")

R = join_Pcentroid_at_Pl(R,
                         ["osm_id", "su_id", "DISTRICT", "GaPa_NaPa", "impact", "PGA_osm", "SI", "PGA_su", "FlowR_mean",
                          "geometry"],
                         E4polygons,
                         ["E4", "geometry"])

R["E4"].fillna(0, inplace=True)

################################################################################
# Get threshold scores per aggregated layer
agg_levels = ["osm_id", "su_id", "GaPa_NaPa", "DISTRICT"]

r2_list = []
timestr = time.strftime("%Y%m%d")
for a in agg_levels:
    thr, r_square = plot_qc_f1_auc_threshold(R, a)
    r2_list.append(r_square)
    thr.to_csv(datadir.parent / "results" / f"{timestr}_LS_impacts_thr_{a[1:-1]}")

print(agg_levels, r2_list)