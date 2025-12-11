# heatmaps_and_plots.py
import rasterio, numpy as np, matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.plot import show

ndvi_path = "data/processed/indices/sentinel_ndvi.tif"
evi_path = "data/processed/indices/sentinel_evi.tif"
rlvi_path = "outputs/rlvi_map.tif"  # created after applying learned RL-VI on all pixels

def plot_map(path, title, out):
    with rasterio.open(path) as src:
        arr = src.read(1)
        plt.figure(figsize=(8,6))
        plt.imshow(arr, cmap='RdYlGn')
        plt.colorbar()
        plt.title(title)
        plt.axis('off')
        plt.savefig(out, dpi=300)
        plt.close()

plot_map(ndvi_path, "Sentinel-2 NDVI", "figures/figure11_ndvi.png")
plot_map(evi_path, "Sentinel-2 EVI", "figures/figure11_evi.png")
if os.path.exists(rlvi_path):
    plot_map(rlvi_path, "RL-VI Map", "figures/figure9_rlvi.png")
print("Saved maps.")
