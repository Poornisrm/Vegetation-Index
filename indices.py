# indices.py
import numpy as np
import rasterio
from rasterio.transform import from_origin
import os
from skimage import io

OUT = "data/processed/indices"
os.makedirs(OUT, exist_ok=True)

def ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-9)

def evi(nir, red, blue):
    return 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + 1e-9)

# RGB indices
def exg(img):  # img in 0-1 float
    R,G,B = img[:,:,0], img[:,:,1], img[:,:,2]
    return 2*G - R - B

def vari(img):
    R,G,B = img[:,:,0], img[:,:,1], img[:,:,2]
    return (G - R) / (G + R - B + 1e-9)

# For sentinel GeoTIFFs: read arrays and compute
SENTINEL_BAND_PATHS = {
    'B04': 'data/processed/sentinel/B04_10m.tif', # Red
    'B05': 'data/processed/sentinel/B05_10m.tif', # Red-edge
    'B08': 'data/processed/sentinel/B08_10m.tif', # NIR
    'B11': 'data/processed/sentinel/B11_10m.tif', # SWIR1
    'B12': 'data/processed/sentinel/B12_10m.tif', # SWIR2
    'B02': 'data/processed/sentinel/B02_10m.tif', # Blue
    'B03': 'data/processed/sentinel/B03_10m.tif', # Green
}
# load minimal set
def load_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        meta = src.meta
    return arr, meta

# compute NDVI/EVI using available bands
if os.path.exists(SENTINEL_BAND_PATHS['B08']) and os.path.exists(SENTINEL_BAND_PATHS['B04']):
    nir, meta = load_band(SENTINEL_BAND_PATHS['B08'])
    red, _ = load_band(SENTINEL_BAND_PATHS['B04'])
    blue, _ = load_band(SENTINEL_BAND_PATHS['B02']) if os.path.exists(SENTINEL_BAND_PATHS['B02']) else (np.zeros_like(nir), None)
    ndvi_arr = ndvi(nir, red)
    evi_arr = evi(nir, red, blue)
    # save
    meta.update(dtype=rasterio.float32, count=1)
    with rasterio.open(os.path.join(OUT,'sentinel_ndvi.tif'),'w',**meta) as dst:
        dst.write(ndvi_arr.astype(np.float32),1)
    with rasterio.open(os.path.join(OUT,'sentinel_evi.tif'),'w',**meta) as dst:
        dst.write(evi_arr.astype(np.float32),1)
    print("Saved sentinel NDVI/EVI")

# compute RGB indices for processed mobile images
mobile_dir = "data/processed/mobile"
for fname in os.listdir(mobile_dir):
    if not fname.lower().endswith(('.jpg','.png','.jpeg')): continue
    img = io.imread(os.path.join(mobile_dir,fname))/255.0
    exg_arr = exg(img)
    vari_arr = vari(img)
    np.save(os.path.join(OUT, f"{fname}_exg.npy"), exg_arr)
    np.save(os.path.join(OUT, f"{fname}_vari.npy"), vari_arr)
print("Saved RGB indices.")
