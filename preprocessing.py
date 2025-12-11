# preprocessing.py
import os
import numpy as np
from skimage import io, transform
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Paths (edit)
SENTINEL_DIR = "data/sentinel_raw"   # or leave None if using GEE
OUTPUT_DIR = "data/processed"
MOBILE_DIR = "data/mobile_raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR,"sentinel"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR,"mobile"), exist_ok=True)

# --- Simple local Sentinel resampling function (if you have sentinel bands as GeoTIFFs) ---
def resample_to_10m(src_path, dst_path, dst_crs='EPSG:4326', dst_resolution=10):
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=dst_resolution)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count+1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear)

# Example: copy/resample specific bands B2,B3,B4,B5,B6,B8,B11,B12
BAND_FILES = {
    'B02': 'B02.tif', 'B03':'B03.tif', 'B04':'B04.tif', 'B05':'B05.tif',
    'B06':'B06.tif','B08':'B08.tif','B11':'B11.tif','B12':'B12.tif'
}
for name, fname in BAND_FILES.items():
    src = os.path.join(SENTINEL_DIR, fname)
    dst = os.path.join(OUTPUT_DIR, "sentinel", f"{name}_10m.tif")
    if os.path.exists(src):
        print("Resampling", src)
        resample_to_10m(src, dst)
    else:
        print("Missing", src, "- skip (use GEE or provide file)")

# --- Mobile preprocessing: crop and resize ---
TARGET_SIZE = (256,256)
for fname in os.listdir(MOBILE_DIR):
    if not fname.lower().endswith(('.jpg','.jpeg','.png')): continue
    img = io.imread(os.path.join(MOBILE_DIR, fname))
    # naive center crop to square then resize
    h, w = img.shape[:2]
    m = min(h,w)
    cy, cx = h//2, w//2
    crop = img[cy-m//2:cy+m//2, cx-m//2:cx+m//2]
    small = transform.resize(crop, TARGET_SIZE, anti_aliasing=True)
    out_path = os.path.join(OUTPUT_DIR, "mobile", fname)
    io.imsave(out_path, (small*255).astype(np.uint8))
    print("Saved mobile:", out_path)

print("Preprocessing complete.")
