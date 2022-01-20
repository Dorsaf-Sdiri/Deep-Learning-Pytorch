from contextlib import closing
import shutil
import urllib.request as request
from osgeo import gdal
import numpy as np
import sys

n_normal = 200
n_tumor = 200

base_url = (
    "ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/"
)
normal = "training/normal/normal_{}.tif"
tumor = "training/tumor/tumor_{}.tif"

base_url = (
    "ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/"
)
normal = "training/normal/normal_{}.tif"
tumor = "training/tumor/tumor_{}.tif"
normal_paths = [
    base_url + normal.format(str(i).zfill(3)) for i in range(1, min(160, n_normal))
]
tumor_paths = [
    base_url + tumor.format(str(i).zfill(3)) for i in range(1, min(111, n_tumor))
]
print(len(tumor_paths))
print(len(normal_paths))


def download_tif(ftp_path, dbfs_path):
    with closing(request.urlopen(ftp_path)) as r:
        with open(dbfs_path, "wb") as f:
            shutil.copyfileobj(r, f)


for i in normal_paths:
    download_tif(i, "./train_images/" + i.split("/")[-1])
for i in tumor_paths:
    download_tif(i, "./train_images/" + i.split("/")[-1])
# plot tiffs with less memory consumption by 66%


try:
    src_ds = gdal.Open("./train_images_WSI/normal_001.tif")
except RuntimeError as e:
    print("Unable to open INPUT.tif")
    print(e)
    sys.exit(1)
try:
    src_ds = src_ds.GetRasterBand(1)
except RuntimeError as e:
    # for example, try GetRasterBand(10)
    print("Enable to allocate memory of 20 GB")
    print(e)
    sys.exit(1)

src_ds.FlushCache()

channel = np.array(src_ds.ReadAsArray())
channel.FlushCache()
