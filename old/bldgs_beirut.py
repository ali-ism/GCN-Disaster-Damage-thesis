import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import cv2
from tqdm import tqdm

gdf = gpd.read_file("D:/ArcGIS/BeirutPort/buffered_masks.shp")
gdf = gdf[['damage_num', 'Longitude', 'Latitude', 'geometry']]

preimg = rasterio.open("D:/ArcGIS/BeirutPort/PreExp.tif")
postimg = rasterio.open("D:/ArcGIS/BeirutPort/PostExp.tif")

target_path = 'datasets/beirut_bldgs/'

for index, row in tqdm(gdf.iterrows()):
    out_pre, _ = mask(preimg, shapes=[row['geometry']], crop=True)
    out_post, _ = mask(postimg, shapes=[row['geometry']], crop=True)
    cv2.imwrite(target_path+f'bldg_{index}_pre.png', np.moveaxis(out_pre,0,-1))
    cv2.imwrite(target_path+f'bldg_{index}_post.png', np.moveaxis(out_post,0,-1))

preimg.close()
postimg.close()

gdf.drop(columns='geometry').to_csv(target_path+'beirut_labels.csv')