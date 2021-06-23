# add environmental variables to all the pres-background datatset (for all the 101 species)
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, mapping
import matplotlib.pyplot as plt
import pyproj
from pyproj import Proj, transform
import fiona
#import shapefile
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
import numpy as np
from pyreproj import Reprojector
import fiona
import rasterio
import rasterio.mask
import time

def createZone(lat,lon):
    start = time.time()
    rp = Reprojector()

    p1 = Point(lat,lon)
    #print(p1)
    print("initial point: {}".format(p1))
    p2 = rp.transform(p1, from_srs=4326, to_srs=2056)
    #print("point after CRS transformation: {}".format(p2))

    buffer = p2.buffer(1000)
    envelope = buffer.envelope
    #crs={'init': 'epsg:2056'}
    Zone = gpd.GeoDataFrame(crs=2056,geometry=gpd.GeoSeries(envelope))
    # save the neighbourhood shp file
    #Zone.to_file('C:/notes/Location_validation/API_Django/env/zone2.shp')
    # now we need to extract the environmental variables from the Landcover classes and DEM
    # we aggregated the landcover classes to the following groups
    LC_types = [1, 211, 22, 231, 24, 311, 312, 313, 321, 322, 324, 3312, 3334,335, 4, 51]

    all_prop=[]
    all_prop=np.zeros(len(LC_types))
    sum_lc=0
    geom= Zone.geometry
    #extracting landscape metrices
    #print("start")
    with rasterio.open("CLC_Aggr_100_new.tif") as src:
        no_data=src.nodata
        out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
    #print(out_image.shape)
    out_image = np.extract(out_image != no_data, out_image)
    (unique, counts) = np.unique(out_image, return_counts=True)
    #print(out_image.shape)
    frequencies = np.asarray((unique, counts)).T
    #print(frequencies)
    for el in frequencies:
      #print(el)
      sum_lc = sum_lc + el[1]
      #print(el[0],el[1])
    #print(sum_lc)
    for i in range(len(LC_types)):
      for el in frequencies:
        if LC_types[i] == el[0]:
          all_prop[i] = el[1]/sum_lc
    all_prop = all_prop.tolist()

 # extrcating average slope and average elevation in the zone
    sum_elev=0
    sum_slope=0
    with rasterio.open("elev200_2.tif") as src:
        no_data=src.nodata
        elev, out_transform = mask(src, geom, crop=True)
    elev = np.extract(elev != no_data, elev)
    for el in elev:
      sum_elev = sum_elev + el
    average_elev = sum_elev/100

    with rasterio.open("slope.tif") as src:
        no_data=src.nodata
        slope, out_transform = mask(src, geom, crop=True)
    slope = np.extract(slope != no_data, slope)
    for el in slope:
      sum_slope = sum_slope + el
    average_slope = sum_slope/100



    # extrcating average NDVI
    sum=0
    value =0

    with rasterio.open("NDVI_mean_2056.tif") as src:
        no_data=src.nodata

        out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
        #print(out_image.shape)
        out_image = np.extract(out_image != no_data, out_image)
    (unique, counts) = np.unique(out_image, return_counts=True)
    #print(out_image.shape)
    frequencies = np.asarray((unique, counts)).T
    for el in frequencies:
      sum = sum + el[1]
      value = value+ el[0]
    #print(sum)
    if sum==0:
      average_ndvi=-999
    else:
      average_ndvi = value/sum

    all_prop.append(average_elev)
    all_prop.append(average_slope)
    all_prop.append(average_ndvi)

    input = pd.DataFrame(all_prop)
    input= input.transpose()
    input.columns=['artificial_surfaces ', 'Non-irrigated_arable', 'Permanent_crops',
       'Pastures', 'Heterogeneous_agri', 'Broad-leaved_forest ',
       'Coniferous_forest ', 'Mixed_forest ', 'Natural_grassland',
       'Moors_heathland', 'Transitional_woodland', 'beaches_bare rocks',
       'sparsely_vegetated', 'Glaciers', 'wetland', 'waterbodie', 'Elev_meanm',
       'slope_mean', 'ndvi_mean']

    #print(input)
    #print(time.time() - start)
    return input

# get grid_id in order to see where is the observation point in the swiss grids,
# and based on the grid id query from the db and give suggestions for the possible species to be observed in a given location

def getGridId(lat,lon, grid):
    lst_ind =[]
    #rp = Reprojector()
    # wgs84 = pyproj.Proj(init='epsg:4326')
    # ch1903 = pyproj.Proj(init='epsg:2056')
    # #p = Point(lat,lon)
    # lon2, lat2 = pyproj.transform(wgs84, ch1903, lon, lat)
    p = Point(lon,lat)
    print(p)
    bool = grid.geometry.contains(p)
    bool = pd.DataFrame(bool)
    for index , row in bool.iterrows():
        if row[0] == True:
            lst_ind.append(index)
    print(lst_ind)
    return lst_ind
