import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, mapping
import matplotlib.pyplot as plt
from pyproj import Proj, transform
import fiona
import shapefile
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
import numpy as np
from pyreproj import Reprojector
import fiona
import rasterio
import rasterio.mask

def createZone(lat,lon):
    rp = Reprojector()

    p1 = Point(lat, lon)
    print("initial point: {}".format(p1))
    p2 = rp.transform(p1, from_srs=4326, to_srs=2056)
    print("point after CRS transformation: {}".format(p2))

    buffer = p2.buffer(500)
    envelope = buffer.envelope
    #crs={'init': 'epsg:2056'}
    Zone = gpd.GeoDataFrame(crs=2056,geometry=gpd.GeoSeries(envelope))
    # save the neighbourhood shp file
    Zone.to_file('C:/notes/Location_validation/API_Django/env/zone2.shp')
    # now we need to extract the environmental variables from the Landcover classes and DEM
    # we aggregated the landcover classes to the following groups
    LC_types = [11, 12, 13, 14, 211, 22, 231, 24, 311, 312, 313, 321, 322, 324, 33, 4, 51]

    all_prop=[]
    all_prop=np.zeros(len(LC_types))
    sum=0
    geom= Zone.geometry
    #extracting landscape metrices
    with rasterio.open("C:/notes/Location_validation/API_Django/env/CLC_Aggr_100.tif") as src:
        no_data=src.nodata
        out_image, out_transform = rasterio.mask.mask(src, geom, crop=True)
    print(out_image.shape)
    out_image = np.extract(out_image != no_data, out_image)
    (unique, counts) = np.unique(out_image, return_counts=True)
    print(out_image.shape)
    frequencies = np.asarray((unique, counts)).T
    for el in frequencies[1:]:
        sum = sum + el[1]
        print(el[0],el[1])
        #print(sum)
    for i in range(len(LC_types)):
        for el in frequencies:
            if LC_types[i] == el[0]:
                all_prop[i] = el[1]/sum
    all_prop = all_prop.tolist()

 # extrcating average slope and average elevation in the zone
    sum_elev=0
    sum_slope=0
    with rasterio.open("C:/notes/Location_validation/API_Django/env/elev200_2.tif") as src:
        no_data=src.nodata
        elev, out_transform = mask(src, geom, crop=True)
    elev = np.extract(elev != no_data, elev)
    for el in elev:
        sum_elev = sum_elev + el
    average_elev = sum_elev/25

    with rasterio.open("C:/notes/Location_validation/API_Django/env/slope.tif") as src:
        no_data=src.nodata
        slope, out_transform = mask(src, geom, crop=True)
    slope = np.extract(slope != no_data, slope)
    for el in slope:
        sum_slope = sum_slope + el
    average_slope = sum_slope/25

    all_prop.append(average_elev)
    all_prop.append(average_slope)
    print(all_prop)
    input = pd.DataFrame(all_prop)
    input= input.transpose()
    input.columns=["Urban_fabr","Industrial","constructi","artificial","non_irriga","permanent_","pastues","agricultur","Broad_leav","Coniferous","Mixed_fore",
        "Natural_gr",
        "Moors_heat",
        "woodland_s",
        "no-vegetat",
        "wetland",
        "waterbodie",
        "Elev_meanm",
        "slope_mean"]

    print(input)
    return input
