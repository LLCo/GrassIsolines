'''
该函数被被用于从DEM求取山脉，以找到潜在草线区域。
'''
from osgeo import ogr,osr,gdal, gdal_array
import UNIT, cv2, os
import numpy as np


def shp_merge(path_src, path_out):
    output = os.popen("ogrmerge.py -single -o {} {}".format(path_out, path_src))
    print(output.read())


def createBuffer(inputfn, outputBufferfn, bufferDist):
    inputds = ogr.Open(inputfn)
    inputlyr = inputds.GetLayer()

    shpdriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputBufferfn):
        shpdriver.DeleteDataSource(outputBufferfn)
    outputBufferds = shpdriver.CreateDataSource(outputBufferfn)
    bufferlyr = outputBufferds.CreateLayer(outputBufferfn, geom_type=ogr.wkbPolygon)
    featureDefn = bufferlyr.GetLayerDefn()

    for feature in inputlyr:
        ingeom = feature.GetGeometryRef()
        geomBuffer = ingeom.Buffer(bufferDist)

        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(geomBuffer)
        bufferlyr.CreateFeature(outFeature)
        outFeature = None


def raster_to_shape(rasterfile,shapefile):
    data = gdal.Open(rasterfile, gdal.GA_ReadOnly)
    inband = data.GetRasterBand(1)
    drv = ogr.GetDriverByName('ESRI Shapefile')
    Polygon = drv.CreateDataSource(shapefile)
    prj = osr.SpatialReference()
    prj.ImportFromWkt(data.GetProjection()) ## 使用栅格的投影信息
    Polygon_layer = Polygon.CreateLayer(shapefile, srs=prj, geom_type = ogr.wkbMultiPolygon)
    newField = ogr.FieldDefn('Value', ogr.OFTInteger)
    Polygon_layer.CreateField(newField)
    gdal.FPolygonize(inband, inband, Polygon_layer, 0)


def processing(path_img_dem, path_mountain_raster, binary_thres=5500, size_thres=40):
    # binary
    img_dem, proj, geot = UNIT.img2numpy(path_img_dem, geoinfo=True)
    img_dem = np.where(img_dem > binary_thres, 1, 0)
    img_dem = img_dem.astype("uint8")
    num, labels, status, centroids = cv2.connectedComponentsWithStats(img_dem)
    # remove samll parcels
    for i in range(1, num):
        x, y, xlen, ylen, mountain_size = status[i]
        if mountain_size > size_thres:
            continue
        img_dem_subset = labels[y:y+ylen, x:x+xlen]
        img_dem_subset = np.where(img_dem_subset == i, 0, img_dem_subset)
        labels[y:y+ylen, x:x+xlen] = img_dem_subset
    UNIT.numpy2img(path_mountain_raster, labels, proj=proj, geot=geot)


if __name__ == "__main__":
    os.chdir("D:\licong\GrassIsolines\data\DEM")
    # processing("DangXiong_Mosaic_DEM_500.dat", "DangXiong_mountain_raster.tif")
    # raster_to_shape("DangXiong_mountain_raster.tif", "DangXiong_mountain.shp")
    # createBuffer("DangXiong_mountain.shp", "DangXiong_grassisoline_buffer.shp", 0.1)
    shp_merge("DangXiong_grassisoline_buffer_merge.shp", "DangXiong_grassisoline_buffer.shp")
    pass