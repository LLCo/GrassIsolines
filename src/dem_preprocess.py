'''
该函数被被用于从DEM求取山脉，以找到存在潜在草线区域的山脉。

1. 输入数据为 500m 的DEM数据，获得各像素点。
2. 将500m的DEM数据做宽度为50KM的均值滤波，求取像元所在位置的整体坡度，小于2000m均视为Nan值，以排除非高原地区干扰。
3. 将 500m DEM 与 50KM DEM 做差，求取得山地相对于局部海拔高度。
4. 山体DEM > 4000m，局部 DEM > 200m的山峰，才作为我们的研究对象。
5. 同时，删除面积小于10KM2的山体。

'''
from osgeo import ogr,osr,gdal, gdal_array
import UNIT, cv2, os
import numpy as np
from scipy.ndimage import generic_filter, binary_closing


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


def raster_to_shape_multiple(rasterfile, shapefile):
    dts = gdal.Open(rasterfile, gdal.GA_ReadOnly)
    data = dts.ReadAsArray()
    unique = np.unique(data)

    gdal_type = UNIT.NP2GDAL_CONVERSION[str(data.dtype)]
    src_ds = gdal.GetDriverByName('MEM').Create('', data.shape[1], data.shape[0], 1, gdal_type)
    src_ds.SetProjection(dts.GetProjection())
    src_ds.SetGeoTransform(dts.GetGeoTransform())
    inband = src_ds.GetRasterBand(1)

    for i in range(1, len(unique)):
        data_temp = np.where(data == unique[i], 1, 0)
        data_temp = binary_closing(data_temp)
        drv = ogr.GetDriverByName('ESRI Shapefile')
        Polygon = drv.CreateDataSource(shapefile.format(i))
        prj = osr.SpatialReference()
        prj.ImportFromWkt(dts.GetProjection()) ## 使用栅格的投影信息
        Polygon_layer = Polygon.CreateLayer(shapefile, srs=prj, geom_type = ogr.wkbMultiPolygon)
        newField = ogr.FieldDefn('Value', ogr.OFTInteger)
        Polygon_layer.CreateField(newField)

        inband.WriteArray(data_temp)
        gdal.FPolygonize(inband, inband, Polygon_layer, 0)
    return len(unique)


def raster_to_shape(rasterfile, shapefile):
    dts = gdal.Open(rasterfile, gdal.GA_ReadOnly)
    inband = dts.GetRasterBand(1)
    drv = ogr.GetDriverByName('ESRI Shapefile')
    Polygon = drv.CreateDataSource(shapefile)
    prj = osr.SpatialReference()
    prj.ImportFromWkt(dts.GetProjection()) ## 使用栅格的投影信息
    Polygon_layer = Polygon.CreateLayer(shapefile, srs=prj, geom_type=ogr.wkbMultiPolygon)
    newField = ogr.FieldDefn('Value', ogr.OFTInteger)
    Polygon_layer.CreateField(newField)
    gdal.FPolygonize(inband, inband, Polygon_layer, 0)
    return


def processing(path_img_dem, path_mountain_raster, dem_thres=5100, local_thres=200, size_thres=100):
    # binary
    img_dem, proj, geot = UNIT.img2numpy(path_img_dem, geoinfo=True)
    img_dem[np.where(img_dem < 2000)] = np.nan  # filter the no-plateau area
    img_dem_50km = generic_filter(img_dem, np.nanmean, size=100, mode='nearest')

    area_dem = np.where(img_dem > dem_thres, 1, 0)
    area_local = np.where(img_dem - img_dem_50km > local_thres, 1, 0)
    area = np.where(np.logical_and(area_dem, area_local), 1, 0).astype("uint8")
    num, labels, status, centroids = cv2.connectedComponentsWithStats(area)

    # remove samll parcels
    for i in range(1, num):
        x, y, xlen, ylen, mountain_size = status[i]
        if mountain_size > size_thres:
            continue
        # removing
        area_subset = labels[y:y+ylen, x:x+xlen]
        area_subset = np.where(area_subset == i, 0, area_subset)
        labels[y:y+ylen, x:x+xlen] = area_subset
    labels = np.where()
    UNIT.numpy2img(path_mountain_raster, labels, proj=proj, geot=geot)
    # UNIT.numpy2img("labels.tif", labels, proj=proj, geot=geot)
    # UNIT.numpy2img("area_dem.tif", area_dem, proj=proj, geot=geot)
    # UNIT.numpy2img("area_local.tif", area_local, proj=proj, geot=geot)


if __name__ == "__main__":
    os.chdir("D:\licong\GrassIsolines\data\DEM")
    processing("DangXiong_Mosaic_DEM_500.dat", "DangXiong_mountain_raster.tif")
    raster_to_shape("DangXiong_mountain_raster.tif", "DangXiong_mountain.shp")
    createBuffer("DangXiong_mountain.shp", "DangXiong_mountain_buffer.shp", 0.05)
    # shp_merge("DangXiong_grassisoline_buffer_merge.shp", "DangXiong_grassisoline_buffer.shp")
    pass
