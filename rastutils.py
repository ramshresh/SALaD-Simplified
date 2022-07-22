# https://gis.stackexchange.com/questions/199477/gdal-python-cut-geotiff-image
from osgeo import gdal, osr, ogr
import os

def raise_(e): raise e

def get_shp_extent(SHP):
        drv = ogr.GetDriverByName("ESRI Shapefile")
        if not os.path.isfile(SHP):
            raise Exception("file Not found {}".format(SHP))
        ds_region = drv.Open(SHP)
        lyr_region = ds_region.GetLayer()
        ulx, lrx, lry, uly  = lyr_region.GetExtent() # x_min, x_max, y_min, y_max
        assert( ulx < lrx)
        assert( uly > lry)
        lyr_region=None
        ds_region=None
        return (ulx, lrx, lry, uly)

def get_raster_extent(RAST):
    # Compute mean of each objects
    img = gdal.Open( RAST )
    rows = img.RasterYSize
    cols = img.RasterXSize
    geo=img.GetGeoTransform()
    img = None
    ulx = geo[0]
    uly = geo[3]
    lrx = ulx + geo[1] * cols
    lry = uly + geo[5] * rows

    return (ulx, lrx, lry, uly)

def extent_to_poly(extent, fpath):
    ulx, lrx, lry, uly  = extent
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(ulx, uly)
    ring.AddPoint(ulx, lry)
    ring.AddPoint(lrx, lry)
    ring.AddPoint(lrx, uly)
    ring.AddPoint(ulx, uly)
    geom = ogr.Geometry(ogr.wkbPolygon)
    geom.AddGeometry(ring)

    # Now convert it to a shapefile with OGR    
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(fpath)
    layer = ds.CreateLayer('', None, ogr.wkbPolygon)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()
    # Create a new feature (attribute and geometry)
    feat = ogr.Feature(defn)
    feat.SetField('id', 1)

    # Make a geometry, from Shapely object
    feat.SetGeometry(geom)
    layer.CreateFeature(feat)
    feat = geom = None  # destroy these
    # Save and close everything
    ds = layer = feat = geom = None


def check_extent(extent_shp, extent_rast):
    ulx_shp, lrx_shp, lry_shp, uly_shp = extent_shp
    ulx_rast, lrx_rast, lry_rast, uly_rast = extent_rast

    assert all([ulx_shp >= ulx_rast, uly_shp <= uly_rast, lrx_shp <= lrx_rast, lry_shp >= lry_rast]), \
        raise_(Exception('Extent of landslides shapefile not within the extent of raster'))
    


"""
p1 = TopLeft = (ulx, uly)
p2 = BottomRight = (lrx, lry)
extent = ulx, lrx, lry, uly
"""
def clip(filename, output_file, extent):
    ulx, lrx, lry, uly = extent
    # p1 = (355217.199739, 4473171.2377)
    # p2 = (355911.113396, 4472582.9196)
    p1 = (ulx, uly)
    p2 = (lrx, lry)
    
    driver = gdal.GetDriverByName('GTiff')

    dataset = gdal.Open(filename)


    cols = dataset.RasterXSize
    rows = dataset.RasterYSize

    print (cols, rows)

    transform = dataset.GetGeoTransform()

    print (transform)

    

    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    print (xOrigin, yOrigin)

    i1 = int((p1[0] - xOrigin) / pixelWidth)
    j1 = int((yOrigin - p1[1] ) / pixelHeight)
    i2 = int((p2[0] - xOrigin) / pixelWidth)
    j2 = int((yOrigin - p2[1]) / pixelHeight)

    print (i1, j1)
    print (i2, j2)

    new_cols = i2-i1+1
    new_rows = j2-j1+1

    new_x = xOrigin + i1*pixelWidth
    new_y = yOrigin - j1*pixelHeight

    print (new_x, new_y)
    
    new_transform = (new_x, transform[1], transform[2], new_y, transform[4], transform[5])

    # Create gtif file 
    driver = gdal.GetDriverByName("GTiff")
    nband = dataset.RasterCount
    dst_ds = driver.Create(output_file, new_cols, new_rows, nband,  gdal.GDT_Float32)
    
    for band in range( nband ):
        band += 1
        srcband = dataset.GetRasterBand(band)
        # band = dataset.GetRasterBand(1)
        srcData = srcband.ReadAsArray(i1, j1, new_cols, new_rows)
        print (srcData)
        #writting output raster
        dst_ds.GetRasterBand(band).WriteArray( srcData )

    #setting extension of output raster
    # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    dst_ds.SetGeoTransform(new_transform)

    wkt = dataset.GetProjection()

    # setting spatial reference of output raster 
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dst_ds.SetProjection( srs.ExportToWkt() )

    #Close output raster dataset 
    dataset = None
    dst_ds = None