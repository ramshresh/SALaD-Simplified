import os
import glob
from osgeo import gdal, ogr, osr
import otbApplication
import parameters
import filehelpers

os.environ["OTB_MAX_RAM_HINT"] = "50000"

class Segmentation:
    # names
    final_name, dem_name  = None, None
    # Directory and Paths
    TEMP_DIR = None
    
    # Output File Names
    outfile_sm, outfile_seg, outfile_merg, outfile_segment = None, None, None, None

    def __init__(self, IMG_TIFF, OUT_DIR, window=None, logging=None,) -> None:
        self.logging = logging
        self.window = window
        self.IMG_TIFF = IMG_TIFF
        self.OUT_DIR = OUT_DIR

        self.hs=parameters.hs
        self.hr=parameters.hr
        self.min_size=parameters.min_size

        # Initialize result directory for intermediate results of training
        self.TEMP_DIR = os.path.join(self.OUT_DIR, 'segmentation/')
        filehelpers.ensure_dir(self.TEMP_DIR)

        # Initialize name identifiers 
        self.final_name = os.path.basename(self.IMG_TIFF).split('.')[0]
        

        #set output filenames
        self.outfile_sm = os.path.join(self.TEMP_DIR, "sm_"+self.final_name+".tif")
        self.outfile_seg = os.path.join(self.TEMP_DIR, "seg_"+self.final_name+".tif")
        self.outfile_merg = os.path.join(self.TEMP_DIR, "merg_"+self.final_name+".tif")
        self.outfile_segment = os.path.join(self.TEMP_DIR, "segment_"+self.final_name+".shp")


    def run(self):
        logging = self.logging
        os.chdir(self.TEMP_DIR)
        logging.info("[Segmentation] - MeanShiftSmoothing")
            
        # The following line creates an instance of the MeanShiftSmoothing application
        MeanShiftSmoothing = otbApplication.Registry.CreateApplication("MeanShiftSmoothing")
        # The following lines set all the application parameters:
        MeanShiftSmoothing.SetParameterString("in", self.IMG_TIFF)
        MeanShiftSmoothing.SetParameterString("fout", self.outfile_sm)
        MeanShiftSmoothing.SetParameterInt("spatialr", self.hs)
        MeanShiftSmoothing.SetParameterFloat("ranger", self.hr)
        MeanShiftSmoothing.SetParameterFloat("thres", 0.1)
        MeanShiftSmoothing.SetParameterInt("maxiter", 100)
        # The following line execute the application
        MeanShiftSmoothing.ExecuteAndWriteOutput()

        logging.info("[Segmentation] - LSMSSegmentation")
        # The following line creates an instance of the LSMSSegmentation application
        LSMSSegmentation = otbApplication.Registry.CreateApplication("LSMSSegmentation")
        # The following lines set all the application parameters:
        LSMSSegmentation.SetParameterString("in", self.outfile_sm)
        LSMSSegmentation.SetParameterString("out", self.outfile_seg)
        LSMSSegmentation.SetParameterFloat("spatialr", self.hs)
        LSMSSegmentation.SetParameterFloat("ranger", self.hr)
        LSMSSegmentation.SetParameterInt("minsize", 0)
        LSMSSegmentation.SetParameterInt("tilesizex", 500)
        LSMSSegmentation.SetParameterInt("tilesizey", 500)
        # The following line execute the application
        LSMSSegmentation.ExecuteAndWriteOutput()

        logging.info("[Segmentation] - LSMSSmallRegionsMerging")
        # The following line creates an instance of the LSMSSmallRegionsMerging application
        LSMSSmallRegionsMerging = otbApplication.Registry.CreateApplication("LSMSSmallRegionsMerging")
        # The following lines set all the application parameters:
        LSMSSmallRegionsMerging.SetParameterString("in", self.IMG_TIFF)
        LSMSSmallRegionsMerging.SetParameterString("inseg", self.outfile_seg)
        LSMSSmallRegionsMerging.SetParameterString("out", self.outfile_merg)
        LSMSSmallRegionsMerging.SetParameterInt("minsize", self.min_size)
        LSMSSmallRegionsMerging.SetParameterInt("tilesizex", 500)
        LSMSSmallRegionsMerging.SetParameterInt("tilesizey", 500)
        # The following line execute the application
        LSMSSmallRegionsMerging.ExecuteAndWriteOutput()

        logging.info("[Segmentation] -  Convert LSMSSmallRegionsMerging output raster to polygon")
        # Convert LSMSSmallRegionsMerging output raster to polygon
        # Get spatial reference from LSMSSmallRegionsMerging raster 
        src_ds = gdal.Open( self.outfile_merg )
        srcband = src_ds.GetRasterBand(1)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(src_ds.GetProjection())
        
        #Create output shapefile
        logging.info("[Segmentation] -  Convert LSMSSmallRegionsMerging output raster to polygon - Create output shapefile")
        drv = ogr.GetDriverByName("ESRI Shapefile")
        dst_ds = drv.CreateDataSource(self.outfile_segment)
        dst_layer = dst_ds.CreateLayer(self.outfile_segment, srs = srs )
        gdal.Polygonize( srcband, None, dst_layer, -1, [], callback=None )
        dst_ds = None

        logging.info("[Segmentation] -  Delete temporary files < seg_*_FINAL.tif > created during segmentation")
        # Delete temporary files created during segmentation
        for f in glob.glob("seg_*_FINAL.tif"):
            os.remove(f)
        logging.info("[Segmentation] -  Successfully Completed!!")
        
        return (self.outfile_segment, )
# img_tiff = 'C:/wbtraining/image/pasang_2015_12_28_clip.tif'
# out_dir = 'C:/wbtraining/image/results/'
# segmentation = Segmentation(img_tiff, out_dir)
# segmentation.run()