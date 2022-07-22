import os
import glob
from rasterstats import zonal_stats
import geopandas as gpd
import numpy as np
import time
from affine import Affine
from osgeo import gdal
import filehelpers



class ZonalStats:
    def __init__(self, OUT_DIR, window=None, logging=None,) -> None:
        self.window = window
        self.logging = logging
        t1=time.time()
        self.OUT_DIR = OUT_DIR
        self.SEGMENTATION_DIR = os.path.join(self.OUT_DIR, 'segmentation/')
        os.chdir(self.SEGMENTATION_DIR)
        segmented_poly=glob.glob("segment*.shp")
        segmented_filename_with_ext=segmented_poly[0]
        self.segmented_shp = os.path.join(self.SEGMENTATION_DIR, segmented_filename_with_ext)
        print("Finished found segment*.shp in",((time.time()-t1)/60), "mins")

        # Initialize result directory for intermediate results of training
        self.TEMP_DIR = os.path.join(self.OUT_DIR, 'zonalstats/')
        filehelpers.ensure_dir(self.TEMP_DIR)

        
        self.EXPLANATOR_DIR = os.path.join(self.OUT_DIR, 'explanator/')
        os.chdir(self.EXPLANATOR_DIR)
        ndvi_file=glob.glob("ndvi_*.tif")
        ndvi_filename_with_ext=ndvi_file[0]
        self.ndvi_tiff = os.path.join(self.EXPLANATOR_DIR, ndvi_filename_with_ext)
        print("Finished found ndvi_*.tif in",((time.time()-t1)/60), "mins")

        bright_file=glob.glob("bright_*.tif")
        bright_filename_with_ext=bright_file[0]
        self.bright_tiff = os.path.join(self.EXPLANATOR_DIR, bright_filename_with_ext)
        print("Finished found bright_*.tif in",((time.time()-t1)/60), "mins")

        slope_file=glob.glob("slope_clipped_*.tif")
        slope_filename_with_ext=slope_file[0]
        self.slope_tiff = os.path.join(self.EXPLANATOR_DIR, slope_filename_with_ext)
        print("Finished found slope_*.tif in",((time.time()-t1)/60), "mins")

        homog_file=glob.glob("homog_*.tif")
        homog_filename_with_ext=homog_file[0]
        self.homog_tiff = os.path.join(self.EXPLANATOR_DIR, homog_filename_with_ext)
        print("Finished found homog_*.tif in",((time.time()-t1)/60), "mins")

        mean_file=glob.glob("mean_*.tif")
        mean_filename_with_ext=mean_file[0]
        self.mean_tiff = os.path.join(self.EXPLANATOR_DIR, mean_filename_with_ext)
        print("Finished found mean_*.tif in",((time.time()-t1)/60), "mins")

        # Output File Path
        self.segmfinal_shp = os.path.join(self.TEMP_DIR, "segfinal_"+segmented_filename_with_ext)
        
        if not os.path.isfile(self.segmented_shp):
            raise Exception("File Doesnot Exist: {}".format(self.segmented_shp))
        if not os.path.isfile(self.ndvi_tiff):
            raise Exception("File Doesnot Exist: {}".format(self.ndvi_tiff))
        if not os.path.isfile(self.bright_tiff):
            raise Exception("File Doesnot Exist: {}".format(self.bright_tiff))
        if not os.path.isfile(self.slope_tiff):
            raise Exception("File Doesnot Exist: {}".format(self.slope_tiff))
        if not os.path.isfile(self.mean_tiff):
            raise Exception("File Doesnot Exist: {}".format(self.mean_tiff))
        if not os.path.isfile(self.homog_tiff):
            raise Exception("File Doesnot Exist: {}".format(self.homog_tiff))
        if not os.path.isdir(self.TEMP_DIR):
            raise Exception("Directory Doesnot Exist: {}".format(self.TEMP_DIR))

    def run(self):
        window = self.window
        logging = self.logging

        t0=time.time()

        # Compute mean of each objects
        logging.info('[Zonal Statistics] - Compute mean of each predictors')
        img = gdal.Open(self.bright_tiff)
        geo=img.GetGeoTransform()
        af_transf = Affine.from_gdal(*geo)
        img = None

        logging.info('[Zonal Statistics] - Compute mean of each bright')
        bright=zonal_stats(self.segmented_shp, self.bright_tiff, stats="mean", nodata=np.nan, affine=af_transf)
        bright_list=[d["mean"] for d in bright]
        print("Finished zonal_stats-bright in",((time.time()-t0)/60), "mins")

        logging.info('[Zonal Statistics] - Compute mean of each ndvi')
        img = gdal.Open(self.ndvi_tiff)
        geo=img.GetGeoTransform()
        af_transf = Affine.from_gdal(*geo)
        img = None
        ndvi=zonal_stats(self.segmented_shp, self.ndvi_tiff, stats="mean" , nodata=np.nan, affine=af_transf)
        ndvi_list=[d["mean"] for d in ndvi]
        print("Finished zonal_stats-ndvi in",((time.time()-t0)/60), "mins")

        logging.info('[Zonal Statistics] - Compute mean of each slope')
        img = gdal.Open(self.slope_tiff)
        geo=img.GetGeoTransform()
        af_transf = Affine.from_gdal(*geo)
        img = None
        slope=zonal_stats(self.segmented_shp, self.slope_tiff, stats="mean", nodata=np.nan, affine=af_transf)
        slope_list=[d["mean"] for d in slope]
        print("Finished zonal_stats-slope in",((time.time()-t0)/60), "mins")

        logging.info('[Zonal Statistics] - Compute mean of each GLCM_homogeneity')
        glcmhomog=zonal_stats(self.segmented_shp, self.homog_tiff,stats="mean", nodata=np.nan, affine=af_transf)
        glcmhomog_list=[d["mean"] for d in glcmhomog]
        print("Finished zonal_stats-GLCM_homogeneity in",((time.time()-t0)/60), "mins")

        logging.info('[Zonal Statistics] - Compute mean of each GLCM_mean')
        glcmmean=zonal_stats(self.segmented_shp, self.mean_tiff,stats="mean", nodata=np.nan, affine=af_transf)
        glcmmean_list=[d["mean"] for d in glcmmean]
        print("Finished zonal_stats-GLCM_mean in",((time.time()-t0)/60), "mins")

        #Open segmented shapefile and save mean
        logging.info('[Zonal Statistics] - Open segmented shapefile and save mean')
        df=gpd.read_file(self.segmented_shp)
        df["glcmmean"]=glcmmean_list
        df["Meanndvi"]=ndvi_list
        df["Meanslope"]=slope_list
        df["glcmhomog"]=glcmhomog_list
        df["Meanbright"]=bright_list
        df_final = df.replace([np.inf, -np.inf], np.nan)
        df_final=df_final.fillna(0)
        df_final.to_file(self.segmfinal_shp)
        df=None
        print("Finished Saving file in",((time.time()-t0)/60), "mins")
        # Delete original polygon created during segmentation
        # for f in glob.glob("segment*.*"):
        #     os.remove(f)

        logging.info('[Zonal Statistics] - Successfully Completed!!!')
        return (self.segmfinal_shp, )
# out_dir = 'C:/wbtraining/image/results/'

# z = ZonalStats(out_dir)    
# z.run()