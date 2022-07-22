import os
import numpy as np
from osgeo import gdal
import filehelpers
from otb_helpers import HaralickTextureExtraction_Mean, HaralickTextureExtraction_Homogeneity

class Explanator:
    # names
    final_name, dem_name = None, None

    # Directory and Paths
    TEMP_DIR = None
    
    #Set output filenames
    ndvi_path, bright_path = None, None

    driverGTIFF=gdal.GetDriverByName("GTiff")

    def __init__(self, IMG_TIFF, DEM_TIFF, OUT_DIR, window=None, logging=None,) -> None:
        self.IMG_TIFF = IMG_TIFF
        self.DEM_TIFF = DEM_TIFF
        self.OUT_DIR = OUT_DIR

        self.window=window
        self.logging = logging
        
        # Initialize result directory for intermediate results of training
        if not os.path.isdir(self.OUT_DIR):
            filehelpers.ensure_dir(self.OUT_DIR)

        self.TEMP_DIR = os.path.join(self.OUT_DIR, 'explanator/')
        if not os.path.isdir(self.TEMP_DIR):
            filehelpers.ensure_dir(self.TEMP_DIR)

        # Initialize name identifiers 
        self.final_name = os.path.basename(self.IMG_TIFF).split('.')[0]
        self.dem_name = os.path.basename(self.DEM_TIFF).split('.')[0]


        # Initialize paths
        self.image_path = os.path.join(self.TEMP_DIR,self.final_name+"_expln"+".tif") 
        self.dem_path = os.path.join(self.TEMP_DIR,self.dem_name+"_expln"+".tif") 
        self.ndvi_path = os.path.join(self.TEMP_DIR,"ndvi_"+self.final_name+"_expln"+".tif")
        self.bright_path = os.path.join(self.TEMP_DIR,"bright_"+self.final_name+"_expln"+".tif")
        self.slope_path = os.path.join(self.TEMP_DIR,"slope_"+self.final_name+"_expln"+".tif")
        self.clipped_slope_path = os.path.join(self.TEMP_DIR, "slope_clipped_"+self.final_name+"_expln"+".tif")

        self.mean_output_path = os.path.join(self.TEMP_DIR,"mean_"+self.final_name+"_expln"+".tif")
        self.homog_output_path = os.path.join(self.TEMP_DIR,"homog_"+self.final_name+"_expln"+".tif")
        


    def run(self):
        logging = self.logging
        
        logging.info('[Explanator] - Start Process')
        if not os.path.isfile(self.IMG_TIFF):
            raise Exception("File: {} not found".format(self.IMG_TIFF))

        if not os.path.isfile(self.DEM_TIFF):
            raise Exception("File: {} not found".format(self.IMG_TIFF))

        # Open the image file and get red band
        
        logging.info("[Explanator] - Open the image file and get red band:  {}".format(self.IMG_TIFF))
        img = gdal.Open( self.IMG_TIFF )

        # get extent and projections
        geo=img.GetGeoTransform()
        proj=img.GetProjection()
        # get image size
        rows = img.RasterYSize
        cols = img.RasterXSize
        # get bands
        blue_band = img.GetRasterBand(1)
        green_band = img.GetRasterBand(2)
        red_band = img.GetRasterBand(3)
        nir_band = img.GetRasterBand(4)

        blue = blue_band.ReadAsArray()
        green = green_band.ReadAsArray()
        nir = nir_band.ReadAsArray()
        red = red_band.ReadAsArray()
        maxvalue = red.max()
        # Close Image File
        img = None

        # Compute NDVI 
        logging.info("[Explanator] - Compute NDVI")
        red_flt = red.astype(np.float32)
        nir_flt = nir.astype(np.float32)
        np.seterr(divide="ignore", invalid="ignore")
        ndvi = (nir_flt - red_flt)/(nir_flt + red_flt)

        # Compute Brightness
        logging.info("[Explanator] - Compute Brightness")
        bright=(blue+green+red+nir)/4
        
        if not os.path.isfile(self.ndvi_path):
            logging.info("[Explanator] - Saving NDVI")
            output=self.driverGTIFF.Create(self.ndvi_path,cols,rows,1,gdal.GDT_Float32)
            output.SetGeoTransform(geo)
            output.SetProjection(proj)
            output.GetRasterBand(1).WriteArray(ndvi)
            output = None
            if not os.path.isfile(self.ndvi_path):
                raise Exception("{} : Failed to create ndvi from \n{}".format(self.ndvi_path, ndvi))
        else:
            pass
            logging.info('Skipping NDVI: File already exist')

        if not os.path.isfile(self.bright_path):
            logging.info("[Explanator] - Saving Bright")
            output1=self.driverGTIFF.Create(self.bright_path,cols,rows,1,gdal.GDT_Float32)
            output1.SetGeoTransform(geo)
            output1.SetProjection(proj)
            output1.GetRasterBand(1).WriteArray(bright)
            output1 = None
        else:
            logging.info('Skipping Bright: File already exist')
            pass
        
        #Compute slope and clip it to extent and resoultion of image
        logging.info("[Explanator] - Compute slope from DEM")
        gdal.DEMProcessing(self.slope_path, self.DEM_TIFF, "slope")
        ulx = geo[0]
        uly = geo[3]
        lrx = ulx + geo[1] * cols
        lry = uly + geo[5] * rows
        logging.info("[Explanator] - Clip Slope to Image extent")
        gdal.Translate(self.clipped_slope_path, self.slope_path, width=cols,height=rows,resampleAlg=0,format='GTiff',projWin=[ulx,uly,lrx,lry])



        # Compute GLCM all-direction mean and homogeneity
        # print("Compute GLCM all-direction mean and homogeneity")
        logging.info("[Explanator] - Compute GLCM all-direction mean and homogeneity")
        mean_stack=np.zeros((4,rows,cols))
        homog_stack=np.zeros((4,rows,cols))

        for x in range(4):
            if x==0:
                cx=0
                cy=1
                ang=0
            elif x==1:
                cx=1
                cy=1
                ang=45
            elif x==2:
                cx=1
                cy=0
                ang=90
            else:
                cx=1
                cy=-1
                ang=135
            mean_out=os.path.join(self.TEMP_DIR , "HaralickTextures_mean_"+str(ang)+".tif")
            homog_out=os.path.join(self.TEMP_DIR ,"HaralickTextures_homog_"+str(ang)+".tif")
            
            logging.info("[Explanator] - HaralickTextureExtraction_Mean for angle {}".format(ang))
            mean_array = HaralickTextureExtraction_Mean(self.IMG_TIFF, xoff=cx, yoff=cy, out=mean_out, maxvalue=maxvalue)
            
            logging.info("[Explanator] - HaralickTextureExtraction_Homogeneity for angle {}".format(ang))
            homog_array = HaralickTextureExtraction_Homogeneity(self.IMG_TIFF, xoff=cx, yoff=cy, out=homog_out, maxvalue=maxvalue)
            
            mean_stack[x,:]=mean_array
            homog_stack[x,:]=homog_array
        
        logging.info("[Explanator] - Grey Level Coocurance Matrix (GLCM) for all angles ")
        glcm_mean = np.mean(mean_stack, axis=0)
        glcm_homog = np.mean(homog_stack, axis=0)

        logging.info("[Explanator] -Saving GLCMs - Mean ")
        mean_output=self.driverGTIFF.Create(self.mean_output_path, cols, rows, 1, gdal.GDT_Float32)
        mean_output.SetGeoTransform(geo)
        mean_output.SetProjection(proj)
        mean_output.GetRasterBand(1).WriteArray(glcm_mean) 
        mean_output = None

        logging.info("[Explanator] -Saving GLCMs - Homogeneity ")
        homog_output=self.driverGTIFF.Create(self.homog_output_path, cols,rows,1,gdal.GDT_Float32)
        homog_output.SetGeoTransform(geo)
        homog_output.SetProjection(proj)
        homog_output.GetRasterBand(1).WriteArray(glcm_homog) 
        homog_output = None   

        logging.info("[Explanator] - Succesfully completed!!")

        return (self.mean_output_path, self.homog_output_path, self.clipped_slope_path, self.bright_path)
# Usage:
# img_tiff = 'C:/wbtraining/image/pasang_2015_12_28_clip.tif'
# dem_tiff ='C:/wbtraining/image/srtm_pasang.tif' 
# out_dir = 'C:/wbtraining/image/results/'
# explanator = Explanator(img_tiff, dem_tiff, out_dir)

# explanator.run()
