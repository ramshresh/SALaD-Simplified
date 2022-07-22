import os
from tkinter.messagebox import NO
import numpy as np
from osgeo import ogr, gdal, osr
import otbApplication
import glob
from rasterstats import zonal_stats
import geopandas as gpd
import rastutils
import filehelpers
from otb_helpers import HaralickTextureExtraction_Mean, HaralickTextureExtraction_Homogeneity
import parameters

class Training:
    # names
    final_name, dem_name, ls_name, suffix_train = None, None, None, None
    # Directory and Paths
    TEMP_DIR = None
    training_image_path, training_dem_path = None, None
    training_ndvi_path, training_bright_path, training_slope_path = None, None, None
    training_mean_output_path = None
    #Set output filenames
    training_sm_path= None# os.path.join(temp_dir,"sm_"+training_file_name+".tif")  # outfile1
    training_seg_path= None# os.path.join(temp_dir,"seg_"+training_file_name+".tif")  # outfile2
    training_merg_path= None# os.path.join(temp_dir,"merg_"+training_file_name+".tif")  # outfile3
    training_segment_path= None# os.path.join(self.out_dir,"segment_"+training_file_name+".shp")  # outfile4
    training_shp_path = None # os.path.join(self.out_dir,training_file_name+".shp")  # training
    segfinal_path = None #os.path.join(self.out_dir,"segfinal_"+self.final_name+"_train.shp")

    # Training extent
    ulx, lrx, lry, uly = None, None, None, None
    # Image Details
    driverGTIFF = None
    geo, proj, rows, cols, blue, green, nir, red, maxvalue = None, None, None, None, None, None, None, None, None
    
    # For Segmentation
    hs, hr, min_size = None, None, None

    # For Training
    overlap=50
    
    def __init__(self, LS_SHP, IMG_TIFF, DEM_TIFF, OUT_DIR, window=None, logging=None,) -> None:
        self.window = window
        self.logging = logging

        self.LS_SHP = LS_SHP
        self.IMG_TIFF = IMG_TIFF
        self.DEM_TIFF = DEM_TIFF
        self.OUT_DIR = OUT_DIR
        
        self.window=window
        self.logging = logging

        self.hs=parameters.hs
        self.hr=parameters.hr
        self.min_size=parameters.min_size

        # Initialize result directory for intermediate results of training
        self.TEMP_DIR = os.path.join(self.OUT_DIR, 'training/')
        filehelpers.ensure_dir(self.TEMP_DIR)

        # Initialize name identifiers 
        self.final_name = os.path.basename(self.IMG_TIFF).split('.')[0]
        self.dem_name = os.path.basename(self.DEM_TIFF).split('.')[0]
        self.ls_name = os.path.basename(self.LS_SHP).split('.')[0]
        self.suffix_train ="_train"

        # Initialize paths
        self.training_image_path = os.path.join(self.TEMP_DIR,self.final_name+"_train"+".tif") 
        self.training_dem_path = os.path.join(self.TEMP_DIR,self.dem_name+"_train"+".tif") 
        self.training_ndvi_path = os.path.join(self.TEMP_DIR,"ndvi_"+self.final_name+"_train"+".tif")
        self.training_bright_path = os.path.join(self.TEMP_DIR,"bright_"+self.final_name+"_train"+".tif")
        self.training_slope_path = os.path.join(self.TEMP_DIR,"slope_"+self.final_name+"_train"+".tif")
        self.training_mean_output_path = os.path.join(self.TEMP_DIR,"mean_"+self.final_name+"_train"+".tif")
        self.training_homog_output_path = os.path.join(self.TEMP_DIR,"homog_"+self.final_name+"_train"+".tif")
        
        

        self.training_sm_path= os.path.join(self.TEMP_DIR,"sm_"+self.final_name+"_train"+".tif")  # outfile1
        self.training_seg_path= os.path.join(self.TEMP_DIR,"seg_"+self.final_name+"_train"+".tif")  # outfile2
        self.training_merg_path= os.path.join(self.TEMP_DIR,"merg_"+self.final_name+"_train"+".tif")  # outfile3
        self.training_segment_path= os.path.join(self.TEMP_DIR,"segment_"+self.final_name+"_train"+".shp")  # outfile4
        self.training_shp_path = os.path.join(self.TEMP_DIR, self.final_name+"_trained_segment"+".shp")  # training

        self.segfinal_path = os.path.join(self.TEMP_DIR,"segfinal_"+self.final_name+"_train.shp")

        
        # Set the extent of training dataset bsed on manual landslide shapefile
        self.extent = self.get_training_extent()

        # Read Image and then initialize values
        self.driverGTIFF=gdal.GetDriverByName("GTiff")
        
        

    def get_image_details(self):
        # Open the image file and get red band
        img = gdal.Open( self.IMG_TIFF )
        
        geo=img.GetGeoTransform()
        proj=img.GetProjection()

        rows = img.RasterYSize
        cols = img.RasterXSize

        blue_band = img.GetRasterBand(1)
        green_band = img.GetRasterBand(2)
        red_band = img.GetRasterBand(3)
        nir_band = img.GetRasterBand(4)

        blue = blue_band.ReadAsArray()
        green = green_band.ReadAsArray()
        nir = nir_band.ReadAsArray()
        red = red_band.ReadAsArray()
        maxvalue = self.red.max()
        # Close Image File
        img = None

        return (geo, proj, rows, cols, blue, green, nir, red, maxvalue, )



    def get_training_extent(self):
        drv = ogr.GetDriverByName("ESRI Shapefile")
        if not os.path.isfile(self.LS_SHP):
            raise Exception("file Not found {}".format(self.LS_SHP))
        ds_region = drv.Open(self.LS_SHP)
        lyr_region = ds_region.GetLayer()
        ulx, lrx, lry, uly  = lyr_region.GetExtent() # x_min, x_max, y_min, y_max
        assert( ulx < lrx)
        assert( uly > lry)
        lyr_region=None
        ds_region=None
        return (ulx, lrx, lry, uly)

    def HaralickTextureExtraction_Mean(self, IMG_TIFF=None,  xoff=None, yoff=None, out=None, nbbin=32, maxvalue=None, window=None, logging=None):
        IMG_TIFF = IMG_TIFF if IMG_TIFF is not None else self.IMG_TIFF


        # window = window if window is not None else self.window
        # logging = logging if logging is not None else self.logging

        if os.path.isfile(out):
            # logging.info('Reading From existing file ')
            mean = gdal.Open( out)
            mean_band=mean.GetRasterBand(1)
            mean_array=mean_band.ReadAsArray()
            mean = None
            return mean_array
        

        #GLCM Mean 
        # The following line creates an instance of the HaralickTextureExtraction application
        HaralickTextureExtraction = otbApplication.Registry.CreateApplication("HaralickTextureExtraction")

        # The following lines set all the application parameters:
        HaralickTextureExtraction.SetParameterString("in", IMG_TIFF)

        HaralickTextureExtraction.SetParameterInt("channel", 3)

        HaralickTextureExtraction.SetParameterInt("parameters.xoff", xoff)

        HaralickTextureExtraction.SetParameterInt("parameters.yoff", yoff)

        HaralickTextureExtraction.SetParameterInt("parameters.xrad", 3)

        HaralickTextureExtraction.SetParameterInt("parameters.yrad", 3)

        HaralickTextureExtraction.SetParameterInt("parameters.min", 0)

        HaralickTextureExtraction.SetParameterInt("parameters.max", int(maxvalue))

        HaralickTextureExtraction.SetParameterInt("parameters.nbbin", nbbin)    

        HaralickTextureExtraction.SetParameterString("texture","advanced")

        HaralickTextureExtraction.SetParameterString("out", out)
        
        # window['-STATUS-'].update('Writing HaralickTextureExtraction - mean to {}'.format(out))
        # logging.info('Writing HaralickTextureExtraction - mean to {}'.format(out))
        # # The following line execute the application
        HaralickTextureExtraction.ExecuteAndWriteOutput()

        mean = gdal.Open( out)
        mean_band=mean.GetRasterBand(1)
        mean_array=mean_band.ReadAsArray()
        mean = None
        return mean_array

    def HaralickTextureExtraction_Homogeneity(self, IMG_TIFF=None,  xoff=None, yoff=None, maxvalue=None, out=None, nbbin=32, window=None, logging=None):
        IMG_TIFF = IMG_TIFF if IMG_TIFF is not None else self.IMG_TIFF
        # window = window if window is not None else self.window
        # logging = logging if logging is not None else self.logging
        if os.path.isfile(out):
            # logging.info('Reading From existing file ')
            homog = gdal.Open( out)

            homog_band=homog.GetRasterBand(4)

            homog_array=homog_band.ReadAsArray()
            homog = None
            return homog_array
            
        #GLCM Homogeneity 
        # The following line creates an instance of the HaralickTextureExtraction application
        HaralickTextureExtraction = otbApplication.Registry.CreateApplication("HaralickTextureExtraction")

        # The following lines set all the application parameters:
        HaralickTextureExtraction.SetParameterString("in", IMG_TIFF)

        HaralickTextureExtraction.SetParameterInt("channel", 3)

        HaralickTextureExtraction.SetParameterInt("parameters.xoff", xoff)

        HaralickTextureExtraction.SetParameterInt("parameters.yoff", yoff)

        HaralickTextureExtraction.SetParameterInt("parameters.xrad", 3)

        HaralickTextureExtraction.SetParameterInt("parameters.yrad", 3)

        HaralickTextureExtraction.SetParameterInt("parameters.min", 0)

        HaralickTextureExtraction.SetParameterInt("parameters.max", int(maxvalue))

        HaralickTextureExtraction.SetParameterInt("parameters.nbbin", 32)

        HaralickTextureExtraction.SetParameterString("texture","simple")

        HaralickTextureExtraction.SetParameterString("out", out)
        
        # window['-STATUS-'].update('Writing HaralickTextureExtraction - homogeneity to {}'.format(out))
        # logging.info('Writing HaralickTextureExtraction - homogeneity to {}'.format(out))
        
        # The following line execute the application
        HaralickTextureExtraction.ExecuteAndWriteOutput()

        homog = gdal.Open( out)

        homog_band=homog.GetRasterBand(4)

        homog_array=homog_band.ReadAsArray()
        homog = None
        return homog_array

    
    def run(self):
        window = self.window
        logging = self.logging

        logging.info('[Training] - checking if extents of shapefile, image file and dem file match')
        training_extent = [self.ulx, self.uly, self.lrx, self.lry]
        overlap=50
        # Provide segmentation parameter (same as parameters for the whole image)
        hs=self.hs
        hr=self.hr
        min_size=self.min_size


        extent_dem = rastutils.get_raster_extent(self.DEM_TIFF)
        extent_img = rastutils.get_raster_extent(self.IMG_TIFF)
        extent_shp = rastutils.get_shp_extent(self.LS_SHP)
        
        rastutils.extent_to_poly(extent_shp, os.path.join(self.TEMP_DIR, self.ls_name+'_extent.shp'))
        rastutils.extent_to_poly(extent_dem, os.path.join(self.TEMP_DIR, self.dem_name+'_extent.shp'))
        rastutils.extent_to_poly(extent_img, os.path.join(self.TEMP_DIR, self.final_name+'_extent.shp'))

        rastutils.check_extent(extent_shp, extent_dem)
        rastutils.check_extent(extent_shp, extent_img)

        # Cut whole image to extent of training area 
        logging.info('[Training] - Cut whole image to extent of training area ')
        if not os.path.isdir(os.path.dirname(self.training_image_path)):
            raise Exception("{} doesnot exist to write".format(os.path.dirname(self.training_image_path)))
        
        # I have GeoTiff files (*.tif) with WGS84 coordinate system, (for example 35 N 532402 4892945) and need to cut this image (tif file) by specified coordinates (minX=27.37 maxX=27.42 minY=44.15 maxY=44.20) without gdalwarp utility
        # https://gis.stackexchange.com/questions/199477/gdal-python-cut-geotiff-image
        if not os.path.isfile(self.training_image_path):
            rastutils.clip(self.IMG_TIFF, self.training_image_path, extent_shp)
        else:
            # logging.info('Skipping: File already exist')
            pass
        
        
        logging.info('[Training] - Cut whole DEM to extent of training area ')
        if not os.path.isfile(self.training_dem_path):
            rastutils.clip(self.DEM_TIFF, self.training_dem_path, extent_shp)
        else:
            # logging.info('Skipping: File already exist')
            pass
        
        
        # Slope raster of clipped DEM
        
        logging.info('[Training] - Slope raster of clipped DEM')
        if not os.path.isfile(self.training_slope_path):
            # Compute and save Slope from DEM clipped to training extent
            tempDS = gdal.DEMProcessing(self.training_slope_path, self.training_dem_path, "slope")
            tempDS = None
            if not os.path.isfile(self.training_slope_path):
                raise Exception("{} : Failed to create slope from \n{}".format(self.training_slope_path, self.training_dem_path))
        else:
            # logging.info('Skipping: File already exist')
            pass

        
        # 1. Explanator --> Image 
        logging.info('[Training] - Explanator objects')
        # Open the image file and get red band
        if not os.path.isfile(self.training_image_path):
            raise Exception('{} file not found'.format(self.training_image_path))
        training_img = gdal.Open( self.training_image_path )
        # get extent and projections
        training_geo=training_img.GetGeoTransform()
        training_proj=training_img.GetProjection()
        # get image size
        training_rows = training_img.RasterYSize
        training_cols = training_img.RasterXSize
        # get bands
        training_blue_band = training_img.GetRasterBand(1)
        training_green_band = training_img.GetRasterBand(2)
        training_red_band = training_img.GetRasterBand(3)
        training_nir_band = training_img.GetRasterBand(4)

        training_blue = training_blue_band.ReadAsArray()
        training_green = training_green_band.ReadAsArray()
        training_nir = training_nir_band.ReadAsArray()
        training_red = training_red_band.ReadAsArray()
        training_maxvalue = training_red.max()
        # Close Image File
        training_img = None

        # Compute NDVI 
        logging.info('[Training] - Compute NDVI ')
        training_red_flt = training_red.astype(np.float32)
        training_nir_flt = training_nir.astype(np.float32)
        np.seterr(divide="ignore", invalid="ignore")
        training_ndvi = (training_nir_flt - training_red_flt)/(training_nir_flt + training_red_flt)
        # Compute Brightness
        logging.info('[Training] - Compute Brightness ')
        training_bright=(training_blue+training_green+training_red+training_nir)/4
        

        logging.info('[Training] - SAVING NDVI ')
        if not os.path.isfile(self.training_ndvi_path):
            output=self.driverGTIFF.Create(self.training_ndvi_path,training_cols,training_rows,1,gdal.GDT_Float32)
            output.SetGeoTransform(training_geo)
            output.SetProjection(training_proj)
            output.GetRasterBand(1).WriteArray(training_ndvi)
            output = None
            if not os.path.isfile(self.training_ndvi_path):
                raise Exception("{} : Failed to create ndvi from \n{}".format(self.training_ndvi_path, training_ndvi))
        else:
            logging.info('[Training] - Skipping: File already exist')
            
        logging.info('[Training] - SAVING BRIGHT ')
        if not os.path.isfile(self.training_bright_path):
            output1=self.driverGTIFF.Create(self.training_bright_path,training_cols,training_rows,1,gdal.GDT_Float32)
            output1.SetGeoTransform(training_geo)
            output1.SetProjection(training_proj)
            output1.GetRasterBand(1).WriteArray(training_bright)
            output1 = None
        else:
            logging.info('[Training] - Skipping: File already exist')
            

        # GLCM (Grey Level Co-Occurance Matrix for Mean and Homogeneity)
        logging.info('[Training] - GLCM (Grey Level Co-Occurance Matrix for Mean and Homogeneity)')
            
        training_mean_stack=np.zeros((4, training_rows, training_cols))
        training_homog_stack=np.zeros((4,training_rows, training_cols))

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
        
            logging.info('[Training] - HaralickTextures_mean and  HaralickTextures_homog for angle {}'.format(ang))
            training_mean_out=os.path.join(self.TEMP_DIR , "HaralickTextures_mean_"+str(ang)+".tif")
            training_homog_out=os.path.join(self.TEMP_DIR ,"HaralickTextures_homog_"+str(ang)+".tif")
            
            training_mean_array = HaralickTextureExtraction_Mean(self.training_image_path, xoff=cx, yoff=cy, out=training_mean_out, maxvalue=training_maxvalue)
            training_homog_array = HaralickTextureExtraction_Homogeneity(self.training_image_path, xoff=cx, yoff=cy, out=training_homog_out, maxvalue=training_maxvalue)
            
            training_mean_stack[x,:]=training_mean_array
            training_homog_stack[x,:]=training_homog_array
        
        training_glcm_mean = np.mean(training_mean_stack, axis=0)
        training_glcm_homog = np.mean(training_homog_stack, axis=0)
        
        logging.info('[Training] - Saving GLCM Mean to file')
            
        training_mean_output=self.driverGTIFF.Create(self.training_mean_output_path, training_cols, training_rows, 1, gdal.GDT_Float32)
        training_mean_output.SetGeoTransform(training_geo)
        training_mean_output.SetProjection(training_proj)
        training_mean_output.GetRasterBand(1).WriteArray(training_glcm_mean) 
        training_mean_output = None
        
        logging.info('[Training] - Saving GLCM Homogeneity to file')
        training_homog_output=self.driverGTIFF.Create(self.training_homog_output_path, training_cols,training_rows,1,gdal.GDT_Float32)
        training_homog_output.SetGeoTransform(training_geo)
        training_homog_output.SetProjection(training_proj)
        training_homog_output.GetRasterBand(1).WriteArray(training_glcm_homog) 
        training_homog_output = None                
        

        # training_sm_path= 1
        # training_seg_path= 2
        # training_merg_path= 3
        # training_segment_path 4
        # training_shp_path = N

        logging.info('[Training] - MeanShiftSmoothing to file')
        # The following line creates an instance of the MeanShiftSmoothing application
        MeanShiftSmoothing = otbApplication.Registry.CreateApplication("MeanShiftSmoothing")
        # The following lines set all the application parameters:
        MeanShiftSmoothing.SetParameterString("in", self.training_image_path)
        MeanShiftSmoothing.SetParameterString("fout", self.training_sm_path)
        MeanShiftSmoothing.SetParameterInt("spatialr", hs)
        MeanShiftSmoothing.SetParameterFloat("ranger", hr)
        MeanShiftSmoothing.SetParameterFloat("thres", 0.1)
        MeanShiftSmoothing.SetParameterInt("maxiter", 100)
        # The following line execute the application
        MeanShiftSmoothing.ExecuteAndWriteOutput()

        logging.info('[Training] - LSMSSegmentation to file')
        # The following line creates an instance of the LSMSSegmentation application
        LSMSSegmentation = otbApplication.Registry.CreateApplication("LSMSSegmentation")
        # The following lines set all the application parameters:
        LSMSSegmentation.SetParameterString("in", self.training_sm_path)
        LSMSSegmentation.SetParameterString("out", self.training_seg_path)
        LSMSSegmentation.SetParameterFloat("spatialr", hs)
        LSMSSegmentation.SetParameterFloat("ranger", hr)
        LSMSSegmentation.SetParameterInt("minsize", 0)
        LSMSSegmentation.SetParameterInt("tilesizex", 500)
        LSMSSegmentation.SetParameterInt("tilesizey", 500)
        # The following line execute the application
        LSMSSegmentation.ExecuteAndWriteOutput()

        logging.info('[Training] - LSMSSmallRegionsMerging to file')
        # The following line creates an instance of the LSMSSmallRegionsMerging application
        LSMSSmallRegionsMerging = otbApplication.Registry.CreateApplication("LSMSSmallRegionsMerging")
        # The following lines set all the application parameters:
        LSMSSmallRegionsMerging.SetParameterString("in", self.training_image_path)
        LSMSSmallRegionsMerging.SetParameterString("inseg", self.training_seg_path)
        LSMSSmallRegionsMerging.SetParameterString("out", self.training_merg_path)
        LSMSSmallRegionsMerging.SetParameterInt("minsize", min_size)
        LSMSSmallRegionsMerging.SetParameterInt("tilesizex", 500)
        LSMSSmallRegionsMerging.SetParameterInt("tilesizey", 500)
        # The following line execute the application
        LSMSSmallRegionsMerging.ExecuteAndWriteOutput()

        logging.info('[Training] - Convert LSMSSmallRegionsMerging output raster to polygo')
        #Convert LSMSSmallRegionsMerging output raster to polygon
        #Get spatial reference from LSMSSmallRegionsMerging raster 
        src_ds = gdal.Open( self.training_merg_path )
        srcband = src_ds.GetRasterBand(1)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(src_ds.GetProjection())

        #Create output shapefile
        drv = ogr.GetDriverByName("ESRI Shapefile")

        if(os.path.isfile(self.training_segment_path)):
            os.remove(self.training_segment_path)

        dst_ds = drv.CreateDataSource(self.training_segment_path)

        logging.info('[Training] - Saving shapefile of LSMSSmallRegionsMerging output raster to polygo')  
        dst_layer = dst_ds.CreateLayer(self.training_segment_path, srs = srs )
        gdal.Polygonize( srcband, None, dst_layer, -1, [], callback=None )
        dst_layer = None
        dst_ds.Destroy()

        # Delete temporary files created during segmentation
        logging.info('[Training] - Delete temporary files < seg_*_FINAL.tif >created during segmentation')  
        
        for f in glob.glob("seg_*_FINAL.tif"):
            os.remove(f)


        # Compute zonal stats
        logging.info('[Training] - Compute zonal stats')  
        
        logging.info('[Training] - Compute zonal stats - bright')  
        
        bright=zonal_stats(self.training_segment_path, self.training_bright_path, nodata=np.nan)
        bright_list=[d["mean"] for d in bright]
        
        logging.info('[Training] - Compute zonal stats - ndvi')
        ndvi=zonal_stats(self.training_segment_path, self.training_ndvi_path, stats="mean", nodata=np.nan)
        ndvi_list=[d["mean"] for d in ndvi]
        
        logging.info('[Training] - Compute zonal stats - slope')
        slope=zonal_stats(self.training_segment_path, self.training_slope_path, stats="mean", nodata=np.nan)
        slope_list=[d["mean"] for d in slope]
        
        logging.info('[Training] - Compute zonal stats - glcmhomog')
        glcmhomog=zonal_stats(self.training_segment_path, self.training_homog_output_path, stats="mean", nodata=np.nan)
        glcmhomog_list=[d["mean"] for d in glcmhomog]
        
        logging.info('[Training] - Compute zonal stats - glcmmean')
        glcmmean=zonal_stats(self.training_segment_path, self.training_mean_output_path, stats="mean", nodata=np.nan)
        glcmmean_list=[d["mean"] for d in glcmmean]

        logging.info('[Training] - Reading Training Segmentation')
        df=gpd.read_file(self.training_segment_path)
        logging.info('[Training] - Append glcmmean, Meanndvi, Meanslope, glcmhomog,Meanbright ')
        df["glcmmean"]=glcmmean_list
        df["Meanndvi"]=ndvi_list
        df["Meanslope"]=slope_list
        df["glcmhomog"]=glcmhomog_list
        df["Meanbright"]=bright_list

        logging.info('[Training] - Saving')
        df_final = df.replace([np.inf, -np.inf], np.nan)
        df_final=df_final.fillna(0)
        df_final.to_file(self.segfinal_path)
        df=None

        # Delete original polygon created during segmentation
        # for f in glob.glob(os.path.join(self.out_dir,"segment*.*")):
        #     os.remove(f)


        # Create training shapefile
        logging.info('[Training] -  Create training shapefile')
        #Select intersecting polygons
        select_feature = gpd.read_file(self.LS_SHP) 
        input_feature = gpd.read_file(self.segfinal_path, encoding="utf-8")
        selection= gpd.sjoin(input_feature, select_feature, how="inner", op="intersects")
        selection["segment_ar"] = selection["geometry"].area
        # final_select=selection[selection["index_right"]>0]
        final_select =selection


        # Calculate overlap 
        logging.info('[Training] -  Calculate overlap ')
        intersections=gpd.overlay(select_feature, final_select, how="intersection")
        intersections["overlap_ar"] = intersections["geometry"].area
        intersections["percentage"] = intersections["overlap_ar"]/intersections["segment_ar"]*100
        intersections = intersections.loc[:, ["geometry","percentage"]]
        final_intersect=intersections[intersections["percentage"]>=overlap]

        # Combine landslide and non-landslide objects
        logging.info('[Training] -   Combine landslide and non-landslide objects')
        landslide= gpd.sjoin(input_feature, final_intersect, how="inner", op="contains")
        landslide["landslide"]=1
        landslide.drop(["percentage"], axis=1, inplace=True)
        landslide.drop(["index_right"], axis=1, inplace=True)
        non_landslide = input_feature.drop(landslide["FID"], errors="ignore")
        non_landslide["landslide"]=0

        # Join and save the training data
        logging.info('[Training] -   Join and save the training data')
        training = landslide.append(non_landslide)
        training.sort_values(by=["FID"])
        training.to_file(self.training_shp_path)

        # Delete final segmentation shapefile
        # for f in glob.glob(os.path.join(out_dir,"segfinal_*_train.*")):
        #     os.remove(f)
        
        logging.info('[Training] -   Successfully Completed!!!')
        return (self.training_shp_path, )

# ls_shp = 'C:/wbtraining/image/manual_landslide.shp'
# img_tiff = 'C:/wbtraining/image/pasang_2015_12_28_clip.tif'
# dem_tiff ='C:/wbtraining/image/srtm_pasang.tif' 
# out_dir = 'C:/wbtraining/image/results/'
# training = Training(ls_shp,img_tiff, dem_tiff, out_dir)

# training.run()