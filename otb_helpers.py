import otbApplication
import os
from osgeo import gdal


def HaralickTextureExtraction_Mean(IMG_TIFF,  xoff=None, yoff=None, out=None, nbbin=32, maxvalue=None, window=None, logging=None):
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

def HaralickTextureExtraction_Homogeneity(IMG_TIFF,  xoff=None, yoff=None, maxvalue=None, out=None, nbbin=32, window=None, logging=None):
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
