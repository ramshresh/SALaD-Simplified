import os
import logging
import PySimpleGUI as sg   
from explanator import Explanator
from segmentation import Segmentation
from zonalstats import ZonalStats
from train import Training
from randomforest import RandomForest
import filehelpers

log_file = 'run_log.txt'
buffer = ''

class Handler(logging.StreamHandler):
    def __init__(self, window=None):
        logging.StreamHandler.__init__(self)
        self.window = window
    def emit(self, record):
        global buffer
        record = '{}, [{}], {}'.format(record.name,[record.levelname], record.message)
        buffer = '{}\n{}'.format(buffer, record ).strip()
        if self.window is not None:
            self.window['log'].update(value=buffer)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s, %(asctime)s, [%(levelname)s], %(message)s',
    filename=log_file,
    filemode='w')


def validate(LS_SHP=None, IMG_TIFF=None, DEM_TIFF=None, OUTPUT_FOLDER=None):
    errors = {}
    if  not os.path.isfile(LS_SHP):
        
        errors['-LS_SHP-'] = {
            "valueof": "-LS_SHP-",
            "type": "fileNotFound",
            "description": "Landslide shapefile is required"
        } 
    if  not os.path.isfile(DEM_TIFF):
        errors['-DEM_TIFF-'] = {
            "valueof": "-DEM_TIFF-",
            "type": "fileNotFound",
            "description": "DEM file is required"
        } 

    if  not os.path.isdir(OUTPUT_FOLDER):
        try:
            filehelpers.ensure_dir(OUTPUT_FOLDER)
        except:
            pass
    
    if  not os.path.isdir(OUTPUT_FOLDER):
        errors['-OUTPUT_FOLDER-'] = {
            "valueof": "-OUTPUT_FOLDER-",
            "type": "fileNotFound",
            "description": "Please set a valid output folder"
        } 
        
    if  not os.path.isfile(IMG_TIFF):
        errors['-IMG_TIFF-'] = {
            "valueof": "-IMG_TIFF-",
            "type": "fileNotFound",
            "description": "Image file is required"
        }
    return errors

    
        
# 1- the layout
layout = [
    # [sg.Text('Your typed chars appear here:'), sg.Text(size=(15,1), key='-OUTPUT-')],
    # [sg.Input(key='-IN-')],
    # [sg.Button('Show')],
    
    [sg.Text('Landslide Polygons(Manually Mapped) - Training Dataset  (Vector, .shp): ')], [sg.Input(key='-LS_SHP-', size=(70, 20), enable_events=True), sg.FileBrowse(target='-LS_SHP-')],
    [sg.Text('Image File (Raster, .tiff): ')], [sg.Input(key='-IMG_TIFF-', size=(70, 20), enable_events=True), sg.FileBrowse(target='-IMG_TIFF-')],
    [sg.Text('DEM File (Raster, .tiff): ')], [sg.Input(key='-DEM_TIFF-', size=(70, 20), enable_events=True), sg.FileBrowse(target='-DEM_TIFF-')],
    [sg.Text('Output Folder: ')], [sg.Input(key='-OUTPUT_FOLDER-', size=(70, 20), enable_events=True), sg.FolderBrowse(target='-OUTPUT_FOLDER-', key='-BROWSE_OUTPUT_FOLDER-')],
    [sg.Button('1_Explanator')],
    [sg.Button('2_Segmentation')],
    [sg.Button('3_Zonal_Statistics')],
    [sg.Button('4_Train')],
    [sg.Button('5_Random_Forest')],
    [sg.Button('CancelProcess')],
    [sg.Text(s=(100,2), k='-STATUS-')],
    [sg.Output(size=(100,10), key='log')],
    
    [sg.Button('Exit')]
    ]
# 2 - the window
window = sg.Window('SALaD - Training Model', layout)


ch = Handler(window=window)
ch.setLevel(logging.INFO)
logging.getLogger('').addHandler(ch)

timer_running, counter = False, 0

def init_test_data(window):
    # window['-LS_SHP-'].update('C:/wbtraining/image/manual_landslide.shp')
    # window['-IMG_TIFF-'].update('C:/wbtraining/image/pasang_2015_12_28_clip.tif')
    # window['-DEM_TIFF-'].update('C:/wbtraining/image/srtm_pasang.tif')
    # window['-OUTPUT_FOLDER-'].update('C:/wbtraining/image/out/')

    window['-LS_SHP-'].update('C:/SajadDemo/SALaD/indata/test/bk5_epoch_14_manual_trainingdataset.shp')
    window['-IMG_TIFF-'].update('C:/SajadDemo/SALaD/indata/test/bk5_epoch_14_sentinel_B_G_R_NIR_T45RUL_20201113T045039_comp.tif')
    window['-DEM_TIFF-'].update('C:/SajadDemo/SALaD/indata/test/bk5_DEM_10m.tif')
    window['-OUTPUT_FOLDER-'].update('C:/SajadDemo/SALaD/indata/test/out/')

# 3 - the event loop
while True:
    event, values = window.read(timeout=10)
    # init_test_data(window)
    # print(event, values)
    if event in( sg.WIN_CLOSED,'Exit'):
        logging.info('Closed')
        break
    if event == 'CancelProcess':
        timer_running = False
        logging.info('CancelProcess')

    
    if event == '1_Explanator':
        timer_running, counter = False, 0
        logging.info('StartProcess')
        
        LS_SHP = values['-LS_SHP-'] if '-LS_SHP-' in values else None
        IMG_TIFF = values['-IMG_TIFF-']  if '-IMG_TIFF-' in values else None
        DEM_TIFF = values['-DEM_TIFF-']  if '-DEM_TIFF-' in values else None
        OUTPUT_FOLDER = values['-OUTPUT_FOLDER-']  if '-OUTPUT_FOLDER-' in values else None

        errors = validate(
            LS_SHP=LS_SHP,
            IMG_TIFF=IMG_TIFF,
            DEM_TIFF=DEM_TIFF,
            OUTPUT_FOLDER=OUTPUT_FOLDER
            )
        if(bool(errors)):
            err_list = [errors[k]['description'] for k  in errors]
            err_txt = '\n'.join(err_list)
            sg.Popup('Error: {}'.format(err_txt)) 
            logging.info('Error in Input:\n{}'.format(err_txt))
                
        else:
            window['-STATUS-'].update('Input looks good...')
            logging.info('Input looks good...')
            try:
                # window['-STATUS-'].update('Computating GLCM of Mean and Homogeneity in 4 directions...')
                # logging.info('Computating GLCM of Mean and Homogeneity in 4 directions...')
                explanator = Explanator(IMG_TIFF, DEM_TIFF, OUTPUT_FOLDER, window=window, logging=logging)

                window['-STATUS-'].update('PROCESSING - GLCM_Mean_Homogeneity')
                window.perform_long_operation(
                    lambda :explanator.run(), 
                    '-END_1_Explanator-'
                )

            except Exception as e:
                logging.info('An error happened.  Here is the info: \n*****EXCEPTION*****{}\n**********************'.format(e))
                sg.popup_error_with_traceback('An error happened.  Here is the info: {}'.format(e))
    if event == '2_Segmentation':
        timer_running, counter = False, 0
        logging.info('StartProcess')
        
        

        LS_SHP = values['-LS_SHP-'] if '-LS_SHP-' in values else None
        IMG_TIFF = values['-IMG_TIFF-']  if '-IMG_TIFF-' in values else None
        DEM_TIFF = values['-DEM_TIFF-']  if '-DEM_TIFF-' in values else None
        OUTPUT_FOLDER = values['-OUTPUT_FOLDER-']  if '-OUTPUT_FOLDER-' in values else None

        errors = validate(
            LS_SHP=LS_SHP,
            IMG_TIFF=IMG_TIFF,
            DEM_TIFF=DEM_TIFF,
            OUTPUT_FOLDER=OUTPUT_FOLDER
            )
        if(bool(errors)):
            err_list = [errors[k]['description'] for k  in errors]
            err_txt = '\n'.join(err_list)
            sg.Popup('Error: {}'.format(err_txt)) 
            logging.info('Error in Input:\n{}'.format(err_txt))
                
        else:
            window['-STATUS-'].update('Input looks good...')
            logging.info('Input looks good...')
            try:
                window['-STATUS-'].update('PROCESSING - SEGMENTATION')
                segmentation = Segmentation(IMG_TIFF, OUTPUT_FOLDER, window=window, logging=logging)
                window.perform_long_operation(
                    lambda : segmentation.run(), 
                    '-END_2_Segmentation-'
                )
            except Exception as e:
                logging.info('An error happened.  Here is the info: \n*****EXCEPTION*****{}\n**********************'.format(e))
                sg.popup_error_with_traceback('An error happened.  Here is the info: {}'.format(e))
    if event == '3_Zonal_Statistics':
        timer_running, counter = False, 0
        logging.info('StartProcess: Zonal Statistics')
        
        

        LS_SHP = values['-LS_SHP-'] if '-LS_SHP-' in values else None
        IMG_TIFF = values['-IMG_TIFF-']  if '-IMG_TIFF-' in values else None
        DEM_TIFF = values['-DEM_TIFF-']  if '-DEM_TIFF-' in values else None
        OUTPUT_FOLDER = values['-OUTPUT_FOLDER-']  if '-OUTPUT_FOLDER-' in values else None

        errors = validate(
            LS_SHP=LS_SHP,
            IMG_TIFF=IMG_TIFF,
            DEM_TIFF=DEM_TIFF,
            OUTPUT_FOLDER=OUTPUT_FOLDER
            )
        if(bool(errors)):
            err_list = [errors[k]['description'] for k  in errors]
            err_txt = '\n'.join(err_list)
            sg.Popup('Error: {}'.format(err_txt)) 
            logging.info('Error in Input:\n{}'.format(err_txt))
                
        else:
            window['-STATUS-'].update('Input looks good...')
            logging.info('Input looks good...')
            try:
                window['-STATUS-'].update('PROCESSING - Zonal Statistics')
                z = ZonalStats(OUTPUT_FOLDER, window=window, logging=logging)  
                window.perform_long_operation(
                    lambda :z.run(), 
                    '-END_3_Zonal_Statistics-'
                )
            except Exception as e:
                logging.info('An error happened.  Here is the info: \n*****EXCEPTION*****{}\n**********************'.format(e))
                sg.popup_error_with_traceback('An error happened.  Here is the info: {}'.format(e))
    if event == '4_Train':
        timer_running, counter = False, 0
        logging.info('StartProcess: Train')
        
        LS_SHP = values['-LS_SHP-'] if '-LS_SHP-' in values else None
        IMG_TIFF = values['-IMG_TIFF-']  if '-IMG_TIFF-' in values else None
        DEM_TIFF = values['-DEM_TIFF-']  if '-DEM_TIFF-' in values else None
        OUTPUT_FOLDER = values['-OUTPUT_FOLDER-']  if '-OUTPUT_FOLDER-' in values else None

        errors = validate(
            LS_SHP=LS_SHP,
            IMG_TIFF=IMG_TIFF,
            DEM_TIFF=DEM_TIFF,
            OUTPUT_FOLDER=OUTPUT_FOLDER
            )
        if(bool(errors)):
            err_list = [errors[k]['description'] for k  in errors]
            err_txt = '\n'.join(err_list)
            sg.Popup('Error: {}'.format(err_txt)) 
            logging.info('Error in Input:\n{}'.format(err_txt))
                
        else:
            window['-STATUS-'].update('Input looks good...')
            logging.info('Input looks good...')
            try:
                window['-STATUS-'].update('PROCESSING - Training Model')
                training = Training(LS_SHP, IMG_TIFF, DEM_TIFF, OUTPUT_FOLDER, window=window, logging=logging)
                window.perform_long_operation(
                    lambda :training.run(),
                    '-END_4_Train-'
                )
            except Exception as e:
                logging.info('An error happened.  Here is the info: \n*****EXCEPTION*****{}\n**********************'.format(e))
                sg.popup_error_with_traceback('An error happened.  Here is the info: {}'.format(e))
    
    if event == '5_Random_Forest':
        timer_running, counter = False, 0
        logging.info('StartProcess: Random Forest')
        
        LS_SHP = values['-LS_SHP-'] if '-LS_SHP-' in values else None
        IMG_TIFF = values['-IMG_TIFF-']  if '-IMG_TIFF-' in values else None
        DEM_TIFF = values['-DEM_TIFF-']  if '-DEM_TIFF-' in values else None
        OUTPUT_FOLDER = values['-OUTPUT_FOLDER-']  if '-OUTPUT_FOLDER-' in values else None

        errors = validate(
            LS_SHP=LS_SHP,
            IMG_TIFF=IMG_TIFF,
            DEM_TIFF=DEM_TIFF,
            OUTPUT_FOLDER=OUTPUT_FOLDER
            )
        if(bool(errors)):
            err_list = [errors[k]['description'] for k  in errors]
            err_txt = '\n'.join(err_list)
            sg.Popup('Error: {}'.format(err_txt)) 
            logging.info('Error in Input:\n{}'.format(err_txt))
                
        else:
            window['-STATUS-'].update('Input looks good...')
            logging.info('Input looks good...')
            try:
                window['-STATUS-'].update('PROCESSING - Random Forest')
                rf = RandomForest(OUTPUT_FOLDER, window=window, logging=logging)
                
                window.perform_long_operation(
                    lambda :rf.run(),
                    '-END_5_Random_Forest-'
                )
            except Exception as e:
                logging.info('An error happened.  Here is the info: \n*****EXCEPTION*****{}\n**********************'.format(e))
                sg.popup_error_with_traceback('An error happened.  Here is the info: {}'.format(e))
    
    # Events for ending long duration functions
    if event == '-END_1_Explanator-':
        results = values[event]
        logging.info('Completed. Returned: {}'.format(list(results)))
        window['-STATUS-'].update('COMPLETED - 1_Explanator')
    if event == '-END_2_Segmentation-':
        results = values[event]
        logging.info('Completed. Returned: {}'.format(list(results)))
        window['-STATUS-'].update('COMPLETED - 2_Segmentation')
    if event == '-END_3_Zonal_Statistics-':
        results = values[event]
        logging.info('Completed. Returned: {}'.format(list(results)))
        window['-STATUS-'].update('COMPLETED - 3_Zonal_Statistics')
    if event == '-END_4_Train-':
        results = values[event]
        logging.info('Completed. Returned: {}'.format(list(results)))
        window['-STATUS-'].update('COMPLETED - 4_Train')
    if event == '-END_5_Random_Forest-':
        results = values[event]
        logging.info('Completed. Returned: {}'.format(list(results)))
        window['-STATUS-'].update('COMPLETED - 5_Random_Forest')


# 4 - the close
window.close()