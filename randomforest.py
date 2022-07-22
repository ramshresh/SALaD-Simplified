import os
import glob
from sklearn.ensemble import RandomForestClassifier
import geopandas as gpd
import numpy as np
import time
import filehelpers

class RandomForest:
    def __init__(self, OUT_DIR, window=None, logging=None) -> None:
        self.window = window
        self.logging = logging
        self.OUT_DIR = OUT_DIR
        self.TRAINING_DIR = os.path.join(self.OUT_DIR, 'training/')
        self.ZONALSTATS_DIR = os.path.join(self.OUT_DIR, 'zonalstats/')
        # Initialize result directory for intermediate results of training
        self.TEMP_DIR = os.path.join(self.OUT_DIR, 'randomforest/')
        filehelpers.ensure_dir(self.TEMP_DIR)

        self.PREDICTED_LS_SHP = os.path.join(self.TEMP_DIR, 'predicted_landslides.shp') 
    def run(self):
        window = self.window
        logging = self.logging
        t0=time.time()
        logging.info('[RandomForest] - Start Process')

        logging.info('[RandomForest] - Search trained dataset ')
        os.chdir(self.TRAINING_DIR)
        trained_poly=glob.glob("*_trained_segment.shp")
        trained_poly_filename_with_ext=trained_poly[0]
        trained_shp = os.path.join(self.TRAINING_DIR, trained_poly_filename_with_ext)
        print("Finished found *_trained_segment.shp as trained dataset in",((time.time()-t0)/60), "mins")

        
        logging.info('[RandomForest] - Search segmented dataset ')
        os.chdir(self.ZONALSTATS_DIR)
        segmented_poly=glob.glob("segfinal_*.shp")
        segmented_poly_filename_with_ext=segmented_poly[0]
        segmented_shp = os.path.join(self.ZONALSTATS_DIR, segmented_poly_filename_with_ext)
        print("Finished found segfinal_*.shp as test dataset in",((time.time()-t0)/60), "mins")


        # Train the RF model
        logging.info('[RandomForest] - Train the RF model')
        print("Train Random Forest Model-- elapsed:{elapsed}".format(elapsed=((time.time()-t0)/60)))
        print("[READING] File {}-- elapsed:{elapsed}".format(trained_shp, elapsed=((time.time()-t0)/60)))
        df_train = gpd.read_file(trained_shp , encoding="utf-8")
        print("[OK]: File {}-- elapsed:{elapsed}".format(trained_shp, elapsed=((time.time()-t0)/60)))
        predictor_vars = ["Meanbright","Meanndvi","Meanslope","glcmhomog","glcmmean"]
        print("[TRAINING] data {}-- elapsed:{elapsed}".format(df_train.head(), elapsed=((time.time()-t0)/60)))
        
        logging.info('[RandomForest] - predictor using vars: {}'.format(predictor_vars))
        x,y = df_train[predictor_vars],df_train.landslide
        modelRandom = RandomForestClassifier(n_estimators=5000)
        modelRandom.fit(x,y)
        print("[OK]:  {}-- elapsed:{elapsed}".format(modelRandom, elapsed=((time.time()-t0)/60)))


        # Predict using the train model
        logging.info('[RandomForest] - Predict using the train model')
        
        df_test=gpd.read_file(segmented_shp,encoding="utf-8")
        predictions=modelRandom.predict(df_test[predictor_vars])
        df_test["outcomes"]= predictions

        # Dissolve objects and save final landslides
        
        logging.info('[RandomForest] - Dissolve objects and save final landslides')
        crs=df_test.crs
        df_land=df_test[df_test["outcomes"]>0]
        # print(df_land.unary_union)

        # df_land_dissolve = gpd.geoseries.GeoSeries([geom for geom in df_land.unary_union.geoms])
        df_land_dissolve = gpd.geoseries.GeoSeries([df_land.unary_union])
        df_land_dissolve.crs=crs
        df_land_dissolve.to_file(self.PREDICTED_LS_SHP)

        #Print total time for processing
        final=((time.time()-t0)/60)
        print('Finished in',final, 'mins')
        logging.info('[RandomForest] - Successfully Completed!!')
        return (self.PREDICTED_LS_SHP, )

# out_dir = 'C:/wbtraining/image/results/'
# rf = RandomForest(out_dir)
# rf.run()
