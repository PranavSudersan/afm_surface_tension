
##from afm_analyze import JPKAnalyze, DataFit, DataAnalyze, ImageAnalyze
##from afm_plot import AFMPlot
import sys
##import pandas as pd
##import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
import wetting

app = QApplication(sys.argv)
file_path, _ = QFileDialog.getOpenFileName()

##mode = 'Adhesion' #'Snap-in distance', 'Adhesion' 'Force-distance'
##fd_file_path = 'data/force-save-drop-2021.03.21-23.58.25.964.jpk-force'
##file_path = 'data/qi-petri-dish-data-2021.03.21-23.34.20.564.jpk-qi-data'
##file_path = 'data/qi-petri-dish-data-2021.03.21-23.28.09.998.jpk-qi-data'
##fd_file_path = 'data/force-save-bottomleft-drop-2021.03.21-22.13.18.662.jpk-force'
##file_path = 'data/qi-petri-dish-data-2021.03.21-22.04.48.038.jpk-qi-data'
##fd_file_path = 'data/force-save-drop_speed0.5-2021.03.24-15.06.40.432.jpk-force'
##file_path = 'data/qi-drop-data-2021.03.24-14.57.43.653.jpk-qi-data'

simu_folderpath = 'E:/Work/Surface Evolver/afm_pyramid/data/20210325_nps/height=0/'

#drop analysis of AFM data
drop_df, file_name = wetting.get_drop_prop(file_path)

#get simulation data for tip geometry
simu_df = wetting.combine_simul_data(simu_folderpath)

#calculate contact angle from fd curve
##label = 1 #INPUT
##label_df = drop_df[drop_df['Label']==label]
##s = label_df['s'].iloc[0]
##R = round(label_df['R/s'].iloc[0])
##contact_angle = wetting.get_contact_angle(fd_file_path, simu_df,
##                                          [72,82], R, s)

#calculate surface tension
contact_angle = 60 #INPUT
output_df = wetting.get_surface_tension(drop_df, simu_df, contact_angle,
                                        file_name, True)
