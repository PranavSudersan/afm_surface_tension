
##from afm_analyze import JPKAnalyze, DataFit, DataAnalyze, ImageAnalyze
##from afm_plot import AFMPlot
import sys
##import pandas as pd
##import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
import wetting

app = QApplication(sys.argv)
##file_path, _ = QFileDialog.getOpenFileName()
##fd_file_path = '../20210420 silicone oil tip-pdms brush/force-save-area4-f2_s10-2021.04.20-17.57.38.004.jpk-force'
##
##simu_folderpath = 'E:/Work/Surface Evolver/afm_pyramid/data/20210325_nps/height=0/'
##
###drop analysis of AFM data
##drop_df, file_name = wetting.get_drop_prop(file_path)
##
###get simulation data for tip geometry
##simu_df = wetting.combine_simul_data(simu_folderpath)
##
###calculate contact angle from fd curve
##label = 11 #INPUT
##label_df = drop_df[drop_df['Label']==label]
##s = label_df['s'].iloc[0]
##R = round(label_df['R/s'].iloc[0])
##contact_angle = wetting.get_contact_angle(fd_file_path, simu_df,
##                                          [62,64], R, s)
##
###calculate surface tension
####contact_angle = 10 #INPUT
##output_df = wetting.get_surface_tension(drop_df, simu_df, contact_angle,
##                                        file_name, True)

#combine multiple fd curves
fd_file_paths, _ = QFileDialog.getOpenFileNames()
wetting.combine_fd(fd_file_paths)
