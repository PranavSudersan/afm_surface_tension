
##from afm_analyze import JPKAnalyze, DataFit, DataAnalyze, ImageAnalyze
##from afm_plot import AFMPlot
import sys
##import pandas as pd
##import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
import wetting

app = QApplication(sys.argv)

#file_path, _ = QFileDialog.getOpenFileName()
file_path = 'data/qi-area2-data-2021.07.10-19.22.05.499.jpk-qi-data'
fd_file_paths, _ = QFileDialog.getOpenFileNames()
##fd_file_path, _ = QFileDialog.getOpenFileName()
##fd_file_path = '../20210420 silicone oil tip-pdms brush/force-save-area4-f2_s10-2021.04.20-17.57.38.004.jpk-force'
##
#simu_folderpath = 'E:/Work/Surface Evolver/afm_pyramid/data/20210325_nps/height=0/'
simu_folderpath = 'E:/Work/Surface Evolver/afm_pyramid/data/20210803_oltespa/'

###drop analysis of AFM data
drop_df, file_name = wetting.get_drop_prop(file_path, fd_file_paths)

#get simulation data for tip geometry
simu_df = wetting.combine_simul_data(simu_folderpath)

#calculate contact angle from fd curve
##label = 5 #INPUT
##label_df = drop_df[drop_df['Label']==label]
##s = label_df['s'].iloc[0]
##R = round(label_df['R/s'].iloc[0])
##contact_angle = wetting.get_contact_angle(fd_file_path, simu_df,
##                                          [65,74], R, s)

###calculate surface tension
contact_angle = 10 #INPUT
output_df = wetting.get_surface_tension(drop_df, simu_df, contact_angle,
                                        file_name, True)

###combine multiple fd curves
##fd_file_paths, _ = QFileDialog.getOpenFileNames()
wetting.combine_fd(fd_file_paths)
##wetting.get_adhesion_from_fd(fd_file_paths)
