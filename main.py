
##from afm_analyze import JPKAnalyze, DataFit, DataAnalyze, ImageAnalyze
##from afm_plot import AFMPlot
import sys
##import pandas as pd
##import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
import wetting

app = QApplication(sys.argv)

file_path, _ = QFileDialog.getOpenFileName(caption='Select JPK/QI data')
#file_path = 'data/qi-area2-data-2021.07.10-19.22.05.499.jpk-qi-data'
fd_file_paths, _ = QFileDialog.getOpenFileNames(caption='Select JPK force data')
##fd_file_path, _ = QFileDialog.getOpenFileName()
##fd_file_path = '../20210420 silicone oil tip-pdms brush/force-save-area4-f2_s10-2021.04.20-17.57.38.004.jpk-force'
##
simu_folderpath1 = 'E:/Work/Surface Evolver/afm_pyramid/data/20220113_rfesp_np_height0/'
simu_folderpath2 = 'E:/Work/Surface Evolver/afm_pyramid/data/20220201_rfesp_np_fd/'

#drop analysis of AFM data
drop_df, output_path = wetting.get_drop_prop(file_path, fd_file_paths,
                                             level_order=2, fit_range=[60,62])#5000 [60,62]

#get simulation data for tip geometry
simu_df1, _ = wetting.combine_simul_data(simu_folderpath1)
##simu_df2 = wetting.combine_simul_dirs(simu_folderpath2)

#calculate contact angle from fd curve
##label = 5 #INPUT
##label_df = drop_df[drop_df['Label']==label]
##s = label_df['s'].iloc[0]
##R = round(label_df['R/s'].iloc[0])
##contact_angle = wetting.get_contact_angle(fd_file_paths[0], simu_df,
##                                          R, s, fit_index=5000)

#calculate surface tension (from wetted length)
contact_angle = 75 #INPUT (None for auto calculation from FD)
output_df1 = wetting.get_surface_tension(drop_df, simu_df1, contact_angle,
                                        fd_file_paths, output_path, True)
#calculate surface tension from FD fitting
##output_df2 = wetting.get_surface_tension2(drop_df, simu_df2,
##                                          tension_guess_orig=1,tolerance=1,
##                                          fd_file_paths=fd_file_paths,
##                                          file_path=output_path, save=True)

#combine multiple fd curves
##output_path = ''
##fd_file_paths, _ = QFileDialog.getOpenFileNames()
#wetting.combine_fd(fd_file_paths, output_path)
#wetting.get_adhesion_from_fd(fd_file_paths)
