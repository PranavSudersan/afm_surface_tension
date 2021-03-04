
from afm_analyze import JPKAnalyze, DataFit, DataAnalyze
from afm_plot import AFMPlot
import sys

mode = 'Adhesion' #'Snap-in distance', 'Adhesion' 'Force-distance'
##file_path = 'data/force-save-2021.01.08-17.24.07.418.jpk-force'
file_path = 'data/drops-data-2021.01.08-17.17.02.956.jpk-qi-data'
##file_path = '../20210227_silicone oil/qi-area6-drop1-data-2021.02.27-19.26.57.873.jpk-qi-data'

#import data
jpk_data = JPKAnalyze(file_path, mode, None)

###analyze jpk data
##anal_data = DataAnalyze(jpk_data)
##clusters = anal_data.get_kmeans(2)
##zero_height = clusters.min()
##volume = anal_data.get_volume(zero=zero_height)
##max_height = anal_data.get_max_height(zero_height)

##print('Volume:', volume)
##print('Max height:', max_height)
##print('Zero height:', zero_height)

#plot data
afm_plot = AFMPlot(jpk_data)

###fit data
##data_fit = DataFit(jpk_data, afm_plot, 'Sphere-RC', "Height>=0.5e-7",
##                   guess=[1e-5,-1e-5])
##
##curvature = data_fit.fit_output['R']
##contact_radius = anal_data.get_contact_radius(data_fit.fit_output, zero_height)
##h, V, a = anal_data.get_cap_prop(curvature, data_fit.z_max, zero_height)
##
##print('Curvature radius:', curvature)
##print('Contact radius:', contact_radius)
##print(f'Cap height: {h}, Cap volume: {V}, Cap contact angle: {a}')

#jpk_data.data_zip.close()
