
from afm_analyze import JPKAnalyze, DataFit, DataAnalyze, ImageAnalyze
from afm_plot import AFMPlot
import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog

##app = QApplication(sys.argv)
##file_path, _ = QFileDialog.getOpenFileName()

##mode = 'Adhesion' #'Snap-in distance', 'Adhesion' 'Force-distance'
##file_path = 'data/force-save-2021.01.08-17.24.07.418.jpk-force'
##file_path = 'data/drops-data-2021.01.08-17.17.02.956.jpk-qi-data'
file_path = '../20210321 silicone oil fluorinated tip/qi-petri-dish-data-2021.03.21-23.28.09.998.jpk-qi-data'
##file_path = '../20210221_glycerin drop/qi-drop6-full2-data-2021.02.21-21.56.03.718.jpk-qi-data'
##file_path = '../20210227_silicone oil/qi-area6-data-2021.02.27-19.40.31.051.jpk-qi-data'

#import data
jpk_data = JPKAnalyze(file_path, None)

###analyze jpk data
#TODO: do this for each label of segment
anal_data = DataAnalyze(jpk_data, 'Snap-in distance')
clusters = anal_data.get_kmeans(2)
zero_height = clusters.min()
volume = anal_data.get_volume(zero=zero_height)
max_height = anal_data.get_max_height(zero_height)

##print('Volume:', volume)
##print('Max height:', max_height)
##print('Zero height:', zero_height)

#plot data
afm_plot = AFMPlot(jpk_data)

#segment image
##anal_adh = DataAnalyze(jpk_data, 'Adhesion')
##clusters_adh = anal_adh.get_kmeans(2)
img_anal = ImageAnalyze(jpk_data, 'Snap-in distance')
img_anal.segment_image(bg=[-1e10,clusters.min()],
                       fg=[clusters.max(),1e10])
##img_anal.segment_image(bg=[0, 1e-7],
##                       fg=[3e-7, 4e-7])

###fit data
data_fit = DataFit(jpk_data, afm_plot, 'Sphere-RC', img_anal,
                   zero = zero_height)#,"Height>=0.5e-7",
                   #guess=[1.5e-5,-1e-5],bounds=([1e-6,-np.inf],[1e-4,np.inf]),
                   

#output params
output_dict = {'Label': [], 'Curvature':[], 'Contact Radius': [],
               'Max Height': [], 'Volume': [], 'Contact angle': [],
               'Max Adhesion': []}
for key in data_fit.fit_output.keys():
    curvature = data_fit.fit_output[key]['R']
    contact_radius = anal_data.get_contact_radius(data_fit.fit_output[key],
                                                  zero_height)
    h, V, a = anal_data.get_cap_prop(curvature,
                                     data_fit.fit_output[key]['z_max'],
                                     zero_height)
    f_max = anal_data.get_max_adhesion(jpk_data, 'Adhesion',
                                       img_anal.coords[key])
    output_dict['Label'].append(key)
    output_dict['Curvature'].append(curvature)
    output_dict['Contact Radius'].append(contact_radius)
    output_dict['Max Height'].append(h)
    output_dict['Volume'].append(V)
    output_dict['Contact angle'].append(a)
    output_dict['Max Adhesion'].append(f_max)

output_df = pd.DataFrame(output_dict)
output_df['s'] = ((3*output_df['Volume'])/(4*np.pi))**(1/3)
output_df['R/s'] = output_df['Contact Radius']/output_df['s']

#import simulation data
simu_filepath = 'E:/Work/Surface Evolver/afm_pyramid/data/20210325_nps/height=0/'\
                'data-CA_p30-h 0-Rsi1.5_Rsf3.5.txt'
simu_df = pd.read_csv(simu_filepath,delimiter='\t')
simu_df['ys/F'] = -1/(2*np.pi*simu_df['Force_Calc']) #inverse

#3rd order polynomial fit of Force-Contact radius data
fr_fit = np.polyfit(simu_df['Contact_Radius'], simu_df['ys/F'], 3)

output_df['ys/F'] = np.polyval(fr_fit,output_df['R/s'])

#surface tension in mN/m
output_df['Surface Tension (mN)'] = 1000*output_df['ys/F']*output_df['Max Adhesion']/\
                               output_df['s']

#miscellaneous data
output_df['Simulation contact angle'] = simu_df['Top_Angle'][0]
output_df['Simulation file'] = simu_filepath
output_df['AFM file'] = file_path

#save final output
file_name = file_path.split('/')[-1][:-len(jpk_data.file_format)-1]
output_df.to_excel(f'{file_name}-output.xlsx', index=None)


#jpk_data.data_zip.close()
