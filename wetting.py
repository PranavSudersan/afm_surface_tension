from afm_analyze import JPKAnalyze, DataFit, DataAnalyze, ImageAnalyze
from afm_plot import AFMPlot
import sys
import os
import pandas as pd
import numpy as np


#drop volume/contact radius/contact angle
def get_drop_prop(file_path):
    #import data
    jpk_data = JPKAnalyze(file_path, None)
    ###analyze jpk data
    
    anal_data = DataAnalyze(jpk_data, 'Snap-in distance')
    clusters = anal_data.get_kmeans(2)
    zero_height = clusters.min()
##    volume = anal_data.get_volume(zero=zero_height)
##    max_height = anal_data.get_max_height(zero_height)

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
                   'Max Height': [], 'Max Height raw': [],
                   'Volume': [], 'Volume raw':[],
                   'Contact angle': [], 'Max Adhesion': []}
    for key in data_fit.fit_output.keys():
        curvature = data_fit.fit_output[key]['R']
        contact_radius = anal_data.get_contact_radius(data_fit.fit_output[key],
                                                      zero_height)
        h, V, a = anal_data.get_cap_prop(curvature,
                                         data_fit.fit_output[key]['z_max'],
                                         zero_height)
        f_max = anal_data.get_max_adhesion(jpk_data, 'Adhesion',
                                           img_anal.coords[key])

        h_raw = anal_data.get_max_height(img_anal.coords[key],
                                         zero=zero_height)
        V_raw = anal_data.get_volume(img_anal.coords[key],
                                     zero=zero_height)
        print(key,h,h_raw,V,V_raw)
        
        output_dict['Label'].append(key)
        output_dict['Curvature'].append(curvature)
        output_dict['Contact Radius'].append(contact_radius)
        output_dict['Max Height'].append(h)
        output_dict['Max Height raw'].append(h)
        output_dict['Volume'].append(V)
        output_dict['Volume raw'].append(V_raw)
        output_dict['Contact angle'].append(a)
        output_dict['Max Adhesion'].append(f_max)

    output_df = pd.DataFrame(output_dict)
    output_df['s'] = ((3*output_df['Volume'])/(4*np.pi))**(1/3)
    output_df['R/s'] = output_df['Contact Radius']/output_df['s']
    output_df['AFM file'] = file_path
    file_name = file_path.split('/')[-1][:-len(jpk_data.file_format)-1]

    return output_df, file_name
##    print('s:', output_df['s'], 'R/s', output_df['R/s'])

def get_surface_tension(output_df, simu_filepath, file_name, save=False):
    #import simulation data
##    simu_filepath = 'E:/Work/Surface Evolver/afm_pyramid/data/20210325_nps/height=0/'\
##                    'data-CA_p30-h 0-Rsi1.5_Rsf3.5.txt'
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
    
    print(output_df['Surface Tension (mN)'])
    #save final output
    if save == True:                    
        output_df.to_excel(f'{file_name}-output.xlsx', index=None)

def get_contact_angle(file_path, simu_folderpath):
    #import data
    jpk_data = JPKAnalyze(file_path, None)
    df = jpk_data.df['Force-distance']
    fit_range = [72,80] #fitting range in percentage INPUT
    num_points = len(df.index)
    fit_slice = slice(int(fit_range[0]*num_points/100),
                      int(fit_range[1]*num_points/100)-1)
    retract_fit = np.polyfit(df['Distance'][fit_slice],
                             df['Force'][fit_slice],1)
    d_retract = df['Distance'][int(num_points/2):]
    f_fit = np.polyval(retract_fit,d_retract)
    wetted_length = (df['Force'].iloc[0]-retract_fit[1])/retract_fit[0] - \
                    d_retract.iloc[0]
##    print(retract_fit, wetted_length, df['Force'].iloc[0])
    afm_plot = AFMPlot(jpk_data)
    afm_plot.ax_fd.plot(d_retract,f_fit)

    #get contact angle from wetted length simulation data
##    folder = os.fsencode(simu_folderpath)
    simu_df = pd.DataFrame()
    with os.scandir(simu_folderpath) as folder:
        for file in folder:
            if file.is_file() and file.path.endswith( ('.txt') ):
                df_temp = pd.read_csv(file.path,delimiter='\t')
                simu_df = simu_df.append(df_temp)
       
    R = 2.8 # R/s INPUT
    s = 2.44e-6 # s value (in meters from jumpin analysis INPUT
    simu_df_filtered = simu_df[simu_df['Contact_Radius'] == R].\
                       sort_values(by=['Average Wetted Height'])
    print(simu_df_filtered['Average Wetted Height'],simu_df_filtered['Top_Angle'])
    #3rd order polynomial fit of Wetted length-Contact angle data
    wa_fit = np.polyfit(simu_df_filtered['Average Wetted Height'],
                        simu_df_filtered['Top_Angle'], 3)
    contact_angle = np.polyval(wa_fit,wetted_length/s)
    print('Contact angle', contact_angle)
