from afm_analyze import JPKAnalyze, DataFit, DataAnalyze, ImageAnalyze
from afm_plot import AFMPlot, simul_plot
import sys
import os
import pandas as pd
import numpy as np


#drop volume/contact radius/contact angle
def get_drop_prop(file_path, fd_file_paths = None):
    #import data
    jpk_data = JPKAnalyze(file_path, None)
    
##    volume = anal_data.get_volume(zero=zero_height)
##    max_height = anal_data.get_max_height(zero_height)

    ##print('Volume:', volume)
    ##print('Max height:', max_height)
    ##print('Zero height:', zero_height)

    #make output directory
    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    output_dir = f'{file_dir}/analysis/{file_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    #plot data
    afm_plot = AFMPlot(jpk_data, output_path=output_dir)

    #analyze adhesion data (get fg/bg)
    anal_data_adh = DataAnalyze(jpk_data, 'Adhesion')
    clusters_adh = anal_data_adh.get_kmeans(2)
    #segment adhesion data
    img_anal = ImageAnalyze(jpk_data, mode='Adhesion')
    img_anal.segment_image(bg=[-1e10,clusters_adh[-1]],
                           fg=[clusters_adh[-1],1e10],
                           output_path=output_dir) #using cutoff from clustering
    ##img_anal.segment_image(bg=[0, 1e-7],
    ##                       fg=[3e-7, 4e-7])

    #analyze height data
    anal_data_h = DataAnalyze(jpk_data, 'Snap-in distance')
    clusters_h = anal_data_h.get_kmeans(2)
    zero_height = clusters_h.min()
    
    ###fit data
    data_fit = DataFit(jpk_data, 'Snap-in distance',
                       afm_plot, 'Sphere-RC', img_anal,
                       zero = zero_height,
                       output_path=output_dir)#,"Height>=0.5e-7",
                       #guess=[1.5e-5,-1e-5],bounds=([1e-6,-np.inf],[1e-4,np.inf]),

    
    #output params
    output_dict = {'Label': [], 'Curvature':[], 'Contact Radius': [],
                   'Max Height': [], 'Max Height raw': [],
                   'Volume': [], 'Volume raw':[],
                   'Contact angle': [], 'Max Adhesion': []}
    print('label','h','h_raw','V','V_raw')

    if fd_file_paths != None:                   
        fd_adhesion_dict = get_adhesion_from_fd(fd_file_paths, jpk_data, img_anal)
        output_dict['Adhesion (FD)'] = []
        output_dict['FD X position'] = []
        output_dict['FD Y position'] = []
        output_dict['FD file'] = []
                
    for key in data_fit.fit_output.keys():
        curvature = data_fit.fit_output[key]['R']
##        contact_radius = anal_data.get_contact_radius(data_fit.fit_output[key],
##                                                      zero_height)
        contact_radius = data_fit.fit_output[key]['base_r_fit']
        h = data_fit.fit_output[key]['h_fit']
        
        V, a = anal_data_h.get_cap_prop(curvature, h)
        f_max = anal_data_h.get_max_adhesion(jpk_data, 'Adhesion',
                                           img_anal.coords[key])

        h_raw = anal_data_h.get_max_height(img_anal.coords[key],
                                         zero=zero_height)
        V_raw = anal_data_h.get_volume(img_anal.coords[key],
                                     zero=zero_height)
        print(key,h,h_raw,V,V_raw)
        
        output_dict['Label'].append(key)
        output_dict['Curvature'].append(curvature)
        output_dict['Contact Radius'].append(contact_radius)
        output_dict['Max Height'].append(h)
        output_dict['Max Height raw'].append(h_raw)
        output_dict['Volume'].append(V)
        output_dict['Volume raw'].append(V_raw)
        output_dict['Contact angle'].append(a)
        output_dict['Max Adhesion'].append(f_max)

        if fd_file_paths != None:
            if key in fd_adhesion_dict.keys():
                val = fd_adhesion_dict[key]
                output_dict['Adhesion (FD)'].append(val[0])
                output_dict['FD X position'].append(val[1])
                output_dict['FD Y position'].append(val[2])
                output_dict['FD file'].append(val[5])
            else:
                output_dict['Adhesion (FD)'].append(0)
                output_dict['FD X position'].append(0)
                output_dict['FD Y position'].append(0)
                output_dict['FD file'].append('')

    output_df = pd.DataFrame(output_dict)
    output_df['s'] = ((3*output_df['Volume'])/(4*np.pi))**(1/3)
    output_df['R/s'] = output_df['Contact Radius']/output_df['s']
    output_df['AFM file'] = file_path
    #file_name = file_path.split('/')[-1][:-len(jpk_data.file_format)-1]

    return output_df, output_dir
##    print('s:', output_df['s'], 'R/s', output_df['R/s'])

def get_surface_tension(output_df, simu_df, contact_angle, fd_file_paths,
                        file_path, save=False):
    
    #import simulation data
##    simu_filepath = 'E:/Work/Surface Evolver/afm_pyramid/data/20210325_nps/height=0/'\
##                    'data-CA_p30-h 0-Rsi1.5_Rsf3.5.txt'
##    simu_df = pd.read_csv(simu_filepath,delimiter='\t')
    ca_nearest = min(simu_df['Top_Angle'].unique(),
                     key=lambda x:abs(x-contact_angle))
    simu_df_filtered = simu_df[simu_df['Top_Angle'] == ca_nearest].reset_index()
##    simu_df_filtered['ys/F'] = -1/(2*np.pi*simu_df_filtered['Force_Calc']) #inverse

    #3rd order polynomial fit of Force-Contact radius data
    fr_fit = np.polyfit(simu_df_filtered['Contact_Radius'], simu_df_filtered['Force_Calc'], 3)
    output_df['F_fit'] = np.polyval(fr_fit,output_df['R/s'])
    output_df['ys/F'] = -1/(2*np.pi*output_df['F_fit']) #inverse

    #surface tension in mN/m
    output_df['Surface Tension (mN)'] = 1000*output_df['ys/F']*output_df['Max Adhesion']/\
                                   output_df['s']

    #miscellaneous data
    output_df['Simulation contact angle'] = ca_nearest
    output_df['Simulation file'] = simu_df_filtered['File path'].iloc[0]
    
    print(output_df['Surface Tension (mN)'])

    if fd_file_paths != None:
        output_df['Surface Tension FD (mN)'] = 1000*output_df['ys/F']*output_df['Adhesion (FD)']/\
                                               output_df['s']
    
    #save final output
    if save == True:
        output_df.to_excel(f'{file_path}/output.xlsx', index=None)

    return output_df

def get_contact_angle(file_path, simu_df, fit_range, R, s):
    #import data
    jpk_data = JPKAnalyze(file_path, None)
    df = jpk_data.df['Force-distance']
##    fit_range = [72,80] #fitting range in percentage INPUT
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
    ylim = afm_plot.ax_fd.get_ylim()
    afm_plot.ax_fd.plot(d_retract,f_fit)
    afm_plot.ax_fd.set_ylim(ylim)

    #get contact angle from wetted length simulation data
##    folder = os.fsencode(simu_folderpath)
##    simu_df = pd.DataFrame()
##    with os.scandir(simu_folderpath) as folder:
##        for file in folder:
##            if file.is_file() and file.path.endswith( ('.txt') ):
##                df_temp = pd.read_csv(file.path,delimiter='\t')
##                simu_df = simu_df.append(df_temp)

    R = min([3.5,R]) #CHECK THIS   
##    R = 2.8 # R/s INPUT
##    s = 2.44e-6 # s value (in meters from jumpin analysis INPUT
    #TODO: use interpolated values instead of filtering
    simu_df_filtered = simu_df[simu_df['Contact_Radius'] == R].\
                       sort_values(by=['Average Wetted Height'])
##    print(simu_df_filtered['Average Wetted Height'],simu_df_filtered['Top_Angle'])
    #3rd order polynomial fit of Wetted length-Contact angle data
    wa_fit = np.polyfit(simu_df_filtered['Average Wetted Height'],
                        simu_df_filtered['Top_Angle'], 3)
    contact_angle = np.polyval(wa_fit,wetted_length/s)
    print('Contact angle', contact_angle)
    return contact_angle

def combine_simul_data(simu_folderpath):
    simu_df = pd.DataFrame()
    with os.scandir(simu_folderpath) as folder:
        for file in folder:
            if file.is_file() and file.path.endswith( ('.txt') ):
                df_temp = pd.read_csv(file.path,delimiter='\t')
                df_temp['ys/F'] = -1/(2*np.pi*df_temp['Force_Calc']) #inverse
                df_temp['File path'] = file.path
                simu_df = simu_df.append(df_temp)
                 
    simu_df['Simulation folder'] = simu_folderpath
    #simul_plot(simu_df)
    return simu_df

def get_adhesion_from_fd(fd_file_paths, jpk_map, img_anal):
    df_adh = jpk_map.df['Adhesion']
    x_array = df_adh['X']
    y_array = df_adh['Y']
    data_dict = {}
    for file_path in fd_file_paths:
        #import data
        jpk_data = JPKAnalyze(file_path, None)
        df = jpk_data.df['Force-distance']
        #find label of fd point
        x_pos = df['X'].loc[0]
        y_pos = df['Y'].loc[0]
        x_real = df_adh['X'].loc[np.abs(x_array - x_pos).argmin()]
        y_real = df_adh['Y'].loc[np.abs(y_array - y_pos).argmin()]
        x_index = np.where(img_anal.im_df.columns == x_real)[0]
        y_index = np.where(img_anal.im_df.index == y_real)[0]
        #print(y_index,x_index,img_anal.masked[y_index,x_index])
        try:
            label = int(img_anal.masked[y_index,x_index])
            #get adhesion
            force_data = df['Force'].to_numpy()
            adhesion = force_data[-1] - force_data.min()
            data_dict[label] = [adhesion, x_pos, y_pos, x_real, y_real, file_path]
        except Exception as e:
            print(e)
            pass
    return data_dict

    
def combine_fd(file_paths, output_dir=None):
    afm_plot = AFMPlot()
    mode = 'Force-distance'
    for file_path in file_paths:
        fd_data = JPKAnalyze(file_path, None)        
        plot_params = fd_data.ANALYSIS_MODE_DICT[mode]['plot_parameters']
        label_text = file_path.split('/')[-1].split('-')[4] #CHANGE INDEX   
        afm_plot.plot_line(fd_data.df[mode], plot_params, label_text)

    #legend remove duplicates
    handles, labels = afm_plot.ax_fd.get_legend_handles_labels()            
    leg_dict = dict(zip(labels[::-1],handles[::-1]))
    afm_plot.ax_fd.get_legend().remove()
    leg = afm_plot.ax_fd.legend(leg_dict.values(), leg_dict.keys())
    leg.set_draggable(True, use_blit=True)

    afm_plot.fig_fd.savefig(f'{output_dir}/FD_curves.png', bbox_inches = 'tight',
                            transparent = True)
    
    afm_plot.fig_fd.show()
        
#jpk_data.data_zip.close()
