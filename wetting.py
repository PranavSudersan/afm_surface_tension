from afm_analyze import JPKAnalyze, DataFit, DataAnalyze, ImageAnalyze
from afm_plot import AFMPlot, simul_plot
from matplotlib.pyplot import cm
import sys
import os
import pandas as pd
import numpy as np


#drop volume/contact radius/contact angle
def get_drop_prop(file_path, fd_file_paths = None, level_order=1, fit_range=10000):
    #import data
    jpk_data = JPKAnalyze(file_path, None)

    if jpk_data.file_format == 'jpk':
        segment_mode = 'Height (measured)'
        volume_mode = 'Height (measured)'
    elif jpk_data.file_format == 'jpk-qi-data':
        segment_mode = 'Adhesion'
        volume_mode = 'Snap-in distance'

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
    points_data = afm_plot.points_data #collect points to fit plane

    #analyze height data    
    anal_data_h = DataAnalyze(jpk_data, volume_mode)
    if len(points_data)!=0:#tilt correction
        anal_data_h.level_data(points_data, order=level_order)
        jpk_data.df[volume_mode] = anal_data_h.df.copy()
        jpk_data.ANALYSIS_MODE_DICT[volume_mode]['plot_parameters']['z'] += ' corrected'
        jpk_data.ANALYSIS_MODE_DICT[volume_mode]['plot_parameters']['title'] += ' corrected'
        afm_plot.__init__(jpk_data, output_path=output_dir)
        zero_height = 0
    else:   
        clusters_h = anal_data_h.get_kmeans(2)
        zero_height = clusters_h.min()

    #analyze adhesion data for segmentation (get fg/bg)
    anal_data_adh = DataAnalyze(jpk_data, segment_mode)
    clusters_adh = anal_data_adh.get_kmeans(2)
    
    #segment adhesion data
    img_anal = ImageAnalyze(jpk_data, mode=segment_mode)
    img_anal.segment_image(bg=[-1e10,clusters_adh[-1]],
                           fg=[clusters_adh[-1],1e10],
                           output_path=output_dir) #using cutoff from clustering initially
    ##img_anal.segment_image(bg=[0, 1e-7],
    ##                       fg=[3e-7, 4e-7])
    
    ###fit data
    data_fit = DataFit(jpk_data, volume_mode,
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
        fd_adhesion_dict = get_adhesion_from_fd(fd_file_paths, jpk_data,
                                                img_anal, segment_mode,
                                                fit_range = fit_range,
                                                output_path = output_dir)
        output_dict['Adhesion (FD)'] = []
        output_dict['Slope (FD)'] = []
        output_dict['Wetted length (FD)'] = []
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
        if jpk_data.file_format == 'jpk-qi-data':
            f_max = anal_data_h.get_max_adhesion(jpk_data, 'Adhesion',
                                               img_anal.coords[key])
        else:
            f_max = None

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
                output_dict['Slope (FD)'].append(val[6])
                output_dict['Wetted length (FD)'].append(val[7])
                output_dict['FD X position'].append(val[1])
                output_dict['FD Y position'].append(val[2])
                output_dict['FD file'].append(val[5])
            else:
                output_dict['Adhesion (FD)'].append(0)
                output_dict['Slope (FD)'].append(0)
                output_dict['Wetted length (FD)'].append(0)
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
    if contact_angle != None:
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
        if contact_angle != None:
            output_df['Surface Tension FD (mN)'] = 1000*output_df['ys/F']*output_df['Adhesion (FD)']/\
                                                   output_df['s']
        else:
            output_df['F_fit'] = 0.0
            output_df['ys/F'] = 0.0
            output_df['Simulation contact angle'] = 0.0
            output_df['Simulation file'] = ''
            for i in output_df.index:
##                R = min([3.5,round(output_df['R/s'].loc[i])]) #CHECK 3.5 (limit)
                R = min(simu_df['Contact_Radius'].unique(),
                        key=lambda x:abs(x-output_df['R/s'].loc[i]))
                wetted_length = output_df['Wetted length (FD)'].loc[i]
                s = output_df['s'].loc[i]
                F_adh = output_df['Adhesion (FD)'].loc[i]
                print(output_df['R/s'].loc[i],R,s)
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
                ca_nearest = min(simu_df['Top_Angle'].unique(),
                                 key=lambda x:abs(x-contact_angle))
                print('Contact angle', contact_angle,ca_nearest)
                simu_df_filtered2 = simu_df[simu_df['Top_Angle'] == ca_nearest].reset_index()
            ##    simu_df_filtered['ys/F'] = -1/(2*np.pi*simu_df_filtered['Force_Calc']) #inverse

                #3rd order polynomial fit of Force-Contact radius data
                fr_fit = np.polyfit(simu_df_filtered2['Contact_Radius'],
                                    simu_df_filtered2['Force_Calc'], 3)
                F_fit = np.polyval(fr_fit,R)
                ys_f = -1/(2*np.pi*F_fit) #inverse
                tension = 1000*ys_f*F_adh/s
                
                output_df.at[i,'Surface Tension FD (mN)'] = tension
                output_df.at[i,'Simulation contact angle'] = contact_angle
                output_df.at[i,'F_fit'] = F_fit
                output_df.at[i,'ys/F'] = ys_f
                output_df.at[i,'Simulation file'] = simu_df_filtered2['File path'].iloc[0]
                
    
        print(output_df['Surface Tension FD (mN)'])
        
    #save final output
    if save == True:
        afm_filename = output_df['AFM file'].iloc[0].split('/')[-1][:-4]
        output_df.to_excel(f'{file_path}/output_FR-{afm_filename}.xlsx', index=None)

    return output_df

def get_contact_angle(file_path, simu_df, R, s, fit_index=10000):
    #import data
    jpk_data = JPKAnalyze(file_path, None)
    df = jpk_data.df['Force-distance']
##    fit_range = [72,80] #fitting range in percentage INPUT
    num_points = len(df.index)
    force_data = df['Force'].to_numpy()
    adh_id = np.argmin(force_data)
    fit_range = [adh_id, adh_id+fit_index] #CHECK 10000, make adjustable
    fit_slice = slice(fit_range[0],fit_range[1])
##    fit_slice = slice(int(fit_range[0]*num_points/100),
##                      int(fit_range[1]*num_points/100)-1)
    retract_fit = np.polyfit(df['Distance'][fit_slice],
                             df['Force'][fit_slice],1)
    print('FD Fit:', retract_fit)
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

#calculate surface tension iteratively from force-distance curves
def get_surface_tension2(output_df, simu_df, tension_guess_orig, tolerance, fd_file_paths,
                        file_path, save=False):
    if fd_file_paths != None:
        output_df['ys/F'] = 0.0
        output_df['Surface Tension FD (mN)'] = 0.0
        output_df['Simulation contact angle'] = 0.0  
        output_df['Simulation file'] = ''
        output_df_fd = output_df[output_df['FD file'] != ''] #filter out drops without force data
        for i in output_df_fd.index:#range(len(output_df.index)):
            contact_radius = output_df['R/s'].loc[i]
            rs_nearest = min(simu_df['Contact_Radius'].unique(),
                             key=lambda x:abs(x-contact_radius))
            simu_df_filtered = simu_df[simu_df['Contact_Radius'] == rs_nearest].reset_index()
        ##    simu_df_filtered['ys/F'] = -1/(2*np.pi*simu_df_filtered['Force_Calc']) #inverse

            #linear fit of Force-distance data
            fd_fit_dict = {}
            fa_dict = {} #fmax - angle data
            for top_angle in simu_df_filtered['Top_Angle'].unique():
                df_temp = simu_df_filtered[simu_df_filtered['Top_Angle'] == top_angle]
                fd_fit = np.polyfit(df_temp['Height'], df_temp['Force_Calc'], 1)
                fd_fit_dict[top_angle]= fd_fit[0]
                fa_dict[top_angle]= df_temp[df_temp['Height'] == 0.0]['Force_Calc'].iloc[0]
            print(fd_fit_dict, fa_dict)
            angle_slope_fit = np.polyfit(list(map(float,fd_fit_dict.values())),
                                         list(map(float,fd_fit_dict.keys())), 2)
            print(angle_slope_fit)
            force_angle_fit = np.polyfit(list(map(float,fa_dict.keys())),
                                         list(map(float,fa_dict.values())), 2)
            print(force_angle_fit)
            #tension_guess_orig = np.linspace(70,1,140)
            tension_guess = tension_guess_orig
            #j = 0
            while True:
                #tension_guess = tension_guess_orig[j]
                slope_expt = output_df['Slope (FD)'].loc[i]/(2*np.pi*tension_guess/1000)
                #slope_expt = 0.042/(2*np.pi*tension_guess/1000)
##                contact_angle = angle_slope_fit[0]*slope_expt + angle_slope_fit[1]
                contact_angle = np.polyval(angle_slope_fit,slope_expt)
                contact_angle = 1 if contact_angle < 0 else contact_angle
                contact_angle = 90 if contact_angle > 90 else contact_angle
##                contact_angle_nearest = min(simu_df_filtered['Top_Angle'].unique(),
##                                            key=lambda x:abs(x-contact_angle))
##                df_temp2 = simu_df_filtered[simu_df_filtered['Top_Angle'] == contact_angle_nearest]
##                fmax_simul = df_temp2[df_temp2['Height'] == 0.0]['Force_Calc'].iloc[0]
                fmax_simul = np.polyval(force_angle_fit,contact_angle)
                ys_f = -1/(2*np.pi*fmax_simul) #inverse y*s/F
                tension_new = 1000*ys_f*output_df['Adhesion (FD)'].loc[i]/\
                                               output_df['s'].loc[i]
                print('tension guess', tension_guess, 'tension new;', tension_new,
                      'contact angle', contact_angle, 'slope', slope_expt)
                #if abs((tension_new-tension_guess)*100/tension_guess) > tolerance:
                if abs(tension_new-tension_guess) > tolerance:
                    tension_guess = tension_new
                    #j += 1
                else:
                    break
            print('here')
            output_df.at[i,'Surface Tension FD (mN)'] = tension_new
            output_df.at[i,'Simulation contact angle'] = contact_angle
            output_df.at[i,'ys/F'] = ys_f
            output_df.at[i,'Simulation file'] = simu_df_filtered['File path'].iloc[0]

        print(output_df['Surface Tension FD (mN)'])
        
        #save final output
        if save == True:
            afm_filename = output_df['AFM file'].iloc[0].split('/')[-1][:-4]
            output_df.to_excel(f'{file_path}/output_FD-{afm_filename}.xlsx', index=None)

    return output_df


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

#combine data from subfolders
def combine_simul_dirs(simu_folderpath):
    simu_df = pd.DataFrame()
    with os.scandir(simu_folderpath) as folder:
        for fdr in folder:
            if fdr.is_dir():
                df_temp = combine_simul_data(fdr.path)
                simu_df = simu_df.append(df_temp)
    #simu_df.to_excel('simu_out.xlsx')
    return simu_df

#get adhesion and slope from force data files
def get_adhesion_from_fd(fd_file_paths, jpk_map, img_anal, segment_mode,
                         fit_range=10000, output_path=None):
    df_adh = jpk_map.df[segment_mode]
    x_array = df_adh['X']
    y_array = df_adh['Y']
    data_dict = {}
    afm_plot = AFMPlot()
    #colors
    evenly_spaced_interval = np.linspace(0, 1, len(fd_file_paths))
    colors = [cm.rainbow(x) for x in evenly_spaced_interval]
    for i, file_path in enumerate(fd_file_paths):
        #import data
        jpk_data = JPKAnalyze(file_path, None)
        df = jpk_data.df['Force-distance']
        plot_params = jpk_data.ANALYSIS_MODE_DICT['Force-distance']['plot_parameters']
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

            #get FD slope at retract
            num_points = len(df.index)
            adh_id = np.argmin(force_data)
            if type(fit_range).__name__ == 'int': #fit based on adhesion point
                fit_slice = slice(adh_id,adh_id+fit_range)
            elif type(fit_range).__name__ == 'list': #fit based on range percentage
                #fit_range = [70,75] 
                fit_slice = slice(int(fit_range[0]*num_points/100),
                                  int(fit_range[1]*num_points/100)-1)
            
            retract_fit = np.polyfit(df['Distance'][fit_slice],
                                     df['Force'][fit_slice],1)
            print('FD Fit:', retract_fit)
            #get wetted length
            d_retract = df['Distance'][int(num_points/2):]
            f_fit = np.polyval(retract_fit,d_retract)
            wetted_length = (df['Force'].iloc[0]-retract_fit[1])/retract_fit[0] - \
                            d_retract.iloc[0]

            #plot fit line
            label_text = str(label)
            
            afm_plot.plot_line(df, plot_params, label_text=label_text, color=colors[i])
            ylim = afm_plot.ax_fd.get_ylim()
            afm_plot.ax_fd.plot(d_retract,f_fit, label=label_text, color=colors[i])
            afm_plot.ax_fd.set_ylim(ylim)
        
            data_dict[label] = [adhesion, x_pos, y_pos, x_real, y_real,
                                file_path, retract_fit[0], wetted_length]
        except Exception as e:
            print(e)
            pass
    if len(fd_file_paths) != 0:
        #legend remove duplicates and sort
        handles, labels = afm_plot.ax_fd.get_legend_handles_labels()
        handles, labels = zip(*[ (handles[j], labels[j]) for j in sorted(range(len(handles)),
                                                                         key=lambda k: labels[k],
                                                                         reverse=True)])
        leg_dict = dict(zip(labels[::-1],handles[::-1]))
        afm_plot.ax_fd.get_legend().remove()
        leg = afm_plot.ax_fd.legend(leg_dict.values(), leg_dict.keys())
        leg.set_draggable(True, use_blit=True)

        afm_plot.ax_fd.autoscale(enable=True)
        afm_plot.fig_fd.savefig(f'{output_path}/FD_curves.png', bbox_inches = 'tight',
                                transparent = False)
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
