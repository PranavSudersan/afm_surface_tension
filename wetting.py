from afm_analyze import JPKAnalyze, DataFit, DataAnalyze, ImageAnalyze
from afm_plot import AFMPlot, simul_plot
from matplotlib.pyplot import cm
from matplotlib import pyplot as plt
import sys
import os
import copy
import pandas as pd
import numpy as np
from scipy import integrate


def get_afm_image(file_path, output_dir, level_order=1, jump_tol=0.8):
    #import data
    jpk_data = JPKAnalyze(file_path, None, jump_tol=jump_tol)
    
    if jpk_data.file_format == 'jpk':
        segment_mode = 'Height (measured)'
        volume_mode = 'Height (measured)'
        rotation_info = jpk_data.rotation_info
    elif jpk_data.file_format == 'jpk-qi-data':
        segment_mode = 'Snap-in distance' #Adhesion
        volume_mode = 'Snap-in distance'
        rotation_info = None #CHECK for QI!

##    volume = anal_data.get_volume(zero=zero_height)
##    max_height = anal_data.get_max_height(zero_height)

    ##print('Volume:', volume)
    ##print('Max height:', max_height)
    ##print('Zero height:', zero_height)

#     #make output directory
#     if file_dir == '':
#         file_dir = os.path.dirname(file_path) + '/analysis'
#     file_name = os.path.basename(file_path)
#     output_dir = f'{file_dir}/{file_name}'
#     os.makedirs(output_dir, exist_ok=True)
    
    
    #plot data
    afm_plot = AFMPlot(jpk_data, output_path=output_dir)
    points_data = afm_plot.points_data #collect points to fit plane

    #correct height data  
    anal_data_h = DataAnalyze(jpk_data, volume_mode)
    if len(points_data)!=0:#tilt correction
        anal_data_h.level_data(points_data, order=level_order)
        jpk_data.df[volume_mode] = anal_data_h.df.copy()
        #jpk_data.ANALYSIS_MODE_DICT[volume_mode]['plot_parameters']['z'] += ' corrected'
        #jpk_data.ANALYSIS_MODE_DICT[volume_mode]['plot_parameters']['title'] += ' corrected'
        #afm_plot.__init__(jpk_data, output_path=output_dir)
        #zero_height = 0
    else:   
        clusters_h = anal_data_h.get_kmeans(2)
        zero_height = clusters_h.min()
        z_param = jpk_data.ANALYSIS_MODE_DICT[volume_mode]['plot_parameters']['z']
        jpk_data.df[volume_mode][z_param + ' corrected'] = jpk_data.df[volume_mode][z_param]-zero_height
    jpk_data.ANALYSIS_MODE_DICT[volume_mode]['plot_parameters']['z'] += ' corrected'
    jpk_data.ANALYSIS_MODE_DICT[volume_mode]['plot_parameters']['title'] += ' corrected'
    afm_plot.__init__(jpk_data, output_path=output_dir)
    
    fig_list = jpk_data.ANALYSIS_MODE_DICT['Misc']['figure_list']
    
    return jpk_data, anal_data_h, fig_list
    
        
#drop volume/contact radius/contact angle
def get_drop_prop(jpk_data, anal_data_h, output_dir, level_order=1):          
    #import data
    # jpk_data = JPKAnalyze(file_path, None)
    
    #output data params
    output_dict = {'Label': [], 'Curvature':[], 'Contact Radius': [],
                   'Max Height': [], 'Max Height raw': [],
                   'Volume': [], 'Volume raw':[],
                   'Drop contact angle': []}
    
    if jpk_data.file_format == 'jpk':
        segment_mode = 'Height (measured)'
        volume_mode = 'Height (measured)'
        #rotation_info = jpk_data.rotation_info
    elif jpk_data.file_format == 'jpk-qi-data':
        segment_mode = 'Snap-in distance' #Adhesion
        volume_mode = 'Snap-in distance'
        #rotation_info = None #CHECK for QI!
        output_dict['Max Adhesion'] = []

##    volume = anal_data.get_volume(zero=zero_height)
##    max_height = anal_data.get_max_height(zero_height)

    ##print('Volume:', volume)
    ##print('Max height:', max_height)
    ##print('Zero height:', zero_height)

#     #make output directory
#     if file_dir == '':
#         file_dir = os.path.dirname(file_path) + '/analysis'
#     file_name = os.path.basename(file_path)
#     output_dir = f'{file_dir}/{file_name}'
#     os.makedirs(output_dir, exist_ok=True)
    
    fig_list = []
    #plot data
#     afm_plot = AFMPlot(jpk_data, output_path=output_dir)
#     points_data = afm_plot.points_data #collect points to fit plane

#     #analyze height data    
#     anal_data_h = DataAnalyze(jpk_data, volume_mode)
#     if len(points_data)!=0:#tilt correction
#         anal_data_h.level_data(points_data, order=level_order)
#         jpk_data.df[volume_mode] = anal_data_h.df.copy()
#         jpk_data.ANALYSIS_MODE_DICT[volume_mode]['plot_parameters']['z'] += ' corrected'
#         jpk_data.ANALYSIS_MODE_DICT[volume_mode]['plot_parameters']['title'] += ' corrected'
#         afm_plot.__init__(jpk_data, output_path=output_dir)
#         zero_height = 0
#     else:   
#         clusters_h = anal_data_h.get_kmeans(2)
#         zero_height = clusters_h.min()
    zero_height = 0
    
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
    fig_list.append(img_anal.fig)
    
    ###fit data
    data_fit = DataFit(jpk_data, volume_mode,
                       'Sphere-RC', img_anal,
                       zero = zero_height,
                       output_path=output_dir)#,"Height>=0.5e-7",
                       #guess=[1.5e-5,-1e-5],bounds=([1e-6,-np.inf],[1e-4,np.inf]),
    fig_list.append(data_fit.fig)

    
    #print('label','h','h_raw','V','V_raw')

#     if len(fd_file_paths) != 0:                   
#         fd_adhesion_dict = get_adhesion_from_fd(fd_file_paths, jpk_data,
#                                                 img_anal, segment_mode,
#                                                 force_cycle = force_cycle,
#                                                 rotation_info = rotation_info,
#                                                 output_path = output_dir)
#         output_dict['Adhesion (FD)'] = []
#         output_dict['Slope (FD)'] = []
#         output_dict['Wetted length (FD)'] = []
#         output_dict['Rupture distance (FD)'] = []
#         output_dict['Adhesion energy (FD)'] = []
#         output_dict['FD X position'] = []
#         output_dict['FD Y position'] = []
#         output_dict['FD file'] = []
    #anal_data_h = DataAnalyze(jpk_data, volume_mode)            
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
            output_dict['Max Adhesion'].append(f_max)
        else:
            f_max = None

        h_raw = anal_data_h.get_max_height(img_anal.coords[key],
                                         zero=zero_height)
        V_raw = anal_data_h.get_volume(img_anal.coords[key],
                                     zero=zero_height)
        #print(key,h,h_raw,V,V_raw)
        
        output_dict['Label'].append(key)
        output_dict['Curvature'].append(curvature)
        output_dict['Contact Radius'].append(contact_radius)
        output_dict['Max Height'].append(h)
        output_dict['Max Height raw'].append(h_raw)
        output_dict['Volume'].append(V)
        output_dict['Volume raw'].append(V_raw)
        output_dict['Drop contact angle'].append(a)
        

#         if len(fd_file_paths) != 0:
#             if key in fd_adhesion_dict.keys():
#                 val = fd_adhesion_dict[key]
#                 output_dict['Adhesion (FD)'].append(val[0])
#                 output_dict['Slope (FD)'].append(val[6])
#                 output_dict['Wetted length (FD)'].append(val[7])
#                 output_dict['Rupture distance (FD)'].append(val[8])
#                 output_dict['Adhesion energy (FD)'].append(val[9])
#                 output_dict['FD X position'].append(val[1])
#                 output_dict['FD Y position'].append(val[2])
#                 output_dict['FD file'].append(val[5])
#             else:
#                 output_dict['Adhesion (FD)'].append(0)
#                 output_dict['Slope (FD)'].append(0)
#                 output_dict['Wetted length (FD)'].append(0)
#                 output_dict['Rupture distance (FD)'].append(0)
#                 output_dict['Adhesion energy (FD)'].append(0)
#                 output_dict['FD X position'].append(0)
#                 output_dict['FD Y position'].append(0)
#                 output_dict['FD file'].append('')

    output_df = pd.DataFrame(output_dict)
    output_df['s'] = ((3*output_df['Volume'])/(4*np.pi))**(1/3)
    output_df['R/d'] = output_df['Contact Radius']/output_df['Max Height']
    #output_df['AFM file'] = file_path
    #file_name = file_path.split('/')[-1][:-len(jpk_data.file_format)-1]

    return output_df, img_anal, fig_list
##    print('s:', output_df['s'], 'R/s', output_df['R/s'])

                
#get adhesion and slope from force data files
def analyze_drop_fd(fd_file_paths, jpk_map, img_anal,
                    force_cycle='approach', fit_order=1, output_path=None):
    
    if jpk_map.file_format == 'jpk':
        segment_mode = 'Height (measured)'
        #volume_mode = 'Height (measured)'
        rotation_info = jpk_map.rotation_info
    elif jpk_map.file_format == 'jpk-qi-data':
        segment_mode = 'Snap-in distance' #Adhesion
        #volume_mode = 'Snap-in distance'
        rotation_info = None #CHECK for QI!
    
    fig_list = []
    
    df_adh = jpk_map.df[segment_mode]
    x_array = df_adh['X']
    y_array = df_adh['Y']
    data_dict = {'Label': [],
                'Adhesion (FD)': [],
                'Slope (FD)': [],
                'Wetted length (FD)': [],
                'Rupture distance (FD)': [],
                'Adhesion energy (FD)': [],
                'FD X position': [],
                'FD Y position': [],
                'FD file' : []
                }
    afm_plot = AFMPlot()
    #colors
    evenly_spaced_interval = np.linspace(0, 1, len(fd_file_paths))
    colors = [cm.rainbow(x) for x in evenly_spaced_interval]
    for i, file_path in enumerate(fd_file_paths):
        #import data
        print('Force file:', file_path)
        jpk_data = JPKAnalyze(file_path, None)
        df = jpk_data.df['Force-distance']
        plot_params = jpk_data.ANALYSIS_MODE_DICT['Force-distance']['plot_parameters']
        #find label of fd point
        x_pos = df['X'].loc[0]
        y_pos = df['Y'].loc[0]
        if rotation_info != None:
            x0, y0, scan_angle = rotation_info[0], rotation_info[1], rotation_info[2]
            x_rot = x0 + (x_pos-x0)*np.cos(scan_angle) + (y_pos-y0)*np.sin(scan_angle)
            y_rot = y0 -(x_pos-x0)*np.sin(scan_angle) + (y_pos-y0)*np.cos(scan_angle)
        else:
            x_rot = x_pos
            y_rot = y_pos
        #print(x_pos, y_pos, x_rot, y_rot)
        x_real = df_adh['X'].loc[np.abs(x_array - x_rot).argmin()]
        y_real = df_adh['Y'].loc[np.abs(y_array - y_rot).argmin()]
        x_index = np.where(img_anal.im_df.columns == x_real)[0]
        y_index = np.where(img_anal.im_df.index == y_real)[0]
        #print(y_index,x_index,img_anal.masked[y_index,x_index])
        num_points = len(df.index)
        try:
            label = int(img_anal.masked[y_index,x_index])
            #get adhesion
            force_data = df['Force'].to_numpy()
            distance_data = df['Distance'].to_numpy()
            #d_retract = df['Distance'][int(num_points/2):]
            
            force_data_dict = {'approach': force_data[:int(num_points/2)],
                               'retract': force_data[int(num_points/2):]}
            distance_data_dict = {'approach': distance_data[:int(num_points/2)],
                                  'retract': distance_data[int(num_points/2):]}
            #print(force_data_dict)
            #print(distance_data_dict)
            force_zero = np.average(force_data[:100]) #CHECK RANGE FOR ZERO FORCE
            distance_zero = np.average(distance_data[int(num_points/2)]) #CHECK  DISTANCE OF CONTACT POINT
            #adh_id = np.argmin(force_data)
            #distance_zero = distance_data[adh_id] #CHECK THIS!
            
            adhesion = force_zero - force_data_dict[force_cycle].min()
            
            
            #get FD slope at retract
##            num_points = len(df.index)
#             adh_id = np.argmin(force_data)
#             if type(fit_range).__name__ == 'int': #fit based on adhesion point
#                 fit_slice = slice(adh_id,adh_id+fit_range)
#             elif type(fit_range).__name__ == 'list': #fit based on range percentage
#                 #fit_range = [70,75] 
#                 fit_slice = slice(int(fit_range[0]*num_points/100),
#                                   int(fit_range[1]*num_points/100)-1)
            
            

            #plot FD
            label_text = str(label)
            
            afm_plot.plot_line(df, plot_params, label_text=label_text, color='r')
##            ylim = afm_plot.ax_fd.get_ylim()
##            afm_plot.ax_fd.plot(d_retract,f_fit, label=label_text, color=colors[i])
##            afm_plot.ax_fd.set_ylim(ylim)
            #retraction data to afm_plot for cursor based updating (eg. positions, fitting)
            afm_plot.xAxisData = distance_data_dict[force_cycle]#d_retract.to_numpy()
            afm_plot.yAxisData = force_data_dict[force_cycle]#force_data[int(num_points/2):]
            afm_plot.fd_fit_order = fit_order #CHECK FIT ORDER
        
            afm_plot.plotWidget.wid.updateCursor(afm_plot.plotWidget.wid.cursor1,
                                                 afm_plot.xAxisData.min())
            afm_plot.plotWidget.wid.updateCursor(afm_plot.plotWidget.wid.cursor2,
                                                 afm_plot.xAxisData.max())
            

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
            afm_plot.ax_fd.relim()
            afm_plot.ax_fd.autoscale_view()

            afm_plot.plotWidget.wid.draw_idle()
            #plt.show()
            afm_plot.ax_fd.axhline(y=force_zero, linestyle=':')
            afm_plot.plotWidget.showWindow()

            cursor_index = afm_plot.cursor_index #positions of cursors
            rupture_distance = abs(afm_plot.xAxisData[cursor_index[1]] - afm_plot.xAxisData[cursor_index[0]])
            print('rupture distance', rupture_distance)
            
            #FD fitting
#             force_shifted = [x-force_zero for x in force_data]
#             distance_shifted = df['Distance']-d_retract.iloc[0]
#             fit_slice = slice(int(num_points/2)+cursor_index[0],
#                               int(num_points/2)+cursor_index[1])
#             retract_fit = np.polyfit(distance_shifted[fit_slice],
#                                      force_shifted[fit_slice],afm_plot.fd_fit_order) 
            force_shifted = [x-force_zero for x in force_data_dict[force_cycle]]
            distance_shifted = [x-distance_zero for x in distance_data_dict[force_cycle]]
            fit_slice = slice(cursor_index[0],cursor_index[1])
            retract_fit = np.polyfit(distance_shifted[fit_slice],
                                     force_shifted[fit_slice],afm_plot.fd_fit_order)
#             fit_poly = np.poly1d(retract_fit)
#             afm_plot.ax_fd.autoscale(enable=False)
#             afm_plot.ax_fd.plot(d_retract, fit_poly(d_retract-d_retract.iloc[0])+force_zero, ':') #plot fit
            #print('fitting:', label, retract_fit)            
            #get wetted length
            fd_roots = np.roots(retract_fit)
            fd_roots_filtered = fd_roots[(fd_roots>0)]
            wetted_length = min(fd_roots_filtered)
            print('FD wetted length:', wetted_length)
            
            #get area under curve
#             energy_slice = slice(int(num_points/2)+cursor_index[0],
#                                  int(num_points/2)+cursor_index[1])
            energy_adhesion = integrate.simps(force_shifted[fit_slice],
                                              distance_shifted[fit_slice])
            print('energy',label, energy_adhesion)
            
            afm_plot.ax_fd.fill_between(distance_data_dict[force_cycle][fit_slice],
                                        force_zero,
                                        force_data_dict[force_cycle][fit_slice],
                                        color = 'black', alpha=0.5)
            
            fig_list.append(copy.copy(afm_plot.fig_fd))
            #plt.show(block=True)
            afm_plot.fig_fd.savefig(f'{output_path}/FD_curves_{label_text}.png', bbox_inches = 'tight',
                                    transparent = False)
            afm_plot.CLICK_STATUS = False #done to reinitialize fd plot

#             data_dict[label] = [adhesion, x_rot, y_rot, x_real, y_real,
#                                 file_path, retract_fit[0], wetted_length,
#                                 rupture_distance, energy_adhesion]
            data_dict['Label'].append(label)
            data_dict['Adhesion (FD)'].append(adhesion)
            data_dict['Slope (FD)'].append(retract_fit[0])
            data_dict['Wetted length (FD)'].append(wetted_length)
            data_dict['Rupture distance (FD)'].append(rupture_distance)
            data_dict['Adhesion energy (FD)'].append(energy_adhesion)
            data_dict['FD X position'].append(x_rot)
            data_dict['FD Y position'].append(y_rot)
            data_dict['FD file'].append(file_path)
            
        except Exception as e:
            print(e)
            continue
    
    output_df = pd.DataFrame(data_dict)
        #afm_plot.fig_fd.show()
##    if len(fd_file_paths) != 0:
##        #legend remove duplicates and sort
##        handles, labels = afm_plot.ax_fd.get_legend_handles_labels()
##        handles, labels = zip(*[ (handles[j], labels[j]) for j in sorted(range(len(handles)),
##                                                                         key=lambda k: labels[k],
##                                                                         reverse=True)])
##        leg_dict = dict(zip(labels[::-1],handles[::-1]))
##        afm_plot.ax_fd.get_legend().remove()
##        leg = afm_plot.ax_fd.legend(leg_dict.values(), leg_dict.keys())
##        leg.set_draggable(True, use_blit=True)
##
##        afm_plot.ax_fd.autoscale(enable=True)
##        afm_plot.fig_fd.savefig(f'{output_path}/FD_curves.png', bbox_inches = 'tight',
##                                transparent = False)
    return output_df, fig_list


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
def get_surface_tension2(output_df, simu_df, tolerance, fd_file_paths,
                        file_path, save=False):
    if fd_file_paths != None:
        output_df['yd/F'] = 0.0
        output_df['Surface Tension FD (mN)'] = 0.0
        output_df['Simulation contact angle'] = 0.0  
        output_df['Simulation file'] = ''
        output_df_fd = output_df[output_df['FD file'] != ''] #filter out drops without force data
        for i in output_df_fd.index:#range(len(output_df.index)):
            contact_radius = output_df['R/d'].loc[i]
            rs_nearest = min(simu_df['Contact_Radius'].unique(),
                             key=lambda x:abs(x-contact_radius))
            simu_df_filtered = simu_df[simu_df['Contact_Radius'] == rs_nearest].reset_index()
        ##    simu_df_filtered['ys/F'] = -1/(2*np.pi*simu_df_filtered['Force_Calc']) #inverse

            #linear fit of Force-distance data
            fd_fit_dict = {}
            fa_dict = {} #fmax - angle data
            force_var = 'Force_fit' #Force_Calc,Force_Eng,Force_fit
            for top_angle in simu_df_filtered['Top_Angle'].unique():
                df_temp = simu_df_filtered[simu_df_filtered['Top_Angle'] == top_angle]
                #df_temp = df_temp.query('`Height`<0.1').reindex() #CHECK filtering
                fd_fit = np.polyfit(df_temp['Height'], df_temp[force_var], 1)
                fd_fit_dict[top_angle]= fd_fit[0]
                fa_dict[top_angle]= df_temp[df_temp['Height'] == 0.0][force_var].iloc[0]
            #print(fd_fit_dict, fa_dict)
            angle_slope_fit = np.polyfit(list(map(float,fd_fit_dict.values())),
                                         list(map(float,fd_fit_dict.keys())), 2)
            #print(angle_slope_fit)
            force_angle_fit = np.polyfit(list(map(float,fa_dict.keys())),
                                         list(map(float,fa_dict.values())), 2)
            #print(force_angle_fit)
            tension_guess_orig = np.linspace(70,1,300)
            #tension_guess = tension_guess_orig
            j = 0
            while j<300:
                tension_guess = tension_guess_orig[j]
                slope_expt = output_df['Slope (FD)'].loc[i]/(2*np.pi*tension_guess/1000)
                #slope_expt = 0.042/(2*np.pi*tension_guess/1000)
##                contact_angle = angle_slope_fit[0]*slope_expt + angle_slope_fit[1]
                contact_angle = np.polyval(angle_slope_fit,slope_expt)
#                 contact_angle = 1 if contact_angle < 0 else contact_angle
#                 contact_angle = 90 if contact_angle > 90 else contact_angle
##                contact_angle_nearest = min(simu_df_filtered['Top_Angle'].unique(),
##                                            key=lambda x:abs(x-contact_angle))
##                df_temp2 = simu_df_filtered[simu_df_filtered['Top_Angle'] == contact_angle_nearest]
##                fmax_simul = df_temp2[df_temp2['Height'] == 0.0]['Force_Calc'].iloc[0]
                fmax_simul = np.polyval(force_angle_fit,contact_angle)
                yd_f = -1/(2*np.pi*fmax_simul) #inverse y*d/F
                tension_new = 1000*yd_f*output_df['Adhesion (FD)'].loc[i]/\
                                               output_df['Max Height'].loc[i]
#                 print('tension guess', tension_guess, 'tension new;', tension_new,
#                       'contact angle', contact_angle, 'slope', slope_expt)
                #if abs((tension_new-tension_guess)*100/tension_guess) > tolerance:
                if abs(tension_new-tension_guess) > tolerance:
                    #tension_guess = (tension_new+tension_guess)/2
                    j += 1
                else:
                    break
            #print('here')
            output_df.at[i,'Surface Tension FD (mN)'] = tension_new
            output_df.at[i,'Simulation contact angle'] = contact_angle
            output_df.at[i,'yd/F'] = yd_f
            output_df.at[i,'Simulation file'] = simu_df_filtered['File path'].iloc[0]

        #print(output_df['Surface Tension FD (mN)'])
        
        #save final output
        if save == True:
            afm_filename = output_df['AFM file'].iloc[0].split('/')[-1][:-4]
            output_df.to_excel(f'{file_path}/output_FDiter-{afm_filename}.xlsx', index=None)

    return output_df


def get_surface_tension3(drop_df, simu_df_anal, fixed_contact_angle, fd_file_paths,
                        file_path, save=False):
    
    output_df = drop_df.copy()
    if fd_file_paths != None:
#         output_df['F_fit'] = 0.0
#         output_df['ys/F'] = 0.0
#         output_df['Simulation contact angle'] = 0.0
#         output_df['Simulation file'] = ''
        for i in output_df.index:
##                R = min([3.5,round(output_df['R/s'].loc[i])]) #CHECK 3.5 (limit)
            R = min(simu_df_anal['Contact_Radius'].unique(),
                    key=lambda x:abs(x-output_df['R/d'].loc[i]))
            rupture_distance = output_df['Rupture distance (FD)'].loc[i]
            wetted_length = output_df['Wetted length (FD)'].loc[i]
            s = output_df['s'].loc[i]
            d = output_df['Max Height'].loc[i] #max drop height
            F_adh = output_df['Adhesion (FD)'].loc[i]
            #print(output_df['R/s'].loc[i],R,s)
        ##    R = 2.8 # R/s INPUT
        ##    s = 2.44e-6 # s value (in meters from jumpin analysis INPUT
            #TODO: use interpolated values instead of filtering
            simu_df_filtered = simu_df_anal[simu_df_anal['Contact_Radius'] == R].\
                               sort_values(by=['Rupture_Distance'])
        ##    print(simu_df_filtered['Average Wetted Height'],simu_df_filtered['Top_Angle'])
            #3rd order polynomial fit of rupture distance-Contact angle data
            wa_fit = np.polyfit(simu_df_filtered['Rupture_Distance'],
                                simu_df_filtered['Top_Angle'], 3)
            contact_angle = np.polyval(wa_fit,wetted_length/d) #choose:rupture_distance,wetted_length
            
            
#             ca_nearest = min(simu_df['Top_Angle'].unique(),
#                              key=lambda x:abs(x-contact_angle))
#             print('Contact angle', contact_angle,ca_nearest)
#             simu_df_filtered2 = simu_df[simu_df['Top_Angle'] == ca_nearest].reset_index()
#         ##    simu_df_filtered['ys/F'] = -1/(2*np.pi*simu_df_filtered['Force_Calc']) #inverse

#             #5th order polynomial fit of Force-Contact radius data
#             fr_fit = np.polyfit(simu_df_filtered2['Contact_Radius'],
#                                 simu_df_filtered2['Force_Calc'], 5)
#             F_fit = np.polyval(fr_fit,output_df['R/s'].loc[i])
#             ys_f = -1/(2*np.pi*F_fit) #inverse
#             tension = 1000*ys_f*F_adh/s
            
            #3rd order polynomial fit of adhesion-contact angle data
            fa_fit = np.polyfit(simu_df_filtered['Top_Angle'],
                                simu_df_filtered['Adhesion'], 3)
            F_fit_actual = np.polyval(fa_fit,contact_angle)
            tension_actual = 1000*(-1/(2*np.pi*F_fit_actual))*F_adh/d
            F_fit_fixed = np.polyval(fa_fit,fixed_contact_angle)
            tension_fixed = 1000*(-1/(2*np.pi*F_fit_fixed))*F_adh/d
            
            output_df.at[i,'Simulation R/s'] = R
            output_df.at[i,'Surface Tension (rupture, mN)'] = tension_actual
            output_df.at[i,'Tip contact angle (rupture)'] = contact_angle
            output_df.at[i,'F_fit_actual'] = F_fit_actual
            output_df.at[i,'Surface Tension (fixed, mN)'] = tension_fixed
            output_df.at[i,'Tip contact angle (fixed)'] = fixed_contact_angle            
            output_df.at[i,'F_fit_fixed'] = F_fit_fixed
            output_df.at[i,'Simulation file'] = simu_df_filtered['File path'].iloc[0]
                
    
        #print(output_df['Surface Tension FD (mN)'])
        
    #save final output
    if save == True:
        afm_filename = output_df['AFM file'].iloc[0].split('/')[-1][:-4]
        output_df.to_excel(f'{file_path}/output_rupture-{afm_filename}.xlsx', index=None)

    return output_df

def combine_simul_data(simu_folderpath, fit=False, plot=False):
    simu_df = pd.DataFrame()
    simu_df_anal = pd.DataFrame()
    fd_fit_dict = {}
    fd_fit_order = 2 #(CHECK ORDER!)
    force_var = 'Force_fit' #Force_Calc,Force_Eng,Force_fit
    with os.scandir(simu_folderpath) as folder:
        for file in folder:
            if file.is_file() and file.path.endswith( ('.txt') ):
                df_temp = pd.read_csv(file.path,delimiter='\t')
                
                angle = df_temp['Top_Angle'].iloc[0]
                Rs = df_temp['Contact_Radius'].iloc[0]
                                    
                #3rd order Polynomial fit of ED data to get force by derivative
                ed_fit = np.polyfit(df_temp['Height'], df_temp['Energy'], 3)
                #print('Energy fit', ed_fit)
                df_temp['Force_fit'] = -(3*ed_fit[0]*(df_temp['Height']**2) \
                                         + (2*ed_fit[1]*df_temp['Height']) + \
                                         ed_fit[2])/(2*np.pi)
                               
                
                #calculate max drop height without cantilever based on spherical cap approximation
                #h^3 + (3*R^2*h) - 8 = 0
                sph_cap_eq = [1,0,3*Rs**2,-8]
                sph_roots = np.roots(sph_cap_eq)
                sph_roots_filtered = sph_roots[(sph_roots>0) & (sph_roots<1)]
                h_max = min(sph_roots_filtered).real

                #CHECK USE OF h_max EVERYWHERE!
                df_temp['Height'] = df_temp['Height']/h_max
                df_temp['Drop_height'] = h_max
                df_temp['Contact_Radius'] = df_temp['Contact_Radius']/h_max
                df_temp[force_var] = df_temp[force_var]/h_max
                
                df_temp['ys/F'] = -1/(2*np.pi*df_temp[force_var]) #inverse
                df_temp['File path'] = file.path
                
                
                df_temp = df_temp.query('`Height`<=0.5').reindex() #CHECK filtering
                simu_df = simu_df.append(df_temp)
                if fit == True:

                    adhesion = min(df_temp[force_var])
                    yd_F = -1/(2*np.pi*adhesion) #yd/F
                    #Polynomial fit of FD data
                    fd_fit_dict[angle] = np.polyfit(df_temp['Height'],
                                                    df_temp[force_var], fd_fit_order)
                    #print('FD fit simulation',fd_fit_dict[angle],Rs/h_max,angle)
                    fd_roots = np.roots(fd_fit_dict[angle])
                    fd_roots_filtered = fd_roots[(fd_roots>0)] # & (fd_roots<2)
                    if len(fd_roots_filtered) > 0:
                        rupture_distance = min(fd_roots_filtered).real
                    else:
                        print(f'No ROOTS FOUND FOR angle={angle}, Rs={Rs/h_max}')
                        pass
                    
                    #intercept of fd slope at d=0
                    fd_fit_der = np.polyder(fd_fit_dict[angle])
                    slope_intercept = abs(np.polyval(fd_fit_dict[angle],0)/np.polyval(fd_fit_der,0))
                    print(Rs/h_max, slope_intercept, angle)

                    #print(Rs, angle, rupture_distance) #CHECK!
                    simu_df_anal_temp = pd.DataFrame({'Contact_Radius':[Rs/h_max],
                                                      'Drop_height':[h_max],
                                                      'Top_Angle':[angle],
                                                      'Adhesion': [adhesion],
                                                      'yd/F':[yd_F],
                                                      'Rupture_Distance':[rupture_distance],
                                                     'File path': [file.path]})
                    simu_df_anal = simu_df_anal.append(simu_df_anal_temp)
                
    
    if not simu_df.empty:
        simu_df['Average tip angle'] = (simu_df['Top_Angle6']+simu_df['Top_Angle7']+
                                        simu_df['Top_Angle8']+simu_df['Top_Angle9'])/4
        simu_df['Simulation folder'] = simu_folderpath        
    
    if plot == True and not simu_df.empty:
        Rs = simu_df['Contact_Radius'].iloc[0]
        fig = simul_plot(simu_df,
                         x_var='Height',
                         y_var=force_var,#force_var,Average Wetted Height,Average tip angle,Energy
                         hue_var='Top_Angle',
                         title=f'Simulation data (FD): R/d={Rs:.2f}',
                         xlabel='Height, h/d',
                         ylabel=r'$F/2\pi \gamma d$',
                         leglabel='Contact angle',
                         fit_order=fd_fit_order)
#     elif plot_type == 'FR':
#         fig = simul_plot1(simu_df)
    else:
        fig = None

    return simu_df, simu_df_anal, fig

#combine data from subfolders
def combine_simul_dirs(simu_folderpath, plot=False):
    simu_df = pd.DataFrame()
    simu_df_anal = pd.DataFrame()
    #plot_type = 'FD' if plot == True else None
    fig_list = []
    with os.scandir(simu_folderpath) as folder:
        for fdr in folder:
            if fdr.is_dir():
                df_temp, simu_df_anal_temp, fig = combine_simul_data(fdr.path,
                                                                fit=True,
                                                                plot=plot)
                #simul_plot2(df_temp)
                simu_df = simu_df.append(df_temp)
                simu_df_anal = simu_df_anal.append(simu_df_anal_temp)
                fig_list.append(fig)
    #simu_df.to_excel('simu_out.xlsx')

    if plot == True:
        simu_df_anal = simu_df_anal.query('`Top_Angle`>30 & `Contact_Radius`<7') #CHECK filtering
        fig1 = simul_plot(simu_df_anal,
                          x_var='Rupture_Distance',
                          y_var='Top_Angle',
                          hue_var='Contact_Radius',
                          title='Simulation data: rupture distance',
                          xlabel='Rupture distance, r/d', 
                          ylabel='Contact angle', 
                          leglabel='Drop size, R/d',
                          fit_order=3)
        fig_list.append(fig1)
        
        fig2 = simul_plot(simu_df_anal,
                          x_var='Top_Angle',
                          y_var='yd/F',#yd/F,Adhesion
                          hue_var='Contact_Radius',
                          title='Simulation data: surface tension',
                          xlabel='Contact angle', 
                          ylabel=r'$\gamma d/F$', 
                          leglabel='Drop size, R/d',
                          fit_order=3)
        fig_list.append(fig2)

    return simu_df, simu_df_anal, fig_list

    
def combine_fd(file_paths, zero_shift=False, output_dir=None, save=False):
    afm_plot = AFMPlot()
    mode = 'Force-distance'
    #colors
    evenly_spaced_interval = np.linspace(0, 1, len(file_paths))
    colors = [cm.rainbow(x) for x in evenly_spaced_interval]
    for i,file_path in enumerate(file_paths):
        fd_data = JPKAnalyze(file_path, None)
        plot_params = fd_data.ANALYSIS_MODE_DICT[mode]['plot_parameters']
        if zero_shift == True:
            num_pts = len(fd_data.df[mode][plot_params['y']])
            force_zero = fd_data.df[mode][plot_params['y']].iloc[0]
            distance_zero = fd_data.df[mode][plot_params['x']].iloc[int(num_pts/2)]
            fd_data.df[mode][plot_params['y']] -= force_zero
            fd_data.df[mode][plot_params['x']] -= distance_zero
        label_text = file_path.split('/')[-1].split('-')[3] #CHANGE INDEX   
        afm_plot.plot_line(fd_data.df[mode], plot_params, label_text,
                          color=colors[i])
    
    afm_plot.plotWidget.wid.cursor1.remove()
    afm_plot.plotWidget.wid.cursor2.remove()
    
    #legend remove duplicates
    handles, labels = afm_plot.ax_fd.get_legend_handles_labels()            
    leg_dict = dict(zip(labels[::-1],handles[::-1]))
    afm_plot.ax_fd.get_legend().remove()
    leg = afm_plot.ax_fd.legend(leg_dict.values(), leg_dict.keys())
    leg.set_draggable(True, use_blit=True)
    
    afm_plot.ax_fd.autoscale(enable=True)
    afm_plot.ax_fd.relim()
    afm_plot.ax_fd.autoscale_view()

    afm_plot.plotWidget.wid.draw_idle()
            
    if save==True:
        afm_plot.fig_fd.savefig(f'{output_dir}/FD_curves.png', bbox_inches = 'tight',
                                transparent = True)
    
    #afm_plot.fig_fd.show()
    afm_plot.plotWidget.showWindow()
        
#jpk_data.data_zip.close()
def combine_result_spreadsheets(folder_paths):
    summary_df = pd.DataFrame()
    for f_p in folder_paths:
        with os.scandir(f_p) as folder:
            for file in folder:
                if file.is_file() and file.path.endswith( ('.xlsx') ):
                    df_temp = pd.read_excel(file.path)
                    df_temp['Folder name'] = os.path.basename(f_p)
                    df_temp['File path'] = file.path
                    summary_df = summary_df.append(df_temp)
    return summary_df
