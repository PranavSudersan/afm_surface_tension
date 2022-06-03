import seaborn as sns
import pandas as pd
import numpy as np
import os

def combine_simul_data(simu_folderpath, fit=False, fd_fit_order = 2, plot=False):
    simu_df = pd.DataFrame()
    simu_df_anal = pd.DataFrame()
    fd_fit_dict = {}
    #fd_fit_order = 2 #(CHECK ORDER!)
    force_var = 'Force_fit' #Force_Calc,Force_Eng,Force_fit
    with os.scandir(simu_folderpath) as folder:
        for file in folder:
            if file.is_file() and file.path.endswith( ('.txt') ):
                df_temp = pd.read_csv(file.path,delimiter='\t')
                
                angle = df_temp['Top_Angle'].iloc[0]
                Rs = df_temp['Contact_Radius'].iloc[0]
                                    
                #higher order Polynomial fit of ED data to get force by derivative
                ed_fit = np.polyfit(df_temp['Height'], df_temp['Energy'], fd_fit_order+1)
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
                    #print(Rs/h_max, slope_intercept, angle)

                    #print(Rs, angle, rupture_distance) #CHECK!
                    simu_df_anal_temp = pd.DataFrame({'Contact_Radius':[Rs/h_max],
                                                      'Drop_height':[h_max],
                                                      'Top_Angle':[angle],
                                                      'Adhesion': [adhesion],
                                                      'yd/F':[yd_F],
                                                      'Rupture_Distance':[rupture_distance],
                                                      'Fit params': [fd_fit_dict[angle]],
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
def combine_simul_dirs(simu_folderpath, fd_fit_order=2, plot=False):
    simu_df = pd.DataFrame()
    simu_df_anal = pd.DataFrame()
    #plot_type = 'FD' if plot == True else None
    fig_list = []
    with os.scandir(simu_folderpath) as folder:
        for fdr in folder:
            if fdr.is_dir():
                df_temp, simu_df_anal_temp, fig = combine_simul_data(fdr.path,
                                                                     fit=True,
                                                                     fd_fit_order=fd_fit_order,
                                                                     plot=plot)
                #simul_plot2(df_temp)
                simu_df = simu_df.append(df_temp)
                simu_df_anal = simu_df_anal.append(simu_df_anal_temp)
                fig_list.append(fig)
    #simu_df.to_excel('simu_out.xlsx')

    if plot == True:
        #simu_df_anal = simu_df_anal.query('`Top_Angle`>30 & `Contact_Radius`<7') #CHECK filtering
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


def simul_plot(simu_df, x_var, y_var, hue_var, title, xlabel, ylabel, leglabel, fit_order=None):
    sns.set_context("talk")
    sns.set_style("ticks")

    g = sns.lmplot(x=x_var,y=y_var,hue=hue_var,
                   data=simu_df,
                   legend='full',palette='flare',
                   order=fit_order, ci=None,
                   height=8, aspect=1.3)
    ax1 = g.ax
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    leg = g.legend
    leg.set_title(leglabel)
    for t in leg.texts:
    # truncate label text to 4 characters
        t.set_text(t.get_text()[:4])
    fig = g.fig
    #plt.show(block=True)

    return fig