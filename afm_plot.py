import matplotlib
import copy
import numpy as np
#import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.graph_objects as go

from afm_analyze import JPKAnalyze
from plotly_viewer import PlotlyViewer

class AFMPlot:
    
    def __init__(self, jpk_anal):
        self.plotwin = None
        #mapping plot types (from ANALYSIS_MODE_DICT) to functions
        PLOT_DICT = {'2d': self.plot_2d,
                     '3d': self.plot_3d,
                     'line': self.plot_line}
        self.CLICK_STATUS = False #True when clicked first and plot doesn't exist

        self.jpk_anal = jpk_anal
        self.file_path = jpk_anal.file_path
        #plot data
        for mode in jpk_anal.df.keys():
            plot_params =  jpk_anal.anal_dict[mode]['plot_parameters']
            for plot_type in plot_params['type']:
                PLOT_DICT[plot_type](jpk_anal.df[mode], plot_params)

        plt.show(block=False)
##        plt.pause(0.05)
##        if self.plotwin != None:
####            self.plotwin.show()
##            self.plotwin.app.exec_()
    
    def plot_2d(self, df, plot_params):
        x = plot_params['x']
        y = plot_params['y']
        z = plot_params['z']
        title = plot_params['title']
        #organize data into matrix for heatmap plot
        df_data = df.pivot_table(values=z, index=y, columns=x,
                                 aggfunc='first')
        df_reference = df.pivot_table(values='Segment folder', index=y,
                                      columns=x, aggfunc='first')
        #plot
        fig2d = plt.figure(f'{title}')
        ax2d = fig2d.add_subplot(111)
        im2d = ax2d.pcolormesh(df_data.columns, df_data.index,
                               df_data, cmap='afmhot')
        ax2d.ticklabel_format(style='sci', scilimits=(0,0))
        ax2d.set_xlabel(x)
        ax2d.set_ylabel(y)
        fig2d.colorbar(im2d, ax=ax2d, label=z)
        fig2d.suptitle(title)

        canvas = fig2d.canvas

##        rgb_data = np.fromstring(canvas.tostring_rgb(),
##                                 dtype=np.uint8, sep='')
##        rgb_data = rgb_data.reshape(canvas.get_width_height()[::-1] + (3,))

        
        self.cid = fig2d.canvas.mpl_connect('button_press_event',
                                       lambda event: self.on_click(event,
                                                                   df_reference))
        #BUG: program doesn't end after callbacks
        fig2d.canvas.mpl_connect('close_event',
                                 lambda event: self.on_figclose(event, fig2d))


    def plot_3d(self, df, plot_params):
        x = plot_params['x']
        y = plot_params['y']
        z = plot_params['z']
        #organize data into matrix for 3d plot
        df_data = df.pivot_table(values=z, index=y, columns=x,
                                 aggfunc='first')
        df_reference = df.pivot_table(values='Segment folder', index=y,
                                      columns=x, aggfunc='first')
        
        #plot
        fig = go.Figure(data=[go.Surface(z=df_data,
                                         x=df_data.columns,
                                         y=df_data.index)])
        fig.update_layout(title=plot_params['title'],
                          scene = dict(xaxis_title=x,
                                       yaxis_title=y,
                                       zaxis_title=z),
                          autosize=True)
        self.plotwin  = PlotlyViewer(fig)

        
##    def plot_3d(self, df, plot_params):
##        x = plot_params['x']
##        y = plot_params['y']
##        z = plot_params['z']
##
##        self.fig3d = plt.figure('3D Surface')
##        self.ax3d = self.fig3d.gca(projection='3d')
##        surf = self.ax3d.plot_trisurf(df[x], df[y], df[z])#, cmap='afmhot')
##        self.ax3d.ticklabel_format(style='sci', scilimits=(0,0))
##        self.ax3d.set_xlabel(x)
##        self.ax3d.set_ylabel(y)
##        self.ax3d.set_zlabel(z)
##        self.fig3d.suptitle(plot_params['title'])
##        ##fig3d.colorbar(surf, shrink=0.5, aspect=5)

    def plot_line(self, df, plot_params, label_text=None):
        x = plot_params['x']
        y = plot_params['y']
        style = plot_params['style']
        
        if self.CLICK_STATUS == False:
            self.init_fd_plot()
            self.CLICK_STATUS = True

        sns.lineplot(x=x, y=y, style=style,
                     data=df, ax=self.ax_fd,
                     label = label_text, sort=False)
##        self.ax_fd.plot(df[x], df[y])
        self.ax_fd.ticklabel_format(style='sci', scilimits=(0,0))
        self.ax_fd.set_xlabel(x)
        self.ax_fd.set_ylabel(y)
        self.fig_fd.suptitle(plot_params['title'])
##        df.to_excel('test-fd-data.xlsx')

    def init_fd_plot(self): #initialize force-distance plot
        sns.set_theme(palette = 'Set2')
        self.fig_fd = plt.figure('Line plot')
        self.ax_fd = self.fig_fd.add_subplot(111)
        self.fig_fd.canvas.mpl_connect('close_event',
                                       lambda event: self.on_close(event))

    def on_close(self, event):
        self.CLICK_STATUS = False

    def on_figclose(self, event, fig):
        fig.canvas.mpl_disconnect(self.cid)
##        self.jpk_anal.data_zip.close() #CHECK THIS
        
    def on_click(self, event, df_ref):
        print('click')
        x, y = event.xdata, event.ydata
        if x != None and y != None:
            #get segment path corresponding to clicked position
            col = sorted([[abs(a - x), a] for a in df_ref.columns],
                         key=lambda l:l[0])[0][1]
            ind = sorted([[abs(a - y), a] for a in df_ref.index],
                         key=lambda l:l[0])[0][1]
            segment_path = df_ref[col][ind]
            print(segment_path, col, ind)

            #TODO: clean and organize this up
            mode = 'Force-distance'
            fd_data = self.jpk_anal
            segment_path_old = fd_data.segment_path
##            anal_dict_old = fd_data.anal_dict.copy()
            fd_data.clear_output(mode) #clear previous data
            fd_data.segment_path = segment_path
##            fd_data.anal_dict = fd_data.ANALYSIS_MODE_DICT[mode].copy()
##            print('old')
##            print(mode_dict_old)
            fd_data.get_data([mode])
            
            #fd_data = JPKAnalyze(self.file_path, mode, segment_path)
            plot_params = fd_data.ANALYSIS_MODE_DICT[mode]['plot_parameters']

            label_text = f'x={"{:.2e}".format(col)}, y={"{:.2e}".format(ind)}'
            self.plot_line(fd_data.df[mode], plot_params, label_text)

            #legend remove duplicates
            handles, labels = self.ax_fd.get_legend_handles_labels()            
            leg_dict = dict(zip(labels[::-1],handles[::-1]))
            self.ax_fd.get_legend().remove()
            leg = self.ax_fd.legend(leg_dict.values(), leg_dict.keys())
            leg.set_draggable(True, use_blit=True)

            #CHECK
            fd_data.segment_path = None
##            fd_data.anal_dict = anal_dict_old

            self.fig_fd.show()

    def plot_2dfit(self, fit_data, df_raw, plot_params):
        x = plot_params['x']
        y = plot_params['y']
        z = plot_params['z']
        z_raw = f'{z}_raw'
        z_fit = f'{z}_fit'

##        color_limits = [df[z_raw].min(), df[z_raw].max()]
##        title_text = ', '.join([f'{k}' + '={:.1e}'.format(v) \
##                                for k, v in fit_output.items()])
        title_text = 'Fitting result'

        #data reshape
##        df_raw = df.pivot_table(values=z_raw, index=y, columns=x,
##                                       aggfunc='first')
##        df_filtered = df.query(f'{z_fit}>={color_limits[0]}')
##        df_fit = fit_data.pivot_table(values=z_fit, index=y,
##                                           columns=x, aggfunc='first')

        #plot
        fig = go.Figure()
        fig.add_trace(go.Surface(legendgroup="Raw",
                                 name='Raw',
                                 z=df_raw,
                                 x=df_raw.columns,
                                 y=df_raw.index,
                                 opacity=0.7,
                                 colorscale ='Greens',
                                 reversescale=True,
                                 showlegend=True,
                                 showscale=False))
        i = 0
        for key, val in fit_data.items():
##            print(key)
            leg = True if i==0 else False
            fig.add_trace(go.Surface(legendgroup="Fit",
                                     name='Fit',
                                     z=val[z],#df_fit,
                                     x=val[x],#df_fit.columns,
                                     y=val[y],#df_fit.index,
                                     opacity=0.7,
                                     colorscale ='Reds',
                                     reversescale=True,
                                     showlegend=leg,
                                     showscale=False))
            i = 1
##        fig.update_traces(contours_z=dict(show=True, usecolormap=False,
##                                          highlightcolor="limegreen",
##                                          project_z=True))
        z_range = [df_raw.min().min(),1.2*df_raw.max().max()]
##        print(z_range)
        fig.update_layout(title=title_text,
                          scene = dict(xaxis_title=x,
                                       yaxis_title=y,
                                       zaxis_title=z,
                                       xaxis=dict(ticksuffix='m'),
                                       yaxis=dict(ticksuffix='m'),                                       
                                       zaxis=dict(range=z_range,
                                                  ticksuffix='m'),
                                       aspectratio={"x": 1, "y": 1, "z": 0.4}
                                       ),
                          autosize=True)
        self.plotwin  = PlotlyViewer(fig)

def simul_plot(simu_df):
    sns.set_context("talk")
    sns.set_style("ticks")
    fig = plt.figure('Simulation data')
    
    ax1 = fig.add_subplot(1,2,1)
    mk_num = len(simu_df['Top_Angle'].unique())
    sns.lineplot(x='Contact_Radius',y='Force_Calc',hue='Top_Angle',
                 style='Top_Angle',data=simu_df,
                 markers=['o']*mk_num,dashes=False,
                 legend='full',palette='flare', ax=ax1)
    ax1.axhline(y=0, color='0.8', dashes=(1, 1), zorder=0)
    
    ax1.set_title('Adhesion force')
    ax1.set_xlabel('Drop size, R/s')
    ax1.set_ylabel(r'$F/2\pi \gamma s$')
    leg = ax1.get_legend()
    leg.remove()
    
    ax2 = fig.add_subplot(1,2,2)
    mk_num = len(simu_df['Top_Angle'].unique())
    sns.lineplot(x='Contact_Radius',
                 y='Average Wetted Height',hue='Top_Angle',
                 style='Top_Angle',data=simu_df,
                 markers=['o']*mk_num,dashes=False,
                 legend='full',palette='flare', ax=ax2)

    ax2.set_title('Wetted length')
    ax2.set_xlabel('Drop size, R/s')
    ax2.set_ylabel('w/s')
    leg = ax2.get_legend()
    leg.set_title('Contact angle')
    
    plt.show(block=False)
        

        
##    def plot_2dfit(self, df, plot_params, fit_output):
##        x = plot_params['x']
##        y = plot_params['y']
##        z = plot_params['z']
##        z_raw = f'{z}_raw'
##        z_fit = f'{z}_fit'
##
##        color_limits = [df[z_raw].min(), df[z_raw].max()]
##
##        #fit data reshape
##        df_fit = df.pivot_table(values=z_fit, index=y, columns=x,
##                                aggfunc='first')
##        #plot
##        fig = plt.figure('Fit: 2D map')
##        ax_fit = fig.add_subplot(111)
##        im_fit = ax_fit.pcolormesh(df_fit.columns, df_fit.index,
##                                   df_fit, cmap='afmhot',
##                                   vmin=color_limits[0],
##                                   vmax=color_limits[1])
##        ax_fit.ticklabel_format(style='sci', scilimits=(0,0))
##        ax_fit.set_xlabel(x)
##        ax_fit.set_ylabel(y)
##        title_text = ', '.join([f'{k}' + '={:.1e}'.format(v) \
##                                for k, v in fit_output.items()])
##        ax_fit.set_title(title_text)
##        fig.colorbar(im_fit, ax=ax_fit, label=z)
##        
##        fig.tight_layout()
##
##        #3d plot
##        z_lim = self.ax3d.get_zlim3d() #z limits of raw 3D
##        df_filtered = df.query(f'{z_fit}>={z_lim[0]}')
##        fig3d = plt.figure('Fit: 3D')
##        ax3d = fig3d.gca(projection='3d')
##        surf = ax3d.plot_trisurf(df_filtered[x],df_filtered[y],
##                                 df_filtered[z_fit])#, cmap='afmhot')
##        ax3d.ticklabel_format(style='sci', scilimits=(0,0))
##        ax3d.set_xlabel(x)
##        ax3d.set_ylabel(y)
##        ax3d.set_zlabel(z)
##        fig3d.suptitle(title_text)        
##        ax3d.set_zlim3d(color_limits)
##        ax3d.set_xlim((df[x].min(), df[x].max()))
##        ax3d.set_ylim(df[y].min(), df[y].max())
##        
##        plt.show(block=False)
##    
