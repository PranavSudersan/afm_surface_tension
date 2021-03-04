# include functions to analyze and get output data from afm data

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans
import statistics

from afm_read import JPKRead

class JPKAnalyze(JPKRead):

    #mapping of keys in header files which depend on file format
    FORMAT_KEY_DICT = {'jpk-qi-data': {'position': 'position'},
                       'jpk-force': {'position': 'start-position'}}
    def __init__(self, file_path, mode, segment_path):
        #Make sure variable keys in result dict of 'function' definition
        #is same as 'output' keys of below
        self.ANALYSIS_MODE_DICT = {'Adhesion': {'function': self.get_adhesion,
                                                'output': {'Adhesion': [],'X': [], 'Y':[],
                                                           'Segment folder': []},
                                                'plot_parameters': {'x': 'X',
                                                                    'y': 'Y',
                                                                    'z': 'Adhesion',
                                                                    'title': 'Adhesion',
                                                                    'type': ['2d', '3d']}
                                                },
                                   'Snap-in distance': {'function': self.get_snapin_distance,
                                                        'output': {'Height': [],
                                                                   'X': [], 'Y':[],
                                                                   'Segment folder': []},
                                                        'plot_parameters': {'x': 'X',
                                                                            'y': 'Y',
                                                                            'z': 'Height',
                                                                            'title': 'Snap-in distance',
                                                                            'type': ['2d']}
                                                        },
                                   'Force-distance': {'function': self.get_force_distance,
                                                      'output': {'Force': [],
                                                                 'Distance': [],
                                                                 'Segment': [],
                                                                 'Segment folder': []},
                                                      'plot_parameters': {'x': 'Distance',
                                                                          'y': 'Force',
                                                                          'style': 'Segment',
                                                                          'title': 'Force-distance curve',
                                                                          'type': ['line']
                                                                          }
                                                      }
                                   }
        #initialize JPKRead and get data
        super().__init__(file_path, self.ANALYSIS_MODE_DICT[mode],
                         segment_path)

    #clear output data in ANALYSIS_MODE_DICT
    def clear_output(self, mode):
        for key in self.ANALYSIS_MODE_DICT[mode]['output'].keys():
            self.ANALYSIS_MODE_DICT[mode]['output'][key] = []
    
    def get_adhesion(self, dirpath, *args, **kwargs):    
        retract_dir = f'{dirpath}1'  #retract folder  
        segment_header_path = f'{retract_dir}/segment-header.properties'
        segment_header = self.data_zip.read(segment_header_path).decode().split('\n')
        segment_header_dict = self.parse_header_file(segment_header)
        #get retract force data
        force_data = self.decode_data('vDeflection', segment_header_dict,
                                      retract_dir)
        #calculate adhesion
        adhesion = force_data[-1] - force_data.min()

        #get position
        x_pos, y_pos = self.get_xypos(segment_header_dict)

        #USE SAME KEYS AS ANALYSIS_MODE_DICT!
        result_dict = {'Adhesion': adhesion,
                       'X': x_pos, 'Y':y_pos,
                       'Segment folder': dirpath} 
        return result_dict

    def get_snapin_distance(self, dirpath, *args, **kwargs):    
        extend_dir = f'{dirpath}0'  #extend folder
        #get segment header file
        segment_header_path = f'{extend_dir}/segment-header.properties'
        segment_header = self.data_zip.read(segment_header_path).decode().split('\n')
        segment_header_dict = self.parse_header_file(segment_header)
        #get extend force data
        force_data = self.decode_data('vDeflection', segment_header_dict,
                                      extend_dir)
        #get extend height data
        height_data = self.decode_data('measuredHeight', segment_header_dict,
                                       extend_dir)

        #calculate snapin distance
        idx_min = np.argmin(force_data) #minimum force during extend
        #tolerance method to get jumpin distance
##        zero_points = 10
##        zero_force = statistics.median(force_data[:zero_points])
##        zero_dev = statistics.stdev(force_data[:zero_points])
##        tol = 100 #deviation from zero
##        for idx, a in enumerate(force_data): 
##            if abs(a-zero_force) > tol*zero_dev:
##                idx_min = idx
##                #idx_min = force_data.index(a)
##                #idx, = np.where(force_data == a)
##                #print(idx)
##                #idx_min = idx[0]
##                break
##            else:
##                idx_min = len(force_data)-1
        #print(idx_min)
        snapin_distance = height_data[idx_min] - height_data[-1]
        #TODO: define as fraction of extend distance range
##        total_distance = height_data[0] - height_data[-1]
##        tolerence = 0.8
##        snapin_distance = 0 if snapin_distance >= tolerence * total_distance \
##                          else snapin_distance

        #get position
        x_pos, y_pos = self.get_xypos(segment_header_dict)
        
        adhesion = force_data[-1] - force_data.min()
        #USE SAME KEYS AS ANALYSIS_MODE_DICT!
        result_dict = {'Height': snapin_distance,
                       'X': x_pos, 'Y':y_pos,
                       'Segment folder': dirpath}
        return result_dict

    def get_force_distance(self, dirpath, *args, **kwargs):
        #USE SAME KEYS AS ANALYSIS_MODE_DICT
        result_dict = {'Force': [], 'Distance': [], 'Segment': [],
                       'Segment folder': []}
        #TODO: get segment info from header files
        segment_name = {'0': 'extend', '1': 'retract'}
        for seg_num, seg_name in segment_name.items():        
            #get segment header file
            segment_dir = f'{dirpath}{seg_num}'  #extend folder
            segment_header_path = f'{segment_dir}/segment-header.properties'
            segment_header = self.data_zip.read(segment_header_path).decode().split('\n')
            segment_header_dict = self.parse_header_file(segment_header)
            #get segment force data
            force_data = self.decode_data('vDeflection', segment_header_dict,
                                          segment_dir)
            #get segment height data
            height_data = self.decode_data('measuredHeight', segment_header_dict,
                                           segment_dir)

            result_dict['Force'] = np.append(result_dict['Force'],force_data)
            result_dict['Distance'] = np.append(result_dict['Distance'],height_data)            
            len_data = len(force_data)
            result_dict['Segment'] = np.append(result_dict['Segment'],
                                               len_data * [seg_name])
            result_dict['Segment folder'] = np.append(result_dict['Segment folder'],
                                                      len_data * [dirpath])

        return result_dict

    def get_xypos(self, segment_header_dict): #get xy position    
        pos_dict = segment_header_dict['force-segment-header']\
                   ['environment']['xy-scanner-position-map']['xy-scanner']\
                   ['tip-scanner'][self.FORMAT_KEY_DICT[self.file_format]['position']]
        return float(pos_dict['x']), float(pos_dict['y'])
    

class DataFit:
    def __init__(self, jpk_anal, afm_plot, func, filter_string,
                 guess=None, bounds=(-np.inf, np.inf)):
        FIT_DICT = {'Sphere-RC': {'function': self.sphere_rc,
                                  'params': 'R,c'
                                  }
                    } #'func' arg keys and params
        df_filt = jpk_anal.df.query(filter_string)
        plot_params =  jpk_anal.anal_dict['plot_parameters']
        x = plot_params['x']
        y = plot_params['y']
        z = plot_params['z']
        data = np.array([[df_filt[x][i],
                          df_filt[y][i],
                          df_filt[z][i]] for i in df_filt.index])

        #fit
        popt, _ = curve_fit(FIT_DICT[func]['function'], data, data[:,2],
                            p0=guess, bounds=bounds)

        self.fit_output = dict(zip(FIT_DICT[func]['params'].split(','), popt))
        print(f'Fit output {func}:', self.fit_output)
        #get fitted data
        data_full = np.array([[jpk_anal.df[x][i],
                               jpk_anal.df[y][i],
                               jpk_anal.df[z][i]] for i in jpk_anal.df.index])

        z_fit = FIT_DICT[func]['function'](data_full, *popt)
        self.fit_data = pd.DataFrame({x: data_full[:,0],
                                      y: data_full[:,1],
                                      f'{z}_raw': data_full[:,2],
                                      f'{z}_fit': z_fit})
        self.z_max = z_fit.max() #maximum fitted z
        #plot
        afm_plot.plot_2dfit(self.fit_data, plot_params, self.fit_output)

    def sphere_rc(self, X, R, C): #sphere function (only R and C)
        i, j, k = np.argmax(X, axis=0)
        a, b = X[k, 0], X[k, 1]
        x, y, z = X.T
        return C + ((R**2)-((x-a)**2)-((y-b)**2))**0.5

#analyze processed data from JPKRead
class DataAnalyze:
    def __init__(self, jpk_anal):
        self.plot_params =  jpk_anal.anal_dict['plot_parameters']        
        self.df = jpk_anal.df.copy()

    #K-means cluster Z data
    def get_kmeans(self, n_clusters=2):
        kmeans = KMeans(n_clusters=2)
        data = np.array(self.df[self.plot_params['z']]).reshape(-1,1)
        k_fit = kmeans.fit(data)
        return k_fit.cluster_centers_.flatten()

    #TODO: calculate volume of smoothed surface
    #calculate volume    
    def get_volume(self, zero=0):
        
        x = self.plot_params['x']
        y = self.plot_params['y']
        z = self.plot_params['z']

        #organize data into matrix for heatmap plot
        df_filter = self.df.query(f'{z}>=1e-8')#remove zeros
        
##        df_matrix = df_filter.pivot_table(values=z,
##                                          index=y, columns=x,
##                                          aggfunc='first')
##        
##
##        nx, ny = len(self.df[x].unique()), len(self.df[y].unique()) #grid size
##        grid_x, grid_y = np.mgrid[self.df[x].min():self.df[x].max():complex(0,nx),
##                                  self.df[y].min():self.df[y].max():complex(0,ny)]
##        
##        
##        #interpolate
##        points = df_filter[[x,y]].to_numpy()
##        values = df_filter[z]        
##        grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')
##
##        import matplotlib.pyplot as plt
##        Z2 = ndimage.gaussian_filter(df_matrix.to_numpy(), sigma=1.0, order=0)
##        plt.subplot(121)
##        plt.imshow(df_matrix.to_numpy(), origin='lower')
##        plt.plot(points[:,0], points[:,1], 'k.', ms=1)
##        plt.title('Original')
##        plt.subplot(122)
##        plt.imshow(Z2, origin='lower')
##        plt.title('Nearest')
##        plt.show()

        
        #organize data into matrix for heatmap plot
        df_matrix = df_filter.pivot_table(values=z,
                                          index=y, columns=x,
                                          aggfunc='first')
        df_shifted = df_matrix-zero
        df_shifted.fillna(0, inplace=True)
        vol = np.trapz(np.trapz(df_shifted-zero,
                                df_shifted.columns),
                       df_shifted.index)
        return vol

    #get volume and contact angle of fitted cap
    #reference: https://mathworld.wolfram.com/SphericalCap.html
    def get_cap_prop(self, R, z_max, zero):
        h = z_max - zero #cap height
        vol = (1/3)*np.pi*((3*R)-h)*(h**2)
        angle = 90 - (np.arcsin((R-h)/R)*180/np.pi)
        return h, vol, angle


    def get_max_height(self, zero):
        return self.df[self.plot_params['z']].max()-zero

    def get_contact_radius(self, fit_out, zero):
        return (fit_out['R']**2 - (zero-fit_out['c'])**2)**0.5
    
