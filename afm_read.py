import numpy as np
import pandas as pd
import struct
import zipfile
import tifffile as tiff
# import time

class JPKRead:        

    DATA_TYPES = {'short':(2,'h'),'short-data':(2,'h'), 'unsignedshort':(2,'H'),
                  'integer-data':(4,'i'), 'signedinteger':(4,'i'),
                  'float-data':(4,'f')}
    #anal_dict: ANALYSIS_MODE_DICT[mode] dictionary reference
    def __init__(self, file_path, anal_dict, segment_path):
        self.file_path = file_path
        self.anal_dict = anal_dict
        self.segment_path = segment_path
        self.df = {}
        self.file_format = self.file_path.split('.')[-1]
        if self.file_format == 'jpk-qi-data':
            modes = ['Adhesion', 'Snap-in distance']
        elif self.file_format == 'jpk-force':
            modes = ['Force-distance']
        elif self.file_format == 'jpk':
            modes = ['Height (measured)']   
        if self.file_format == 'jpk':
            self.jpk_data_dict = self.read_jpk(self.file_path)
            for mode in modes:
                result = self.anal_dict[mode]['function'](channel=mode,
                                                          trace=self.anal_dict[mode]['misc']['trace'],
                                                          calibration=self.anal_dict[mode]['misc']['calibration']) 
#                 result = self.anal_dict[mode]['function'](dirpath=self.segment_path)
                for key, value in result.items():
                    output = self.anal_dict[mode]['output']
                    output[key] = np.append(output[key], value)
                self.df[mode] = pd.DataFrame(self.anal_dict[mode]['output'])
#             self.get_height_measured(file_path, modes)
        else:
            self.data_zip = zipfile.ZipFile(self.file_path, 'r')
            self.get_data(modes)

        
    #read jpk TIFF file and return data matrixes (check JPK TIFF specification for hex code details)
    def read_jpk(self, filepath):
        with tiff.TiffFile(filepath) as tif:
            data_dict = {}
            for page in tif.pages:        
                tag_dict = {}
                for tag in page.tags:

                    tag_formatted = hex(tag.code) if len(tag.name)==5 else tag.name
        #             print(tag_formatted,tag.name,tag.value)
                    tag_dict[tag_formatted] = tag.value

                image_array = page.asarray()

                if tag_dict[hex(0x8050)] == 'thumbnail':
                    feedback_prop = tag_dict[hex(0x803E)]
                    feedback_dict = {}
                    for k in feedback_prop.splitlines():
                        k_split = k.split(':')
                        feedback_dict[k_split[0].strip()] = k_split[1].strip()
        #             data_dict['Feedback properties'] = feedback_dict

                    x0 = tag_dict[hex(0x8040)]
                    y0 = tag_dict[hex(0x8041)]
                    x_len = tag_dict[hex(0x8042)]
                    y_len = tag_dict[hex(0x8043)]
                    scan_angle = tag_dict[hex(0x8044)]
                    grid_reflect = tag_dict[hex(0x8045)]
                    x_num = tag_dict[hex(0x8046)]
                    y_num = tag_dict[hex(0x8047)]            

                    data_dict['header'] = {'Feedback_Mode': tag_dict[hex(0x8030)],
                                           'Grid-x0':x0,
                                           'Grid-y0':y0,
                                           'Grid-uLength':x_len,
                                           'Grid-vLength':y_len,
                                           'Grid-Theta':scan_angle,
                                           'Grid-Reflect':grid_reflect,
                                           'Grid-iLength':x_num,
                                           'Grid-jLength':y_num,
                                           'Lineend':tag_dict[hex(0x8048)],
                                           'Scanrate-Frequency':tag_dict[hex(0x8049)],
                                           'Scanrate-Dutycycle':tag_dict[hex(0x804A)],
                                           'Motion':tag_dict[hex(0x804B)],
                                           'Feedback properties':feedback_dict}

                    x_data = np.linspace(x0, x0+x_len, num=x_num)
                    y_data = np.linspace(y0, y0+y_len, num=y_num)
                    xx_data, yy_data = np.meshgrid(x_data, y_data)
                else:
                    channel_name = tag_dict[hex(0x8052)]
                    channel_trace = 'trace' if tag_dict[hex(0x8051)] == 0 else 'retrace'
                    if channel_name not in data_dict.keys():
                        data_dict[channel_name] = {}
                    data_dict[channel_name][channel_trace] = {}
                    num_slots = tag_dict[hex(0x8080)]
                    for i in range(num_slots):
                        slot_name = tag_dict[hex(0x8090 + i*0x30)]
                        slot_type = tag_dict[hex(0x8091 + i*0x30)]
                        calibration_name = tag_dict[hex(0x80A0 + i*0x30)]
                        data_dict[channel_name][channel_trace][slot_name] = {}
                        if 'absolute' in slot_type.lower():
                            data_dict[channel_name][channel_trace][slot_name]['data'] = image_array
                            unit = ''
                        elif 'relative' in slot_type.lower():
                            slot_parent = 'raw'#tag_dict[hex(0x8092 + i*0x30)]
                            unit = tag_dict[hex(0x80A2 + i*0x30)]
                            scaling_type = tag_dict[hex(0x80A3 + i*0x30)]
                            if 'linear' in scaling_type.lower():
                                multiplier = tag_dict[hex(0x80A4 + i*0x30)]
                                offset = tag_dict[hex(0x80A5 + i*0x30)]
                                data_dict[channel_name][channel_trace][slot_name]['data'] = offset + \
                                    (multiplier*data_dict[channel_name][channel_trace][slot_parent]['data'])
                        else:
                            continue
                        data_dict[channel_name][channel_trace][slot_name]['info'] = {'Calibration name': calibration_name,
                                                                                    'Unit': unit}
        return data_dict

                
    #import datafile and get output dataframe
    def get_data(self, modes):
        #print(self.segment_path)
##        self.file_format = self.file_path.split('.')[-1]
##        self.data_zip = zipfile.ZipFile(self.file_path, 'r')
        #with zipfile.ZipFile(self.file_path, 'r') as self.data_zip:
        shared_header = self.data_zip.read('shared-data/header.properties').decode().split('\n')
        self.shared_header_dict = self.parse_header_file(shared_header)    
        file_list = self.data_zip.namelist()
        
        if self.segment_path == None: #all segments taken
            for file in file_list:
                if file.endswith('segments/'): # segments folder
                    for mode in modes:
                        result = self.anal_dict[mode]['function'](dirpath=file)
                        for key, value in result.items():
                            output = self.anal_dict[mode]['output']
                            output[key] = np.append(output[key], value)
        else: #specific segment taken
            for mode in modes:
                result = self.anal_dict[mode]['function'](dirpath=self.segment_path)
                for key, value in result.items():
                    output = self.anal_dict[mode]['output']
                    output[key] = np.append(output[key], value)

        for mode in modes:
            self.df[mode] = pd.DataFrame(self.anal_dict[mode]['output'])

    def parse_header_file(self, header):
        header_dict = {}
        header_dict['Date'] = header[0]
        for line in header[1:]:
            if line != '':
                line_data = line.split('=')
                keys = line_data[0].split('.')
                value = line_data[1]
                temp_dict = header_dict
                for idx, key in enumerate(keys):        
                    if key not in temp_dict.keys():
                        if idx == len(keys)-1:
                            temp_dict[key] = value
                        else:
                            temp_dict[key] = {}
                    temp_dict = temp_dict[key]
        return header_dict

    def decode_data(self, channel, segment_header_dict, segment_dir):
        info_id = segment_header_dict['channel'][channel]['lcd-info']['*']
        decode_dict = self.shared_header_dict['lcd-info'][info_id]
        data = self.data_zip.read(f'{segment_dir}/channels/{channel}.dat')
        
        point_length, type_code  = self.DATA_TYPES[decode_dict['type']]
        num_points = len(data) // point_length
        data_unpack = np.array(struct.unpack_from(f'!{num_points}{type_code}', data))
        encod_multiplier = float(decode_dict['encoder']['scaling']['multiplier'])
        encod_offset = float(decode_dict['encoder']['scaling']['offset'])
        data_conv = (data_unpack * encod_multiplier) + encod_offset
        conv_list = decode_dict['conversion-set']['conversions']['list'].split(' ')
        data_dict = {} #data dictionary of each level of converted data (eg. nominal/calibrated, distance/force)
        for conv in conv_list:
            multiplier = float(decode_dict['conversion-set']['conversion'][conv]['scaling']['multiplier'])
            offset = float(decode_dict['conversion-set']['conversion'][conv]['scaling']['offset'])
            data_conv = (data_conv * multiplier) + offset
            data_dict[conv] = data_conv

        return data_dict

    #Read JPK file format from matlab code
    #check https://in.mathworks.com/help/matlab/matlab_external/start-the-matlab-engine-for-python.html      
    #for installation procedure for MATLAB linking
    def get_height_measured2(self, filepath, modes):
        import matlab.engine
        eng = matlab.engine.start_matlab()

        matlab_output = eng.open_JPK(filepath)
        z_data = matlab_output['Height_measured_']['Trace']['AFM_image']
        #print(matlab_output['header'], matlab_output.keys())
        x0 = matlab_output['header']['x_Origin']
        y0 = matlab_output['header']['y_Origin']
        x_len = matlab_output['header']['x_scan_length']
        y_len = matlab_output['header']['y_scan_length']
        x_num = int(matlab_output['header']['x_scan_pixels'])
        y_num = int(matlab_output['header']['y_scan_pixels'])
        scan_angle = matlab_output['header']['Scanangle']*np.pi/180 #in radians
        x_data = np.linspace(x0, x0+x_len, num=x_num)
        y_data = np.linspace(y0, y0+y_len, num=y_num)
        xx_data, yy_data = np.meshgrid(x_data, y_data)
        mode = modes[0] #CHECK
        self.rotation_info = [x0, y0, scan_angle]
#         start = time.process_time()
#         print(x_num,y_num)
        output = self.anal_dict[mode]['output']
        
        output['Height'] = np.array(z_data).flatten()
        output['X'] = xx_data.flatten()
        output['Y'] = yy_data.flatten()
        output['Segment folder']= [None]*(x_num*y_num)
        
#         for i in range(x_num-1):
#             for j in range(y_num-1):
#                 #x_rotated = x0 + (x_data[i]-x0)*np.cos(scan_angle) + (y_data[j]-y0)*np.sin(scan_angle)
#                 #y_rotated = y0 -(x_data[i]-x0)*np.sin(scan_angle) + (y_data[j]-y0)*np.cos(scan_angle)
#                 output = self.anal_dict[mode]['output']
#                 output['Height'] = np.append(output['Height'], z_data[j][i])
#                 output['X'] = np.append(output['X'], x_data[i])
#                 output['Y'] = np.append(output['Y'], y_data[j])
#                 output['Segment folder'].append(None) #CHECK
#         print(time.process_time() - start)
        self.df[mode] = pd.DataFrame(self.anal_dict[mode]['output'])
        eng.quit()
