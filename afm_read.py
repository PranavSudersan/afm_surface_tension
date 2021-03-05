import numpy as np
import pandas as pd
import struct
import zipfile


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
        self.data_zip = zipfile.ZipFile(self.file_path, 'r')
        self.get_data(modes)

    #import datafile and get output dataframe
    def get_data(self, modes):
        print(self.segment_path)
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
        for conv in conv_list:
            multiplier = float(decode_dict['conversion-set']['conversion'][conv]['scaling']['multiplier'])
            offset = float(decode_dict['conversion-set']['conversion'][conv]['scaling']['offset'])
            data_conv = (data_conv * multiplier) + offset

        return data_conv
