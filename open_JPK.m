function varargout2=open_JPK(varargin)

%
% This function opens .JPK image files and .JPK-force curves.
% It is also able to import a number of parameters relative to the hile.
%
% Image:
% [Image,Details of the image]=open_JPK(Path_to_File)
%
% Force Curve:
% [Force Curve]=open_JPK(Path_to_File);
%
% <<Path_to_File>> is optional, if not found then the user is prompted to
% select file through GUI.
%
% In case a JPK Nanowizard AFM is used and an image is imported, in combination with the following
% micromash [https://www.spmtips.com/] tips HQ:CSC38/Cr-Au or HQ:CSC37/Cr-Au, the SW also calibrates
% for lateral force microscopy. For further information on this functionality please turn to the
% following scientific article:
%
% Ortuso, Roberto D., Kaori Sugihara.
% "Detailed Study on the Failure of the Wedge Calibration Method at Nanonewton Setpoints for Friction Force Microscopy."
% The Journal of Physical Chemistry C 122.21 (2018): 11464-11474.
%
% Thank you to Mr. Ptak F. (PUC-Rio) for pointing out an error and the example files that caused the error for allowing corrections in the code.
%
% Author: Dr. Ortuso, R.D.
% Adolph Merkle Institute, Fribourg, CH.
% Contact: roberto.ortuso@unifr.ch
%
% Last update 17.June.2020


%#ok<*FNDSB>


flag_manual_select=1;
Uncalibrated=0;

valid_extensions={'.jpk';'.jpk-force';'.jpk-force-map'};
valid_extensions_getfile={'*.jpk';'*.jpk-force';'*.jpk-force-map'};

if(~isempty(varargin))
    if(isfile(varargin{1,1}))
        [~,~,extension]=fileparts(varargin{1,1});
        if(any(strcmp(extension,valid_extensions)))
            complete_path_to_afm_file=varargin{1,1};
            flag_manual_select=0;
        else
            clearvars extension
        end
    end
end

while(flag_manual_select==1)
    [afm_file_name,AFM_file_path,afm_file_index]=uigetfile(valid_extensions_getfile,'Choose AFM File');
    complete_path_to_afm_file=sprintf('%c%c',AFM_file_path,afm_file_name);
    [~,~,extension]=fileparts(complete_path_to_afm_file);
    if(afm_file_index==0)
        error('No File Selected')
    else
        if(any(strcmp(extension,valid_extensions)))
            fprintf('\n\nDetails of storage location:\n %s\n',sprintf('%c%c',AFM_file_path,afm_file_name))
            flag_manual_select=0;
        else
            clearvars afm_file_name AFM_file_path afm_file_index complete_path_to_afm_file extension
            waitfor(warndlg({'Accepted file formats limited to:','*.jpk','*.jpk-force','*.jpk-force-map (currently not supported)'},'Warning'));
        end
    end
end

% if(strcmp(extension,valid_extensions{1,1}))
% elseif(strcmp(extension,valid_extensions{1,2}))
%     unzip(complete_path_to_afm_file,'Test_me_Force')
% elseif(strcmp(extension,valid_extensions{1,3}))
%     zipJavaFile  = java.io.File(complete_path_to_afm_file);
%     zipFile = org.apache.tools.zip.ZipFile(zipJavaFile);
%     entries = zipFile.getEntries;
%     filelist={};
%     while entries.hasMoreElements
%         filelist = cat(1,filelist,char(entries.nextElement));
%     end
% end



if(strcmp(extension,valid_extensions{1,1}))
    if(~isempty(varargin))
        file_info=imfinfo(varargin{1,1});
    else
        file_info=imfinfo(complete_path_to_afm_file);
    end
    number_of_images=numel(file_info);
    wb=waitbar(0/number_of_images,sprintf('Loading Channel %.0f of %.0f',0,number_of_images),...
        'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
    setappdata(wb,'canceling',0);
    
    for i=1:number_of_images
        
        if(i==1)
            
            waitbar(i/number_of_images,wb,sprintf('Metadata of Image'));
            
            Type=(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32816)).Value);
            
            if(~isempty(find([file_info(i).UnknownTags.ID]==32832)))
                x_Origin=(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32832)).Value);
            else
                x_Origin=nan;
            end
            
            if(~isempty(find([file_info(i).UnknownTags.ID]==32833)))
                y_Origin=(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32833)).Value);
            else
                y_Origin=nan;
            end
            
            if(~isempty(find([file_info(i).UnknownTags.ID]==32834)))
                x_scan_length=(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32834)).Value);
            else
                x_scan_length=nan;
            end
            
            if(~isempty(find([file_info(i).UnknownTags.ID]==32835)))
                y_scan_length=(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32835)).Value);
            else
                y_scan_length=nan;
            end
            
            if(~isempty(find([file_info(i).UnknownTags.ID]==32838)))
                x_scan_pixels=(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32838)).Value);
            else
                x_scan_pixels=nan;
            end
            
            
            if(~isempty(find([file_info(i).UnknownTags.ID]==32839)))
                y_scan_pixels=(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32839)).Value);
            else
                y_scan_pixels=nan;
            end

            if(~isempty(find([file_info(i).UnknownTags.ID]==32836)))
                scanangle=rad2deg(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32836)).Value);
            else
                warning('No Scan Angle slot found')
                scanangle=nan;
            end
            
            if(strcmp(Type,'contact'))
                
                flag_data=strsplit(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32830)).Value)';
                
                flag=find(~cellfun(@isempty,strfind(flag_data,'setpoint-feedback-settings.i-gain')));
                if(~isempty(flag))
                    I_Gain=cellfun(@str2double, flag_data(flag+2,1));
                else
                    I_Gain=nan;
                end
                flag=find(~cellfun(@isempty,strfind(flag_data,'setpoint-feedback-settings.p-gain')));
                if(~isempty(flag))
                    P_Gain=cellfun(@str2double, flag_data(flag+2,1));
                else
                    P_Gain=nan;
                end
                
                if(~isempty(find([file_info(i).UnknownTags.ID]==33028)))&&(~isempty(find([file_info(i).UnknownTags.ID]==32980)))
                    Vertical_Sn=(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==33028)).Value)/(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32980)).Value);
                else
                    warning('No Sensitivity calibration slot found, image is uncalibarated')
                    Uncalibrated=1;
                    Vertical_Sn=1;
                end
                
                if(~isempty(find([file_info(i).UnknownTags.ID]==33076)))&&(~isempty(find([file_info(i).UnknownTags.ID]==32980)))
                    Vertical_kn=(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==33076)).Value)/(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==33028)).Value);
                else
                    warning('No Spring Constant calibration slot found, image is uncalibarated')
                    Uncalibrated=1;
                    Vertical_kn=1;
                end
                
                if(Uncalibrated==0)
                    Alpha=Vertical_Sn*Vertical_kn*14.75; % For further detail please refer to aforementioned publication
                else
                    Alpha=nan;
                end
                
%                if(~isempty(find([file_info(i).UnknownTags.ID]==32836)))
%                    scanangle=rad2deg(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32836)).Value);
%                else
%                    warning('No Scan Angle slot found')
%                    scanangle=nan;
%                end
                
                if(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32820)).Value==1)
                    Baseline_Raw=((file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32819)).Value-file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32821)).Value)-file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32980)).Value);
                    Bline_adjust='Yes';
                else
                    Baseline_Raw=((file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32819)).Value)-file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32980)).Value);
                    Bline_adjust='No';
                end
                
                SetP_V=Baseline_Raw;
                
                if(~isempty(find([file_info(i).UnknownTags.ID]==32981)))&&(~isempty(find([file_info(i).UnknownTags.ID]==32980)))
                    Raw=(Baseline_Raw-(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32981)).Value))/(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32980)).Value); % Setpoint in Volts [V]
                else
                    warning('No Baseline correction found')
                    Raw=SetP_V;
                    
                end
                
                if(~isempty(find([file_info(i).UnknownTags.ID]==33028)))&&(~isempty(find([file_info(i).UnknownTags.ID]==33029)))
                    SetP_m=(Raw)*(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==33028)).Value)+(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==33029)).Value); % Setpoint in meters [m]
                else
                    warning('No Setpoint (in meters) calibration slot found, image might be uncalibarated')
                    SetP_m=(Raw);
                end
                
                if(~isempty(find([file_info(i).UnknownTags.ID]==33076)))&&(~isempty(find([file_info(i).UnknownTags.ID]==33077)))
                    SetP_N=(Raw)*(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==33076)).Value)+(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==33077)).Value); % Setpoint in force [N]
                else
                    warning('No Setpoint (in Newton) calibration slot found, image might be uncalibarated')
                    SetP_N=(Raw);
                end
                
                Details_Img=struct(...
                    'Type', Type,...
                    'x_Origin', x_Origin,...
                    'y_Origin', y_Origin,...
                    'Scanangle', scanangle,...
                    'x_scan_length', x_scan_length,...
                    'y_scan_length', y_scan_length,...
                    'x_scan_pixels', x_scan_pixels,...
                    'y_scan_pixels', y_scan_pixels,...
                    'I_Gain', I_Gain,...
                    'P_Gain', P_Gain,...
                    'Baseline_V', Baseline_Raw,...
                    'Baseline_N', nan,...
                    'SetP_V', SetP_V,...
                    'SetP_m', SetP_m,...
                    'SetP_N', SetP_N,...
                    'Vertical_Sn', Vertical_Sn,...
                    'Vertical_kn', Vertical_kn,...
                    'Alpha', Alpha);
                
            elseif(strcmp(Type,'ac') || strcmp(Type,'ac-direct-drive'))
                if(~isempty(find([file_info(i).UnknownTags.ID]==32818)))
                    I_Gain=abs((file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32818)).Value));
                else
                    I_Gain=nan;
                end
                
                if(~isempty(find([file_info(i).UnknownTags.ID]==32821)))
                    Reference_Amplitude=(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32821)).Value);
                else
                    Reference_Amplitude=nan;
                end
                
                if(~isempty(find([file_info(i).UnknownTags.ID]==33028)))
                    Set_Amplitude=(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32822)).Value);
                else
                    Set_Amplitude=nan;
                end
                
                if(~isempty(find([file_info(i).UnknownTags.ID]==33028)))
                    Oscillation_Freq=(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32823)).Value);
                else
                    Oscillation_Freq=nan;
                end
                
                if(~isempty(find([file_info(i).UnknownTags.ID]==33028)))
                    Reference_Phase_shift=(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32824)).Value);
                else
                    Reference_Phase_shift=nan;
                end
                
                if(~isempty(find([file_info(i).UnknownTags.ID]==33028)))
                    Scan_Rate=(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32841)).Value);
                else
                    Scan_Rate=nan;
                end
                
                if(~isempty(find([file_info(i).UnknownTags.ID]==33028)))&&(~isempty(find([file_info(i).UnknownTags.ID]==32980)))
                    Vertical_Sn=(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==33028)).Value)/(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32980)).Value);
                else
                    warning('No Sensitivity calibration slot found, image is uncalibarated')
                    Uncalibrated=1;
                    Vertical_Sn=1;
                end
                
                if(~isempty(find([file_info(i).UnknownTags.ID]==33076)))&&(~isempty(find([file_info(i).UnknownTags.ID]==33028)))
                    Vertical_kn=(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==33076)).Value)/(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==33028)).Value);
                else
                    Uncalibrated=1;
                    warning('No Vertical spring constant slot found, image is uncalibarated')
                    Vertical_kn=1;
                end
                
                Details_Img=struct(...
                    'Type', Type,...
                    'x_Origin', x_Origin,...
                    'y_Origin', y_Origin,...
                    'Scanangle', scanangle,...
                    'x_scan_length', x_scan_length,...
                    'y_scan_length', y_scan_length,...
                    'x_scan_pixels', x_scan_pixels,...
                    'y_scan_pixels', y_scan_pixels,...
                    'I_Gain', I_Gain,...
                    'Reference_Amp', Reference_Amplitude,...
                    'Set_Amplitude', Set_Amplitude,...
                    'Oscillation_Freq', Oscillation_Freq,...
                    'Regerence_Ph_Shift', Reference_Phase_shift,...
                    'Scan_Rate', Scan_Rate,...
                    'Vertical_Sn', Vertical_Sn,...
                    'Vertical_kn', Vertical_kn);
                
            else
                error('Code not valid for type of AFM imaging...');
                
            end
            
            
        else
            waitbar(i/number_of_images,wb,sprintf('Loading Channel %.0f of %.0f',i,number_of_images));
            
            Channel_Name=(file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32850)).Value);
            
            strsp=(strsplit((file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32851)).Value)))';
            
            for k=1:size(strsp,1)
                if(strcmp(strsp{k,1},'retrace')==1)
                    if(strcmp(strsp{k+2,1},'true'))
                        trace_type_flag='ReTrace';
                    else
                        trace_type_flag='Trace';
                    end
                    break
                end
            end
            
            typpe_of_ch=file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32897)).Value;
            
            if(strcmp(typpe_of_ch,'nominal')||(strcmp(typpe_of_ch,'voltsamplitude')))
                m_ID=33028;
                off_ID=33029;
            elseif ((strcmp(typpe_of_ch,'force'))||(strcmp(typpe_of_ch,'calibrated'))||(strcmp(typpe_of_ch,'distanceamplitude')))
                m_ID=33076;
                off_ID=33077;
            elseif(strcmp(typpe_of_ch,'volts'))
                typpe_of_ch_det=file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==32848)).Value;
                if(strcmp(typpe_of_ch_det,'capacitiveSensorXPosition'))||(strcmp(typpe_of_ch_det,'servoDacY'))||(strcmp(typpe_of_ch_det,'servoDacX'))||(strcmp(typpe_of_ch_det,'capacitiveSensorYPosition'))
                    m_ID=33028;
                    off_ID=33029;
                else
                    m_ID=32980;
                    off_ID=32981;
                end
            else
                m_ID=32980;
                off_ID=32981;
            end
            
            if(~isempty(find([file_info(i).UnknownTags.ID]==m_ID)))
                multiplyer=file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==m_ID)).Value;
            else
                warning('No multiplyer slot found, image might be uncalibrated uncalibarated')
                multiplyer=1;
            end
            
            if(~isempty(find([file_info(i).UnknownTags.ID]==off_ID)))
                offset=file_info(i).UnknownTags(find([file_info(i).UnknownTags.ID]==off_ID)).Value;
            else
                warning('No offset slot found, image might be uncalibrated uncalibarated')
                offset=0;
            end
            
            if(strcmp(Channel_Name,'Vertical Deflection') && strcmp(Type,'contact'))
                if(strcmp(Bline_adjust,'No'))%Bline_adjust
                    afm_image=((double(imread(complete_path_to_afm_file,i))*multiplyer))+offset;
                else
                    Details_Img.Baseline_N=(Baseline_Raw*multiplyer)+offset;
                    afm_image=((double(imread(complete_path_to_afm_file,i))*multiplyer))+offset;
                end
            else
                afm_image=((double(imread(complete_path_to_afm_file,i))*multiplyer))+offset;
                
            end
            
            Image(i-1)=struct(...
                'Channel_name',...
                Channel_Name,...
                'Trace_type',...
                trace_type_flag,...
                'Raw_afm_image',...
                imread(complete_path_to_afm_file,i),...
                'Scale_factor',...
                multiplyer,...
                'Offset',...
                offset,...
                'AFM_image',...
                afm_image); %#ok<AGROW>
        end
        
        if(exist('wb','var'))
            if getappdata(wb,'canceling')
                delete (wb)
                break
            end
        end
        
    end
    [~,index] = sortrows({Image.Channel_name}.'); Image = Image(index); clear index
    varargout{1}=Image;
    varargout{2}=Details_Img;
    varargout{3}=complete_path_to_afm_file; %AFM_file_path;
elseif(strcmp(extension,valid_extensions{2,1}))
    
    [~,FileName,~]=fileparts(complete_path_to_afm_file);
    if ~exist('ForceCurvesMatalabExtracted', 'dir')
        mkdir('ForceCurvesMatalabExtracted')
    end
    unzip(complete_path_to_afm_file,sprintf('ForceCurvesMatalabExtracted\\%s',FileName))
    InfoDir=dir(sprintf('ForceCurvesMatalabExtracted\\%s',FileName));
    InfoDirSize=size(InfoDir,1);
    
    index_header=find(strcmp({InfoDir.name},'shared-data')==1);
    header_location=fullfile(InfoDir(index_header).folder,InfoDir(index_header).name,'header.properties');
    fid=fopen(header_location);
    header_metadata_raw=textscan(fid,'%s');
    header_metadata_raw=header_metadata_raw{1,1};
    fclose(fid);
    j=1;f=0;g=1;Conversion_Set_Id=0;Encoder_Id=0;
    for i=1:size(header_metadata_raw,1)
        temp=strsplit(header_metadata_raw{i,1},'=');
        if(size(temp,2)==2)
            temp_2=strsplit(temp{1,1},'.');
            for z=1:size(temp_2,2)+1
                if(z<=size(temp_2,2))
                    header_metadata_split{j,z}=temp_2{1,z};
                end
                if(z==size(temp_2,2)+1)
                    header_metadata_split{j,z}=temp{1,2};
                end
            end
            if(size(temp_2,2)>=4)
                if(strcmp(header_metadata_split{j,3},'channel'))&&(strcmp(header_metadata_split{j,4},'name'))
                    f=f+1;
                    FC_Data(f).Channel_name=header_metadata_split{j,5};
                    FC_Data(f).Encoded.Offset=[];
                    FC_Data(f).Encoded.Multiplier=[];
                end
            end
            t=99;
            if(size(temp_2,2)==7)
                if(strcmp(header_metadata_split{j,3},'conversion-set'))&&(strcmp(header_metadata_split{j,4},'conversion'))&&(strcmp(temp{1,2},'offsetmultiplier'))
                    Conversion_Set_Id=1;
                    Flag_Conversion_Set_Id=j;
                end
            end
            if(size(temp_2,2)==5)
                if(strcmp(header_metadata_split{j,3},'encoder'))&&(strcmp(header_metadata_split{j,4},'scaling'))&&(strcmp(header_metadata_split{j,5},'style'))&&(strcmp(temp{1,2},'offsetmultiplier'))
                    Encoder_Id=1;
                    Flag_Encoder_Id=j;
                end
                
            end
            if((Conversion_Set_Id==1)&&(Flag_Conversion_Set_Id+4==j))
                FC_Data(f).(header_metadata_split{j-2,5}).Offset=header_metadata_split{j-3,8};
                FC_Data(f).(header_metadata_split{j-2,5}).Multiplier=header_metadata_split{j-2,8};
                FC_Data(f).(header_metadata_split{j-2,5}).Unit=header_metadata_split{j,9};
                Conversion_Set_Id=0;
            end
            if((Encoder_Id==1)&&(Flag_Encoder_Id+4==j))
                FC_Data(f).Encoded.Offset=header_metadata_split{j-3,6};
                FC_Data(f).Encoded.Multiplier=header_metadata_split{j-2,6};
                FC_Data(f).Encoded.Unit=header_metadata_split{j,7};
                Encoder_Id=0;
            end
            j=j+1;
        end
    end
    [~,index] = sortrows({FC_Data.Channel_name}.'); FC_Data = FC_Data(index); clear index
    z=1;
    InfoDir_Segments=dir(sprintf('ForceCurvesMatalabExtracted\\%s\\%s',FileName,'segments'));
    for i=1:size(InfoDir_Segments,1)
        if(~isnan(str2double(InfoDir_Segments(i).name)))
            folderToEval(1,z)=i;
            z=z+1;
        end
    end
    Channel_Name_Imprint_Raw=sprintf('%s_%s','Data','Raw');
    Channel_Name_Imprint_Encoded=sprintf('%s_%s','Data','Encoded');
    F_Names_FC_Data=fieldnames(FC_Data);
    for i=1:size(folderToEval,2)
        Info_location=fullfile('ForceCurvesMatalabExtracted',FileName,'segments',InfoDir_Segments(folderToEval(1,i)).name,'segment-header.properties');
        fid=fopen(Info_location);
        segment_metadata_raw=textscan(fid,'%s');
        fclose(fid);
        for j=1:size(segment_metadata_raw{1,1},1)
            temp=strsplit(segment_metadata_raw{1,1}{j,1},'=');
            if(size(temp,2)==2)&&(strcmp(temp{1,1},'force-segment-header.settings.style'))
                Data_Type=temp{1,2};
                break
            end
        end
        Temp_InfoDir_Segments=dir(sprintf('ForceCurvesMatalabExtracted\\%s\\%s\\%s\\%s',FileName,'segments',InfoDir_Segments(folderToEval(1,i)).name,'channels'));
        [~,index1] = sortrows({Temp_InfoDir_Segments.name}.'); Temp_InfoDir_Segments = Temp_InfoDir_Segments(index1); clear index1
        j=1;
        while (j<=size(Temp_InfoDir_Segments,1))
            temp=strsplit(Temp_InfoDir_Segments(j).name,'.');
            if (~strcmp(temp{1,2},'dat'))
                Temp_InfoDir_Segments(j)=[];
                j=1;
            else
                j=j+1;
            end
        end
        
        if(size(Temp_InfoDir_Segments,1)~=size(FC_Data,2))
            error('Something whent wrong!! Different data sets')
        end
        
        for j=1:size(Temp_InfoDir_Segments,1)
            fid=fopen(sprintf('ForceCurvesMatalabExtracted\\%s\\%s\\%s\\%s\\%s',FileName,'segments',InfoDir_Segments(folderToEval(1,i)).name,'channels',Temp_InfoDir_Segments(j).name));
            flag_temp=fread(fid,'int32');
            fclose(fid);
            temp=strsplit(Temp_InfoDir_Segments(j).name,'.');
            if(~strcmp(FC_Data(j).Channel_name,temp{1,1}))
                error('Something whent wrong!!')
            end
            
            
            FC_Data(j).(Data_Type).(Channel_Name_Imprint_Raw) = flag_temp;
            FC_Data(j).(Data_Type).(Channel_Name_Imprint_Encoded)=flag_temp*str2double(FC_Data(j).Encoded.Multiplier)+str2double(FC_Data(j).Encoded.Offset);
            
            for w=3:size(F_Names_FC_Data,1)
                if(~isempty(FC_Data(j).(F_Names_FC_Data{w,1})))
                    Channel_Name_Imprint_Calibrated=sprintf('%s_%s','Data',F_Names_FC_Data{w,1});
                    FC_Data(j).(Data_Type).(Channel_Name_Imprint_Calibrated)=(FC_Data(j).(Data_Type).(Channel_Name_Imprint_Encoded))*str2double(FC_Data(j).(F_Names_FC_Data{w,1}).Multiplier)+str2double(FC_Data(j).(F_Names_FC_Data{w,1}).Offset);
                end
            end
            
        end
        clearvars segment_metadata_raw Temp_InfoDir_Segments
    end
    
    varargout{1}=FC_Data;
    
end

F = findall(0,'type','figure','tag','TMWWaitbar');
delete(F);
clearvars F
%varargout2 = varargout
varargout2 = struct;
channel_num=size(struct2table(varargout{1}));
for j=1:channel_num(1)
    channel_name = matlab.lang.makeValidName(varargout{1}(j).Channel_name);
    trace_type = varargout{1}(j).Trace_type;
    if any(strcmp(fieldnames(varargout2),channel_name))==0
        varargout2.(channel_name) = struct;
    end
    varargout2.(channel_name).(trace_type) = varargout{1}(j);
end
varargout2.header = varargout{2};
varargout2.filepath = varargout{3};
%varargout2.fileinfo = file_info;

% varargout2 = struct(strcat(varargout{1}(1).Channel_name," - ",varargout{1}(1).Trace_type),varargout{1}(3),...
%     'header',varargout{2},...
%     'filepath',varargout{3})
end