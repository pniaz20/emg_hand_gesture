clc; clear all;

%% CSV Folder Generator

current_path = string(cd);
DIRECTION = dir;

FOLDER_NUMBER = size(DIRECTION,1);
for j = 1 : FOLDER_NUMBER
    SPLITTED_NAME = string(DIRECTION(j).name).split("dbc-preprocessed-");
    if( size(SPLITTED_NAME,1) == 2 )
        file_name = "dbc-preprocessed-" + SPLITTED_NAME(2);
    
        file_path = convertStringsToChars(current_path + "\" + file_name);
        cd(file_path);
        mkdir csv_Files
        A = cd;
        f_raw = dir(fullfile(A,'*.csv'));
        B = fullfile(A,'csv_Files');
        for ix = 1 :  size(f_raw,1) 
            movefile(fullfile(f_raw(ix).folder,f_raw(ix).name),B)
        end
    end
end
