clc; clear all;

%% mat to csv converter BEASTTTT

current_path = string(cd);
DIRECTION = dir;

FOLDER_NUMBER = size(DIRECTION,1);
for j = 1 : FOLDER_NUMBER
    SPLITTED_NAME = string(DIRECTION(j).name).split("dbc-preprocessed-");
    if( size(SPLITTED_NAME,1) == 2 )
        file_name = "dbc-preprocessed-" + SPLITTED_NAME(2);
    
        file_path = convertStringsToChars(current_path + "\" + file_name);
        cd(file_path);
        direction = dir;
        file_NUMBER = size(direction,1);
        for i = 1 : file_NUMBER 
            [~,File_NAME_without_mat,ext] = fileparts(which(direction(i).name));
            if ext == ".mat"
                %% Read and Change file
                FileData = load(direction(i).name);
                FileData.trial = double(FileData.trial);
                FileData.gesture = double(FileData.gesture);
                FileData.subject = double(FileData.subject);
                Row_NUMBER = size(FileData.data,1);
                FileData.trial = FileData.trial * ones(Row_NUMBER,1);
                FileData.gesture = FileData.gesture * ones(Row_NUMBER,1);
                FileData.subject = FileData.subject * ones(Row_NUMBER,1);
                FileData.trial = int8(FileData.trial);
                FileData.gesture = int8(FileData.gesture);
                FileData.subject = int8(FileData.subject);
                File_Combination = [FileData.subject, FileData.gesture, FileData.trial, FileData.data];

                NEW_CSV_File_name = string(File_NAME_without_mat) + ".csv";
                NEW_CSV_File_name = convertStringsToChars(NEW_CSV_File_name);
                writematrix(File_Combination,NEW_CSV_File_name)
            end
        end
    end
end
