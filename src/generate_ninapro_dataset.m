% Generate NinaPro DB5 dataset and convert to Python-friendly format.
% NOTE: After running this file, the generated CSV files will need to be
% optimized for data format and memory.

clc; clear;

% Navigate to the correct directory
cd("../data/ninapro_db5/");

% Generate column names for EMG data
cols = ["subject","exercise"];
for chan = 1:16 % There are 16 sEMG channels
    cols = [cols, "emg_"+string(chan)];
end

% Generate column names for acceleration data
for chan = 1:3
    cols = [cols, "acc_"+string(chan)];
end

% Generate column names for glove data
for chan = 1:22
    cols = [cols, "glove_"+string(chan)];
end

% Generate column names for repetition and stimulus data
cols = [cols, "repetition","rerepetition","stimulus","restimulus"];

% Parse the entire NinaPro DB5 data

% File system hierarchy-related operations
subjects_dirs = dir(".");
subjects_dirs = subjects_dirs(3:end);
num_subjects = size(subjects_dirs,1);
fprintf("Number of subjects: %i\n",num_subjects);


% Loop through the subjects
for subj = 1:num_subjects
    DATA = [];
    files = dir(subjects_dirs(subj).name);
    files = files(3:end);
    num_files = size(files, 1);
    % Loop through the files inside the directory of the subject
    for f = 1:num_files
        filename = files(f).name;
        fprintf("Processing %s ...\n", filename);
        loaded = load(subjects_dirs(subj).name+"/"+filename);
        acc = double(loaded.acc);
        emg = double(loaded.emg);
        glove = double(loaded.glove);
        rep = double(loaded.repetition);
        rerep = double(loaded.rerepetition);
        stim = double(loaded.stimulus);
        restim = double(loaded.restimulus);
        data = [emg,acc,glove,rep,rerep,stim,restim];
        exercise = double(loaded.exercise);
        s = split(filename, "_");
        c = char(s(1));
        c = c(2:end);
        subject = str2double(c);
        % Generate and update data
        N = size(data, 1);
        subjects_data = subject*ones(N,1);
        exercise_data = exercise*ones(N,1);
        DATA = [DATA;[subjects_data, exercise_data, data]];
    end
    % Save data in csv file for specific subject of dataset
    savepath = subjects_dirs(subj).name+"/"+"subj_"+string(subject)+".csv";
    fprintf("Saving %s ...\n",savepath);
    writematrix(cols, savepath);
    writematrix(DATA, savepath, "writemode", "append");
    clear DATA; % For memory and computational efficiency
end