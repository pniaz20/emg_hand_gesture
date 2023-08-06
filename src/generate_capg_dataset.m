% Generate CAPG dataset and convert to Python-friendly format.
% NOTE: After running this file, the generated CSV files will need to be
% optimized.

clc; clear;

% Navigate to the correct directory
cd("../data/CAPG/");

% Generate strings for the three datasets' directories
database_strings = ["dba","dbb","dbc"];

% Generate empty lists for general info
num_subjects_list = [];
num_gestures_list = [];
num_trials_list = [];

% Generate column names for CSV files
cols = ["subject","gesture","trial"];
for band = 1:8              % There are 8 HD-sEMG sensors attached
    for chan = 1:16         % Each band has 16 dense channels of info
        cols = [cols, "b_"+string(band)+"_c_"+string(chan)];
    end
end

% Parse the entire CAPG data
for db = 1:3
    % File system hierarchy-related operations
    db_str = database_strings(db);
    subjects_dirs = dir(db_str);
    subjects_dirs = subjects_dirs(3:end);
    num_subjects = size(subjects_dirs,1);
    % Initializing variables for numbers of gestures and trials per dataset
    gestures_list = [];
    trials_list = [];
    num_gestures = 0;
    num_trials = 0;
    % Loop through the subjects
    for subj = 1:num_subjects
        DATA = [];
        files = dir(db_str+"/"+subjects_dirs(subj).name);
        files = files(3:end);
        num_files = size(files, 1);
        % Loop through the files inside the directory of the subject
        for f = 1:num_files
            filename = files(f).name;
            fprintf("Processing %s ...\n", filename);
            loaded = load(db_str+"/"+subjects_dirs(subj).name+"/"+filename);
            % Get gesture & trial and convert to double for compatibility
            gesture = double(loaded.gesture);
            trial = double(loaded.trial);
            subject = double(loaded.subject);
            % Get data (1000 x 128)
            data = loaded.data;
            % Update number of gestures and trials for subject if need be
            if ~any(gestures_list==gesture)
                gestures_list = [gestures_list; gesture];
                num_gestures = num_gestures + 1;
            end
            if ~any(trials_list==trial)
                trials_list = [trials_list; trial];
                num_trials = num_trials + 1;
            end
            % Generate and update data
            N = size(data, 1);
            subjects_data = subject*ones(N,1);
            gestures_data = gesture*ones(N,1);
            trials_data = trial*ones(N,1);
            DATA = [DATA;[subjects_data, gestures_data, trials_data, data]];
        end
        % Save data in csv file for specific subject of specific dataset
        savepath = db_str+"/"+subjects_dirs(subj).name+"/"+db_str+"_subj_"+string(subj)+".csv";
        fprintf("Saving %s ...\n",savepath);
        writematrix(cols, savepath);
        writematrix(DATA, savepath, "writemode", "append");
        clear DATA; % For memory anb computational efficiency
    end
    % Generate lists for gesture and trial counts of different datasets
    num_subjects_list = [num_subjects_list; num_subjects];
    num_gestures_list = [num_gestures_list; num_gestures];
    num_trials_list = [num_trials_list; num_trials];
end

% Sav e general information of the whole dataset in CSV file
general_info = [num_subjects_list, num_gestures_list, num_trials_list];
writematrix(["subjects","gestures","trials"], "general_info.csv");
writematrix(general_info, "general_info.csv", "writemode", "append");