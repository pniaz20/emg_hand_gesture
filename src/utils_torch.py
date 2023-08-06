"""
This script holds some useful functions for training and testing deep learning models in PyTorch.
"""

# Necessary imports
import sys, os, warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
sys.path.append("../data/")
from models_torch import *
from datasets_torch import *
from recipe_577504_1 import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics                        import accuracy_score, mean_squared_error, f1_score
from torch.utils.data                       import DataLoader, random_split, Dataset
from timeit                                 import default_timer as timer
from datetime                               import datetime
from scipy.signal                           import butter, filtfilt, lfilter




def make_ann_dataset(ann_hparams:dict, inputs, outputs, **kwargs):
    """Makes a dataset appropriate to be used by a PyTorch-compatible `models.ANN` model, from timeseries data.

    ### Args:
        - `ann_hparams` (dict): Hyperparameters dictionary containing **ALL** of the following keys:
            - `in_seq_len_sec` (float): Input sequence length, in seconds
            - `out_seq_len_sec` (float): Output sequence length, in seconds
            - `data_downsampling` (int): Downsampling rate for generating the dataset. Must be >= 1.
            - `sequence_downsampling` (int): Downsampling rate for generating the sequences. Must be >= 1.
            - `data_sampling_rate_Hz` (float): Sampling rate of the timeseries data, in Hertz.
            - `in_features` (int): Number of input features. Default is 1.
            - `out_features` (int): Number of output features. Default is 1.
        - `inputs` (numpy array): Matrix of pure timeseries inputs, as in (num_data, features)
        - `outputs` (numpy arrays): Matrix of pure timeseries outputs, as in (num_data, features)

    ### Returns:
        - `datasets.TabularDataset`: Dataset object, inherited from `torch.utils.data.Dataset` class,fully preprocessed.
        Can be used later for making data loaders, etc.
        - `dict`: Modified dictionary of hyperparameters, also including the `"in_size"` and `"out_size"` keys, 
        so that a `models.ANN` model can be built afterwards.
    
    ### Usage:
    `dataset, hparams = make_ann_dataset(hparams, inputs, outputs)`
    
    **NOTE** A modified version of the input hyperparams dictionary will also be returned for later use when 
    constructing the ANN model.
    """
    in_window_size_sec = ann_hparams["in_seq_len_sec"]
    out_window_size_sec = ann_hparams["out_seq_len_sec"]
    data_downsampling = ann_hparams.get('data_downsampling') if ann_hparams.get('data_downsampling') else 1
    seq_downsampling = ann_hparams.get('sequence_downsampling') if ann_hparams.get('sequence_downsampling') else 1
    data_sampling_rate_Hz = ann_hparams['data_sampling_rate_Hz']
    
    if not ann_hparams.get('sequence_downsampling') and not ann_hparams.get('data_downsampling') and \
        ann_hparams.get('downsampling'):
        data_downsampling = 1
        seq_downsampling = ann_hparams['downsampling']

    # Why are we doing the following?
    # if in_window_size_sec < out_window_size_sec:
    #     out_window_size_sec = in_window_size_sec

    if in_window_size_sec > 0:
        in_sequence_length = int(in_window_size_sec*data_sampling_rate_Hz)
    else:
        in_sequence_length = 1
    if out_window_size_sec > 0:
        out_sequence_length = int(out_window_size_sec*data_sampling_rate_Hz)
    else:
        out_sequence_length = 1

    # Generating dataset
    dataset = TabularDataset(
        inputs, in_sequence_length, in_features=ann_hparams['in_features'], in_squeezed=True, 
        output_vec=outputs, out_seq_length=out_sequence_length, out_features=ann_hparams['out_features'], 
        out_squeezed=True, data_downsampling_rate=data_downsampling, 
        sequence_downsampling_rate=seq_downsampling, 
        **kwargs)

    ann_hparams["input_size"] = dataset.table_in.shape[-1]
    ann_hparams["output_size"] = dataset.table_out.shape[-1]
    ann_hparams["in_seq_len"] = dataset._in_seq_length_final
    ann_hparams["out_seq_len"] = dataset._out_seq_length_final
    return dataset, ann_hparams





def make_seq2dense_dataset(lstm_hparams:dict, inputs, outputs, **kwargs):
    """Makes a dataset appropriate to be used by a PyTorch-compatible `models.Seq2Dense` model, from timeseries data.

    ### Args:
        - `lstm_hparams` (dict): Hyperparameters dictionary containing ALL OF the following keys:
            - `in_seq_len_sec` (float): Input sequence length, in seconds, or zero, if there is no sequence
            - `out_seq_len_sec` (float): Output sequence length, in seconds, or zero, if there is no sequence
            - `data_downsampling` (int): Downsampling rate for generating the dataset. Must be >= 1.
            - `sequence_downsampling` (int): Downsampling rate for generating the sequences. Must be >= 1.
            - `data_sampling_rate_Hz` (float): Sampling rate of the timeseries data, in Hertz.
            - `in_features` (int): Number of input features. Default is 1.
            - `out_features` (int): Number of output features. Default is 1.
        - `inputs` (numpy array): Matrix of pure timeseries inputs, as in (num_data, features)
        - `outputs` (numpy arrays): Matrix of pure timeseries outputs, as in (num_data, features)

    ### Returns:
        - `datasets.TabularDataset`: Dataset object, inherited from `torch.utils.data.Dataset` class,fully preprocessed.
        Can be used later for making data loaders, etc.
        - `dict`: Modified dictionary of hyperparameters, also including the `"in_seq_len"` and `"out_seq_len"` keys, 
        so that a `models.Seq2Dense` model can be built afterwards.
    
    ### Usage:
    `dataset, hparams = make_seq2dense_dataset(hparams, inputs, outputs)`
    
    **NOTE** A modified version of the input hyperparams dictionary will also be returned for later use when 
    constructing the Seq2Dense model.
    """
    in_window_size_sec = lstm_hparams["in_seq_len_sec"]
    out_window_size_sec = lstm_hparams["out_seq_len_sec"]
    data_sampling_rate_Hz = lstm_hparams['data_sampling_rate_Hz']
    data_downsampling = lstm_hparams.get('data_downsampling') if lstm_hparams.get('data_downsampling') else 1
    seq_downsampling = lstm_hparams.get('sequence_downsampling') if lstm_hparams.get('sequence_downsampling') else 1
    
    if not lstm_hparams.get('sequence_downsampling') and not lstm_hparams.get('data_downsampling') and \
        lstm_hparams.get('downsampling'):
        data_downsampling = 1
        seq_downsampling = lstm_hparams['downsampling']

    if in_window_size_sec != 0:
        in_sequence_length = int(in_window_size_sec*data_sampling_rate_Hz)
    else:
        in_sequence_length = 1
    
    if out_window_size_sec != 0:
        out_sequence_length = int(out_window_size_sec*data_sampling_rate_Hz)
    else:
        out_sequence_length = 1

    # print("Inside make_seq2dense_dataset:")
    # print("Input sequence length:", in_sequence_length)
    # print("Output sequence length:", out_sequence_length)
    
    dataset = TabularDataset(
        input_vec=inputs, in_seq_length=in_sequence_length, in_features=lstm_hparams['in_features'], in_squeezed=False,
        output_vec=outputs, out_seq_length=out_sequence_length, out_features=lstm_hparams['out_features'], 
        out_squeezed=True, data_downsampling_rate=data_downsampling, sequence_downsampling_rate=seq_downsampling,
        **kwargs)

    # Sequence length of downsampled data is different, so we have to read it from the dataset object
    in_size = dataset._in_seq_length_final
    out_size = dataset._out_seq_length_final
    lstm_hparams['in_seq_len'] = in_size
    lstm_hparams['out_seq_len'] = out_size

    return dataset, lstm_hparams




# Designing lowpass filters
# Making different functions for smoothening (back-to-back filtering) and typical forward-filtering of timeseries data
def butter_lowpass_filter_back_to_back(data, cutoff, fs, order):
    """Low-pass filter the given data back-to-back (using filtfilt) using a Butterworth filter.

    ### Args:
        `data` (numpy array): Data, where each column is a timeseries, and each row is a time step.
        `cutoff` (float): Cutoff frequency, Hz
        `fs` (float): Sampling frequency, Hz
        `order` (int): Order of the filter

    ### Returns:
        Filtered Data
    """
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff / nyquist_freq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # print("a: ")
    # print(a)
    # print("b: ")
    # print(b)
    y = filtfilt(b, a, data.T).T
    # y = lfilter(b, a, data)
    return y


def butter_lowpass_filter_forward(data, cutoff, fs, order):
    """Low-pass filter the given data by only moving forwards (using filt, not filtfilt) using a Butterworth filter.

    ### Args:
        `data` (numpy array): Data, where each column is a timeseries, and each row is a time step.
        `cutoff` (float): Cutoff frequency, Hz
        `fs` (float): Sampling frequency, Hz
        `order` (int): Order of the filter

    ### Returns:
        Filtered Data
    """
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff / nyquist_freq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # y = filtfilt(b, a, data)
    y = lfilter(b, a, data.T).T
    
    return y




def generate_cell_array(dataframe, hparams,
    subjects_column:str=None, conditions_column:str=None, trials_column:str=None,
    input_cols:list=None, output_cols:list=None,
    input_preprocessor=None, input_postprocessor=None, 
    output_preprocessor=None, output_postprocessor=None,
    specific_subjects:list=None, specific_conditions:list=None, specific_trials:list=None,
    num_subjects_for_testing:int=None, subjects_for_testing:list=None, 
    conditions_for_testing:list=None, trials_for_testing:list=None,
    use_filtered_data:bool=False, lpcutoff:float=None, lporder:int=None, lpsamplfreq:float=None,
    data_squeezed:bool=True, 
    return_data_arrays_orig:bool=True, 
    return_data_arrays_processed:bool=True,
    return_train_val_test_data:bool=True,
    return_train_val_test_arrays:bool=True,
    verbosity:int=0,
    **kwargs):
    """
    Generates a cell array of data arrays, given a dataframe, and a set of parameters.
    
    Comes in handy when trying to separate the subjects, conditions, and trials of some timeseries experiments,
    and processing them separately such as scaling them, etc., including or excluding some of them from training or
    testing, and trying to generate training, validation, and testing sets out of those experiments.
    
    In many occasions, because input data to models will be sequence data, data for trials need to be analyzed and 
    scanned via a sliding window separately. On the other hand, a single preprocessed, scaled dataset is required for
    training. Sometimes, differnet subjects or conditions need to be kept for testing, and only osme of them are
    desired to be used for training. This function is useful for all of those cases.
    
    As a bonus, this function is also capable of rearranging and returning the original data, that can later be used 
    for plotting time plots, etc. It is also capable of preprocessign or postprocessing any sequential data before 
    or after processing with a sliding window.

    ### Args:
    
        - `dataframe` (`pd.DataFrame`): DataFrame holding all the raw timeseries data, all subjects and conditions.
        - `hparams` (dict): Dictionary to be used for extracting useful hyperparameters for the sliding window.
        
            This dicitonary will be sent to, and returned by, the `make_ann_dataset` or `make_seq2dense_dataset`
            function, depending on whether or not squeezed or unsqueezed data is desired (see below).
            This dictionary should contain the following keys:
             - `in_seq_len_sec` (float): Length of the input sequence, in seconds.
             - `out_seq_len_sec` (float): Length of the output sequence, in seconds.
             - `downsampling` (int): Downsampling factor, i.e. one out of how many samples will be extracted.
             - `data_sampling_rate_Hz` (float): Sampling rate of the data, in Hz.
             - `validation_data` (float|tuple, optional): Portion of the data to be used for validation while training.
                These datapoints will be selected randomly out of all the data. The data will be shuffled.
                If this is a tuple, it should hold the portion, followed by which set it comes from, 
                'trainset' or 'testset'. If it is a float, it will by default come from test set, if any.
                If there is no test set applicable according to the settings, training set will be used.
             
            
            Depending on whether squeezed or unsqueezed data is required (see below), the passed dictionary will be
            updated as follows:
            
             - If `data_squeezed` is True: The `input_size` and `output_size` hyperparameters will be updated.
            This is useful when `Keras_ANN` or `Pytorch_ANN` or generally MLP models will be deployed,
            where inputs are 2D.
            
             - If `data_squeezed` is False: The `in_seq_len` and `out_seq_len` hyperparameters will be updated.
            This is useful when `Keras_Seq2Dense` or `Pytorch_Seq2Dense` or generally RNN-based or CNN-based models
            will be deployed, where inputs are 3D.
            
        - `subjects_column` (str, optional): Column of the DataFrame holding subject numbers, if any. Defaults to None.
        - `conditions_column` (str, optional): Column of the DataFrame holding condition numbers. Defaults to None.
        - `trials_column` (str, optional): Column of the DataFrame holding trial numbers, if any. Defaults to None.
            **NOTE**: If any of the above arguments are None, it is assumed only one subject/condition/trial is
            involved in the timeseries experiments.
        - `input_cols` (list, optional): List of columns of input data. Defaults to None, meaning all data is input.
        - `output_cols` (list, optional): List of columns of output data. Defaults to None, meaning there is no output.
        - `input_preprocessor` (function, optional): Function to perform before processing. Defaults to None.
            **Note** This function should take in a DataFrame extracted from a specific subject/condition/trial,
            and return a numpy array containing preprocessed data.
        - `input_postprocessor` (function, optional): Function to perform after processing. Defaults to None.
            **Note** This function should take a numpy array which represents the processed data, and return a new
            numpy array with the same number of rows, containing the postprocessed data.
            By processing we mean passing through the sliding window function.
            **Note** Scaling of inputs and outputs can be done automatically (see below) and do not need to be
            manually employed as pre/postprocessing steps.
        - `output_preprocessor` (function, optional): Same as the one for inputs. Defaults to None.
        - `output_postprocessor` (function, optional): Same as the one for inputs. Defaults to None.
        - `specific_subjects` (list, optional): List of 1-indexed subject numbers to use. Defaults to None.
        - `specific_conditions` (list, optional): List of 1-indexed condition numbers to use. Defaults to None.
            **Note** If any of the above arguments are given, the cell array will be complete, but the corresponding
            elements of the cell arrays to the subjects/conditions not included in the list will be empty.
            These data will also not be used for generating training or testing data for a model. 
        - `num_subjects_for_testing` (int, optional): Self-explanatory. Defaults to None.
        - `subjects_for_testing` (list, optional): Self-explanatory. Defaults to None.
            **Note** If `num_subjects_for_testing` is given but `subjects_for_testing` is not given, the subjects
            for testing will be chosen randomly from all subjects.
        - `conditions_for_testing` (list, optional): List of 1-indexed condition numebrs used for testing.
            Defaults to None.
        - `trials_for_testing` (list, optional): List of 1-indexed trial numbers used for testing 
            out of every condition. Defaults to None.
        - `use_filtered_data` (bool, optional): Whether to low-pass filter data before processing. Defaults to False.
            **Note** If given, a digital Butterworth filter will be deployed, only forward-facing, that is,
            using `filt`, NOT `filtfilt`.
        - `lpcutoff` (float, optional): Lowpass filter cutoff frequency, Hz. Defaults to None.
        - `lporder` (int, optional): Lowpass filter order. Defaults to None.
        - `lpsamplfreq` (float, optional): Lowpass filter sampling frequency, Hz. Defaults to None.
        - `data_squeezed` (bool, optional): Whether the input data should be squeezed. Defaults to True.
            Squeezed data are 2D, as in (num_data, sequence_length*num_features),
            but unsqueezed data are 3D, as in (num_data, sequence_length, num_features).
        - `return_data_arrays` (bool, optional): Whether to return data arrays. Defaults to True.
            If True, the function will also return raw and scaled data arrays, which are not passed through the 
            sliding window, and they just contain typical timeseries data as in (num_steps, num_features).
            If False, the function will not return unprocessed data arrays, which can be useful for saving memory.
            Unprocessed data come in handy for drawing time plots, for instance.
        - `return_train_val_test_data`(bool, optional): Whether to return model training-related arrays and data. 
            Defaults to True.
            If True, the function will return processed, scaled and shuffled numpy arrays ready to be plugged into
            a learning model. They will include training, validation and testsets.
            If False, the function will not return such data. This can come in handy if no training or machine learning
            is involved in what you are trying to do, and you are trying to save memory.
        - `verbosity` (int, optional): Verbosity level. Defaults to 0. If 1, only the shape of the resulting databases
            are printed, as well as some fundamental information about the database. If 2, then basically everything is
            printed, along all the steps of the way.

    ### Returns:
    
        This function returns a dictionary holding some of the following, according to the arguments.
        - `is_test` (numpy nested cell arrays): Whether every trial is for testing data or not.
        - `x_train`,`x_val`,`x_test`,`y_train`,`y_val`,`y_test` (numpy arrays): Training, validation and testing
            arrays of input and output data, respectively, processed, scaled and processed with sliding window, fully
            ready to be fed to a learning algorithm. The data is also shuffled.
            If `hparams["validatoin_data"]` does not exist, `x_val` and `y_val` will be None, or empty.
            If testing-related parameters like `num_subjects_for_testing` or `subjects_for_testing` are not given,
            The `x_test` and `y_test` will be None, or empty.
            If `return_train_val_test_data` is False, `x_train`, `x_val`, `x_test`, `y_train`, `y_val`, `y_test` will
            not be included in the output dictionary.
        - `x_arrays` and `y_arrays` (numpy nested cell arrays): These include the same data as `x_train`, `x_val`, etc.,
            only they are separated for subjects, conditions and trials.
            `x_arrays[subj,cond,trial]` holds the corresponding data of one timeseries experiment, for instance.
        - `data_arrays_orig` and `data_arrays_processed` hold the data itself, nested and rearranged,
            but not passed through the sliding window. The `orig` one holds the raw data before preprocessing function,
            if any, and the `processed` one contains preprocessed and scaled data.
            
            **Note** One of the preprocessing steps that takes place by default, is downsampling.
            
            `data_arrays_orig[subj,cond,trial]` is a dictionary holding `input` and `output` keys, whose values are the
            original or processed timeseries data for the corresponding timeseries experiment, containing the
            inputs and outputs, respectively.
            
        - `hparams` (dictionary): Dictionary of hyperparameters used in this function, modified and updated.
            **Note** Returning this object is not actually necessary. The `hparam` parameter is already modified and 
            updated, because the function modifies the reference to the hparams object, so there is no real need
            for returning it, unless for back-up or storage reasons.
        
        - `subjects_test` (list): List of subject numbers used for testing.
        - `conditions_test` (list): List of condition numbers used for testing.
        - `trials_test` (list): List of trial numbers used for testing.
        - `num_subjects`, `num_conditions`, `num_trials` (int): Number of subjects, conditions and trials, respectively.
        
    """
    
    # Lists of arrays holding trial data, to concatenate later
    x_lst_lrn = []
    y_lst_lrn = []
    x_lst_tst = []
    y_lst_tst = []
    
    
    num_subjects = len(dataframe[subjects_column].value_counts()) if subjects_column else 1
    num_conditions = len(dataframe[conditions_column].value_counts()) if conditions_column else 1
    num_trials = len(dataframe[trials_column].value_counts()) if trials_column else 1
    
    if verbosity > 0:
        print("# subjects:   ", num_subjects)
        print("# conditions: ", num_conditions)
        print("# trials:     ", num_trials)
        print("\n")
    

    # Cellular arrays holding data of each trial
    if verbosity == 2: print("Initializing data arrays ...")
    x_arrays = np.empty((num_subjects,num_conditions,num_trials), dtype=np.ndarray)
    y_arrays = np.empty((num_subjects,num_conditions,num_trials), dtype=np.ndarray) 
    data_arrays_orig = np.empty((num_subjects,num_conditions,num_trials), dtype=dict)   
    data_arrays_processed = np.empty((num_subjects,num_conditions,num_trials), dtype=dict) 
    is_test_arr = np.empty((num_subjects,num_conditions,num_trials), dtype=object)

    # Determining which trials will be used for training+validation, and which ones will be used for testing
    if verbosity == 2: print("Determining training/validation, and testing trials...")
    
    if subjects_for_testing:
        subjects_test = subjects_for_testing
    elif num_subjects_for_testing:
        subjects_test = np.random.choice(np.arange(1,num_subjects+1), size=num_subjects_for_testing, replace=False)
    else:
        subjects_test = []
    
    if conditions_for_testing:
        conds_test = conditions_for_testing
    else:
        conds_test = []
        
    if trials_for_testing:
        trials_test = trials_for_testing
    else:
        trials_test = []
        
    if verbosity > 0:
        print("subjects used for testing:   ", subjects_test)
        print("conditions used for testing: ", conds_test)
        print("trials used for testing:     ", trials_test)
        print("\n")
    
    # Iterate through each trial
    if verbosity > 0: print("Iterating through all trials ...\n")
    for subj in range(num_subjects):
        if verbosity == 2: print("  > Subject %d ... "%(subj+1), end="")
        is_test_subject = (subj+1 in subjects_test) if subjects_test else False
        if verbosity == 2: print("(testing)" if is_test_subject else "")
        for ctrl in range(num_conditions):
            if verbosity==2: print("    >> Condition %d ... "%(ctrl+1), end="")
            is_test_cond = (ctrl+1 in conds_test) if conds_test else False
            if verbosity == 2: print("(testing)" if is_test_cond else "")
            for trial in range(num_trials):
                if verbosity == 2: print("      >>> Trial %d ... "%(trial+1), end="")
                is_test_trial = (trial+1 in trials_test) if trials_test else False
                #if verbosity == 2: print("(testing) " if is_test_trial else " ", end="")
                
                # Figure out if the trial is for training or testing
                is_test = (is_test_subject or is_test_cond or is_test_trial)
                if verbosity == 2: print("(testing) " if is_test else " ", end="")
                
                # Figure out if the trial is one of the specific ones that we want
                if specific_subjects is not None:
                    if subj+1 not in specific_subjects:
                        x_arrays[subj, ctrl, trial] = []
                        y_arrays[subj, ctrl, trial] = []
                        data_arrays_orig[subj, ctrl, trial] = {}
                        data_arrays_processed[subj, ctrl, trial] = {}
                        is_test_arr[subj, ctrl, trial] = None
                        if verbosity == 2: print("[skip]")
                        continue
                if specific_conditions is not None:
                    if ctrl+1 not in specific_conditions:
                        x_arrays[subj, ctrl, trial] = []
                        y_arrays[subj, ctrl, trial] = []
                        data_arrays_orig[subj, ctrl, trial] = {}
                        data_arrays_processed[subj, ctrl, trial] = {}
                        is_test_arr[subj, ctrl, trial] = None
                        if verbosity == 2: print("[skip]")
                        continue
                if specific_trials is not None:
                    if trial+1 not in specific_trials:
                        x_arrays[subj, ctrl, trial] = []
                        y_arrays[subj, ctrl, trial] = []
                        data_arrays_orig[subj, ctrl, trial] = {}
                        data_arrays_processed[subj, ctrl, trial] = {}
                        is_test_arr[subj, ctrl, trial] = None
                        if verbosity == 2: print("[skip]")
                        continue
                        
                is_test_arr[subj, ctrl, trial] = is_test
                # data_trial = dataframe[
                #     (dataframe[subjects_column] == subj+1) & \
                #     (dataframe[conditions_column] == ctrl+1) & \
                #     (dataframe[trials_column] == trial+1)]
                
                # Extract relevant columns of the data
                data_trial = dataframe
                if subjects_column:
                    data_trial = data_trial[data_trial[subjects_column] == subj+1]
                if conditions_column:
                    data_trial = data_trial[data_trial[conditions_column] == ctrl+1]
                if trials_column:
                    data_trial = data_trial[data_trial[trials_column] == trial+1]
                if verbosity == 2: print(data_trial.shape, end="")
                
                # Input Preprocessing
                data_in = data_trial[input_cols] if input_cols else data_trial
                if input_preprocessor:
                    x = input_preprocessor(data_in)
                else:
                    x = data_in.to_numpy().astype(np.float32)
                if len(x.shape) == 1:
                    x = x.reshape(-1, 1)
                if verbosity == 2: print("; in: ",x.shape, end="")
                
                # Output Preprocessing
                data_out = data_trial[output_cols] if output_cols else None
                if output_cols:
                    if output_preprocessor:
                        y = output_preprocessor(data_out)
                    else:
                        y = data_out.to_numpy().astype(np.float32)
                    if len(y.shape) == 1:
                        y = y.reshape(-1, 1)
                    if verbosity == 2: print(", out: ",y.shape, end="")
                else:
                    y = None
                
                # Construct DATA_ARRAYS_ORIG
                data_arrays_orig[subj, ctrl, trial] = {"input": x, "output": y} if return_data_arrays_orig else {}
                
                # Low-pass filter dta
                if use_filtered_data:
                    data_features = butter_lowpass_filter_forward(x, lpcutoff, lpsamplfreq, lporder)
                else:
                    data_features = x
                
                # GENERATE DATASET OBJECT
                if data_squeezed:
                    dataset, hparams = make_ann_dataset(hparams, data_features, y, verbose=False, **kwargs)
                else:
                    dataset, hparams = make_seq2dense_dataset(hparams, data_features, y, verbose=False, **kwargs)
                
                # Get input and output trables   
                x_processed = dataset.table_in
                y_processed = dataset.table_out
                if verbosity == 2:
                    print("; x: ",x_processed.shape, end="")
                    print(", y: ",y_processed.shape, end="")
                
                # POSTPROCESSING
                if input_postprocessor:
                    x_processed = input_postprocessor(x_processed)
                if output_postprocessor:
                    y_processed = output_postprocessor(y_processed)
                if verbosity == 2:
                    print("; x: ",x_processed.shape, end="")
                    print(", y: ",y_processed.shape, end="")
                
                # Construct data arrays    
                if return_train_val_test_arrays:
                    x_arrays[subj, ctrl, trial] = x_processed
                    y_arrays[subj, ctrl, trial] = y_processed
                else:
                    x_arrays[subj, ctrl, trial] = []
                    y_arrays[subj, ctrl, trial] = []
                if return_data_arrays_processed:
                    data_arrays_processed[subj, ctrl, trial] = {"input":dataset._invec, "output":dataset._outvec}
                else:
                    data_arrays_processed[subj, ctrl, trial] = {}
                # for debugging:
                # print("-------------------------------------")
                # Construct train-val-test arrays    
                if return_train_val_test_data:
                    if is_test:
                        x_lst_tst.append(x_processed)
                        y_lst_tst.append(y_processed)
                        # For debugging
                        # print("x_lst_tst: ")
                        # print(x_lst_tst)
                        # print("length of x_lst_tst: ")
                        # print(len(x_lst_tst))
                        # print("y_lst_tst: ")
                        # print(y_lst_tst)
                        # print("length of y_lst_tst: ")
                        # print(len(y_lst_tst))
                    else:
                        x_lst_lrn.append(x_processed)
                        y_lst_lrn.append(y_processed)
                        # For debugging
                        # print("x_lst_lrn: ")
                        # print(x_lst_lrn)
                        # print("length of x_lst_lrn: ")
                        # print(len(x_lst_lrn))
                        # print("y_lst_lrn: ")
                        # print(y_lst_lrn)
                        # print("length of y_lst_lrn: ")
                        # print(len(y_lst_lrn))
                
                # Go to the next line
                if verbosity == 2: print("\n")
    
    if verbosity > 0 and return_data_arrays_orig:
        print("Size of data_arrays_orig in bytes:      ", total_size(data_arrays_orig))
    if verbosity > 0 and return_data_arrays_processed:
        print("Size of data_arrays_processed in bytes: ", total_size(data_arrays_processed))           
    # Concatenate arrays to make all inputs and outputs, tabulated, scaled
    if verbosity > 0: print("Concatenating arrays and generating outputs ...")
    if return_train_val_test_data:
        if verbosity == 2: print("Returning train-val-test data ...")
        if verbosity == 2: print("Concatenating all training inputs and outputs ...")
        # For debugging
        # print("x_lst_lrn includes %d arrays." % len(x_lst_lrn))
        # print("y_lst_lrn includes %d arrays." % len(y_lst_lrn))
        # print("x_lst_tst includes %d arrays." % len(x_lst_tst))
        # print("y_lst_tst includes %d arrays." % len(y_lst_tst))
        #
        x_all_lrn = np.concatenate(x_lst_lrn, axis=0)
        y_all_lrn = np.concatenate(y_lst_lrn, axis=0) if output_cols else None
        # for debugging
        # print("shape of x_all_lrn: ", x_all_lrn.shape)
        # if output_cols: print("shape of y_all_lrn: ", y_all_lrn.shape)
        # Up to this point everything is correct, and training data size is correct.
        if hparams.get("validation_data"):
            if verbosity == 2: print("Calculating validation data ...")
            valdata = hparams["validation_data"]
            if isinstance(valdata, tuple) or isinstance(valdata, list):
                val_portion, val_set = valdata
                if "testset" not in val_set and "trainset" not in val_set:
                    if verbosity > 0: 
                        print("WARNING: validation set is neither 'trainset' nor 'testset'. 'trainset' will be used.")
                    val_set = "trainset"
            else:
                val_portion = valdata
                val_set = "testset" if subjects_test or conds_test or trials_test else "trainset"
            if verbosity > 0:
                print("Validation data source:  ",val_set)
                print("Validation data portion: ",val_portion)
            if val_set == "trainset" or not (subjects_test or conds_test or trials_test):
                val_data_set_x = x_all_lrn
                val_data_set_y = y_all_lrn
            else:
                x_all_tst = np.concatenate(x_lst_tst, axis=0)
                y_all_tst = np.concatenate(y_lst_tst, axis=0) if output_cols else None
                val_data_set_x = x_all_tst
                val_data_set_y = y_all_tst
            if verbosity == 2: print("Splitting data to extract validation dataset") 
            if output_cols:
                x_else, x_val, y_else, y_val = train_test_split(val_data_set_x, val_data_set_y, 
                    test_size=val_portion, random_state=SEED, shuffle=True)
            else:
                x_else, x_val = train_test_split(val_data_set_x, 
                    test_size=val_portion, random_state=SEED, shuffle=True)
                y_else = None
                y_val = None
                
            if val_set=="trainset":
                x_train = x_else
                y_train = y_else if output_cols else None
                if subjects_test or conds_test or trials_test:
                    x_test = np.concatenate(x_lst_tst, axis=0)
                    y_test = np.concatenate(y_lst_tst, axis=0) if output_cols else None
                    # for debugging
                    print("shape of x_test: ", x_test.shape)
                    print("shape of y_test: ", y_test.shape)
                    #
                    idx = np.random.permutation(x_test.shape[0])
                    x_test = x_test[idx]
                    y_test = y_test[idx] if output_cols else None
                else:
                    x_test = None
                    y_test = None
            else:
                x_train = x_all_lrn
                y_train = y_all_lrn
                idx = np.random.permutation(x_train.shape[0])
                x_train = x_train[idx]
                y_train = y_train[idx] if output_cols else None
                x_test = x_else
                y_test = y_else if output_cols else None
            #
            # For debugging
            # print("Shape of x_train: ", x_train.shape)
            # if output_cols: print("Shape of y_train: ", y_train.shape)
            # print("Shape of x_val: ", x_val.shape)
            # if output_cols: print("Shape of y_val: ", y_val.shape)
            # print("Shape of x_test: ", x_test.shape)
            # if output_cols: print("Shape of y_test: ", y_test.shape)
            # This conditional block seems to be working fine.
        else:
            if verbosity == 2: print("No validation data specified. Using all data for training ...")
            x_train = x_all_lrn
            y_train = y_all_lrn
            # For debugging
            # print("Shape of x_train before shuffling: ", x_train.shape)
            # if output_cols: print("Shape of y_train before shuffling: ", y_train.shape)
            # Up to this point everything is correct.
            x_val = None
            y_val = None
            idx = np.random.permutation(x_train.shape[0])
            x_train = x_train[idx]
            y_train = y_train[idx] if output_cols else None
            # For debugging
            # print("Shape of x_train after shuffling: ", x_train.shape)
            # if output_cols: print("Shape of y_train after shuffling: ", y_train.shape)
            # Up to this point everything is correct.
            if subjects_test or conds_test or trials_test:
                x_test = np.concatenate(x_lst_tst, axis=0)
                y_test = np.concatenate(y_lst_tst, axis=0) if output_cols else None
                # for debugging
                # print("shape of x_test: ", x_test.shape)
                # print("shape of y_test: ", y_test.shape)
                #
                idx = np.random.permutation(x_test.shape[0])
                x_test = x_test[idx]
                y_test = y_test[idx] if output_cols else None
            else:
                x_test = None
                y_test = None
            
        
        if verbosity > 0:
            print("x_train: ",x_train.shape)
            if y_train is not None: print("y_train: ",y_train.shape)
            if x_val is not None: print("x_val: ",x_val.shape)
            if y_val is not None: print("y_val: ",y_val.shape)
            if x_test is not None: print("x_test: ",x_test.shape)
            if y_test is not None: print("y_test: ",y_test.shape)
        # if subjects_test or conds_test or trials_test:
        #     x_test = np.concatenate(x_lst_tst, axis=0)
        #     y_test = np.concatenate(y_lst_tst, axis=0) if output_cols else None
        #     # for debugging
        #     print("shape of x_test: ", x_test.shape)
        #     print("shape of y_test: ", y_test.shape)
        #     #
        #     idx = np.random.permutation(x_test.shape[0])
        #     x_test = x_test[idx]
        #     y_test = y_test[idx] if output_cols else None
        # else:
        #     x_test = None
        #     y_test = None
            
        
    else:
        if verbosity == 2: print("Not returning train-val-test data ...")
        x_train = []
        x_val = []
        x_test = []
        y_train = []
        y_val = []
        y_test = []
        
        
    if verbosity > 0: print("Constructing output dictionary ...")    
    outdict = {
        "is_test": is_test_arr,
        "x_train": x_train, "x_val": x_val, "x_test": x_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "x_arrays": x_arrays, "y_arrays": y_arrays, 
        "data_arrays_orig": data_arrays_orig, "data_arrays_processed": data_arrays_processed,
        "hparams": hparams, "subjects_test": subjects_test, "conditions_test": conds_test, "trials_test": trials_test,
        "num_subjects": num_subjects, "num_conditions": num_conditions, "num_trials": num_trials
    }
    if verbosity > 0:
        print("Size of output dictionary in bytes: ", total_size(outdict))
        print("Done.\n")
    return outdict










def tsc_metrics(y_true, y_pred, transition_radius:int=None, consistency_penalty:float=1.0):
    """
    Calculate metrics for a time series classification problem.
    Metrics include:
     - `accuracy`: Overall accuracy score
     - `f1_score`: Overall F1 score
     - `concurrency`: Accuracy of the model in transition regions.
     - `consistency`: Accuracy of the model in consistent regions, reduced using a penalty.
     
     ### Parameters
     
        - `y_true`: True labels for the data, of a single timeseries experiment.
        - `y_pred`: Predicted labels for the data, of a single timeseries experiment.
        - `transition_radius`: Radius of transition regions, in number of time steps.
        - `consistency_penalty`: Penalty for consistency.
        
    ### Returns
        
    A dictionary holding the above metrics.
    
    **NOTE** The `consistency_penalty` multiplies the number of erroneous predictions in the consistent regions 
    by this value, before calculating the accuracy.
        
    """
    
    # Squeeze data
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    num_points = len(y_true)
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Auto-radius
    if not transition_radius:
        transition_radius = len(y_true)//20
    
    # Find transition points
    trans_points = np.where(np.diff(y_true) != 0)[0] + 1
    
    # Find indices of transition points
    indices = np.concatenate([np.arange(trans-transition_radius, trans+transition_radius+1) \
        for trans in trans_points]).astype(int)
    indices_list = indices.tolist()
    indices_list_opposite = [i for i in range(num_points)]
    for idx in indices_list:
        indices_list_opposite.remove(idx)
    
    # Extract indices of transition points and their vicinity from both y_true and y_pred
    y_true_transition = y_true[indices].astype(np.float32)
    y_pred_transition = y_pred[indices].astype(np.float32)
    
    # Calculate concurrency of chosen data
    concurrency = accuracy_score(y_true_transition, y_pred_transition)
    
    y_true_consistent = y_true[indices_list_opposite].astype(np.float32)
    y_pred_consistent = y_pred[indices_list_opposite].astype(np.float32)
    
    # Calculate consistency of chosen data
    consistency = 1-consistency_penalty+consistency_penalty*accuracy_score(y_true_consistent, y_pred_consistent)
    
    return {"accuracy":accuracy, "f1_score":f1, "consistency":consistency, "concurrency":concurrency}




def autoname(name):
    """
    Genereate a unique name for a file, based on the current time and the given name.
    Gets the `name` as a string and adds the time stamp to the end of it before returning it.
    """
    return name + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")