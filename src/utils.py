"""
This script holds some useful functions for training and testing deep learning models in Keras and PyTorch.
"""

# Necessary imports
import sys, os, warnings
from sklearn.preprocessing import OneHotEncoder
sys.path.append("../data/")
if os.path.isdir("keras2cpp"):
    sys.path.append("keras2cpp") 
from models import *
from datasets import *
from recipe_577504_1 import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import tensorflow as tf
import torch#, keras, optuna, statsmodels
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
#from optuna.samplers                       import TPESampler
#from sklearn.preprocessing                 import StandardScaler
from sklearn.model_selection                import train_test_split
from sklearn.metrics                        import accuracy_score, mean_squared_error, f1_score
from torch.utils.data                       import DataLoader, random_split, Dataset
#from statsmodels.graphics.tsaplots         import plot_acf
#from statsmodels.tsa.seasonal              import seasonal_decompose
from timeit                                 import default_timer as timer
from datetime                               import datetime
#from keras.models                          import Sequential, Model, load_model
#from keras.activations                     import softmax, relu, tanh, sigmoid
#from keras.layers.advanced_activations     import LeakyReLU
#from keras.layers                          import Dense, LSTM, Flatten, BatchNormalization, Dropout, Activation
from keras.optimizers                       import Adam, SGD, RMSprop
from tensorflow.keras.optimizers.schedules  import ExponentialDecay
from keras.callbacks                        import EarlyStopping
from keras2cpp.keras2cpp                    import export_model
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





def make_seq2dense_dataset_keras(lstm_hparams:dict, inputs, outputs, **kwargs):
    """Makes a dataset appropriate to be used by a Keras-compatible `models.Keras_Seq2Dense` model from timeseries data.

    ### Args:
        - `lstm_hparams` (dict): Hyperparameters dictionary containing ALL of the following keys:
            - `in_seq_len_sec` (float): Input sequence length, in seconds
            - `out_seq_len_sec` (float): Output sequence length, in seconds
            - `data_downsampling` (int): Downsampling rate for generating the dataset. Must be >= 1.
            - `sequence_downsampling` (int): Downsampling rate for generating the sequences. Must be >= 1.
            - `data_sampling_rate_Hz` (float): Sampling rate of the timeseries data, in Hertz.
            - `in_features` (int): Number of input features. Default is 1.
            - `out_features` (int): Number of output features. Default is 1.
            - `validation_data` (float): Percentage of the data used for validation
        - `inputs` (numpy array): Matrix of pure timeseries inputs, as in (num_data, features)
        - `outputs` (numpy arrays): Matrix of pure timeseries outputs, as in (num_data, features)

    ### Returns:
        - `numpy.ndarray`: Training inputs, fully scaled and preprocessed.
        - `numpy.ndarray`: Validation inputs, fully scaled and preprocessed.
        - `numpy.ndarray`: Training outputs, fully scaled and preprocessed.
        - `numpy.ndarray`: Validation outputs, fully scaled and preprocessed.
        - `datasets.TabularDataset`: Dataset object, inherited from `torch.utils.data.Dataset` class. 
        Can be used later for making data loaders, etc.
        - `dict`: Modified dictionary of hyperparameters, also including the `"in_size"` and `"out_size"` keys, 
        so that a `models.Keras_Seq2Dense` model can be built afterwards.
    
    ### Usage:
    `x_train, x_val, y_train, y_val, dataset, hparams = make_seq2dense_dataset_keras(hparams, inputs, outputs)`
    
    **NOTE** A modified version of the input hyperparams dictionary will also be returned for later use when 
    constructing the `models.Keras_Seq2Dense` model.
    """
    seq2d_dataset, lstm_hparams = make_seq2dense_dataset(lstm_hparams, inputs, outputs, **kwargs)
    x_train, x_val, y_train, y_val = train_test_split(seq2d_dataset.table_in, seq2d_dataset.table_out, 
    test_size=lstm_hparams['validation_data'], random_state=SEED, shuffle=True)
    return x_train, x_val, y_train, y_val, seq2d_dataset, lstm_hparams






def train_model_keras(model, x_train, x_val, y_train, y_val, hparams:dict, verbose:bool=True, 
    saveto:str=None, export:str=None):
    """Train a Keras-compatible model, given some hyperparameters.

    ### Args:
        - `model` (Keras model): A keras-compatible model
        - `x_train` (numpy array): Training inputs
        - `x_val` (numpy array): Validation inputs
        - `y_train` (numpy array): Training target outputs
        - `y_val` (numpy array): Validation target outputs
        - `hparams` (dict): Hyperparameters, containing the following keys:
            - `learning_rate` (float): Initial learning rate of training.
            - `learning_rate_decay_gamma` (float): Exponential decay rate gamma for learning rate, if any.
            - `optimizer` (str): Optimizer, options are "sgd" and "adam" for now.
            - `optimizer_params` (dict): Additional parameters of the optimizer, if any.
            - `batch_size` (int): Minibatch size for training.
            - `epochs` (int): Maximum number of epochs for training.
        - `verbose` (bool, optional): Verbosity of training. Defaults to True.
        - `saveto` (str, optional): Save Keras model in path. Defaults to None.
        - `export` (str, optional): Save Keras model in .model file using keras2cpp for later use in C++. 
        Defaults to None.

    ### Returns:
        - `model`: Trained Keras-compatible model
        - `history`: Keras model history object
    
    **NOTE** This function uses MSE loss function for now.
    
    **NOTE**
    THIS FUNCTION IS DEPRACATED. PLEASE USE EITHER THE BUILT-IN METHODS OF THE CORRESPONDING MODEL CLASSES IN MODELS.PY,
    OR USE THE GLOBALLY USABLE TRAINING FUNCTIONS IN THE PREAMBLE OF MODELS.PY
    """
    warnings.warn("This function is deprecated. Please use the built-in methods of the corresponding model classes \
        in models.py, or use the globally usable training functions in the preamble of models.py", DeprecationWarning)
    _earlystop = hparams.get("validation_tolerance_epochs")
    _es = EarlyStopping(monitor='val_loss', mode="min", patience=_earlystop) if _earlystop else None

    opt_dict = {'adam':Adam, 'sgd':SGD, 'rmsprop':RMSprop}
    if hparams.get('learning_rate_decay_gamma'):
        itersPerEpoch = x_train.shape[0]//hparams['batch_size']
        sch = ExponentialDecay(initial_learning_rate=hparams['learning_rate'], 
        decay_steps=itersPerEpoch, decay_rate=hparams['learning_rate_decay_gamma'])
        lr = sch
    else:
        lr = hparams['learning_rate']
    if hparams.get("optimizer_params"):
        optparam = hparams['optimizer_params']
        opt = opt_dict[hparams['optimizer']](learning_rate=lr, **optparam)
    else:
        opt = opt_dict[hparams['optimizer']](learning_rate=lr)
    model.compile(optimizer=opt, loss=hparams["loss_function"], metrics=hparams["metrics"])
    hist = model.fit(x_train, y_train, batch_size=hparams['batch_size'], epochs=hparams['epochs'], 
    validation_data=(x_val, y_val), verbose=verbose, callbacks=[_es] if _es else None)
    if saveto:
        try:
            model.save(saveto)
        except Exception as e:
            #if verbose:
                print(e)
                print("Cannot serialize Keras model.")
    if export:
        try:
            export_model(model, export)
        except Exception as e:
            #if verbose:
                print(e)
                print("Cannot export Keras model.")
    return model, hist
    




def train_model(model, hparams, dataset, verbose=True, script_before_save=True, saveto=None):
    """Train a Keras-compatible model, given some hyperparameters.

    ### Args:
        - `model` (Keras model): A keras-compatible model
        - `hparams` (dict): Hyperparameters, containing the following keys:
            - `learning_rate` (float): Initial learning rate of training.
            - `learning_rate_decay_gamma` (float): Exponential decay rate gamma for learning rate, if any.
            - `optimizer` (str): Optimizer, options are "sgd" and "adam" for now.
            - `optimizer_params` (dict): Additional parameters of the optimizer, if any.
            - `batch_size` (int): Minibatch size for training.
            - `epochs` (int): Maximum number of epochs for training.
            - `validation_tolerance_epochs` (int): Number of epochs to tolerate unimprovement in validation loss
            for early stopping before maximum epochs.
        - `dataset` (`datasets.TabularDataset`|`torch.utils.data.Dataset`): Dataset object to be used
        - `verbose` (bool, optional): Verbosity of training. Defaults to True.
        - `script_before_save` (bool, optional): Use TorchScript for serializing the model. Defaults to True.
        - `saveto` (str, optional): Save PyTorch model in path. Defaults to None.
        Defaults to None.

    ### Returns:
        - `model`: Trained PyTorch-compatible model
        - `history`: PyTorch model history dictionary, containing the following keys:
            - `training_loss`: List containing training loss values of epochs.
            - `validation_loss`: List containing validation loss values of epochs.
            - `learning_rate`: List containing learning rate values of epochs.
    
    **NOTE** This function uses MSE loss function for now.
    
    **NOTE**
    THIS FUNCTION IS DEPRACATED. PLEASE USE EITHER THE BUILT-IN METHODS OF THE CORRESPONDING MODEL CLASSES IN MODELS.PY,
    OR USE THE GLOBALLY USABLE TRAINING FUNCTIONS IN THE PREAMBLE OF MODELS.PY
    """
    warnings.warn("This function is deprecated. Please use the built-in methods of the corresponding model classes \
        in models.py, or use the globally usable training functions in the preamble of models.py", DeprecationWarning)
    criterion_dict = {"mse":nn.MSELoss()}
    validation_data = hparams['validation_data']

    hist_training_loss = []
    hist_validation_loss = []
    hist_learning_rate = []

    torch.cuda.empty_cache()

    num_all_data = dataset.size
    num_val_data = int(validation_data*num_all_data)
    num_train_data = num_all_data - num_val_data
    (trainset, valset) = random_split(dataset, (num_train_data, num_val_data), 
    generator=torch.Generator().manual_seed(42))

    trainloader = DataLoader(trainset, batch_size=hparams['batch_size'], shuffle=True, num_workers=0)
    validloader = DataLoader(valset, batch_size=hparams['batch_size'], shuffle=True, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("selected device: ",device)
    
    model.to(device)
    model.train()
    
    criterion = criterion_dict[hparams['loss_function']]
    if hparams['optimizer']=='adam':
        if hparams.get('optimizer_params'):
            optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'], **hparams['optimizer_params'])
        else:
            optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    elif hparams['optimizer']=='sgd':
        if hparams.get('optimizer_params'):
            optimizer = optim.SGD(model.parameters(), lr=hparams['learning_rate'], **hparams['optimizer_params'])
        else:
            optimizer = optim.SGD(model.parameters(), lr=hparams['learning_rate'])
    else:
        raise ValueError("Sorry, only 'adam' and 'sgd' are supported for now.")
    
    if hparams.get("learning_rate_decay_gamma"):
        if verbose:
            print("The learning rate has an exponential decay rate of %.5f."%hparams["learning_rate_decay_gamma"])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=hparams["learning_rate_decay_gamma"])
        lr_sch = True
    else:
        lr_sch = False
    
    
    # Training
    num_training_batches = len(trainloader)
    num_validation_batches = len(validloader)
    
    progress_bar_size = 20
    ch = "█"
    intvl = num_training_batches/progress_bar_size;
    valtol = hparams["validation_tolerance_epochs"]
    minvalerr = 1000.0
    badvalcount = 0
    
    tStart = timer()
    for epoch in range(hparams["epochs"]):
        
        tEpochStart = timer()
        epoch_loss_training = 0.0
        epoch_loss_validation = 0.0
        newnum = 0
        oldnum = 0
    
        if verbose: print("Epoch %3d/%3d ["%(epoch+1, hparams["epochs"]), end="")
        model.train()
        for i, data in enumerate(trainloader):
            seqs, targets = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            predictions = model(seqs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            if lr_sch:
                scheduler.step()
            epoch_loss_training += loss.item()
            # Visualization of progressbar
            if verbose:
                newnum = int(i/intvl)
                if newnum > oldnum:
                    print((newnum-oldnum)*ch, end="")
                    oldnum = newnum
        if verbose: print("] ", end="")
    
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(validloader):
                seqs, targets = data[0].to(device), data[1].to(device)
                predictions = model(seqs)
                loss = criterion(predictions, targets)
                epoch_loss_validation += loss.item()
        epoch_loss_training /= num_training_batches
        epoch_loss_validation /= num_validation_batches
        tEpochEnd = timer()
        if verbose:
            print("Trn Loss: %5.4f |Val Loss: %5.4f |Time: %6.3f sec" % (
                epoch_loss_training, 
                epoch_loss_validation, tEpochEnd-tEpochStart))

        hist_training_loss.append(epoch_loss_training)
        hist_validation_loss.append(epoch_loss_validation)
        if lr_sch:
            hist_learning_rate.append(scheduler.get_last_lr())
        else:
            hist_learning_rate.append(hparams['learning_rate'])

        # Checking for early stopping
        if epoch_loss_validation < minvalerr:
            minvalerr = epoch_loss_validation
            badvalcount = 0
        else:
            badvalcount += 1
            if badvalcount > valtol:
                if verbose:
                    print("Validation loss not improved for more than %d epochs."%badvalcount)
                    print("Early stopping criterion with validation loss has been reached. \
                        Stopping training at %d epochs..."%epoch)
                break
    
    tFinish = timer()
    if verbose:        
        print('Finished Training.')
        print("Training process took %.2f seconds."%(tFinish-tStart))
    if saveto:
        try:
            if verbose: print("Saving model...")
            if script_before_save:
                example,_ = next(iter(trainloader))
                example = example[0,:].unsqueeze(0)
                model.cpu()
                with torch.no_grad():
                    traced = torch.jit.trace(model, example)
                    traced.save(saveto)
            else:
                with torch.no_grad():
                    torch.save(model, saveto)
        except Exception as e:
            if verbose:
                print(e)
                print("Failed to save the model.")
        if verbose: print("Done.")
    
    history = {'training_loss':hist_training_loss, 'validation_loss':hist_validation_loss, 
    'learning_rate':hist_learning_rate}
    return model, history




def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=np.float32, axis=0)
    ret[n:,...] = ret[n:,...] - ret[:-n,...]
    return ret[n-1:,...]/n




def evaluate_model_keras(model, x_val, y_val, 
    rmse_seq_file_name:str=None, rmse_seq_fig_title:str=None, rmse_seq_x_label:str=None, rmse_seq_y_label:str=None,
    timeplots_file_name:str=None, timeplots_fig_title:str=None, timeplots_x_label:str=None, timeplots_y_label:str=None,
    timeplots_gridsize:tuple=(5,4), moving_average_size=3, give_unscaled:bool=False, scaler=None):
    """Evaluate Keras-compatible model on timeseries forecasting.

    Args:
        - `model` (Keras model): Keras model
        - `x_val` (numpy array): Validation inputs
        - `y_val` (numpy array): Validation outputs
        - `rmse_seq_file_name` (str, optional): RMSE sequence image file name. Defaults to None.
        - `rmse_seq_fig_title` (str, optional): RMSE sequence figure title. Defaults to None.
        - `rmse_seq_x_label` (str, optional): RMSE sequence figure X label. Defaults to None.
        - `rmse_seq_y_label` (str, optional): RMSE sequence figure Y label. Defaults to None.
        - `timeplots_file_name` (str, optional): Timeplot image file name. Defaults to None.
        - `timeplots_fig_title` (str, optional): Timeplot figure super title. Defaults to None.
        - `timeplots_x_label` (str, optional): Timeplot figure X label. Defaults to None.
        - `timeplots_y_label` (str, optional): Timeplot figure Y label. Defaults to None.
        - `timeplots_gridsize` (tuple, optional): Timeplot grid size, in number of images. Defaults to (5,4).
        - `moving_average_size` (int, optional): Moving average size for forecast smoothing. Defaults to 3.
        - `give_unscaled` (bool, optional): Unscale predictions to postprocess them. Defaults to False.
        - `scaler` (sklearn scaler, optional): sklearn Scaler object used for scaling outputs. Defaults to None.

    Returns:
        dict: results dictionary with the following keys:
            - `rmse_val_seq`: Sequence of RMSE values.
            - `rmse_val_tot`: Overall RMSE value.
    """

    rmse_val_seq = []
    rmse_val_tot = 0

    pred_val = model.predict(x_val)

    if give_unscaled:
        y_val = scaler.inverse_transform(y_val)
        pred_val = scaler.inverse_transform(pred_val)

    rmse_val_tot = mean_squared_error(y_val, pred_val, squared=False, multioutput="uniform_average")
    rmse_val_seq = mean_squared_error(y_val, pred_val, squared=False, multioutput="raw_values")
    num_rows, num_cols = [int(i) for i in timeplots_gridsize]
    num_plots = num_rows * num_cols
    pred_unscaled_plot = pred_val[:num_plots]
    target_unscaled_plot = y_val[:num_plots]
    print("Total RMSE of " + "unscaled " if give_unscaled else " " + "data: ", rmse_val_tot)
    print("Sequence RMSE of " + "unscaled" if give_unscaled else " " + "is shown below: ")
    plt.figure(figsize=(15,7))
    plt.plot(rmse_val_seq)
    plt.xlabel(rmse_seq_x_label)
    plt.ylabel(rmse_seq_y_label)
    plt.title(rmse_seq_fig_title)
    plt.grid(True)
    if rmse_seq_file_name: plt.savefig(rmse_seq_file_name, dpi=600)


    plt.figure(figsize=(num_plots, num_plots))
    plt.suptitle(timeplots_fig_title)    
    for i in range(num_plots):
        plt.subplot(num_rows,num_cols,i+1)
        plt.grid(True)
        plt.plot(target_unscaled_plot[i], "-b")
        plt.plot(moving_average(pred_unscaled_plot[i], moving_average_size), "-r")
        if i == 0:
            plt.legend(["target", "pred"])
        if i >= num_plots - num_cols:
            plt.xlabel(timeplots_x_label)
        if i%num_cols==0:
            plt.ylabel(timeplots_y_label)

    if timeplots_file_name: plt.savefig(timeplots_file_name, dpi=600)
    return {"rmse_val_seq":rmse_val_seq, "rmse_val_tot":rmse_val_tot}







def evaluate_model(model, dataloader,
    rmse_seq_file_name:str=None, rmse_seq_fig_title:str=None, rmse_seq_x_label:str=None, rmse_seq_y_label:str=None,
    timeplots_file_name:str=None, timeplots_fig_title:str=None, timeplots_x_label:str=None, timeplots_y_label:str=None, 
    timeplots_gridsize:tuple=(5,4), moving_average_size:int=3, give_unscaled:bool=False, scaler=None):
    """Evaluate PyTorch-compatible model on timeseries forecasting.

    Args:
        - `model` (PyTorch model): Trained PyTorch model
        - `dataloader` (PyTorch DataLoader): Validation DataLoader
        - `rmse_seq_file_name` (str, optional): RMSE sequence image file name. Defaults to None.
        - `rmse_seq_fig_title` (str, optional): RMSE sequence figure title. Defaults to None.
        - `rmse_seq_x_label` (str, optional): RMSE sequence figure X label. Defaults to None.
        - `rmse_seq_y_label` (str, optional): RMSE sequence figure Y label. Defaults to None.
        - `timeplots_file_name` (str, optional): Timeplot image file name. Defaults to None.
        - `timeplots_fig_title` (str, optional): Timeplot figure super title. Defaults to None.
        - `timeplots_x_label` (str, optional): Timeplot figure X label. Defaults to None.
        - `timeplots_y_label` (str, optional): Timeplot figure Y label. Defaults to None.
        - `timeplots_gridsize` (tuple, optional): Timeplot grid size, in number of images. Defaults to (5,4).
        - `moving_average_size` (int, optional): Moving average size for forecast smoothing. Defaults to 3.
        - `give_unscaled` (bool, optional): Unscale predictions to postprocess them. Defaults to False.
        - `scaler` (sklearn scaler, optional): sklearn Scaler object used for scaling outputs. Defaults to None.

    Returns:
        dict: results dictionary with the following keys:
            - `rmse_val_seq`: Sequence of RMSE values.
            - `rmse_val_tot`: Overall RMSE value.
    """

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("selected device: ", device)

    model.eval()
    model.to(device)
    # Prediction on the validation set:
    rmse_val_seq = []
    rmse_val_tot = 0
    num_validation_batches = len(dataloader)
    with torch.no_grad():
        for i,batch in enumerate(dataloader):
            seq, target = batch[0].to(device), batch[1].to(device)
            pred = model(seq)
            # Inversing the scale
            pred_unscaled = scaler.inverse_transform(pred.cpu()) if give_unscaled else pred.cpu()
            target_unscaled = scaler.inverse_transform(target.cpu()) if give_unscaled else target.cpu()
            if i == 0:
                pred_unscaled_plot = pred_unscaled
                target_unscaled_plot = target_unscaled
            # Errors
            err_seq = mean_squared_error(target_unscaled, pred_unscaled, squared=False, multioutput="raw_values")
            err_tot = mean_squared_error(target_unscaled, pred_unscaled, squared=False, multioutput="uniform_average")

            rmse_val_tot += err_tot
            rmse_val_seq.append(err_seq)

    rmse_val_tot /= num_validation_batches
    rmse_val_seq = np.mean(np.vstack([arr.reshape(1,-1) for arr in rmse_val_seq]), axis=0)
    print("Total RMSE of " + "unscaled " if give_unscaled else " " + "data: ", rmse_val_tot)
    print("Sequence validation RMSE of " + "unscaled " if give_unscaled else " " + "data is shown below: ")
    plt.figure(figsize=(15,7))
    plt.plot(rmse_val_seq)
    plt.xlabel(rmse_seq_x_label)
    plt.ylabel(rmse_seq_y_label)
    plt.title(rmse_seq_fig_title)
    plt.grid(True)
    #plt.show()
    if rmse_seq_file_name: plt.savefig(rmse_seq_file_name, dpi=600)

    num_rows, num_cols = [int(i) for i in timeplots_gridsize]
    num_plots = num_rows * num_cols

    plt.figure(figsize=(num_plots, num_plots))
    plt.suptitle(timeplots_fig_title)
    for i in range(num_plots):
        plt.subplot(num_rows,num_cols,i+1)
        plt.grid(True)
        plt.plot(target_unscaled_plot[i], "-b")
        plt.plot(moving_average(pred_unscaled_plot[i], moving_average_size), "-r")
        if i == 0:
            plt.legend(["target", "pred"])
        if i >= num_plots - num_cols:
            plt.xlabel(timeplots_x_label)
        if i%num_cols == 0:
            plt.ylabel(timeplots_y_label)
    if timeplots_file_name: plt.savefig(timeplots_file_name, dpi=600)
    return {"rmse_val_tot":rmse_val_tot, "rmse_val_seq":rmse_val_seq}




# Generating scalers and encoders out of data frames, given the column names
def generate_scaler(df, cols, scaler):
    """Generate scaler out of data frame, given the column names

    ### Args:
        `df` (Pandas DataFrame): DataFrame to be used
        `cols` (list): List of column names to be extracted
        `scaler` (sklearn scaler): The basic (unfit) scaler to be used, e.g. StandardScaler()

    ### Returns:
        The scaler, fit to the selected data
    """
    data = df[cols].to_numpy().astype(np.float32)
    scaler.fit(data)
    return scaler


def generate_encoder(df, cols):
    """Generate one-hot encoder out of a data frame, given column names

    Args:
        df (Pandas DataFrame): Pandas DataFrame to be used
        cols (list): List of column names to be extracted

    Returns:
        The one-hot encoder, fit to the selected data
    """
    ncod = OneHotEncoder()
    data = df[cols].to_numpy().astype(int)
    ncod.fit(data)
    return ncod



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