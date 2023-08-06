import sys
import os
import warnings
from pathlib import Path
import math
import json
from timeit import default_timer as timer
from datetime import datetime
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
from torch.nn.init import xavier_uniform_, zeros_, orthogonal_, calculate_gain
# Note: Custom random initializations should NOT be implemented without the gain values from calculate_gain.
from sklearn.metrics import r2_score
import numpy as np
import collections
from tqdm import tqdm

########################################################################################################################
# Global variables, functions, and classes
########################################################################################################################

# Set random seed
SEED = 42

def make_path(path:str):
    ''' Make a path if it doesn't exist.'''
    Path.mkdir(Path(path).parent, parents=True, exist_ok=True)
    return path


def autoname(name):
    """
    Genereate a unique name for a file, based on the current time and the given name.
    Gets the `name` as a string and adds the time stamp to the end of it before returning it.
    """
    return name + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


actdict_pytorch = {
    'relu':nn.ReLU, 'leakyrelu':nn.LeakyReLU, 'sigmoid':nn.Sigmoid, 'tanh':nn.Tanh, 'softmax':nn.Softmax, 'logsoftmax':nn.LogSoftmax,
    'softplus':nn.Softplus, 'softshrink':nn.Softshrink,'elu':nn.ELU, 'selu':nn.SELU, 'softsign':nn.Softsign, 'softmin':nn.Softmin, 'softmax2d':nn.Softmax2d}

lossdict_pytorch = {
    "mse":nn.MSELoss, "crossentropy":nn.CrossEntropyLoss, "binary_crossentropy":nn.BCELoss, "categorical_crossentropy":nn.CrossEntropyLoss, "nll":nn.NLLLoss, 
    "poisson":nn.PoissonNLLLoss, "kld":nn.KLDivLoss, "hinge":nn.HingeEmbeddingLoss, "l1":nn.L1Loss, "mae": nn.L1Loss, "l2":nn.MSELoss, "smoothl1":nn.SmoothL1Loss, 
    "bce_with_logits":nn.BCEWithLogitsLoss
}

optdict_pytorch = {'adam':optim.Adam, 'sgd':optim.SGD, 'rmsprop':optim.RMSprop}

convdict_pytorch = {'conv1d':nn.Conv1d, 'conv2d':nn.Conv2d, 'conv3d':nn.Conv3d}

pooldict_pytorch = {
    "maxpool1d": nn.MaxPool1d, "avgpool1d": nn.AvgPool1d, "adaptivemaxpool1d": nn.AdaptiveMaxPool1d, "adaptiveavgpool1d": nn.AdaptiveAvgPool1d,
    "maxpool2d": nn.MaxPool2d, "avgpool2d": nn.AvgPool2d, "adaptivemaxpool2d": nn.AdaptiveMaxPool2d, "adaptiveavgpool2d": nn.AdaptiveAvgPool2d,
    "maxpool3d": nn.MaxPool3d, "avgpool3d": nn.AvgPool3d, "adaptivemaxpool3d": nn.AdaptiveMaxPool3d, "adaptiveavgpool3d": nn.AdaptiveAvgPool3d}

normdict_pytorch = {
    "batchnorm1d": nn.BatchNorm1d, "batchnorm2d": nn.BatchNorm2d, "batchnorm3d": nn.BatchNorm3d, "instancenorm1d": nn.InstanceNorm1d, "instancenorm2d": nn.InstanceNorm2d, 
    "instancenorm3d": nn.InstanceNorm3d, "layernorm": nn.LayerNorm, "groupnorm": nn.GroupNorm, "localresponsenorm": nn.LocalResponseNorm,
}

dropoutdict_pytorch = {"dropout1d": nn.Dropout1d, "dropout2d": nn.Dropout2d, "dropout3d": nn.Dropout3d}



def _calc_image_size(size_in:int, kernel_size:int, padding:int, stride:int, dilation:int):
    if padding == 'same':
        return size_in
    else:
        if padding == 'valid':
            padding = 0
        if isinstance(size_in, (list, tuple)):
            if isinstance(padding, int): padding = [padding]*len(size_in)
            if isinstance(kernel_size, int): kernel_size = [kernel_size]*len(size_in)
            if isinstance(stride, int): stride = [stride]*len(size_in)
            if isinstance(dilation, int): dilation = [dilation]*len(size_in)
            return [math.floor((size_in[i] + 2*padding[i] - dilation[i]*(kernel_size[i]-1) - 1)/stride[i] + 1) for i in range(len(size_in))]
        else:
            assert isinstance(size_in, int), "size_in must be an integer or a list/tuple of integers."
            assert isinstance(padding, int), "padding must be an integer or a list/tuple of integers."
            assert isinstance(kernel_size, int), "kernel_size must be an integer or a list/tuple of integers."
            assert isinstance(stride, int), "stride must be an integer or a list/tuple of integers."
            assert isinstance(dilation, int), "dilation must be an integer or a list/tuple of integers."
            return math.floor((size_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)


def _generate_geometric_array(init, count, direction, powers_of_two=True):
    """Generate array filled with incrementally doubling/halving values, optionally with powers of two.

    ### Args:
        `init` (int): The first value to begin.
        `count` (int): Number of elements to generate.
        `direction` (str): Direction of the array. Can be either 'up' or 'down', i.e. increasing or decreasing.
        `powers_of_two` (bool, optional): Generate numbers that are powers of two. Defaults to True.

    ### Returns:
        list: List containing elements
    """
    lst = []
    old = int(2**math.ceil(math.log2(init))) if powers_of_two else init
    new = old
    for _ in range(count):
        lst.append(new)
        old = new
        new = (old * 2) if direction == 'up' else (old // 2)
    return lst


def _generate_array_for_hparam(
    hparam, count_if_not_list:int, 
    hparam_name:str='parameter', count_if_not_list_name:str='its count',
    check_auto:bool=False, init_for_auto:int=2, powers_of_two_if_auto:bool=True,
    direction_if_auto:str=None):
    """Generate array for a hyperparameter, regardless of if it is a list or not. This function is for use in APIs
    that generate models with hyperparameters as inputs, which can be lists, a single item, or "auto".
    Examples include width of a neural network's hidden layers, channels of conv layers, etc.
    For these hyperparameters, the user is typically free to specify an array-like, a single item to be repeated,
    or "auto" for automatic calculation of the parameter.
    This function is meant to be used in the body of the code of class constructors and other functions in the API.

    ### Args:
        `hparam` (var): A specific hyperparameter, e.g., input by user to your network's constructor.
        `count_if_not_list` (int): Number of elements to generate if `hparam` is not an array-like.
        `hparam_name` (str, optional): Name of the hyperparameter. Defaults to 'parameter'.
        `count_if_not_list_name` (str, optional): Name of the "count" that must be provided. Defaults to 'its count'.
        `check_auto` (bool, optional): Check for the "auto" case. Defaults to False.
        `init_for_auto` (int, optional): Initial value in case of "auto". Defaults to 2.
        `powers_of_two_if_auto` (bool, optional): Generate powers of two in case of "auto". Defaults to True.
        `direction_if_auto` (str, optional): Direction of geometric increment in case of "auto". Defaults to None.
        This can be "up" or "down". If check_for_auto is True, then this argument must be specified.

    ### Returns:
        list: List containing elements
    """
    assert count_if_not_list is not None, \
        "Since %s may not be a list/tuple, %s must always be specified."%(hparam_name, count_if_not_list_name)
    if isinstance(hparam, (list,tuple)) and len(hparam) == count_if_not_list:
        lst = hparam
    elif hparam == "auto" and check_auto:
        assert init_for_auto is not None, \
            "If %s is 'auto', then %s must be specified."%(hparam_name, "init_for_auto")
        assert direction_if_auto in ['up','down'], \
            "If %s is 'auto', then %s must be specified as 'up' or 'down'."%(hparam_name, "direction_if_auto")
        lst = _generate_geometric_array(init_for_auto, count_if_not_list, direction_if_auto, powers_of_two_if_auto)
    else:
        lst = [hparam]*count_if_not_list
    return lst




def generate_sample_batch(model):
    x = np.random.rand(*model.batch_input_shape).astype(np.float32)
    y = np.random.rand(*model.batch_output_shape).astype(np.float32)
    if model.hparams['loss_function'] in ['CrossEntroplyLoss', 'NLLLoss']:
        y = np.argmax(y, axis=1)
    return (x,y)


def test_pytorch_model_class(model_class):
    print("Constructing model...\n")
    model = model_class()
    print("Summary of model:")
    print(model)
    print("\nGenerating random dataset...\n")
    (x_train, y_train) = generate_sample_batch(model)
    (x_val, y_val) = generate_sample_batch(model)
    x_train_t = torch.Tensor(x_train)
    y_train_t = torch.Tensor(y_train)
    x_val_t = torch.Tensor(x_val)
    y_val_t = torch.Tensor(y_val)
    trainset = TensorDataset(x_train_t, y_train_t)
    validset = TensorDataset(x_val_t, y_val_t)
    dataset = (trainset, validset)
    print("\nTraining model...\n")
    model.train_model(dataset, verbose=1)
    print("\nEvaluating model...\n")
    print("Done.")
    

def _update_metrics_for_batch(
    predictions:torch.Tensor, targets:torch.Tensor, loss_str:str, classification:bool, regression:bool, 
    verbose:int, batch_num:int, epoch:int, metric:float, num_logits:int):
    if loss_str == "BCELoss":
        # Output layer already includes sigmoid.
        class_predictions = (predictions > 0.5).int()
    elif loss_str == "BCEWithLogitsLoss":
        # Output layer does not include sigmoid. Sigmoid is a part of the loss function.
        class_predictions = (torch.sigmoid(predictions) > 0.5).int()
    elif loss_str in ["NLLLoss", "CrossEntropyLoss"]:
        # nll -> Output layer already includes log_softmax.
        # crossentropy -> Output layer has no log_softmax. It's implemented as a part of the loss function.
        class_predictions = torch.argmax(predictions, dim=1)
        if predictions.shape == targets.shape: # Targets are one-hot encoded probabilities
            target_predictions = torch.argmax(targets, dim=1)
        else: # Targets are class indices
            target_predictions = targets

    if classification:
        if verbose>=2 and batch_num==0 and epoch ==0: 
            print("Shape of model outputs:     ", predictions.shape)
            print("Shape of class predictions: ", class_predictions.shape)
            print("Shape of targets:           ", targets.shape)
        # Calculate accuracy
        correct = (class_predictions == target_predictions).int().sum().item()
        num_logits += target_predictions.numel()
        metric += correct
        if verbose==3 and epoch==0: 
            print("Number of correct answers (this batch - total): %10d - %10d"%(correct, metric))
        # Calculate F1 score
        # f1 = f1_score(targets.cpu().numpy(), class_predictions.cpu().numpy(), average="macro")
    elif regression:
        if verbose==3 and batch_num==0 and epoch==0: 
            print("Shape of predictions: ", predictions.shape)
            print("Shape of targets:     ", targets.shape)
        # Calculate r2_score
        metric += r2_score(targets.cpu().numpy(), predictions.cpu().numpy())
    
    return metric, num_logits



def _test_shapes(predictions:torch.Tensor, targets:torch.Tensor, classification:bool):
    if classification:
        assert predictions.shape[0] == targets.shape[0], "Batch size of targets and predictions must be the same.\n"+\
            "Target shape: %s, Prediction shape: %s\n"%(str(targets.shape), str(predictions.shape))
        if len(predictions.shape) == 1:
            assert targets.shape == predictions.shape, "For 1D predictions, the targets must also be 1D.\n"+\
                "Predictions shape: %s, Targets shape: %s\n"%(str(predictions.shape), str(targets.shape))
        if len(predictions.shape) == 2:
            assert len(targets.shape)==1 or targets.shape == predictions.shape, \
                "For 2D predictions, the targets must be 1D class indices are 2D [N x K] one-hot encoded array, with the same shape as the predictions.\n"+\
                "Predictions shape: %s, Targets shape: %s\n"%(str(predictions.shape), str(targets.shape))
        if len(predictions.shape) > 2:
            assert len(predictions.shape)==len(targets.shape) or len(predictions.shape)==len(targets.shape)+1, \
                "Target dimensionality must be equal to or one less than the prediction dimensionality.\n"+\
                "Target shape: %s, Prediction shape: %s\n"%(str(targets.shape), str(predictions.shape))+\
                "If targets are class indices, they must be of shape (N,), or (N, d1, ..., dm). "+\
                "Otherwise, they must be (N, K) or (N, K, d1, ..., dm) arrays of one-hot encoded probabilities. "+\
                "Predictions must in any case be (N, K) or (N, K, d1, ..., dm).\n"+\
                "N is batch size, K is number of classes and d1 to dm are other dimensionalities of classification, if any."
            if len(predictions.shape) == len(targets.shape):
                assert predictions.shape == targets.shape, "If predictions and targets have the same dimensionality, they must be the same shape.\n"+\
                    "Target shape: %s, Prediction shape: %s\n"%(str(targets.shape), str(predictions.shape))
            else:
                assert predictions.shape[2:] == targets.shape[1:], \
                    "If predictions have shape (N, K, d1, ..., dm) then targets must either have the same shape, or (N, d1, ..., dm).\n"+\
                    "Target shape: %s, Prediction shape: %s\n"%(str(targets.shape), str(predictions.shape))
    else:
        assert predictions.shape == targets.shape, \
            "Target shape must be equal to the prediction shape.\n"+\
            "Target shape: %s, Prediction shape: %s\n"%(str(targets.shape), str(predictions.shape))




def _calculate_epoch_loss_and_metrics(
    cumulative_epoch_loss:float, num_batches:int, verbose:int, epoch:int, 
    hist_loss:dict, hist_metric:dict, display_metrics:bool, cumulative_metric:float, metric_denominator:int):
    # Calculate training epoch loss
    epoch_loss = cumulative_epoch_loss / num_batches
    if verbose==3 and epoch==0: print("Epoch loss (training): %.5f"%epoch_loss)
    if hist_loss is not None: hist_loss.append(epoch_loss)
    # Calculate training epoch metric (accuracy or r2-score)
    if display_metrics:
        epoch_metric = cumulative_metric / metric_denominator
        if verbose==3 and epoch==0: print("Epoch metric: %.5f"%epoch_metric)
        if hist_metric is not None: hist_metric.append(epoch_metric)
    return epoch_loss, epoch_metric, hist_loss, hist_metric



def save_pytorch_model(model:torch.nn.Module, saveto:str, dataloader, script_before_save:bool=True, verbose:int=1):
    try:
        if verbose > 0: print("Saving model...")
        if script_before_save:
            example,_ = next(iter(dataloader))
            example = example[0,:].unsqueeze(0)
            model.cpu()
            with torch.no_grad():
                traced = torch.jit.trace(model, example)
                traced.save(saveto)
        else:
            with torch.no_grad():
                torch.save(model, saveto)
    except Exception as e:
        if verbose > 0:
            print(e)
            print("Failed to save the model.")
    if verbose > 0: print("Done Saving.")
    
    
    

def train_pytorch_model(model, dataset, batch_size:int, loss_str:str, optimizer_str:str, optimizer_params:dict=None, loss_function_params:dict=None, learnrate:float=0.001, 
    learnrate_decay_gamma:float=None, epochs:int=10, validation_patience:int=10000, validation_data:float=0.1, verbose:int=1, script_before_save:bool=True, saveto:str=None, 
    num_workers=0):
    """Train a Pytorch model, given some hyperparameters.

    ### Args:
        - `model` (`torch.nn`): A torch.nn model
        - `dataset` (`torch.utils.data.Dataset`): Dataset object to be used
        - `batch_size` (int): Batch size
        - `loss_str` (str): Loss function to be used. Examples: "CrossEntropyLoss", "BCELoss", "BCEWithLogitsLoss", "MSELoss", etc.
        - `optimizer_str` (str): Optimizer to be used. Examples: "Adam", "SGD", "RMSprop", etc.
        - `optimizer_params` (dict, optional): Parameters for the optimizer.
        - `loss_function_params` (dict, optional): Parameters for the loss function.
        - `learnrate` (float, optional): Learning rate. Defaults to 0.001.
        - `learnrate_decay_gamma` (float, optional): Learning rate exponential decay rate. Defaults to None.
        - `epochs` (int, optional): Number of epochs. Defaults to 10.
        - `validation_patience` (int, optional): Number of epochs to wait before stopping training. Defaults to 10000.
        - `validation_data` (float, optional): Fraction of the dataset to be used for validation. Defaults to 0.1.
        - `verbose` (int, optional): Logging the progress. Defaults to 1. 0 prints nothing, 2 prints everything.
        - `script_before_save` (bool, optional): Use TorchScript for serializing the model. Defaults to True.
        - `saveto` (str, optional): Save PyTorch model in path. Defaults to None.
        - `num_workers` (int, optional): Number of workers for the dataloader. Defaults to 0.
        
    ### Returns:
        - `model`: Trained PyTorch-compatible model
        - `history`: PyTorch model history dictionary, containing the following keys:
            - `training_loss`: List containing training loss values of epochs.
            - `validation_loss`: List containing validation loss values of epochs.
            - `learning_rate`: List containing learning rate values of epochs.
            - `training_metrics`: List containing training metric values of epochs.
            - `validation_metrics`: List containing validation metric values of epochs.
    """
    # Initialize necessary lists
    hist_training_loss = []
    hist_validation_loss = []
    hist_learning_rate = []
    hist_trn_metric = []
    hist_val_metric = []
    
    # Empty CUDA cache
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # Check if validation data is provided or not, and calculate number of training and validation data
    if isinstance(dataset, (list, tuple)):
        assert len(dataset)==2, "If dataset is a tuple, it must have only two elements, the training dataset and the validation dataset."
        trainset, valset = dataset
        num_val_data = int(len(valset))
        num_train_data = int(len(trainset))
        num_all_data = num_train_data + num_val_data
    else:
        num_all_data = len(dataset)
        num_val_data = int(validation_data*num_all_data)
        num_train_data = num_all_data - num_val_data
        (trainset, valset) = random_split(dataset, (num_train_data, num_val_data), generator=torch.Generator().manual_seed(SEED))

    if verbose > 0:
        print("Total number of data points:      %d"%num_all_data)
        print("Number of training data points:   %d"%num_train_data)
        print("Number of validation data points: %d"%num_val_data)
    
    # Generate training and validation dataloaders    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    if verbose > 0:
        print("Number of training batches:    %d"%len(trainloader))
        print("Number of validation batches:  %d"%len(validloader))
        print("Batch size:                    %d"%batch_size)
        for x,y in trainloader:
            print("Shape of training input from the dataloader:  ", x.shape)
            print("Shape of training output from the dataloader: ", y.shape)
            break
        for x,y in validloader:
            print("Shape of validation input from the dataloader:  ", x.shape)
            print("Shape of validation output from the dataloader: ", y.shape)
            break
    
    # Select the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if verbose > 0: print("Selected device: ", device)
    model.to(device)
    
    # Instantiate the loss function
    loss_func = getattr(nn, loss_str)
    criterion = loss_func(**loss_function_params) if loss_function_params else loss_func()
        
    # Instantiate the optimizer
    optimizer_func = getattr(optim, optimizer_str)
    optimizer = optimizer_func(model.parameters(), lr=learnrate, **optimizer_params) if optimizer_params else optimizer_func(model.parameters(), lr=learnrate)
    
    # Defining learning rate scheduling
    if learnrate_decay_gamma:
        if verbose > 0: print("The learning rate has an exponential decay rate of %.5f."%learnrate_decay_gamma)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=learnrate_decay_gamma)
        lr_sch = True
    else:
        lr_sch = False
    
    # Find out if we will display any metric along with the loss.
    display_metrics = True
    classification = False
    regression = False
    if loss_str in ["BCELoss", "BCEWithLogitsLoss", "CrossEntropyLoss", "NLLLoss", "PoissonNLLLoss", "GaussianNLLLoss"]:
        classification = True
        regression = False
        trn_metric_name = "Acc"
        val_metric_name = "Val Acc"
    elif loss_str in ["MSELoss", "L1Loss", "L2Loss", "HuberLoss", "SmoothL1Loss"]:
        classification = False
        regression = True
        trn_metric_name = "R2"
        val_metric_name = "Val R2"
    else:
        classification = False
        regression = False
        display_metrics = False
    if verbose > 0:
        if classification: print("Classification problem detected. We will look at accuracies.")
        elif regression: print("Regression problem detected. We will look at R2 scores.")
        else: print("We have detected neither classification nor regression problem. No metric will be displayed other than loss.")
    
                    
    # Calculating number of training and validation batches
    num_training_batches = len(trainloader)
    num_validation_batches = len(validloader)
    
    # Preparing progress bar
    progress_bar_size = 40
    ch = "█"
    intvl = num_training_batches/progress_bar_size;
    valtol = validation_patience if validation_patience else 100000000
    minvalerr = 10000000000.0
    badvalcount = 0
    
    # Commencing training loop
    tStart = timer()
    loop = tqdm(range(epochs), desc='Training Progress', ncols=100) if verbose==1 else range(epochs)
    for epoch in loop:
        
        # Initialize per-epoch variables
        tEpochStart = timer()
        epoch_loss_training = 0.0
        epoch_loss_validation = 0.0
        newnum = 0
        oldnum = 0
        trn_metric = 0.0
        val_metric = 0.0
        num_train_logits = 0
        num_val_logits = 0
    
        if verbose>=2 and epoch > 0: print("Epoch %3d/%3d ["%(epoch+1, epochs), end="")
        if verbose==3 and epoch ==0: print("First epoch ...")
        
        ##########################################################################
        # Training
        if verbose==3 and epoch==0: print("\nTraining phase ...")
        model.train()
        for i, data in enumerate(trainloader):
            # Fetch data
            seqs, targets = data[0].to(device), data[1].to(device)
            # Forward propagation
            predictions = model(seqs)
            # Test shapes
            _test_shapes(predictions, targets, classification)
            # Loss calculation and accumulation
            loss = criterion(predictions, targets)
            epoch_loss_training += loss.item()
            # Backpropagation and optimizer update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Metrics calculation
            if display_metrics:
                with torch.no_grad():
                    trn_metric, num_train_logits = _update_metrics_for_batch(
                        predictions, targets, loss_str, classification, regression, verbose, i, epoch, trn_metric, num_train_logits)
                    
            # Visualization of progressbar within the batch
            if verbose>=2 and epoch > 0:
                newnum = int(i/intvl)
                if newnum > oldnum:
                    print((newnum-oldnum)*ch, end="")
                    oldnum = newnum 
        
        # Update learning rate if necessary
        if lr_sch: scheduler.step()
        
        # Calculate epoch loss and metrics
        epoch_loss_training, trn_metric, hist_training_loss, hist_trn_metric = _calculate_epoch_loss_and_metrics(epoch_loss_training, num_training_batches, verbose, epoch, 
            hist_training_loss, hist_trn_metric, display_metrics, trn_metric, (num_train_logits if classification else num_training_batches))
            
        if verbose>=2 and epoch > 0: print("] ", end="")
        
        ##########################################################################
        # Validation
        if verbose==3 and epoch==0: print("\nValidation phase ...")
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(validloader):
                seqs, targets = data[0].to(device), data[1].to(device)
                predictions = model(seqs)
                loss = criterion(predictions, targets)
                epoch_loss_validation += loss.item()
                # Do prediction for metrics
                if display_metrics:
                    val_metric, num_val_logits = _update_metrics_for_batch(
                        predictions, targets, loss_str, classification, regression, verbose, i, epoch, val_metric, num_val_logits)
        # Calculate epoch loss and metrics
        epoch_loss_validation, val_metric, hist_validation_loss, hist_val_metric = _calculate_epoch_loss_and_metrics(epoch_loss_validation, num_validation_batches, verbose, epoch, 
            hist_validation_loss, hist_val_metric, display_metrics, val_metric, (num_val_logits if classification else num_validation_batches))
        
        # Log the learning rate, if there is any scheduling.
        if lr_sch: hist_learning_rate.append(scheduler.get_last_lr()[0])
        else: hist_learning_rate.append(learnrate)
        
        ##########################################################################
        # Post Processing Training Loop            
        tEpochEnd = timer()
        if verbose>=2:
            if display_metrics:
                print("Loss: %5.4f |Val Loss: %5.4f |%s: %5.4f |%s: %5.4f | %6.3f s" % (
                    epoch_loss_training, epoch_loss_validation, trn_metric_name, trn_metric,val_metric_name, val_metric, tEpochEnd-tEpochStart))
            else:
                print("Loss: %5.4f |Val Loss: %5.4f | %6.3f s" % (epoch_loss_training, epoch_loss_validation, tEpochEnd-tEpochStart))
        
        # Checking for early stopping
        if epoch_loss_validation < minvalerr:
            minvalerr = epoch_loss_validation
            badvalcount = 0
        else:
            badvalcount += 1
            if badvalcount > valtol:
                if verbose > 0:
                    print("Validation loss not improved for more than %d epochs."%badvalcount)
                    print("Early stopping criterion with validation loss has been reached. " + 
                        "Stopping training at %d epochs..."%epoch)
                break
    # End for loop
    model.eval()
    ##########################################################################
    # Epilogue
    tFinish = timer()
    if verbose > 0:        
        print('Finished Training.')
        print("Training process took %.2f seconds."%(tFinish-tStart))
    if saveto:
       save_pytorch_model(model, saveto, trainloader, script_before_save, verbose)
    # Clear CUDA cache    
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    # Generate output dictionaries
    history = {
        'training_loss':hist_training_loss, 
        'validation_loss':hist_validation_loss, 
        'learning_rate':hist_learning_rate}
    if display_metrics:
        history["training_metrics"] = hist_trn_metric
        history["validation_metrics"] = hist_val_metric
    if verbose > 0: print("Done training.")
    
    return history




def evaluate_pytorch_model(model, dataset, loss_str:str, loss_function_params:dict=None, batch_size:int=16, device_str:str="cuda", verbose:bool=True, num_workers:int=0):
    """
    Evaluates a PyTorch model on a dataset.
    
    ### Parameters
    
    `model` (`torch.nn.Module`): The model to evaluate.
    `dataset` (`torch.utils.data.Dataset`): The dataset to evaluate the model on.
    `loss_str` (str): The loss function to use when evaluating the model.
    `loss_function_params` (dict, optional) : Parameters to pass to the loss function.
    `batch_size` (int, optional) : The batch size to use when evaluating the model. Defaults to 16.
    `device_str` (str, optional) : The device to use when evaluating the model. Defaults to "cuda".
    `verbose` (bool, optional) : Whether to print out the evaluation metrics. Defaults to True.
    `num_workers` (int, optional) : The number of workers to use when making dataloader. Defaults to 0.
    
    
    ### Returns
    
    A dictionary containing the evaluation metrics, including "loss" and "metrics" in case any metric is available.
    """
    # Clear CUDA cache
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    if verbose: print("Preparing data...")
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    num_batches = len(testloader)
        
    if "cuda" in device_str:
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    print("selected device: ", device)
    model.eval()
    model.to(device)
    
    loss_func = getattr(nn, loss_str)
    criterion = loss_func(**loss_function_params) if loss_function_params else loss_func()
    
    display_metrics = True
    classification = False
    regression = False
    if loss_str in ["BCELoss", "BCEWithLogitsLoss", "CrossEntropyLoss", "NLLLoss", "PoissonNLLLoss", "GaussianNLLLoss"]:
        classification = True
        regression = False
        metric_name = "Accuracy"
    elif loss_str in ["MSELoss", "L1Loss", "L2Loss", "HuberLoss", "SmoothL1Loss"]:
        classification = False
        regression = True
        metric_name = "R2-Score"
    else:
        classification = False
        regression = False
        display_metrics = False
        
    progress_bar_size = 40
    ch = "█"
    intvl = num_batches/progress_bar_size;
    if verbose: print("Evaluating model...")
    model.eval()
    newnum = 0
    oldnum = 0
    totloss = 0.0
    if verbose: print("[", end="")
    val_metric = 0.0
    num_val_logits = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, targets = data[0].to(device), data[1].to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            totloss += loss.item()
            
            # Do prediction for metrics
            if display_metrics:
                val_metric, num_val_logits = _update_metrics_for_batch(
                        predictions, targets, loss_str, classification, regression, verbose, i, 0, val_metric, num_val_logits)
                    
            # Visualization of progressbar
            if verbose:
                newnum = int(i/intvl)
                if newnum > oldnum:
                    print((newnum-oldnum)*ch, end="")
                    oldnum = newnum 
    
    totloss, val_metric, _, _ = _calculate_epoch_loss_and_metrics(
            totloss, num_batches, verbose, 0, None, None, display_metrics, val_metric, (num_val_logits if classification else num_batches))
        
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if verbose: print("] ", end="") 
    if verbose:
        if display_metrics:
            print("Loss: %5.4f | %s: %5.4f" % (totloss, metric_name, val_metric))
        else:
            print("Loss: %5.4f" % totloss)
            
    if verbose: print("Done.")
    d = {"loss":totloss}
    if display_metrics:
        d["metrics"] = val_metric
    return d
            
            

################################
def predict_pytorch_model(model, dataset, loss_str:str, batch_size:int=16, device_str:str="cuda", return_in_batches:bool=True, return_inputs:bool=False, 
                          return_raw_predictions:bool=False, verbose:bool=True, num_workers:int=0):
    """
    Predicts the output of a pytorch model on a given dataset.

    ### Args:
        - `model` (`torch.nn.Module`): The PyTorch model to use.
        - `dataset` (`torch.utils.data.Dataset`): Dataset containing the input data
        - `loss_str` (str): Loss function used when training. Used only for determining whether a classification or a regression model is used.
        - `batch_size` (int, optional): Batch size to use when evaluating the model. Defaults to 16.
        - `device_str` (str, optional): Device to use when performing inference. Defaults to "cuda".
        - `return_in_batches` (bool, optional): Whether the predictions should be batch-separated. Defaults to True.
        - `return_inputs` (bool, optional): Whether the output should include the inputs as well. Defaults to False.
        - `return_raw_predictions` (bool, optional): Whether raw predictions should also be returned. Defaults to False.
        - `verbose` (bool, optional): Verbosity of the function. Defaults to True.
        - `num_workers` (int, optional): Number of workers to use when making dataloader. Defaults to 0.

    ### Returns:
        List: A List containing the output predictions, and optionally, the inputs and raw predictions.
        
    ### Notes:
        - If `return_in_batches` is True, the output will be a list of lists. output[i] contains the i'th batch.
        - If `return_inputs` is true, the first element of the output information will be the inputs.
        - If `return_raw_predictions` is true, the second element of the output information will be the raw predictions.
            Please note that this is only meaningful for classification problems. Otherwise, predictions will only include raw predictions. For classification problems, 
            if this setting is True, the third element of the output information will be the class predictions.
        - "output information" here is a list containing [input, raw_predictions, class_predictions].
            For non-classification problems, "output information" will only contain [input, raw_predictions].
            If `return_inputs` is False, the first element of the output information will be omitted; [raw_predictions].
            If `return_in_batches` is True, the output will be a list of "output information" for every batch.
            Otherwise, the output will be one "output information" for the whole dataset.
    """
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if verbose: print("Preparing data...")
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    num_batches = len(testloader)
    if "cuda" in device_str:
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    if verbose: print("selected device: ", device)
    model.to(device)
    if loss_str in ["BCELoss", "BCEWithLogitsLoss", "CrossEntropyLoss", "NLLLoss", "PoissonNLLLoss", "GaussianNLLLoss"]:
        classification = True
    else:
        classification = False
    output_list = []
    progress_bar_size = 40
    ch = "█"
    intvl = num_batches/progress_bar_size;
    if verbose: print("Performing Prediction...")
    model.eval()
    newnum = 0
    oldnum = 0
    if verbose: print("[", end="")
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs = data[0].to(device)
            predictions = model(inputs)
            
            # Do prediction
            if classification:
                if loss_str == "BCELoss":
                    # Output layer already includes sigmoid.
                    class_predictions = (predictions > 0.5).float()
                elif loss_str == "BCEWithLogitsLoss":
                    # Output layer does not include sigmoid. Sigmoid is a part of the loss function.
                    class_predictions = (torch.sigmoid(predictions) > 0.5).float()
                else:
                    class_predictions = torch.argmax(predictions, dim=1).float()
            
            # Add batch predictions to output dataset
            obatch = []
            if return_inputs:
                obatch.append(inputs.cpu().numpy())
            if classification:
                if return_raw_predictions:
                    obatch.append(predictions.cpu().numpy())
                obatch.append(class_predictions.cpu().numpy())
            else:
                obatch.append(predictions.cpu().numpy())
                
            if return_in_batches:
                output_list.append(obatch)
            elif i==0:
                output_array = obatch
            else:
                for j in range(len(obatch)):
                    output_array[j] = np.append(output_array[j], obatch[j], axis=0)
            
            # Visualization of progressbar
            if verbose:
                newnum = int(i/intvl)
                if newnum > oldnum:
                    print((newnum-oldnum)*ch, end="")
                    oldnum = newnum 
        
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if verbose: print("] ")
    if return_in_batches:
        return output_list
    else:
        return output_array
        

########################################################################################################################
########################################################################################################################
########################################################################################################################


class Dense_Block(nn.Module):
    def __init__(self, input_size:int, output_size:int=None, activation:str=None, activation_params:dict=None, norm_layer_type:str=None, norm_layer_position:str='before', 
                 norm_layer_params:dict=None, dropout:float=None):
        """Dense (fully connected) block containing one linear layer, followed optionally by a normalization layer, an activation function and a Dropout layer.

        ### Args:
            - `input_size` (int): Number of input features.
            - `output_size` (int, optional): Number of output features. Defaults to None, in which case it will be input_size.
            - `activation` (str, optional): Activation function in string form. Defaults to None. Examples: 'ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid', etc.
            - `activation_params` (dict, optional): kwargs to pass to the activation function constructor. Defaults to None.
            - `norm_layer_type` (str, optional): Type of normalization layer. Defaults to None. Examples: 'BatchNorm1d', 'LayerNorm', etc.
            - `norm_layer_position` (str, optional): Position of norm layer relative to activation. Defaults to 'before'. Alternative is 'after'.
            - `norm_layer_params` (dict, optional): kwargs to pass to the norm layer constructor. Defaults to None.
            - `dropout` (float, optional): Dropout rate at the end. Defaults to None. Must be a float between 0 and 1.
            
        ### Returns:
        An `nn.Module` object.
        """
        super(Dense_Block, self).__init__()
        if not output_size: output_size = input_size
        self._activation_module = getattr(nn, activation) if activation else None
        self._norm_layer_module = getattr(nn, norm_layer_type) if norm_layer_type else None
        self._dropout_module = nn.Dropout if dropout else None
        layers_vec = []
        layers_vec.append(nn.Linear(input_size, output_size))
        if norm_layer_type and norm_layer_position=='before': 
            if norm_layer_params: layers_vec.append(self._norm_layer_module(output_size, **norm_layer_params))
            else: layers_vec.append(self._norm_layer_module(output_size))
        if activation: 
            if activation_params: layers_vec.append(self._activation_module(**activation_params))
            else: layers_vec.append(self._activation_module())
        if norm_layer_type and norm_layer_position=='after': 
            if norm_layer_params: layers_vec.append(self._norm_layer_module(output_size, **norm_layer_params))
            else: layers_vec.append(self._norm_layer_module(output_size))
        if dropout: layers_vec.append(self._dropout_module(dropout))
        self.net = nn.Sequential(*layers_vec)
    
    def forward(self, x):
        return self.net(x)



class Conv_Block(nn.Module):
    def __init__(self, in_channels:int, out_channels:int=None, conv_dim:int=1, input_image:list=[30], conv_kernel_size=3, conv_padding='valid', conv_stride=1, conv_dilation=1, 
                 conv_params:dict=None, conv_activation:str='ReLU', conv_activation_params:dict=None, norm_layer_position:str=None, norm_layer_type:str=None, 
                 norm_layer_params:dict=None, pool_type:str=None, pool_kernel_size=2, pool_padding:int=0, pool_stride=1, pool_dilation=1, pool_params:dict=None, 
                 adaptive_pool_output_size=None, dropout:float=None, min_image_dim:int=1):
        """Convolutional block, containing one convolution layer, followed optionally by a normalization layer,
        an activation layer, a pooling layer and a dropout layer. The convolution layer is mandatory, but the other ones
        are optional. The convolution layer can be 1D, 2D or 3D. The normalization layer can be any such layer defined
        in PyTorch. The activation layer can also be anything, and the pooling layer can be any of the pooling layers
        defined for PyTorch. The dropout layer is a standard dropout layer. The dimension of any existing batch-norm and
        dropout layer will match the dimension of the convolution layer. For Conv1d, BatchNorm1d and Dropout1d will be
        used, if desired.
        
        This module is meant to be used as a building block for arger modules available here.

        ### Args:
        
        - `in_channels` (int): Channels of input image.
        - `out_channels` (int, optional): Number of convolution filters. Defaults to the input channels size.
        - `conv_dim` (int, optional): Dimension of the convolution. Defaults to 1. 1 means Conv1d, 2 means Conv2d etc.
        - `input_image` (list, optional): Size of the input image. Defaults to [30]. This must be a list/tuple of integers, with legth equal to `conv_dim`.
        - `conv_kernel_size` (int, optional): Convolution kernel size. Defaults to 3. It is strongly recommended to provide a list of integers, with length equal to `conv_dim`.
        - `conv_padding` (str, optional): Convolution padding. Defaults to 'same'. Arrays are recommended over integers.
        - `conv_stride` (int, optional): Convolution stride. Defaults to 1.
        - `conv_dilation` (int, optional): Convolution dilation. Defaults to 1.
        - `conv_params` (dict, optional): Additional dictionary of kwargs for Conv?d module. Defaults to None.
        - `conv_activation` (str, optional): String representing activation function. Defaults to 'ReLU'. Examples: 'LeakyReLU', 'Sigmoid', 'Tanh' etc.
        - `conv_activation_params` (dict, optional): kwargs dictionary for activation function. Defaults to None.
        - `norm_layer_position` (str, optional): Position of the normalization layer relative to activation. Defaults to None. It should be 'before' or 'after' or None.
        - `norm_layer_type` (str, optional): Type of the normalization layer. Defaults to None. Examples: 'BatchNorm', 'LayerNorm', etc.
        - `norm_layer_params` (dict, optional): kwargs dictionary for normalization layer. Defaults to None.
        - `pool_type` (str, optional): Type of pooling layer, if any. Defaults to None. For example, 'Max', 'Avg', 'AdaptiveMax', 'AdaptiveAvg' etc.
        - `pool_kernel_size` (int, optional): Pooling kernel size. Defaults to 2. Arrays are recommended over integers.
        - `pool_padding` (int, optional): Padding for pooling layer. Defaults to 0. 'same' is NOT an option here.
        - `pool_stride` (int, optional): Pooling stride. Defaults to 1.
        - `pool_dilation` (int, optional): Pooling dilation. Defaults to 1.
        - `pool_params` (dict, optional): kwargs dictionary for pooling layer module. Defaults to None.
        - `adaptive_pool_output_size` (list, optional): Output size for adaptive pooling, if any. Defaults to None.
        - `dropout` (float, optional): Dropout rate, if any. Defaults to None. for Conv?d, Dropout?d is used.
        - `min_image_dim` (int, optional): Minimum image dimension. Defaults to 1. This is used for preventing the image dimension from becoming too small. 
            It can automatically adjust padding and stride for convolution and pooling layers to keep the image dimensions larger than this argument.
        
        ### Returns:
        A `torch.nn.Module` instance representing a single convolutional block.
        
        ### Attributes:
        `self.output_image` (list): Size of the output image.
        `self.net` (torch.nn.Module): The actual network, a `torch.nn.Sequential` instance.
        """
        super(Conv_Block, self).__init__()
        # Input channels check
        assert isinstance(in_channels,int) and in_channels>0, "`in_channels` must be a positive integer, not {} which has type {}.".format(in_channels, str(type(in_channels)))
        self._in_channels = in_channels
        # Output channels check
        if isinstance(out_channels,int) and out_channels > 0:
            self._out_channels = out_channels
        else:
            warnings.warn("Invalid value out_channels={}. Using value {} equal to in_channels.".format(out_channels,in_channels), UserWarning)
            self._out_channels = in_channels
        # Convolution dimension check    
        assert isinstance(conv_dim, int) and conv_dim in [1,2,3], "`conv_dim` must be an integer among [1,2,3], not {} which has type {}.".format(conv_dim, str(type(conv_dim)))
        self._conv_dim = conv_dim
        # Determine convolution module used
        self._conv_module = convdict_pytorch["conv{}d".format(self._conv_dim)]
        # Iniput imager size check
        assert isinstance(input_image, (list,tuple)) and len(input_image)==self._conv_dim, \
            "`input_image` must be a list or tuple of length equal to `conv_dim`, not {} which has type {}.".format(input_image, str(type(input_image)))
        self._input_image = input_image
        # Store convolution parameters as class attributes
        self._conv_kernel_size = conv_kernel_size
        self._conv_padding = conv_padding
        self._conv_stride = conv_stride
        self._conv_dilation = conv_dilation
        self._conv_params = conv_params
        self._conv_activation = conv_activation
        # Check activation function and module
        self._conv_activation_module = getattr(nn, self._conv_activation) if self._conv_activation else None
        self._conv_activation_params = conv_activation_params
        # Check position, type and parameters of normalization layer
        if norm_layer_position in ['before', 'after', None]:
            self._norm_layer_position = norm_layer_position
        else:
            warnings.warn(("Invalid value {} for `norm_layer_position`: It can only be 'before' (before activation), 'after' (after activation) or None. "+
                          "Using default value of None. There will be no normalization.").format(norm_layer_position), UserWarning)
            self._norm_layer_position = None
        self._norm_layer_type = norm_layer_type
        if self._norm_layer_position is None: self._norm_layer_type = None
        if self._norm_layer_type is None: self._norm_layer_position = None
        self._norm_layer_params = norm_layer_params
        if self._norm_layer_type:
            self._norm_layer_module = getattr(nn, 'BatchNorm{}d'.format(self._conv_dim)) if 'BatchNorm' in self._norm_layer_type else getattr(nn, self._norm_layer_type)
        else:
            self._norm_layer_module = None
        # Check pooling layer type, module and parameters
        self._pool_type = pool_type
        self._pool_module = getattr(nn, "{}Pool{}d".format(self._pool_type, self._conv_dim)) if self._pool_type else None
        self._pool_kernel_size = pool_kernel_size
        self._pool_padding = pool_padding
        self._pool_stride = pool_stride
        self._pool_dilation = pool_dilation
        self._pool_params = pool_params
        self._adaptive_pool_output_size = adaptive_pool_output_size
        # Check Dropout parameters
        self._dropout = dropout if dropout else None
        self._dropout_module = getattr(nn, 'Dropout{}d'.format(self._conv_dim)) if self._dropout else None
        # Store minimum desired image size
        self._min_image_dim = min_image_dim if min_image_dim>0 else 1
        # Initialize vector of layers, and image size
        layers_vec = []
        img_size = self._input_image
        # -----------------------------------------------------------------------------        
        # Check if output image size is smaller than min_image_dim, and adjust parameters if necessary
        temp_img_size = _calc_image_size(img_size, kernel_size=self._conv_kernel_size, stride=self._conv_stride, padding=self._conv_padding, dilation=self._conv_dilation)
        if min(temp_img_size) < self._min_image_dim:
            warnings.warn(
                "Output image is smaller in one or more dimensions than min_image_dim={} for Conv_Block. ".format(self._min_image_dim)+ 
                "Using padding='same' and stride=1 instead of padding={} and stride={}".format(self._conv_padding, self._conv_stride), UserWarning)
            self._conv_padding = 'same'
            self._conv_stride = 1
        # Construct convolutional layer
        if self._conv_params:
            layers_vec.append(self._conv_module(in_channels, out_channels, kernel_size=self._conv_kernel_size, 
            stride=self._conv_stride, padding=self._conv_padding, dilation=self._conv_dilation, **self._conv_params))
        else:
            layers_vec.append(self._conv_module(in_channels, out_channels, kernel_size=self._conv_kernel_size, 
            stride=self._conv_stride, padding=self._conv_padding, dilation=self._conv_dilation))
        # Calculate output image size
        img_size = _calc_image_size(img_size, kernel_size=self._conv_kernel_size, stride=self._conv_stride, padding=self._conv_padding, dilation=self._conv_dilation)
        # ---------------------------------------------------------------------------
        # Construct normalization layer, if it should be here.
        if self._norm_layer_position=='before':
            if self._norm_layer_params:
                layers_vec.append(self._norm_layer_module(out_channels, **self._norm_layer_params))
            else:
                layers_vec.append(self._norm_layer_module(out_channels))
        # Construct activation layer
        if self._conv_activation:
            if self._conv_activation_params:
                layers_vec.append(self._conv_activation_module(**self._conv_activation_params))
            else:
                layers_vec.append(self._conv_activation_module())
        # Construct normalization layer, if it should be here.
        if self._norm_layer_position=='after':
            if self._norm_layer_params:
                layers_vec.append(self._norm_layer_module(out_channels, **self._norm_layer_params))
            else:
                layers_vec.append(self._norm_layer_module(out_channels))
        # ---------------------------------------------------------------------------
        # Check type and parameters of the pooling layer, and calculate output image size
        if self._pool_type is not None and 'adaptive' in self._pool_type.lower():
            assert self._adaptive_pool_output_size is not None, "adaptive_pool_output_size must be specified for adaptive pooling."
            if self._pool_params:
                layers_vec.append(self._pool_module(output_size=self._adaptive_pool_output_size, **self._pool_params))
            else:
                layers_vec.append(self._pool_module(output_size=self._adaptive_pool_output_size))
            img_size = list(self._adaptive_pool_output_size)
        elif self._pool_type is not None:
            temp_img_size = _calc_image_size(img_size, kernel_size=self._pool_kernel_size, stride=self._pool_stride, padding=self._pool_padding, dilation=self._pool_dilation)
            if min(temp_img_size) < self._min_image_dim:
                warnings.warn(
                    "Output image is smaller in one or more dimensions than min_image_dim={} for Conv_Block. ".format(self._min_image_dim)+ 
                    "Using padding={} and stride=1 instead of padding={} and stride={}".format(self._pool_kernel_size//2, self._pool_padding, self._pool_stride), UserWarning)
                self._pool_padding = self._pool_kernel_size//2
                self._pool_stride = 1
            layers_vec.append(self._pool_module(kernel_size=self._pool_kernel_size, stride=self._pool_stride, padding=self._pool_padding, dilation=self._pool_dilation))
            img_size = _calc_image_size(img_size, kernel_size=self._pool_kernel_size, stride=self._pool_stride, padding=self._pool_padding, dilation=self._pool_dilation)
        # ---------------------------------------------------------------------------
        # Construct Dropout layer    
        if self._dropout: layers_vec.append(self._dropout_module(self._dropout))
        # Store output image size as attribute    
        self.output_image = img_size
        # Construct Sequential module
        self.net = nn.Sequential(*layers_vec)

    def forward(self, x):
        return self.net(x)
        

########################################################################################################################


class PyTorchSmartModule(nn.Module):
    
    sample_hparams = {
        'model_name': 'dummy_Pytorch_Smart_Module',
        'l2_reg': 0.0001,
        'batch_size': 16,
        'epochs': 2,
        'validation_data': 0.1,
        'validation_tolerance_epochs': 10,
        'learning_rate': 0.0001,
        'learning_rate_decay_gamma': 0.99,
        'loss_function': 'MSELoss',
        'loss_function_params': None,
        'optimizer': 'Adam',
        'optimizer_params': {'eps': 1e-07}
    }
    
    def __init__(self, hparams:dict=None):
        """
        Base class for smart, trainable pytorch modules. All hyperparameters are contained within the `hparams`
        dictionary. Some training-related hyperparameters are common across almost all kinds of PyTorch modules,
        which can be overloaded by the child class. The module includes functions for training, evaluation, and
        prediction. These functions cane be modified or overloaded by any child subclass.

        ### Usage

        `net = PyTorchSmartModule(hparams)` where `hparams` is dictionary of hyperparameters containing the following:
            - `model_name` (str): Name of the model.
            - `batch_size` (int): Minibatch size, the expected input size of the network.
            - `learning_rate` (float): Initial learning rate of training.
            - `learning_rate_decay_gamma` (float): Exponential decay rate gamma for learning rate, if any.
            - `optimizer` (str): Optimizer. Examples are "Adam", "SGD", "RMSprop", "Adagrad", etc.
            - `optimizer_params` (dict): Additional parameters of the optimizer, if any.
            - `epochs` (int): Maximum number of epochs for training.
            - `validation_tolerance_epochs` (int): Epochs to tolerate unimproved val loss, before early stopping.
            - `l2_reg` (float): L2 regularization parameter.
            - `loss_function` (str): Loss function. Examples: "MSELoss", "CrossEntropyLoss", "NLLLoss", etc.
            - `loss_function_params` (dict): Additional parameters for the loss function, if any.

        ### Returns

        Returns a `torch.nn.Module` object that can be trained and used accordingly.
        Run `print(net)` afterwards to see what you have inside the network.
        
        ### Notes:
        - The `self.batch_input_shape` attribute must be set in the `__init__` method.
        - The `self.batch_output_shape` attribute must be set in the `__init__` method.
        """
        super(PyTorchSmartModule, self).__init__()
        if not hparams: hparams = self.sample_hparams
        self.hparams = hparams
        self._batch_size = int(hparams["batch_size"])
        self._loss_function = hparams.get("loss_function")
        self._loss_function_params = hparams.get("loss_function_params")
        self._optimizer = hparams.get("optimizer")
        self._optimizer_params = hparams.get("optimizer_params")
        self._validation_tolerance_epochs = hparams.get("validation_tolerance_epochs")
        self._learning_rate = hparams.get("learning_rate")
        self._learning_rate_decay_gamma = hparams.get("learning_rate_decay_gamma")
        self._validation_data = hparams.get("validation_data")
        self._epochs = hparams.get("epochs")
        self._l2_reg = hparams.get("l2_reg") if hparams.get("l2_reg") else 0.0
        self.history = None
        self.batch_input_shape = (self._batch_size, 1)
        self.batch_output_shape = (self._batch_size, 1)
        if self._l2_reg > 0.0:
            if self._optimizer_params is not None:
                self._optimizer_params["weight_decay"] = self._l2_reg
            else:
                self._optimizer_params = {"weight_decay": self._l2_reg}
    
    def train_model(self, dataset, verbose:bool=True, script_before_save:bool=False, saveto:str=None, **kwargs):
        self.history = train_pytorch_model(self, dataset, self._batch_size, self._loss_function, self._optimizer, self._optimizer_params, self._loss_function_params, 
        self._learning_rate, self._learning_rate_decay_gamma, self._epochs, self._validation_tolerance_epochs, self._validation_data, verbose, script_before_save, saveto, **kwargs)
        return self.history
    
    def evaluate_model(self, dataset, verbose:bool=True, **kwargs):
        return evaluate_pytorch_model(self, 
            dataset, loss_str=self._loss_function, loss_function_params=self._loss_function_params, batch_size=self._batch_size, verbose=verbose, **kwargs)
    
    def predict_model(self, dataset, 
        return_in_batches:bool=True, return_inputs:bool=False, return_raw_predictions:bool=False, verbose:bool=True, **kwargs):
        return predict_pytorch_model(self, dataset, self._loss_function, self._batch_size, return_in_batches=return_in_batches, return_inputs=return_inputs, 
            return_raw_predictions=return_raw_predictions, verbose=verbose, **kwargs)


                

########################################################################################################################
        
class Recurrent_Network(PyTorchSmartModule):
    
    sample_hparams = {
        'model_name': 'Recurrent_Network',
        'in_features': 10,
        'out_features': 3,
        'in_seq_len': 13,
        'out_seq_len': 1,
        'rnn_type': 'LSTM',
        'rnn_hidden_sizes': 8,
        'rnn_bidirectional': False,
        'rnn_depth': 2,
        'rnn_dropout': 0.1,
        'rnn_params': None,
        'lstm_proj_size':None,
        'final_rnn_return_sequences': False,
        'apply_dense_for_each_time_step': True,
        'permute_output': False,
        'dense_width': 16,
        'dense_depth': 2,
        'dense_dropout': 0.2,
        'dense_activation': 'ReLU',
        'dense_activation_params': None,
        'output_activation': None,
        'output_activation_params': None,
        'norm_layer_type': 'BatchNorm1d',
        'norm_layer_params': None,
        'norm_layer_position': 'before',
        'l2_reg': 0.0001,
        'batch_size': 16,
        'epochs': 2,
        'validation_data': [0.05,'testset'],
        'validation_tolerance_epochs': 10,
        'learning_rate': 0.0001,
        'learning_rate_decay_gamma': 0.99,
        'loss_function': 'CrossEntropyLoss',
        'loss_function_params': None,
        'optimizer': 'Adam',
        'optimizer_params': {'eps': 1e-07}
    }
    
    def __init__(self, hparams:dict=None):
        """Sequence to Dense network with RNN for time-series classification, regression, and forecasting, as well as NLP applications.
        This network uses any RNN layers as encoders to extract information from input sequences, and fully-connected 
        multilayer perceptrons (Dense) to decode the sequence into an output, which can be class probabilitites 
        (timeseries classification), a continuous number (regression), or an unfolded sequence (forecasting) of a 
        target timeseries.

        ### Usage

        `net = Recurrent_Network(hparams)` where `hparams` is dictionary of hyperparameters containing the following:

            - `rnn_type` (str): RNN type, options are "LSTM", "GRU", "RNN", etc.
            - `in_seq_len` (int): Input sequence length, in number of timesteps
            - `out_seq_len` (int): Output sequence length, in number of timesteps, assuming output is also a sequence. This will affect the output layer in the dense section.
                Use 1 for when the output is not a sequence, or do not supply this key.
            - `in_features` (int): Number of features of the input.
            - `out_features` (int): Number of features of the output.
            - `rnn_hidden_sizes` ("auto"|int): RNN layer hidden size. "auto" decides automatically, and a number sets them all the same. Default is 'auto'.
            - `rnn_bidirectional` (bool): Whether the RNN layers are bidirectional or not. Default is False.
            - `rnn_depth` (int): Number of stacked RNN layers. Default is 1.
            - `rnn_dropout` (float): Dropout rates, if any, of the RNN layers. PyTorch ignores this if there is only one RNN layer.
                Please note that using dropout in RNN layers is generally discouraged, for it decreases determinism during inference.
            - `rnn_params` (dict): Additional parameters for the RNN layer constructor. Default is None.
            - `lstm_proj_size` (int): If the RNN type is LSTM, this is the projection size of the LSTM. Default is None.
            - `final_rnn_return_sequences` (bool): Whether the final RNN layer returns sequences of hidden state. 
                **NOTE** Setting this to True will make the model much, much larger.
            - `apply_dense_for_each_time_step` (bool): Whether to apply the Dense network to each time step of the 
                RNN output. If False, the Dense network is applied to the last time step only if 
                `final_rnn_retrurn_sequences` is False, or applied to a flattened version of the output sequence
                otherwise (the dimensionality of the input feature space to the dense network will be multiplied
                by the sequence length. PLEASE NOTE that this only works if the entered sequence is exactly as long
                as the priorly defined sequence length according to the hyperparameters).
            - `permute_output` (bool): Whether to permute the output sequence to be (N, D*H_out, L_out)
            - `dense_width` (int|list): (list of) Widths of the Dense network. It can be a number (for all) or a list holding width of each hidden layer.
            - `dense_depth` (int): Depth (number of hidden layers) of the Dense network.
            - `dense_activation` (str|list): (list of) Activation functions for hidden layers of the Dense network. Examples: "ReLU", "LeakyReLU", "Sigmoid", "Tanh", etc.
            - `dense_activation_params` (dict|list): (list of) Dictionaries of parameters for the activation func constructors of the Dense network.
            - `output_activation` (str): Activation function for the output layer of the Dense network, if any. Examples: "Softmax", "LogSoftmax", "Sigmoid", etc.
                **NOTE** If the loss function is cross entropy, then no output activation is erquired.
                However, if the loss function is nll (negative loglikelihood), then you must specify an output activation as in "LogSoftmax".
            - `output_activation_params` (dict): Dictionary of parameters for the activation func constructor of the output layer.
            - `norm_layer_type` (str|list): (list of) Types of normalization layers to use in the dense section, if any. Options are "BatchNorm1d", "LayerNorm", etc.
            - `norm_layer_params` (dict|list): (list of) Dictionaries of parameters for the normalization layer constructors.
            - `norm_layer_position` (str|list): (list of) Whether the normalization layer should come before or after the activation of each hidden layer in the dense network.
            - `dense_dropout` (float|list): (list of) Dropout rates (if any) for the hidden layers of the Dense network.
            - `batch_size` (int): Minibatch size, the expected input size of the network.
            - `learning_rate` (float): Initial learning rate of training.
            - `learning_rate_decay_gamma` (float): Exponential decay rate gamma for learning rate, if any.
            - `optimizer` (str): Optimizer. Examples: 'Adam', 'SGD', 'RMSProp', etc.
            - `optimizer_params` (dict): Additional parameters of the optimizer, if any.
            - `epochs` (int): Maximum number of epochs for training.
            - `validation_tolerance_epochs` (int): Epochs to tolerate unimproved val loss, before early stopping.
            - `validation_data` (list): Portion of validation data. Should be a tuple like [validation split, dataset as in 'trainset' or 'testset']
            - `l2_reg` (float): L2 regularization parameter.
            - `loss_function` (str): Loss function. Examples: 'BCELoss', 'CrossEntropyLoss', 'MSELoss', etc.
            - `loss_function_params` (dict): Additional parameters for the loss function, if any.

        ### Returns
        
        Returns a `torch.nn.Module` object that can be trained and used accordingly.
        Run `print(net)` afterwards to see what you have inside the network.
        The returned model is an instance of `PyTorchSmartModule`, which has builkt-in functions for training, evaluation, prediction, etc.
        """
        super(Recurrent_Network, self).__init__(hparams)
        hparams = hparams if hparams is not None else self.sample_hparams
        self._rnn_type = hparams["rnn_type"]
        self._rnn = getattr(nn, self._rnn_type)
        self._denseactivation = hparams["dense_activation"] if hparams.get("dense_activation") else "ReLU"
        self._denseactivation_params = hparams.get("dense_activation_params")
        self._outactivation = hparams.get("output_activation")
        self._outactivation_params = hparams.get("output_activation_params")
        self._normlayer_type = hparams.get("norm_layer_type")
        self._normlayer_params = hparams.get("norm_layer_params")
        self._normlayer_position = hparams.get("norm_layer_position")
        self._infeatures = hparams["in_features"]
        self._outfeatures = hparams["out_features"]
        self._rnnhidsizes = hparams["rnn_hidden_sizes"] if hparams.get("rnn_hidden_sizes") else "auto"
        self._densehidsizes = hparams["dense_width"] if hparams.get("dense_width") else "auto"
        self._densedepth = hparams["dense_depth"] if hparams.get("dense_depth") else 0
        self._rnndepth = hparams["rnn_depth"] if hparams.get("rnn_depth") else 1
        self._bidirectional = True if hparams.get("rnn_bidirectional") else False
        self._rnndropout = hparams["rnn_dropout"] if hparams.get("rnn_dropout") else 0
        self._densedropout = hparams["dense_dropout"] if hparams.get("dense_dropout") else None
        self._final_rnn_return_sequences = True if hparams.get("final_rnn_return_sequences") else False
        self._apply_dense_for_each_timestep = True if hparams.get("apply_dense_for_each_timestep") else False
        self._permute_output = True if hparams.get("permute_output") else False
        self._N = int(hparams["batch_size"])
        self._L_in = int(hparams["in_seq_len"])
        self._L_out = int(hparams["out_seq_len"]) if hparams.get("out_seq_len") else 1
        self._D = int(2 if self._bidirectional else 1)
        self._rnn_params = hparams.get("rnn_params")
        self._lstmprojsize = hparams.get("lstm_proj_size") if hparams.get("lstm_proj_size") else 0
        self._H_in = int(self._infeatures)
        if self._rnnhidsizes == "auto": self._H_cell = int(2**(np.round(math.log2(self._H_in*self._L_in))))
        else: self._H_cell = int(self._rnnhidsizes)
        self._H_out = int(self._lstmprojsize if self._lstmprojsize and self._lstmprojsize > 0 else self._H_cell)
        self.batch_input_shape = (self._N, self._L_in, self._H_in)
        if self._final_rnn_return_sequences and self._apply_dense_for_each_timestep:
            if self._permute_output: self.batch_output_shape = (self._N, self._outfeatures, self._L_out)
            else: self.batch_output_shape = (self._N, self._L_out, self._outfeatures)
        else: self.batch_output_shape = (self._N, self._L_out * self._outfeatures)
        
        # Constructing RNN layers
        if self._rnn_type == "LSTM" and self._lstmprojsize > 0:
            if self._rnn_params:
                self.rnn = nn.LSTM(input_size=self._H_in, hidden_size=self._H_cell, num_layers=self._rnndepth, batch_first=True, dropout=self._rnndropout, 
                    bidirectional=self._bidirectional, proj_size=self._lstmprojsize, **self._rnn_params)
            else:
                self.rnn = nn.LSTM(input_size=self._H_in, hidden_size=self._H_cell, num_layers=self._rnndepth, batch_first=True, dropout=self._rnndropout, 
                    bidirectional=self._bidirectional, proj_size=self._lstmprojsize)
        else:
            if self._rnn_params:
                self.rnn = self._rnn(input_size=self._H_in, hidden_size=self._H_cell, num_layers=self._rnndepth, 
                                    batch_first=True, dropout=self._rnndropout, bidirectional=self._bidirectional, **self._rnn_params)
            else:
                self.rnn = self._rnn(input_size=self._H_in, hidden_size=self._H_cell, num_layers=self._rnndepth, 
                                    batch_first=True, dropout=self._rnndropout, bidirectional=self._bidirectional)
        # for attrib in dir(self.rnn):
        #     if attrib.startswith("weight_ih"): xavier_uniform_(self.rnn.__getattr__(attrib))
        #     elif attrib.startswith("weight_hh"): orthogonal_(self.rnn.__getattr__(attrib))
        #     elif attrib.startswith("bias_"): zeros_(self.rnn.__getattr__(attrib))
        
        # Calculating Dense layers widths
        cf = self._L_in if (self._final_rnn_return_sequences and not self._apply_dense_for_each_timestep) else 1 
        self._dense_input_size = self._H_out * self._D * cf
        if self._final_rnn_return_sequences and not self._apply_dense_for_each_timestep:
            self._dense_output_size = int(self._L_out*self._outfeatures)
        else:
            self._dense_output_size = int(self._outfeatures)
            
        # Generate arrays containing parameters of each Dense Block (Every block contains a linear, normalization, activation, and dropout layer).
        self._dense_width_vec = self._gen_hparam_vec_for_dense(self._densehidsizes, 'dense_width')
        self._dense_activation_vec = self._gen_hparam_vec_for_dense(self._denseactivation, 'dense_activation')
        self._dense_activation_params_vec = self._gen_hparam_vec_for_dense(self._denseactivation_params, 'dense_activation_params')
        self._dense_norm_layer_type_vec = self._gen_hparam_vec_for_dense(self._normlayer_type, 'norm_layer_type')
        self._dense_norm_layer_params_vec = self._gen_hparam_vec_for_dense(self._normlayer_params, 'norm_layer_params')
        self._dense_norm_layer_position_vec = self._gen_hparam_vec_for_dense(self._normlayer_position, 'norm_layer_position')
        self._dense_dropout_vec = self._gen_hparam_vec_for_dense(self._densedropout, 'dense_dropout')
        
        # Construct the dense layers
        in_size = self._dense_input_size
        layers = []
        for i in range(self._densedepth):
            out_size = self._dense_width_vec[i]
            temp_dropout_rate = self._dense_dropout_vec[i] if (i != self._densedepth-1) else None # The hidden layer just before the output layer rarely has Dropout.
            layers.append(Dense_Block(in_size, out_size, self._dense_activation_vec[i], self._dense_activation_params_vec[i], 
                self._dense_norm_layer_type_vec[i], self._dense_norm_layer_position_vec[i], self._dense_norm_layer_params_vec[i], temp_dropout_rate))
            in_size = out_size
        
        # Output layer
        layers.append(nn.Linear(in_size, self._dense_output_size))
        if self._outactivation:
            if self._outactivation_params:
                layers.append(getattr(nn, self._outactivation)(**self._outactivation_params))
            else:
                layers.append(getattr(nn, self._outactivation)())
        
        # Sequentiating the layers
        self.fc = nn.Sequential(*layers)
        
        self.rnn.flatten_parameters()
        
        
    
    def _gen_hparam_vec_for_dense(self, hparam, hparam_name, **kwargs):
        return _generate_array_for_hparam(hparam, self._densedepth, hparam_name=hparam_name, count_if_not_list_name='dense_depth', **kwargs)
    
    def forward(self, x):
        self.rnn.flatten_parameters()
        # self._rnn_output, (self._rnn_final_hidden_states, self._lstm_final_cell_states) = self.rnn(x)
        rnn_output, _ = self.rnn(x)
        if self._final_rnn_return_sequences:
            if self._apply_dense_for_each_timestep: rnn_output_flattened = rnn_output
            else: rnn_output_flattened = rnn_output.view(rnn_output.shape[0], -1)
        else:
            # RNN output is of shape  (N, L, D * H_out)
            rnn_output_flattened = rnn_output[:,-1,:]
        out = self.fc(rnn_output_flattened)
        if self._permute_output:
            return out.permute(0,2,1)
        else:
            return out



########################################################################################################################

class LanguageModel(Recurrent_Network):
    def __init__(self, hparams:dict=None):
        super(LanguageModel, self).__init__(hparams)
        self.embed_dim = hparams['embedding_dim'] if hparams.get('embedding_dim') else hparams['in_features']
        self.vocab_size = hparams['vocab_size'] if hparams.get('vocab_size') else 27
        assert self.embed_dim == self._infeatures, "Embedding dim (%d) must be equal to input feature dim (%d)."%(self.embed_dim, self._infeatures)
        self.embed_layer = nn.Embedding(self.vocab_size, self.embed_dim)
        self._embed_output = None
        self.permute_output = hparams['permute_output'] if hparams.get('permute_output') else False
        self.batch_input_shape = [hparams['batch_size'], hparams['in_seq_len']]
        if self.permute_output:
            self.batch_output_shape = [hparams['batch_size'], hparams['out_features'], hparams['out_seq_len']]
        else:
            self.batch_output_shape = [hparams['batch_size'], hparams['out_seq_len'], hparams['out_features']]
        
    def forward(self, x):
        # TODO: Transfer the "permute_output" logic to the base class as well.
        # self._rnn_output, (self._rnn_final_hidden_states, self._lstm_final_cell_states) = self.rnn(x)
        # Shape of x should be: [N, L]
        self._embed_output = self.embed_layer(x)    # [N, L, embed_dim]
        self._rnn_output, _ = self.rnn(self._embed_output)
        if self._final_rnn_return_sequences:
            if self._apply_dense_for_each_timestep:
                self._rnn_output_flattened = self._rnn_output
            else:
                self._rnn_output_flattened = self._rnn_output.view(self._rnn_output.shape[0], -1)
        else:
            # RNN output is of shape  (N, L, D * H_out)
            self._rnn_output_flattened = self._rnn_output[:,-1,:]
        out = self.decoder(self._rnn_output_flattened)
        if self.permute_output:
            return out.permute(self.permute_output)
        else:
            return out
    
    def test(self):
        print("------------------------------------------------------------------")
        print("Testing RNN_Language_Network")
        print("Constructing random inputs and outputs ...")
        print("Batch size:              %d"%self._batchsize)
        print("Input sequence length:   %d"%self._L_in)
        print("Output sequence length:  %d"%self._L_out)
        print("Input feature dimension: %d"%self._infeatures)
        print("Construjcting random torch.long tensor for input ...")
        x = torch.randint(0, self.vocab_size, self.batch_input_shape, dtype=torch.long)
        print("Input shape:  %s"%str(x.shape))
        print("Constructing random torch.float tensor for output ...")
        y_true = torch.rand(size=self.batch_output_shape)
        print("Output shape from truth: %s"%str(y_true.shape))
        print("Calling the forward method ...")
        y_pred = self.forward(x)
        print("Output shape from preds: %s"%str(y_pred.shape))
        assert y_true.shape == y_pred.shape, \
            "Output shape (%s) does not match expected shape (%s)"%(str(y_pred.shape), str(y_true.shape))
        print("Testing complete. Output shape matches expected shape.")
        print("------------------------------------------------------------------")


########################################################################################################################


class ANN(PyTorchSmartModule):
    
    sample_hparams = {
        "model_name": "ANN",
        "input_size": 10,
        "output_size": 3,
        "width": 32,
        "depth": 2,
        "hidden_activation": "ReLU",
        "hidden_activation_params": None,
        "output_activation": None,
        "output_activation_params": None,
        "norm_layer_type":"BatchNorm1d",
        "norm_layer_position": "before",
        "norm_layer_params": None,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "learning_rate_decay_gamma": 0.99,
        "optimizer": "Adam",
        "optimizer_params": {"eps": 1e-08},
        "batch_size": 32,
        "epochs": 2,
        "validation_tolerance_epochs": 2,
        "validation_data":[0.05,'trainset'],
        "l2_reg": 0.0001,
        "loss_function": "CrossEntropyLoss",
        "loss_function_params": None
    }
    
    
    def __init__(self, hparams:dict=None):
        """Typical Artificial Neural Network class, also known as multilayer perceptron. This class will create a fully connected feedforward artificial neural network.
        It can be used for classification, regression, etc. It basically encompasses enough options to build all kinds of ANNs with any number of 
        inputs, outputs, layers with custom or arbitrary width or depth, etc. Supports multiple activation functions for hidden layers and the output layer,
        but the activation function of the hidden layers are all the same.
        
        ### Usage
        `net = ANN(hparams)` where `hparams` is the dictionary of hyperparameters.

        It can include the following keys:
            - `input_size` (int): number of inputs to the ANN, i.e. size of the input layer.
            - `output_size` (int): number of outputs to predict, i.e. size of the output layer.
            - `width` (int|list): (list of) hidden layer widths. 
                a number sets them all the same, and a list/array sets each hidden layer according to the list.
            - `depth` (int): Specifies the depth of the network (number of hidden layers).
                It must be specified unless `width` is provided as a list. Then the depth will be inferred form it.
            - `hidden_activation` (str): (list of) Activations of the hidden layers. Examples include "ReLU", "LeakuReLU", "Sigmoid", "Tanh", "Softmax", "LogSoftmax", etc.
            - `hidden_activation_params` (dict): (list of) Parameters for the hidden activation function, if any.
            - `output_activation` (str): Activation of the output layer, if any.
                **Note**: For classification problems, you may want to choose "Sigmoid", "Softmax" or "LogSoftmax".
                That being said, you usually don't need to specify an activation for the output layer at all.
                Some loss functions in PyTorch have the classification activation functions embedded in them.
                **Note**: For regression problems, no activation is needed. It is by default linear, unless you want to manually specify an activation.
            - `output_activation_params` (dict): Parameters for the output activation function, if any.
            - `norm_layer_type` (str): (list of) Types of normalization layers to use for each hidden layer. Options are "BatchNorm1d", "LayerNorm", "GroupNorm", etc.
            - `norm_layer_position` (str): (list of) where the normalization layer should be included relative to the activation function.
            - `norm_layer_params` (dict): (list of) Dictionaries of parameters for the normalization layers.
            - `dropout` (float): (list of) the dropout rates after every hidden layer. It should be a probability value between 0 and 1.
            - `learning_rate` (float): Initial learning rate of training.
            - `learning_rate_decay_gamma` (float): Exponential decay rate gamma for learning rate, if any.
            - `optimizer` (str): Optimizer. Examples: "Adam", "SGD" ,"RMSProp", etc.
            - `optimizer_params` (dict): Additional parameters of the optimizer, if any.
            - `batch_size` (int): Minibatch size for training.
            - `epochs` (int): Maximum number of epochs for training.
            - `validation_tolerance_epochs` (int): Epochs to tolerate unimproved val loss, before early stopping.
            - `validation_data` (list): List of [validation_split, 'trainset'|'testset'].
            - `l2_reg` (float): L2 regularization parameter.
            - `loss_function` (str): Loss function. Examples: "MSELoss", "BCELoss", "CrossEntropyLoss", etc.
            - `loss_function_params` (dict): Additional parameters for the loss function, if any.
        
        Note that for all such hyperparameters that have a (list of) at the beginning, the entry can be a single item repeated for all hidden layers, or it can be a list of items
        for all hidden layers. If a list is provided, it must have the same length as the depth of the network. Also note that depth does not include the input and output layers.
        This gives you the ability to specify different width, dropout rate, normalization layer and its parameters, and so forth.
        
        Also note that the hidden layer just before the output layer will not have any dropout, which is typical.

        ### Returns
        It returns a `torch.nn.Module` object that corresponds with an ANN model.
        run `print(net)` afterwards to see what the ANN holds.
        The returned module is a `PyTorchSmartModule` object, which is a subclass of `torch.nn.Module`. It has built-in functions for training, evaluation, prediction, etc.
        """
        super(ANN, self).__init__(hparams)
        # Read and store hyperparameters
        layers = []
        self._insize = hparams["input_size"]
        self._outsize = hparams["output_size"]
        self._dropout = hparams.get("dropout")
        self._width = hparams.get("width")
        self._depth = hparams.get("depth")
        self._denseactivation = actdict_pytorch[hparams["hidden_activation"]]
        self._denseactivation_params = hparams.get("hidden_activation_params")
        self._outactivation = actdict_pytorch[hparams.get("output_activation")] if hparams.get("output_activation") else None
        self._outactivation_params = hparams.get("output_activation_params")
        self._norm_layer_type = hparams.get("norm_layer_type")
        self._norm_layer_position = hparams.get("norm_layer_position")
        self._norm_layer_params = hparams.get("norm_layer_params")
        self.batch_input_shape = (self._batchsize, self._insize)
        self.batch_output_shape = (self._batchsize, self._outsize)
        
        # Generate arrays containing parameters of each Dense Block (Every block contains a linear, normalization, activation, and dropout layer).
        self._dense_width_vec = self._gen_hparam_vec_for_dense(self._width, 'width')
        self._dense_activation_vec = self._gen_hparam_vec_for_dense(self._denseactivation, 'hidden_activation')
        self._dense_activation_params_vec = self._gen_hparam_vec_for_dense(self._denseactivation_params, 'hidden_activation_params')
        self._dense_norm_layer_type_vec = self._gen_hparam_vec_for_dense(self._norm_layer_type, 'norm_layer_type')
        self._dense_norm_layer_params_vec = self._gen_hparam_vec_for_dense(self._norm_layer_params, 'norm_layer_params')
        self._dense_norm_layer_position_vec = self._gen_hparam_vec_for_dense(self._norm_layer_position, 'norm_layer_position')
        self._dense_dropout_vec = self._gen_hparam_vec_for_dense(self._dropout, 'dropout')
        
        # Construct the dense layers
        in_size = self._insize
        for i in range(self._depth):
            out_size = self._dense_width_vec[i]
            temp_dropout_rate = self._dense_dropout_vec[i] if (i != self._depth-1) else None # The hidden layer just before the output layer rarely has Dropout.
            layers.append(Dense_Block(in_size, out_size, self._dense_activation_vec[i], self._dense_activation_params_vec[i], 
                                            self._dense_norm_layer_type_vec[i], self._dense_norm_layer_position_vec[i], self._dense_norm_layer_params_vec[i], temp_dropout_rate))
            in_size = out_size
        
        # Output layer
        layers.append(nn.Linear(out_size, self._outsize))
        if self._outactivation:
            if self._outactivation_params:
                layers.append(getattr(nn, self._outactivation)(**self._outactivation_params))
            else:
                layers.append(getattr(nn, self._outactivation)())
        
        # Sequentiating the layers
        self.net = nn.Sequential(*layers)

    def _gen_hparam_vec_for_dense(self, hparam, hparam_name, **kwargs):
        return _generate_array_for_hparam(hparam, self._depth, hparam_name=hparam_name, count_if_not_list_name='depth', **kwargs)
        
    def forward(self, x):
        return self.net(x)


########################################################################################################################

class Conv_Network(PyTorchSmartModule):
    sample_hparams = {
        "model_name": "Conv_Network",
        # I/O shapes (without the batch dimension)
        "input_shape": [3, 28, 28],
        "output_shape": [10],
        # Convolution blocks
        "num_conv_blocks": 2,
        "conv_dim": 2,
        "conv_params": None,
        "conv_channels": "auto",
        "conv_kernel_size": 3,
        "conv_padding": "valid",
        "conv_stride": 1,
        "conv_dilation": 1,
        "conv_activation": "LeakyReLU",
        "conv_activation_params": {"negative_slope": 0.1},
        "conv_norm_layer_type": "BatchNorm",
        "conv_norm_layer_position": "before",
        "conv_norm_layer_params": None,
        "conv_dropout": 0.1,
        "pool_type": "Max",
        "pool_kernel_size": 2,
        "pool_padding": 0,
        "pool_stride": 1,
        "pool_dilation": 1,
        "pool_params": None,
        "min_image_size": 4,
        "adaptive_pool_output_size": None,
        # Fully connected blocks
        "dense_width": "auto",
        "dense_depth": 2,
        "dense_activation": "ReLU",
        "dense_activation_params": None,
        "output_activation": "LogSoftmax",
        "output_activation_params": None,
        "dense_norm_layer_type": "BatchNorm1d",
        "dense_norm_layer_position": "before",
        "dense_norm_layer_params": None,
        "dense_dropout": 0.1,
        # Training procedure
        "l2_reg": 0.0001,
        "batch_size": 32,
        "epochs": 40,
        "validation_data": [0.05,'testset'],
        "validation_tolerance_epochs": 5,
        "learning_rate": 0.01,
        "learning_rate_decay_gamma": 0.9,
        "loss_function": "NLLLoss",
        "optimizer": "Adam",
        "optimizer_params": {"eps": 1e-07}
    }
    
    
    def __init__(self, hparams:dict=None):
        """Standard Convolutional Neural Network, containing convolutional blocks followed by fully-connected blocks.
        It supports 1D, 2D, and 3D convolutions, and can be used for image classification, timeseries classification,
        video classification, and so forth. The module can easily be trained and evaluated using its own methods,
        because it inherits from `PyTorchSmartModule`.

        ### Usage

        `model = Conv_Network(hparams)` where `hparams` is dictionary of hyperparameters containing the following:

        #### I/O shapes
        
        - `input_shape` (list): Input shape *WITHOUT* the batch dimension. For instance, for 2D images, input should be [N, C, H, W], therefore `input_shape` should be [C, H, W].
        - `output_shape` (int): Output shape *WITHOUT* the batch dimension. For instance, for K-class classification, model outputs can be [N, K], so `output_shape` should be [K].
            
        #### Convolution blocks
        
        - `num_conv_blocks` (int): Number of convolutional blocks. Every block contains a convolutional layer, and
            optionally a normalization layer, an activation layer, a pooling layer, and finally a dropout layer.
        - `conv_dim` (int): Dimensionality of the convolution. 1, 2, or 3.
        - `conv_params` (dict): kwargs dict to pass to the convolution constructor in *ALL* blocks. Defaults to None.
        - `conv_channels` (int|list|str): Number of filters of the convolution layers. If `auto`, it will start
            with the input channels, and double with every block, in powers of two. If `list`, it should be a list
            of channels for each conv block. If `int`, it will be the same for all conv blocks. Default is `auto`.
        - `conv_kernel_size` (int|list): Kernel size of the convolution layers. Should be a list of integers,
            a list of tuples of integers (for conv2d or conv3d), or an integer. If it is a list, it MUST have the same 
            length as `num_conv_blocks`. If it is an integer, it will be the same for all conv blocks. Defaults to 3.
        - `conv_padding` (int|str|list): Padding of convolution layers. Format is as `conv_kernel_size`. Defaults to "valid".
        - `conv_stride` (int|list): Stride of convolution layers. Format is as `conv_kernel_size`. Defaults to 1.
        - `conv_dilation` (int|list): Dilation of convolution layers. Format is as `conv_kernel_size`. Defaults to 1.
        - `conv_activation` (str|list): (list of) string(s) representing activation func of the convolution layers. Examples: 'ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', etc.
        - `conv_activation_params` (dict|list): (list of) dicts for the convolution activation functions' constructors. Defaults to None.
        - `conv_norm_layer_type` (str|list): (list of) types of normalization layers to use in the conv blocks. Examples: 'BatchNorm', 'LayerNorm', etc.
            If 'BatchNorm' is used, its dimensionality will match that of `conv_dim`. Defaults to None.
        - `conv_norm_layer_position` ("before"|"after"|list): (list of) positions of the normalization layers in the 
            convolutional blocks relative to the activation functions. Defaults to "before". If it is a list, it should be a list of strings of the same length as `num_conv_blocks`
        - `conv_norm_layer_params` (dict|list): kwargs dict for the convolution normalization layers' constructors. Defaults to None.    
        - `conv_dropout` (float|list): (list of) Dropout rates of the convolution blocks. Defaults to None.
        - `pool_type` (str|list): (list of) types of pooling layer. "Max", "Avg", "AdaptiveMax", "AdaptiveAvg", etc. Defaults to None, in which case there will be no pooling layer.
        - `pool_kernel_size` (int|list): (list of) kernel sizes of the pooling layers, with similar format to 
            `conv_kernel_size`. Again, it can be a list of integers, a list of tuples of integers, or an integer.
        - `pool_padding` (int|list): (list of) paddings of the pooling layers.
        - `pool_stride` (int|list): (list of) strides of the pooling layers.
        - `pool_dilation` (int|list): (list of) dilations of the pooling layers.
        - `pool_params` (dict|list): (list of) kwargs dicts for the pooling layers' constructors.
        - `adaptive_pool_output_size` (int|list): (list of) output sizes of the adaptive pooling layers, if any.
            If it is a list, it should contain one element (integer or tuple) per adaptive pooling layer.
        - `min_image_size` (int): Minimum size of the image to be reduced to in convolutions and poolings.
            After this point, the padding and striding will be chosen such that image size does not decrease further. Defaults to 1.
            
        #### Dense blocks
        
        - `dense_width` ("auto"|int|list): Width of the hidden layers of the Dense network. "auto", a number (for all of them) or a list holding width of each hidden layer.
            If "auto", it will start with the output size of the Flatten() layer, halving at every Dense block.
        - `dense_depth` (int): Depth (number of hidden layers) of the Dense network.
        - `dense_activation` (str|list): (list of) activation function for hidden layers of the Dense network. Examples: 'ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', etc.
        - `dense_activation_params` (dict|list): (list of) dicts for the dense activation functions' constructors.
        - `output_activation` (str): Activation function for the output layer of the Dense network, if any.
            **NOTE** If the loss function is cross entropy, then no output activation is erquired. However, if the loss function is `NLLLoss` (negative loglikelihood), 
            then you MUST specify an output activation as in `LogSoftmax`.
        - `output_activation_params` (dict): Dictionary of parameters for the output activation function's constructor.
        - `dense_norm_layer_type` (str|list): (list of) types of normalization layers to use in the dense blocks. Examples: 'BatchNorm', 'LayerNorm', etc.
            If 'BatchNorm' is used, it will be `BatchNorm1d`. Defaults to None, in which case no normalization layer will be used.
        - `dense_norm_layer_position` ("before"|"after"|list): (list of) positions of the normalization layers in the dense blocks relative to the activation functions. 
            Defaults to "before". If it is a list, it should be a list of strings of the same length as `dense_depth`.
        - `dense_norm_layer_params` (dict|list): (list of) kwargs dict for the dense normalization layers' constructors.
        - `dense_dropout` (float|list): (list of) Dropout rates (if any) for the hidden layers of the Dense network.
        
        #### Training procedure
        
        - `batch_size` (int): Minibatch size, the expected input size of the network.
        - `learning_rate` (float): Initial learning rate of training.
        - `learning_rate_decay_gamma` (float): Exponential decay rate gamma for learning rate, if any.
        - `optimizer` (str): Optimizer. Examples: 'Adam', 'SGD', 'RMSprop', etc.
        - `optimizer_params` (dict): Additional parameters of the optimizer, if any.
        - `epochs` (int): Maximum number of epochs for training.
        - `validation_tolerance_epochs` (int): Epochs to tolerate unimproved val loss, before early stopping.
        - `l2_reg` (float): L2 regularization parameter.
        - `loss_function` (str): Loss function. Examples: 'CrossEntropyLoss', 'NLLLoss', 'MSELoss', etc.
        - `loss_function_params` (dict): Additional parameters for the loss function, if any.
        - `validation_data` (tuple): Validation data, if any. It should be a tuple of (portion, from_dataset). For instance, [0.05, 'testset'] means 5% of the testset will be used 
            for validation.The second element of the tuple can only be 'trainset' and 'testset'. The first element must be a float between 0 and 1. 
            If the second element is not specified, testset will be used by default.
        
        ### Returns
        
        - Returns a `nn.Module` object that can be trained and used accordingly.
        - Run `print(net)` afterwards to see what you have inside the network.
        - A `PyTorchSmartModule` object is returned, which is a subclass of `nn.Module`. This module has its own functions for training, evaluation, etc.
        """
        super(Conv_Network, self).__init__(hparams)
        if not hparams: hparams = self.sample_hparams
        # Input and output shapes
        self.model_name = hparams["model_name"] if hparams.get("model_name") else "Conv_Network"
        self.input_shape = hparams["input_shape"]
        self.output_shape = hparams["output_shape"]
        self._N = int(hparams["batch_size"])
        self.batch_input_shape = list(self.input_shape).copy()
        self.batch_input_shape.insert(0, self._N)
        self.batch_output_shape = list(self.output_shape).copy()
        self.batch_output_shape.insert(0, self._N)
        self.size_list = [self.input_shape]
        modules_list = []
        
        # Convolutional layers hyperparameters
        self._num_conv_blocks = hparams.get("num_conv_blocks")
        self._conv_dim = hparams.get("conv_dim")
        self._conv_params = hparams.get("conv_params")    
        self._conv_channels = hparams.get("conv_channels") if hparams.get("conv_channels") else "auto"
        self._conv_kernel_size = hparams.get("conv_kernel_size") if hparams.get("conv_kernel_size") else 3
        self._conv_padding = hparams["conv_padding"] if hparams.get("conv_padding") else "valid"
        self._conv_stride = hparams["conv_stride"] if hparams.get("conv_stride") else 1
        self._conv_dilation = hparams["conv_dilation"] if hparams.get("conv_dilation") else 1
        self._conv_activation = hparams["conv_activation"] if hparams.get("conv_activation") else "relu"
        self._conv_activation_params = hparams.get("conv_activation_params")
        self._conv_norm_layer_type = hparams.get("conv_norm_layer_type")
        self._conv_norm_layer_position = hparams.get("conv_norm_layer_position")
        self._conv_norm_layer_params = hparams.get("conv_norm_layer_params")
        self._conv_dropout = hparams.get("conv_dropout")
        self._pool_type = hparams.get("pool_type")
        self._pool_kernel_size = hparams.get("pool_kernel_size") if hparams.get("pool_kernel_size") else 2
        self._pool_padding = hparams["pool_padding"] if hparams.get("pool_padding") else 0
        self._pool_stride = hparams["pool_stride"] if hparams.get("pool_stride") else 1
        self._pool_dilation = hparams["pool_dilation"] if hparams.get("pool_dilation") else 1
        self._pool_params = hparams.get("pool_params")
        self._min_image_size = hparams["min_image_size"] if hparams.get("min_image_size") else 1
        self._adaptive_pool_output_size = hparams.get("adaptive_pool_output_size")
        
        
        # Generate lists of hyperparameters for conv/pool layers
        self._conv_channels_vec = self._gen_hparam_vec_for_conv(self._conv_channels, "conv_channels", 
            check_auto=True, init_for_auto=self.input_shape[0], powers_of_two_if_auto=True, direction_if_auto="up")
        self._conv_kernel_size_vec = self._gen_hparam_vec_for_conv(self._conv_kernel_size, "conv_kernel_size")
        self._pool_kernel_size_vec = self._gen_hparam_vec_for_conv(self._pool_kernel_size, 'pool_kernel_size')
        self._conv_padding_vec = self._gen_hparam_vec_for_conv(self._conv_padding, 'conv_padding')
        self._pool_padding_vec = self._gen_hparam_vec_for_conv(self._pool_padding, 'pool_padding')
        self._conv_stride_vec = self._gen_hparam_vec_for_conv(self._conv_stride, 'conv_stride')
        self._pool_stride_vec = self._gen_hparam_vec_for_conv(self._pool_stride, 'pool_stride')
        self._conv_dilation_vec = self._gen_hparam_vec_for_conv(self._conv_dilation, 'conv_dilation')
        self._pool_dilation_vec = self._gen_hparam_vec_for_conv(self._pool_dilation, 'pool_dilation')
        self._conv_activation_vec = self._gen_hparam_vec_for_conv(self._conv_activation, 'conv_activation')
        self._conv_activation_params_vec = self._gen_hparam_vec_for_conv(self._conv_activation_params, 'conv_activation_params')
        self._pool_type_vec = self._gen_hparam_vec_for_conv(self._pool_type, 'pool_type')
        self._pool_params_vec = self._gen_hparam_vec_for_conv(self._pool_params, 'pool_params')
        self._conv_params_vec = self._gen_hparam_vec_for_conv(self._conv_params, 'conv_params')
        self._adaptive_pool_output_size_vec = self._gen_hparam_vec_for_conv(self._adaptive_pool_output_size, 'adaptive_pool_output_size')
        self._conv_norm_layer_type_vec = self._gen_hparam_vec_for_conv(self._conv_norm_layer_type, 'conv_norm_layer_type')
        self._conv_norm_layer_params_vec = self._gen_hparam_vec_for_conv(self._conv_norm_layer_params, 'conv_norm_layer_params')
        self._conv_norm_layer_position_vec = self._gen_hparam_vec_for_conv(self._conv_norm_layer_position, 'conv_norm_layer_position')
        self._conv_dropout_vec = self._gen_hparam_vec_for_conv(self._conv_dropout, 'conv_dropout')
        
        # Constructing the encoder (convolutional blocks)
        # print("input_shape: ", self.input_shape)
        in_channels = self.input_shape[0]
        input_image = list(self.input_shape[1:])
        for i in range(self._num_conv_blocks):
            out_channels = self._conv_channels_vec[i]
            # print("in_channels: ", in_channels)
            # print("out_channels: ", out_channels)
            # print("input_image: ", input_image)
            block = Conv_Block(in_channels, out_channels, self._conv_dim, input_image, self._conv_kernel_size_vec[i], self._conv_padding_vec[i], self._conv_stride_vec[i], 
                 self._conv_dilation_vec[i], self._conv_params_vec[i], self._conv_activation_vec[i], self._conv_activation_params_vec[i], self._conv_norm_layer_position_vec[i], 
                 self._conv_norm_layer_type_vec[i],  self._conv_norm_layer_params_vec[i], self._pool_type_vec[i], self._pool_kernel_size_vec[i], 
                 self._pool_padding_vec[i], self._pool_stride_vec[i], self._pool_dilation_vec[i], self._pool_params_vec[i], self._adaptive_pool_output_size_vec[i], 
                 self._conv_dropout_vec[i], self._min_image_size)
            modules_list.append(block)
            output_image = block.output_image
            self.size_list.append([out_channels]+output_image)
            in_channels = out_channels
            input_image = output_image
            
        # Flattening (Image embedding)
        modules_list.append(nn.Flatten())
        self._dense_input_size = np.prod(output_image) * out_channels
        self.size_list.append([self._dense_input_size])
        
        # Dense layers hyperparameters
        self._dense_width = hparams["dense_width"]
        self._dense_depth = hparams["dense_depth"]
        self._dense_activation = hparams["dense_activation"] if hparams.get("dense_activation") else "ReLU"
        self._dense_activation_params = hparams.get("dense_activation_params")
        self._output_activation = hparams.get("output_activation") if hparams.get("output_activation") else None
        self._output_activation_params = hparams.get("output_activation_params")
        self._dense_norm_layer_type = hparams.get("dense_norm_layer_type")
        self._dense_norm_layer_params = hparams.get("dense_norm_layer_params")
        self._dense_norm_layer_position = hparams.get("dense_norm_layer_position")
        self._dense_dropout = hparams.get("dense_dropout")
        
        # Generate lists of hyperparameters for the dense layers
        self._dense_width_vec = self._gen_hparam_vec_for_dense(self._dense_width, 'dense_width',
            check_auto=True, init_for_auto=self._dense_input_size, powers_of_two_if_auto=True, direction_if_auto="down")
        self._dense_activation_vec = self._gen_hparam_vec_for_dense(self._dense_activation, 'dense_activation')
        self._dense_activation_params_vec = self._gen_hparam_vec_for_dense(self._dense_activation_params, 'dense_activation_params')
        self._dense_norm_layer_type_vec = self._gen_hparam_vec_for_dense(self._dense_norm_layer_type, 'dense_norm_layer_type')
        self._dense_norm_layer_params_vec = self._gen_hparam_vec_for_dense(self._dense_norm_layer_params, 'dense_norm_layer_params')
        self._dense_norm_layer_position_vec = self._gen_hparam_vec_for_dense(self._dense_norm_layer_position, 'dense_norm_layer_position')
        self._dense_dropout_vec = self._gen_hparam_vec_for_dense(self._dense_dropout, 'dense_dropout')
        
        # Construct the dense layers
        in_size = self._dense_input_size
        for i in range(self._dense_depth):
            out_size = self._dense_width_vec[i]
            temp_dropout_rate = self._dense_dropout_vec[i] if (i != self._dense_depth-1) else None # The hidden layer just before the output layer rarely has Dropout.
            modules_list.append(Dense_Block(in_size, out_size, self._dense_activation_vec[i], self._dense_activation_params_vec[i], 
                                            self._dense_norm_layer_type_vec[i], self._dense_norm_layer_position_vec[i], self._dense_norm_layer_params_vec[i], temp_dropout_rate))
            in_size = out_size
            self.size_list.append([out_size])
        
        # Output layer
        modules_list.append(nn.Linear(in_size, self.output_shape[0]))
        if self._output_activation:
            if self._output_activation_params:
                modules_list.append(getattr(nn, self._output_activation)(**self._output_activation_params))
            else:
                modules_list.append(getattr(nn, self._output_activation)())
        
        # Building Sequential Model
        self.net = nn.Sequential(*modules_list)

    
    def _gen_hparam_vec_for_conv(self, hparam, hparam_name, **kwargs):
        return _generate_array_for_hparam(hparam, self._num_conv_blocks, 
                hparam_name=hparam_name, count_if_not_list_name='num_conv_blocks', **kwargs)
    
    def _gen_hparam_vec_for_dense(self, hparam, hparam_name, **kwargs):
        return _generate_array_for_hparam(
            hparam, self._dense_depth, hparam_name=hparam_name, count_if_not_list_name='dense_depth', **kwargs)
    
    def forward(self, inputs):
        return self.net(inputs)



########################################################################################################################

class Dummy(PyTorchSmartModule):
    sample_hparams = {
        'model_name': 'dummy_Pytorch_Smart_Module',
        'l2_reg': 0.0001,
        'batch_size': 16,
        'epochs': 2,
        'validation_data': 0.1,
        'validation_tolerance_epochs': 10,
        'learning_rate': 0.0001,
        'learning_rate_decay_gamma': 0.99,
        'loss_function': 'MSELoss',
        'loss_function_params': None,
        'optimizer': 'Adam',
        'optimizer_params': {'eps': 1e-07},
        'some_new_feature': True
    }
    def __init__(self, hparams:dict=None):
        super(Dummy, self).__init__(hparams)
        self._some_new_feature = self.hparams.get("some_new_feature")
    def forward(self, x):
        return x
    
########################################################################################################################

def test_dummy_model():
    model = Dummy()
    print(model.hparams)
    print(model._some_new_feature)
    print(model._optimizer_params)

def test_calc_image_size():
    size = _calc_image_size(size_in=[28,28,28], kernel_size=3, padding='valid', stride=2, dilation=1)
    print(size)

def test_generate_geometric_array():
    array = _generate_geometric_array(init=256, count=4, direction='up', powers_of_two=True)
    print(array)
    
def test_generate_array_for_hparam():
    array = _generate_array_for_hparam(
        [1,2], count_if_not_list=4, 
        hparam_name='parameter', count_if_not_list_name='its count',
        check_auto=True, init_for_auto=2, powers_of_two_if_auto=True,
        direction_if_auto='up')
    print(array)

def test_dense_block():
    dense_block = Dense_Block(256, 128, 'ReLU', None, 'BatchNorm1d', 'before', None, 0.1)
    print(dense_block)
    x = torch.randn(32,256)
    y = dense_block(x)
    print("Input shape:  ", x.shape)
    print("Output shape: ", y.shape)

def test_conv_block():
    conv_block = Conv_Block(3, 32, conv_dim=2, input_image=[28,28], conv_kernel_size=3, conv_padding='valid', conv_stride=1, conv_dilation=1, conv_params=None, 
                            conv_activation='ReLU', conv_activation_params=None, norm_layer_position='before', norm_layer_type='BatchNorm', norm_layer_params=None, 
                            pool_type='Max', pool_kernel_size=2, pool_padding=0, pool_stride=1, pool_dilation=1, pool_params=None, adaptive_pool_output_size= None, 
                            dropout=0.1, min_image_dim=8)
    print(conv_block)
    x = torch.randn(32,3,28,28)
    y = conv_block(x)
    print("Input shape:  ", x.shape)
    print("Output shape: ", y.shape)
    
def test_conv_network():
    test_pytorch_model_class(Conv_Network)

def test_recurrent_network():
    test_pytorch_model_class(Recurrent_Network)




if __name__ == '__main__':
    
    # test_dummy_model()
    # test_calc_image_size()
    # test_generate_geometric_array()
    # test_generate_array_for_hparam()
    # test_dense_block()
    # test_conv_block()
    # test_conv_network()
    # test_recurrent_network()
    
    
    pass