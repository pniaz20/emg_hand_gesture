# %% [markdown]
# # Gesture Recognition with CAPG DB-a Dataset Using 3D CNN with EMGNet Architecture (one subject for testing)
# 
# In this preliminary effort, we will try to perform hand gesture recognition on CAPG DBA dataset.
# We will use the EMGNet architecture and training procedure, but instead of CWT, we will use 3D CNN on sequences of 2D images.
# 
# In this version:
# 
# - EMG data is normalized with the recorded MVC data
# - The **EMGNet** architecture will be used, along with the training procedure.
# - A **3D CNN** architecture will be adopted into the EMGNet architecture.
# - **Raw EMG data** will be used, there will be no preproccessing or feature engineering.
# - **Training data:** 17 subjects
# - **Test data:** 1 subject
# - K-fold cross-validation will be performed.
# 
# **NOTE** This code has been tested with:
# ```
#     numpy version:        1.23.5
#     scipy version:        1.9.3
#     sklearn version:      1.2.0
#     seaborn version:      0.12.1
#     pandas version:       1.5.2
#     torch version:        1.12.1+cu113
#     matplotlib version:   3.6.2
#     CUDA version:         11.2
# ```

# %% [markdown]
# ## 1- Preliminaries

# %% [markdown]
# ### Imports

# %%
import sys, os
direc = os.getcwd()
print("Current Working Directory is: ", direc)
KUACC = False
if "scratch" in direc: # We are using the cluster
    KUACC = True
    homedir = os.path.expanduser("~")
    os.chdir(os.path.join(homedir,"comp541-project/capg_3dcnn/"))
    direc = os.getcwd()
    print("Current Working Directory is now: ", direc)
sys.path.append("../src/")
sys.path.append("../data/")
import torch
import torch.nn as nn
from datasets_torch import *
from models_torch import *
from utils_torch import *
from cwt import calculate_wavelet_vector, calculate_wavelet_dataset
from datetime import datetime
import pandas as pd
import numpy as np
import scipy as sp
import sklearn
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
import statistics
import json
from IPython.display import display

# Print versions
print("numpy version:       ", np.__version__)
print("scipy version:       ", sp.__version__)
print("sklearn version:     ", sklearn.__version__)
print("seaborn version:     ", sns.__version__)
print("pandas version:      ", pd.__version__)
print("torch version:       ", torch.__version__)
print("matplotlib version:  ", matplotlib.__version__)


# Checking to see if CUDA is available for us
print("Checking to see if PyTorch recognizes GPU...")
print(torch.cuda.is_available())

# Whether to use latex rendering in plots throughout the notebook
USE_TEX = False
FONT_SIZE = 12

# Setting matplotlib plotting variables
if USE_TEX:
    plt.rcParams.update({
        "text.usetex": True,
        "font.size": FONT_SIZE,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"]
    })
else:
    plt.rcParams.update({
        "text.usetex": False,
        "font.size": FONT_SIZE,
        "font.family": "serif",
        "font.serif": ["Times New Roman"]
    })

# Do not plot figures inline (only useful for cluster)
# %matplotlib

# %% [markdown]
# ## 2- Hyperparameters and Settings

# %% [markdown]
# ### General settings of the study

# %%
k_fold_study = {
    'code':'capg_3dcnn/capg_dba_v003',
    'package':'torch',
    'dataset':'capg',
    'subdataset':'dba',
    "training_accuracies": [],
    "validation_accuracies": [],
    "testset_accuracies": [],
    "history_training_loss": [],
    "history_training_metrics": [],
    "history_validation_loss": [],
    "history_validation_metrics": [],
    "preprocessing":None,
    "feature_engineering":None,
    "k_fold_mode":"1 subject for testing",
    "global_downsampling":10
}

# %%
hparams = {
    "model_name": autoname("capg_3dcnn_dba_v003"),
    # General hyperparameters
    "in_features": 128,
    "out_features": 1,
    # Sequence hyperparameters
    "in_seq_len_sec": 0.16,
    "out_seq_len_sec": 0,
    "data_sampling_rate_Hz": 1000.0,
    "data_downsampling": 5,
    "sequence_downsampling": 1,
    "in_seq_len": 0,
    "out_seq_len": 0,
    # Convolution blocks
    "num_conv_blocks": 4,
    "conv_dim": 3,
    "conv_params": None,
    "conv_channels": [16, 32, 32, 64],
    "conv_kernel_size": 3,
    "conv_padding": "same",
    "conv_stride": 1,
    "conv_dilation": 1,
    "conv_activation": "ReLU",
    "conv_activation_params": None,#{"negative_slope": 0.1},
    "conv_norm_layer_type": "BatchNorm",
    "conv_norm_layer_position": "before",
    "conv_norm_layer_params": None,
    "conv_dropout": None,
    "pool_type": [None, None, None, "AdaptiveAvg"],
    "pool_kernel_size": 2,
    "pool_padding": 0,
    "pool_stride": 1,
    "pool_dilation": 1,
    "pool_params": None,
    "min_image_size": 1,
    "adaptive_pool_output_size": [1,1,1],
    # Fully connected blocks
    "dense_width": "auto",
    "dense_depth": 0,
    "dense_activation": "ReLU",
    "dense_activation_params": None,
    "output_activation": None,
    "output_activation_params": None,
    "dense_norm_layer_type": None,
    "dense_norm_layer_position": None,
    "dense_norm_layer_params": None,
    "dense_dropout": None,
    # Training procedure
    "l2_reg": 0.0001,
    "batch_size": 512,
    "epochs": 1,
    "validation_data": [0.05,'testset'],
    "validation_tolerance_epochs": 1000,
    "learning_rate": 0.01,
    "learning_rate_decay_gamma": 0.9,
    "loss_function": "CrossEntropyLoss",
    "optimizer": "Adam",
    "optimizer_params": None
}

# %% [markdown]
# ## 3- Data Processing

# %% [markdown]
# ### Load and concatenate data

# %%
data_dir = "../data/CAPG/parquet"
def load_single_capg_dataset(data_dir, db_str:str="dba"):
    data_lst = []
    for i,file in enumerate(os.listdir(data_dir)):
        if file.endswith(".parquet") and db_str in file:
            print("Loading file: ", file)
            data_lst.append(pd.read_parquet(os.path.join(data_dir, file)))
    data = pd.concat(data_lst, axis=0, ignore_index=True)
    return data
dba_tot = load_single_capg_dataset(data_dir, db_str="dba")
dba_mvc = dba_tot.loc[dba_tot["gesture"].isin([100, 101])]
dba = dba_tot.loc[~dba_tot["gesture"].isin([100, 101])]
print("dba_tot shape: ", dba_tot.shape)
print("dba_mvc shape: ", dba_mvc.shape)
print("dba shape: ", dba.shape)
print("Columns: ")
print(dba_tot.columns)
print("Description: ")
print(dba.iloc[:,:3].describe())

# %% [markdown]
# ### Normalize EMG Data
# 
# Here the recorded MVC values will be used for normalizaing EMG data

# %%
max_mvc = dba_mvc.iloc[:,3:].max(axis=0)
del dba_mvc
# print("max_mvc for 5 first channels: ")
# print(max_mvc[:5])
# print("shape of max_mvc: ", max_mvc.shape)
# print("max of dba before normalization: (first five)")
# print(dba.iloc[:,3:].max(axis=0)[:5])
dba.iloc[:,3:] = dba.iloc[:,3:].div(max_mvc, axis=1)
# print("max of dba_norm after normalization: ")
# print(dba_norm.iloc[:,3:].max(axis=0)[:5])

# %% [markdown]
# ## 4- k-fold study

# %% [markdown]
# ### EMGNet model

# %%

class EMGNet(PyTorchSmartModule):
    def __init__(self, hparams):
        super(EMGNet, self).__init__(hparams)
        self.prep_block = nn.Sequential(
            nn.BatchNorm3d(1),
            nn.ReLU()
        )
        self.main_block = Conv_Network(hparams)
    
    def forward(self, x):
        x = self.prep_block(x)
        x = self.main_block(x)
        return x

# %% [markdown]
# ### Perform k-fold cross-validation study

# %%
# Define input columns
input_cols = list(dba.iloc[:,3:].columns)

# Hard-code total number of subjects
num_subjects = dba['subject'].nunique()

ds = k_fold_study['global_downsampling']


for k in range(num_subjects):
    
    print("\n#################################################################")
    print("Using subject %d for testing ..." % (k+1))
    print("#################################################################\n")
    
    subj_for_testing = [k+1]
    
    # Un-Correct the output feature count (this is buggy behavior and should be fixed)
    hparams['out_features'] = 1
    
    # Get processed data cell
    # CWT: N x C x L --> N x C x H x L
    print("Generating data cell ...")
    data_processed = generate_cell_array(
        dba, hparams,
        subjects_column="subject", conditions_column="gesture", trials_column="trial",
        input_cols=input_cols, output_cols=["gesture"], specific_conditions=None,
        input_preprocessor=None,
        output_preprocessor=None,
        # Convert N x L x C data to N x C x L and then to N x C' x D x H x W where C'=1, D=L, H=8, W=16
        input_postprocessor=lambda arr: arr.reshape(arr.shape[0], 1, arr.shape[1], 8, 16),
        output_postprocessor = lambda arr:(arr-1).squeeze(), # torch CrossEntropyLoss needs (N,) array of 0-indexed class labels
        subjects_for_testing=subj_for_testing, 
        trials_for_testing=None,
        input_scaling=False, output_scaling=False, input_forward_facing=True, output_forward_facing=True, 
        data_squeezed=False,
        input_towards_future=False, output_towards_future=False, 
        output_include_current_timestep=True,
        use_filtered_data=False, #lpcutoff=CUTOFF, lporder=FILT_ORDER, lpsamplfreq=SAMPL_FREQ,
        return_data_arrays_orig=False,
        return_data_arrays_processed=False,
        return_train_val_test_arrays=False,
        return_train_val_test_data=True,
        verbosity=1
    )
    
    # Correct the output feature count (this is buggy behavior and should be fixed)
    hparams['out_features'] = 8
    
    print("Extracting downsampled input and output data from the datacell ...")
    # Inputs MUST have correct shape
    x_train = data_processed["x_train"][::ds]
    x_val = data_processed["x_val"][::ds]
    x_test = data_processed["x_test"][::ds]
    # Outputs MUST be zero-indexed class labels
    y_train = data_processed["y_train"][::ds]
    y_val = data_processed["y_val"][::ds]
    y_test = data_processed["y_test"][::ds]
    print("x_train shape: ", x_train.shape)
    print("x_val shape: ", x_val.shape)
    print("x_test shape: ", x_test.shape)
    print("y_train shape: ", y_train.shape)
    print("y_val shape: ", y_val.shape)
    print("y_test shape: ", y_test.shape)
    del data_processed
    # Make datasets from training, validation and test sets
    print("Generating the TensorDataset objects ...")
    train_set = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    val_set = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long())
    test_set = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())
    
    # If it is the first iteration of the loop, save the hyperparameters dictionary in the k-fold study dictionary
    if k==0:
        k_fold_study['hparams'] = hparams
    
    # Construct model
    print("Constructing the model ...")
    hparams['input_shape'] = list(x_train.shape[1:])
    hparams['output_shape'] = [8]
    print("Model input shape: ", hparams['input_shape'])
    print("Model output shape: ", hparams['output_shape'])
    model = EMGNet(hparams)
    if k == 0: print(model)
    
    # Train model
    print("Training the model ...")
    # history = train_pytorch_model(
    #     model, [train_set, val_set], batch_size=1024, loss_str='crossentropy', optimizer_str='adam', 
    #     optimizer_params={'weight_decay':0.0001}, loss_function_params=None, learnrate=0.1, 
    #     learnrate_decay_gamma=0.95, epochs=200, validation_patience=1000000, 
    #     verbose=1, script_before_save=True, saveto=None, num_workers=0)
    history = model.train_model([train_set, val_set], verbose=1)    
    
    # Update relevant fields in the k-fold study dictionary
    print("Updating the dictinoary for logging ...")
    k_fold_study['history_training_loss'].append(history["training_loss"])
    k_fold_study["history_validation_loss"].append(history["validation_loss"])
    k_fold_study["history_training_metrics"].append(history["training_metrics"])
    k_fold_study["history_validation_metrics"].append(history["validation_metrics"])
    k_fold_study["training_accuracies"].append(history["training_metrics"][-1])
    k_fold_study["validation_accuracies"].append(history["validation_metrics"][-1])
    
    # Evaluate the model on the test set
    print("Evaluating the model on the test set ...")
    # results = evaluate_pytorch_model(model, test_set, loss_str='crossentropy', loss_function_params=None,
    # batch_size=1024, device_str="cuda", verbose=True, num_workers=0)
    results = model.evaluate_model(test_set, verbose=True)
    k_fold_study["testset_accuracies"].append(results["metrics"])
    print("Done with this fold of the K-fold study.")

print("Done with the K-fold study.")

# %% [markdown]
# ### Saving k-fold study

# %%
print("Dumping the JSON file ...")
json.dump(k_fold_study, open(make_path("../results/"+hparams['model_name']+"/k_fold_study.json"), "w"), indent=4)
print("Saved the JSON file.")

# %% [markdown]
# ### Saving general statistics

# %%
print("Saving the general statistics ...")
trn_acc_arr = np.array(k_fold_study["training_accuracies"])
val_acc_arr = np.array(k_fold_study["validation_accuracies"])
tst_acc_arr = np.array(k_fold_study["testset_accuracies"])
general_dict = {"training_accuracy":trn_acc_arr, "validation_accuracy":val_acc_arr, "testset_accuracy":tst_acc_arr}
general_results = pd.DataFrame(general_dict)
print("Description of general results:")
general_results_describe = general_results.describe()
display(general_results_describe)
general_results_describe.to_csv(
    make_path("../results/"+hparams['model_name']+"/general_results.csv"), header=True, index=True)
print("Saved general statistics.")

# %% [markdown]
# ### Plotting training histories

# %%
# import numpy as np
# import json
# import pandas as pd

# %%
# k_fold_study = json.load(open("../results/capg_replica_dba_v002_2023_01_07_20_07_25/k_fold_study.json", "r"))

# %%
print("Plotting the taining curve ...")
train_loss = np.array(k_fold_study["history_training_loss"])
val_loss = np.array(k_fold_study["history_validation_loss"])
train_acc = np.array(k_fold_study["history_training_metrics"])
val_acc = np.array(k_fold_study["history_validation_metrics"])

print("Shape of train_loss: ", train_loss.shape)

train_loss_mean = np.mean(train_loss, axis=0)
train_loss_std = np.std(train_loss, axis=0)# / 2
val_loss_mean = np.mean(val_loss, axis=0)
val_loss_std = np.std(val_loss, axis=0)# / 2
train_acc_mean = np.mean(train_acc, axis=0)
train_acc_std = np.std(train_acc, axis=0)# / 2
val_acc_mean = np.mean(val_acc, axis=0)
val_acc_std = np.std(val_acc, axis=0)# / 2

print("Shape of train_loss_mean: ", train_loss_mean.shape)
print("Shape of train_loss_std: ", train_loss_std.shape)

epochs = train_loss_mean.shape[0]
epochs = np.arange(1, epochs+1)
plt.figure(figsize=(8,8), dpi=100)
plt.subplot(2,1,1)
plt.grid(True)
plt.plot(epochs, train_loss_mean, label="Training", color="blue")
plt.fill_between(epochs, train_loss_mean-train_loss_std, train_loss_mean+train_loss_std, 
                 color='blue', alpha=0.2)
plt.plot(epochs, val_loss_mean, label="Validation", color="orange")
plt.fill_between(epochs, val_loss_mean-val_loss_std, val_loss_mean+val_loss_std,
                 color='orange', alpha=0.2)
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.subplot(2,1,2)
plt.grid(True)
plt.plot(epochs, train_acc_mean, color="blue")
plt.fill_between(epochs, train_acc_mean-train_acc_std, train_acc_mean+train_acc_std,
                 color='blue', alpha=0.2)
plt.plot(epochs, val_acc_mean, color="orange")
plt.fill_between(epochs, val_acc_mean-val_acc_std, val_acc_mean+val_acc_std,
                 color='orange', alpha=0.2)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.subplots_adjust(hspace=0.2)
plt.savefig(make_path("../results/"+k_fold_study['hparams']['model_name']+"/training_history.png"), dpi=300)

print("Done plotting the training curve.")
print("ALL DONE. GOOD BYE!")


