"""
This script includes some tools and functionality for making tabular datasets from sequential data such as time series.

"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    DTYPE = np.float32
    POWERS_OF_TWO = True
    def __init__(self, 
        input_vec, in_seq_length:int, in_features:int, in_squeezed:bool,
        output_vec=None, out_seq_length:int=1, out_features:int=0, out_squeezed:bool=True,
        data_downsampling_rate:int=1, sequence_downsampling_rate:int=1, 
        input_scaling:bool=True, output_scaling:bool=True,
        input_forward_facing:bool=True, output_forward_facing:bool=True,
        input_include_current_timestep:bool=True, output_include_current_timestep:bool=True,
        input_towards_future:bool=False, output_towards_future:bool=True,
        stacked=True, extern_input_scaler=None, extern_output_scaler=None, scaling:str='standard', verbose=True):
        """Generate dataset to be used later in dataloaders using tabulated sequential data.

        ### Parameters:

        :param `input_vec` (numpy array):               [Nx1] vector of timeseries inputs.
        :param `in_seq_length` (int):                   Input sequence length (sliding time window length)
        :param `in_features` (int)                      Number of features of input
        :param `in_squeezed` (bool):                    Squeezed inputs (batch, seqlen*features) as in ANNs 
                                                        or unsqueezed (batch, seqlen, features) as in LSTM
        :param `output_vec` (numpy array):              [Nx1] vector of timeseries outputs (targets). 
                                                        Default is the same as inputs (for autoregression).
        :param `out_seq_length` (int):                  Output (target) sequence length (sliding time window length).
                                                        Default is 1 (estimation rather than forecasting)
        :param `out_features` (int):                    Number of features of the output target. Default is 0.
        :param `out_squeezed` (bool):                   Squeezed inputs (batch, seqlen*features) as in Dense outputs
                                                        or unsqueezed (batch, seqlen, features) as in RNN outputs
        :param `data_downsampling_rate` (int):          Downsampling rate, if the data has too high sampling rate.
                                                        This integer must be greater than or equal to 1.
        :param `sequence_downsampling_rate` (int):      Downsampling rate, if the sequence has too high sampling rate.
        :param `input_scaling` (bool):                  Whether standard scaling should be applied to inputs. 
                                                        Default is True.
        :param `output_scaling` (bool):                 Whether standard scaling should be applied to output targets. 
                                                        Default is True.
        :param `input_forward_facing` (bool):           Whether the input sequences should be forward facing 
                                                        (timestep t-K towards K) or not (backward facing). 
                                                        Default is True.
        :param `output_forward_facing` (bool):          Whether the output target sequences should be forward facing. 
                                                        Default is True.
        :param `input_include_current_timestep`(bool):  Whether the input sequences include the current time step. 
                                                        Default is True.
        :param `output_include_current_timestep`(bool): Whether the input sequences include the current time step.
                                                        Default is False, for autoregression.
        :param `input_towards_future` (bool):           Whether the input sequences come from future data at 
                                                        every time step rather than past data. Default is False.
        :param `output_towards_future` (bool):          Whether the output sequences come from future data at 
                                                        every time step rather than past data. 
                                                        Default is True for autoregression.
        :param `stacked` (bool):                        If True (default), squeezed columns will be 
                                                        sequence of first feature, 
                                                        then second feature, etc. Otherwise columns will have a
                                                        cascaded arrangement, i.e. features of first time step,
                                                        features of second time step, etc. 
                                                        Only applies to squeezed inputs/outputs.
        :param `extern_input_scaler` (sklearn scaler):  If not None, this scaler will be used to scale the inputs.
                                                        Default is None.
        :param `extern_output_scaler` (sklearn scaler): If not None, this scaler will be used to scale the outputs.
                                                        Default is None.
        :param `scaling` (str):                         Scaling method to use. Default is 'standard'. Other is 'minmax'.


        ### Attributes:

        self.`size` (int): Total number of timesteps in the dataset, after downsampling.
        self.`inscaler` (sklearn.StandardScaler): The standard scaler object, fit to the input data
        self.`outscaler` (sklearn.StandardScaler): The standard scaler object, fit to the output data
        self.`table_in` (numpy matrix): Table of input data, rows are datapoints, columns are features (time steps)
        self.`table_out` (numpy matrix): Table of output data, rows are datapoints, columns are features (time steps)
        self.`shape` (dict): Shape of the dataset object, keys are "in" and "out", 
            which correspond to shapes of numpy matrices representing tabulated inputs and outputs.
        self.`downsampling_rate` (int): Downsampling rate
        self.`in_squeezed` (bool): Whether the input is squeezed (for ANN) or not (for LSTM)
        self.`out_squeezed` (bool): Whether the output is squeezed (for ANN) or not (for LSTM)
        self.`stacked` (bool): Whether the data should be stacked or not
        self.`_scaling` (str): Scaling method to use
        self.`_invec` (numpy array): Array of inputs, after downsampling and scaling
        self.`_outvec` (numpy array): Array of outputs, after downsampling and scaling
        self.`_in_seq_length` (int): Input sequence length (sliding time window length)
        self.`_in_seq_length_ds` (int): Input sequence length (sliding time window length) after downsampling
        self.`_out_seq_length` (int): Output sequence length (sliding time window length)
        self.`_out_seq_length_ds` (int): Output sequence length (sliding time window length) after downsampling
        self.`_in_features` (int):Number of features of input
        self.`_out_features` (int):Number of features of input



        
        """

        # Make output:
        if output_vec is None:
            output_vec = input_vec.astype(self.DTYPE)
        
        # Reshape inputs and outputs
        if len(input_vec.shape)==1:
            input_vec=input_vec.reshape(-1,1).astype(self.DTYPE)
        if len(output_vec.shape)==1:
            output_vec=output_vec.reshape(-1,1).astype(self.DTYPE)

        if out_features==0:
            out_features = output_vec.shape[-1]

        # Reassure size of data.
        assert(len(input_vec.shape)==2 and input_vec.shape[1]==in_features, 
            "All inputs and outputs must be NumData x NumFeatures vectors. \
            Inputs and Outputs must have the same number of rows (datapoints).")
        assert(len(output_vec.shape)==2 and output_vec.shape[1]==out_features, 
            "All inputs and outputs must be NumData x NumFeatures vectors. \
            Inputs and Outputs must have the same number of rows (datapoints).")

        # Reassure inputs and outputs have the same size.
        assert(input_vec.shape[0]==output_vec.shape[0],
            "Inputs and Outputs must have the same time length (number of rows).")

        # Reassure type of scaler.
        assert('standard' in scaling.lower() or 'minmax' in scaling.lower(), 
            "Scaling method must include 'standard' or 'minmax', not case-sensitive.")

        
        # Model type and arrangement
        self.in_squeezed = in_squeezed
        self.out_squeezed = out_squeezed
        self.stacked = stacked
        self._scaling = scaling
        self._in_seq_length = in_seq_length
        self._out_seq_length = out_seq_length
        self._input_is_sequence = (in_seq_length > 1)
        self._output_is_sequence = (out_seq_length > 1)
        self.in_features = in_features
        self.out_features = out_features
        self.data_downsampling_rate = data_downsampling_rate
        self.sequence_downsampling_rate = sequence_downsampling_rate

        # Downsampling inputs and outputs
        if data_downsampling_rate > 1:
            idx = np.arange(start=0, stop=input_vec.shape[0], step=data_downsampling_rate).astype(int)
            input_vec = input_vec[idx,:]
            output_vec = output_vec[idx,:]
        self.size = input_vec.shape[0]
        if data_downsampling_rate > 1:
            # int(2**np.round(np.log2(in_seq_length/data_downsampling_rate))) if self._input_is_sequence else 1
            self._in_seq_length_ds = \
                int(np.round(in_seq_length/data_downsampling_rate)) if self._input_is_sequence else 1
            self._out_seq_length_ds = \
                int(np.round(out_seq_length/data_downsampling_rate)) if self._output_is_sequence else 1
        else:
            self._in_seq_length_ds = in_seq_length
            self._out_seq_length_ds = out_seq_length
        


        # Scaling inputs and outputs
        if input_scaling:
            if extern_input_scaler:
                inscaler = extern_input_scaler
            else:
                inscaler = StandardScaler() if 'standard' in scaling.lower() else MinMaxScaler()
                inscaler.fit(input_vec)
            input_vec = inscaler.transform(input_vec).astype(self.DTYPE)
        else:
            inscaler = None

        if output_scaling:
            if extern_output_scaler:
                outscaler = extern_output_scaler
            else:
                outscaler = StandardScaler() if 'standard' in scaling.lower() else MinMaxScaler()
                outscaler.fit(output_vec)
            output_vec = outscaler.transform(output_vec).astype(self.DTYPE)
        else:
            outscaler = None
        
        self.inscaler = inscaler
        self.outscaler = outscaler
        self._invec = input_vec
        self._outvec = output_vec

        ### Tabulating the inputs and outputs
        # Generate tabular input data out of sequential data
        if self._input_is_sequence:
            inputObj = sliding_window(input_vec, self._in_seq_length_ds, self.sequence_downsampling_rate, 
                forward_facing=input_forward_facing, 
                include_current_step=input_include_current_timestep, squeezed=self.in_squeezed, stacked=self.stacked,
                includes_future_data=input_towards_future, dtype=self.DTYPE)
            table_in = inputObj["table"]
            self._in_seq_length_final = inputObj["seq_len_ds"]
        else:
            table_in = input_vec
            self._in_seq_length_final = 1
        if verbose: print("Input Table Shape: ", table_in.shape)

        # Generate tabular output data out of sequential data
        if self._output_is_sequence:
            outputObj = sliding_window(output_vec, self._out_seq_length_ds, self.sequence_downsampling_rate,
                forward_facing=output_forward_facing, 
                include_current_step=output_include_current_timestep, squeezed=self.out_squeezed, stacked=self.stacked,
                includes_future_data=output_towards_future, dtype=self.DTYPE)
            table_out = outputObj["table"]
            self._out_seq_length_final = outputObj["seq_len_ds"]
        else:
            table_out = output_vec
            self._out_seq_length_final = 1
        if verbose: print("Output Table Shape: ", table_out.shape)

        self.table_in = table_in.astype(self.DTYPE)
        self.table_out = table_out.astype(self.DTYPE)
        self.shape = {"in":self.table_in.shape, "out":self.table_out.shape}
        if verbose: print("Dataset constructed successfully.")
    


    def __len__(self):
        return self._invec.size


    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return (torch.from_numpy(self._table_in[key,...].copy()), 
                torch.from_numpy(self._table_out[key,...].copy()))
        else:
            raise TypeError("Invalid argument type.")



def sliding_window(input_matrix, sequence_length:int, downsampling:int=1, squeezed:bool=True, forward_facing:bool=True, 
        include_current_step:bool=True, stacked:bool=True, includes_future_data:bool=False, dtype=np.float32,
        powers_of_two:bool=True):
        """Generate tabulated data out of a multivariate timeseries sequence. 
        The tabulated data will be the result of a sliding window passing through the sequence.
        The output of the function is typically used for generating input variables 
        for machine learning algorithms such as deep learning.

        ### Args:
            :param `input_matrix` (numpy matrix): Matrix of input data
            :param `sequence_length` (int): Sequence length of the sliding window
            :param `downsampling` (int): Downsampling rate of the input data. Sequence length is divided by this number.
            :param `squeezed` (bool, optional): Squeezed outputs have size (batch, seqlen*features) 
            useful for ANNs whereas unsqueezed outputs have shape (batch, seqlen, features) useful for LSTM. 
            Default is True.
            :param `forward_facing` (bool, optional): Whether the tabulated sequences will be forward-facing
                like t - K to t, or vise versa, like t to t - K. Defaults to True.
            :param `include_current_step` (bool, optional): Whether the current time step is a part of the 
                sequence being processed or not. Defaults to True.
            :param `stacked`: If True (default), ANN output columns will be sequence of first feature, 
                then second feature, etc. Otherwise ANN output matrices will have a
                cascaded arrangement, i.e. features of first time step,
                features of second time step, etc. Only applies to ANN, not LSTM.
            :param `includes_future_data`: If True, the future data will be processed rather than past data.
                This is useful for forecasting.
            :param `dtype`: Data type of the output. Defaults to np.float32.
            :param `powers_of_two`: If True, the sequence length will be rounded to the nearest power of two.

        ### Returns:
            numpy matrix: Matrix of size (NumDataPoints, SeqLength, NumFeatures) for LSTM (unsqueezed), or 
                          (NumDataPoints, SeqLength*NumFeatures) for ANN (squeezed).
        """
        # Processing direction
        if includes_future_data:
            input_matrix = np.flipud(input_matrix).astype(dtype)
            if forward_facing:
                forward_facing = False


        # Calculate Sequence Lengths
        if sequence_length > 1:
            seq_len_ds = sequence_length // downsampling + 1
            if powers_of_two:
                seq_len_ds = 2**(int(np.log2(seq_len_ds)) + 1)
            sequence_length = seq_len_ds * downsampling
        else:
            sequence_length = 1
            downsampling = 1
        
        # Processing inputs
        num_features = input_matrix.shape[-1]
        buffer = np.ones((sequence_length,num_features), dtype=dtype)*input_matrix[0,:].astype(dtype)
        buffer_ds = buffer[::downsampling,...]
        output = []
        for x in input_matrix:
            if include_current_step:
                buffer = np.append(buffer[1:,:],x.reshape(1,-1),axis=0) if forward_facing else \
                    np.append(x.reshape(1,-1),buffer[:-1,:],axis=0)
                buffer_ds = buffer[::downsampling,...]
            output.append(buffer_ds)
            if not include_current_step:
                buffer = np.append(buffer[1:,:],x.reshape(1,-1),axis=0) if forward_facing else \
                    np.append(x.reshape(1,-1),buffer[:-1,:],axis=0)
                buffer_ds = buffer[::downsampling,...]
        if squeezed:
            output = np.vstack(
                [buffer_ds.reshape((1, num_features*seq_len_ds), 
                order='F' if stacked else 'A') for buffer_ds in output])
        else:
            output = np.stack(output, axis=0)
        
        if includes_future_data:
            output = np.flipud(output)
            
        return {"table":output, "seq_len":sequence_length, "ds":downsampling, "seq_len_ds":seq_len_ds}
    


    
if __name__ == '__main__':
    pass