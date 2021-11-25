# coding=utf-8
# Copyright 2021 DAF Trucks NV.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
from libs.tft_model import get_decoder_mask
import os
import gc
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import libs.utils as utils
from libs.loss.quantile_loss_calculator import QuantileLossCalculator

from data_formatters.base import InputTypes


print(tf.version.VERSION)

def linear_layer(size,
                activation = None,
                use_time_distributed = False,
                use_bias = True):
    """Returns simple Keras linear layer.

    Args:
        size: Output size
        activation: Activation function to apply if required
        use_time_distributed: Whether to apply layer across time
        use_bias: Whether bias should be included in layer
    """
    linear = tf.keras.layers.Dense(size,
                                activation=activation,
                                use_bias=use_bias)
    if use_time_distributed:
        linear = tf.keras.layers.TimeDistributed(linear)
    return linear

def apply_gating_layer(x,
                    hidden_layer_size,
                    dropout_rate = None,
                    use_time_distributed = None,
                    activation = None):
    """Applies a Gated Linear Unit (GLU) to an input.

    Args:
        x: Input to gating layer
        hidden_layer_size: Dimension of GLU
        dropout_rate: Dropout rate to apply if any
        use_time_distributed: Whether to apply across time
        activation: Activation function to apply to the linear feature transform if
        necessary

    Returns:
        Tuple of tensors for: (GLU output, gate)
    """
    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    activation_layer = tf.keras.layers.Dense(hidden_layer_size, activation = activation)
    gated_layer = tf.keras.layers.Dense(hidden_layer_size, activation = 'sigmoid')
    if use_time_distributed:
        activation_layer_output = tf.keras.layers.TimeDistributed(activation_layer)(x)
        gated_layer_output = tf.keras.layers.TimeDistributed(gated_layer)(x)
    else:
        activation_layer_output = activation_layer(x)
        gated_layer_output = gated_layer(x)
    
    return tf.keras.layers.Multiply()([activation_layer_output, gated_layer_output]), gated_layer_output

def add_and_norm(x, name=None):
    """Applies skip connection followed by layer normalisation.

    Args:
        x: List of inputs to sum for skip connection

    Returns:
        Tensor output from layer.
    """
    add_layer_output = tf.keras.layers.Add()(x)
    norm_layer_output = tf.keras.layers.LayerNormalization(name=name)(add_layer_output)
    return norm_layer_output

def gated_residual_network(x,
                        hidden_layer_size,
                        output_size = None,
                        dropout_rate = None,
                        use_time_distributed = True,
                        additional_context = None,
                        return_gate = False,
                        name=None):
    """Applies the gated residual network (GRN) as defined in paper.

    Args:
        x: Network inputs
        hidden_layer_size: Internal state size
        output_size: Size of output layer
        dropout_rate: Dropout rate if dropout is applied
        use_time_distributed: Whether to apply network across time dimension
        additional_context: Additional context vector to use if relevant
        return_gate: Whether to return GLU gate for diagnostic purposes

    Returns:
        Tuple of tensors for: (GRN output, GLU gate)
    """
    # Setup skip connection
    if output_size is None:
        output_size = hidden_layer_size
        residual_connection = x
    else:
        linear = tf.keras.layers.Dense(output_size)
        if use_time_distributed:
            linear = tf.keras.layers.TimeDistributed(linear)
        residual_connection = linear(x)
    
    primary_input_hidden_layer = linear_layer(hidden_layer_size,
                                            activation=None,
                                            use_time_distributed=use_time_distributed)(x)
    if additional_context is not None:
        context_input_layer = linear_layer(hidden_layer_size,
                                        activation=None,
                                        use_time_distributed=use_time_distributed,
                                        use_bias=False)(additional_context)
        primary_input_hidden_layer += context_input_layer
    
    elu_layer_output = tf.keras.layers.ELU()(primary_input_hidden_layer)

    n1_dense_layer_output = linear_layer(hidden_layer_size,
                                        activation=None,
                                        use_time_distributed=use_time_distributed)(elu_layer_output)
    
    gating_layer, gate = apply_gating_layer(n1_dense_layer_output,
                                            output_size,
                                            dropout_rate=dropout_rate,
                                            use_time_distributed=use_time_distributed,
                                            activation=None)
    if return_gate:
        return add_and_norm([residual_connection,gating_layer], name=name), gate
    else:
        return add_and_norm([residual_connection, gating_layer], name=name)

class ScaledDotProductAttention():
    """Defines scaled dot product attention layer (Equation 10).

    Attributes:
        dropout: Dropout rate to use
        activation: Normalisation function for scaled dot product attention (e.g.
        softmax by default)
    """
    def __init__(self, attn_dropout=0.0):
        self.dropout = tf.keras.layers.Dropout(attn_dropout)
        self.activation = tf.keras.layers.Softmax()

    def __call__(self, q, k, v, mask):
        """Applies scaled dot product attention.

        Args:
        q: Queries
        k: Keys
        v: Values
        mask: Masking if required -- sets softmax to very large value

        Returns:
        Tuple of (layer outputs, attention weights)
        """
        root_squared_attn_size = tf.math.sqrt(tf.cast(k.shape[-1], dtype=tf.float32))
        # root_squared_attn_size = tf.math.sqrt(tf.cast(tf.shape(k)[-1], dtype=tf.float32))
        attn = tf.keras.layers.Lambda(lambda x: tf.keras.backend.batch_dot(x[0], x[1], axes = [2,2]) / root_squared_attn_size)([q,k])
        if mask is not None:
            mask_layer = tf.keras.layers.Lambda(lambda x: (-1e+9) * (1. - tf.cast(x, dtype=tf.float32)))(mask)
            attn = tf.keras.layers.Add()([attn,mask_layer])
        softmax_output = self.activation(attn)
        dropout_output = self.dropout(softmax_output)
        output = tf.keras.layers.Lambda(lambda x: tf.keras.backend.batch_dot(x[0],x[1]))([dropout_output,v])
        return output, dropout_output

class InterpretableMultiHeadAttention():
    """Defines interpretable multi-head attention layer.

    Attributes:
        n_head: Number of heads
        d_k: Key/query dimensionality per head
        d_v: Value dimensionality
        dropout: Dropout rate to apply
        qs_layers: List of queries across heads
        ks_layers: List of keys across heads
        vs_layers: List of values across heads
        attention: Scaled dot product attention layer
        w_o: Output weight matrix to project internal state to the original TFT
        state size
    """
    def __init__(self, n_head, d_model, dropout):
        """Initialises layer.

        Args:
        n_head: Number of heads
        d_model: TFT state dimensionality
        dropout: Dropout discard rate
        """

        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout

        self.queries_layers = []
        self.keys_layers = []
        self.values_layers = []

        values_layer = tf.keras.layers.Dense(d_v, use_bias=False)

        for _ in range(n_head):
            self.queries_layers.append(tf.keras.layers.Dense(d_k, use_bias=False))
            self.keys_layers.append(tf.keras.layers.Dense(d_k, use_bias=False))
            self.values_layers.append(values_layer)
        
        self.attention = ScaledDotProductAttention()
        self.weights_output = tf.keras.layers.Dense(d_model, use_bias=False)

    def __call__(self, q, k, v, mask=None):
        """Applies interpretable multihead attention.

        Using T to denote the number of time steps fed into the transformer.

        Args:
        q: Query tensor of shape=(?, T, d_model)
        k: Key of shape=(?, T, d_model)
        v: Values of shape=(?, T, d_model)
        mask: Masking if required with shape=(?, T, T)

        Returns:
        Tuple of (layer outputs, attention weights)
        """

        n_head = self.n_head

        heads = []
        attentions = []
        for i in range(n_head):
            queries = self.queries_layers[i](q)
            keys = self.keys_layers[i](k)
            values = self.values_layers[i](v)
            head, attention = self.attention(queries, keys, values, mask)

            head_dropout = tf.keras.layers.Dropout(self.dropout)(head)
            heads.append(head_dropout)
            attentions.append(attention)
        head = tf.keras.backend.stack(heads) if n_head > 1 else heads[0]
        attention = tf.keras.backend.stack(attentions)

        outputs = tf.keras.backend.mean(head, axis = 0) if n_head > 1 else head
        outputs = self.weights_output(outputs)
        outputs = tf.keras.layers.Dropout(self.dropout)(outputs)

        return outputs, attention

class TFTDataCache(object):
    """Caches data for the TFT."""

    _data_cache = {}

    @classmethod
    def update(cls, data, key):
        """Updates cached data.

        Args:
        data: Source to update
        key: Key to dictionary location
        """
        cls._data_cache[key] = data

    @classmethod
    def get(cls, key):
        """Returns data stored at key location."""
        return cls._data_cache[key].copy()

    @classmethod
    def contains(cls, key):
        """Retuns boolean indicating whether key is present in cache."""

        return key in cls._data_cache

# TFT model definitions.
class TemporalFusionTransformer(object):
    """Defines Temporal Fusion Transformer.

    Attributes:
    name: Name of model
    time_steps: Total number of input time steps per forecast date (i.e. Width
        of Temporal fusion decoder N)
    input_size: Total number of inputs
    output_size: Total number of outputs
    category_counts: Number of categories per categorical variable
    n_multiprocessing_workers: Number of workers to use for parallel
        computations
    column_definition: List of tuples of (string, DataType, InputType) that
        define each column
    quantiles: Quantiles to forecast for TFT
    use_cudnn: Whether to use Keras CuDNNLSTM or standard LSTM layers
    hidden_layer_size: Internal state size of TFT
    dropout_rate: Dropout discard rate
    max_gradient_norm: Maximum norm for gradient clipping
    learning_rate: Initial learning rate of ADAM optimizer
    minibatch_size: Size of minibatches for training
    num_epochs: Maximum number of epochs for training
    early_stopping_patience: Maximum number of iterations of non-improvement
        before early stopping kicks in
    num_encoder_steps: Size of LSTM encoder -- i.e. number of past time steps
        before forecast date to use
    num_stacks: Number of self-attention layers to apply (default is 1 for basic
        TFT)
    num_heads: Number of heads for interpretable mulit-head attention
    model: Keras model for TFT
    """

    def __init__(self, raw_params):
        """Builds TFT from parameters.

        Args:
        raw_params: Parameters to define TFT
        use_cudnn: Whether to use CUDNN GPU optimised LSTM
        """

        self.name = self.__class__.__name__

        params = dict(raw_params)

        self.time_steps = int(params['total_time_steps'])
        self.input_size = int(params['input_size'])
        self.output_size = int(params['output_size'])
        self.category_counts = json.loads(str(params['category_counts']))
        self.n_multiprocessing_workers = int(params['multiprocessing_workers'])

        # Relevant indices for TFT
        self._input_obs_loc = json.loads(str(params['input_obs_loc']))
        self._static_input_loc = json.loads(str(params['static_input_loc']))
        self._known_regular_input_idx = json.loads(
            str(params['known_regular_inputs']))
        self._known_categorical_input_idx = json.loads(
            str(params['known_categorical_inputs']))

        self.column_definition = params['column_definition']

        # Network params
        self.quantiles = [0.1, 0.5, 0.9]
        self.hidden_layer_size = int(params['hidden_layer_size'])
        self.dropout_rate = float(params['dropout_rate'])
        self.max_gradient_norm = float(params['max_gradient_norm'])
        self.learning_rate = float(params['learning_rate'])
        self.minibatch_size = int(params['minibatch_size'])
        self.num_epochs = int(params['num_epochs'])
        self.early_stopping_patience = int(params['early_stopping_patience'])

        self.num_encoder_steps = int(params['num_encoder_steps'])
        self.num_stacks = int(params['stack_size'])
        self.num_heads = int(params['num_heads'])

        # Serialisation options
        self._temp_folder = os.path.join(params['model_folder'], 'tmp')
        self.reset_temp_folder()

        # Extra components to store Tensorflow nodes for attention computations
        self._input_placeholder = None
        self._attention_components = None
        self._prediction_parts = None

        #Build model
        self.model = self.build_model()

    def get_tft_embeddings(self, all_inputs):
        """Transforms raw inputs to embeddings.

        Applies linear transformation onto continuous variables and uses embeddings
        for categorical variables.

        Args:
        all_inputs: Inputs to transform

        Returns:
        Tensors for transformed inputs.
        """

        time_steps = self.time_steps

        # Sanity checks
        for i in self._known_regular_input_idx:
            if i in self._input_obs_loc:
                raise ValueError('Observation cannot be known a priori!')
        for i in self._input_obs_loc:
            if i in self._static_input_loc:
                raise ValueError('Observation cannot be static!')

        if all_inputs.get_shape().as_list()[-1] != self.input_size:
            raise ValueError(
                'Illegal number of inputs! Inputs observed={}, expected={}'.format(
                    all_inputs.get_shape().as_list()[-1], self.input_size))

        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size - num_categorical_variables

        embedding_sizes = [self.hidden_layer_size for i, size in enumerate(self.category_counts)]

        embeddings = []
        for i in range(num_categorical_variables):
            embedding = tf.keras.Sequential([
                tf.keras.layers.InputLayer([time_steps]),
                tf.keras.layers.Embedding(
                    self.category_counts[i],
                    embedding_sizes[i],
                    input_length=time_steps,
                    dtype=tf.float32
                )
            ])
            embeddings.append(embedding)
        
        regular_inputs, categorical_inputs = all_inputs[:,:,:num_regular_variables], all_inputs[:,:,num_regular_variables:]

        embedded_inputs = [embeddings[i](categorical_inputs[Ellipsis, i]) for i in range(num_categorical_variables)] #Embadding layers applied to categorical inputs

        # Static inputs
        if self._static_input_loc:
            static_inputs = [tf.keras.layers.Dense(self.hidden_layer_size)(
                regular_inputs[:,0,i:i+1]) for i in range(num_regular_variables)
                if i in self._static_input_loc
            ] + [embedded_inputs[i][:,0,:]
                for i in range(num_categorical_variables)
                if i + num_regular_variables in self._static_input_loc]
            static_inputs = tf.stack(static_inputs, axis = 1)
        
        else:
            static_inputs = None

        def convert_real_to_embedding(x):
            return tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.hidden_layer_size)
            )(x)

        # Targets
        observed_inputs = tf.stack([
            convert_real_to_embedding(regular_inputs[Ellipsis, i:i+1])
            for i in self._input_obs_loc
        ], axis = -1)

        #Observed (a priori unknow) inputs
        wired_embeddings = []
        for i  in range(num_categorical_variables):
            if i not in self._known_categorical_input_idx and i + num_regular_variables not in self._input_obs_loc:
                embedding = embeddings[i](categorical_inputs[:,:,i])
                wired_embeddings.append(embedding)
        
        unknown_inputs = []
        for i in range(regular_inputs.shape[-1]):
            if i not in self._known_regular_input_idx and i not in self._input_obs_loc:
                embedding = convert_real_to_embedding(regular_inputs[Ellipsis, i:i+1])
                unknown_inputs.append(embedding)
        
        if unknown_inputs + wired_embeddings:
            unknown_inputs = tf.stack(unknown_inputs + wired_embeddings, axis = -1)
        else:
            unknown_inputs = None
        
        # A priori known inputs
        known_regular_inputs = [
            convert_real_to_embedding(regular_inputs[Ellipsis,i:i+1])
            for i in self._known_regular_input_idx
            if i not in self._static_input_loc
        ]
        known_categorical_inputs = [
            embedded_inputs[i]
            for i in self._known_categorical_input_idx
            if i + num_regular_variables not in self._static_input_loc
        ]

        known_combined_layer = tf.stack(
            known_regular_inputs + known_categorical_inputs, axis = -1
        )

        return unknown_inputs, known_combined_layer, observed_inputs, static_inputs

    def _get_single_col_by_type(self, input_type):
        """Returns name of single column for input type."""

        return utils.get_single_col_by_input_type(input_type, self.column_definition)

    def training_data_cached(self):
        """Returns boolean indicating if training data has been cached."""

        return TFTDataCache.contains('train') and TFTDataCache.contains('valid')

    def cache_batched_data(self, data, cache_key, num_samples=-1):
        """Batches and caches data once for using during training.

        Args:
            data: Data to batch and cache
            cache_key: Key used for cache
            num_samples: Maximum number of samples to extract (-1 to use all data)
        """

        if num_samples > 0:
            TFTDataCache.update(
                self._batch_sampled_data(data, max_samples=num_samples), cache_key)
        else:
            TFTDataCache.update(self._batch_data(data), cache_key)

        print('Cached data "{}" updated'.format(cache_key))

    def _batch_sampled_data(self, data, max_samples):
        """Samples segments into a compatible format.

        Args:
        data: Sources data to sample and batch
        max_samples: Maximum number of samples in batch

        Returns:
        Dictionary of batched data with the maximum samples specified.
        """

        if max_samples < 1:
            raise ValueError(
            'Illegal number of samples specified! samples={}'.format(max_samples))

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)

        data.sort_values(by=[id_col, time_col], inplace=True)

        print('Getting valid sampling locations.')
        valid_sampling_locations = []
        split_data_map = {}
        for identifier, df in data.groupby(id_col):
            print('Getting locations for {}'.format(identifier))
            num_entries = len(df)
            if num_entries >= self.time_steps:
                valid_sampling_locations += [
                (identifier, self.time_steps + i)
                for i in range(num_entries - self.time_steps + 1)
            ]
            split_data_map[identifier] = df

        inputs = np.zeros((max_samples, self.time_steps, self.input_size))
        outputs = np.zeros((max_samples, self.time_steps, self.output_size))
        time = np.empty((max_samples, self.time_steps, 1), dtype=object)
        identifiers = np.empty((max_samples, self.time_steps, 1), dtype=object)

        if max_samples > 0 and len(valid_sampling_locations) > max_samples:
            print('Extracting {} samples...'.format(max_samples))
            ranges = [
            valid_sampling_locations[i] for i in np.random.choice(
                len(valid_sampling_locations), max_samples, replace=False)
            ]
        else:
            print('Max samples={} exceeds # available segments={}'.format(
                max_samples, len(valid_sampling_locations)))
            ranges = valid_sampling_locations

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [
            tup[0]
            for tup in self.column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        for i, tup in enumerate(ranges):
            if ((i + 1) % 10000) == 0:
                print(i + 1, 'of', max_samples, 'samples done...')
            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx - self.time_steps:start_idx]
            inputs[i, :, :] = sliced[input_cols]
            outputs[i, :, :] = sliced[[target_col]]
            time[i, :, 0] = sliced[time_col]
            identifiers[i, :, 0] = sliced[id_col]

        sampled_data = {
            'inputs': inputs,
            'outputs': outputs[:, self.num_encoder_steps:, :],
            'active_entries': np.ones_like(outputs[:, self.num_encoder_steps:, :]),
            'time': time,
            'identifier': identifiers
        }

        return sampled_data

    def _batch_data(self, data): # TODO Change to Tensorflow data input
        """Batches data for training.

        Converts raw dataframe from a 2-D tabular format to a batched 3-D array
        to feed into Keras model.

        Args:
            data: DataFrame to batch

        Returns:
            Batched Numpy array with shape=(?, self.time_steps, self.input_size)
        """
        # Functions.
        def _batch_single_entity(input_data):
            time_steps = len(input_data)
            lags = self.time_steps
            x = input_data.values
            if time_steps >= lags:
                return np.stack(
                    [x[i:time_steps - (lags - 1) + i, :] for i in range(lags)], axis = 1
                )
            else:
                return None

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [
            tup[0]
            for tup in self.column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        data_map = {}
        for _, sliced in data.groupby(id_col):
            col_mappings = {
                'identifier': [id_col],
                'time': [time_col],
                'outputs': [target_col],
                'inputs': input_cols
            }

            for k in col_mappings:
                cols = col_mappings[k]
                arr = _batch_single_entity(sliced[cols].copy())

                if k not in data_map:
                    data_map[k] = [arr]
                else:
                    data_map[k].append(arr)

        # Combine all data
        for k in data_map:
            data_map[k] = np.concatenate(data_map[k], axis = 0)

        # Shorten target so we only get decoder steps
        data_map['outputs'] = data_map['outputs'][:, self.num_encoder_steps:,:]

        active_entries = np.ones_like(data_map['outputs'])
        if 'active_entries' not in data_map:
            data_map['active_entries'] = active_entries
        else:
            data_map['active_entries'].append(active_entries)
        
        return data_map

    def _get_active_locations(self, x):
        """Formats sample weights for Keras training."""
        return (np.sum(x, axis=-1) > 0.0) * 1.0
    
    def _build_base_graph(self):
        """Returns graph defining layers of the TFT."""

        # Size definitions.
        time_steps = self.time_steps
        combined_input_size = self.input_size
        encoder_steps = self.num_encoder_steps

        # Inputs
        all_inputs = tf.keras.layers.Input(
            shape = (
                time_steps,
                combined_input_size
            )
        )

        unknown_inputs, known_combined_layer, observed_inputs, static_inputs = self.get_tft_embeddings(all_inputs)

        # Isolate known and observed historical inputs
        if unknown_inputs is not None:
            historical_inputs = tf.concat([
                unknown_inputs[:, :encoder_steps, :],
                known_combined_layer[:, :encoder_steps, :],
                observed_inputs[:, :encoder_steps, :]
            ], -1)

        else:
            historical_inputs = tf.concat([
                known_combined_layer[:, :encoder_steps, :],
                observed_inputs[:, :encoder_steps, :]
            ], -1)

        future_inputs = known_combined_layer[:, encoder_steps:, :]

        def static_combine_and_mask(embedding):
            """Applies variable selection network to static inputs.

            Args:
                embedding: Transformed static inputs

            Returns:
                Tensor output for variable selection network
            """

            # Add temporal features

            _, num_static, _ = embedding.get_shape().as_list()

            flatten = tf.keras.layers.Flatten()(embedding)

            # Nonlinear transformation with gated residual network

            mlp_outputs = gated_residual_network(
                flatten,
                self.hidden_layer_size,
                output_size=num_static,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                additional_context=None
            )

            sparse_weights = tf.keras.layers.Softmax()(mlp_outputs)
            sparse_weights = tf.expand_dims(sparse_weights, -1)
            trans_emb_list = []
            for i in range(num_static):
                embed = gated_residual_network(
                    embedding[:,i:i+1,:],
                    self.hidden_layer_size,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=False
                )
                trans_emb_list.append(embed)

            transformed_embedding = tf.concat(trans_emb_list, 1)

            combined = tf.keras.layers.Multiply()([sparse_weights, transformed_embedding])

            static_vec = tf.math.reduce_sum(combined, axis=1)

            return static_vec, sparse_weights

        static_encoder, static_weights = static_combine_and_mask(static_inputs)

        static_context_variable_selection = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)
        static_context_enrichment = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)
        static_context_state_h = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)
        static_context_state_c = gated_residual_network(
            static_encoder,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False)

        def lstm_combine_and_mask(embedding, name=None):
            """Apply temporal variable selection networks.

            Args:
                embedding: Transformed inputs.

            Returns:
                Processed tensor outputs.
            """

            # Add temporal features

            _, time_steps, embedding_dim, num_inputs = embedding.get_shape().as_list()

            flatten = tf.reshape(embedding, [-1, time_steps, embedding_dim * num_inputs]) # Adapts the number of batches to (?, time_steps, embedding_dim * num_inputs)

            expanded_static_context = tf.expand_dims(static_context_variable_selection, 1)

            # Variable selection weights

            mlp_outputs, static_gate = gated_residual_network(
                flatten,
                self.hidden_layer_size,
                output_size=num_inputs,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                additional_context=expanded_static_context,
                return_gate=True
            )

            sparse_weights = tf.keras.layers.Softmax()(mlp_outputs)
            sparse_weights = tf.expand_dims(sparse_weights,2,name=name)

            # Non-linear processing & weight application
            trans_emb_list = []
            for i in range(num_inputs):
                grn_output = gated_residual_network(
                    embedding[Ellipsis,i],
                    self.hidden_layer_size,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=True
                )
                trans_emb_list.append(grn_output)

            transformed_embedding = tf.stack(trans_emb_list, axis=-1)

            combined = tf.keras.layers.Multiply()([sparse_weights, transformed_embedding])
            temporal_ctx = tf.math.reduce_sum(combined, axis=-1)

            return temporal_ctx, sparse_weights, static_gate
        
        historical_features, historical_flags, _ = lstm_combine_and_mask(historical_inputs, name='Historical_Inputs_GRN')
        future_features, future_flags, _ = lstm_combine_and_mask(future_inputs, name='Future_Inputs_GRN')

        # LSTM Layer

        def get_lstm(return_state):
            """Returns LSTM cell initialized with default parameters."""
            return tf.keras.layers.LSTM(
                self.hidden_layer_size,
                return_sequences=True,
                return_state=return_state,
                stateful=False,
                activation='tanh',
                recurrent_activation='sigmoid',
                recurrent_dropout=0,
                unroll=False,
                use_bias=True
            )
        
        history_lstm, state_h, state_c = get_lstm(True)(
            historical_features,
            initial_state=[static_context_state_h, static_context_state_c]
        )

        future_lstm = get_lstm(False)(
            future_features, initial_state=[state_h, state_c]
        )

        lstm_layer = tf.concat([history_lstm, future_lstm], axis=1)

        # Apply gated skip connection

        input_embeddings = tf.concat([historical_features, future_features], axis = 1)

        lstm_layer, _ = apply_gating_layer(
            lstm_layer, self.hidden_layer_size, self.dropout_rate, activation=None
        )
        temporal_feature_layer = add_and_norm([lstm_layer, input_embeddings])

        #Static enrichment layers
        expanded_static_context = tf.expand_dims(static_context_enrichment, 1)
        enriched = gated_residual_network(
            temporal_feature_layer,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            additional_context=expanded_static_context
        )

        # Decoder self attention
        self_attn_layer = InterpretableMultiHeadAttention(
            self.num_heads,
            self.hidden_layer_size,
            dropout=self.dropout_rate
        )

        mask = get_decoder_mask(enriched)
        x, self_attn = self_attn_layer(enriched, enriched, enriched, mask=mask)

        x, _ = apply_gating_layer(
            x,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            activation=None
        )

        x = add_and_norm([x, enriched])

        # Nonlinear processing on outputs
        decoder = gated_residual_network(
            x,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True
        )

        # Final skip connection
        decoder, _ = apply_gating_layer(
            decoder, self.hidden_layer_size, activation=None)
        transformer_layer = add_and_norm([decoder, temporal_feature_layer])

        # Attention components for explainability
        attention_components = {
            # Temporal attention weights
            'decoder_self_attn': self_attn,
            # Static variable selection weights
            'static_flags': static_weights[Ellipsis, 0],
            # Variable selection weights of past inputs
            'historical_flags': historical_flags[Ellipsis, 0, :],
            # Variable selection weights of future inputs
            'future_flags': future_flags[Ellipsis, 0, :]
        }

        return transformer_layer, all_inputs, attention_components
    
    def build_model(self):
        """Build model and defines training losses.

        Returns:
            Fully defined Keras model.
        """

        transformer_layer, all_inputs, attention_components = self._build_base_graph()
        outputs = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.output_size * len(self.quantiles))
        )(transformer_layer[Ellipsis, self.num_encoder_steps:, :])

        self._attention_components = attention_components

        adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=self.max_gradient_norm)

        model = tf.keras.Model(inputs=all_inputs, outputs=outputs)

        valid_quantiles = self.quantiles
        output_size = self.output_size
        quantile_loss = QuantileLossCalculator(valid_quantiles, output_size).quantile_loss

        model.compile(loss=quantile_loss, optimizer=adam, sample_weight_mode='temporal',metrics=[
                tf.keras.metrics.RootMeanSquaredError(),
                tf.keras.metrics.MeanAbsoluteError(),
                ],)

        self._input_placeholder = all_inputs

        return model

    def fit(self, train_df=None, valid_df=None):
        """Fits deep neural network for given training and validation data.

        Args:
            train_df: DataFrame for training data
            valid_df: DataFrame for validation data
        """
        print('*** Fitting {} ***'.format(self.name))

        # Add relevant callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                min_delta=1e-4),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.get_keras_saved_path(self._temp_folder),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.TerminateOnNaN()
        ]

        print('Getting batched_data')
        if train_df is None:
            print('Using cached training data')
            train_data = TFTDataCache.get('train')
        else:
            train_data = self._batch_data(train_df)

        if valid_df is None:
            print('Using cached validation data')
            valid_data = TFTDataCache.get('valid')
        else:
            valid_data = self._batch_data(valid_df)

        print('Using keras standard fit')

        def _unpack(data):
            return data['inputs'], data['outputs'], self._get_active_locations(data['active_entries'])

        data, labels, active_flags = _unpack(train_data)
        val_data, val_labels, val_flags = _unpack(valid_data)

        all_callbacks = callbacks

        self.model.fit(
            x=data,
            y=np.concatenate([labels, labels, labels], axis=-1),
            sample_weight=active_flags,
            epochs=self.num_epochs,
            batch_size=self.minibatch_size,
            validation_data=(val_data, 
                np.concatenate([val_labels, val_labels,val_labels], axis=-1),
                val_flags),
            callbacks=all_callbacks,
            shuffle=True,
            use_multiprocessing=True,
            workers=self.n_multiprocessing_workers,
            verbose=1
        )

        # Load best checkpoint again
        tmp_checkpoint = self.get_keras_saved_path(self._temp_folder)
        if os.path.exists(tmp_checkpoint):
            self.load(self._temp_folder)
        else:
            print('Cannot load from {}, skipping ...'.format(self._temp_folder))
    
    def evaluate(self, data=None, eval_metric='loss'):
        """Applies evaluation metric to the training data.

        Args:
            data: Dataframe for evaluation
            eval_metric: Evaluation metic to return, based on model definition.

        Returns:
            Computed evaluation loss.
        """

        if data is None:
            print('Using cached validation data')
            raw_data = TFTDataCache.get('valid')
        else:
            raw_data = self._batch_data(data)

        inputs = raw_data['inputs']
        outputs = raw_data['outputs']
        active_entries = self._get_active_locations(raw_data['active_entries'])

        metric_values = self.model.evaluate(
            x=inputs,
            y=np.concatenate([outputs,outputs,outputs], axis=-1),
            sample_weight=active_entries,
            workers=self.n_multiprocessing_workers,
            use_multiprocessing=True
        )

        metrics = pd.Series(metric_values, self.model.metrics_names)

        return metrics[eval_metric]

    def predict(self, df, return_targets=False):
        """Computes predictions for a given input dataset.

        Args:
            df: Input dataframe
            return_targets: Whether to also return outputs aligned with predictions to
                faciliate evaluation

        Returns:
            Input dataframe or tuple of (input dataframe, algined output dataframe).
        """

        data = self._batch_data(df)

        inputs = data['inputs']
        time = data['time']
        identifier = data['identifier']
        outputs = data['outputs']

        combined = self.model.predict(
            inputs,
            workers=self.n_multiprocessing_workers,
            use_multiprocessing=True,
            batch_size=self.minibatch_size
        )

        if self.output_size != 1:
            raise NotImplementedError('Current version only supports 1D targets!')

        def format_outputs(prediction):
            """Returns formatted dataframes for prediction."""

            flat_prediction = pd.DataFrame(
                prediction[:, :, 0],
                columns=[
                    't+{}'.format(i)
                    for i in range(self.time_steps - self.num_encoder_steps)
                ])
            cols = list(flat_prediction.columns)
            flat_prediction['forecast_time'] = time[:, self.num_encoder_steps - 1, 0]
            flat_prediction['identifier'] = identifier[:, 0, 0]

            # Arrange in order
            return flat_prediction[['forecast_time', 'identifier'] + cols]

        # Extract predictions for each quantile into different entries
        process_map = {
            'p{}'.format(int(q * 100)):
            combined[Ellipsis, i * self.output_size:(i + 1) * self.output_size]
            for i, q in enumerate(self.quantiles)
        }

        if return_targets:
        # Add targets if relevant
            process_map['targets'] = outputs

        return {k: format_outputs(process_map[k]) for k in process_map}
    
    def get_attention(self,df):
        """Computes TFT attention weights for a given dataset.

        Args:
            df: Input dataframe

        Returns:
            Dictionary of numpy arrays for temporal attention weights and variable
            selection weights, along with their identifiers and time indices
        """
        data = self._batch_data(df)
        inputs = data['inputs']
        identifiers = data['identifier']
        time = data['time']

        def get_batch_attention_weights(input_batch):
            """Returns weights for a given minibatch of data."""
            input_placeholder = self._input_placeholder
            attention_weights = {}
            for k in self._attention_components:
                attention_weight = self._attention_components[k].eval(input_batch.astype(np.float32))
                attention_weights[k] = attention_weight
            return attention_weights

        # Compute number of batches
        batch_size = self.minibatch_size
        n = inputs.shape[0]
        num_batches = n // batch_size
        if n - (num_batches * batch_size) > 0:
            num_batches += 1
        
        # Split up inputs into batches
        batched_inputs = [
            inputs[i*batch_size:(i+1)*batch_size,Ellipsis]
            for i in range(num_batches)
        ]

        # Get attention weights, while avoiding large memory increases
        attention_by_batch = [
            get_batch_attention_weights(batch) for batch in batched_inputs
        ]

        attention_weights = {}
        for k in self._attention_components:
            attention_weights[k] = []
            for batch_weights in attention_by_batch:
                attention_weights[k].append(batch_weights[k])
            
            if len(attention_weights[k][0].shape) == 4:
                tmp = np.concatenate(attention_weights[k], axis=1)
            else:
                tmp = np.concatenate(attention_weights[k], axis=0)

            del attention_weights[k]
            gc.collect()
            attention_weights[k] = tmp

        attention_weights['identifiers'] = identifiers[:,0,0]
        attention_weights['time'] = time[:,:,0]

        return attention_weights

    def reset_temp_folder(self):
        """Deletes and recreates folder with temporary Keras training outputs."""
        print('Resetting temp folder')
        utils.create_folder_if_not_exist(self._temp_folder)
        shutil.rmtree(self._temp_folder)
        os.makedirs(self._temp_folder)
    
    def get_keras_saved_path(self, model_folder):
        """Returns path to keras checkpoint."""
        return os.path.join(model_folder, '{}.ckpt'.format(self.name))

    def save(self, model_folder):
        """Saves optimal TFT weights.

        Args:
        model_folder: Location to serialze model.
        """
        # Allows for direct serialisation of tensorflow variables to avoid spurious
        # issue with Keras that leads to different performance evaluation results
        # when model is reloaded (https://github.com/keras-team/keras/issues/4875).

        utils.save(
            self.model,
            model_folder,
            cp_name=self.name)
        
    def load(self, model_folder):
        """Loads TFT weights.

        Args:
            model_folder: Folder containing serialized models.
            use_keras_loadings: Whether to load from Keras checkpoint.

        Returns:

        """
        
        # Loads temporary Keras model saved during training
        serialisation_path = self.get_keras_saved_path(model_folder)
        print('Loading model from {}'.format(serialisation_path))
        self.model.load_weights(serialisation_path)
        
        
    @classmethod
    def get_hyperparam_choices(cls):
        """Returns hyperparameter ranges for random search."""
        return {
            'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
            'hidden_layer_size': [10, 20, 40, 80, 160, 240, 320],
            'minibatch_size': [64, 128, 256],
            'learning_rate': [1e-4, 1e-3, 1e-2],
            'max_gradient_norm': [0.01, 1.0, 100.0],
            'num_heads': [1, 4],
            'stack_size': [1],
        }
