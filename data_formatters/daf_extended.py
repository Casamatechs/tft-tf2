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
from data_formatters.base import GenericDataFormatter, DataTypes, InputTypes
import libs.utils as utils
import sklearn.preprocessing

import pandas as pd
import os

import calendar
import time

class DafExtendedFormatter(GenericDataFormatter):
    """Defines and formats data for the DAF extended dataset.

    Attributes:
        column_definition: Defines input and data type of column used in the
        experiment.
        identifiers: Entity identifiers used in experiments.
    """

    _column_definition = [
        ('END_DATETIME', DataTypes.DATE, InputTypes.TIME),
        ('DRIVERID', DataTypes.CATEGORICAL, InputTypes.ID),
        ('TRUCK_TYPE', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('AXLE_CONF', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('COMMERCIAL_NAME', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('TRUCK_SERIES', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('TRUCK_ENGINE', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('GROSS_WEIGHT', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('TRIP_DISTANCE', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('ALTITUDE_DELTA', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('USED_FUEL', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('IDLE_FUEL', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('FUEL_CONSUMPTION', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('CC_FUEL', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('CC_FUEL_CONSUMPTION', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('CC_DIST', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('CC_ENABLED', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('BRAKEDURATION', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('HARSHBRAKEDURATION', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('DPA_SCORE', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('CATEGORICAL_ID', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('INDEX_TRIP', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('DAYOFWEEK', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('HOUROFDAY', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
    ]

    def __init__(self):
        """Initialises formatter."""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self._train_filename = 'tf_input_train.csv'
        self._valid_filename = 'tf_input_valid.csv'
        self._test_filename = 'tf_input_test.csv'

    def split_data(self, df_train, df_valid, df_test, valid_boundary = None, test_boundary = None, path = None, overwrite = False):
        """Splits data frame into training-validation-test data frames.

        This also calibrates scaling object, and transforms data for each split.

        Args:
            df: Source data frame to split.
            valid_boundary: Starting year for validation data
            test_boundary: Starting year for test data

        Returns:
            Tuple of transformed (train, valid, test) data.
        """
        # if not overwrite and path is not None and os.path.exists(os.path.join(path,self._train_filename)):
        #     print('Loading stored training dataset...')
        #     train = pd.read_csv(os.path.join(path,self._train_filename), index_col=0)
        #     valid = pd.read_csv(os.path.join(path,self._valid_filename), index_col=0)
        #     test = pd.read_csv(os.path.join(path,self._test_filename), index_col=0)
        #     filtered_df = pd.concat([train, valid, test])
        #     self.set_scalers(train)

        #     return (self.transform_inputs(data) for data in [train, valid, test])
        

        # total_time_steps = self.get_fixed_params()['total_time_steps']
        # filtered_df = df.copy()
        # trip_amount = df.DRIVERID.value_counts()

        # if valid_boundary is None:
        #     valid_boundary = 0.2
        # if test_boundary is None:
        #     test_boundary = 0.1
        
        # Gets rid of all the drivers that don't have enough trips recorded to fit the encoders for training, validation and test
        # filtered_df = filtered_df[filtered_df['DRIVERID'].isin(trip_amount[trip_amount * test_boundary > total_time_steps].index)]

        # def split_series(df, valid_boundary, test_boundary):
        #     train, valid, test = [], [], []
        #     counter = 0
        #     number_drivers = df['DRIVERID'].nunique()
        #     ts_o = time.time()
        #     for driverid in df['DRIVERID'].unique():
                # if counter % 1000 == 0:
                #     ts_f = time.time() - ts_o
                #     print('Processed {} out of {} drivers in {}ms'.format(counter, number_drivers, (ts_f * 1000)))
                #     ts_o = time.time()
        #         driver_df = df[df['DRIVERID'] == driverid].copy()
        #         number_steps = len(driver_df)
        #         driver_df['CATEGORICAL_ID'] = driver_df['DRIVERID']
        #         driver_df['TIME_STEPS'] = list(range(0,number_steps))
        #         valid_size = int(number_steps * valid_boundary)
        #         test_size = int(number_steps * test_boundary)
        #         train_size = number_steps - valid_size - test_size
        #         train.append(driver_df[:train_size])
        #         valid.append(driver_df[train_size:train_size+valid_size])
        #         test.append(driver_df[train_size+valid_size:])
        #         counter += 1
        #     train_df = pd.concat(train, ignore_index=True)
        #     valid_df = pd.concat(valid, ignore_index=True)
        #     test_df = pd.concat(test, ignore_index=True)

        #     if path is not None:
        #         print('Writing splits in {}'.format(path))
        #         train_df.to_csv(os.path.join(path,self._train_filename))
        #         valid_df.to_csv(os.path.join(path,self._valid_filename))
        #         test_df.to_csv(os.path.join(path,self._test_filename))
            
        #     return train_df, valid_df, test_df

        # train, valid, test = split_series(filtered_df, valid_boundary, test_boundary)

        def preprocess_df(df_train, df_valid, df_test):
            number_drivers = df_train['DRIVERID'].nunique()
            counter = 0
            ts_o = time.time()
            df_train['CATEGORICAL_ID'] = df_train['DRIVERID']
            df_valid['CATEGORICAL_ID'] = df_valid['DRIVERID']
            df_test['CATEGORICAL_ID'] = df_test['DRIVERID']
            # for driverid in df_train['DRIVERID'].unique():
            #     if counter % 10 == 0:
            #         ts_f = time.time() - ts_o
            #         print('Processed {} out of {} drivers in {}ms'.format(counter, number_drivers, (ts_f * 1000)))
            #         ts_o = time.time()
            #     len_train = len(df_train[df_train['DRIVERID'] == driverid])
            #     len_valid = len(df_valid[df_valid['DRIVERID'] == driverid])
            #     len_test = len(df_test[df_test['DRIVERID'] == driverid])
            #     df_valid[df_valid['DRIVERID'] == driverid]['INDEX_TRIP'] = list(range(len_train+1, len_train+1+len_valid))
            #     df_test[df_test['DRIVERID'] == driverid]['INDEX_TRIP'] = list(range(len_train+len_valid+1, len_train+1+len_valid+len_test))
            #     counter += 1
            return df_train, df_valid, df_test
        
        df_train, df_valid, df_test = preprocess_df(df_train, df_valid, df_test)
        
        self.set_scalers(df_train)

        return (self.transform_inputs(data) for data in [df_train, df_valid, df_test])



    
    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.

        Args:
            df: Data to use to calibrate scalers.
        """
        print('Setting scalers with training data...')

        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID, column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definitions)

        self.identifiers = list(df[id_column].unique())

        # Format real scalers

        real_inputs = utils.extract_cols_from_data_type(DataTypes.REAL_VALUED, column_definitions, {InputTypes.ID, InputTypes.TIME})
        data = df[real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(df[[target_column]].values)

        # Format categorical scalers

        categorical_inputs = utils.extract_cols_from_data_type(DataTypes.CATEGORICAL, column_definitions, {InputTypes.ID, InputTypes.TIME})
        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(srs.values)
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        """Performs feature transformations.

        This includes both feature engineering, preprocessing and normalisation.

        Args:
            df: Data frame to transform.

        Returns:
            Transformed data frame.

        """

        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        column_definitions = self.get_column_definition()

        real_inputs = utils.extract_cols_from_data_type(DataTypes.REAL_VALUED, column_definitions, {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = utils.extract_cols_from_data_type(DataTypes.CATEGORICAL, column_definitions, {InputTypes.ID, InputTypes.TIME})

        # Format real inputs
        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

        # Format categorical inputs

        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df.values)
        
        return output


    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.

        Args:
            predictions: Dataframe of model predictions.

        Returns:
            Data frame of unnormalised predictions.
        """

        output = predictions.copy()

        column_names = predictions.columns

        for col in column_names:
            if col not in {'forecast_time', 'identifier'}:
                output[col] = self._target_scaler.inverse_transform(predictions[col])

        return output

    def get_fixed_params(self):
        """Returns fixed model parameters for experiments"""

        fixed_params = {
            'total_time_steps': 14 + 1,
            'num_encoder_steps': 14,
            'num_epochs': 100,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 5,
        }

        return fixed_params

    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        model_params = {
            'dropout_rate': 0.2,
            'hidden_layer_size': 20,
            'learning_rate': 0.001,
            'minibatch_size': 64,
            'max_gradient_norm': 1.0,
            'num_heads': 4,
            'stack_size': 1
        }

        return model_params

    