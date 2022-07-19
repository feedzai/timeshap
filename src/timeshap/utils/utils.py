#  Copyright 2022 Feedzai
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import List, Union
import pandas as pd
import numpy as np
import copy
from scipy import stats


def make_list(element) -> list:
    """Wraps element in list, if element not a list already

    Parameters
    ----------
    element

    Returns
    -------
    list
    """
    if isinstance(element, (int, float)):
        element = [element]
    elif isinstance(element, list):
        pass
    else:
        raise ValueError("Unrecognized parameter format")
    return element


def calculate_list_intersection(a: list, b: list) -> list:
    """Calculates list intersections

    Parameters
    ----------
    a: list

    b: list

    Returns
    -------
    list
    """
    if len(b) == 0:
        return a
    elif len(a) == 0:
        return b
    intersection = list(set(a).intersection(set(b)))
    assert len(intersection) > 0
    return intersection


def get_tolerances_to_test(pruning_data: pd.DataFrame,
                           explanation_dict: dict,
                           ):
    """Converts datasets into list of numpy arrays for smooth global explanation calculation

    Parameters
    ----------
    pruning_data: pd.DataFrame
        Data to explain

    explanation_dict: dict
        Data schema

    Returns
    -------
    List[np.ndarray]
        List of sequences
    """
    if pruning_data is None:
        if explanation_dict.get('tol', False):
            print("No pruning data provided and no pruning tolerances provided. No pruning will take place")
            tolerances_to_calc = [-1.0]
        else:
            tolerances_to_calc = explanation_dict.get('tol')
            print(f"No pruning data provided. TimeSHAP will calculate pruning on-the fly using provided tolerances: {list(tolerances_to_calc)} ")
    else:
        tolerances_to_calc = np.unique(pruning_data['Tolerance'].values)
        tolerances_to_calc = tolerances_to_calc[~(tolerances_to_calc == -1)]
        input_tols = explanation_dict.get('tol')
        if input_tols is not None:
            assert np.array([x in tolerances_to_calc for x in input_tols]).all(), "Inputed tolerances are not present on the provided pruning data"
            tolerances_to_calc = input_tols

    return tolerances_to_calc


def convert_data_to_3d(data: Union[pd.DataFrame, np.ndarray],
                       entity_col_index: int = None,
                       time_col_index: int = None,
                       ) -> List[np.ndarray]:
    """Converts datasets into list of numpy arrays for smooth global explanation calculation

    Parameters
    ----------
    data: Union[pd.DataFrame, np.ndarray]
        Data to explain

    entity_col_index: int
        Index of the entity column

    time_col_index: int
        Index of the time column

    Returns
    -------
    List[np.ndarray]
        List of sequences
    """
    if isinstance(data, list) and isinstance(data[0], np.ndarray) and len(data[0].shape) == 3:
        return data
    dataset = []
    if isinstance(data, np.ndarray):
        if len(data.shape) == 3:
            for sequence_idx in range(data.shape[0]):
                sequence = data[sequence_idx, :, :]
                if entity_col_index is not None:
                    entity = sequence[:, entity_col_index]
                    assert np.all(entity == entity[0]), "All entities of a sequence must be the same"
                if time_col_index is not None:
                    sequence = sequence[sequence[:, time_col_index].argsort()]
                dataset.append(np.expand_dims(sequence, axis=0))

        elif len(data.shape) == 2:
            assert entity_col_index is not None
            for entity in np.unique(data[:, entity_col_index]):
                sequence = data[data[:, entity_col_index] == entity]
                if time_col_index is not None:
                    sequence = sequence[sequence[:, time_col_index].argsort()]
                dataset.append(np.expand_dims(sequence, axis=0))
        else:
            raise ValueError("Unsupported number of dimensions on the dataset")

    elif isinstance(data, pd.DataFrame):
        for entity in data.iloc[:, entity_col_index].unique():
            sequence = data[data.iloc[:, entity_col_index] == entity]
            if time_col_index is not None:
                sequence = sequence.sort_values(by=sequence.columns[time_col_index])
            dataset.append(np.expand_dims(sequence.values, axis=0))
    else:
        raise ValueError("Unsupported data type")
    return dataset


def convert_to_indexes(model_features: List[Union[int, str]] = None,
                       schema: List[str] = None,
                       entity_col: Union[int, str] = None,
                       time_col: Union[int, str] = None,
                       ):
    """Converts features into indexes given a schema

    Parameters
    ----------
    model_features: List[Union[int, str]]
        Model features

    schema: List[str]
        Data schema

    entity_col: Union[int, str]
        Entity column

    time_col: Union[int, str]
        Time column

    Returns
    -------
    List[int]
        Model features index

    int
        entity column index

    int
        time column index
    """
    model_features_index, entity_col_index, time_col_index = None, None, None
    if model_features is not None:
        if isinstance(model_features[0], str):
            model_features_index = [schema.index(x) for x in model_features]
        elif isinstance(model_features[0], int):
            model_features_index = model_features

    if entity_col is not None:
        if isinstance(entity_col, str):
            entity_col_index = schema.index(entity_col)
        elif isinstance(entity_col, int):
            entity_col_index = entity_col

    if time_col is not None:
        if isinstance(time_col, str):
            time_col_index = schema.index(time_col)
        elif isinstance(entity_col, int):
            time_col_index = time_col

    return model_features_index, entity_col_index, time_col_index


def calc_avg_sequence(data: Union[pd.DataFrame, np.ndarray],
                      numerical_feats: List[Union[str, int]],
                      categorical_feats: List[Union[str, int]],
                      model_features=None,
                      entity_col: str = None,
                      ) -> np.ndarray:
    """
    Calculates the average sequence of a dataset. Requires all sequences of the
    dataset to be the same size and ordered by time.

    Calculates the median of numerical features, and the mode for categorical
    features of a pandas DataFrame

    Parameters
    ----------
    data: Union[pd.DataFrame, np.ndarray]
        Dataset to use for baseline calculation

    numerical_feats: List[Union[str, int]]
        List of numerical features or corresponding indexes to calculate median of

    categorical_feats: List[Union[str, int]]
        List of numerical features or corresponding indexes to calculate mode of

    model_features: List[str]
        Model features to infer the indexes of schema. Needed when using strings to identify features

    Returns
    -------
    np.ndarray
        Average sequence to use in TimeSHAP
    """
    if isinstance(data, pd.DataFrame):
        assert entity_col is not None, "To calculate average sequence from DataFrame, entity_col is required"
        sequences = data.groupby(entity_col)
        seq_lens = sequences.size().values
        assert np.array([x == seq_lens[0] for x in seq_lens]).all(), "All sequences must be the same length"
        data = np.array([x[1].values for x in sequences])
    elif isinstance(data, np.ndarray) and len(data.shape) == 3:
        pass
    else:
        raise ValueError("Unrecognized data format")
    if len(numerical_feats) > 0 and isinstance(numerical_feats[0], str) or  len(categorical_feats) > 0 and isinstance(categorical_feats[0], str):
        # given features are not indexes
        assert model_features is not None and len(model_features), "When using feature names to identify them, specify the model features. Alternatively you can pass the indexes of the features directly"
        numerical_indexes = [model_features.index(x) for x in numerical_feats]
        categorical_indexes = [model_features.index(x) for x in categorical_feats]
    else:
        numerical_indexes = numerical_feats
        categorical_indexes = categorical_feats

    numerical = np.median(data[:, :,  numerical_indexes], axis=0)
    if len(categorical_indexes) > 0:
        categorical = stats.mode(data[:, :,  categorical_indexes], axis=0)[0][0, :, :]
        numerical = np.concatenate((numerical, categorical), axis=1)
    return numerical.astype(float)


def calc_avg_event(data: Union[pd.DataFrame, np.ndarray],
                   numerical_feats: List[Union[str, int]],
                   categorical_feats: List[Union[str, int]],
                   model_features: List[str] = None,
                   ) -> pd.DataFrame:
    """
    Calculates the average event of a dataset. This event is repeated N times
    to form the background sequence to be used in TimeSHAP.

    Calculates the median of numerical features, and the mode for categorical
    features of a pandas DataFrame

    Parameters
    ----------
    data: pd.DataFrame
        Dataset to use for baseline calculation

    numerical_feats: List[Union[str, int]]
        List of numerical features or corresponding indexes to calculate median of

    categorical_feats: List[Union[str, int]]
        List of numerical features or corresponding indexes to calculate mode of

    model_features: List[str]
        Model features to infer the indexes of schema. Needed when using strings to identify features

    Returns
    -------
    pd.DataFrame
        DataFrame with the median/mode of the features
    """
    if len(numerical_feats) > 0 and isinstance(numerical_feats[0], str) or  len(categorical_feats) > 0 and isinstance(categorical_feats[0], str):
        # given features are not indexes
        if isinstance(data, pd.DataFrame):
            model_features = list(data.columns)
        else:
            assert model_features is not None and len(model_features), "When using feature names to identify them, specify the model features. Alternatively you can pass the indexes of the features directly"
        numerical_indexes = [model_features.index(x) for x in numerical_feats]
        categorical_indexes = [model_features.index(x) for x in categorical_feats]
        ordered_feats = numerical_feats + categorical_feats
    else:
        numerical_indexes = numerical_feats
        categorical_indexes = categorical_feats
        ordered_feats = numerical_indexes + categorical_indexes

    if len(data.shape) == 3:
        data = np.squeeze(data, axis=1)
    elif len(data.shape) == 2:
        data = data.values
    else:
        raise ValueError

    numerical = np.median(data[:,  numerical_indexes], axis=0)
    if len(categorical_indexes) > 0:
        categorical = stats.mode(data[:, categorical_indexes], axis=0)[0][0, :]
        numerical = np.concatenate((numerical,categorical), axis=0)

    return pd.DataFrame([numerical], columns=ordered_feats)


def get_score_of_avg_sequence(model, data: np.ndarray):
    """Scores the average sequence

    Parameters
    ----------
    model: Union[TimeSHAPWrapper, torch.nn.Module, tf.Module]
        An RNN model.

    data:
        Average sequence

    Returns
    -------
    float
        Score of average sequence
    """
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)

    if len(model.__code__.co_varnames) == 1:
        pred = model(data)
    else:
        pred, _ = model(data)
    return pred[0]


def get_avg_score_with_avg_event(model, med, top=1000):
    """Repeats the avg event N times and returns the score of the last
    event

    Parameters
    ----------
    model: Union[TimeSHAPWrapper, torch.nn.Module, tf.Module]
        An RNN model.

    med:
        Average event of the dataset.

    top:
        Limit to which repeat the evaluation
    Returns
    -------
    Dict
        Keys are seq lens,
        Values are the score of the last event
    """
    avg_score = {}
    hs = None
    expanded = np.expand_dims(med, axis=0)
    for x in range(1, top+1):
        expanded_copy = copy.deepcopy(expanded)
        if len(model.__code__.co_varnames) == 1:
            tiled_background = np.tile(expanded_copy, (1, x, 1))
            pred = model(tiled_background)
        else:
            pred, hs = model(expanded_copy, hs)
        avg_score[x] = float(pred[0])
    return avg_score
