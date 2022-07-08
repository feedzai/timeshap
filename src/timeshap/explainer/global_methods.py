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

from typing import Callable, Union, List
import numpy as np
import pandas as pd
from timeshap.explainer import prune_all, pruning_statistics, event_explain_all, feat_explain_all
from timeshap.explainer.pruning import verify_pruning_dict
from timeshap.explainer.event_level import verify_event_dict
from timeshap.explainer.feature_level import verify_feature_dict

from timeshap.plot import plot_global_report
import os
from timeshap.utils import convert_to_indexes, convert_data_to_3d


def validate_global_input(f: Callable[[np.ndarray], np.ndarray],
                          data: Union[pd.DataFrame, np.array],
                          pruning_dict: dict,
                          event_dict: dict,
                          feature_dict: dict,
                          model_features: List[Union[int, str]] = None,
                          schema: List[str] = None,
                          entity_col: Union[int, str] = None,
                          time_col: Union[int, str] = None,
                          baseline: Union[pd.DataFrame, np.array] = None,
                          append_to_files: bool = False,
                          verbose: bool = False,
                          ):
    assert isinstance(f, Callable), "Provided model must be callable"
    assert isinstance(data, (pd.DataFrame, np.ndarray)), "Provided data must be an numpy array or pandas DataFrame"
    assert baseline is None or isinstance(baseline, (pd.DataFrame, np.ndarray)), "Provided baseline must be an numpy array or pandas DataFrame"
    assert model_features is None or (isinstance(model_features, list) and isinstance(model_features[0], (int, str))), "Model features must be a list of features (str) or their corresponding indexes(ints)"
    assert entity_col is None or isinstance(entity_col, (int, str)), "Provided entity column must be a feature name (str) or the corresponding index (int)"
    assert time_col is None or isinstance(time_col, (int, str)), "Provided time column must be a feature name (str) or the corresponding index (int)"

    if model_features is None:
        raise NotImplementedError()

    elif isinstance(model_features[0], str) or \
        (entity_col is not None and isinstance(entity_col, str)) or \
        (time_col is not None and isinstance(time_col, str)):
        # we have strings to obtain indexes, therefore we need the schema
        assert schema is not None, "When model features, entity column or time column are strings, data schema must be provided"
        assert isinstance(schema, list), "Provided schema must be a list of strings"
        assert isinstance(schema[0], str), "Provided schema must be a list of strings"
        assert data.shape[-1] >= len(model_features), "Provided model features do not match data"
        assert data.shape[-1] == len(schema), "Provided schema does not match data"
        assert set(model_features).issubset(set(schema)), "Provided model features must be in the provided schema"

        if time_col is not None and isinstance(time_col, str):
            assert time_col in schema, "Provided time feature must be in the provided schema"

        if entity_col is not None and isinstance(entity_col, str):
            assert entity_col in schema, "Provided entity feature must be in the provided schema"
    else:
        # we are dealing with indexes
        # these columns must not be model features
        assert entity_col is None or entity_col not in model_features, "Provided entity col index must not be on model feature indexes"
        assert time_col is None or time_col not in model_features, "Provided time col index must not be on model feature indexes"
        assert data.shape[-1] >= len(model_features), "Higher number of model features provided than present in data"

    if isinstance(data, pd.DataFrame):
        assert entity_col is not None, "Entity column must be provided when using DataFrames as data"

    if isinstance(data, np.ndarray):
        if len(data.shape) == 2:
            assert entity_col is not None, "Entity column must be provided when using 2D numpy arrays as data"

    verify_pruning_dict(pruning_dict)
    verify_event_dict(event_dict)
    verify_feature_dict(feature_dict)
    if pruning_dict.get("path"):
        if os.path.exists(pruning_dict.get("path")) and not append_to_files:
            print("The defined path for pruning data already exists and the append option is turned off. TimeSHAP will only read from this file and will not create new explanation data")
    else:
        print("No path to persist pruning data provided.")

    if event_dict.get("path"):
        if os.path.exists(event_dict.get("path")) and not append_to_files:
            print("The defined path for event explanations already exists and the append option is turned off. TimeSHAP will only read from this file and will not create new explanation data")
    else:
        print("No path to persist event explanations provided.")

    if feature_dict.get("path"):
        if os.path.exists(feature_dict.get("path")) and not append_to_files:
            print("The defined path for feature explanations already exists and the append option is turned off. TimeSHAP will only read from this file and will not create new explanation data")
    else:
        print("No path to persist feature explanations provided.")


def calc_global_explanations(f: Callable[[np.ndarray], np.ndarray],
                             data: Union[pd.DataFrame, np.array],
                             pruning_dict: dict,
                             event_dict: dict,
                             feature_dict: dict,
                             baseline: Union[pd.DataFrame, np.array] = None,
                             model_features: List[Union[int, str]] = None,
                             schema: List[str] = None,
                             entity_col: Union[int, str] = None,
                             time_col: Union[int, str] = None,
                             append_to_files: bool = False,
                             max_instances: int = 10000,
                             verbose: bool = False,
                             ):

    if schema is None and isinstance(data, pd.DataFrame):
        schema = list(data.columns)

    validate_global_input(
        f, data, pruning_dict, event_dict, feature_dict, model_features,
        schema, entity_col, time_col, baseline, append_to_files, verbose)

    model_features_index, entity_col_index, time_col_index = convert_to_indexes(model_features, schema, entity_col, time_col)
    data = convert_data_to_3d(data, entity_col_index, time_col_index)
    if len(data) > max_instances:
        selected_sequences = np.random.choice(np.arange(len(data)), max_instances, False)
        data = [data[idx] for idx in selected_sequences]
    print("Calculating pruning algorithm")
    prun_indexes = prune_all(f, data, pruning_dict, baseline, model_features_index, schema, entity_col_index, time_col_index, append_to_files, verbose)

    print("Calculating event data")
    event_data = event_explain_all(f, data, event_dict, prun_indexes, baseline, model_features_index, schema, entity_col_index, time_col_index, append_to_files, verbose)

    print("Calculating feat data")
    feat_data = feat_explain_all(f, data, feature_dict, prun_indexes, baseline, model_features_index, schema, entity_col_index, time_col_index, append_to_files, verbose)

    return prun_indexes, event_data, feat_data


def global_report(f: Callable[[np.ndarray], np.ndarray],
                  data: Union[pd.DataFrame, np.array],
                  pruning_dict: dict,
                  event_dict: dict,
                  feature_dict: dict,
                  baseline: Union[pd.DataFrame, np.array] = None,
                  model_features: List[Union[int, str]] = None,
                  schema: List[str] = None,
                  entity_col: Union[int, str] = None,
                  time_col: Union[int, str] = None,
                  append_to_files: bool = False,
                  max_instances: int = 10000,
                  verbose: bool = False,
                  ):
    """Plots a global report of a model applied to a dataset

    Parameters
    ----------

    Returns
    -------
    pd.DataFrame
    """
    prun_indexes, event_data, feat_data = calc_global_explanations(f,
                  data,
                  pruning_dict,
                  event_dict,
                  feature_dict,
                  baseline,
                  model_features,
                  schema,
                  entity_col,
                  time_col,
                  append_to_files,
                  max_instances,
                  verbose,
                  )

    prun_stats, global_plot = plot_global_report(pruning_dict,
                                                 event_dict,
                                                 feature_dict,
                                                 prun_indexes,
                                                 event_data,
                                                 feat_data
                                                 )

    return prun_stats, global_plot
