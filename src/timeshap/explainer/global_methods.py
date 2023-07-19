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

from typing import Callable, Union, List, Tuple
import numpy as np
import pandas as pd
from timeshap.explainer import prune_all, event_explain_all, feat_explain_all
from timeshap.explainer.pruning import verify_pruning_dict
from timeshap.explainer.event_level import verify_event_dict
from timeshap.explainer.feature_level import verify_feature_dict

from timeshap.plot import plot_global_report
import os
from timeshap.utils import convert_to_indexes, convert_data_to_3d, validate_input


def validate_global_input(f: Callable[[np.ndarray], np.ndarray],
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
                          verbose: bool = False,
                          ):
    """ Validates the inputs for global reports

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: Union[List[np.ndarray], pd.DataFrame, np.array]
        Sequences to be explained.
        Must contain columns with names disclosed on `model_features`.

    pruning_dict: dict
        Information required for the pruning algorithm

    event_dict: dict
        Information required for the event level explanation calculation

    feature_dict: dict
        Information required for the feature level explanation calculation

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.
        The baseline can be an average event or an average sequence

    model_features: List[str]
        Features to be used by the model. Requires same order as training dataset

    schema: List[str]
        Schema of provided data

    entity_col: str
        Entity column to identify sequences

    time_col: str
        Data column that represents the time feature in order to sort sequences
        temporally

    append_to_files: bool
        Append explanations to files if file already exists

    verbose: bool
        If process is verbose

    """
    validate_input(f, data, baseline, model_features, schema, entity_col, time_col)

    if isinstance(data, np.ndarray):
        if len(data.shape) == 2:
            assert entity_col is not None, "Entity column must be provided when using 2D numpy arrays as data"

    if pruning_dict is not None:
        verify_pruning_dict(pruning_dict)

        if pruning_dict.get("path"):
            if os.path.exists(pruning_dict.get("path")) and not append_to_files:
                print(
                    "The defined path for pruning data already exists and the append option is turned off. TimeSHAP will only read from this file and will not create new explanation data")
        else:
            print("No path to persist pruning data provided.")
    verify_event_dict(event_dict)
    verify_feature_dict(feature_dict)

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
                             ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Calculates global report explanations

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: Union[List[np.ndarray], pd.DataFrame, np.array]
        Sequences to be explained.
        Must contain columns with names disclosed on `model_features`.

    pruning_dict: dict
        Information required for the pruning algorithm

    event_dict: dict
        Information required for the event level explanation calculation

    feature_dict: dict
        Information required for the feature level explanation calculation

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.
        The baseline can be an average event or an average sequence

    model_features: List[str]
        Features to be used by the model. Requires same order as training dataset

    schema: List[str]
        Schema of provided data

    entity_col: str
        Entity column to identify sequences

    time_col: str
        Data column that represents the time feature in order to sort sequences
        temporally

    append_to_files: bool
        Append explanations to files if file already exists

    max_instances: int
        Max number of instances to explain

    verbose: bool
        If process is verbose

    Returns
    -------
    pd.DataFrame
        Global pruning algorithm information

    pd.DataFrame
        Global event explanations

    pd.DataFrame
        Global feature explanations
    """
    if schema is None and isinstance(data, pd.DataFrame):
        schema = list(data.columns.map(str))

    validate_global_input(
        f, data, pruning_dict, event_dict, feature_dict, baseline, model_features,
        schema, entity_col, time_col, append_to_files, verbose)

    model_features_index, entity_col_index, time_col_index = convert_to_indexes(model_features, schema, entity_col, time_col)
    data = convert_data_to_3d(data, entity_col_index, time_col_index)
    if len(data) > max_instances:
        selected_sequences = np.random.choice(np.arange(len(data)), max_instances, False)
        data = [data[idx] for idx in selected_sequences]

    if pruning_dict is None:
        prun_indexes = None
    else:
        print("Calculating pruning algorithm")
        prun_indexes = prune_all(f, data, pruning_dict, baseline,
                                 model_features_index, schema, entity_col_index,
                                 time_col_index, append_to_files, verbose)

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
    """ Calculates the global report and plots it.\

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: Union[List[np.ndarray], pd.DataFrame, np.array]
        Sequences to be explained.
        Must contain columns with names disclosed on `model_features`.

    pruning_dict: dict
        Information required for the pruning algorithm

    event_dict: dict
        Information required for the event level explanation calculation

    feature_dict: dict
        Information required for the feature level explanation calculation

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.
        The baseline can be an average event or an average sequence

    model_features: List[str]
        Features to be used by the model. Requires same order as training dataset

    schema: List[str]
        Schema of provided data

    entity_col: str
        Entity column to identify sequences

    time_col: str
        Data column that represents the time feature in order to sort sequences
        temporally

    append_to_files: bool
        Append explanations to files if file already exists

    max_instances: int
        Max instances to use for global plots and explanations.
        Used to limit explanation dump file sizes and allow for feasable
        plot time

    verbose: bool
        If process is verbose

    Returns
    -------
    pd.DataFrame

    altair.plot
    """
    prun_indexes, event_data, feat_data = \
        calc_global_explanations(f, data, pruning_dict, event_dict,
                                 feature_dict, baseline, model_features,
                                 schema, entity_col, time_col, append_to_files,
                                 max_instances, verbose
                                 )

    prun_stats, global_plot = \
        plot_global_report(pruning_dict, event_dict, feature_dict, prun_indexes,
                           event_data, feat_data
                           )

    return prun_stats, global_plot
