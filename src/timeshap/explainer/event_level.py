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

from typing import Callable, List, Union
import numpy as np
import pandas as pd
from timeshap.explainer.kernel import TimeShapKernel
import os
import re
import csv
from pathlib import Path
from timeshap.utils import convert_to_indexes, convert_data_to_3d
from timeshap.explainer import temp_coalition_pruning
from timeshap.utils import get_tolerances_to_test


def event_level(f: Callable,
                data: np.ndarray,
                baseline: np.ndarray,
                pruned_idx: int,
                random_seed: int,
                nsamples: int,
                display_events: List[str] = None,
                ) -> pd.DataFrame:
    """Method to calculate event level explanations

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: pd.DataFrame
        Sequence to explain.

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.

    pruned_idx: int
        Index to prune the sequence. All events up to this point are grouped

    random_seed: int
        Used random seed for the sampling process.

    nsamples: int
        The number of coalitions for TimeSHAP to sample.

    display_events: List[str]
        In-order list of event names to be displayed

    Returns
    -------
    pd.DataFrame
    """
    explainer = TimeShapKernel(f, baseline, random_seed, "event")
    shap_values = explainer.shap_values(data, pruning_idx=pruned_idx, nsamples=nsamples)

    if display_events is None:
        display_events = ["Event {}".format(str(-int(i))) for i in np.arange(1, data.shape[1]-pruned_idx+1)]
    else:
        display_events = display_events[-len(shap_values)+1:]
    if pruned_idx > 0:
        display_events += ["Pruned Events"]

    ret_data = []
    for exp, event in zip(shap_values, display_events):
        ret_data += [[random_seed, nsamples, event, exp]]
    return pd.DataFrame(ret_data, columns=['Random seed', 'NSamples', 'Feature', 'Shapley Value'])


def local_event(f: Callable[[np.ndarray], np.ndarray],
                data: Union[pd.DataFrame, np.array],
                event_dict: dict,
                entity_uuid: Union[str, int, float],
                entity_col: str,
                baseline: Union[pd.DataFrame, np.array],
                pruned_idx: int,
                ):
    """Method to calculate event level explanations or load them if path is provided

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: pd.DataFrame
        Sequence to explain.

    event_dict: dict
        Information required for the event level explanation calculation

    entity_uuid: Union[str, int, float]
        The indentifier of the sequence that is being pruned.
        Used when fetching information from a csv of explanations

    entity_col: str
        Entity column to identify sequences

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.

    pruned_idx: int
        Index to prune the sequence. All events up to this point are grouped

    Returns
    -------
    pd.DataFrame
    """
    if event_dict.get("path") is None or not os.path.exists(event_dict.get("path")):
        #print("No path to event data provided. Calculating data")
        event_data = event_level(f, data, baseline, pruned_idx, event_dict.get("rs"), event_dict.get("nsamples"))
        if event_dict.get("path") is not None:
            # create directory
            if '/' in event_dict.get("path"):
                Path(event_dict.get("path").rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
            event_data.to_csv(event_dict.get("path"), index=False)
    elif event_dict.get("path") is not None and os.path.exists(event_dict.get("path")):
        event_data = pd.read_csv(event_dict.get("path"))
        if len(event_data.columns) == 5 and entity_col is not None:
            event_data = event_data[event_data[entity_col] == entity_uuid]
        elif len(event_data.columns) == 4:
            pass
        else:
            # TODO
            # the provided csv should be generated by timeshap, by either
            # explaining the whole dataset with TODO or just the instance in question
            raise ValueError
    else:
        raise ValueError

    return event_data


def verify_event_dict(event_dict: dict):
    if event_dict.get('path'):
        assert isinstance(event_dict.get('path'), str), "Provided path must be a string"

    if event_dict.get('rs', False):
        if isinstance(event_dict.get('rs'), int):
            event_dict['rs'] = [event_dict.get('rs')]
        elif isinstance(event_dict.get('rs'), list):
            assert np.array([isinstance(x, int) for x in event_dict.get('rs')]).all(), "All provided random seeds must be ints."
        else:
            raise ValueError("Unsuported format of random seeds(s). Please provide one seed or a list of them.")
    else:
        print("No random seed provided for event-level explanations. Using default: 42")
        event_dict['rs'] = [42]

    if event_dict.get('nsamples', False):
        if isinstance(event_dict.get('nsamples'), int):
            event_dict['nsamples'] = [event_dict.get('nsamples')]
        elif isinstance(event_dict.get('nsamples'), list):
            assert np.array([isinstance(x, int) for x in event_dict.get('nsamples')]).all(), "All provided nsamples must be ints."
        else:
            raise ValueError("Unsuported format of nsamples. Please provide value or a list of them.")
    else:
        print("No nsamples provided for event-level explanations. Using default: 32000")
        event_dict['nsamples'] = [32000]

    if event_dict.get('tol', False):
        tolerances = event_dict.get('tol')
        if isinstance(tolerances, float):
            event_dict['tol'] = [tolerances]
        elif isinstance(tolerances, list):
            assert np.array([isinstance(x, float) for x in tolerances]).all(), "All provided tolerances must be floats."


def event_explain_all(f: Callable,
                      data: Union[List[np.ndarray], pd.DataFrame, np.array],
                      event_dict: dict,
                      pruning_data: pd.DataFrame = None,
                      baseline: Union[pd.DataFrame, np.array] = None,
                      model_features: List[Union[int, str]] = None,
                      schema: List[str] = None,
                      entity_col: Union[int, str] = None,
                      time_col: Union[int, str] = None,
                      append_to_files: bool = False,
                      verbose: bool = False,
                      ) -> pd.DataFrame:
    """Calculates event level explanations for all entities on the provided
    DataFrame applying pruning if explicit

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: pd.DataFrame
        Sequences to be explained.
        Must contain columns with names disclosed on `model_features`.

    entity_col: str
        Entity column to identify sequences

    baseline: Union[pd.DataFrame, np.array]
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.

    event_dict: dict
        Information required for the event level explanation calculation

    pruning_data: pd.DataFrame
        Pruning indexes for all sequences being explained.
        Produced by `prune_all`

    model_features: List[str]
        Features to be used by the model. Requires same order as training dataset

    time_col: str
        Data column that represents the time feature in order to sort sequences
        temporally

    verbose: bool
        If process is verbose

    Returns
    -------
    pd.DataFrame
    """
    if schema is None and isinstance(data, pd.DataFrame):
        schema = list(data.columns)
    verify_event_dict(event_dict)
    file_path = event_dict.get('path')
    make_predictions = True
    event_data = None

    tolerances_to_calc = get_tolerances_to_test(pruning_data, event_dict, entity_col)

    if file_path is not None and os.path.exists(file_path) and not append_to_files:
        event_data = pd.read_csv(file_path)
        make_predictions = False

        # TODO resume explanations
        # conditions = []
        # necessary_entities = set(np.unique(data[entity_col].values))
        # event_data = pd.read_csv(file_path)
        # present_entities = set(np.unique(event_data[entity_col].values))
        # if necessary_entities.issubset(present_entities):
        #     conditions.append(True)
        #     event_data = event_data[event_data[entity_col].isin(necessary_entities)]
        #
        # necessary_tols = set(tolerances_to_calc)
        # loaded_csv = pd.read_csv(file_path)
        # present_tols = set(np.unique(loaded_csv['Tolerance'].values))
        # if necessary_tols.issubset(present_tols):
        #     conditions.append(True)
        #     event_data = event_data[event_data['Tolerance'].isin(necessary_tols)]
        #
        # make_predictions = ~np.array(conditions).all()

    if make_predictions:
        random_seeds = event_dict.get('rs')
        nsamples = event_dict.get('nsamples')
        names = ["Random Seed", "NSamples", "Event", "Shapley Value", "t (event index)", "Entity", 'Tolerance']

        if file_path is not None:
            if os.path.exists(file_path):
                assert append_to_files, "The defined path for event explanations already exists and the append option is turned off. If you wish to append the explanations please use the flag `append_to_files`, otherwise change the provided path."
            else:
                if '/' in file_path:
                    Path(file_path.rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
                with open(file_path, 'w', newline='') as file:
                    writer = csv.writer(file, delimiter=',')
                    writer.writerow(names)

        if time_col is None:
            print("No time col provided, assuming dataset is ordered ascendingly by date")

        model_features_index, entity_col_index, time_col_index = convert_to_indexes(model_features, schema, entity_col, time_col)
        data = convert_data_to_3d(data, entity_col_index, time_col_index)

        ret_event_data = []
        for rs in random_seeds:
            for ns in nsamples:
                for sequence in data:
                    if entity_col is not None:
                        entity = sequence[0, 0, entity_col_index]
                    if model_features:
                        sequence = sequence[:, :, model_features]
                    sequence = sequence.astype(np.float64)
                    event_data = None
                    prev_pruning_idx = None
                    for tol in tolerances_to_calc:
                        if pruning_data is None:
                            #we need to perform the pruning on the fly
                            coal_prun_idx, _ = temp_coalition_pruning(f, sequence, baseline, tol)
                            pruning_idx = data.shape[1] + coal_prun_idx
                        else:
                            instance = pruning_data[pruning_data["Entity"] == entity]
                            pruning_idx = instance[instance['Tolerance'] == tol]['Pruning idx'].iloc[0]
                            pruning_idx = sequence.shape[1] + pruning_idx

                        if prev_pruning_idx == pruning_idx:
                            # we have already calculated this, let's use it from the last iteration
                            event_data['Tolerance'] = tol
                        else:
                            local_event_dict = {'rs': rs, 'nsamples': ns}
                            event_data = local_event(f, sequence, local_event_dict, entity, entity_col, baseline, pruning_idx)
                            event_data['Event index'] = event_data['Feature'].apply(lambda x: 1 if x == 'Pruned Events' else -int(re.findall(r'\d+', x)[0])+1)
                            event_data[entity_col] = entity
                            event_data['Tolerance'] = tol

                        if file_path is not None:
                            with open(file_path, 'a', newline='') as file:
                                writer = csv.writer(file, delimiter=',')
                                writer.writerows(event_data.values)
                        ret_event_data.append(event_data.values)
                        prev_pruning_idx = pruning_idx

        event_data = pd.DataFrame(np.concatenate(ret_event_data), columns=names)
        event_data = event_data.astype({'NSamples': 'int', 'Random Seed': 'int', 'Tolerance': 'float', 'Shapley Value': 'float', 't (event index)': 'int'})
    return event_data
