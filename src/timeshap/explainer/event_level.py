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
        print("No path to event data provided. Calculating data")
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


def event_explain_all(f: Callable,
                      data: pd.DataFrame,
                      entity_col: str,
                      baseline: Union[pd.DataFrame, np.ndarray],
                      event_dict: dict,
                      pruning_data: pd.DataFrame,
                      model_features: List[str],
                      time_col: str = None,
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
    file_path = event_dict.get('path')
    tolerances_to_calc = np.unique(pruning_data['Tolerance'].values)
    make_predictions = True
    event_data = None
    if os.path.exists(file_path):
        conditions = []
        necessary_entities = set(np.unique(data[entity_col].values))
        event_data = pd.read_csv(file_path)
        present_entities = set(np.unique(event_data[entity_col].values))
        if necessary_entities.issubset(present_entities):
            conditions.append(True)
            event_data = event_data[event_data[entity_col].isin(necessary_entities)]

        necessary_tols = set(tolerances_to_calc)
        loaded_csv = pd.read_csv(file_path)
        present_tols = set(np.unique(loaded_csv['Tolerance'].values))
        if necessary_tols.issubset(present_tols):
            conditions.append(True)
            event_data = event_data[event_data['Tolerance'].isin(necessary_tols)]

        make_predictions = ~np.array(conditions).all()

    if make_predictions:
        random_seeds = event_dict.get('rs')
        if isinstance(random_seeds, int):
            random_seeds = [random_seeds]
        nsamples = event_dict.get('nsamples')
        if isinstance(nsamples, int):
            nsamples = [nsamples]
        if time_col is None:
            print("No time col provided, assuming dataset is ordered ascendingly by date")

        names = ["Random Seed", "NSamples", "Event",  "Shapley Value", "t (event index)", entity_col, 'Tolerance']
        # create directory
        if '/' in file_path:
            Path(file_path.rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(names)

        for rs in random_seeds:
            for ns in nsamples:
                for uuid in np.unique(data[entity_col].values):
                    event_data = None
                    prev_pruning_idx = None
                    for tol in tolerances_to_calc:
                        seq = data[data[entity_col] == uuid]
                        if time_col:
                            seq = seq.sort_values(time_col)
                        seq = seq[model_features]
                        seq = np.expand_dims(seq.to_numpy().copy(), axis=0)
                        if pruning_data is None:
                            raise NotImplementedError
                        else:
                            instance = pruning_data[pruning_data[entity_col] == uuid]
                            pruning_idx = instance[instance['Tolerance'] == tol]['Pruning idx'].iloc[0]
                            pruning_idx = seq.shape[1] + pruning_idx
                            if prev_pruning_idx == pruning_idx:
                                # we have already calculated this, let's use it from the last iteration
                                event_data['Tolerance'] = tol
                            else:
                                local_event_dict = {'rs': rs, 'nsamples': ns}
                                event_data = local_event(f, seq, local_event_dict, uuid, entity_col, baseline, pruning_idx)
                                event_data['Event index'] = event_data['Feature'].apply(lambda x: 1 if x == 'Pruned Events' else -int(re.findall(r'\d+', x)[0])+1)
                                event_data[entity_col] = uuid
                                event_data['Tolerance'] = tol
                                if file_path is not None:
                                    with open(file_path, 'a', newline='') as file:
                                        writer = csv.writer(file, delimiter=',')
                                        writer.writerows(event_data.values)
                            prev_pruning_idx = pruning_idx
        event_data = pd.read_csv(file_path)
    return event_data
