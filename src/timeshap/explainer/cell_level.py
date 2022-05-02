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
from typing import Callable, Union, Tuple, List
import numpy as np
import pandas as pd
from timeshap.explainer.kernel import TimeShapKernel
import os
from pathlib import Path


def cell_top_events(event_data: pd.DataFrame,
                    event_threshold: float = None,
                    top_x_events: int = None,
                    **kwargs,
                    ) -> Tuple[list, list]:
    """Calculates the indexes and names of events to participate in the cell level
    given the conditions

    Parameters
    ----------
    event_data: pd.DataFrame
        Event level explanations.

    event_threshold: float
        The threshold to consider an event relevant.

    top_x_events: float
        Number of events to include as relevant

    Returns
    -------
    Tuple[list, list]
        Tuple containing two lists.
        The first list contains the event indexes ordered ascendingly.
        The second list contains the correspondent event names.
    """
    # order explanations by absolute contribution
    ordered_exp = event_data.iloc[(-event_data['Shapley Value'].abs()).argsort()].reset_index()

    top_events_idx = []
    for _, row in ordered_exp.iterrows():
        if row['Feature'] == 'Pruned Events':
            # we want to skip previous events
            continue
        if event_threshold is not None and abs(row['Shapley Value']) < event_threshold:
            # we have reached the sopping point
            break

        top_events_idx += [[-row['index'] -1, row['Feature']]]

        if top_x_events is not None and len(top_events_idx) == top_x_events:
            # we have added enough events
            break

    df = pd.DataFrame(top_events_idx, columns=['idx', 'name']).sort_values('idx')
    return list(df['idx'].values), list(df['name'].values)


def cell_top_feats(feat_data: pd.DataFrame,
                   feat_threshold: float = None,
                   top_x_feats: int = None,
                   **kwargs,
                   ) -> Tuple[list, list]:
    """Calculates the indexes and names of events to participate in the cell level
    given the conditions

    Parameters
    ----------
    feat_data: pd.DataFrame
        Feature level explanations.

    feat_threshold: float
        The threshold to consider a feature relevant.

    top_x_feats: float
        Number of features to include as relevant

    Returns
    -------
    Tuple[list, list]
        Tuple containing two lists.
        The first list contains the feature indexes ordered ascendingly.
        The second list contains the correspondent feature names.
    """
    ordered_feats = feat_data.iloc[(-feat_data['Shapley Value'].abs()).argsort()].reset_index()

    top_feats_idx = []
    for _, row in ordered_feats.iterrows():
        if row['Feature'] == 'Pruned Events':
            # we want to skip previous events
            continue
        if feat_threshold is not None and abs(row['Shapley Value']) < feat_threshold:
            # we have reached the sopping point
            break
        top_feats_idx += [[row['index'], row['Feature']]]
        if top_x_feats is not None and len(top_feats_idx) == top_x_feats:
            # we have added enough events
            break

    df = pd.DataFrame(top_feats_idx, columns=['feat', 'name']).sort_values('feat')
    return list(df['feat'].values), list(df['name'].values)


def considered_cells(event_data: pd.DataFrame,
                     feat_data: pd.DataFrame,
                     **kwargs,
                     ) -> Tuple[Tuple[List, List], Tuple[List, List]]:
    """Calculates the indexes and names of events to participate in the cell level
    given the conditions

    Parameters
    ----------
    event_data: pd.DataFrame
        Event level explanations.

    feat_data: pd.DataFrame
        Feature level explanations.

    Returns
    -------
    Tuple[Tuple[List, List], Tuple[List, List]]
        Events and features to include in cell levle computations
        and their respective name
    """
    top_events_idx, top_events_names = cell_top_events(event_data, **kwargs)
    top_feats_idx, top_feats_names = cell_top_feats(feat_data, **kwargs)

    return (top_events_idx, top_feats_idx), (top_events_names,top_feats_names)


def cell_level(f: Callable,
               data: np.ndarray,
               baseline: Union[pd.DataFrame, np.ndarray],
               event_data: pd.DataFrame,
               feat_data: pd.DataFrame,
               random_seed: int,
               nsamples: int,
               cell_dict: dict,
               pruned_idx: int,
               model_feats=None,
               ) -> pd.DataFrame:
    """Cell level given relevant events and features

    Parameters
    ----------
    f : Callable
        Prediction method of model being explained.
        Will be called with a 3-D input

    data: numpy.ndarray
        Input matrix to use. First element of the first dimension is explained,
        using the rest of the elements as context/hidden state.

    baseline: numpy.ndarray
        Baseline event to use. Median of numerical and mode for categorical.

    event_data: pd.DataFrame
        Event level explanations.

    feat_data: pd.DataFrame
        Feature level explanations.

    random_seed: int
        Used random seed for the sampling process.

    nsamples: int
        The number of coalitions for TimeSHAP to sample.

    cell_dict: dict
        Information required for the cell level explanation calculation

    pruned_idx: int
        Index to prune the sequence. All events up to this point are grouped\

    model_feats: List
        The list of feature names.
        If none is provided, "Feature 1" format is used

    Returns
    -------
    pd.DataFrame
    """
    kwargs = {}
    if cell_dict.get('threshold', False):
        # single threshold for everything
        kwargs['event_threshold'] = cell_dict.get('threshold')
        kwargs['feat_threshold'] = cell_dict.get('threshold')
    elif cell_dict.get('event_threshold', False) and cell_dict.get('feat_threshold', False):
        kwargs['event_threshold'] = cell_dict.get('event_threshold')
        kwargs['feat_threshold'] = cell_dict.get('feat_threshold')
    elif cell_dict.get('top_x', False):
        kwargs['top_x_events'] = cell_dict.get('top_x')
        kwargs['top_x_feats'] = cell_dict.get('top_x')
    elif cell_dict.get('top_x_events', False) and cell_dict.get('top_x_feats', False):
        kwargs['top_x_events'] = cell_dict.get('top_x_events')
        kwargs['top_x_feats'] = cell_dict.get('top_x_feats')
    else:
        raise ValueError("No threshold condition provided for cell level")

    varying_cells, names = considered_cells(event_data, feat_data, **kwargs)

    negative_indexes = np.array([False if x >= 0 else True for x in varying_cells[0]])
    if any(negative_indexes):
        if not all(negative_indexes):
            raise ValueError("All indexes must be positive or negative. Not both")
        varying_cells = ([data.shape[1] + x for x in varying_cells[0]], varying_cells[1])

    explainer = TimeShapKernel(f, baseline, random_seed, "cell", varying=varying_cells)
    explanation = explainer.shap_values(data, pruning_idx=pruned_idx, nsamples=nsamples)

    ret_df_data = []
    i = 0
    for event in names[0]:
        for feat in names[1]:
            row = [event, feat, explanation[i]]
            i += 1
            ret_df_data += [row]

    if explainer.special_cells[0]:
        for event in names[0]:
            row = [event, 'Other Features', explanation[i]]
            i += 1
            ret_df_data += [row]

    if explainer.special_cells[1]:
        for feat in names[1]:
            row = ['Other Events', feat, explanation[i]]
            i += 1
            ret_df_data += [row]

    if explainer.special_cells[2]:
        ret_df_data += [["Other Events", "Other Features", explanation[i]]]
        i += 1
    if explainer.special_cells[3]:
        ret_df_data += [["Pruned Events", "Pruned Events", explanation[i]]]
    return pd.DataFrame(ret_df_data, columns=['Event', 'Feature',
                                              'Shapley Value']).sort_values(
        'Shapley Value', ascending=False)


def local_cell_level(f: Callable[[np.ndarray], np.ndarray],
                     data: Union[pd.DataFrame, np.array],
                     cell_dict: dict,
                     event_data: pd.DataFrame,
                     feat_data: pd.DataFrame,
                     entity_uuid,
                     entity_col,
                     baseline,
                     pruned_idx,
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

    cell_dict: dict
        Information required for the cell level explanation calculation

    event_data: pd.DataFrame
        Event level explanations.

    feat_data: pd.DataFrame
        Feature level explanations.

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

    if cell_dict.get("path") is None or not os.path.exists(cell_dict.get("path")):
        print("No path to cell data provided. Calculating data")
        cell_data = cell_level(f, data, baseline, event_data, feat_data, cell_dict.pop("rs"), cell_dict.pop("nsamples"), cell_dict, pruned_idx)
        if cell_dict.get("path") is not None:
            # create directory
            if '/' in cell_dict.get("path"):
                Path(cell_dict.get("path").rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
            cell_data.to_csv(cell_dict.get("path"), index=False)
    elif cell_dict.get("path") is not None and os.path.exists(cell_dict.get("path")):
        cell_data = pd.read_csv(cell_dict.get("path"))
        if len(cell_data.columns) == 5 and entity_col is not None:
            cell_data = cell_data[cell_data[entity_col] == entity_uuid]
        elif len(cell_data.columns) == 4:
            pass
        else:
            # TODO
            # the provided csv should be generated by timeshap, by either
            # explaining the whole dataset with TODO or just the instance in question
            raise ValueError
    else:
        raise ValueError
    return cell_data
