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

from typing import Callable, Union
import numpy as np
import pandas as pd
from timeshap.plot import plot_temp_coalition_pruning, plot_event_heatmap, plot_feat_barplot, plot_cell_level
from timeshap.explainer import local_pruning, local_event, local_feat, local_cell_level


def validate_local_input(f: Callable[[np.ndarray], np.ndarray],
                         data: Union[pd.DataFrame, np.array],
                         pruning_dict: dict,
                         event_dict: dict,
                         feature_dict: dict,
                         model_features=None,
                         entity_col=None,
                         time_col=None,
                         ):
    """Verifies for local inputs if inputs are according"""

    assert isinstance(f, Callable), "Provided model must be callable"
    assert isinstance(data, (pd.DataFrame, np.ndarray)), "Provided data must be an numpy array with 3 dimensions"
    if isinstance(data, pd.DataFrame):
        data_cols = set(data.columns)
        if model_features:
            assert set(model_features).issubset(data_cols), "When providing model features, these should be on the given DataFrame"
        if entity_col:
            assert set(entity_col).issubset(data_cols), "When providing entity feature, these should be on the given DataFrame"
        if time_col:
            assert set(time_col).issubset(data_cols), "When providing time feature, these should be on the given DataFrame"
    else:
        assert len(data.shape) == 3, "Provided data must be an numpy array with 3 dimensions"
    assert data.shape[0] == 1, "For local report, provided data must contain one instance only"

    assert pruning_dict.get("tol") is not None, "Prunning dict must have tolerance attribute"
    assert isinstance(pruning_dict.get("tol"), (int, float)), "Provided tolerance must be a int or float"

    if event_dict.get('rs'):
        assert isinstance(event_dict.get("rs"), int), "Provided random seed must be a int"
    if event_dict.get('nsamples'):
        assert isinstance(event_dict.get("nsamples"), int), "Provided nsamples must be a int"

    if feature_dict.get('rs'):
        assert isinstance(feature_dict.get("rs"), int), "Provided random seed must be a int or float"
    if feature_dict.get('nsamples'):
        assert isinstance(feature_dict.get("nsamples"), int), "Provided nsamples must be a int"
    if feature_dict.get('top_feats'):
        assert isinstance(feature_dict.get("top_feats"), int), "Provided top_feats must be a int"
    if feature_dict.get('plot_features'):
        assert isinstance(feature_dict.get("plot_features"), dict), "Provided plot_features must be a dict, mapping model features, to plot features"


def local_report(f: Callable[[np.ndarray], np.ndarray],
                 data: Union[pd.DataFrame, np.array],
                 pruning_dict: dict,
                 event_dict: dict,
                 feature_dict: dict,
                 cell_dict: dict = None,
                 entity_uuid=None,
                 entity_col=None,
                 time_col=None,
                 model_features=None,
                 baseline=None,
                 verbose=False,
                 ):
    """Plots local feature explanations

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: Union[pd.DataFrame, np.array]
        Sequence to be explained.

    pruning_dict: dict
        Information required for pruning algorithm

    event_dict: dict
        Information required for the event level explanation calculation

    feature_dict: dict
        Information required for the feature level explanation calculation

    cell_dict: dict
        Information required for the cell level explanation calculation

    entity_uuid: Union[str, int, float]
        The indentifier of the sequence that is being pruned.
        Used when fetching information from a csv of explanations

    entity_col: str
        Entity column to identify sequences

    time_col: str
        Data column that represents the time feature in order to sort sequences
        temporally

    model_features: List[str]
        Features to be used by the model. Requires same order as training dataset

    baseline: Union[pd.DataFrame, np.array]
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.

    verbose: bool
        If process is verbose

    Returns
    -------
    pd.DataFrame
    """
    validate_local_input(f, data, pruning_dict, event_dict, feature_dict, model_features, entity_col, time_col)
    # deals with given date being a DataFrame
    if isinstance(data, pd.DataFrame):
        if time_col:
            data = data.sort_values(time_col)
        data = data[model_features]
        data = np.expand_dims(data.to_numpy().copy(), axis=0)

    coal_plot_data, coal_prun_idx = local_pruning(f, data, pruning_dict, baseline, entity_uuid, entity_col, verbose)
    # coal_prun_idx is in negative terms
    pruning_idx = data.shape[1] + coal_prun_idx
    plot_lim = max(abs(coal_prun_idx)+10, 40)
    pruning_plot = plot_temp_coalition_pruning(coal_plot_data, coal_prun_idx, plot_lim)

    event_data = local_event(f, data, event_dict, entity_uuid, entity_col, baseline, pruning_idx)
    event_plot = plot_event_heatmap(event_data)

    feature_data = local_feat(f, data, feature_dict, entity_uuid, entity_col, baseline, pruning_idx)
    feature_plot = plot_feat_barplot(feature_data, feature_dict.get('top_feats'), feature_dict.get('plot_features'))

    if cell_dict:
        cell_data = local_cell_level(f, data, cell_dict, event_data, feature_data, entity_uuid, entity_col, baseline, pruning_idx)
        feat_names = list(feature_data['Feature'].values)[:-1] # exclude pruned events
        cell_plot = plot_cell_level(cell_data, feat_names, feature_dict.get('plot_features'))
        plot_report = (pruning_plot | event_plot | feature_plot | cell_plot).resolve_scale(color='independent')
    else:
        plot_report = (pruning_plot | event_plot | feature_plot).resolve_scale(color='independent')

    return plot_report
