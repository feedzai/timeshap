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
import pandas as pd
from timeshap.plot import plot_temp_coalition_pruning, plot_event_heatmap, plot_feat_barplot, plot_cell_level
from timeshap.explainer import prune_given_data


def plot_local_report(pruning_dict: dict,
                      event_dict: dict,
                      feature_dict: dict,
                      cell_dict: dict,
                      coal_plot_data: pd.DataFrame = None,
                      event_data: pd.DataFrame = None,
                      feat_data: pd.DataFrame = None,
                      cell_data: pd.DataFrame = None,
                      ):
    """Plots a local report given explanations

    Parameters
    ----------
    pruning_dict: dict
        Information required for the pruning algorithm

    event_dict: dict
        Information required for the event level explanation calculation

    feature_dict: dict
        Information required for the feature level explanation calculation

    cell_dict: dict
        Information required for the cell level explanation calculation

    coal_plot_data: pd.DataFrame
        Pruning algorithm data to plot

    event_data: pd.DataFrame
        Event explanations to plot

    feat_data: pd.DataFrame
        Feature explanations to plot

    cell_data: pd.DataFrame
        Cell explanations to plot

    Returns
    -------
    altair.plot
        The local report
    """
    if coal_plot_data is None:
        assert pruning_dict.get('path', False), "No data or path to data provided to calculate pruning statistics"
    if event_data is None:
        assert event_dict.get('path', False), "No data or path to data provided to plot event explanations"
    if feat_data is None:
        assert feature_dict.get('path', False), "No data or path to data provided to plot feature explanations"
    if cell_data is None and cell_dict is not None:
        assert cell_dict.get('path', False), "No data or path to data provided to plot feature explanations"

    if coal_plot_data is None:
        coal_plot_data = pd.read_csv(pruning_dict.get('path'))
    if event_data is None:
        event_data = pd.read_csv(event_dict.get('path'))
    if feat_data is None:
        feat_data = pd.read_csv(feature_dict.get('path'))
    if cell_data is None and cell_dict is not None:
        cell_data = pd.read_csv(cell_dict.get('path'))

    coal_prun_idx = prune_given_data(coal_plot_data, pruning_dict.get('tol'))
    plot_lim = max(abs(coal_prun_idx)+10, 40)
    pruning_plot = plot_temp_coalition_pruning(coal_plot_data, coal_prun_idx, plot_lim)

    event_plot = plot_event_heatmap(event_data)

    feature_plot = plot_feat_barplot(feat_data, feature_dict.get('top_feats'), feature_dict.get('plot_features'))

    if cell_dict:
        feat_names = list(feat_data['Feature'].values)[:-1]  # exclude pruned events
        cell_plot = plot_cell_level(cell_data, feat_names, feature_dict.get('plot_features'))
        plot_report = (pruning_plot | event_plot | feature_plot | cell_plot).resolve_scale(color='independent')
    else:
        plot_report = (pruning_plot | event_plot | feature_plot).resolve_scale(color='independent')

    return plot_report
