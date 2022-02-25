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
from timeshap.explainer import prune_all, pruning_statistics, event_explain_all, feat_explain_all
from timeshap.plot import plot_global_event, plot_global_feat


def validate_global_input():
    assert True


def global_report(f: Callable[[np.ndarray], np.ndarray],
                  data: Union[pd.DataFrame, np.array],
                  pruning_dict: dict,
                  event_dict: dict,
                  feature_dict: dict,
                  entity_col,
                  time_col,
                  model_features,
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
    validate_global_input()

    print("Calculating pruning algorithms")
    prun_indexes = prune_all(f, data, entity_col, baseline, pruning_dict, model_features, time_col)
    print("Calculating pruning indexes")
    pruning_stats = pruning_statistics(prun_indexes, pruning_dict.get('tol'), entity_col)
    print("Calculating event data")
    event_data = event_explain_all(f, data, entity_col, baseline, event_dict, prun_indexes, model_features, time_col)
    print("Calculating global event plot")
    event_global_plot = plot_global_event(event_data)

    print("Calculating feat data")
    feat_data = feat_explain_all(f, data, entity_col, baseline, feature_dict, prun_indexes, model_features, time_col)
    print("Calculating global feat plot")
    feat_global_plot = plot_global_feat(feat_data, **feature_dict)
    return pruning_stats, (event_global_plot | feat_global_plot).resolve_scale(color='independent')
