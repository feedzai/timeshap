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

from typing import List
import pandas as pd
import numpy as np
import copy


def calc_avg_event(data: pd.DataFrame,
                   numerical_feats: List[str],
                   categorical_feats: List[str],
                   ) -> pd.DataFrame:
    """Calculates the median of numerical features, and the mode for categorical
    features of a pandas DataFrame

    Parameters
    ----------
    data: pd.DataFrame
        Dataset to use for baseline calculation

    numerical_feats: List
        List of numerical features to calculate median of

    categorical_feats: List
        List of categorical features to calculate mode of

    Returns
    -------
    pd.DataFrame
        DataFrame with the median/mode of the features
    """
    numerical = data[numerical_feats].astype(float).describe().loc[["50%"]].reset_index(drop=True)
    categorical = data[categorical_feats].mode()
    ordered_feats = [x for x in list(data.columns) if x in numerical_feats + categorical_feats]
    return pd.concat([numerical, categorical], axis=1)[ordered_feats]


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
    for x in range(1, top):
        expanded_copy = copy.deepcopy(expanded)
        if len(model.__code__.co_varnames) == 1:
            pred = model(expanded_copy)
        else:
            pred, hs = model(expanded_copy, hs)
        avg_score[x] = float(pred[0])
    return avg_score
