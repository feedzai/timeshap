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
import numpy as np


def filter_dataset(data: pd.DataFrame,
                   tol: float,
                   rs: int,
                   nsamples: int
                   ) -> pd.DataFrame:
    filtered_data = data[
        (data['Tolerance'] == tol) & (data['Random Seed'] == rs) & (
                    data['NSamples'] == nsamples)
        ]
    return filtered_data


def find_parameters_to_plot(event_dict: dict,
                           feature_dict: dict,
                           event_data: pd.DataFrame,
                           feat_data: pd.DataFrame
                           ):
    def make_list(element):
        if isinstance(element, (int, float)):
            element = [element]
        elif isinstance(element, list):
            pass
        else:
            raise ValueError("Unrecognized parameter format")
        return element

    def calculate_list_intersection(A: list, B: list):
        intersection = list(set(A).intersection(set(B)))
        assert len(intersection) > 0, "Asking to plot parameter not present on data"
        return intersection

    event_data_nsamples = list(np.unique(event_data["NSamples"].values))
    event_data_rs = list(np.unique(event_data["Random Seed"].values))
    event_data_tol = list(np.unique(event_data["Tolerance"].values))

    event_nsamples = make_list(event_dict.get('nsamples', event_data_nsamples))
    event_rs = make_list(event_dict.get('rs', event_data_rs))
    event_tol = make_list(event_dict.get('tol', event_data_tol))

    allowed_event_nsamples = calculate_list_intersection(event_data_nsamples, event_nsamples)
    allowed_event_rs = calculate_list_intersection(event_data_rs, event_rs)
    allowed_event_tol = calculate_list_intersection(event_data_tol, event_tol)

    feat_data_nsamples = list(np.unique(feat_data["NSamples"].values))
    feat_data_rs = list(np.unique(feat_data["Random Seed"].values))
    feat_data_tol = list(np.unique(feat_data["Tolerance"].values))

    feature_nsamples = make_list(feature_dict.get('nsamples', feat_data_nsamples))
    feature_rs = make_list(feature_dict.get('rs', feat_data_rs))
    feature_tol = make_list(feature_dict.get('tol', feat_data_tol))

    allowed_feat_nsamples = calculate_list_intersection(feat_data_nsamples, feature_nsamples)
    allowed_feat_rs = calculate_list_intersection(feat_data_rs, feature_rs)
    allowed_feat_tol = calculate_list_intersection(feat_data_tol, feature_tol)

    return calculate_list_intersection(allowed_event_tol, allowed_feat_tol), \
           calculate_list_intersection(allowed_event_rs, allowed_feat_rs), \
           calculate_list_intersection(allowed_event_nsamples, allowed_feat_nsamples)