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
import csv
from pathlib import Path
import copy
from timeshap.utils import get_tolerances_to_test
from timeshap.utils import convert_to_indexes, convert_data_to_3d
from timeshap.explainer import temp_coalition_pruning


def feature_level(f: Callable,
                  data: np.ndarray,
                  baseline: np.ndarray,
                  pruned_idx: int,
                  random_seed: int,
                  nsamples: int,
                  model_feats=None,
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

    model_feats: List
        The list of feature names.
        If none is provided, "Feature 1" format is used

    Returns
    -------
    pd.DataFrame
    """
    if pruned_idx == -1:
        pruned_idx = 0

    explainer = TimeShapKernel(f, baseline, random_seed, "feature")
    shap_values = explainer.shap_values(data, pruning_idx=pruned_idx, nsamples=nsamples)

    if model_feats is None:
        model_feats = ["Feature {}".format(i) for i in np.arange(data.shape[2])]

    model_feats = copy.deepcopy(model_feats)
    if pruned_idx > 0:
        model_feats += ["Pruned Events"]

    ret_data = []
    for exp, feature in zip(shap_values, model_feats):
        ret_data += [[random_seed, nsamples, feature, exp]]
    return pd.DataFrame(ret_data, columns=['Random seed', 'NSamples', 'Feature', 'Shapley Value'])


def local_feat(f: Callable[[np.ndarray], np.ndarray],
               data: Union[pd.DataFrame, np.array],
               feature_dict: dict,
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

    feature_dict: dict
        Information required for the feature level explanation calculation

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
    if feature_dict.get("path") is None or not os.path.exists(feature_dict.get("path")):
        #print("No path to feature data provided. Calculating data")
        feat_data = feature_level(f, data, baseline, pruned_idx, feature_dict.get("rs"), feature_dict.get("nsamples"), model_feats=feature_dict.get("feature_names"))
        if feature_dict.get("path") is not None:
            # create directory
            if '/' in feature_dict.get("path"):
                Path(feature_dict.get("path").rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
            feat_data.to_csv(feature_dict.get("path"), index=False)
    elif feature_dict.get("path") is not None and os.path.exists(feature_dict.get("path")):
        feat_data = pd.read_csv(feature_dict.get("path"))
        if len(feat_data.columns) == 5 and entity_col is not None:
            feat_data = feat_data[feat_data[entity_col] == entity_uuid]
        elif len(feat_data.columns) == 4:
            pass
        else:
            # TODO
            # the provided csv should be generated by timeshap, by either
            # explaining the whole dataset with TODO or just the instance in question
            raise ValueError
    else:
        raise ValueError
    return feat_data


def verify_feature_dict(feature_dict: dict):
    if feature_dict.get('path'):
        assert isinstance(feature_dict.get('path'), str), "Provided path must be a string"

    if feature_dict.get('rs', False):
        if isinstance(feature_dict.get('rs'), int):
            feature_dict['rs'] = [feature_dict.get('rs')]
        elif isinstance(feature_dict.get('rs'), list):
            assert np.array([isinstance(x, int) for x in feature_dict.get('rs')]).all(), "All provided random seeds must be ints."
        else:
            raise ValueError(
                "Unsuported format of random seeds(s). Please provide one seed or a list of them.")
    else:
        print("No random seed provided for event-level explanations. Using default: 42")
        feature_dict['rs'] = [42]

    if feature_dict.get('nsamples', False):
        if isinstance(feature_dict.get('nsamples'), int):
            feature_dict['nsamples'] = [feature_dict.get('nsamples')]
        elif isinstance(feature_dict.get('nsamples'), list):
            assert np.array([isinstance(x, int) for x in feature_dict.get('nsamples')]).all(), "All provided nsamples must be ints."
        else:
            raise ValueError("Unsuported format of nsamples. Please provide value or a list of them.")
    else:
        print("No nsamples provided for event-level explanations. Using default: 32000")
        feature_dict['nsamples'] = [32000]

    if feature_dict.get('tol', False):
        tolerances = feature_dict.get('tol')
        if isinstance(tolerances, float):
            feature_dict['tol'] = [tolerances]
        elif isinstance(tolerances, list):
            assert np.array([isinstance(x, float) for x in tolerances]).all(), "All provided tolerances must be floats."

    if feature_dict.get('plot_features', False):
        assert isinstance(feature_dict.get('plot_features'), dict)
        assert np.array([isinstance(x, str) for x in feature_dict.get('plot_features').values()]).all(), "All provided plot features must be strings."

    if feature_dict.get('feature_names', False):
        assert isinstance(feature_dict.get('feature_names'), list)
        assert np.array([isinstance(x, str) for x in feature_dict.get('feature_names')]).all(), "All provided features must be strings."


def feat_explain_all(f: Callable,
                     data: Union[List[np.ndarray], pd.DataFrame, np.array],
                     feat_dict: dict,
                     pruning_data: pd.DataFrame,
                     baseline: Union[pd.DataFrame, np.array] = None,
                     model_features: List[Union[int, str]] = None,
                     schema: List[str] = None,
                     entity_col: Union[int, str] = None,
                     time_col: Union[int, str] = None,
                     append_to_files: bool = False,
                     verbose: bool = False,
                     ):
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

    feat_dict: dict
        Information required for the feature level explanation calculation

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
    verify_feature_dict(feat_dict)
    file_path = feat_dict.get('path')
    make_predictions = True
    feat_data = None

    tolerances_to_calc = get_tolerances_to_test(pruning_data, feat_dict, entity_col)

    if file_path is not None and os.path.exists(file_path) and not append_to_files:
        feat_data = pd.read_csv(file_path)
        make_predictions = False

        # TODO resume explanations
        # conditions = []
        # necessary_entities = set(np.unique(data[entity_col].values))
        # feat_data = pd.read_csv(file_path)
        # present_entities = set(np.unique(feat_data[entity_col].values))
        # if necessary_entities.issubset(present_entities):
        #     conditions.append(True)
        #     feat_data = feat_data[feat_data[entity_col].isin(necessary_entities)]
        #
        # necessary_tols = set(tolerances_to_calc)
        # loaded_csv = pd.read_csv(file_path)
        # present_tols = set(np.unique(loaded_csv['Tolerance'].values))
        # if necessary_tols.issubset(present_tols):
        #     conditions.append(True)
        #     feat_data = feat_data[loaded_csv['Tolerance'].isin(necessary_tols)]
        #
        # make_predictions = ~np.array(conditions).all()

    if make_predictions:
        random_seeds = feat_dict.get('rs')
        nsamples = feat_dict.get('nsamples')
        names = ["Random Seed", "NSamples", "Feature",  "Shapley Value", "Entity", 'Tolerance']

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

        ret_feat_data = []
        for rs in random_seeds:
            for ns in nsamples:
                for sequence in data:
                    if entity_col is not None:
                        entity = sequence[0, 0, entity_col_index]
                    if model_features:
                        sequence = sequence[:, :, model_features]
                    sequence = sequence.astype(np.float64)
                    feat_data = None
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
                            feat_data['Tolerance'] = tol
                        else:
                            local_feat_dict = {'rs': rs, 'nsamples': ns}
                            if feat_dict.get('feature_names'):
                                local_feat_dict['feature_names'] = feat_dict.get('feature_names')
                            feat_data = local_feat(f, sequence, local_feat_dict, entity, entity_col, baseline, pruning_idx)
                            feat_data[entity_col] = entity
                            feat_data['Tolerance'] = tol

                        if file_path is not None:
                            with open(file_path, 'a', newline='') as file:
                                writer = csv.writer(file, delimiter=',')
                                writer.writerows(feat_data.values)
                        ret_feat_data.append(feat_data.values)
                        prev_pruning_idx = pruning_idx

        feat_data = pd.DataFrame(np.concatenate(ret_feat_data), columns=names)
        feat_data = feat_data.astype({'NSamples': 'int', 'Random Seed': 'int', 'Tolerance': 'float', 'Shapley Value': 'float'})
    return feat_data
