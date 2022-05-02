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
import csv
from pathlib import Path


def pruning_statistics(df: pd.DataFrame,
                       tol: Union[float, list],
                       entity_col: str,
                       ):
    """Calculates global pruning statistics with the given tolerances.

    Parameters
    ----------
    df: pd.DataFrame
        Pruning data to be analysed produced by `prune_all`

    tol: Union[float, list]
        The tolerances to analyze the pruning

    entity_col: str
        Entity column to identify sequences

    Returns
    -------
    pd.DataFrame
    """
    resume = []
    orig = df[df['Tolerance'] == -1]
    for idx, row in orig.iterrows():
        resume += [["Original", 'No Pruning', row[entity_col], -row['Pruning idx']]]
    if isinstance(tol, float):
        tol = [tol]
    for tol in tol:
        tolerance_sequences = df[df['Tolerance'] == tol]
        for idx, row in tolerance_sequences.iterrows():
            resume.append(["Pruning", tol,  row[entity_col], -row['Pruning idx']])

    resume_df = pd.DataFrame(resume, columns=["Algorithm", "Tolerance", entity_col, "Sequence Length"])
    resume_df['Mean'] = resume_df['Sequence Length']
    resume_df['Std'] = resume_df['Sequence Length']
    resume_df = resume_df.groupby("Tolerance").agg({"Mean": "mean", "Std": "std"})
    resume_df.reset_index(inplace=True)
    resume_df = resume_df.rename(columns={'index': 'Tolerance'})
    return resume_df


def prune_given_data(data: pd.DataFrame, tolerance: float) -> int:
    """Calculates the pruning index to prune the sequence to with a given tolerance

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe containing the pruning algorithm information

    tolerance: str
        Tolerance to prun the sequence

    Returns
    -------
    int
    """
    data = data[data['Coalition'] == 'Sum of contribution of events \u2264 t']
    if tolerance == 0:
        # to filter float unprecision out
        tolerance = 0.000001
    respecting_lens = data[data['Shapley Value'].abs() <= tolerance]
    if respecting_lens.shape[0] == 0:
        return -data['t (event index)'].min()

    return respecting_lens.iloc[0]['t (event index)']


def temp_coalition_pruning(f: Callable,
                           data: np.ndarray,
                           baseline: Union[np.ndarray, pd.DataFrame],
                           tolerance: float = None,
                           ret_plot_data=False,
                           verbose=False,
                           ) -> Union[int, pd.DataFrame, Tuple[int, pd.DataFrame]]:
    """Temporal coalition pruning method

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: numpy.ndarray
        Input matrix to use. First element of the first dimension is explained,
        using the rest of the elements as context/hidden state.

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.

    tolerance: float
        Temporal coalition explainer tolerance.
        Represents the maximum allowed Shapley Value of the older grouped events.

    ret_plot_data: bool
        If method returns pruning algorithm across the whole sequence

    verbose: bool
        If process is verbose

    Returns
    -------
    Union[int, pd.DataFrame, Tuple[int, pd.DataFrame]]:
        int:
            Pruning index
        pd.DataFrame
            Pruning data over the whole sequence
         Tuple[int, pd.DataFrame]]
            Pruning index and Pruning data over the whole sequence
    """
    if verbose:
        print("Allowed importance for pruned events: {}".format(tolerance))

    if ret_plot_data:
        plot_data = []
    pruning_idx = 0
    for seq_len in range(data.shape[1], -1, -1):
        explainer = TimeShapKernel(f, baseline, 0, "pruning")
        shap_values = explainer.shap_values(data, pruning_idx=seq_len, **{'nsamples': 4})
        if ret_plot_data:
            plot_data += [['Sum of contribution of events \u003E t', -data.shape[1]+seq_len, shap_values[0]]]
            plot_data += [['Sum of contribution of events \u2264 t', -data.shape[1]+seq_len, shap_values[1]]]

        if verbose:
            print("len {} | importance {}".format(-data.shape[1] + seq_len, shap_values[1]))
        if seq_len < data.shape[1] and tolerance and abs(shap_values[1]) <= tolerance:
            if pruning_idx == 0:
                pruning_idx = -data.shape[1] + seq_len
            if not ret_plot_data:
                return pruning_idx

    if tolerance is not None and pruning_idx == 0:
        pruning_idx = -data.shape[1]

    if tolerance is not None and ret_plot_data:
        # used for plotting
        return pruning_idx,  pd.DataFrame(plot_data, columns=['Coalition', 't (event index)', 'Shapley Value'])
    if tolerance is not None and not ret_plot_data:
        # used for event level
        return pruning_idx
    return pd.DataFrame(plot_data, columns=['Coalition', 't (event index)', 'Shapley Value'])


def local_pruning(f: Callable[[np.ndarray], np.ndarray],
                  data: np.ndarray,
                  pruning_dict: dict,
                  baseline: Union[np.ndarray, pd.DataFrame],
                  entity_uuid: Union[str, int, float] = None,
                  entity_col: str = None,
                  verbose: bool = False,
                  ) -> Tuple[pd.DataFrame, int]:
    """Method to prune a sequence or fetch the respective information if a path
    is provided

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: numpy.ndarray
        Input matrix to use. First element of the first dimension is explained,
        using the rest of the elements as context/hidden state.

    pruning_dict: dict
        Information required for pruning algorithm

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.

    entity_uuid: Union[str, int, float]
        The indentifier of the sequence that is being pruned.
        Used when fetching information from a csv of explanations

    entity_col: str
        Column that contains the sequence identifiers
        Used when fetching information from a csv of explanations

    verbose: bool
        If process is verbose

    Returns
    -------
    Tuple[int, pd.DataFrame]]
            Pruning index and Pruning data over the whole sequence
    """
    def calculate_pruning():
        if baseline is None:
            raise ValueError("Baseline is not defined")
        coal_prun_idx, coal_plot_data = temp_coalition_pruning(f,
                                                               data,
                                                               baseline,
                                                               pruning_dict['tol'],
                                                               ret_plot_data=True,
                                                               verbose=verbose)

        return coal_prun_idx, coal_plot_data

    if pruning_dict.get("path") is None or not os.path.exists(pruning_dict.get("path")):
        print("No path to explainer data provided. Calculating data")
        if baseline is None:
            raise ValueError("Baseline is not defined")
        coal_prun_idx, coal_plot_data = calculate_pruning()
        if pruning_dict.get("path") is not None:
            # create directory
            if '/' in pruning_dict.get("path"):
                Path(pruning_dict.get("path").rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
            coal_plot_data.to_csv(pruning_dict.get("path"), index=False)

    elif pruning_dict.get("path") is not None and os.path.exists(pruning_dict.get("path")):
        coal_plot_data = pd.read_csv(pruning_dict.get("path"))
        if len(coal_plot_data.columns) > 3:
            # global df
            assert entity_uuid is not None, "When using a dataset with several instances, a uuid needs to be provided"
            coal_plot_data = coal_plot_data[coal_plot_data[entity_col] == entity_uuid]
        coal_prun_idx = prune_given_data(coal_plot_data, pruning_dict.get('tol'))
    else:
        raise ValueError('Unrecognized explainer procedure.')
    return coal_plot_data, coal_prun_idx


def prune_all(f: Callable,
              data: pd.DataFrame,
              entity_col: str,
              baseline: Union[pd.DataFrame, np.ndarray],
              pruning_dict: dict,
              model_features: List[str],
              time_col: str = None,
              verbose: bool = False,
              ) -> pd.DataFrame:
    """Temporal coalition pruning method

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: pd.DataFrame
        Sequence to explain.

    entity_col: str
        Column that contains the sequence identifiers

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.

    pruning_dict: dict
        Information required for pruning algorithm

    model_features: List[str]
        In-order list of features to select and input to the model

    time_col: str
        Data column that represents the time feature in order to sort sequences
        temporally

    verbose: bool
        If process is verbose

    Returns
    -------
    pd.DataFrame

    """
    file_path = pruning_dict.get('path')
    tolerances = pruning_dict.get('tol')
    make_predictions = True
    prun_data = None
    if os.path.exists(file_path):
        necessary_entities = set(np.unique(data[entity_col].values))
        loaded_csv = pd.read_csv(file_path)
        present_entities = set(np.unique(loaded_csv[entity_col].values))
        if necessary_entities.issubset(present_entities):
            make_predictions = False
            prun_data = loaded_csv[loaded_csv[entity_col].isin(necessary_entities)]

    if make_predictions:
        if time_col is None:
            print("No time col provided, assuming dataset is ordered ascendingly by date")
        names = ["Coalition", "t (event index)", "Shapley Value", entity_col]
        # create directory
        if '/' in file_path:
            Path(file_path.rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(names)
        for uuid in np.unique(data[entity_col].values):
            seq = data[data[entity_col] == uuid]
            if time_col:
                seq = seq.sort_values(time_col)
            seq = seq[model_features]
            seq = np.expand_dims(seq.to_numpy().copy(), axis=0)

            local_pruning_data = temp_coalition_pruning(f, seq, baseline, None, ret_plot_data=True, verbose=verbose)
            local_pruning_data[entity_col] = uuid
            if file_path is not None:
                 with open(file_path, 'a', newline='') as file:
                     writer = csv.writer(file, delimiter=',')
                     writer.writerows(local_pruning_data.values)
        prun_data = pd.read_csv(file_path)
    if isinstance(tolerances, float):
        tolerances = [tolerances]

    pruning_data = []
    for uuid in np.unique(data[entity_col].values):
        uuid_data = prun_data[prun_data[entity_col] == uuid]
        pruning_data.append([uuid, -1, -(uuid_data.shape[0]/2)+1])
        for tol in tolerances:
            pruning_idx = prune_given_data(uuid_data, tol)
            pruning_data.append([uuid, tol, pruning_idx])

    return pd.DataFrame(pruning_data, columns=[entity_col, 'Tolerance', 'Pruning idx'])
