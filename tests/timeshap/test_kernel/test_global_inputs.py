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
"""Global explanations on inputs that timeshap 1.0.4 failed on."""
import copy

import numpy as np
import pandas as pd
import pytest

from timeshap.explainer import (calc_global_explanations, event_explain_all,
                                feat_explain_all, prune_all)

BASELINE = np.zeros((1, 4))


def model(x):
    # only the last three events change the score, so older events can be pruned
    return x[:, -3:, :].sum(axis=(1, 2))[:, None] / 10


@pytest.fixture
def sequences():
    return np.random.default_rng(0).normal(size=(2, 30, 4))


@pytest.mark.parametrize("explain_all", [event_explain_all, feat_explain_all])
def test_pruning_on_the_fly_matches_precomputed_pruning(sequences, explain_all):
    explanation_dict = {"rs": 1, "nsamples": 100, "tol": [0.05]}
    on_the_fly = explain_all(model, sequences, explanation_dict, None, BASELINE)
    precomputed = explain_all(model, sequences, explanation_dict,
                              prune_all(model, sequences, {"tol": 0.05}, BASELINE), BASELINE)
    pd.testing.assert_frame_equal(on_the_fly, precomputed)


def test_event_explanations_group_unused_events(sequences):
    events = event_explain_all(model, sequences, {"rs": 1, "nsamples": 100, "tol": [0.05]}, None, BASELINE)
    for _, entity_events in events.groupby("Entity"):
        assert "Pruned Events" in set(entity_events["Event"])
        assert len(entity_events) < sequences.shape[1]


def test_global_explanations_without_entity_column(sequences):
    pruning, events, features = calc_global_explanations(
        model, sequences, {"tol": 0.05}, {"rs": 1, "nsamples": 100}, {"rs": 1, "nsamples": 100}, BASELINE)
    # sequences are identified by their position in the data
    assert set(pruning["Entity"]) == {0, 1}
    assert set(events["Entity"]) == {0, 1}
    assert set(features["Entity"]) == {0, 1}


def test_global_explanations_do_not_change_the_given_dicts(sequences):
    pruning_dict = {"tol": 0.05}
    event_dict = {"rs": 1, "nsamples": 100}
    feature_dict = {"rs": 1, "nsamples": 100}
    given = copy.deepcopy((pruning_dict, event_dict, feature_dict))
    calc_global_explanations(model, sequences, pruning_dict, event_dict, feature_dict, BASELINE)
    assert (pruning_dict, event_dict, feature_dict) == given


def test_pruning_csv_is_utf8(sequences, tmp_path):
    # the coalition labels contain "≤", which the Windows default encoding cannot write
    path = tmp_path / "pruning.csv"
    prune_all(model, sequences, {"tol": 0.05, "path": str(path)}, BASELINE)
    assert "≤" in path.read_text(encoding="utf-8")
