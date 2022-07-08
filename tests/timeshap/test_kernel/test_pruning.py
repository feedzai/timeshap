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
import unittest
import numpy as np

from unittest.mock import patch, MagicMock
from tests.timeshap import get_baseline, get_instance
from timeshap.explainer.kernel import TimeShapKernel

roundresult = lambda x: round(x, 8)
vectorize_round = np.vectorize(roundresult)


class TestPruning(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = get_baseline()
        self.instance = get_instance()
        self.model_mock = MagicMock()
        self.f =lambda x: self.model_mock(x)

    def test_pruning_beggining(self):
        model_answers = []
        model_answers.append(np.array([np.array([0.01820675])]))
        model_answers.append(np.array([np.array([0.7290683])]))
        model_answers.append(np.array([np.array([0.01820675]), np.array([0.7290683])]))
        self.model_mock.side_effect = model_answers
        expected_result = np.array([0, 0.71086155])

        kernel = TimeShapKernel(self.f, self.baseline, 0, "pruning")
        result = kernel.shap_values(self.instance, pruning_idx=480, **{'nsamples': 4})
        result = vectorize_round(result)
        assert (expected_result == result).all()
        assert self.model_mock.call_count == 3

    def test_pruning_middle(self):
        model_answers = []
        model_answers.append(np.array([np.array([0.01820675])]))
        model_answers.append(np.array([np.array([0.7290683])]))
        model_answers.append(np.array([np.array([0.019985398]), np.array([0.6154269])]))
        self.model_mock.side_effect = model_answers
        expected_result = np.array([0.05771002, 0.65315153])

        kernel = TimeShapKernel(self.f, self.baseline, 0, "pruning")
        result = kernel.shap_values(self.instance, pruning_idx=479, **{'nsamples': 4})
        result = vectorize_round(result)

        assert (expected_result == result).all()
        assert self.model_mock.call_count == 3

    def test_pruning_end(self):
        model_answers = []
        model_answers.append(np.array([np.array([0.01820675])]))
        model_answers.append(np.array([np.array([0.7290683])]))
        model_answers.append(np.array([np.array([0.7290683]), np.array([0.01820675])]))
        self.model_mock.side_effect = model_answers
        expected_result = np.array([0.71086155, 0])

        kernel = TimeShapKernel(self.f, self.baseline, 0, "pruning")
        result = kernel.shap_values(self.instance, pruning_idx=479, **{'nsamples': 4})
        result = vectorize_round(result)

        assert (expected_result == result).all()
        assert self.model_mock.call_count == 3
