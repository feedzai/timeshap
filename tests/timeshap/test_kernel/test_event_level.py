#  Copyright 2022 Feedzai
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#vectorize_round
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


class TestEventLevel(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = get_baseline()
        self.instance = get_instance()
        self.model_mock = MagicMock()

    def test_event_hs(self):
        self.f_hs = lambda x, y=None: self.model_mock(x, y)
        model_answers = []
        model_answers.append([np.array([[0.01820675]]), np.zeros((2,1,32))])
        model_answers.append([np.array([[0.01820675]]), np.zeros((2,1,32))])
        model_answers.append([np.array([[0.6505512]]), np.zeros((2,1,32))])
        model_answers.append([np.array([[0.7290683]]), np.zeros((2,1,32))])
        model_answers.append([np.array([[0.0199854], [0.6154269], [0.02305947], [0.5342902], [0.3835847], [0.02890363]]), np.zeros((2,1,32))])

        self.model_mock.side_effect = model_answers
        expected_result = np.array([0.06456496, 0.10667034, 0.53962625])

        kernel = TimeShapKernel(self.f_hs, self.baseline, 42, "event")
        result = kernel.shap_values(self.instance, pruning_idx=478, nsamples=100)

        result = vectorize_round(result)
        assert (expected_result == result).all()
        assert self.model_mock.call_count == 5

    def test_event_no_hs(self):
        self.f = lambda x: self.model_mock(x)
        model_answers = []
        model_answers.append(np.array([[0.01820675]]))
        model_answers.append(np.array([[0.7290683]]))
        model_answers.append(np.array([[0.0199854], [0.6154269], [0.02305947], [0.5342902], [0.3835847], [0.02890363]]))

        self.model_mock.side_effect = model_answers
        expected_result = np.array([0.06456496, 0.10667034, 0.53962625])

        kernel = TimeShapKernel(self.f, self.baseline, 42, "event")
        result = kernel.shap_values(self.instance, pruning_idx=478, nsamples=100)

        result = vectorize_round(result)
        assert (expected_result == result).all()
        assert self.model_mock.call_count == 3
