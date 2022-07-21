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

class TestFeatureLevel(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = get_baseline()
        self.instance = np.expand_dims(get_instance().values[:, :6], axis=0)
        self.model_mock = MagicMock()

    def test_feature_hs(self):
        self.f_hs = lambda x, y=None: self.model_mock(x, y)
        model_answers = []
        model_answers.append([np.array([[0.01820675]]), np.zeros((2,1,32))])
        model_answers.append([np.array([[0.01820675]]), np.zeros((2,1,32))])
        model_answers.append([np.array([[0.6505512]]), np.zeros((2,1,32))])
        model_answers.append([np.array([[0.7290683]]), np.zeros((2,1,32))])
        model_answers.append([np.array([[[0.0184], [0.7250], [0.0187], [0.7045], [0.0190], [0.6681], [0.0186], [0.7023], [0.0189], [0.6556], [0.0193], [0.6882], [0.3836], [0.0289], [0.0190], [0.6975], [0.0193], [0.6630], [0.0189], [0.6947], [0.0192], [0.6476], [0.0197], [0.6782], [0.4030], [0.0270], [0.0197], [0.6358], [0.0192], [0.6722], [0.0196], [0.6231], [0.0204], [0.6395], [0.4473], [0.0250], [0.0195], [0.6367], [0.0199], [0.5830], [0.0206], [0.6135], [0.4690], [0.0250], [0.0195], [0.6233], [0.0200], [0.6469], [0.4336], [0.0261], [0.0205], [0.5994], [0.4854], [0.0250], [0.4864], [0.0231], [0.0216], [0.5530], [0.0199], [0.6110], [0.5690], [0.0211], [0.0209], [0.5427], [0.0204], [0.5850], [0.5498], [0.0224], [0.0203], [0.5737], [0.0199], [0.6274], [0.5343], [0.0223], [0.5344], [0.0213], [0.5039], [0.0238], [0.0197], [0.6614], [0.0222], [0.5559], [0.0201], [0.6113], [0.5735], [0.0223], [0.4664], [0.0239], [0.0215], [0.5909], [0.5411], [0.0232], [0.0211], [0.5850], [0.0205], [0.6326], [0.0202], [0.6277], [0.5028], [0.0222]]]), np.zeros((2,1,32))])

        self.model_mock.side_effect = model_answers
        expected_result = np.array([[[0.00488924,0.02023113,0.0382695,0.01906325,0.0427651,0.03395559,0.55168774]]])

        kernel = TimeShapKernel(self.f_hs, self.baseline, 42, "feature")
        result = kernel.shap_values(self.instance, pruning_idx=478, nsamples=100)

        result = vectorize_round(result)
        assert (expected_result == result).all()
        assert self.model_mock.call_count == 5

    def test_feature_no_hs(self):
        self.f = lambda x: self.model_mock(x)
        model_answers = []
        model_answers.append(np.array([[0.01820675]]))
        model_answers.append(np.array([[0.7290683]]))
        model_answers.append(np.array([[[0.0184], [0.7250], [0.0187], [0.7045], [0.0190], [0.6681], [0.0186], [0.7023], [0.0189], [0.6556], [0.0193], [0.6882], [0.3836], [0.0289], [0.0190], [0.6975], [0.0193], [0.6630], [0.0189], [0.6947], [0.0192], [0.6476], [0.0197], [0.6782], [0.4030], [0.0270], [0.0197], [0.6358], [0.0192], [0.6722], [0.0196], [0.6231], [0.0204], [0.6395], [0.4473], [0.0250], [0.0195], [0.6367], [0.0199], [0.5830], [0.0206], [0.6135], [0.4690], [0.0250], [0.0195], [0.6233], [0.0200], [0.6469], [0.4336], [0.0261], [0.0205], [0.5994], [0.4854], [0.0250], [0.4864], [0.0231], [0.0216], [0.5530], [0.0199], [0.6110], [0.5690], [0.0211], [0.0209], [0.5427], [0.0204], [0.5850], [0.5498], [0.0224], [0.0203], [0.5737], [0.0199], [0.6274], [0.5343], [0.0223], [0.5344], [0.0213], [0.5039], [0.0238], [0.0197], [0.6614], [0.0222], [0.5559], [0.0201], [0.6113], [0.5735], [0.0223], [0.4664], [0.0239], [0.0215], [0.5909], [0.5411], [0.0232], [0.0211], [0.5850], [0.0205], [0.6326], [0.0202], [0.6277], [0.5028], [0.0222]]]))

        self.model_mock.side_effect = model_answers
        expected_result = np.array([[0.00488924,0.02023113,0.0382695,0.01906325,0.0427651,0.03395559,0.55168774]])

        kernel = TimeShapKernel(self.f, self.baseline, 42, "feature")
        result = kernel.shap_values(self.instance, pruning_idx=478, nsamples=100)

        result = vectorize_round(result)
        assert (expected_result == result).all()
        assert self.model_mock.call_count == 3
