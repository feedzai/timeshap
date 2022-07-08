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


class TestCellLevel(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = get_baseline()
        self.instance = get_instance()
        self.model_mock = MagicMock()

    def test_cell_hs(self):
        self.f_hs = lambda x, y=None: self.model_mock(x, y)
        model_answers = []
        model_answers.append([np.array([[0.01820675]]), np.zeros((2,1,32))])
        model_answers.append([np.array([[0.01820675]]), np.zeros((2,1,32))])
        model_answers.append([np.array([[0.6505512]]), np.zeros((2,1,32))])
        model_answers.append([np.array([[0.7290683]]), np.zeros((2,1,32))])
        model_answers.append([np.array([[[0.0189],[0.6720],[0.0183],[0.7158],[0.0182],[0.7259],[0.0187],[0.6712],[0.0206],[0.6378],[0.0190],[0.6873],[0.3836],[0.0289],[0.0191],[0.6561],[0.0190],[0.6681],[0.0196],[0.6049],[0.0225],[0.5625],[0.0199],[0.6196],[0.4648],[0.0251],[0.0184],[0.7125],[0.0189],[0.6556],[0.0210],[0.6115],[0.0192],[0.6714],[0.4122],[0.0280],[0.0188],[0.6677],[0.0207],[0.6333],[0.0191],[0.6839],[0.3875],[0.0286],[0.0218],[0.5699],[0.0199],[0.6188],[0.4549],[0.0257],[0.0223],[0.5735],[0.5203],[0.0214],[0.4552],[0.0251],[0.0243],[0.4995],[0.0242],[0.4890],[0.4602],[0.0249],[0.0210],[0.6067],[0.0222],[0.5413],[0.5400],[0.0211],[0.0226],[0.5574],[0.0197],[0.6009],[0.4163],[0.0277],[0.4841],[0.0244],[0.5973],[0.0202],[0.0198],[0.5871],[0.0192],[0.6679],[0.0231],[0.5343],[0.5241],[0.0213],[0.4954],[0.0245],[0.0201],[0.6008],[0.5970],[0.0202],[0.0249],[0.4902],[0.0210],[0.5437],[0.0192],[0.6521],[0.5365],[0.0224]]]), np.zeros((2,1,32))])

        self.model_mock.side_effect = model_answers
        expected_result = np.array([0.03482471, 0.00926021, 0.00218324, 0.03421063, 0.05676721, 0.02843482, 0.54518073])

        kernel = TimeShapKernel(self.f_hs, self.baseline, 42, "cell", varying=([478, 479], [2, 4]))
        result = kernel.shap_values(self.instance, pruning_idx=478, nsamples=100)

        result = vectorize_round(result)
        assert (expected_result == result).all()
        assert self.model_mock.call_count == 5

    def test_cell_no_hs(self):
        self.f = lambda x: self.model_mock(x)
        model_answers = []
        model_answers.append(np.array([[0.01820675]]))
        model_answers.append(np.array([[0.7290683]]))
        model_answers.append(np.array([[0.0189], [0.6720], [0.0183], [0.7158], [0.0182], [0.7259], [0.0187], [0.6712], [0.0206], [0.6378], [0.0190], [0.6873], [0.3836], [0.0289], [0.0191], [0.6561], [0.0190], [0.6681], [0.0196], [0.6049], [0.0225], [0.5625], [0.0199], [0.6196], [0.4648], [0.0251], [0.0184], [0.7125], [0.0189], [0.6556], [0.0210], [0.6115], [0.0192], [0.6714], [0.4122], [0.0280], [0.0188], [0.6677], [0.0207], [0.6333], [0.0191], [0.6839], [0.3875], [0.0286], [0.0218], [0.5699], [0.0199], [0.6188], [0.4549], [0.0257], [0.0223], [0.5735], [0.5203], [0.0214], [0.4552], [0.0251], [0.0243], [0.4995], [0.0242], [0.4890], [0.4602], [0.0249], [0.0210], [0.6067], [0.0222], [0.5413], [0.5400], [0.0211], [0.0226], [0.5574], [0.0197], [0.6009], [0.4163], [0.0277], [0.4841], [0.0244], [0.5973], [0.0202], [0.0198], [0.5871], [0.0192], [0.6679], [0.0231], [0.5343], [0.5241], [0.0213], [0.4954], [0.0245], [0.0201], [0.6008], [0.5970], [0.0202], [0.0249], [0.4902], [0.0210], [0.5437], [0.0192], [0.6521], [0.5365], [0.0224]]))

        self.model_mock.side_effect = model_answers
        expected_result = np.array([0.03482471, 0.00926021, 0.00218324, 0.03421063, 0.05676721, 0.02843482, 0.54518073])

        kernel = TimeShapKernel(self.f, self.baseline, 42, "cell", varying=([478, 479], [2, 4]))
        result = kernel.shap_values(self.instance, pruning_idx=478, nsamples=100)

        result = vectorize_round(result)
        assert (expected_result == result).all()
        assert self.model_mock.call_count == 3

