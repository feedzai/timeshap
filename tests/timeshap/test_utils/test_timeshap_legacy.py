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
from timeshap import utils
from unittest.mock import MagicMock, patch
from shap.utils._legacy import convert_to_instance, convert_to_model

from tests.timeshap import get_sample_test_instance
from timeshap.utils.timeshap_legacy import TimeShapDenseData, time_shap_convert_to_data, \
    time_shap_match_instance_to_data, time_shap_match_model_to_data

class TestTimeShapDenseData(unittest.TestCase):
    def setUp(self) -> None:
        self.data = get_sample_test_instance()

    def test_with_no_hs(self):
        modes = ["pruning", "event", "feature", "cell"]
        group_names = [['x', 'hidden'],
                       ['Event: 10', 'Event: 9', 'Event: 8', 'Event: 7', 'Event: 6', 'Pruned Events'],
                       ['Feat: 0', 'Feat: 1', 'Feat: 2', 'Feat: 3', 'Feat: 4', 'Feat: 5', 'Pruned Events'],
                       ['(9, 2)', 'Other feats on event 4',
                        'Other events on feature 2', 'Other Cells', 'Pruned Cells']
                       ]
        for mode, group_names in zip(modes, group_names):
            result = TimeShapDenseData(self.data, mode, group_names)
            assert (result.data == self.data).all()
            assert result.group_names == group_names
            assert result.groups == [np.array([i]) for i in range(len(group_names))]
            assert result.groups_size == len(group_names)

            weights = np.ones(self.data.shape[0])
            weights /= np.sum(weights)
            assert result.weights == weights


class TestTimeShapMatchModelToData(unittest.TestCase):
    def setUp(self) -> None:
        self.data = get_sample_test_instance()

    def test_with_no_hs(self):
        model_mock = MagicMock(side_effect=np.array([np.array([0.9])]))
        f = lambda x: model_mock(x)
        converted_model = convert_to_model(f)
        out_val, return_hs = time_shap_match_model_to_data(converted_model, self.data)

        assert out_val == np.array([0.9])
        assert not return_hs
        assert model_mock.call_count == 1
        assert (model_mock.call_args_list[0][0][0].obj == self.data).all()

    def test_with_hs(self):
        model_mock = MagicMock(side_effect=[(np.array([np.array([0.9])]), np.zeros((2,1,32))),])
        f_hs = lambda x, y=None: model_mock(x, y)
        converted_model = convert_to_model(f_hs)
        out_val, return_hs = time_shap_match_model_to_data(converted_model, self.data)

        assert out_val == np.array([0.9])
        assert return_hs
        assert model_mock.call_count == 1
        assert (model_mock.call_args_list[0][0][0].obj == self.data).all()

# class TestTimeShapMatchIntanceToData(unittest.TestCase):
#     def setUp(self) -> None:
#
#     def test_time_shap_match_instance_to_data(self):
#


class TestTimeShapConvertToData(unittest.TestCase):
    def setUp(self) -> None:
        self.data = get_sample_test_instance()

    @patch('timeshap.utils.timeshap_legacy.TimeShapDenseData')
    def test_pruning(self, mock_DenseData):
        mode = "pruning"
        time_shap_convert_to_data(self.data, mode, 5)
        assert mock_DenseData.call_count == 1
        assert (mock_DenseData.call_args_list[0][0][0] == self.data).all()
        assert mock_DenseData.call_args_list[0][0][1] == 'pruning'
        assert mock_DenseData.call_args_list[0][0][2] == ['x', 'hidden']

    @patch('timeshap.utils.timeshap_legacy.TimeShapDenseData')
    def test_event(self, mock_DenseData):
        mode = "event"
        time_shap_convert_to_data(self.data, mode, 5)
        assert mock_DenseData.call_count == 1
        assert (mock_DenseData.call_args_list[0][0][0] == self.data).all()
        assert mock_DenseData.call_args_list[0][0][1] == mode
        assert mock_DenseData.call_args_list[0][0][2] == ['Event: 10', 'Event: 9', 'Event: 8', 'Event: 7', 'Event: 6', 'Pruned Events']

    @patch('timeshap.utils.timeshap_legacy.TimeShapDenseData')
    def test_feature(self, mock_DenseData):
        mode = "feature"
        time_shap_convert_to_data(self.data, mode, 5)
        assert mock_DenseData.call_count == 1
        assert (mock_DenseData.call_args_list[0][0][0] == self.data).all()
        assert mock_DenseData.call_args_list[0][0][1] == mode
        assert mock_DenseData.call_args_list[0][0][2] == ['Feat: 0', 'Feat: 1', 'Feat: 2', 'Feat: 3', 'Feat: 4', 'Feat: 5', 'Pruned Events']

    @patch('timeshap.utils.timeshap_legacy.TimeShapDenseData')
    def test_cell(self, mock_DenseData):
        mode = "cell"
        time_shap_convert_to_data(self.data, mode, 5, varying=([9, 7], [2, 4]))
        assert mock_DenseData.call_count == 1
        assert (mock_DenseData.call_args_list[0][0][0] == self.data).all()
        assert mock_DenseData.call_args_list[0][0][1] == mode
        assert mock_DenseData.call_args_list[0][0][2] == ['(9, 2)', '(9, 4)', '(7, 2)', '(7, 4)',
                'Other feats on event 9', 'Other feats on event 7', 'Other events on feature 2',
                'Other events on feature 4', 'Other Cells', 'Pruned Cells']

