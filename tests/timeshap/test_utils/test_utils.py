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
import pandas as pd
import numpy as np
from timeshap import utils
from unittest.mock import MagicMock


class TestCalcAvgSequence(unittest.TestCase):
    def setUp(self) -> None:
        self.dummy_instance = np.array([[[1, 1000, 0, 0], [1, 4, 0, 1]],
                                        [[100, 100, 0, 1], [2, 8, 2, 4]],
                                        [[10, 10, 20, 1], [10, 100, 0, 1]],]
                                       )
        self.numerical_feats = ['num_dummy1', 'num_dummy2']
        self.categorical_feats = ['cat_dummy1', 'cat_dummy2']
        self.expected_result = np.array([[10, 100, 0, 1], [2, 8, 0, 1]])

    def test_calc_avg_sequence_w_names(self):
        model_features = self.numerical_feats + self.categorical_feats
        result = utils.calc_avg_sequence(self.dummy_instance,
                                         numerical_feats=self.numerical_feats,
                                         categorical_feats=self.categorical_feats,
                                         model_features=model_features
                                         )
        assert (self.expected_result == result).all()

    def test_calc_avg_sequence_w_indexes(self):
        numerical_indeces = [0, 1]
        cat_indeces = [2, 3]
        result = utils.calc_avg_sequence(self.dummy_instance,
                                         numerical_feats=numerical_indeces,
                                         categorical_feats=cat_indeces
                                         )
        assert (self.expected_result == result).all()


class TestCalcAvgEvent(unittest.TestCase):
    def setUp(self) -> None:
        columns = ['timestamp', 'all_id', 'activity',
       'p_avg_rss12_normalized', 'p_var_rss12_normalized',
       'p_avg_rss13_normalized', 'p_var_rss13_normalized',
       'p_avg_rss23_normalized', 'p_var_rss23_normalized', 'dummy_cat', 'label']

        values = \
        [[0, 'sitting_11', 'sitting', -0.3641, -0.3790,-0.6222, 0.7142, 0.5207, -0.3340, 1, 0],
        [250, 'sitting_11', 'sitting', -0.2938, -0.1854, 0.3706, -0.4574, 0.1293, 0.3872, 1, 0],
        [500, 'sitting_11', 'sitting', -0.3641,  -0.1623, 0.3226, -0.4808, -0.0174, -0.2871, 1, 0],
        [750, 'sitting_11', 'sitting', -0.3507,  -0.1854, -0.1728, -0.6917, -0.7513, 0.6335, 1, 0],
        [1000, 'sitting_11', 'sitting', -0.2386,  -0.4343, 0.4666, 0.1284, -0.0663, 0.8504, 1, 0],
        [1250, 'sitting_11', 'sitting', -0.2805, -0.1623, 1.4268, -0.9670, 0.2761, 0.2875, 1, 0],
        [1500, 'sitting_11', 'sitting', -0.2805, -0.3790, 0.9467, -0.0883, 0.4718, -0.2871, 1, 0],
        [1750, 'sitting_11', 'sitting', -0.1968, -0.3790, 0.6586, -0.2523, 0.8143, 0.4165, 2, 0],
        [2000, 'sitting_11', 'sitting', -0.0714,  -0.7617, 0.3706, -0.4574, 1.1078, -0.4630, 2, 0],
        [2250, 'sitting_11', 'sitting', -0.1132,  -0.5635, 0.46663, -0.9670, 0.8143, -0.7563, 3, 0],]

        self.dataset = pd.DataFrame(values, columns=columns)
        self.numerical_feats = ['p_avg_rss12_normalized', 'p_var_rss12_normalized',
                                'p_avg_rss13_normalized', 'p_var_rss13_normalized',
                                'p_avg_rss23_normalized', 'p_var_rss23_normalized']
        self.categorical_feats = ['dummy_cat']

    def test_calc_avg_event(self):
        expected_columns = ['p_avg_rss12_normalized', 'p_var_rss12_normalized',
               'p_avg_rss13_normalized', 'p_var_rss13_normalized',
               'p_avg_rss23_normalized', 'p_var_rss23_normalized', 'dummy_cat']
        expected_values = [[-2.8050e-01, -3.7900e-01, 4.1860e-01, -4.5740e-01, 3.74e-01, 2.0000e-04, 1]]
        expected_result = pd.DataFrame(expected_values, columns=expected_columns)
        result = utils.calc_avg_event(self.dataset, self.numerical_feats, self.categorical_feats).apply(lambda x: round(x, 4) )
        assert result.equals(expected_result)


class TestGetAvgScoreWithAvgEvent(unittest.TestCase):
    def setUp(self) -> None:
        expected_columns = ['p_avg_rss12_normalized', 'p_var_rss12_normalized',
               'p_avg_rss13_normalized', 'p_var_rss13_normalized',
               'p_avg_rss23_normalized', 'p_var_rss23_normalized', 'dummy_cat']
        expected_values = [[-2.8050e-01, -3.7900e-01, 4.1860e-01, -4.5740e-01, 3.74e-01, 2.0000e-04, 1]]
        self.average_event = pd.DataFrame(expected_values, columns=expected_columns)

    def test_with_no_hs(self):
        top = 5
        model_mock = MagicMock(side_effect=[[0.9], [0.8], [0.7], [0.6], [0.5]])
        f = lambda x: model_mock(x)
        result = utils.get_avg_score_with_avg_event(f, self.average_event, top)

        assert model_mock.call_count == 5
        for x in range(0, 5):
            assert model_mock.call_args_list[x][0][0].shape == (1,x+1,7)

        assert result == {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.5}

    def test_with_hs(self):
        top = 5
        model_mock = MagicMock(side_effect=[([0.9], np.zeros((2,1,32))), ([0.8], np.zeros((2,1,32))),
                                            ([0.7], np.zeros((2,1,32))), ([0.6], np.zeros((2,1,32))),
                                            ([0.5], np.zeros((2,1,32)))])
        f_hs = lambda x, y=None: model_mock(x, y)
        result = utils.get_avg_score_with_avg_event(f_hs, self.average_event, top)

        assert model_mock.call_count == 5
        for x in range(0, 5):
            assert model_mock.call_args_list[x][0][0].shape == (1, 1, 7)
            if x == 0:
                assert model_mock.call_args_list[x][0][1] == None
            else:
                assert (model_mock.call_args_list[x][0][1] == np.zeros((2,1,32))).all()

        assert result == {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.5}


class TestGetAvgScoreWithAvgSequence(unittest.TestCase):
    def setUp(self) -> None:
        self.average_event = np.ones((1,480, 6))

    def test_with_no_hs(self):
        model_mock = MagicMock(side_effect=[[0.9]])
        f = lambda x: model_mock(x)
        result = utils.get_score_of_avg_sequence(f, self.average_event)

        assert model_mock.call_count == 1
        assert result == 0.9

    def test_with_hs(self):
        model_mock = MagicMock(side_effect=[([0.9], np.zeros((2, 1, 32)))])
        f_hs = lambda x, y=None: model_mock(x, y)
        result = utils.get_score_of_avg_sequence(f_hs, self.average_event)

        assert model_mock.call_count == 1
        assert result == 0.9

