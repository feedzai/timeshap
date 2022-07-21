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
from timeshap.explainer import validate_local_input, calc_local_report
import pandas as pd
roundresult = lambda x: round(x, 4)
vectorize_round = np.vectorize(roundresult)

def get_model_answers():
    return [[np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0153,0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0153,0.7359]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0154,0.6066]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0155,0.4053]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0157,0.1781]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0157,0.1152]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0158,0.0712]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0159,0.0483]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0160,0.0319]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0160,0.0253]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0160,0.0221]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0161,0.0202]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0161,0.0187]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0161,0.0188]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0161,0.0185]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0161,0.0177]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0161,0.0169]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0161,0.0165]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0161,0.0162]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0161,0.0161]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0161,0.0161]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0161,0.0158]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0161,0.0157]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0161,0.0159]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0162,0.0156]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0165,0.0155]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.8062,0.0155]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.8141,0.0155]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.8145,0.0154]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.8150,0.0154]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.8152,0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.0590]]), np.zeros((2,1,32))],
            [np.array([[0.7067]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0153,0.8234,0.0153,0.2938,0.0153,0.7302,0.0153,0.7348,0.0153,0.8235,0.0153,0.8154,0.0153,0.8151,0.0157,0.0164,0.0153,0.8062,0.0153,0.6475,0.0155,0.8062,0.0154,0.0284,0.0162,0.0155,0.0153,0.8134,0.0153,0.8140,0.0153,0.8181,0.0153,0.7532,0.0156,0.7134,0.0152,0.8239,0.0154,0.0197,0.0153,0.8152,0.0153,0.7687,0.0155,0.6354,0.0155,0.7842,0.0160,0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.0590]]), np.zeros((2,1,32))],
            [np.array([[0.7067]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0152,0.8574,0.0151,0.8739,0.0154,0.6885,0.0152,0.7562,0.0154,0.7006,0.0153,0.4973,0.0155,0.8062,0.0153,0.0191,0.0152,0.8453,0.0507,0.0153,0.0151,0.8504,0.0153,0.7709,0.0151,0.0230,0.0151,0.8998,0.0153,0.0154,0.0154,0.0209,0.0153,0.0166,0.0153,0.7768,0.0161,0.0151,0.0153,0.5979,0.0152,0.8002,0.0152,0.8293,0.0152,0.7914,0.0153,0.0160,0.0156,0.0167]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.0590]]), np.zeros((2,1,32))],
            [np.array([[0.7067]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0152,0.7672,0.7191,0.0152,0.7598,0.0152,0.0177,0.0153,0.8149,0.0152,0.7217,0.0153,0.8152,0.0156,0.2561,0.0153,0.8156,0.0153,0.8146,0.0167,0.7165,0.0152,0.0153,0.8151,0.0153,0.8151,0.0153,0.8158,0.0152,0.7210,0.0152,0.7214,0.0153,0.5038,0.0155,0.8062,0.0153,0.5115,0.0152,0.7209,0.0153,0.4871,0.0152,0.0181,0.0162,0.0154,0.0152,0.7695,0.0153,0.5026]]), np.zeros((2,1,32))],]


def get_calc_local_answers():
    pruning_data = pd.DataFrame(
        [['Sum of contribution of events > t', 0, 0.0000],
         ['Sum of contribution of events ≤ t', 0, 0.7999],
         ['Sum of contribution of events > t', -1, 0.0397],
         ['Sum of contribution of events ≤ t', -1, 0.7603],
         ['Sum of contribution of events > t', -2, 0.1043],
         ['Sum of contribution of events ≤ t', -2, 0.6956],
         ['Sum of contribution of events > t', -3, 0.2051],
         ['Sum of contribution of events ≤ t', -3, 0.5949],
         ['Sum of contribution of events > t', -4, 0.3187],
         ['Sum of contribution of events ≤ t', -4, 0.4812],
         ['Sum of contribution of events > t', -5, 0.3502],
         ['Sum of contribution of events ≤ t', -5, 0.4497],
         ['Sum of contribution of events > t', -6, 0.3723],
         ['Sum of contribution of events ≤ t', -6, 0.4276],
         ['Sum of contribution of events > t', -7, 0.3838],
         ['Sum of contribution of events ≤ t', -7, 0.4162],
         ['Sum of contribution of events > t', -8, 0.3920],
         ['Sum of contribution of events ≤ t', -8, 0.4079],
         ['Sum of contribution of events > t', -9, 0.3953],
         ['Sum of contribution of events ≤ t', -9, 0.4046],
         ['Sum of contribution of events > t', -10, 0.3969],
         ['Sum of contribution of events ≤ t', -10, 0.4030],
         ['Sum of contribution of events > t', -11, 0.3979],
         ['Sum of contribution of events ≤ t', -11, 0.4020],
         ['Sum of contribution of events > t', -12, 0.3987],
         ['Sum of contribution of events ≤ t', -12, 0.4012],
         ['Sum of contribution of events > t', -13, 0.3986],
         ['Sum of contribution of events ≤ t', -13, 0.4013],
         ['Sum of contribution of events > t', -14, 0.3988],
         ['Sum of contribution of events ≤ t', -14, 0.4012],
         ['Sum of contribution of events > t', -15, 0.3992],
         ['Sum of contribution of events ≤ t', -15, 0.4007],
         ['Sum of contribution of events > t', -16, 0.3996],
         ['Sum of contribution of events ≤ t', -16, 0.4004],
         ['Sum of contribution of events > t', -17, 0.3998],
         ['Sum of contribution of events ≤ t', -17, 0.4002],
         ['Sum of contribution of events > t', -18, 0.3999],
         ['Sum of contribution of events ≤ t', -18, 0.4000],
         ['Sum of contribution of events > t', -19, 0.4000],
         ['Sum of contribution of events ≤ t', -19, 0.4000],
         ['Sum of contribution of events > t', -20, 0.4000],
         ['Sum of contribution of events ≤ t', -20, 0.4000],
         ['Sum of contribution of events > t', -21, 0.4001],
         ['Sum of contribution of events ≤ t', -21, 0.3998],
         ['Sum of contribution of events > t', -22, 0.4002],
         ['Sum of contribution of events ≤ t', -22, 0.3997],
         ['Sum of contribution of events > t', -23, 0.4001],
         ['Sum of contribution of events ≤ t', -23, 0.3999],
         ['Sum of contribution of events > t', -24, 0.4003],
         ['Sum of contribution of events ≤ t', -24, 0.3997],
         ['Sum of contribution of events > t', -25, 0.4005],
         ['Sum of contribution of events ≤ t', -25, 0.3995],
         ['Sum of contribution of events > t', -26, 0.7953],
         ['Sum of contribution of events ≤ t', -26, 0.0046],
         ['Sum of contribution of events > t', -27, 0.7993],
         ['Sum of contribution of events ≤ t', -27, 0.0007],
         ['Sum of contribution of events > t', -28, 0.7995],
         ['Sum of contribution of events ≤ t', -28, 0.0004],
         ['Sum of contribution of events > t', -29, 0.7997],
         ['Sum of contribution of events ≤ t', -29, 0.0002],
         ['Sum of contribution of events > t', -30, 0.7999],
         ['Sum of contribution of events ≤ t', -30, 0.0000],],
        columns=['Coalition','t (event index)','Shapley Value']
    )

    event_data = pd.DataFrame(
        [[42, 50, 'Event -1', 0.0260],
         [42, 50, 'Event -2', 0.0000],
         [42, 50, 'Event -3', 0.1538],
         [42, 50, 'Event -4', 0.0000],
         [42, 50, 'Event -5', 0.0000],
         [42, 50, 'Event -6', 0.0000],
         [42, 50, 'Event -7', 0.0000],
         [42, 50, 'Event -8', 0.0000],
         [42, 50, 'Event -9', 0.0686],
         [42, 50, 'Event -10', 0.0000],
         [42, 50, 'Event -11', 0.1205],
         [42, 50, 'Event -12', 0.0000],
         [42, 50, 'Event -13', 0.0000],
         [42, 50, 'Event -14', 0.0000],
         [42, 50, 'Event -15', 0.0000],
         [42, 50, 'Event -16', 0.0000],
         [42, 50, 'Event -17', 0.2153],
         [42, 50, 'Event -18', 0.0000],
         [42, 50, 'Event -19', 0.0000],
         [42, 50, 'Event -20', 0.0000],
         [42, 50, 'Event -21', 0.0562],
         [42, 50, 'Event -22', 0.0000],
         [42, 50, 'Event -23', 0.0000],
         [42, 50, 'Event -24', 0.1595],
         [42, 50, 'Event -25', 0.0000],
         [42, 50, 'Event -26', 0.0000],
         [42, 50, 'Pruned Events', 0.0000], ],
        columns=['Random seed','NSamples','Feature','Shapley Value']
    )

    feat_data = pd.DataFrame(
        [[42, 50, 'p_avg_rss12_normalized', -0.0248],
         [42, 50, 'p_var_rss12_normalized', 0.0218],
         [42, 50, 'p_avg_rss13_normalized', 0.0717],
         [42, 50, 'p_var_rss13_normalized', 0.1885],
         [42, 50, 'p_avg_rss23_normalized', 0.0845],
         [42, 50, 'p_var_rss23_normalized', 0.2633],
         [42, 50, 'Pruned Events', 0.1949], ],
        columns=['Random seed','NSamples','Feature','Shapley Value']
    )

    cell_data = pd.DataFrame(
        [['Other Events', 'p_var_rss23_normalized', 0.2603],
         ['Pruned Events', 'Pruned Events', 0.2445],
         ['Other Events', 'p_var_rss13_normalized', 0.162],
         ['Other Events', 'Other Features', 0.1332],
         ['Event -24', 'p_var_rss13_normalized', 0.0],
         ['Event -24', 'p_var_rss23_normalized', 0.0],
         ['Event -17', 'p_var_rss13_normalized', 0.0],
         ['Event -17', 'p_var_rss23_normalized', 0.0],
         ['Event -24', 'Other Features', 0.0],
         ['Event -17', 'Other Features', 0.0],],
        columns=['Event','Feature','Shapley Value']
    )
    return pruning_data, event_data, feat_data, cell_data


class TestValidateLocalInputNumpy(unittest.TestCase):
    def setUp(self) -> None:
        self.f = lambda x: MagicMock()(x)
        self.model_features = ['A', 'B', 'C']
        self.plot_feats = {'A': "a", 'B': "b", 'C': "c", }
        self.data = np.ones((1, 480, 3))
        self.baseline = np.ones((1, 3))

    def test_success(self):
        pruning_dict = {'path': 'a/a.csv', 'tol': 0.05}
        event_dict = {'path': 'a/a.csv', 'rs': 42, 'nsamples': 50}
        feature_dict = {'path': 'a/a.csv', 'rs': 42, 'nsamples': 50, 'top_feats': 3,
                        'feature_names': self.model_features, 'plot_features': self.plot_feats}
        cell_dict = {'rs': 42, 'nsamples': 32000, 'top_x_feats': 2, 'top_x_events': 2}
        validate_local_input(self.f, self.data, pruning_dict, event_dict, feature_dict,
                             cell_dict, self.baseline)


class TestCalcLocalReport(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = get_baseline()
        self.instance = np.expand_dims(get_instance().values[:, :6], axis=0)
        self.model_mock = MagicMock()
        self.model_features = ['p_avg_rss12_normalized', 'p_var_rss12_normalized',
               'p_avg_rss13_normalized', 'p_var_rss13_normalized',
               'p_avg_rss23_normalized', 'p_var_rss23_normalized']
        self.pruning_dict = {'tol': 0.025}
        self.event_dict = {'rs': 42, 'nsamples': 50}
        self.feature_dict = {'rs': 42, 'nsamples': 50, 'feature_names': self.model_features}
        self.cell_dict = {'rs': 42, 'nsamples': 50, 'top_x_feats': 2, 'top_x_events': 2}

    def test_calc_local_report(self):
        self.f_hs = lambda x, y=None: self.model_mock(x, y)

        self.model_mock.side_effect = get_model_answers()

        prun_data, event_data, feature_data, cell_data = \
            calc_local_report(self.f_hs, get_instance().iloc[-30:], self.pruning_dict,
                              self.event_dict, self.feature_dict, self.cell_dict,
                              self.baseline, self.model_features, entity_col='sequence_id',time_col='timestamp')

        prun_data['Shapley Value'] = prun_data['Shapley Value'].apply(roundresult)
        event_data['Shapley Value'] = event_data['Shapley Value'].apply(roundresult)
        feature_data['Shapley Value'] = feature_data['Shapley Value'].apply(roundresult)
        cell_data['Shapley Value'] = cell_data['Shapley Value'].apply(roundresult)
        prun_answer, event_answer, feature_answer, cell_answer = get_calc_local_answers()
        assert prun_answer.equals(prun_data.reset_index(drop=True))
        assert event_answer.equals(event_data.reset_index(drop=True))
        assert feature_answer.equals(feature_data.reset_index(drop=True))
        assert cell_answer.equals(cell_data.reset_index(drop=True))


    def test_calc_local_report_pd_vs_np(self):
        self.f_hs = lambda x, y=None: self.model_mock(x, y)

        self.model_mock.side_effect = get_model_answers() + get_model_answers()

        prun_data_pd, event_data_pd, feature_data_pd, cell_data_pd = \
            calc_local_report(self.f_hs, get_instance().iloc[-30:], self.pruning_dict,
                              self.event_dict, self.feature_dict, self.cell_dict,
                              self.baseline, self.model_features, entity_col='sequence_id',time_col='timestamp')

        prun_data_np, event_data_np, feature_data_np, cell_data_np = \
            calc_local_report(self.f_hs, np.expand_dims(get_instance().values[-30:, :6], axis=0).astype(float), self.pruning_dict,
                              self.event_dict, self.feature_dict, self.cell_dict,
                              self.baseline,)

        assert prun_data_pd.equals(prun_data_np)
        assert event_data_pd.equals(event_data_np)
        assert feature_data_pd.equals(feature_data_np)
        assert cell_data_pd.equals(cell_data_np)
