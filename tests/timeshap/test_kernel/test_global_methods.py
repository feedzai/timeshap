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
from timeshap.explainer import validate_global_input, calc_global_explanations
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
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153, 0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153, 0.7359]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0154, 0.6066]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0155, 0.4053]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0157, 0.1781]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0157, 0.1152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0158, 0.0712]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0159, 0.0483]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0160, 0.0319]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0160, 0.0253]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0160, 0.0221]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0161, 0.0202]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0161, 0.0187]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0161, 0.0188]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0161, 0.0185]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0161, 0.0177]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0161, 0.0169]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0161, 0.0165]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0161, 0.0162]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0161, 0.0161]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0161, 0.0161]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0161, 0.0158]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0161, 0.0157]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0161, 0.0159]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0162, 0.0156]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0165, 0.0155]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.8062, 0.0155]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.8141, 0.0155]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.8145, 0.0154]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.8150, 0.0154]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152, 0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.0590]]), np.zeros((2,1,32))],
            [np.array([[0.7067]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0153,0.8234,0.0153,0.2938,0.0153,0.7302,0.0153,0.7348,0.0153,0.8235,0.0153,0.8154,0.0153,0.8151,0.0157,0.0164,0.0153,0.8062,0.0153,0.6475,0.0155,0.8062,0.0154,0.0284,0.0162,0.0155,0.0153,0.8134,0.0153,0.8140,0.0153,0.8181,0.0153,0.7532,0.0156,0.7134,0.0152,0.8239,0.0154,0.0197,0.0153,0.8152,0.0153,0.7687,0.0155,0.6354,0.0155,0.7842,0.0160,0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.0590]]), np.zeros((2, 1, 32))],
            [np.array([[0.7067]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153, 0.8234, 0.0153, 0.2938, 0.0153, 0.7302, 0.0153,0.7348, 0.0153, 0.8235, 0.0153, 0.8154, 0.0153, 0.8151,0.0157, 0.0164, 0.0153, 0.8062, 0.0153, 0.6475, 0.0155,0.8062, 0.0154, 0.0284, 0.0162, 0.0155, 0.0153, 0.8134,0.0153, 0.8140, 0.0153, 0.8181, 0.0153, 0.7532, 0.0156,0.7134, 0.0152, 0.8239, 0.0154, 0.0197, 0.0153, 0.8152,0.0153, 0.7687, 0.0155, 0.6354, 0.0155, 0.7842, 0.0160,0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.0153]]), np.zeros((2,1,32))],
            [np.array([[0.0590]]), np.zeros((2,1,32))],
            [np.array([[0.7067]]), np.zeros((2,1,32))],
            [np.array([[0.8152]]), np.zeros((2,1,32))],
            [np.array([[0.0152,0.8574,0.0151,0.8739,0.0154,0.6885,0.0152,0.7562,0.0154,0.7006,0.0153,0.4973,0.0155,0.8062,0.0153,0.0191,0.0152,0.8453,0.0507,0.0153,0.0151,0.8504,0.0153,0.7709,0.0151,0.0230,0.0151,0.8998,0.0153,0.0154,0.0154,0.0209,0.0153,0.0166,0.0153,0.7768,0.0161,0.0151,0.0153,0.5979,0.0152,0.8002,0.0152,0.8293,0.0152,0.7914,0.0153,0.0160,0.0156,0.0167]]), np.zeros((2,1,32))],
            [np.array([[0.0153]]), np.zeros((2, 1, 32))],
            [np.array([[0.0590]]), np.zeros((2, 1, 32))],
            [np.array([[0.7067]]), np.zeros((2, 1, 32))],
            [np.array([[0.8152]]), np.zeros((2, 1, 32))],
            [np.array([[0.0152, 0.8574, 0.0151, 0.8739, 0.0154, 0.6885, 0.0152, 0.7562, 0.0154, 0.7006, 0.0153, 0.4973, 0.0155, 0.8062, 0.0153, 0.0191, 0.0152, 0.8453, 0.0507, 0.0153, 0.0151, 0.8504, 0.0153, 0.7709, 0.0151, 0.0230, 0.0151, 0.8998, 0.0153, 0.0154, 0.0154, 0.0209, 0.0153, 0.0166, 0.0153, 0.7768, 0.0161, 0.0151, 0.0153, 0.5979, 0.0152, 0.8002, 0.0152, 0.8293, 0.0152, 0.7914, 0.0153, 0.0160, 0.0156, 0.0167]]), np.zeros((2, 1, 32))],
            ]


def get_calc_global_answers():
    pruning_data = pd.DataFrame(
        [['other', -1.0000, -30.0000],
         ['other', 0.0250, -26.0000],
         ['sequence_id', -1.0000, -30.0000],
         ['sequence_id', 0.0250, -26.0000]],
        columns=['Entity','Tolerance','Pruning idx']
    )

    event_data = pd.DataFrame(
        [[42, 50, 'Event -1', 0.0260, 0, 'sequence_id', 0.0250],
         [42, 50, 'Event -2', 0.0000, -1, 'sequence_id', 0.0250],
         [42, 50, 'Event -3', 0.1538, -2, 'sequence_id', 0.0250],
         [42, 50, 'Event -4', 0.0000, -3, 'sequence_id', 0.0250],
         [42, 50, 'Event -5', 0.0000, -4, 'sequence_id', 0.0250],
         [42, 50, 'Event -6', 0.0000, -5, 'sequence_id', 0.0250],
         [42, 50, 'Event -7', 0.0000, -6, 'sequence_id', 0.0250],
         [42, 50, 'Event -8', 0.0000, -7, 'sequence_id', 0.0250],
         [42, 50, 'Event -9', 0.0686, -8, 'sequence_id', 0.0250],
         [42, 50, 'Event -10', 0.0000, -9, 'sequence_id', 0.0250],
         [42, 50, 'Event -11', 0.1205, -10, 'sequence_id', 0.0250],
         [42, 50, 'Event -12', 0.0000, -11, 'sequence_id', 0.0250],
         [42, 50, 'Event -13', 0.0000, -12, 'sequence_id', 0.0250],
         [42, 50, 'Event -14', 0.0000, -13, 'sequence_id', 0.0250],
         [42, 50, 'Event -15', 0.0000, -14, 'sequence_id', 0.0250],
         [42, 50, 'Event -16', 0.0000, -15, 'sequence_id', 0.0250],
         [42, 50, 'Event -17', 0.2153, -16, 'sequence_id', 0.0250],
         [42, 50, 'Event -18', 0.0000, -17, 'sequence_id', 0.0250],
         [42, 50, 'Event -19', 0.0000, -18, 'sequence_id', 0.0250],
         [42, 50, 'Event -20', 0.0000, -19, 'sequence_id', 0.0250],
         [42, 50, 'Event -21', 0.0562, -20, 'sequence_id', 0.0250],
         [42, 50, 'Event -22', 0.0000, -21, 'sequence_id', 0.0250],
         [42, 50, 'Event -23', 0.0000, -22, 'sequence_id', 0.0250],
         [42, 50, 'Event -24', 0.1595, -23, 'sequence_id', 0.0250],
         [42, 50, 'Event -25', 0.0000, -24, 'sequence_id', 0.0250],
         [42, 50, 'Event -26', 0.0000, -25, 'sequence_id', 0.0250],
         [42, 50, 'Pruned Events', 0.0000, 1, 'sequence_id', 0.0250],
         [42, 50, 'Event -1', 0.0260, 0, 'other', 0.0250],
         [42, 50, 'Event -2', 0.0000, -1, 'other', 0.0250],
         [42, 50, 'Event -3', 0.1538, -2, 'other', 0.0250],
         [42, 50, 'Event -4', 0.0000, -3, 'other', 0.0250],
         [42, 50, 'Event -5', 0.0000, -4, 'other', 0.0250],
         [42, 50, 'Event -6', 0.0000, -5, 'other', 0.0250],
         [42, 50, 'Event -7', 0.0000, -6, 'other', 0.0250],
         [42, 50, 'Event -8', 0.0000, -7, 'other', 0.0250],
         [42, 50, 'Event -9', 0.0686, -8, 'other', 0.0250],
         [42, 50, 'Event -10', 0.0000, -9, 'other', 0.0250],
         [42, 50, 'Event -11', 0.1205, -10, 'other', 0.0250],
         [42, 50, 'Event -12', 0.0000, -11, 'other', 0.0250],
         [42, 50, 'Event -13', 0.0000, -12, 'other', 0.0250],
         [42, 50, 'Event -14', 0.0000, -13, 'other', 0.0250],
         [42, 50, 'Event -15', 0.0000, -14, 'other', 0.0250],
         [42, 50, 'Event -16', 0.0000, -15, 'other', 0.0250],
         [42, 50, 'Event -17', 0.2153, -16, 'other', 0.0250],
         [42, 50, 'Event -18', 0.0000, -17, 'other', 0.0250],
         [42, 50, 'Event -19', 0.0000, -18, 'other', 0.0250],
         [42, 50, 'Event -20', 0.0000, -19, 'other', 0.0250],
         [42, 50, 'Event -21', 0.0562, -20, 'other', 0.0250],
         [42, 50, 'Event -22', 0.0000, -21, 'other', 0.0250],
         [42, 50, 'Event -23', 0.0000, -22, 'other', 0.0250],
         [42, 50, 'Event -24', 0.1595, -23, 'other', 0.0250],
         [42, 50, 'Event -25', 0.0000, -24, 'other', 0.0250],
         [42, 50, 'Event -26', 0.0000, -25, 'other', 0.0250],
         [42, 50, 'Pruned Events', 0.0000, 1, 'other', 0.0250]],
        columns=['Random Seed','NSamples','Event','Shapley Value','t (event index)','Entity','Tolerance']
    )

    feat_data = pd.DataFrame(
        [[42, 50, 'p_avg_rss12_normalized', -0.0248, 'sequence_id', 0.0250],
         [42, 50, 'p_var_rss12_normalized', 0.0218, 'sequence_id', 0.0250],
         [42, 50, 'p_avg_rss13_normalized', 0.0717, 'sequence_id', 0.0250],
         [42, 50, 'p_var_rss13_normalized', 0.1885, 'sequence_id', 0.0250],
         [42, 50, 'p_avg_rss23_normalized', 0.0845, 'sequence_id', 0.0250],
         [42, 50, 'p_var_rss23_normalized', 0.2633, 'sequence_id', 0.0250],
         [42, 50, 'Pruned Events', 0.1949, 'sequence_id', 0.0250],
         [42, 50, 'p_avg_rss12_normalized', -0.0248, 'other', 0.0250],
         [42, 50, 'p_var_rss12_normalized', 0.0218, 'other', 0.0250],
         [42, 50, 'p_avg_rss13_normalized', 0.0717, 'other', 0.0250],
         [42, 50, 'p_var_rss13_normalized', 0.1885, 'other', 0.0250],
         [42, 50, 'p_avg_rss23_normalized', 0.0845, 'other', 0.0250],
         [42, 50, 'p_var_rss23_normalized', 0.2633, 'other', 0.0250],
         [42, 50, 'Pruned Events', 0.1949, 'other', 0.0250],],
        columns=['Random Seed','NSamples','Feature','Shapley Value','Entity','Tolerance']
    )

    return pruning_data, event_data, feat_data


class TestValidateGlobalInputNumpy(unittest.TestCase):
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
        validate_global_input(self.f, self.data, pruning_dict, event_dict, feature_dict, self.baseline)


class TestCalcGlobalReport(unittest.TestCase):
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

    def test_calc_global_report(self):
        self.f_hs = lambda x, y=None: self.model_mock(x, y)

        self.model_mock.side_effect = get_model_answers()

        other_instance = get_instance().iloc[-30:]
        other_instance['sequence_id'] = 'other'
        data = pd.concat((get_instance().iloc[-30:], other_instance))

        prun_data, event_data, feature_data = \
            calc_global_explanations(self.f_hs, data, self.pruning_dict,
                              self.event_dict, self.feature_dict, self.baseline,
                              self.model_features, entity_col='sequence_id',time_col='timestamp')

        event_data['Shapley Value'] = event_data['Shapley Value'].apply(roundresult)
        feature_data['Shapley Value'] = feature_data['Shapley Value'].apply(roundresult)
        prun_answer, event_answer, feature_answer = get_calc_global_answers()
        assert prun_answer.equals(prun_data.reset_index(drop=True))
        assert event_answer.equals(event_data.reset_index(drop=True))
        assert feature_answer.equals(feature_data.reset_index(drop=True))

    def test_calc_local_report_pd_vs_np(self):
        self.f_hs = lambda x, y=None: self.model_mock(x, y)

        self.model_mock.side_effect = get_model_answers() + get_model_answers()

        other_instance = get_instance().iloc[-30:]
        other_instance['sequence_id'] = 'other'
        data = pd.concat((get_instance().iloc[-30:], other_instance))

        prun_data_pd, event_data_pd, feature_data_pd = \
            calc_global_explanations(self.f_hs, data, self.pruning_dict,
                                     self.event_dict, self.feature_dict,
                                     self.baseline,
                                     self.model_features,
                                     entity_col='sequence_id',
                                     time_col='timestamp')

        def pandas_to_numpy(df, group_by_feat, timestamp_Feat):
            sequence_length = len(df[timestamp_Feat].unique())
            data_tensor = np.zeros((len(df[group_by_feat].unique()), sequence_length, len(df.columns)))
            for i, name in enumerate(df[group_by_feat].unique()):
                name_data = df[df[group_by_feat] == name]
                sorted_data = name_data.sort_values(timestamp_Feat)
                try:
                    data_tensor[i, :, :] = sorted_data.values
                except ValueError:
                    data_tensor = data_tensor.astype(np.object)
                    data_tensor[i, :, :] = sorted_data.values

            return data_tensor

        data_np = pandas_to_numpy(data, 'sequence_id', 'timestamp')
        prun_data_np, event_data_np, feature_data_np = \
            calc_global_explanations(self.f_hs, data_np,
                                     self.pruning_dict, self.event_dict,
                                     self.feature_dict, self.baseline,
                                     [0, 1, 2, 3, 4, 5], entity_col=6
                                     )

        assert prun_data_pd.equals(prun_data_np)
        assert event_data_pd.equals(event_data_np)
        assert feature_data_pd.equals(feature_data_np)
