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

# The MIT License (MIT)
#
# Copyright (c) 2018 Scott Lundberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from shap.utils._legacy import Instance, Model, Data


class TimeShapDenseData(Data):
    def __init__(self, data, mode, group_names, *args):
        self.groups = args[0] if len(args) > 0 and args[0] is not None else [np.array([i]) for i in range(len(group_names))]

        if mode in ["event", "feature", "pruning", "cell"]:
            self.weights = args[2] if len(args) > 1 else np.ones(data.shape[0])
            self.weights /= np.sum(self.weights)

            self.transposed = False
            self.group_names = group_names
            self.data = data
            self.groups_size = len(self.groups)

        else:
            raise ValueError("TimeShapDenseData - mode not supported")


def time_shap_match_instance_to_data(instance, data):
    assert isinstance(instance, Instance), "instance must be of type Instance!"

    if isinstance(data, TimeShapDenseData):
        if instance.group_display_values is None:
            instance.group_display_values = [instance.x[0, :, group[0]] if len(group) == 1 else "" for group in data.groups]
        assert len(instance.group_display_values) == len(data.groups)
        instance.groups = data.groups
    else:
        raise NotImplementedError("Type of data not supported")


def time_shap_match_model_to_data(model, data):
    assert isinstance(model, Model), "model must be of type Model!"
    data = data.data
    returns_hs = False
    try:
        out_val = model.f(data)
        if len(out_val) == 2:
            # model returns the hidden state aswell.
            # We can use this hidden state to make the algorithm more efficent
            # as we reduce the computation of all pruned events to a single hidden state
            out_val, _ = out_val
            returns_hs = True
    except:
        print("Provided model function fails when applied to the provided data set.")
        raise

    if model.out_names is None:
        if len(out_val.shape) == 1:
            model.out_names = ["output value"]
        else:
            model.out_names = ["output value " + str(i) for i in range(out_val.shape[0])]

    return out_val, returns_hs


def time_shap_convert_to_data(val, mode, pruning_idx, varying=None):
    if type(val) == np.ndarray:
        if mode == 'event':
            event_names = ["Event: {}".format(i) for i in np.arange(val.shape[1], pruning_idx, -1)]
            if pruning_idx > 0:
                event_names += ["Pruned Events"]
            return TimeShapDenseData(val, mode, event_names)
        elif mode == 'feature':
            feature_names = ["Feat: {}".format(i) for i in np.arange(val.shape[2])]
            if pruning_idx > 0:
                feature_names += ["Pruned Events"]
            return TimeShapDenseData(val, mode, feature_names)
        elif mode == 'cell':
            group_names = []
            for event_idx in varying[0]:
                for feat_idx in varying[1]:
                    group_names += ["({}, {})".format(event_idx, feat_idx)]

            used_index = 0
            special_names = []
            # check if there are pruned cells
            if pruning_idx > 0:
                special_names += ["Pruned Cells"]
                used_index += 1
                pruned_events = used_index
            else:
                pruned_events = False

            # check if there are other cells
            if not(len(varying[0]) == val.shape[1] - pruning_idx or len(varying[1]) == val.shape[2]):
                special_names += ["Other Cells"]
                used_index += 1
                all_other = used_index
            else:
                all_other = False

            if len(varying[0]) < val.shape[1] - pruning_idx:
                special_names += reversed(["Other events on feature {}".format(x) for x in varying[1]])
                used_index += 1
                other_event_rel_feat = used_index
            else:
                other_event_rel_feat = False

            if len(varying[1]) < val.shape[2]:
                special_names += reversed(["Other feats on event {}".format(x) for x in varying[0]])
                used_index += 1
                other_feat_rel_event = used_index
            else:
                other_feat_rel_event = False

            group_names += reversed(special_names)

            return TimeShapDenseData(val, mode, group_names), [other_feat_rel_event, other_event_rel_feat, all_other, pruned_events]
        elif mode == 'pruning':
            return TimeShapDenseData(val, mode, ["x", "hidden"])
        else:
            raise ValueError("`convert_to_data` - mode not supported")

    elif isinstance(val, Data):
        return val
    else:
        assert False, "Unknown type passed as data object: " + str(type(val))
