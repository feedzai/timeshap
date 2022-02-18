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
        print(
            "Provided model function fails when applied to the provided data set.")
        raise

    if model.out_names is None:
        if len(out_val.shape) == 1:
            model.out_names = ["output value"]
        else:
            model.out_names = ["output value " + str(i) for i in
                               range(out_val.shape[0])]

    return out_val, returns_hs


def time_shap_convert_to_data(val, mode, pruning_idx, varying=None):
    if type(val) == np.ndarray:
        if mode == 'event':
            event_names = ["Event: {}".format(i) for i in np.arange(val.shape[1], pruning_idx, -1)]
            event_names += ["Pruned Events"]
            return TimeShapDenseData(val, mode, event_names)
        elif mode == 'feature':
            event_names = ["Feat: {}".format(i) for i in np.arange(val.shape[2])]
            event_names += ["Pruned Events"]
            return TimeShapDenseData(val, mode, event_names)
        elif mode == 'cell':
            group_names = []
            for event_idx in varying[0]:
                for feat_idx in varying[1]:
                    group_names += ["({}, {})".format(event_idx, feat_idx)]
            group_names += ["Other on event {}".format(x) for x in varying[0]]
            group_names += ["Other on feature {}".format(x) for x in varying[1]]
            group_names += ["Pruned Events", "Pruned Events"]
            return TimeShapDenseData(val, mode, group_names)
        elif mode == 'pruning':
            return TimeShapDenseData(val, mode, ["x", "hidden"])
        else:
            raise ValueError("`convert_to_data` - mode not supported")

    elif isinstance(val, Data):
        return val
    else:
        assert False, "Unknown type passed as data object: "+str(type(val))
