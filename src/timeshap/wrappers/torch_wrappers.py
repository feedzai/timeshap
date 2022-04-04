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
"""Wrapper to explain custom PyTorch RNN models.
"""

import numpy as np
import pandas as pd
import copy
import math
from typing import Tuple

import torch

from timeshap.wrappers import TimeSHAPWrapper


class TorchModelWrapper(TimeSHAPWrapper):
    """Wrapper for pytorch machine learning models.

    Encompasses necessary logic to utilize torch models as lambda functions
    required for TimeSHAP explanations.

    This wrapper is responsible to create torch tensors, sending them to the
    required device, batching processes, and obtained predictions from tensors.

    Attributes
    ----------
    model: torch.nn.Module
        Torch model to wrap. This model is required to receive a torch.Tensor
        of sequences and returning the score for each instance of each sequence.

    batch_budget: int
        The number of instances to score at a time. Needed to not overload
        GPU memory.
        Default is 750K. Equates to a 7GB batch

    device: torch.device

    Methods
    -------
    predict_last(data: pd.DataFrame, metadata: Matrix) -> list
        Creates explanations for each instance in ``data``.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 batch_budget: int = 750000,
                 device: torch.device = None,
                 ):
        super().__init__(model, batch_budget)
        if device:
            self.model = self.model.to(device)

    def prepare_input(self, input):
        sequence = copy.deepcopy(input)
        if isinstance(sequence, pd.DataFrame):
            sequence = np.expand_dims(sequence.values, axis=0)
        elif len(sequence.shape) == 2 and isinstance(sequence, np.ndarray):
            sequence = np.expand_dims(sequence, axis=0)

        if not (len(sequence.shape) == 3 and isinstance(sequence, np.ndarray)):
            raise ValueError("Input type not supported")

        return sequence

    def predict_last_hs(self,
                     sequences: np.ndarray,
                     hidden_states: np.ndarray = None,
                     ) -> Tuple[np.ndarray, np.ndarray]:
        sequences = self.prepare_input(sequences)
        device = next(self.model.parameters()).device

        sequence_len = sequences.shape[1]
        batch_size = math.floor(self.batch_budget / sequence_len)
        batch_size = max(1, batch_size)

        if sequences.shape[0] <= batch_size:
            with torch.no_grad():
                self.model.train(False)
                data_tensor = torch.from_numpy(sequences.copy()).float().to(device)
                if hidden_states is not None:
                    hidden_states_tensor = torch.from_numpy(hidden_states)
                    predictions = self.model(data_tensor, hidden_states_tensor)
                else:
                    predictions = self.model(data_tensor)
                self.model.train(True)
                if not isinstance(predictions, tuple):
                    return predictions.cpu().numpy()
                elif isinstance(predictions, tuple) and len(predictions) == 2:
                    predictions, hs = predictions
                    return predictions.cpu().numpy(), hs.cpu().numpy()
                else:
                    raise NotImplementedError(
                        "Only models that return predictions or predictions + hidden states are supported for now.")

        else:
            return_scores = []
            return_hs = []
            hs = None
            for i in range(0, sequences.shape[0], batch_size):
                batch = sequences[i:(i + batch_size), :, :]
                batch_tensor = torch.from_numpy(batch.copy())
                with torch.no_grad():
                    self.model.train(False)
                    batch_tensor = batch_tensor.float().to(device)
                    if hidden_states is not None:
                        batch_hs_tensor = torch.from_numpy(hidden_states[:, i:(i + batch_size), :].copy()).float().to(device)
                        predictions = self.model(batch_tensor, batch_hs_tensor)
                    else:
                        predictions = self.model(batch_tensor)
                    self.model.train(True)

                    if not isinstance(predictions, tuple):
                        predictions = predictions.cpu()
                    elif isinstance(predictions, tuple) and len(predictions) == 2:
                        predictions, hs = predictions
                        predictions, hs = predictions.cpu(), hs.cpu()
                        return_hs.append(hs.numpy())
                    else:
                        raise NotImplementedError(
                            "Only models that return predictions or predictions + hidden states are supported for now.")

                return_scores.append(predictions.numpy())
            if hs is None:
                return np.concatenate(tuple(return_scores), axis=0)

            return np.concatenate(tuple(return_scores), axis=0), np.concatenate(tuple(return_hs), axis=1)
