# Data wrangling libraries
import numpy as np
import pandas as pd
import math

# Machine Learning libraries
## PyTorch 
import torch 
import torch.nn as nn
from typing import Union, Optional 
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
## PyPots 
from pypots.classification.grud.data import DatasetForGRUD
## TQDM
import tqdm


class TemporalDecay(nn.Module):
    """The module used to generate the temporal decay factor gamma in the original paper.

    Attributes
    ----------
    W: tensor,
        The weights (parameters) of the module.
    b: tensor,
        The bias of the module.

    Parameters
    ----------
    input_size : int,
        the feature dimension of the input

    output_size : int,
        the feature dimension of the output

    diag : bool,
        whether to product the weight with an identity matrix before forward processing
    """

    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std_dev = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-std_dev, std_dev)
        if self.b is not None:
            self.b.data.uniform_(-std_dev, std_dev)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """Forward processing of the NN module.

        Parameters
        ----------
        delta : tensor, shape [batch size, sequence length, feature number]
            The time gaps.

        Returns
        -------
        gamma : array-like, same shape with parameter `delta`, values in (0,1]
            The temporal decay factor.
        """
        if self.diag:
            gamma = F.relu(F.linear(delta, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(delta, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma
    

class _GRUD(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        n_classes: int,
        device: Union[str, torch.device],
        saving_path: Optional[str] = None,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes
        self.device = device
        self.saving_path = saving_path

        # create models
        self.rnn_cell = nn.GRUCell(
            self.n_features * 2 + self.rnn_hidden_size, self.rnn_hidden_size
        )
        self.temp_decay_h = TemporalDecay(
            input_size=self.n_features, output_size=self.rnn_hidden_size, diag=False
        )
        self.temp_decay_x = TemporalDecay(
            input_size=self.n_features, output_size=self.n_features, diag=True
        )
        self.classifier = nn.Linear(self.rnn_hidden_size, self.n_classes)
        self.output = nn.Sigmoid()

    def classify(self, X) -> torch.Tensor:
        TEST_SET = {"X": X}
        training_set = DatasetForGRUD(TEST_SET)
        ts = DataLoader(training_set)
 
        values = ts.dataset.X.to(torch.float32).to(self.device)
        masks = ts.dataset.missing_mask.to(torch.float32).to(self.device)
        deltas = ts.dataset.deltas.to(torch.float32).to(self.device)
        empirical_mean = ts.dataset.empirical_mean.to(torch.float32).to(self.device)

        # X_filled
        if type(X) == torch.Tensor:
            X = X.detach().cpu().numpy()

        trans_X = X.transpose((0, 2, 1))
        mask = np.isnan(trans_X)
        n_samples, n_steps, n_features = mask.shape
        idx = np.where(~mask, np.arange(n_features), 0)
        np.maximum.accumulate(idx, axis=2, out=idx)

        collector = []
        for x, i in zip(trans_X, idx):
            collector.append(x[np.arange(n_steps)[:, None], i])
        X_imputed = np.asarray(collector)
        X_imputed = X_imputed.transpose((0, 2, 1))

        # If there are values still missing,
        # they are missing at the beginning of the time-series sequence.
        # Impute them with self.nan
        if np.isnan(X_imputed).any():
            X_imputed = np.nan_to_num(X_imputed)
        X_filledLOCF = torch.from_numpy(X_imputed).to(torch.float32).to(self.device)

        hidden_state = torch.zeros(
            (values.size()[0], self.rnn_hidden_size), device=self.device
        )

        for t in range(self.n_steps):
            # for data, [batch, time, features]
            x = values[:, t, :]  # values
            m = masks[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap
            x_filledLOCF = X_filledLOCF[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)
            hidden_state = hidden_state * gamma_h

            x_h = gamma_x * x_filledLOCF + (1 - gamma_x) * empirical_mean
            x_replaced = m * x + (1 - m) * x_h
            inputs = torch.cat([x_replaced, hidden_state, m], dim=1)
            hidden_state = self.rnn_cell(inputs, hidden_state)

        logits = self.classifier(hidden_state)
        prediction = self.output(logits)
        return prediction

    def forward(self, X) -> dict:
        """Forward processing of GRU-D.

        Parameters
        ----------
        inputs : dict,
            The input data.

        Returns
        -------
        dict,
            A dictionary includes all results.
        """
        prediction = self.classify(X)
        return prediction

    def predict(self, X):
        """Predict the labels of the input data.

        Parameters
        ----------
        inputs : dict,
            The input data.

        Returns
        -------
        dict,
            A dictionary includes all results.
        """
        self.eval()
        yhat = self.classify(X)
        yhat = yhat >= 0.5
        return yhat

