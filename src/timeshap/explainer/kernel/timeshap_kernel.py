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
"""
This file is based on the original SHAP implementation:
https://github.com/slundberg/shap/blob/master/shap/explainers/_kernel.py
"""

import numpy as np
import pandas as pd
import scipy as sp
import logging
import copy
import itertools

from timeshap.utils.timeshap_legacy import time_shap_match_instance_to_data, \
    time_shap_match_model_to_data, time_shap_convert_to_data, TimeShapDenseData
from shap.utils._legacy import convert_to_link, IdentityLink
from shap.utils._legacy import convert_to_instance, convert_to_model
from shap.explainers._kernel import Kernel
from scipy.special import binom
from scipy.sparse import issparse

log = logging.getLogger('shap')


class TimeShapKernel(Kernel):
    """Uses the Kernel SHAP method to explain the output of any function.

    TimeSHAP extends KernelSHAP to explain sequences of features on several axis.
    TimeSHAP calculates, event, feature, and cell level explanations.
    Due to sequences being arbitrarily long, TimeSHAP also implements a pruning
    algorithm based on Shapley values, to select the most relevant, recent,
    consecutive events.

    Parameters
    ----------
    model : function
        User supplied function that takes a 3D array (# samples x # sequence length x # features)
        and computes the output of the model for those samples. The output can be a vector
        (# samples) or a matrix (# samples x # model outputs).

    background : numpy.array or pd.DataFrame
        The background instance to use for integrating out features. To determine the impact
        of a feature, that feature is set to "missing" and the change in the model output
        is observed. Since most models aren't designed to handle arbitrary missing data at test
        time, we simulate "missing" by replacing the feature with the values it takes in the
        background dataset. So if the background dataset is a simple sample of all zeros, then
        we would approximate a feature being missing by setting it to zero.
        Consider using the `time.shap.calc_avg_event` method for this instance

    rs: int
        Random seed for timeshap algorithm

    mode: str
        This method indicates what kind of explanations should be calculated.
        Possible values: ["pruning", "event", "feature", "cell"]
            - "pruning" - used for pruning algorithm
            - "event" - used for event explanations
            - "feature" - used for feature explanations
            - "cell" -used for cell explanations

    varying: Tuple
        index of varying indexes on cell level
        If mode == "cell": varying needs to be of len 2, the first the idx of
            events to preturb, and the second the idx of features

    link : "identity" or "logit"
        A generalized linear model link to connect the feature importance values to the model
        output. Since the feature importance values, phi, sum up to the model output, it often makes
        sense to connect them to the output with a link function where link(output) = sum(phi).
        If the model output is a probability then the LogitLink link function makes the feature
        importance values have log-odds units.
    """
    def __init__(self, model, background, rs, mode, varying=None, link=IdentityLink(), **kwargs):
        self.background = background
        self.random_seed = rs
        self.mode = mode
        self.data = None
        self.varyingInds = None
        self.pruning_idx = None
        self.varying = varying
        self.returns_hs = None
        self.background_hs = None
        self.instance_hs = None
        if mode in 'cell':
            if varying is None:
                # The algorithm supports the calculation using all events and features
                # but its computation is very expensive as the number of cells is very large
                raise ValueError("Cell level needs to receive which cells to calculate")
            self.varying = varying
            cell_idx_keys = []
            for event_idx in self.varying[0]:
                for feat_idx in self.varying[1]:
                    cell_idx_keys += [[event_idx, feat_idx]]
            self.cell_idx_keys = np.array(cell_idx_keys)
            event_idx = self.cell_idx_keys[:, 0]
            self.considered_cells = {}
            for event in np.unique(event_idx):
                self.considered_cells[event] = self.cell_idx_keys[self.cell_idx_keys[:, 0] == event][:, 1]

        # convert incoming inputs to standardized iml objects
        self.link = convert_to_link(link)
        self.model = convert_to_model(model)
        self.keep_index = kwargs.get("keep_index", False)
        self.keep_index_ordered = kwargs.get("keep_index_ordered", False)

    def set_variables_up(self, X):
        sequence = np.tile(self.background, (X.shape[1], 1))
        sequence = np.expand_dims(sequence.copy(), axis=0)

        if self.mode == 'cell':
            self.data, self.special_cells = time_shap_convert_to_data(sequence, self.mode, self.pruning_idx, self.varying)
        else:
            self.data = time_shap_convert_to_data(sequence, self.mode, self.pruning_idx, self.varying)

        model_null, returns_hs = time_shap_match_model_to_data(self.model, self.data)
        self.returns_hs = returns_hs

        if not self.mode == 'pruning' and self.returns_hs:
            if self.pruning_idx == 0:
                _, example_hs = self.model.f(X[:, -1:, :])
                self.instance_hs = np.zeros_like(example_hs)
                self.background_hs = np.zeros_like(example_hs)
            else:
                _, self.background_hs = self.model.f(sequence[:, :self.pruning_idx, :])
                _, self.instance_hs = self.model.f(X[:, :self.pruning_idx, :])
        # enforce our current input type limitations
        assert isinstance(self.data, TimeShapDenseData), \
            "Shap explainer only supports the DenseData input currently."
        assert not self.data.transposed, "Shap explainer does not support transposed DenseData or SparseData currently."

        # warn users about large background data sets
        if len(self.data.weights) > 100:
            log.warning("Using " + str(len(
                self.data.weights)) + " background data samples could cause " +
                        "slower run times. Consider using shap.sample(data, K) or shap.kmeans(data, K) to " +
                        "summarize the background as K samples.")

        # init our parameters
        self.N = self.data.data.shape[0]
        # seq len total
        self.S = self.data.data.shape[1]
        self.P = self.data.data.shape[2]
        self.linkfv = np.vectorize(self.link.f)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

        # find E_x[f(x)]
        if isinstance(model_null, (pd.DataFrame, pd.Series)):
            model_null = np.squeeze(model_null.values)
        self.fnull = np.sum((model_null.T * self.data.weights).T, 0)
        self.expected_value = self.linkfv(self.fnull)

        # see if we have a vector output
        self.vector_out = True
        if len(self.fnull.shape) == 0:
            self.vector_out = False
            self.fnull = np.array([self.fnull])
            self.D = 1
            self.expected_value = float(self.expected_value)
        else:
            self.D = self.fnull.shape[0]

    def shap_values(self, X, pruning_idx=None, **kwargs):
        """ Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array or pandas.DataFrame or any scipy.sparse matrix
            A matrix of samples (# samples x # features) on which to explain the model's output.

        nsamples : "auto" or int
            Number of times to re-evaluate the model when explaining each prediction. More samples
            lead to lower variance estimates of the SHAP values. The "auto" setting uses
            `nsamples = 2 * X.shape[1] + 2048`.

        l1_reg : "num_features(int)", "auto" (default for now, but deprecated), "aic", "bic", or float
            The l1 regularization to use for feature selection (the estimation procedure is based on
            a debiased lasso). The auto option currently uses "aic" when less that 20% of the possible sample
            space is enumerated, otherwise it uses no regularization. THE BEHAVIOR OF "auto" WILL CHANGE
            in a future version to be based on num_features instead of AIC.
            The "aic" and "bic" options use the AIC and BIC rules for regularization.
            Using "num_features(int)" selects a fix number of top features. Passing a float directly sets the
            "alpha" parameter of the sklearn.linear_model.Lasso model used for feature selection.

        Returns
        -------
        For models with a single output this returns a matrix of SHAP values
        (# samples x # features). Each row sums to the difference between the model output for that
        sample and the expected value of the model output (which is stored as expected_value
        attribute of the explainer). For models with vector outputs this returns a list
        of such matrices, one for each output.
        """
        if self.mode == "pruning":
            assert pruning_idx is not None
        else:
            assert pruning_idx < X.shape[1], "Pruning idx must be smaller than the sequence length. If not all events are pruned"
        assert pruning_idx % 1 == 0, "Pruning idx must be integer"
        self.pruning_idx = int(pruning_idx)

        self.set_variables_up(X)

        # Removed the input variability to receive pd.series and DataFrame
        # TODO implement this variability?

        if sp.sparse.issparse(X) and not sp.sparse.isspmatrix_lil(X):
            X = X.tolil()

        # single instance
        if X.shape[0] == 1:
            explanation = self.explain(X, **kwargs)

            out = np.zeros(explanation.shape[0])
            if isinstance(explanation.shape, tuple) and len(explanation.shape) == 2:
                assert explanation.shape[1] == 1
                out[:] = explanation[:, 0]
            else:
                out[:] = explanation
            return out

        elif X.shape[0] > 1:
            # TODO In case we want to make a method to explain a
            raise NotImplementedError()
        else:
            raise ValueError

    def explain(self, incoming_instance, **kwargs):
        # convert incoming input to a standardized iml object
        instance = convert_to_instance(incoming_instance)
        instance.group_display_values = self.data.group_names
        time_shap_match_instance_to_data(instance, self.data)

        # Find the feature groups we will test. If a feature does not change from its
        # current value then we know it doesn't impact the model
        if self.mode == "event":
            self.varyingInds = np.array([x for x in np.arange(incoming_instance.shape[1]-1, self.pruning_idx-1, -1)])
        elif self.mode == 'pruning':
            self.varyingInds = [0, 1]
        elif self.mode == "feature":
            if self.pruning_idx > 0:
                self.varyingInds = self.varying_groups(instance.x, self.data.groups_size - 1)
                # add an index for pruned events
                self.varyingInds = np.concatenate((self.varyingInds, np.array([self.data.groups_size - 1])))
            else:
                self.varyingInds = self.varying_groups(instance.x, self.data.groups_size)
        elif self.mode == 'cell':
            self.varyingInds = np.arange(len(self.data.groups))
        else:
            raise ValueError("`explain` -> mode not suported")

        if self.data.groups is None:
            self.varyingFeatureGroups = np.array([i for i in self.varyingInds])
            self.M = self.varyingFeatureGroups.shape[0]
        else:
            if self.mode in ['event']:
                self.varyingFeatureGroups = self.varyingInds
                self.M = len(self.varyingFeatureGroups)
                if self.pruning_idx > 0:
                    self.M += 1
            elif self.mode in ['feature']:
                if self.pruning_idx > 0:
                    self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds[:-1]]
                else:
                    self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
                self.M = len(self.varyingFeatureGroups)
                if self.pruning_idx > 0:
                    self.M += 1
            else:
                self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
                self.M = len(self.varyingFeatureGroups)

            groups = self.data.groups
            # convert to numpy array as it is much faster if not jagged array (all groups of same length)
            if isinstance(self.varyingFeatureGroups, list) and all(len(groups[i]) == len(groups[0]) for i in range(len(self.varyingFeatureGroups))):
                self.varyingFeatureGroups = np.array(self.varyingFeatureGroups)
                # further performance optimization in case each group has a single value
                if self.varyingFeatureGroups.shape[1] == 1:
                    self.varyingFeatureGroups = self.varyingFeatureGroups.flatten()

        if self.returns_hs:
            # Removed the input variability to receive pd.series and DataFrame
            model_out, _ = self.model.f(instance.x)
        else:
            model_out = self.model.f(instance.x)


        self.fx = model_out[0]
        if not self.vector_out:
            self.fx = np.array([self.fx])

        # if no features vary then no feature has an effect
        if self.M == 0:
            phi = np.zeros((self.data.groups_size, self.D))

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros((self.data.groups_size, self.D))
            diff = self.link.f(self.fx) - self.link.f(self.fnull)
            for d in range(self.D):
                phi[self.varyingInds[0], d] = diff[d]

        # if more than one feature varies then we have to do real work
        else:
            self.l1_reg = kwargs.get("l1_reg", "auto")

            # pick a reasonable number of samples if the user didn't specify how many they wanted
            self.nsamples = kwargs.get("nsamples", "auto")
            if self.nsamples == "auto":
                self.nsamples = 2 * self.M + 2**11

            # if we have enough samples to enumerate all subsets then ignore the unneeded samples
            self.max_samples = 2 ** 30
            if self.M <= 30:
                self.max_samples = 2 ** self.M - 2
                if self.nsamples > self.max_samples:
                    self.nsamples = self.max_samples

            # reserve space for some of our computations
            self.allocate()

            # weight the different subset sizes
            num_subset_sizes = np.int(np.ceil((self.M - 1) / 2.0))
            num_paired_subset_sizes = np.int(np.floor((self.M - 1) / 2.0))
            weight_vector = np.array([(self.M - 1.0) / (i * (self.M - i)) for i in range(1, num_subset_sizes + 1)])
            weight_vector[:num_paired_subset_sizes] *= 2
            weight_vector /= np.sum(weight_vector)
            log.debug("weight_vector = {0}".format(weight_vector))
            log.debug("num_subset_sizes = {0}".format(num_subset_sizes))
            log.debug("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
            log.debug("M = {0}".format(self.M))

            # fill out all the subset sizes we can completely enumerate
            # given nsamples*remaining_weight_vector[subset_size]
            num_full_subsets = 0
            num_samples_left = self.nsamples
            group_inds = np.arange(self.M, dtype='int64')
            mask = np.zeros(self.M)
            remaining_weight_vector = copy.copy(weight_vector)
            for subset_size in range(1, num_subset_sizes + 1):

                # determine how many subsets (and their complements) are of the current size
                nsubsets = binom(self.M, subset_size)
                if subset_size <= num_paired_subset_sizes: nsubsets *= 2
                log.debug("subset_size = {0}".format(subset_size))
                log.debug("nsubsets = {0}".format(nsubsets))
                log.debug("self.nsamples*weight_vector[subset_size-1] = {0}".format(
                    num_samples_left * remaining_weight_vector[subset_size - 1]))
                log.debug("self.nsamples*weight_vector[subset_size-1]/nsubsets = {0}".format(
                    num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets))

                # see if we have enough samples to enumerate all subsets of this size
                if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                    num_full_subsets += 1
                    num_samples_left -= nsubsets

                    # rescale what's left of the remaining weight vector to sum to 1
                    if remaining_weight_vector[subset_size - 1] < 1.0:
                        remaining_weight_vector /= (1 - remaining_weight_vector[subset_size - 1])

                    # add all the samples of the current subset size
                    w = weight_vector[subset_size - 1] / binom(self.M, subset_size)
                    if subset_size <= num_paired_subset_sizes: w /= 2.0
                    for inds in itertools.combinations(group_inds, subset_size):
                        mask[:] = 0.0
                        mask[np.array(inds, dtype='int64')] = 1.0
                        self.add_sample(instance.x, mask, w)
                        if subset_size <= num_paired_subset_sizes:
                            mask[:] = np.abs(mask - 1)
                            self.add_sample(instance.x, mask, w)
                else:
                    break
            log.info("num_full_subsets = {0}".format(num_full_subsets))
            # add random samples from what is left of the subset space
            nfixed_samples = self.nsamplesAdded
            samples_left = self.nsamples - self.nsamplesAdded
            log.debug("samples_left = {0}".format(samples_left))
            np.random.seed(self.random_seed)
            if num_full_subsets != num_subset_sizes:
                remaining_weight_vector = copy.copy(weight_vector)
                remaining_weight_vector[:num_paired_subset_sizes] /= 2 # because we draw two samples each below
                remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
                remaining_weight_vector /= np.sum(remaining_weight_vector)
                log.info("remaining_weight_vector = {0}".format(remaining_weight_vector))
                log.info("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
                ind_set = np.random.choice(len(remaining_weight_vector), 4 * samples_left, p=remaining_weight_vector)
                ind_set_pos = 0
                used_masks = {}
                while samples_left > 0 and ind_set_pos < len(ind_set):
                    mask.fill(0.0)
                    ind = ind_set[ind_set_pos] # we call np.random.choice once to save time and then just read it here
                    ind_set_pos += 1
                    subset_size = ind + num_full_subsets + 1
                    mask[np.random.permutation(self.M)[:subset_size]] = 1.0

                    # only add the sample if we have not seen it before, otherwise just
                    # increment a previous sample's weight
                    mask_tuple = tuple(mask)
                    new_sample = False
                    if mask_tuple not in used_masks:
                        new_sample = True
                        used_masks[mask_tuple] = self.nsamplesAdded
                        samples_left -= 1
                        self.add_sample(instance.x, mask, 1.0)
                    else:
                        self.kernelWeights[used_masks[mask_tuple]] += 1.0

                    # add the compliment sample
                    if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1)

                        # only add the sample if we have not seen it before, otherwise just
                        # increment a previous sample's weight
                        if new_sample:
                            samples_left -= 1
                            self.add_sample(instance.x, mask, 1.0)
                        else:
                            # we know the compliment sample is the next one after the original sample, so + 1
                            self.kernelWeights[used_masks[mask_tuple] + 1] += 1.0

                # normalize the kernel weights for the random samples to equal the weight left after
                # the fixed enumerated samples have been already counted
                weight_left = np.sum(weight_vector[num_full_subsets:])
                log.info("weight_left = {0}".format(weight_left))
                self.kernelWeights[nfixed_samples:] *= weight_left / self.kernelWeights[nfixed_samples:].sum()

            # execute the model on the synthetic samples we have created
            self.run()

            # solve then expand the feature importance (Shapley value) vector to contain the non-varying features
            phi = np.zeros((self.data.groups_size, self.D))
            for d in range(self.D):
                vphi, _ = self.solve(self.nsamples / self.max_samples, d)
                if self.mode == 'event':
                    phi[:, d] = vphi
                elif self.mode == 'cell':
                    phi[:, d] = vphi
                else:
                    phi[self.varyingInds, d] = vphi

        if not self.vector_out:
            phi = np.squeeze(phi, axis=1)

        return phi

    @staticmethod
    def not_equal(i, j):
        if isinstance(i, str) or isinstance(j, str):
            return 0 if i == j else 1
        return 0 if np.isclose(i, j, equal_nan=True) else 1

    def varying_groups(self, x, group_size):
        if not sp.sparse.issparse(x):
            varying = np.zeros(group_size)
            for i in range(0, group_size):
                inds = self.data.groups[i]
                x_group = x[0, :, inds]
                if sp.sparse.issparse(x_group):
                    if all(j not in x.nonzero()[1] for j in inds):
                        varying[i] = False
                        continue
                    x_group = x_group.todense()
                num_mismatches = np.sum(np.frompyfunc(self.not_equal, 2, 1)(x_group, self.data.data[:, 0, inds]))
                varying[i] = num_mismatches > 0
            varying_indices = np.nonzero(varying)[0]
            return varying_indices
        else:
            # go over all nonzero columns in background and evaluation data
            # if both background and evaluation are zero, the column does not vary
            varying_indices = np.unique(np.union1d(self.data.data.nonzero()[1], x.nonzero()[1]))
            remove_unvarying_indices = []
            for i in range(0, len(varying_indices)):
                varying_index = varying_indices[i]
                # now verify the nonzero values do vary
                data_rows = self.data.data[:, [varying_index]]
                nonzero_rows = data_rows.nonzero()[0]

                if nonzero_rows.size > 0:
                    background_data_rows = data_rows[nonzero_rows]
                    if sp.sparse.issparse(background_data_rows):
                        background_data_rows = background_data_rows.toarray()
                    num_mismatches = np.sum(np.abs(background_data_rows - x[0, varying_index]) > 1e-7)
                    # Note: If feature column non-zero but some background zero, can't remove index
                    if num_mismatches == 0 and not \
                        (np.abs(x[0, [varying_index]][0, 0]) > 1e-7 and len(nonzero_rows) < data_rows.shape[0]):
                        remove_unvarying_indices.append(i)
            mask = np.ones(len(varying_indices), dtype=bool)
            mask[remove_unvarying_indices] = False
            varying_indices = varying_indices[mask]
            return varying_indices

    def allocate(self):
        if sp.sparse.issparse(self.data.data):
            # We tile the sparse matrix in csr format but convert it to lil
            # for performance when adding samples
            shape = self.data.data.shape
            nnz = self.data.data.nnz
            data_rows, data_cols = shape
            rows = data_rows * self.nsamples
            shape = rows, data_cols
            if nnz == 0:
                self.synth_data = sp.sparse.csr_matrix(shape, dtype=self.data.data.dtype).tolil()
            else:
                data = self.data.data.data
                indices = self.data.data.indices
                indptr = self.data.data.indptr
                last_indptr_idx = indptr[len(indptr) - 1]
                indptr_wo_last = indptr[:-1]
                new_indptrs = []
                for i in range(0, self.nsamples - 1):
                    new_indptrs.append(indptr_wo_last + (i * last_indptr_idx))
                new_indptrs.append(indptr + ((self.nsamples - 1) * last_indptr_idx))
                new_indptr = np.concatenate(new_indptrs)
                new_data = np.tile(data, self.nsamples)
                new_indices = np.tile(indices, self.nsamples)
                self.synth_data = sp.sparse.csr_matrix((new_data, new_indices, new_indptr), shape=shape).tolil()
        else:
            if self.returns_hs and self.mode != 'pruning':
                self.synth_data = np.tile(self.data.data[:, self.pruning_idx:, :], (self.nsamples, 1, 1))
                self.synth_hidden_states = np.tile(self.background_hs, (1, self.nsamples, 1))
            else:
                self.synth_data = np.tile(self.data.data, (self.nsamples, 1, 1))

        self.maskMatrix = np.zeros((self.nsamples, self.M))
        self.kernelWeights = np.zeros(self.nsamples)
        self.y = np.zeros((self.nsamples * self.N, self.D))
        self.ey = np.zeros((self.nsamples, self.D))
        self.lastMask = np.zeros(self.nsamples)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0
        if self.keep_index:
            self.synth_data_index = np.tile(self.data.index_value, self.nsamples)

    def add_sample(self, x, m, w):
        if self.mode == "event":
            self.event_add_sample(x, m)
        elif self.mode == "feature":
            self.feat_add_sample(x, m)
        elif self.mode == 'pruning':
            self.pruning_add_sample(x, m)
        elif self.mode == "cell":
            self.cell_add_sample(x, m)
        else:
            raise ValueError("`add_sample` - Mode not suported")

        self.maskMatrix[self.nsamplesAdded, :] = m
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1

    def cell_add_sample(self, x, m):
        offset = self.nsamplesAdded * self.N
        mask = m == 1.0

        cells_to_preturb = self.cell_idx_keys[mask[: self.cell_idx_keys.shape[0]], :]
        relevent_events = np.unique(self.cell_idx_keys[:, 0])
        relevent_feats = np.unique(self.cell_idx_keys[:, 1])

        feats_by_event = {}
        for event in np.unique(cells_to_preturb[:, 0]):
            feats_by_event[event] = cells_to_preturb[cells_to_preturb[:, 0] == event][:, 1]

        # BACKGROUND IS ACTIVE
        if self.special_cells[3] and mask[-self.special_cells[3]]:
            # in case self.pruning_idx == sequence length, we dont prune anything.
            if not self.pruning_idx == self.S:
                if self.returns_hs:
                    self.synth_hidden_states[:, offset:offset + self.N, :] = self.instance_hs
                else:
                    evaluation_data = x[0:1, :self.pruning_idx, :]
                    self.synth_data[offset:offset + self.N, :self.pruning_idx, :] = evaluation_data

        # other cells are active (no relevant events or feats)
        if self.special_cells[2] and mask[-self.special_cells[2]]:
            other_events = [x for x in np.arange(self.pruning_idx, x.shape[1]) if x not in relevent_events]
            other_feats = [x for x in range(x.shape[2]) if x not in relevent_feats]

            for event in other_events:
                evaluation_data = x[0:1, event, other_feats]
                if self.returns_hs:
                    event = event - self.pruning_idx
                self.synth_data[offset:offset + self.N, event, other_feats] = evaluation_data

        mask_pointer = self.cell_idx_keys.shape[0]
        if self.special_cells[0]:
            # other feats in relevant events
            perturb_events = relevent_events[mask[mask_pointer: mask_pointer + len(self.varying[0])]]
            mask_pointer += len(self.varying[0])
            for event in perturb_events:
                other_feats = [x for x in range(x.shape[2]) if x not in relevent_feats]
                evaluation_data = x[0:1, event, other_feats]
                if self.returns_hs:
                    event = event - self.pruning_idx
                self.synth_data[offset:offset + self.N, event, other_feats] = evaluation_data

        if self.special_cells[1]:
            # other events in relevant feats
            perturb_feats = relevent_feats[mask[mask_pointer: mask_pointer + len(self.varying[1])]]
            mask_pointer += len(self.varying[1])
            other_events = [x for x in np.arange(self.pruning_idx, x.shape[1]) if x not in relevent_events]
            for event in other_events:
                evaluation_data = x[0:1, event, perturb_feats]
                if self.returns_hs:
                    event = event - self.pruning_idx
                self.synth_data[offset:offset + self.N, event, perturb_feats] = evaluation_data

        # activate individual cells
        for event, feats in feats_by_event.items():
            evaluation_data = x[0:1, event, feats]
            if self.returns_hs:
                event = event - self.pruning_idx
            self.synth_data[offset:offset + self.N, event, feats] = evaluation_data

    def pruning_add_sample(self, x, m):
        offset = self.nsamplesAdded * self.N
        mask = m == 1.0
        if not len(mask) == 2:
            raise ValueError("For pruning mode, masks must have size 2")
        if mask[0]:
            # cur active
            evaluation_data = x[0:1, self.pruning_idx:, :]
            self.synth_data[offset:offset + self.N, self.pruning_idx:, :] = evaluation_data
        if mask[1]:
            # background active
            evaluation_data = x[0:1, :self.pruning_idx, :]
            self.synth_data[offset:offset + self.N, :self.pruning_idx, :] = evaluation_data

    def event_add_sample(self, x, m):
        offset = self.nsamplesAdded * self.N
        mask = m == 1.0
        # there is a background and it is active
        if self.pruning_idx > 0 and mask[-1]:
            # in case self.pruning_idx == sequence length, we dont prune anything.
            if not self.pruning_idx == self.S:
                if self.returns_hs:
                    # in case of using hidden state optimization, the background is the instance one
                    self.synth_hidden_states[:, offset:offset + self.N, :] = self.instance_hs
                else:
                    # in case of not using hidden state optimization, we need to set the whole background to the original sequence
                    evaluation_data = x[0:1, :self.pruning_idx, :]
                    self.synth_data[offset:offset + self.N, :self.pruning_idx, :] = evaluation_data
        if self.pruning_idx > 0:
            # there is a background, so the last position of the mask is for it
            groups = self.varyingFeatureGroups[mask[:-1]]
        else:
            groups = self.varyingFeatureGroups[mask]

        evaluation_data = x[0:1, groups, :]
        if self.returns_hs:
            # re-align indexes to the truncated sequence
            groups = [x-self.pruning_idx for x in groups]
        self.synth_data[offset:offset + self.N, groups, :] = evaluation_data

    def feat_add_sample(self, x, m):
        offset = self.nsamplesAdded * self.N
        mask = m == 1.0
        #BACKGROUND IS ACTIVE
        if self.pruning_idx > 0 and mask[-1]:
            # in case self.pruning_idx == sequence length, we dont prune anything.
            if not self.pruning_idx == self.S:
                if self.returns_hs:
                    self.synth_hidden_states[:, offset:offset + self.N, :] = self.instance_hs
                else:
                    evaluation_data = x[0:1, :self.pruning_idx, :]
                    self.synth_data[offset:offset + self.N, :self.pruning_idx,:] = evaluation_data

        if self.pruning_idx > 0:
            # there is a background, so the last position of the mask is for it
            groups = self.varyingFeatureGroups[mask[:-1]]
        else:
            groups = self.varyingFeatureGroups[mask]
        evaluation_data = x[0:1, self.pruning_idx:, groups]
        if self.returns_hs:
            self.synth_data[offset:offset+self.N, :, groups] = evaluation_data
        else:
            self.synth_data[offset:offset+self.N, self.pruning_idx:, groups] = evaluation_data

    def run(self):
        num_to_run = self.nsamplesAdded * self.N - self.nsamplesRun * self.N

        data = self.synth_data[self.nsamplesRun * self.N:self.nsamplesAdded * self.N, :, :]

        if not self.mode == 'pruning' and self.returns_hs:
            hidden_sates = self.synth_hidden_states[:, self.nsamplesRun * self.N: self.nsamplesAdded * self.N,:]
            modelOut, _ = self.model.f(data, hidden_sates)
        elif self.returns_hs:
            modelOut, _ = self.model.f(data)
        else:
            modelOut = self.model.f(data)

        if isinstance(modelOut, (pd.DataFrame, pd.Series)):
            modelOut = modelOut.values
        self.y[self.nsamplesRun * self.N:self.nsamplesAdded * self.N, :] = np.reshape(modelOut, (num_to_run, self.D))

        # find the expected value of each output
        for i in range(self.nsamplesRun, self.nsamplesAdded):
            eyVal = np.zeros(self.D)
            for j in range(0, self.N):
                eyVal += self.y[i * self.N + j, :] * self.data.weights[j]

            self.ey[i, :] = eyVal
            self.nsamplesRun += 1
