#!/usr/bin/python
#
# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standard Errors for //ads/metrics/lib/meterstick."""

from __future__ import division

import google3
import numpy as np
import pandas as pd
import scipy.stats


class StandardErrorMethod(object):
  """Base class for standard error methods."""

  def __init__(self, name, compute, confidence=None):
    """Initializes StandardErrorMethod.

    Args:
      name: A string which will be used in the column name of results.
      compute: A function which computes standard errors.
      confidence: Optional, the level of the confidence interval, must be in
        (0,1).

    Raises:
      ValueError: Confidence is not in (0,1).
    """

    self.name = name
    self._compute = compute
    self.confidence = confidence

    if confidence is not None and not 0 < confidence < 1:
      raise ValueError("Confidence must be in (0,1).")

  def _estimate_comparison(self, data, weights, metric, comparison,
                           full_estimate):
    """Computes standard errors for comparisons.

    Args:
      data: A pandas dataframe.
      weights: A numpy array of weights.
      metric: A Metric object.
      comparison: A comparison object.
      full_estimate: The output of comparison(data, weights, metric).

    Returns:
      An array of standard errors corresponding to the rows of
      comparison(data, weights, metric).
    """

    fn = lambda data, weights: comparison(data, weights, metric)
    baseline_inclusion = (comparison.factors == comparison.baseline_index)

    # We iterate through alternate conditions, zeroing out all but one
    # condition and running the comparison. Because comparison returns
    # a dataframe with all conditions we want to remove only the row
    # with the relevant comparison result. Because comparisons remove
    # the baseline condition, we just remove successive elements from
    # the returned array.

    # stderrs will store our standard error estimates. For dataframe metrics,
    # it is a dataframe indexed like the full estimate: by comparison condition
    # and the dataframe metric index. For scalar metrics, it is an array.
    if metric.metric_type == "dataframe":
      tmp = np.empty_like(full_estimate.values.flatten(), float)
      tmp[:] = np.nan
      stderrs = pd.DataFrame(tmp, index=full_estimate.index,
                             columns=full_estimate.columns)
    else:
      stderrs = pd.DataFrame(np.empty_like(full_estimate.values.flatten(),
                                           float))

    # altfactor is the comparison condition factor.
    for (row, altfactor) in enumerate(comparison.alternate_indices):
      included = baseline_inclusion | (comparison.factors == altfactor)
      new_weights = weights * included
      # se has the standard error estimate for this comparison condition.
      # The index of se has comparison condition as the outermost level.
      # However, because we pass the whole data in and zero out weights, it
      # includes NaN results for all other comparison conditions, so we need to
      # pull out the actual result and store it in stderrs.
      stderr = self._compute(data, new_weights, fn)
      if metric.metric_type == "dataframe":
        key = comparison.alternate_keys[row]
        for v in stderr.loc[key].index:
          # v iterates through indices of the dataframe metric result.
          stderrs.loc[key].loc[v] = stderr.loc[key].loc[v].values
      else:
        # the RHS should only be one value
        stderrs.iloc[row] = stderr.iloc[row].values

    return stderrs

  def __call__(self, df, weights, metric, comparison=None):
    """Computes standard errors for metric and possibly comparison.

    Args:
      df: A pandas dataframe.
      weights: A numpy array of weights.
      metric: A Metric object.
      comparison: Optionally a Comparison object.

    Returns:
      A pandas dataframe with the estimate and either standard errors
      or, if confidence is provided, confidence intervals.
    """

    if comparison is not None:
      full_df = comparison(df, weights, metric)
      stderrs = self._estimate_comparison(df, weights, metric, comparison,
                                          full_df)
    else:
      full_df = metric(df, weights)
      stderrs = self._compute(df, weights, metric)

    if isinstance(full_df, pd.DataFrame):
      base_name = full_df.columns[0]
    else:
      base_name = ""

    if self.confidence is None:
      if metric.metric_type == "dataframe":
        return full_df.join(stderrs, how="outer", lsuffix="",
                            rsuffix=" {} SE".format(self.name))

      columns = (base_name, "{} {} {}".format(base_name, self.name, "SE"))
      if isinstance(full_df, pd.DataFrame):
        return pd.DataFrame(
            np.array(zip(full_df.values.flatten(), stderrs.values.flatten())),
            columns=columns,
            index=full_df.index)
      else:
        return pd.DataFrame(
            np.array([(full_df, stderrs)]),
            columns=columns,
            index=None)

    else:
      degrees_of_freedom = self._degrees_of_freedom(weights)
      multiplier = scipy.stats.t.ppf(
          (1 + self.confidence) / 2, degrees_of_freedom)
      columns = (base_name,
                 "{} {} {}".format(base_name, self.name, "CI-lower"),
                 "{} {} {}".format(base_name, self.name, "CI-upper"))
      if isinstance(full_df, pd.DataFrame):
        full_estimate = full_df.values.flatten()
        lower = full_estimate - multiplier * stderrs.values.flatten()
        upper = full_estimate + multiplier * stderrs.values.flatten()
        return pd.DataFrame(
            zip(full_estimate, lower, upper),
            index=full_df.index,
            columns=columns)

      lower = full_df - multiplier * stderrs
      upper = full_df + multiplier * stderrs
      return pd.DataFrame(
          np.array(zip([full_df], [lower], [upper])),
          columns=columns,
          index=None)


class Jackknife(StandardErrorMethod):
  """Class for Jackknife estimates of standard errors.

  Attributes:
    unit: A string for the column whose values are the levels which define
      the jackknife buckets.
  """

  def __init__(self, unit=None, confidence=None):
    """Initializes the Jackknife.

    Args:
      unit: The variable name for which column to use as Jackknife units.
      confidence: Optional, the level of the confidence interval, must be in
        (0,1).
    """

    self._unit = unit

    def _compute(df, weights, fn):
      """Calculates Jackknife SEs for function fn on the dataframe df.

      Args:
        df: A pandas dataframe.
        weights: A numpy array of weights.
        fn: A function which takes df and produces a dataframe with point
        estimates.

      Returns:
        A pandas dataframe with point estimates and SEs for fn(df).
      """

      labels = self._determine_nonempty_buckets(weights)
      num_buckets = len(labels)

      bucket_estimates = []
      for label in labels:
        bucket_weights = weights * (self._buckets != label)
        bucket_estimates.append(fn(df, bucket_weights))

      means = sum(bucket_estimates) / num_buckets
      rss = sum([(b - means) ** 2 for b in bucket_estimates])
      stderrs = np.sqrt(rss * (num_buckets - 1) / num_buckets)
      return stderrs

    super(Jackknife, self).__init__("Jackknife", _compute, confidence)

  def precalculate_factors(self, df):
    """Precomputes the buckets and labels for the Jackknife object.

    Args:
      df: A pandas dataframe.
    """

    if self._unit is None:
      self._buckets = np.arange(len(df))
      self._bucket_labels = np.arange(len(df))
    else:
      self._buckets, names = pd.factorize(df[self._unit])
      self._bucket_labels = np.arange(len(names))

  def _degrees_of_freedom(self, weights):
    """Calculates the degrees of freedom for SE's T distribution.

    Args:
      weights: A numpy array of weights.

    Returns:
      The degrees of freedom for the SE's T distribution.
    """

    labels = self._determine_nonempty_buckets(weights)
    return len(labels) - 1

  def _determine_nonempty_buckets(self, weights):
    """Creates an array of jackknife buckets.

    Args:
      weights: A numpy array of weights.

    Returns:
       An array of integers in which specify the non-empty buckets

    Raises:
      ValueError: The jackknife is not valid with <= 1 bucket.
    """

    nonempty_labels = [ii for ii in self._bucket_labels
                       if weights.dot(self._buckets == ii) > 0]

    num_buckets = len(nonempty_labels)

    if num_buckets < 2:
      raise ValueError("Jackknife not valid; only {} buckets available.".format(
          num_buckets))

    return nonempty_labels


class Bootstrap(StandardErrorMethod):
  """Class for Bootstrap estimates of standard errors.

  Attributes:
    _num_replicates: A integer for the number of bootstrap replicates.
    _unit: A string for the variable representing the level to be resampled.
  """

  def __init__(self, num_replicates, unit=None, confidence=None):
    """Initializes the Bootstrap.

    Args:
      num_replicates: The number of bootstrap replicates to compute.
      unit: The variable name for which column to use as Bootstrap units.
      confidence: Optional, the level of the confidence interval, must be in
        (0,1).
    """
    self._unit = unit

    def _compute(df, weights, fn):
      """Calculates bootstrap standard errors for fn on data df.

      Args:
        df: A pandas dataframe.
        weights: A numpy array of weights.
        fn: A function which takes df and produces a dataframe with point
        estimates.

      Returns:
        A pandas dataframe with estimates and bootstrap standard errors.
      """

      replicate_estimates = []
      for _ in xrange(num_replicates):
        replicate_weights = self._generate_replicate_weights(weights)
        replicate_estimates.append(fn(df, replicate_weights))

      means = sum(replicate_estimates) / num_replicates
      rss = sum([(b - means) ** 2 for b in replicate_estimates])
      stderrs = np.sqrt(rss / num_replicates)

      return stderrs

    super(Bootstrap, self).__init__("Bootstrap", _compute, confidence)

  def _degrees_of_freedom(self, df, weights):
    """Calculates the degrees of freedom for SE's T distribution.

    Args:
      df: A pandas dataframe.
      weights: A numpy array of weights.

    Returns:
      The degrees of freedom for the SE's T distribution.
    """
    if self._unit is not None:
      values = df[weights > 0][self._unit].unique()
      degrees_of_freedom = len(values) - 1
    else:
      degrees_of_freedom = sum(weights) - 1

    return degrees_of_freedom

  def precalculate_factors(self, df):
    """Initializes the labels for the Bootstrap object.

    Args:
      df: A pandas dataframe.
    """

    if self._unit is not None:
      self._values, labels = pd.factorize(df[self._unit])
      self._labels = [self._values == ii for ii in range(len(labels))]

  def _generate_replicate_weights(self, weights):
    """Generates a replicate copies.

    Args:
      weights: A numpy array of weights.

    Returns:
      A numpy array of resampled weights.
    """

    if self._unit is not None:
      replicate_weights = np.zeros_like(weights)
      for label in self._labels:
        # To resample labels we assign all non-zero elements in a
        # label to have the same replicate weight; we rely on the
        # weights being initalized to zero so this respects any zeroed
        # out weights.
        replicate_weights[(weights > 0) & label] = np.random.poisson(1)
    else:
      replicate_weights = np.random.poisson(weights)

    return replicate_weights
