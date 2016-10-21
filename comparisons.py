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

"""Comparisons for //ads/metrics/lib/meterstick."""

from __future__ import division

import google3
import numpy as np
import pandas as pd


class Comparison(object):
  """Base class for comparisons.

  Attributes:
    condition_column: A string denoting the dataframe column to
      compare against (e.g. "Experiment").
    baseline_key: A string denoting the value of the
      condition_column which represents the baseline comparison
      condition (e.g., "Control").
    factors: A numpy array for the values of condition_column coded as integers
    alternate_keys: The values of factors that are not the baseline
    baseline_index: The integer for the baseline_key in factors
    alternate_indices: The unique values of condition_column ordered by factors
  """

  # These attributes are public because they have to be accessed by
  # standard error methods.

  def __init__(self, condition_column, baseline_key, name, compare,
               include_base=False):
    """Initializes the Comparison object.

    Args:
      condition_column: A string denoting the dataframe column to
        compare against (e.g. "Experiment").
      baseline_key: A string denoting the value of the
        condition_column which represents the baseline comparison
        condition (e.g., "Control").
      name: A string denoting the column name for results of the
        comparison.
      compare: A function which takes a dataframe and weights and
        returns the comparison between the baseline and non-baseline
        conditions.
      include_base: A boolean for whether the base value should be included in
        the list of indices to compare.
    """

    self.condition_column = condition_column
    self.baseline_key = baseline_key
    self._name = name
    self._compare = compare
    self._include_base = include_base

  def precalculate_factors(self, data, sort=True):
    """Initializes the factor variable.

    Args:
      data: A pandas dataframe.
      sort: Boolean indicating whether or not the conditions should be sorted.

    Raises:
      ValueError: The baseline key isn't found.
    """

    self.factors, condition_keys = pd.factorize(data[self.condition_column],
                                                sort=sort)

    self.alternate_indices = [
        ii for ii, label in enumerate(condition_keys)
        if self._include_base or label != self.baseline_key
    ]
    self.alternate_keys = condition_keys[self.alternate_indices]

    if any(condition_keys == self.baseline_key):
      self.baseline_index = np.where(condition_keys == self.baseline_key)[0][0]
    else:
      raise ValueError("Baseline value {} not present in column {}".format(
          self.baseline_key, self.condition_column))

    self._baseline_mask = (self.factors == self.baseline_index)
    self._alternate_masks = {}
    for ii in self.alternate_indices:
      self._alternate_masks[ii] = (self.factors == ii)

  def __call__(self, data, weights, metric):
    """Calculates the comparison for the metric on dataframe data.

    Args:
      data: A pandas dataframe.
      weights: A numpy array of weights.
      metric: A Metric object.

    Returns:
      A pandas DataFrame with the comparisons of baseline against every other
      unique condition.
    """

    ## We want to call it with weights zeroed out for each of the conditions.

    dfs = []

    for ii in self.alternate_indices:
      dfs.append(self._compare(data, weights, metric, ii))

    if metric.metric_type != "dataframe":
      return pd.DataFrame(dfs,
                          index=pd.Index(self.alternate_keys,
                                         name=self.condition_column),
                          columns=[self._name])

    return pd.concat(dfs, keys=self.alternate_keys,
                     names=[self.condition_column])


def _make_simple_comparison(fn, name):
  """Creates a class for comparison which is a function of the metric values.

  Args:
    fn: A function of the metric values.
    name: A string for the column names of the results.

  Returns:
    A subclass of Comparison which implements the passed in comparison function.
  """

  class SimpleComparison(Comparison):

    def __init__(self, condition_column, baseline_key, include_base=False):
      """Initializes the comparison.

      Args:
        condition_column: A string denoting the dataframe column to
          compare against (e.g. "Experiment").
        baseline_key: A string denoting the value of the
          condition_column which represents the baseline comparison
          condition (e.g., "Control").
        include_base: A boolean for whether the base value should be included in
          the list of indices to compare.
      """

      def _compare(data, weights, metric, condition_index):
        """Compares the results in "baseline" and "not baseline" conditions.

        Args:
          data: A pandas dataframe.
          weights: A numpy array of weights.
          metric: A Metric object.
          condition_index: The value of factors which represents the comparison.

        Returns:
          A scalar with the results of the comparison.
        """

        baseline_weights = weights * self._baseline_mask
        comparison_weights = weights * self._alternate_masks[condition_index]

        if metric.metric_type == "dataframe":
          result = fn(metric(data, comparison_weights),
                      metric(data, baseline_weights))
          result.columns = [self._name]
          return result

        baseline_est = metric(data, baseline_weights)
        comparison_est = metric(data, comparison_weights)
        return fn(comparison_est, baseline_est)

      super(SimpleComparison, self).__init__(condition_column, baseline_key,
                                             name, _compare, include_base)

  return SimpleComparison


def _absolute_difference(x, y):
  return x - y


AbsoluteDifference = _make_simple_comparison(_absolute_difference,
                                             "Absolute Difference")


def _percentage_difference(x, y):
  return 100 * (x - y) / y


PercentageDifference = _make_simple_comparison(_percentage_difference,
                                               "Percentage Difference")


## Mantel-Haenszel needs to be implemented so that it can access the
## metric object. Thus, it doesn"t fit into the general
## implementation pattern of comparisons. However, conceptually it is
## a comparison between conditions and thus fits in here.
class MH(Comparison):
  """Class for Mantel-Haenszel estimator for comparing rates."""

  def __init__(self, condition_column, baseline_key, index_var,
               include_base=False):
    """Initializes the MH comparison.

    Args:
      condition_column: A string denoting the column in data to compare
        against (e.g. "Experiment").
      baseline_key: A string denoting the value of
        data[condition_column] that represents the baseline comparison
        condition (e.g., "Control").
      index_var: A string denote the variable which indicates the
        experimental unit to aggregate on (e.g. "AdGroup").
      include_base: A boolean for whether the base value should be included in
        the list of indices to compare.
    """
    name = "MH"

    def _compare(data, weights, metric, condition_index):
      """Calculates the MantelHaenszel estimator.

      Args:
        data: A pandas dataframe.
        weights: A numpy array of weights.
        metric: A Ratio object.
        condition_index: The value of factors which represents the comparison.
      Returns:
        A pandas dataframe with the MH estimator.

      Raises:
        ValueError: MH weight is undefined.
      """

      num = metric.numerator()
      den = metric.denominator()

      baseline_data = data[data[self.condition_column] == self.baseline_key]
      baseline_weights = weights[np.where(self.factors == self.baseline_index)]

      ## We rely upon the __call__ method to pass a dataframe such
      ## that only the baseline and exactly one other condition
      ## remain, so we can just check inequality here while __call__
      ## handles making multiple comparisons.

      comparison_data = data[data[self.condition_column] != self.baseline_key]
      comparison_weights = weights[np.where(self.factors == condition_index)]

      baseline_data[num] *= baseline_weights
      baseline_data[den] *= baseline_weights
      comparison_data[num] *= comparison_weights
      comparison_data[den] *= comparison_weights

      baseline_counts = baseline_data.groupby(index_var).agg({num: np.sum,
                                                              den: np.sum})

      comparison_counts = comparison_data.groupby(index_var).agg({num: np.sum,
                                                                  den: np.sum})

      joined = baseline_counts.join(
          comparison_counts,
          how="inner",
          lsuffix="baseline",
          rsuffix="comparison")

      num_baseline = num + "baseline"
      den_baseline = den + "baseline"
      num_comparison = num + "comparison"
      den_comparison = den + "comparison"

      keep = (joined[den_baseline] + joined[den_comparison]) != 0

      if not any(keep):
        raise ValueError("MH weight is undefined.")

      weights = 1 / (joined[den_baseline][keep] + joined[den_comparison][keep])
      top = (joined[num_comparison][keep] * joined[den_baseline][keep] *
             weights).sum()
      bot = (joined[num_baseline][keep] * joined[den_comparison][keep] *
             weights).sum()

      if top == 0 or bot == 0:
        raise ValueError("MH rate is zero.")

      stat = top / bot
      return stat

    super(MH, self).__init__(condition_column, baseline_key, name, _compare,
                             include_base)
