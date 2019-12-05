# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Comparisons for //ads/metrics/lib/meterstick."""

from __future__ import division

from meterstick import pdutils
import numpy as np


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

  def __init__(self, condition_column, baseline_key, name,
               include_base=False, sort=True):
    """Initializes the Comparison object.

    Args:
      condition_column: A string denoting the dataframe column to
        compare against (e.g. "Experiment").
      baseline_key: A string denoting the value of the
        condition_column which represents the baseline comparison
        condition (e.g., "Control").
      name: A string denoting the column name for results of the
        comparison.
      include_base: A boolean for whether the base value should be included in
        the list of indices to compare.
      sort: A boolean for whether the conditions should be sorted.
    """
    self.condition_column = condition_column
    self.baseline_key = baseline_key
    self.name = name
    self._include_base = include_base
    self.sort = sort

  def compute(self, data_condition, data_baseline, metric):
    """Computes the change in metric across condition and baseline.

    Every comparison method must implement this method.

    Args:
      data_condition: A Pandas DataFrame containing the data for the
        treatment condition.
      data_baseline: A Pandas DataFrame with the data for the
        baseline condition.
      metric: The metric to be compared.

    Returns:
      A Pandas Series with the results of the comparison.

    Raises:
      NotImplementedError
    """
    raise NotImplementedError()

  def precalculate(self, data):
    """Initializes the factor variable.

    Args:
      data: A pandas dataframe.

    Raises:
      ValueError: The baseline key isn't found.
    """
    conditions = data[self.condition_column].unique()
    if self.sort:
      conditions.sort()

    if (conditions != self.baseline_key).all():
      raise ValueError("Baseline value {} not present in column {}".format(
          self.baseline_key, self.condition_column))

    self.conditions = [cond for cond in conditions
                       if cond != self.baseline_key or self._include_base]

  def __call__(self, data, metric):
    """Calculates the comparison for the metric on dataframe data.

    Args:
      data: A pandas dataframe. It is assumed that the condition column
        and slices are the index of the dataframe, with the condition
        column as the first level. The indexing is handled in core.py.
      metric: A Metric object.

    Returns:
      A pandas series with the comparisons of baseline against every other
      unique condition.
    """
    results = []

    # get dataframe for the baseline
    baseline = pdutils.select_by_label(data, self.baseline_key)

    # get dataframe for each of the conditions
    for cond in self.conditions:
      condition = pdutils.select_by_label(data, cond)
      results.append(self.compute(condition, baseline, metric))

    if results:
      output = pdutils.concat(results, keys=self.conditions,
                              name=self.condition_column)
    else:
      # return dataframe of NaNs with the appropriate number of rows
      output = pdutils.concat(
          [np.nan * self.compute(data, data, metric)] * len(self.conditions),
          keys=self.conditions,
          name=self.condition_column)
    output.name = self.name
    return output


def _make_simple_comparison(fn, name):
  """Creates a class for comparison which is a function of the metric values.

  Args:
    fn: A function of the metric values.
    name: A string for the column names of the results.

  Returns:
    A subclass of Comparison which implements the passed in comparison function.
  """

  class SimpleComparison(Comparison):
    """A comparison which can be represented as a difference between two groups.
    """

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
      super(SimpleComparison, self).__init__(condition_column, baseline_key,
                                             name, include_base)

    def compute(self, data_condition, data_baseline, metric):
      """Calculates the comparison across condition and baseline.

      Args:
        data_condition: A pandas dataframe containing a comparison
          condition.
        data_baseline: A pandas dataframe containing the data for the
          baseline condition.
        metric: A Metric object.

      Returns:
        A Pandas series with the results of the comparison.
      """
      return fn(metric(data_condition), metric(data_baseline))

  return SimpleComparison


def _absolute_difference(x, y):
  return x - y


AbsoluteDifference = _make_simple_comparison(_absolute_difference,
                                             "Absolute Difference")


def _percentage_difference(x, y):
  return 100 * (x - y) / y


PercentageDifference = _make_simple_comparison(_percentage_difference,
                                               "Percentage Difference")


class MH(Comparison):
  """Class for Mantel-Haenszel estimator for comparing ratio metrics."""

  def __init__(self, condition_column, baseline_key, index_var,
               include_base=False):
    """Initializes the MH comparison.

    Args:
      condition_column: A string denoting the column in the DataFrame that
        contains the conditions (e.g. "Experiment").
      baseline_key: A string denoting the name of the condition that
        represents the baseline (e.g., "Control"). All conditions will be
        compared to this baseline condition.
      index_var: A string denoting the column in the DataFrame that contains
        the intermediate level of aggregation (e.g. "AdGroup").
      include_base: A boolean for whether the baseline condition should be
        included in the output.
    """
    super(MH, self).__init__(condition_column, baseline_key,
                             "MH Ratio Percentage Difference", include_base)
    self.index_var = index_var

  def compute(self, data_condition, data_baseline, metric):
    """Calculates the MH comparison across condition and baseline.

    To get the MH ratio, we have to hack the dataframes. First, metric has to be
    defined by Ratio() so it has numerator and denominator properties. Then we
    calculate MH values and assign it to the numerator column, and set the
    denominator column to constant 1.
    The slices with invalid MH weights are dropped automatically.

    Args:
      data_condition: A pandas dataframe containing a comparison condition.
      data_baseline: A pandas dataframe containing the baseline condition.
      metric: A Metric object.

    Returns:
      A Pandas series with the results of the comparison.

    Raises:
      AttributeError: If metric doesn't have numerator or denominator property.
      RuntimeError: If numerator or denominator column not found in data.
    """
    try:
      numer = metric.numerator
      denom = metric.denominator
    except AttributeError:
      raise AttributeError(
          "Numerator or/and denominator not found for metric %s. "
          "For MH calculation, pls use Ratio() to define the metric." %
          metric.name)

    # This assumes split_vars has been set to the data_frame, which is the case
    # if you use Meterstick in the way specified in go/meterstick.
    split_vars = data_condition.index.names
    if split_vars[0] is None:
      grpby_vars = self.index_var
    else:
      grpby_vars = split_vars + [self.index_var]
    data_condition_agg = data_condition.groupby(grpby_vars).sum()
    data_baseline_agg = data_baseline.groupby(grpby_vars).sum()
    data_condition_agg_joined = data_condition_agg.join(
        data_baseline_agg, how="inner",
        rsuffix="_other").reset_index(self.index_var)
    data_baseline_agg_joined = data_baseline_agg.join(
        data_condition_agg, how="inner",
        rsuffix="_other").reset_index(self.index_var)
    denom_other = denom + "_other"
    mh_weights = (data_condition_agg_joined[denom] +
                  data_condition_agg_joined[denom_other])
    data_condition_agg_joined[numer] *= (data_condition_agg_joined[denom_other]
                                         / mh_weights)
    data_baseline_agg_joined[numer] *= (data_baseline_agg_joined[denom_other]
                                        / mh_weights)
    data_condition_agg_joined[denom] = 1
    data_baseline_agg_joined[denom] = 1
    mh_ratio = metric(data_condition_agg_joined) / metric(
        data_baseline_agg_joined)
    return (mh_ratio - 1) * 100
