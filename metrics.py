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

"""Metrics for //ads/metrics/lib/meterstick."""

from __future__ import division

import google3
import numpy as np
import pandas as pd


class Metric(object):
  """Base class for metrics.

  Metrics wrap various calculations on Pandas DataFrames.

  Attributes:
    name: A string name for the metric's result (eg, CTR for
      Clicks/Impressions).
    _fn: A function to calculate the metric on a dataframe with weights.
    metric_type: A string indicating whether the metric is a scalar (eg, mean)
      or dataframe (eg, a distribution over a categorical variable). Scalar
      metrics return scalars, dataframe metrics return indexed DataFrames.
  """

  def __init__(self, name, fn, metric_type):
    self._name = name
    self._fn = fn
    # Metrics can be scalar metrics (eg, ratio metric that return a number)
    # or dataframe metrics (eg, distribution metric that returns a distribution)
    self._metric_type = metric_type

  @property
  def name(self):
    """Gets metric name."""
    return self._name

  @property
  def metric_type(self):
    """Gets metric type."""
    return self._metric_type

  def __call__(self, data, weights):
    """Calculates the metric for the given dataframe.

    Args:
      data: A pandas dataframe.
      weights: A numpy array of weights.

    Returns:
      A pandas dataframe with an estimate of the metric.
    """

    return self._fn(data, weights) if weights.sum() > 0 else np.nan


### Utility functions
def _weighted_sum(values, weights):
  return values.dot(weights)


def _weighted_mean(values, weights):
  return np.average(values, weights=weights)


def _weighted_variance(values, weights, ddof):
  mean = _weighted_mean(values, weights)
  return ((values - mean)**2).dot(weights) / (weights.sum() - ddof)


def _weighted_correlation(values1, values2, weights):
  mean1 = _weighted_mean(values1, weights)
  mean2 = _weighted_mean(values2, weights)

  cross = ((values1 - mean1) * (values2 - mean2)).dot(weights)
  sum_squares1 = ((values1 - mean1)**2).dot(weights)
  sum_squares2 = ((values2 - mean2)**2).dot(weights)

  return cross / np.sqrt(sum_squares1 * sum_squares2)


### Classes
class Distribution(Metric):
  """Ratio estimator."""

  def __init__(self, metric, dimensions, name=None):
    """Initializes distribution estimator.

    Args:
      metric: Thing to calculate
      dimensions: Dimensions to distribute things over.
      name: A string for the column name of results.
    """

    def _calculate(data, weights):
      """Calculates distribution metric."""
      total = 1.0 * _weighted_sum(data[metric].values, weights)

      dimension_tuples = pd.lib.fast_zip([data[ii].values for ii in dimensions])
      factors, keys = pd.factorize(dimension_tuples)
      results = np.zeros(len(keys))

      for ii in xrange(len(keys)):
        results[ii] = _weighted_sum(data[metric].values,
                                    weights * (factors == ii)) / total

      output = pd.DataFrame(results,
                            index=pd.MultiIndex.from_tuples(keys,
                                                            names=dimensions),
                            columns=[""])
      return output

    if name is None:
      name = "{} Distribution".format(metric)

    super(Distribution, self).__init__(name, _calculate, "dataframe")


class CumulativeDistribution(Metric):
  """Ratio estimator."""

  def __init__(self, metric, dimensions, ascending=True, name=None):
    """Initializes distribution estimator.

    Args:
      metric: Thing to calculate
      dimensions: Dimensions to distribute things over.
      ascending: list of bools to pass to pandas.sort_index that say to sort
        each dimension ascending or descending.
      name: A string for the column name of results.
    """

    def _calculate(data, weights):
      """Calculates cumulative distribution metric."""
      total = 1.0 * _weighted_sum(data[metric].values, weights)

      dimension_tuples = pd.lib.fast_zip([data[ii].values for ii in dimensions])
      factors, keys = pd.factorize(dimension_tuples, sort=True)
      results = np.zeros(len(keys))

      for ii in xrange(len(keys)):
        results[ii] = _weighted_sum(data[metric].values,
                                    weights * (factors == ii)) / total

      output = pd.DataFrame(results,
                            index=pd.MultiIndex.from_tuples(keys,
                                                            names=dimensions),
                            columns=[""])
      output = output.sort_index(ascending=ascending).cumsum()
      return output

    if name is None:
      name = "{} Cumulative Distribution".format(metric)

    super(CumulativeDistribution, self).__init__(name, _calculate, "dataframe")


class Ratio(Metric):
  """Ratio estimator."""

  def __init__(self, numerator, denominator, name=None):
    """Initializes ratio estimator.

    Args:
      numerator: A string representing the numerator variable.
      denominator: A string representing the denominator variable.
      name: A string for the column name of results.
    """

    def _calculate(data, weights):
      # Follow's numpy's divide-by-zero behavior.
      return _weighted_sum(data[numerator].values, weights) / _weighted_sum(
          data[denominator].values, weights)

    if name is None:
      name = "{}/{}".format(numerator, denominator)

    super(Ratio, self).__init__(name, _calculate, "scalar")

    self._numerator = numerator
    self._denominator = denominator

  # We provide numerator and denominator methods because the
  # Mantel-Haenszel estimator needs acess to these fields.

  def numerator(self):
    return self._numerator

  def denominator(self):
    return self._denominator


class Sum(Metric):
  """Sum estimator."""

  def __init__(self, variable, name=None):
    """Initializes sum estimator.

    Args:
      variable: A string representing the variable to sum.
      name: A string for the column name of results.
    """

    def _calculate(data, weights):
      return _weighted_sum(data[variable].values, weights)

    if name is None:
      name = "sum({})".format(variable)

    super(Sum, self).__init__(name, _calculate, "scalar")


class Mean(Metric):
  """Mean estimator."""

  def __init__(self, variable, name=None):
    """Initializes mean estimator.

    Args:
      variable: A string representing the variable to average.
      name: A string for the column name of results.
    """

    def _calculate(data, weights):
      return _weighted_mean(data[variable].values, weights)

    if name is None:
      name = "mean({})".format(variable)

    super(Mean, self).__init__(name, _calculate, "scalar")


class WeightedMean(Metric):
  """Weighted mean estimator."""

  def __init__(self, variable, weight_variable, name=None):
    """Initializes weighted mean estimator.

    Args:
      variable: A string representing the variable to average.
      weight_variable: A string representing the weight variable.
      name: A string for the column name of results.
    """

    def _calculate(data, weights):
      return _weighted_mean(data[variable].values,
                            weights * data[weight_variable].values)

    if name is None:
      name = "{}_weighted_mean({})".format(weight_variable, variable)

    super(WeightedMean, self).__init__(name, _calculate, "scalar")


class Quantile(Metric):
  """Quantile estimator."""

  def __init__(self, variable, quantile, name=None):
    """Initializes quantile estimator.

    Args:
      variable: A string representing the variable to _calculate the quantile.
      quantile: The quantile to be _calculated (range is [0,1]).
      name: A string for the column name of results.
    """

    def _calculate(data, weights):
      """Calculates the quantile for a weighted array.

      Args:
        data: A pandas dataframe
        weights: A numpy array of weights.

      Returns:
        The weighted quantile of data[variable].
      """

      values = data[variable].values

      indices = np.argsort(values)

      if quantile > 0.5:
        # Because we're interating through the indices we should
        # choose the presumably quicker side to start.
        local_quantile = 1 - quantile
        previous_ii = indices[-1]
        indices = reversed(indices)
      else:
        local_quantile = quantile
        previous_ii = indices[0]

      threshold = weights.sum() * local_quantile
      previous_accumulated_weight = 0

      # We follow the usual convention for the median: let
      # lower=floor(threshold) and upper = ceil(threshold). If
      # lower==upper we return values[lower] otherwise we return the
      # average of values[lower] and values[upper].

      for ii in indices:
        if weights[ii] > 0:
          if previous_accumulated_weight > threshold:
            # The previous element passed the threshold itself and thus
            # values[lower]==values[upper].
            return values[previous_ii]
          elif previous_accumulated_weight == threshold:
            return (values[ii] + values[previous_ii]) / 2
          else:
            previous_ii = ii
            previous_accumulated_weight += weights[ii]

    if name is None:
      name = "quantile({}, {:.2f})".format(variable, quantile)

    super(Quantile, self).__init__(name, _calculate, "scalar")


class Variance(Metric):
  """Variance estimator.
  """

  def __init__(self, variable, unbiased=True, name=None):
    """Initializes variance estimator.

    Args:
      variable: A string for the variable column.
      unbiased: A boolean; if true the unbiased estimate is used,
        otherwise the unbiased MLE estimator is used.
      name: A string for the column name of results.
    """

    ddof = 1 if unbiased else 0

    def _calculate(data, weights):
      return _weighted_variance(data[variable].values, weights, ddof)

    if name is None:
      name = "var({})".format(variable)

    super(Variance, self).__init__(name, _calculate, "scalar")


class StandardDeviation(Metric):
  """Standard Deviation estimator.
  """

  def __init__(self, variable, unbiased=True, name=None):
    """Initializes standard deviation estimator.

    Args:
      variable: A string for the variable column.
      unbiased: A boolean; if true the unbiased estimate is used,
        otherwise the unbiased MLE estimator is used.
      name: A string for the column name of results.

    """

    ddof = 1 if unbiased else 0

    def _calculate(data, weights):
      return np.sqrt(_weighted_variance(data[variable].values, weights, ddof))

    if name is None:
      name = "sd({})".format(variable)

    super(StandardDeviation, self).__init__(name, _calculate, "scalar")


class CV(Metric):
  """Coefficient of variation estimator.
  """

  def __init__(self, variable, unbiased=True, name=None):
    """Initializes CV estimator.

    Args:
      variable: A string for the variable column.
      unbiased: A boolean; if true the unbiased estimate for the standard
        deviation is used, otherwise the unbiased MLE estimator is used.
      name: A string for the column name of results.
    """

    ddof = 1 if unbiased else 0

    def _calculate(data, weights):
      return np.sqrt(_weighted_variance(data[variable].values, weights,
                                        ddof)) / _weighted_mean(
                                            data[variable].values, weights)

    if name is None:
      name = "cv({})".format(variable)

    super(CV, self).__init__(name, _calculate, "scalar")


class Correlation(Metric):
  """Correlation estimator.
  """

  def __init__(self, var1, var2, name=None):
    """Initializes correlation estimator.

    Args:
      var1: A string for the first variable column.
      var2: A string for the second variable column.
      name: A string for the column name of results.
    """

    def _calculate(data, weights):
      return _weighted_correlation(data[var1].values, data[var2].values,
                                   weights)

    if name is None:
      name = "corr({}, {})".format(var1, var2)

    super(Correlation, self).__init__(name, _calculate, "scalar")
