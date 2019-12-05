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

"""Metrics for //ads/metrics/lib/meterstick."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Callable, List, Optional, Union

import attr
from meterstick import pdutils
import numpy as np
import pandas as pd
import six


def _get_name(obj):
  """Gets name and function for obj (either a Metric or a constant)."""
  return obj.name if isinstance(obj, Metric) else str(obj)


@attr.s()
class Metric(object):
  """Base class for metrics.

  Each metric represents a calculation to perform on a pandas
  DataFrame. There are three types of indexing columns you can specify. See
  screenshot/9z0GDaP1FEy for an intuitive display.
  1. split_vars: Think them as part of the data, not part of the metric. Metric
    is calculated for each slice after splitting.
  2. metric_idx: Only exists when you specify argument fn. If fn returns a
    pd.Series, metric_idx is the indexing columns in the Series. If fn returns a
    pd.DataFrame, it's the df.columns. For any column not named in metric_idx,
    it will be renamed to MetricIndex0, MetricIndex1 and so on. By using
    metric_idx, you can use cross-row information in a slice. For example, for
    every slice, you can fit a linear regression and return the fitted values
    for every row.
  3. over: Only exists when you specify argument fn and fn returns a
    pd.DataFrame. In this case, over will be df.index. Over column(s) are always
    displayed vertically. Namely, they're different from split_vars in that they
    are part of the metric, but they are always displayed vertically just like
    split_vars columns. See DistributionMetric() below for a typical use case.

  Attributes:
    name: A string name for the metric's result (e.g., sum(Queries)).
    _parents: A list of Metric objects that specify the parents
      of the current metric object. For example, if the
      metric is defined as Sum("Clicks") / Sum("Impressions"),
      then the parents of this metric are Sum("Clicks") and
      Sum("Impressions").
    index: A Pandas Index which will be used as the index for
      the output.
    split_vars: A list of the names of indices.
    fn: A function to calculate metric. It may return a single value, a
      pd.Series, or a pd.DataFrame.
    metric_idx: A list of the column name(s) when a pd.Series or a pd.DataFrame
      is returned by fn. See the description above for details.
    over:  If a pd.DataFrame with indexing column(s) is returned by fn, this is
      a list of the column name(s). See the description above for details.
  """
  name = attr.ib(type=str)
  _parents = attr.ib(factory=list, type=List["Metric"])
  index = attr.ib(default=None, type=Optional[pd.Index])
  fn = attr.ib(
      default=None,
      type=Optional[
          Callable[[pd.DataFrame], Union[float, pd.Series, pd.DataFrame]]])
  metric_idx = attr.ib(init=False, factory=list, type=List[str])
  over = attr.ib(init=False, factory=list, type=List[str])

  def compute(self, data):
    """Computes metric for given data.

    Every Metric must implement this method.

    Args:
      data: A Pandas DataFrame.

    Returns:
      A Pandas Series or a scalar containing the metric value(s).

    Raises:
      ValueError: When there's no function defined to calculate the metric.
      ValueError: When split_vars but the output isn't a pd.Series after all
        processing.
    """

    def _fill_index(idx, base_name):
      """Fills no-name level in idx with numbered base_name."""
      return [base_name + str(i) if not n else n for i, n in enumerate(idx)]

    if not self.fn:
      raise ValueError("No function found to calculate metric %s." % self.name)

    if self.split_vars:
      output = self._group(data).apply(self.fn)
      # The output must be a pd.Series or a number. When there are split_vars,
      # it can't be a number, so we need to transform the output to a pd.Series
      # if it's not he case.
      if isinstance(output, pd.DataFrame):
        # The index = split_vars + optional over columns.
        if len(output.index.names) > len(self.split_vars):
          self.over = _fill_index(output.index.names[len(self.split_vars):],
                                  "OverColumn")
          output.index.names = self.split_vars + self.over

        self.metric_idx = _fill_index(output.columns.names, "MetricIndex")
        output.columns.names = self.metric_idx
        # Transform to a pd.Series for easier handling.
        output = output.stack(self.metric_idx)
      if not isinstance(output, pd.Series):
        raise ValueError("Output must be a pd.Series.")
    else:
      output = self.fn(data)
      if isinstance(output, pd.Series):
        fn_idx = _fill_index(output.index.names, "MetricIndex")
        output.index.names = fn_idx
        self.metric_idx = fn_idx
      elif isinstance(output, pd.DataFrame):
        self.over = _fill_index(output.index.names, "OverColumn")
        output.index.names = self.over
        self.metric_idx = _fill_index(output.columns.names, "MetricIndex")
        output.columns.names = self.metric_idx
        # Transform to a pd.Series for easier handling.
        output = output.stack(self.metric_idx)

    return output

  def precalculate(self, data, split_index):
    """Sets the index for the metric result.

    Args:
      data: Pandas DataFrame
      split_index: Pandas Index representing the variables in split_by.
    """
    # update split_vars for metric
    self.split_vars = [] if split_index is None else split_index.names
    # set the index if not already set
    if self.index is None:
      self.index = split_index
    # repeat the process for parent metrics
    for parent in self._parents:
      if isinstance(parent, Metric):
        parent.precalculate(data, split_index)

  def _group(self, data):
    """Groups data by split_vars if split_vars is set.

    Pandas does not allow grouping by 0 variables, i.e.,
    .groupby([]). This function provides a unified API
    to handle both the case where we group by 0 variables
    and the case where we group by 1 or more variables.

    Args:
      data: A Pandas DataFrame or Series

    Returns:
      A Pandas NDFrameGroupBy object, if self.split_vars is set.
      Otherwise, returns an ungrouped NDFrame.
    """
    if self.split_vars:
      return data.groupby(level=self.split_vars, sort=False)
    else:
      return data

  def __call__(self, data):
    output = self.compute(data)
    if isinstance(output, pd.Series):
      if not self.fn:
        # When self.fn, the indexing is handled in Metric.compute() already.
        output = output.reindex(self.index)
      output.name = ""
    return output

  def __add__(self, other):
    return CompositeMetric(lambda x, y: x + y, "{} + {}", [self, other])

  def __radd__(self, other):
    return CompositeMetric(lambda x, y: x + y, "{} + {}", [other, self])

  def __sub__(self, other):
    return CompositeMetric(lambda x, y: x - y, "{} - {}", [self, other])

  def __rsub__(self, other):
    return CompositeMetric(lambda x, y: x - y, "{} - {}", [other, self])

  def __mul__(self, other):
    return CompositeMetric(lambda x, y: x * y, "{} * {}", [self, other])

  def __rmul__(self, other):
    return CompositeMetric(lambda x, y: x * y, "{} * {}", [other, self])

  def __neg__(self):
    metric = -1 * self
    metric.name = "-{}".format(self.name)
    return metric

  def __div__(self, other):
    return CompositeMetric(lambda x, y: x / y, "{} / {}", [self, other])

  def __truediv__(self, other):
    return self.__div__(other)

  def __rdiv__(self, other):
    return CompositeMetric(lambda x, y: x / y, "{} / {}", [other, self])

  def __rtruediv__(self, other):
    return self.__rdiv__(other)

  def __pow__(self, other):
    return CompositeMetric(lambda x, y: x ** y, "{} ^ {}", [self, other])

  def __rpow__(self, other):
    return CompositeMetric(lambda x, y: x ** y, "{} ^ {}", [other, self])


class CompositeMetric(Metric):
  """Class for metrics formed by composing two metrics.

  Attributes:
    name: A string name for the metric's result (e.g., CTR for
      Sum(Clicks) / Sum(Impressions)).
    index: A Pandas Index which will be used as the index for
      the output.
  """

  def __init__(self, op, name_format, parents, index=None):
    """Initializes Metric.

    Args:
      op: Binary operation that defines the composite metric.
      name_format: A format string for the metric's result.
      parents: A list of Metric objects that specify the parents
        of the current metric object. For example, if the
        metric is defined as Sum("Clicks") / Sum("Impressions"),
        then the parents of this metric are Sum("Clicks") and
        Sum("Impressions").
      index: A Pandas Index object which will be used as the
        index for the output.
    """
    name = name_format.format(_get_name(parents[0]), _get_name(parents[1]))
    super(CompositeMetric, self).__init__(name, parents, index)
    self._op = op

  def compute(self, data):
    """Calculates composite metric based on parents.

    Args:
      data: A Pandas DataFrame containing the data.

    Returns:
      A Pandas Series or constant value representing the value.
    """
    a = (self._parents[0].compute(data)
         if isinstance(self._parents[0], Metric)
         else self._parents[0])
    b = (self._parents[1].compute(data)
         if isinstance(self._parents[1], Metric)
         else self._parents[1])
    return self._op(a, b)


### Classes
class Count(Metric):
  """Initializes count estimator.

  Args:
    variable: A string representing the variable to count.
    name: A string for the column name of results.
  """

  def __init__(self, variable, name=None):
    if name is None:
      name = "count({})".format(variable)
    super(Count, self).__init__(name)

    self._var = variable

  def compute(self, data):
    return self._group(data)[self._var].count()


class Sum(Metric):
  """Sum estimator."""

  def __init__(self, variable, name=None):
    """Initializes sum estimator.

    Args:
      variable: A string representing the variable to sum.
      name: A string for the column name of results.
    """
    if name is None:
      name = "sum({})".format(variable)
    super(Sum, self).__init__(name)

    self._var = variable

  def compute(self, data):
    return self._group(data)[self._var].sum(min_count=1)


class Mean(Metric):
  """Mean estimator."""

  def __init__(self, variable, name=None):
    """Initializes mean estimator.

    Args:
      variable: A string representing the variable to average.
      name: A string for the column name of results.
    """
    if name is None:
      name = "mean({})".format(variable)
    super(Mean, self).__init__(name)

    self._var = variable

  def compute(self, data):
    return self._group(data)[self._var].mean()


class WeightedMean(Metric):
  """Weighted mean estimator."""

  def __init__(self, variable, weight_variable, name=None):
    """Initializes weighted mean estimator.

    Args:
      variable: A string representing the variable to average.
      weight_variable: A string representing the weight variable.
      name: A string for the column name of results.
    """
    if name is None:
      name = "{}_weighted_mean({})".format(weight_variable, variable)
    super(WeightedMean, self).__init__(name)

    self._var = variable
    self._weight = weight_variable

  def compute(self, data):
    data["weighted_var"] = data[self._var] * data[self._weight]
    data_grouped = self._group(data)
    return (data_grouped["weighted_var"].sum(min_count=1) /
            data_grouped[self._weight].sum(min_count=1))


class Quantile(Metric):
  """Quantile estimator."""

  def __init__(self, variable, quantile, name=None):
    """Initializes quantile estimator.

    Args:
      variable: A string representing the variable to _calculate the quantile.
      quantile: The quantile to be _calculated (range is [0,1]).
      name: A string for the column name of results.
    """
    if name is None:
      name = "quantile({}, {:.2f})".format(variable, quantile)
    super(Quantile, self).__init__(name)

    self._var = variable
    self._quantile = quantile

  def compute(self, data):
    return self._group(data)[self._var].quantile(self._quantile)


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
    if name is None:
      name = "var({})".format(variable)
    super(Variance, self).__init__(name)

    self._var = variable
    self._ddof = 1 if unbiased else 0

  def compute(self, data):
    return self._group(data)[self._var].var(ddof=self._ddof)


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
    if name is None:
      name = "sd({})".format(variable)
    super(StandardDeviation, self).__init__(name)

    self._var = variable
    self._ddof = 1 if unbiased else 0

  def compute(self, data):
    return self._group(data)[self._var].std(ddof=self._ddof)


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
    if name is None:
      name = "cv({})".format(variable)
    super(CV, self).__init__(name)

    self._var = variable
    self._ddof = 1 if unbiased else 0

  def compute(self, data):
    var_grouped = self._group(data)[self._var]
    return var_grouped.std(ddof=self._ddof) / var_grouped.mean()


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
    if name is None:
      name = "corr({}, {})".format(var1, var2)
    super(Correlation, self).__init__(name)

    self._var1 = var1
    self._var2 = var2

  def compute(self, data):
    return self._group(data)[self._var1].corr(data[self._var2])


class WeightedCorrelation(Metric):
  """Weighted correlation estimator."""

  def __init__(self, var1, var2, weight_variable, name=None):
    """Initializes weighted correlation estimator.

    Args:
      var1: A string for the first variable column.
      var2: A string for the second variable column.
      weight_variable: A string representing the weight variable.
      name: A string for the column name of results.
    """
    if name is None:
      name = "{}_weighted_corr({}, {})".format(weight_variable, var1, var2)
    super(WeightedCorrelation, self).__init__(name)

    self._var1 = var1
    self._var2 = var2
    self._weight = weight_variable

  def _weighted_corr(self, df):
    """Computes weighted correlation.

    Args:
      df: A Pandas data frame.

    Returns:
      The weighted correlation batween self._var1 and self._var2 based on
      self._weight.
    """
    wts = df[self._weight]
    mean1 = (df[self._var1] * wts).sum(min_count=1) / wts.sum(min_count=1)
    mean2 = (df[self._var2] * wts).sum(min_count=1) / wts.sum(min_count=1)

    cross = (wts * (df[self._var1] - mean1) * (df[self._var2] - mean2)).sum(
        min_count=1
    )
    ss1 = (wts * (df[self._var1] - mean1) ** 2).sum(min_count=1)
    ss2 = (wts * (df[self._var2] - mean2) ** 2).sum(min_count=1)

    return cross / np.sqrt(ss1 * ss2)

  def compute(self, data):
    if self.split_vars:
      return self._group(data).apply(self._weighted_corr)
    else:
      return self._weighted_corr(data)


# Define Ratio estimator for backwards compatibility.
def Ratio(numerator, denominator, name=None):  # pylint: disable=invalid-name
  """Constructs a Ratio estimator."""

  metric = Sum(numerator) / Sum(denominator)
  metric.numerator = numerator  # pylint: disable=g-missing-from-attributes
  metric.denominator = denominator  # pylint: disable=g-missing-from-attributes

  if name is not None:
    metric.name = name
  else:
    # for compatibility with older versions of Meterstick
    metric.name = "%s / %s" % (numerator, denominator)

  return metric


## Distribution Metrics


def _fillna(dist):
  """Fills NaN values in a distribution with zeros.

  Args:
    dist: A Pandas Series representing a distribution.

  Returns:
    A Pandas Series with the NaN values replaced by zeros, only if the
    distribution is well-defined (i.e., its values do not sum to 0).
  """
  return dist.fillna(0.) if dist.sum() > 0 else dist


class DistributionMetric(Metric):
  """Base class for distribution metrics.

  Attributes:
    name: A string name for the metric's result (e.g., "X Distribution").
    over: A list of strings indicating what variable(s) to calculate the
      distribution over.
    over_index: A pd.Index of the over columns.
  """

  def __init__(self, name, of, over, expand=False,
               sort=True, ascending=True, normalize=True):
    """Initializes DistributionMetric with relevant fields.

    Args:
      name: A string name for the distribution metric result.
      of: A string indicating what variable to calculate the distribution of.
      over: A string or a list of strings indicating what variable(s)
        to calculate the distribution over.
      expand: A boolean indicating whether or not to expand to have the full
        product of all possible levels in all "over" variables.
      sort: A boolean indicating whether or not to sort the levels of the
        "over" variable.
      ascending: A boolean indicating whether the levels in the "over"
        variables should be sorted in ascending or descending order.
      normalize: A boolean indicating whether the distribution should be
        scaled to sum to one.
    """
    super(DistributionMetric, self).__init__(name)

    # ensure that the "over" variables are a list
    if isinstance(over, six.string_types):
      over = [over]
    self._of = of
    self.over = over
    self._expand = expand
    self._sort = sort
    self._ascending = ascending
    self._normalize = normalize
    self.name = name

  def precalculate(self, data, split_index):
    """Precalculates the index for the distribution metric.

    Args:
      data: A Pandas DataFrame
      split_index: A Pandas Index for the variables on which the analysis
        is being split.
    """
    # Get index for the "over" variables.
    over_index = pdutils.index_product_from_vars(data, self.over, self._expand)

    # Sort the index, as appropriate
    if self._sort:
      over_index = over_index.sort_values(ascending=self._ascending)
    self.over_index = over_index

    # Take the product of the two indexes.
    self.index = pdutils.index_product(split_index, over_index)

    super(DistributionMetric, self).precalculate(data, split_index)

  def compute(self, data):
    """Calculates distribution of self._of over the columns in self.over.

    Args:
      data: A Pandas DataFrame

    Returns:
      A Pandas Series, indexed by the split_vars and the over var,
      representing the distribution.
    """
    if self.split_vars:
      group_vars = self.split_vars + self.over
      dist = data.groupby(by=group_vars, sort=False)[self._of].sum(
          min_count=1
      )

      if self._normalize:
        dist = (dist.groupby(level=self.split_vars, sort=False).
                transform(lambda x: x / x.sum(min_count=1)))
    else:
      dist = data.groupby(self.over, sort=False)[self._of].sum(
          min_count=1
      )

      if self._normalize:
        dist /= data[self._of].sum(min_count=1)

    # reindex the distribution (this will possibly sort the levels)
    dist = dist.reindex(self.index)

    return self._group(dist).transform(_fillna)


class Distribution(DistributionMetric):
  """Distribution estimator."""

  def __init__(self, of, over, expand=False, sort=True,
               normalize=True, name=None):
    """Initializes distribution estimator.

    Args:
      of: A string indicating what variable to calculate the distribution of.
      over: A string or a list of strings indicating what variable(s)
        to calculate the distribution over.
      expand: A boolean indicating whether or not to expand to have the full
        product of all possible levels in all "over" variables.
      sort: A boolean indicating whether or not to sort the levels of the
        "over" variable.
      normalize: A boolean indicating whether the distribution should be
        scaled to sum to one.
      name: A string for the column name of results.
    """
    if name is None:
      name = "{} Distribution".format(of)
    super(Distribution, self).__init__(name, of, over,
                                       expand=expand, sort=sort,
                                       normalize=normalize)


class CumulativeDistribution(DistributionMetric):
  """Ratio estimator."""

  def __init__(self, of, over, expand=False, ascending=True,
               normalize=True, name=None):
    """Initializes distribution estimator.

    Args:
      of: A string indicating what variable to calculate the distribution of.
      over: A string or a list of strings indicating what variable(s)
        to calculate the distribution over.
      expand: A boolean indicating whether or not to expand to have the full
        product of all possible levels in all "over" variables.
      ascending: A boolean indicating whether the levels in the "over"
        variables should be sorted in ascending or descending order.
      normalize: A boolean indicating whether the distribution should be
        scaled to sum to one.
      name: A string for the column name of results.
    """
    if name is None:
      name = "{} Cumulative Distribution".format(of)
    super(CumulativeDistribution, self).__init__(name, of, over,
                                                 expand=expand,
                                                 ascending=ascending,
                                                 normalize=normalize)

  def compute(self, data):
    """Computes cumulative distribution of data over variables.

    The cumulative distribution is computed such that the numbers in each
    slice increase from 0 to 1.

    Args:
      data: A Pandas DataFrame.

    Returns:
      A Pandas Series, indexed by the variables in self.split_vars, followed
      by the variables in self.over. The levels of the variables in
      self.over are sorted in ascending order if self._ascending is True.
    """
    dist = super(CumulativeDistribution, self).compute(data)
    return self._group(dist).cumsum()
