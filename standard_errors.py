# Lint as: python2, python3
"""Standard Errors for //ads/metrics/lib/meterstick."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from meterstick import pdutils
import numpy as np
import pandas as pd
import scipy.stats
from six.moves import range


class StandardErrorMethod(object):
  """Base class for standard error methods."""

  def __init__(self, name, confidence=None, unit=None, flat_index=True):
    """Initializes StandardErrorMethod.

    Args:
      name: A string which will be used in the column name of results.
      confidence: Optional, the level of the confidence interval, must be in
        (0,1).
      unit: A string denoting the dataframe column that contains the
        sampling unit (such that observations are independent across units).
      flat_index: If collapse the result's multiindex columns to flat-index
        column.

    Raises:
      ValueError: Confidence is not in (0,1).
    """

    self.name = name
    self.confidence = confidence
    self.unit = unit
    self.flat_index = flat_index

    if confidence is not None and not 0 < confidence < 1:
      raise ValueError("Confidence must be in (0,1).")

  def calculate_ci_lower_upper(self, df, confidence, stderr, value_name,
                               lower_name, upper_name):
    """Add columns for the lower and upper bounds of CI to df.

    Args:
      df: A DataFrame that has a column of center values.
      confidence: The level of the confidence interval, must be in (0,1).
      stderr: A pd.Series in the same length of df. It's the row-wise stderr for
        the values in df.
      value_name: The name of center value column in df.
      lower_name: The name for the added column of CI lower bound.
      upper_name: The name for the added column of CI upper bound.

    Returns:
      A DataFrame that is df plus two columns for CI bounds.
    """
    degrees_of_freedom = self._degrees_of_freedom()
    multiplier = scipy.stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
    # df.loc[:, lower_name] = ... might give a warning
    # "indexing past lexsort depth may impact performance".
    df[lower_name] = df[value_name] - multiplier * stderr
    df[upper_name] = df[value_name] + multiplier * stderr
    return df

  def __call__(self, df, metric, comparison=None):
    """Calculates standard errors for metric, with possibly a comparison.

    This function is a wrapper around the actual logic to compute the
    standard errors, which is contained in the self.compute function.

    Args:
      df: A pandas DataFrame.
      metric: A Metric object.
      comparison: Optionally a Comparison object.

    Returns:
      A pandas DataFrame with the estimate and either standard errors
      or, if confidence is provided, confidence intervals.
    """

    # calculate estimates and standard error
    if comparison is None:
      base_name = ""
      est = pd.Series(metric(df), name=base_name)
      stderr = pd.Series(self.compute(df, metric), name=base_name)
    else:
      base_name = comparison.name
      est = pd.Series(comparison(df, metric),
                      name=base_name)
      # Calculate standard errors (unit is the first level in the index)
      stderr = pd.Series(self.compute(df, metric, comparison),
                         name=base_name)

    # Determine whether CIs are needed or just SEs.
    if self.confidence is None:
      stderr_name = " ".join([base_name, self.name, "SE"]).strip()
      return pd.concat([est, stderr], axis=1,
                       keys=(base_name, stderr_name))
    else:
      res = pd.DataFrame(est)
      lower_name = " ".join([base_name, self.name, "CI-lower"]).strip()
      upper_name = " ".join([base_name, self.name, "CI-upper"]).strip()
      res = self.calculate_ci_lower_upper(res, self.confidence, stderr,
                                          base_name, lower_name, upper_name)

      return res

  def compute(self, data, metric, comparison=None):
    """Computes the standard errors.

    Every standard error method must implement this method.

    Args:
      data: A pandas DataFrame.
      metric: A Metric object.
      comparison: Optionally a Comparison object.

    Returns:
      A Pandas Series or a scalar containing the standard error(s).
    """
    raise NotImplementedError()

  def precalculate(self, df):
    """Precomputes the buckets for the Jackknife object.

    Args:
      df: A pandas dataframe.
    """

    if self.unit is None:
      self._buckets = list(range(len(df)))
    else:
      self._buckets = df[self.unit].unique()

  def _degrees_of_freedom(self):
    """Calculates the degrees of freedom for SE's T distribution.

    Returns:
      The degrees of freedom for the SE's T distribution.
    """
    return len(self._buckets) - 1


class Jackknife(StandardErrorMethod):
  """Class for Jackknife estimates of standard errors.

  Attributes:
    unit: A string for the column whose values are the levels which define
      the jackknife buckets.
    flat_index: If collapse the result's multiindex columns to flat-index
      column.
  """

  def __init__(self, unit=None, confidence=None, flat_index=True):
    """Initializes the Jackknife.

    Args:
      unit: The variable name specifying the column to use as Jackknife units.
      confidence: Optional, the level of the confidence interval, must be in
        (0,1).
      flat_index: If collapse the result's multiindex columns to flat-index
        column.
    """
    super(Jackknife, self).__init__("Jackknife", confidence, unit, flat_index)

  def compute(self, df, metric, comparison=None):
    """Calculates Jackknife SEs for metric, optionally with a comparison.

    Args:
      df: A pandas DataFrame. It is assumed that the DataFrame is indexed
          by the condition column and slices.
      metric: A Metric object to calculate.
      comparison: A Comparison object to calculate.

    Returns:
      A pandas Series with the standard errors.
    """

    if comparison is None:
      fn = metric
    else:
      fn = lambda data: comparison(data, metric)

    # If too few buckets, return a correctly shaped dataframe of NaNs.
    if len(self._buckets) < 2:
      return np.nan * fn(df)

    # Get estimates by dropping each bucket, one at a time.
    bucket_estimates = []
    for bucket in self._buckets:

      # Get two dataframes: one containing the data in the bucket, and
      # another containing the rest of the data *excluding* the bucket.
      if self.unit is None:
        df_bucket = pdutils.select_by_position(df, bucket)
        df_rest = df.iloc[list(range(bucket)) +
                          list(range(bucket + 1, len(df)))]
      else:
        df_bucket = df[df[self.unit] == bucket]
        df_rest = df[df[self.unit] != bucket]

      # Keep only the slices and comparisons that appeared in
      # the dropped bucket.
      if comparison is not None or len(metric.split_vars):
        # Store index names because they get lost when we slice the data.
        index_names = df_rest.index.names
        # slice data
        slices = df_bucket.index.unique()
        df_rest = df_rest[df_rest.index.isin(slices)]
        # restore index names
        df_rest.index.names = index_names

      # Calculate estimate on df with bucket dropped and
      # append to bucket_estimates.
      bucket_estimates.append(fn(df_rest))

    # Turn estimates into a dataframe. (This is necessary because some
    # slices may be missing buckets. We need to take this into account
    # when we compute the jackknife standard error.)
    bucket_estimates = pdutils.concat(bucket_estimates, axis=1)

    means = bucket_estimates.mean(axis=1)
    num_buckets = bucket_estimates.count(axis=1)
    rss = (bucket_estimates.subtract(means, axis=0) ** 2).sum(
        axis=1,
        min_count=1
    )
    stderrs = np.sqrt(rss * (1. - 1. / num_buckets))

    return stderrs


class Bootstrap(StandardErrorMethod):
  """Class for Bootstrap estimates of standard errors.

  Attributes:
    num_replicates: A integer for the number of bootstrap replicates.
    unit: A string for the variable representing the level to be resampled.
    flat_index: If collapse the result's multiindex columns to flat-index
      column.
  """

  def __init__(self,
               num_replicates,
               unit=None,
               confidence=None,
               flat_index=True):
    """Initializes the Bootstrap.

    Args:
      num_replicates: The number of bootstrap replicates to compute.
      unit: The variable name for which column to use as Bootstrap units.
      confidence: Optional, the level of the confidence interval, must be in
        (0,1).
      flat_index: If collapse the result's multiindex columns to flat-index
        column.
    """
    self.num_replicates = num_replicates
    super(Bootstrap, self).__init__("Bootstrap", confidence, unit, flat_index)

  def compute(self, df, metric, comparison=None):
    """Calculates bootstrap standard errors for metric with comparison.

    Args:
      df: A pandas DataFrame. It is assumed that the DataFrame is indexed
          by the condition column and slices.
      metric: A Metric object to calculate.
      comparison: A Comparison object to calculate.

    Returns:
      A pandas DataFrame with estimates and bootstrap standard errors.
    """

    if comparison is None:
      fn = metric
    else:
      fn = lambda data: comparison(data, metric)

    replicate_estimates = []
    for _ in range(self.num_replicates):
      buckets_new = np.random.choice(self._buckets, size=len(self._buckets))
      if self.unit is None:
        df_new = pd.concat([pdutils.select_by_position(df, bucket)
                            for bucket in buckets_new])
      else:
        df_new = pd.concat(df[df[self.unit] == bucket]
                           for bucket in buckets_new)
      replicate_estimates.append(fn(df_new))

    # Turn estimates into a dataframe (This is necessary because some
    # slices may be missing buckets. We need to take this into account
    # when we compute the jackknife standard error.)
    replicate_estimates = pdutils.concat(replicate_estimates, axis=1)

    stderrs = replicate_estimates.std(axis=1, ddof=1)

    return stderrs
