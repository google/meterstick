# Lint as: python2, python3
"""Core module for //ads/metrics/lib/meterstick."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

import collections
import copy
from typing import List, Union

import attr
from meterstick import comparisons
from meterstick import metrics as metrics_module
from meterstick import pdutils
import numpy as np
import pandas as pd
from six import string_types

def _merge_metrics(row):
  non_empty = [str(r) for r in row if pd.notnull(r)]
  if not non_empty:
    return None
  elif len(non_empty) == 1:
    return non_empty[0]
  else:
    return "::".join(non_empty)


# TODO(dlsun): Remove AnalysisParameters and incorporate the
#   attributes directly into the Analyze object.
@attr.s()
class AnalysisParameters(object):
  """Class for the parameters of a data analysis.

  Attributes:
    metrics: A Metric or a list of Metric objects specifying
      which metric(s) to compute.
    split_index: A Pandas index object.
    split_vars: A variable or a list of variables to split the data upon.
    comparison: A Comparison object which performs the comparison.
    se_method: A function which calculates standard errors.
    sort: If to sort on split_vars.
  """
  metrics = attr.ib(default=None)
  split_index = attr.ib(default=None, type=pd.Index)
  split_vars = attr.ib(factory=list, type=Union[str, List[str], None])
  comparison = attr.ib(default=None)
  se_method = attr.ib(default=None)
  sort = attr.ib(type=bool, default=False)


class Analyze(object):
  """Base analysis class; provides base functionality for data analysis.

  Attributes:
    data: A pandas dataframe.
    parameters: An AnalysisParameters object.
  """

  def __init__(self, data, parameters=None):
    """Initializes Analysis object.

    Args:
      data: A pandas dataframe.
      parameters: An AnalysisParameters object.
    """

    self.data = data.copy()
    self.parameters = parameters or AnalysisParameters()

  def where(self, query):
    """Subsets the data according to a query.

    Args:
     query: A string which evaluates to an array of Booleans.

    Returns:
      An Analysis object.

    Raises:
      ValueError: Query does not evaluate to an array of Booleans.
    """

    query_results = self.data.eval(query).values
    if query_results.dtype == np.bool:
      self.data = self.data[query_results]
    else:
      raise ValueError("The query (%s) does not evalulate to "
                       "an array of Booleans." % query)

    return self

  def split_by(self, split_vars, expand=False, sort=True):
    """Splits the analysis by categorical variables.

    Args:
      split_vars: A string or a list of strings representing variables
        to split upon.
      expand: A boolean indicating whether the index that is created
        should contain the full expanded product of all possible
        combinations of levels of the variables in split_vars.
        Otherwise, the index will only contain combinations that
        were actually observed in the data set.
      sort: A boolean indicating whether or not the levels in each
        slice should be sorted.

    Returns:
      An Analysis object with the split_vars attribute set to input.

    Raises:
      ValueError: Split variables are already defined.
      TypeError: Split variable is not a string or list of strings.
    """

    if self.parameters.split_vars:
      raise ValueError("Split variables are already defined.")

    if isinstance(split_vars, string_types):
      self.parameters.split_vars = [split_vars]
    else:
      try:
        all_strings = all(isinstance(var, string_types) for var in split_vars)
      except KeyError:
        raise TypeError("Split variable is not a string or list of strings.")
      if not all_strings:
        raise TypeError("Split variable is not a string or list of strings.")
      self.parameters.split_vars = split_vars

    # Determine the index for split_vars
    self.parameters.split_index = pdutils.index_product_from_vars(
        self.data, self.parameters.split_vars, expand)
    self.parameters.sort = sort

    return self

  def relative_to(self, comparison, sort=True):
    """Specifies the comparison for the analysis.

    Args:
      comparison: A Comparison object.
      sort: Boolean indicating whether to sort the conditions by name.

    Returns:
      An Analysis object with the comparison attribute set to the input.

    Raises:
      ValueError: Comparison is already defined.
    """

    if self.parameters.comparison is not None:
      raise ValueError("Comparison is already defined.")

    comparison.sort = sort
    self.parameters.comparison = copy.deepcopy(comparison)

    return self

  def with_standard_errors(self, method):
    """Specifies standard error method for analysis.

    Args:
      method: A standard error method.

    Returns:
      An Analysis object with the standard error method set.

    Raises:
      ValueError: Standard error method is already defined.
    """

    if self.parameters.se_method is not None:
      raise ValueError("Standard error method is already defined.")

    self.parameters.se_method = copy.deepcopy(method)

    return self

  def calculate(self, metrics):
    """Specifies metrics to calculate.

    Args:
      metrics: A Metric object or a list of Metric objects.

    Returns:
      An Analysis object with the metrics set.

    Raises:
      ValueError: Metrics are already defined.
      TypeError: .calculate() takes a Metric or a list of Metrics.
      ValueError: A metric appears more than once in the list of Metrics.
    """

    if self.parameters.metrics is not None:
      raise ValueError("Metrics are already defined.")

    if isinstance(metrics, collections.Iterable):
      if all(isinstance(metric, metrics_module.Metric) for metric in metrics):
        # Check that no two metrics have the same name.
        metric_names = set()
        for metric in metrics:
          if metric.name in metric_names:
            raise ValueError(
                "'%s' appears more than once in the metrics." % metric.name
            )
          else:
            metric_names.add(metric.name)
        # Create a list of (copies of) each metric
        self.parameters.metrics = [copy.deepcopy(metric) for metric in metrics]
      else:
        raise TypeError(".calculate() takes a Metric or a list of Metrics")
    elif isinstance(metrics, metrics_module.Metric):
      self.parameters.metrics = [copy.deepcopy(metrics)]
    else:
      raise TypeError(".calculate() takes a Metric or a list of Metrics")

    return self

  def run(self,
          melted=False,
          var_name="Metric",
          value_name="Value",
          encoding="utf8"):
    """Runs the analysis, returning the output in the specified form.

    Args:
      melted: Boolean indicating whether to return the output in "melted" form
        (i.e., with a separate row for each metric, as opposed to a separate
        column for each metric).
      var_name: Column name to use for the metric when data is melted
        (equivalent to var_name in pd.melt).
      value_name: Column name to use for the values of the metric when data is
        melted (equivalent to value_name in pd.melt).
      encoding: String encoding to use for string columns.

    Returns:
      A pandas dataframe with the results.

    Raises:
      ValueError: No metrics to calculate.
    """
    results = []

    split_index = self.parameters.split_index
    split_vars = self.parameters.split_vars
    comparison = self.parameters.comparison
    se_method = self.parameters.se_method

    if self.parameters.metrics is None:
      raise ValueError("No metrics to calculate.")

    if (self.data is None) or self.data.empty:
      raise ValueError("The dataset is empty.")

    # Do calculations that only need to be done once.
    index_vars = []
    if se_method is not None:
      se_method.precalculate(self.data)
    if comparison is not None:
      comparison.precalculate(self.data)
      index_vars.append(comparison.condition_column)
    for metric in self.parameters.metrics:
      metric.precalculate(self.data, split_index)
    index_vars.extend(split_vars)

    # Explicitly decode columns that will become part of the index,
    # to avoid UnicodeDecodeErrors.  Since pandas has no way to distinguish
    # between bytes and strings, we just try decoding and catch any failures.
    for col in index_vars:
      # Only convert columns of type "object".
      if self.data[col].dtype == np.object_:
        try:
          decoded = self.data[col].str.decode(encoding)
          self.data.loc[~decoded.isna(), col] = decoded[~decoded.isna()]
        except UnicodeEncodeError:
          pass

    # Set the index of the dataframe.
    if index_vars:
      df = self.data.set_index(index_vars)
    else:
      df = self.data

    def _compute_metric(metric):
      """Computes the metric and process the output.

      1. Computes the metric.
      2. Transforms to pd.DataFrame.
      3. Melts the dataframe if melted=True.
      4. Gives columns reasonable names.
      5. Records the MetricIndex and OverColumns in metric if present.
      6. Orders the indexes and sorts if asked.

      Args:
        metric: A Metric() instance to compute.

      Returns:
        The computed metric in pd.DataFrame.
      """
      if se_method is not None:
        output = se_method(df, metric, comparison)
      elif comparison is not None:
        output = comparison(df, metric)
      else:
        output = metric(df)
      all_metric_indices.extend(metric.metric_idx)
      all_over_columns.extend(metric.over)

      # Convert output to dataframe.
      if np.isscalar(output):
        output = pd.DataFrame([output], columns=[""])
      elif isinstance(output, pd.Series):
        output = pd.DataFrame(output)

      # To melt data, add an index called var_name that stores the metric.
      if melted:
        output[var_name] = metric.name
        output.rename(columns={"": value_name}, inplace=True)
        output.set_index(var_name, append=True, inplace=True)

      # Otherwise, depending on if there's stderr calculated, we either append
      # metric name to the beginning of each of the columns, or make results a
      # MultiIndex DataFrame.
      else:
        if se_method:
          output.columns = pd.MultiIndex.from_product([[metric.name],
                                                       output.columns])
        else:
          output.columns = ["{} {}".format(metric.name, col).strip()
                            for col in output.columns]

      if comparison and len(output.index.names) > 1:
        # Comparison is by default the first index. Put it to the last.
        output.index = output.index.reorder_levels(
            np.roll(output.index.names, -1))
      if split_vars and self.parameters.sort:
        output.sort_index(level=split_vars, inplace=True, sort_remaining=False)

      return output

    # Calculate the metrics. For the meaning of MetricIndex and OverColumns,
    # see the doc of Metric().
    all_metric_indices = []
    all_over_columns = []
    for metric in self.parameters.metrics:
      results.append(_compute_metric(metric))
    all_metric_indices = list(pd.unique(all_metric_indices))
    all_over_columns = list(pd.unique(all_over_columns))

    def _fill_missing_indexes(results):
      """Fills missing index in results so all elements have same index.

      When MetricIndex or OverColumn exist, each metric might have different
      MetricIndex or OverColumn. As we will concat all metrics later, we need
      them to have exactly same indexes. The function fills the missing indexes.

      Args:
        results: The list of metrics computed, with possible missing indexes.

      Returns:
        The list of metrics computed, all having same indexes.
      """
      all_metric_indices_over_columns = all_metric_indices + all_over_columns
      all_idx = split_vars + [var_name] if melted else split_vars[:]
      all_idx += all_metric_indices
      all_idx += [comparison.condition_column] if comparison else []
      all_idx += all_over_columns
      # Metrics might have different MetricIndex and Over columns. To concat
      # metrics later, we need all metrics to have same indexes.
      for i, output in enumerate(results):
        for col in all_metric_indices_over_columns:
          if col not in output.index.names:
            output[col] = ""
            output.set_index(col, append=True, inplace=True)
        # To make the indexes of all metrics be in the same order.
        output = output.reset_index(all_idx).set_index(all_idx)

        if not melted:
          # df.unstack(df.index) is a pd.Series, not a pd.DataFrame anymore.
          # This will introduce trouble when operating with other df. So add a
          # placeholder index column in such case.
          if len(output.index.names) == len(all_metric_indices):
            output[""] = 0
            output.set_index("", append=True, inplace=True)
          output = output.unstack(all_metric_indices)

        results[i] = output
      return results

    if all_metric_indices or all_over_columns:
      results = _fill_missing_indexes(results)
    # concatenate results for each metric into a single dataframe
    results = pdutils.concat(results, axis=0 if melted else 1)

    if not melted and se_method:
      column_multiindex = results.rename(columns={"": value_name},
                                         level=1).columns
      if not se_method.flat_index:
        results.columns = column_multiindex
      else:
        # Merge the first two levels metric and value type.
        if all_metric_indices:
          new_idx = [
              [" ".join(c[:2]).strip()] + list(c[2:]) for c in results.columns
          ]
          names = [None] + list(results.columns.names[2:])
          new_idx = pd.MultiIndex.from_tuples(new_idx, names=names)
        else:
          new_idx = [" ".join(c).strip() for c in results.columns]
        results.columns = new_idx

    return results
