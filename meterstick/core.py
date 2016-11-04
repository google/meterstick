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

"""Core module for meterstick."""

import collections
import copy

import numpy as np
import pandas as pd


class AnalysisParameters(object):
  """Class for the parameters of a data analysis.

  Attributes:
    metrics: A Metric or a list of Metric objects specifying
      which metric(s) to compute.
    split_vars: A variable or a list of variables to split the data upon.
    comparison: A Comparison object which performs the comparison.
    se_method: A function which calculates standard errors.
  """

  def __init__(self,
               metrics=None,
               split_vars=None,
               comparison=None,
               se_method=None):
    """Initializes AnalysisParameters object.

    Args:
      metrics: A Metric object or a list/tuple of Metric objects.
      split_vars: A string or list of strings on which to split the analysis.
      comparison: A Comparison object
      se_method: A StandardErrorMethod object
    """

    self.metrics = metrics
    self.split_vars = split_vars
    self.comparison = comparison
    self.se_method = se_method


class Analyze(object):
  """Base analysis class; provides base functionality for data analysis.

  Attributes:
    data: A pandas dataframe.
    weights: A numpy array of integer weights.
    parameters: An AnalysisParameters object.
  """

  def __init__(self, data, weights=None, parameters=None):
    """Initializes Analysis object.

    Args:
      data: A pandas dataframe.
      weights: A numpy array of integer weights.
      parameters: An AnalysisParameters object.
    """

    self.data = data
    self.weights = np.ones(data.shape[0], int) if weights is None else weights
    self.parameters = parameters or AnalysisParameters()

  def _clone(self):
    return Analyze(self.data.copy(), self.weights.copy(),
                   copy.deepcopy(self.parameters))

  def where(self, query):
    """Subsets the data according to a query.

    Args:
     query: A string which evaluates to an array of Booleans.

    Returns:
      An Analysis object with weights where query is false zeroed out.

    Raises:
      ValueError: Query does not evaluate to an array of Booleans.
    """

    new = self._clone()

    query_results = new.data.eval(query).values
    if query_results.dtype == np.bool:
      new.weights &= query_results
    else:
      raise ValueError("Query does not evalulate to an array of Booleans.")

    return new

  def split_by(self, split_vars):
    """Splits the analysis by categorical variables.

    Args:
      split_vars: A string or a list of strings representing variables
        to split upon.

    Returns:
      An Analysis object with the split_vars attribute set to input.

    Raises:
      ValueError: Split variables are already defined.
      TypeError: Split variable is not a string or list of strings.
    """

    if self.parameters.split_vars is not None:
      raise ValueError("Split variables are already defined.")

    new = self._clone()

    if isinstance(split_vars, basestring):
      new.parameters.split_vars = [split_vars]
    else:
      try:
        all_strings = all(isinstance(var, basestring) for var in split_vars)
      except KeyError:
        raise TypeError("Split variable is not a string or list of strings.")
      if not all_strings:
        raise TypeError("Split variable is not a string or list of strings.")
      new.parameters.split_vars = split_vars

    return new

  def relative_to(self, comparison, sort=True):
    """Specifies the comparison for the analysis.

    This modifies comparison in place by precalculating factor
    variables.

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

    new = self._clone()

    comparison.precalculate_factors(self.data, sort)

    new.parameters.comparison = comparison

    return new

  def with_standard_errors(self, method):
    """Specifies standard error method for analysis.

    This modifies method in place by precalculating factor variables.

    Args:
      method: A standard error method.

    Returns:
      An Analysis object with the standard error method set.

    Raises:
      ValueError: Standard error method is already defined.
    """

    if self.parameters.se_method is not None:
      raise ValueError("Standard error method is already defined.")

    new = self._clone()

    method.precalculate_factors(self.data)

    new.parameters.se_method = method

    return new

  def calculate(self, metrics):
    """Specifies metrics to calculate.

    Args:
      metrics: A Metric object or a list of Metric objects.

    Returns:
      An Analysis object with the metrics set.

    Raises:
      ValueError: Metrics are already defined.
    """

    if self.parameters.metrics is not None:
      raise ValueError("Metrics are already defined.")

    new = self._clone()

    if isinstance(metrics, collections.Iterable):
      new.parameters.metrics = list(metrics)
    else:
      new.parameters.metrics = [metrics]

    return new

  def run(self, melted=False, var_name="Metric", value_name="Value",
          sort=True):
    """Runs the analysis, returning the output in the specified form.

    Args:
      melted: Boolean indicating whether to return the output in "melted" form
        (i.e., with a separate row for each metric, as opposed to a separate
        column for each metric).
      var_name: Column name to use for the metric when data is melted
        (equivalent to var_name in pd.melt).
      value_name: Column name to use for the values of the metric when data is
        melted (equivalent to value_name in pd.melt).
      sort: Boolean indicating whether to sort the returned output.

    Returns:
      A pandas dataframe with the results.

    Raises:
      ValueError: No metrics to calculate.
    """

    results = []

    data = self.data
    metrics = self.parameters.metrics
    split_vars = self.parameters.split_vars
    comparison = self.parameters.comparison
    se_method = self.parameters.se_method

    if metrics is None:
      raise ValueError("No metrics to calculate.")

    for metric in metrics:
      if se_method is not None:

        def run_fn(data, weights, metric=metric):
          # pylint can't recognize that se_method is callable.
          return se_method(data, weights, metric,
                           comparison)  # pylint: disable=not-callable
      elif comparison is not None:

        def run_fn(data, weights, metric=metric):
          # pylint can't recognize that comparison is callable.
          return comparison(data, weights,
                            metric)  # pylint: disable=not-callable
      else:

        def run_fn(data, weights, metric=metric):
          value = metric(data, weights)
          if isinstance(value, pd.DataFrame):
            return value
          return pd.DataFrame(np.array([[value]]),
                              columns=pd.Index([""]))

      if split_vars is None:
        output = run_fn(data, self.weights)
      else:
        # We iterate through the unique tuples of the split_vars
        # variables and zero-out everything except from rows which
        # match the tuple.

        split_tuples = pd.lib.fast_zip([data[ii].values for ii in split_vars])
        factors, keys = pd.factorize(split_tuples, sort=sort)

        dfs = []

        for ii in xrange(len(keys)):
          split_weights = self.weights * (factors == ii)
          dfs.append(run_fn(data, split_weights))

        output = pd.concat(dfs, keys=keys, names=split_vars)

      # Comparisons and dataframe metrics manage indices themselves. Otherwise
      # we need to remove the extraneous last index for aesthetic reasons.
      if comparison is None and metric.metric_type != "dataframe":
        output.reset_index(-1, drop=True, inplace=True)

      # Add a column called metric if data needs to be returned in melted form.
      if melted:
        output.columns = [" ".join(col.split()) for col in output.columns]
        output[var_name] = metric.name
        output.rename(columns={"": value_name}, inplace=True)
      # Otherwise, append metric name to the beginning of each of the columns.
      else:
        # Split column names and rejoin to remove extra whitespace.
        output.columns = [" ".join([metric.name] + col.split())
                          for col in output.columns]
      results.append(output)

    return pd.concat(results, axis=1 * (not melted))
