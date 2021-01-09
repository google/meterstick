# Copyright 2020 Google LLC
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
"""Operation classes for Meterstick."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from typing import Any, Iterable, List, Optional, Text, Tuple, Union
from meterstick import confidence_interval_display
from meterstick import metrics
from meterstick import utils
import numpy as np
import pandas as pd
from scipy import stats


class Operation(metrics.Metric):
  """An meta-Metric that operates on a Metric instance.

  The differences between Metric and Operation are
  1. Operation must take another Metric as the child to operate on.
  2. The name of Operation is reflected in the result differently. A Metric
    usually returns a 1D data and its name could just be used as the column.
    However, Operation often operates on MetricList and one name doesn't fit
    all. What we do is we apply the name_tmpl of Operation to all Metric names

  Attributes:
    name: Name of the Metric.
    name_tmpl: The template to generate the name from child Metrics' names.
    children: A Length-1 tuple of the child Metric(s) whose results will be the
      input to the Operation. Might be None in __init__, but must be assigned
      before compute().
    extra_index: Many Operations rely on adding extra split_by columns to child
      Metric. For example,
      PercentChange('condition', base_value, Sum('X')).compute_on(df, 'grp')
      would compute Sum('X').compute_on(df, ['grp', 'condition']) then get the
      change. As the result, the CacheKey used in PercentChange is different to
      that used in Sum('X'). The latter has more columns in the split_by.
      extra_index records what columns need to be added to children Metrics so
      we can flush the cache correctly. The convention is extra_index comes
      after split_by. If not, you need to overwrite flush_children().
    precomputable_in_jk: Indicates whether it is possible to cut corners to
      obtain leave-one-out (LOO) estimates for the Jackknife. This attribute
      is True if the input df is only used in compute_on() and compute_child().
      This is necessary because Jackknife emits None as the input df for LOO
      estimation when cutting corners. The compute_on() and compute_child()
      functions know to read from cache but other functions may not know what to
      do. If your Operation uses df outside the compute_on() and compute_child()
      functions, you have either to
      1. ensure that your computation doesn't break when df is None.
      2. set attribute 'precomputable_in_jk' to False (which will force the
         jackknife to be computed the manual way, which is slower).
    where: A string that will be passed to df.query() as a prefilter.
    cache_key: What key to use to cache the df. You can use anything that can be
      a key of a dict except '_RESERVED' and tuples like ('_RESERVED', ...).
    And all other attributes inherited from Metric.
  """

  def __init__(self,
               child: Optional[metrics.Metric] = None,
               name_tmpl: Optional[Text] = None,
               extra_index: Optional[Union[Text, Iterable[Text]]] = None,
               name: Optional[Text] = None,
               where: Optional[Text] = None,
               **kwargs):
    if name_tmpl and not name:
      name = name_tmpl.format(utils.get_name(child))
    super(Operation, self).__init__(name, child or (), where, **kwargs)
    self.name_tmpl = name_tmpl
    self.extra_index = [extra_index] if isinstance(extra_index,
                                                   str) else extra_index or []
    self.precomputable_in_jk = True

  def compute_child(self,
                    df: pd.DataFrame,
                    split_by=None,
                    melted=False,
                    return_dataframe=True,
                    cache_key=None):
    child = self.children[0]
    cache_key = cache_key or self.cache_key or self.RESERVED_KEY
    cache_key = self.wrap_cache_key(cache_key, split_by)
    return child.compute_on(df, split_by, melted, return_dataframe, cache_key)

  def manipulate(self, res, melted, return_dataframe=True):
    """Applies name_tmpl to all Metric names."""
    res = super(Operation, self).manipulate(res, melted, return_dataframe)
    res = res.copy()  # Avoid changing the result in cache.
    if melted:
      if len(res.index.names) > 1:
        res.index.set_levels(
            map(self.name_tmpl.format, res.index.levels[0]), 0, inplace=True)
      else:
        res.index = map(self.name_tmpl.format, res.index)
    else:
      res.columns = map(self.name_tmpl.format, res.columns)
    return res

  def flush_children(self,
                     key=None,
                     split_by=None,
                     where=None,
                     recursive=True,
                     prune=True):
    split_by = (split_by or []) + self.extra_index
    super(Operation, self).flush_children(key, split_by, where, recursive,
                                          prune)

  def __call__(self, child: metrics.Metric):
    op = copy.deepcopy(self) if self.children else self
    op.name = op.name_tmpl.format(utils.get_name(child))
    op.children = (child,)
    op.cache = {}
    return op


class Distribution(Operation):
  """Computes the normalized values of a Metric over column(s).

  Attributes:
    extra_index: A list of column(s) to normalize over.
    children: A tuple of a Metric whose result we normalize on.
    And all other attributes inherited from Operation.
  """

  def __init__(self,
               over: Union[Text, List[Text]],
               child: Optional[metrics.Metric] = None,
               **kwargs):
    super(Distribution, self).__init__(child, 'Distribution of {}', over,
                                       **kwargs)

  def compute_slices(self, df, split_by=None):
    lvls = split_by + self.extra_index if split_by else self.extra_index
    res = self.compute_child(df, lvls)
    total = res.groupby(level=split_by).sum() if split_by else res.sum()
    return res / total


Normalize = Distribution  # An alias.


class CumulativeDistribution(Operation):
  """Computes the normalized cumulative sum.

  Attributes:
    extra_index: A list of column(s) to normalize over.
    children: A tuple of a Metric whose result we compute the cumulative
      distribution on.
    order: An iterable. The over column will be ordered by it before computing
      cumsum.
    ascending: Sort ascending or descending.
    And all other attributes inherited from Operation.
  """

  def __init__(self,
               over: Text,
               child: Optional[metrics.Metric] = None,
               order=None,
               ascending: bool = True,
               **kwargs):
    self.order = order
    self.ascending = ascending
    super(CumulativeDistribution,
          self).__init__(child, 'Cumulative Distribution of {}', over, **kwargs)

  def split_data(self, df, split_by=None):
    """Caches the result for the whole df instead of many slices."""
    if not split_by:
      yield self.compute_child(df, self.extra_index), None
    else:
      child = self.compute_child(df, split_by + self.extra_index)
      keys, indices = list(zip(*child.groupby(split_by).groups.items()))
      for i, idx in enumerate(indices):
        yield child.loc[idx.unique()].droplevel(split_by), keys[i]

  def compute(self, df):
    if self.order:
      df = pd.concat((
          df.loc[[o]] for o in self.order if o in df.index.get_level_values(0)))
    else:
      df.sort_values(self.extra_index, ascending=self.ascending, inplace=True)
    dist = df.cumsum()
    dist /= df.sum()
    return dist


class Comparison(Operation):
  """Base class for comparisons like percent/absolute change."""

  def __init__(self,
               condition_column,
               baseline_key,
               child: Optional[metrics.Metric] = None,
               include_base: bool = False,
               name_tmpl: Optional[Text] = None,
               **kwargs):
    self.baseline_key = baseline_key
    self.include_base = include_base
    super(Comparison, self).__init__(child, name_tmpl, condition_column,
                                     **kwargs)


class PercentChange(Comparison):
  """Percent change estimator on a Metric.

  Attributes:
    extra_index: The column(s) that contains the conditions.
    baseline_key: The value of the condition that represents the baseline (e.g.,
      "Control"). All conditions will be compared to this baseline. If
      condition_column contains multiple columns, then baseline_key should be a
      tuple.
    children: A tuple of a Metric whose result we compute percentage change on.
    include_base: A boolean for whether the baseline condition should be
      included in the output.
    And all other attributes inherited from Operation.
  """

  def __init__(self,
               condition_column: Text,
               baseline_key,
               child: Optional[metrics.Metric] = None,
               include_base: bool = False,
               **kwargs):
    super(PercentChange,
          self).__init__(condition_column, baseline_key, child, include_base,
                         '{} Percent Change', **kwargs)

  def compute_slices(self, df, split_by: Optional[List[Text]] = None):
    if split_by:
      to_split = list(split_by) + self.extra_index
      level = self.extra_index[0] if len(
          self.extra_index) == 1 else self.extra_index
    else:
      to_split = self.extra_index
      level = None
    res = self.compute_child(df, to_split)
    res = (res / res.xs(self.baseline_key, level=level) - 1) * 100
    if not self.include_base:
      to_drop = [i for i in res.index.names if i not in self.extra_index]
      idx_to_match = res.index.droplevel(to_drop) if to_drop else res.index
      res = res[~idx_to_match.isin([self.baseline_key])]
    return res


class AbsoluteChange(Comparison):
  """Absolute change estimator on a Metric.

  Attributes:
    extra_index: The column(s) that contains the conditions.
    baseline_key: The value of the condition that represents the baseline (e.g.,
      "Control"). All conditions will be compared to this baseline. If
      condition_column contains multiple columns, then baseline_key should be a
      tuple.
    children: A tuple of a Metric whose result we compute absolute change on.
    include_base: A boolean for whether the baseline condition should be
      included in the output.
    And all other attributes inherited from Operation.
  """

  def __init__(self,
               condition_column: Text,
               baseline_key,
               child: Optional[metrics.Metric] = None,
               include_base: bool = False,
               **kwargs):
    super(AbsoluteChange,
          self).__init__(condition_column, baseline_key, child, include_base,
                         '{} Absolute Change', **kwargs)

  def compute_slices(self, df, split_by: Optional[List[Text]] = None):
    if split_by:
      to_split = list(split_by) + self.extra_index
      level = self.extra_index[0] if len(
          self.extra_index) == 1 else self.extra_index
    else:
      to_split = self.extra_index
      level = None
    res = self.compute_child(df, to_split)
    # Don't use "-=". For multiindex it might go wrong. The reason is DataFrame
    # has different implementations for __sub__ and __isub__. ___isub__ tries
    # to reindex to update in place which sometimes lead to lots of NAs.
    res = res - res.xs(self.baseline_key, level=level)
    if not self.include_base:
      to_drop = [i for i in res.index.names if i not in self.extra_index]
      idx_to_match = res.index.droplevel(to_drop) if to_drop else res.index
      res = res[~idx_to_match.isin([self.baseline_key])]
    return res


class MH(Comparison):
  """Cochran-Mantel-Haenszel statistics estimator on a Metric.

  MH only takes a ratio of two single-column Metrics, or a MetricList of such
  ratios.
  So AbsoluteChange(MetricList([a, b])) / AbsoluteChange(MetricList([c, d]))
  won't work. Instead please use
  MetricList([AbsoluteChange(a) / AbsoluteChange(c),
              AbsoluteChange(b) / AbsoluteChange(d)]).

  Attributes:
    extra_index: The column(s) that contains the conditions.
    baseline_key: The value of the condition that represents the baseline (e.g.,
      "Control"). All conditions will be compared to this baseline. If
      condition_column contains multiple columns, then baseline_key should be a
      tuple.
    stratified_by: The stratification column(s) in the DataFrame.
    children: A tuple of a Metric whose result we compute the MH on.
    include_base: A boolean for whether the baseline condition should be
      included in the output.
    And all other attributes inherited from Operation.
  """

  def __init__(self,
               condition_column: Text,
               baseline_key: Text,
               stratified_by: Union[Text, List[Text]],
               metric: Optional[metrics.Metric] = None,
               include_base: bool = False,
               **kwargs):
    self.stratified_by = stratified_by if isinstance(stratified_by,
                                                     list) else [stratified_by]
    super(MH, self).__init__(condition_column, baseline_key, metric,
                             include_base, '{} MH Ratio', **kwargs)

  def check_is_ratio(self, metric=None):
    metric = metric or self.children[0]
    if isinstance(metric, metrics.MetricList):
      for m in metric:
        self.check_is_ratio(m)
    else:
      if not isinstance(metric,
                        metrics.CompositeMetric) or metric.op(2.0, 2) != 1:
        raise ValueError('MH only makes sense on ratio Metrics.')

  def compute_one_metric(self, metric, df, split_by=None):
    """Computes MH statistics for one Metric."""
    mh_metric = metrics.MetricList(metric.children)
    numer = metric.children[0].name
    denom = metric.children[1].name
    df_all = mh_metric.compute_on(
        df,
        split_by + self.extra_index + self.stratified_by,
        cache_key=self.cache_key or self.RESERVED_KEY)
    level = self.extra_index[0] if len(
        self.extra_index) == 1 else self.extra_index
    df_baseline = df_all.xs(self.baseline_key, level=level)
    suffix = '_base'
    df_mh = df_all.join(df_baseline, rsuffix=suffix)
    ka, na = df_mh[numer], df_mh[denom]
    kb, nb = df_mh[numer + suffix], df_mh[denom + suffix]
    weights = 1. / (na + nb)
    to_split = [i for i in ka.index.names if i not in self.stratified_by]
    res = ((ka * nb * weights).groupby(to_split).sum() /
           (kb * na * weights).groupby(to_split).sum() - 1) * 100
    res.name = metric.name
    to_split = [i for i in to_split if i not in self.extra_index]
    if to_split:
      split_by = split_by or []
      extra_idx = [i for i in to_split if i not in split_by]
      res = res.reorder_levels(split_by + self.extra_index + extra_idx)

    if not self.include_base:
      to_drop = [i for i in res.index.names if i not in self.extra_index]
      idx_to_match = res.index.droplevel(to_drop) if to_drop else res.index
      res = res[~idx_to_match.isin([self.baseline_key])]
    return pd.DataFrame(res.sort_index(level=split_by + self.extra_index))

  def compute_slices(self, df, split_by):
    self.check_is_ratio()
    child = self.children[0]
    if isinstance(child, metrics.MetricList):
      return pd.concat(
          [self.compute_one_metric(m, df, split_by) for m in child],
          axis=1,
          sort=False)
    else:
      return self.compute_one_metric(child, df, split_by)

  def flush_children(self,
                     key=None,
                     split_by=None,
                     where=None,
                     recursive=True,
                     prune=True):
    """Flushes the grandchildren as child is not computed."""
    split_by = (split_by or []) + self.extra_index + self.stratified_by
    if isinstance(self.children[0], metrics.MetricList):
      for c in self.children[0]:
        c.flush_children(key, split_by, where, recursive, prune)
    else:
      self.children[0].flush_children(key, split_by, where, recursive, prune)


def get_display_fn(name,
                   split_by=None,
                   melted=False,
                   value='Value',
                   raw=None,
                   condition_column: Optional[List[Text]] = None,
                   ctrl_id=None,
                   default_metric_formats=None):
  """Returns a function that displays confidence interval nicely.

  Args:
    name: 'Jackknife' or 'Bootstrap'.
    split_by: The split_by passed to Jackknife().compute_on().
    melted: Whether the input res is in long format.
    value: The name of the value column.
    raw: Present if the child is PercentChange or AbsoluteChange. It's the base
      values for comparison. We will use it in the display.
    condition_column: Present if the child is PercentChange or AbsoluteChange.
    ctrl_id: Present if the child is PercentChange or AbsoluteChange. It's the
      baseline_key of the comparison.
    default_metric_formats: How to format the numbers in the display.

  Returns:
    A funtion that takes a DataFrame and displays confidence intervals.
  """

  def display(res,
              aggregate_dimensions=True,
              show_control=None,
              metric_formats=None,
              sort_by=None,
              metric_order=None,
              flip_color=None,
              hide_null_ctrl=True,
              display_expr_info=False,
              auto_add_description=False,
              return_pre_agg_df=False,
              return_formatted_df=False):
    """Displays confidence interval nicely in Colab/Jupyter notebook.

    Args:
      res: The DataFrame returned by Jackknife or Bootstrap with confidence
        level specified.
      aggregate_dimensions: Whether to aggregate all dimensions in to one
        column.
      show_control: If False, only ratio values in non-control rows are shown.
      metric_formats: A dict specifying how to display metric values. Keys can
        be 'Value' and 'Ratio'. Values can be 'absolute', 'percent', 'pp' or a
        formatting string. For example, '{:.2%}' would have the same effect as
          'percent'. By default, Value is in absolute form and Ratio in percent.
      sort_by: In the form of
        [{'column': ('CI_Lower', 'Metric Foo'), 'ascending': False}},
         {'column': 'Dim Bar': 'order': ['Logged-in', 'Logged-out']}]. 'column'
           is the column to sort by. If you want to sort by a metric, use
           (field, metric name) where field could be 'Ratio', 'Value',
           'CI_Lower' and 'CI_Upper'. 'order' is optional and for categorical
           column. 'ascending' is optional and default True. The result will be
           displayed in the order specified by sort_by from top to bottom.
      metric_order: An iterable. The metric will be displayed by the order from
        left to right.
      flip_color: A iterable of metric names that positive changes will be
        displayed in red and negative changes in green.
      hide_null_ctrl: If to hide control value or use '-' to represent it when
        it is null,
      display_expr_info: If to display 'Control_id', 'Is_Control' and
        'Description' columns. Only has effect when aggregate_dimensions is
        False.
      auto_add_description: If add Control/Not Control as descriptions.
      return_pre_agg_df: If to return the pre-aggregated df.
      return_formatted_df: If to return raw HTML df to be rendered.

    Returns:
      Displays confidence interval nicely for df, or aggregated/formatted if
      return_pre_agg_df/return_formatted_df is True.
    """
    if not melted:
      res = utils.melt(res)
    if raw is not None:
      res = raw.join(res)  # raw always has the baseline so needs to be at left.
      comparison_suffix = [
          AbsoluteChange('', '').name_tmpl.format(''),
          PercentChange('', '').name_tmpl.format('')
      ]
      comparison_suffix = '(%s)$' % '|'.join(comparison_suffix)
      # Don't use inplace=True. It will change the index of 'raw' too.
      res.index = res.index.set_levels(
          res.index.levels[0].str.replace(comparison_suffix, ''), 0)
      show_control = True if show_control is None else show_control
    metric_order = list(res.index.get_level_values(
        0).unique()) if metric_order is None else metric_order
    res = res.reset_index()
    control = ctrl_id
    condition_col = condition_column
    if condition_column:
      if len(condition_column) == 1:
        condition_col = condition_column[0]
      else:
        res['_expr_id'] = res[condition_column].agg(', '.join, axis=1)
        control = ', '.join(ctrl_id)
        condition_col = '_expr_id'

    metric_formats = (
        default_metric_formats if metric_formats is None else metric_formats)
    formatted_df = confidence_interval_display.get_formatted_df(
        res,
        split_by,
        aggregate_dimensions,
        show_control,
        metric_formats,
        ratio=value,
        value='_base_value',
        ci_upper=name + ' CI-upper',
        ci_lower=name + ' CI-lower',
        expr_id=condition_col,
        ctrl_id=control,
        sort_by=sort_by,
        metric_order=metric_order,
        flip_color=flip_color,
        hide_null_ctrl=hide_null_ctrl,
        display_expr_info=display_expr_info,
        auto_add_description=auto_add_description,
        return_pre_agg_df=return_pre_agg_df)
    if return_pre_agg_df or return_formatted_df:
      return formatted_df
    display_formatted_df = confidence_interval_display.display_formatted_df
    return display_formatted_df(formatted_df)

  return display


class MetricWithCI(Operation):
  """Base class for Metrics that have confidence interval info in the return.

  The return when melted, has columns like
  Value  Jackknife SE
  or if confidence specified,
  Value  Jackknife CI-lower  Jackknife CI-upper
  if not melted, the columns are pd.MultiIndex like
  Metric1                                          Metric2...
  Value  Jackknife SE (or CI-lower and CI-upper)   Value  Jackknife SE
  The column for point estimate is usually "Value", but could be others like
  "Percent Change" for comparison Metrics so don't rely on the name, but you can
  assume what ever it's called, it's always the first column followed by
  "... SE" or "... CI-lower" and "... CI-upper".
  If confidence is speified, a display function will be bound to the returned
  DataFrame so res.display() will display confidence interval and highlight
  significant changes nicely in Colab and Jupyter notebook.
  As the return has multiple columns even for one Metric, the default DataFrame
  returned is in melted format, unlike vanilla Metric.
  The main computation pipeline is used to compute stderr or confidence interval
  bounds. We compute the point estimates in final_compute() and and combine it
  with stderr or CI in the final_compute.
  Similar to how you derive Metric, if you don't need vectorization, overwrite
  compute(), or even simpler, get_samples(). See Bootstrap for an example. If
  you need vectorization, overwrite compute_slices. See Jackknife for an
  example.

  Attributes:
    unit: The column to go over (kackknife/bootstrap over) to get stderr.
    confidence: The level of the confidence interval, must be in (0, 1). If
      specified, we return confidence interval range instead of standard error.
      Additionally, a display() function will be bound to the result so you can
      visualize the confidence interval nicely in Colab and Jupyter notebook.
    prefix: In the result, the column names will be like "{prefix} SE",
      "{prefix} CI-upper".
    And all other attributes inherited from Operation.
  """

  def __init__(self,
               unit: Optional[Text],
               child: Optional[metrics.Metric] = None,
               confidence: Optional[float] = None,
               name_tmpl: Optional[Text] = None,
               prefix: Optional[Text] = None,
               **kwargs):
    if confidence and not 0 < confidence < 1:
      raise ValueError('Confidence must be in (0, 1).')
    self.unit = unit
    self.confidence = confidence
    super(MetricWithCI, self).__init__(child, name_tmpl, **kwargs)
    self.prefix = prefix
    if not self.prefix and self.name_tmpl:
      self.prefix = prefix or self.name_tmpl.format('').strip()

  def compute_on_samples(self,
                         keyed_samples: Iterable[Tuple[Any, pd.DataFrame]],
                         split_by=None):
    """Iters through sample DataFrames and collects results.

    Args:
      keyed_samples: A tuple. The first element is the cache_key and the second
        is the corresponding DataFrame. Remember a key should correspond to the
        same data.
      split_by: Something can be passed into DataFrame.group_by().

    Returns:
      List of results from samples.
    """
    estimates = []
    for keyed_sample in keyed_samples:
      try:
        cache_key, sample = keyed_sample
        res = self.compute_child(
            sample, split_by, melted=True, cache_key=cache_key)
        estimates.append(res)
      except Exception as e:  # pylint: disable=broad-except
        print(
            'Warning: Failed on sample data for reason %s. If you see many such'
            ' failures, your data might be too sparse.' % repr(e))
      finally:
        # Jackknife keys are unique so can be kept longer.
        if isinstance(self, Bootstrap) and cache_key is not None:
          cache_key = self.wrap_cache_key(cache_key, split_by)
          # In case errors occur so the top Metric was not computed, we don't
          # want to prune because the leaf Metrics still need to be cleaned up.
          self.flush_children(cache_key, split_by, prune=False)
    return estimates

  def compute(self, df):
    estimates = self.compute_on_samples(self.get_samples(df))
    return self.get_stderrs_or_ci_half_width(estimates)

  def manipulate(self,
                 res,
                 melted: bool = False,
                 return_dataframe: bool = True):
    # Always return a melted df and don't add suffix like "Jackknife" because
    # point_est won't have it.
    del melted, return_dataframe  # unused
    return super(Operation, self).manipulate(res, True, True)  # pylint: disable=bad-super-call

  def final_compute(self,
                    std,
                    melted: bool = False,
                    return_dataframe: bool = True,
                    split_by: Optional[List[Text]] = None,
                    df=None):
    """Computes point estimates and returns it with stderrs or CI range."""
    if self.where:
      df = df.query(self.where)
    point_est = self.compute_child(df, split_by, melted=True)
    res = point_est.join(std)

    if self.confidence:
      res[self.prefix +
          ' CI-lower'] = res.iloc[:, 0] - res[self.prefix + ' CI-lower']
      res[self.prefix + ' CI-upper'] += res.iloc[:, 0]

    if not melted:
      res = utils.unmelt(res)

    if self.confidence:
      raw = None
      extra_idx = list(metrics.get_extra_idx(self))
      indexes = split_by + extra_idx if split_by else extra_idx
      if len(self.children) == 1 and isinstance(
          self.children[0], (PercentChange, AbsoluteChange)):
        change = self.children[0]
        to_split = (
            split_by + change.extra_index if split_by else change.extra_index)
        indexes = [i for i in indexes if i not in change.extra_index]
        raw = change.compute_child(df, to_split)
        raw.columns = [change.name_tmpl.format(c) for c in raw.columns]
        raw = utils.melt(raw)
        raw.columns = ['_base_value']
      res = self.add_display_fn(res, indexes, melted, raw)
    return res

  def add_display_fn(self, res, split_by, melted, raw=None):
    """Bounds a display function to res so res.display() works."""
    value = res.columns[0] if melted else res.columns[0][1]
    ctrl_id = None
    condition_col = None
    metric_formats = None
    if len(self.children) == 1 and isinstance(self.children[0],
                                              (PercentChange, AbsoluteChange)):
      change = self.children[0]
      ctrl_id = change.baseline_key
      condition_col = change.extra_index
      if isinstance(self.children[0], PercentChange):
        metric_formats = {'Ratio': 'percent'}

    fn = get_display_fn(self.prefix, split_by, melted, value, raw,
                        condition_col, ctrl_id, metric_formats)
    # pylint: disable=no-value-for-parameter
    res.display = fn.__get__(res)  # pytype: disable=attribute-error
    # pylint: enable=no-value-for-parameter
    return res

  @staticmethod
  def get_stderrs(replicates):
    bucket_estimates = pd.concat(replicates, axis=1, sort=False)
    num_buckets = bucket_estimates.count(axis=1)
    return bucket_estimates.std(1), num_buckets - 1

  def get_ci_width(self, stderrs, dof):
    """You can return asymmetrical confidence interval."""
    half_width = stderrs * stats.t.ppf((1 + self.confidence) / 2, dof)
    return half_width, half_width

  def get_stderrs_or_ci_half_width(self, replicates):
    """Returns confidence interval infomation in an unmelted DataFrame."""
    stderrs, dof = self.get_stderrs(replicates)
    if self.confidence:
      res = pd.DataFrame(self.get_ci_width(stderrs, dof)).T
      res.columns = [self.prefix + ' CI-lower', self.prefix + ' CI-upper']
    else:
      res = pd.DataFrame(stderrs, columns=[self.prefix + ' SE'])
    res = utils.unmelt(res)
    return res

  def get_samples(self, df, split_by=None):
    raise NotImplementedError

  def compute_on_sql(
      self,
      table,
      split_by=None,
      execute=None,
      melted=False):
    res = super(MetricWithCI, self).compute_on_sql(
        table,
        split_by,
        execute)
    sub_dfs = []
    if self.confidence:
      # raw contains the base values passed to comparison.
      raw = None
      split_by = [split_by] if isinstance(split_by, str) else split_by
      extra_idx = list(metrics.get_extra_idx(self))
      indexes = split_by + extra_idx if split_by else extra_idx
      if len(self.children) == 1 and isinstance(
          self.children[0], (PercentChange, AbsoluteChange)):
        if len(res.columns) % 4:
          raise ValueError('Wrong shape for a MetricWithCI with confidence!')
        n_metrics = len(res.columns) // 4
        raw = res.iloc[:, -n_metrics:]
        res = res.iloc[:, :3 * n_metrics]
        change = self.children[0]
        raw.columns = [change.name_tmpl.format(c) for c in raw.columns]
        raw = utils.melt(raw)
        raw.columns = ['_base_value']
        indexes = [i for i in indexes if i not in change.extra_index]

      if len(res.columns) % 3:
        raise ValueError('Wrong shape for a MetricWithCI with confidence!')

      # The columns are like metric1, metric1 jackknife SE, metric1 dof, ...
      metric_names = res.columns[::3]
      sub_dfs = []
      ci_lower = self.prefix + ' CI-lower'
      ci_upper = self.prefix + ' CI-upper'
      for i in range(0, len(res.columns), 3):
        pt_est = res.iloc[:, i]
        half_width = self.get_ci_width(res.iloc[:, i + 1], res.iloc[:, i + 2])
        sub_df = pd.DataFrame(
            {
                'Value': res.iloc[:, i],
                ci_lower: pt_est - half_width[0],
                ci_upper: pt_est + half_width[0]
            },
            columns=['Value', ci_lower, ci_upper])
        sub_dfs.append(sub_df)
    else:
      if len(res.columns) % 2:
        raise ValueError('Wrong shape for a MetricWithCI!')

      # The columns are like metric1, metric1 jackknife SE, ...
      metric_names = res.columns[::2]
      for i in range(0, len(res.columns), 2):
        sub_df = res.iloc[:, [i, i + 1]]
        sub_df.columns = ['Value', self.prefix + ' SE']
        sub_dfs.append(sub_df)

    res = pd.concat((sub_dfs), 1, keys=metric_names, names=['Metric'])
    res = utils.melt(res) if melted else res
    if self.confidence:
      res = self.add_display_fn(res, indexes, melted, raw)
    return res


def get_sum_ct_monkey_patch_fn(unit, original_split_by, original_compute):
  """Gets a function that can be monkey patched to Sum/Count.compute_slices.

  Args:
    unit: The column whose levels define the jackknife buckets.
    original_split_by: The split_by passed to Jackknife().compute_on().
    original_compute: The compute_slices() of Sum or Count. We will monkey patch
      it.

  Returns:
    A function that can be monkey patched to Sum/Count.compute_slices().
  """

  def precompute_loo(self, df, split_by=None):
    """Precomputes leave-one-out (LOO) results to make Jackknife faster.

    For Sum, Count and Mean, it's possible to compute the LOO estimates in a
    vectorized way. For Sum and Count, we can get the LOO estimates by
    subtracting the sum/count of each bucket from the total. Here we precompute
    and cache the LOO results.

    Args:
      self: The Sum or Count instance callling this function.
      df: The DataFrame passed to Sum/Count.compute_slies().
      split_by: The split_by passed to Sum/Count.compute_slies().

    Returns:
      Same as what normal Sum/Count.compute_slies() would have returned.
    """
    total = original_compute(self, df, split_by)
    if isinstance(self, metrics.Count) and self.distinct:
      # For Count distinct, we cannot cut the corner.
      return total
    split_by_with_unit = [unit] + split_by if split_by else [unit]
    each_bucket = original_compute(self, df, split_by_with_unit)
    each_bucket = utils.adjust_slices_for_loo(each_bucket, original_split_by)
    loo = total - each_bucket
    if split_by:
      # total - each_bucket might put the unit as the innermost level, but we
      # want the unit as the outermost level.
      loo = loo.reorder_levels(split_by_with_unit)
    buckets = loo.index.get_level_values(0).unique() if split_by else loo.index
    for bucket in buckets:
      key = self.wrap_cache_key(('_RESERVED', 'Jackknife', unit, bucket),
                                split_by, self.cache_key.where)
      self.save_to_cache(key, loo.loc[bucket])
      self.tmp_cache_keys.add(key)
    return total

  return precompute_loo


def get_mean_monkey_patch_fn(unit, original_split_by):
  """Gets a function that can be monkey patched to Mean.compute_slices.

  Args:
    unit: The column whose levels define the jackknife buckets.
    original_split_by: The split_by passed to Jackknife().compute_on().

  Returns:
    A function that can be monkey patched to Sum/Count.compute_slices().
  """

  def precompute_loo(self, df, split_by=None):
    """Precomputes leave-one-out (LOO) results to make Jackknife faster.

    For Sum, Count and Mean, it's possible to compute the LOO estimates in a
    vectorized way. LOO mean is just LOO sum / LOO count. Here we precompute
    and cache the LOO results.

    Args:
      self: The Mean instance callling this function.
      df: The DataFrame passed to Mean.compute_slies().
      split_by: The split_by passed to Mean.compute_slies().

    Returns:
      Same as what normal Mean.compute_slies() would have returned.
    """
    data = df.copy()
    split_by_with_unit = [unit] + split_by if split_by else [unit]
    if self.weight:
      weighted_var = '_weighted_%s' % self.var
      data[weighted_var] = data[self.var] * data[self.weight]
      total_sum = self.group(data, split_by)[weighted_var].sum()
      total_weight = self.group(data, split_by)[self.weight].sum()
      bucket_sum = self.group(data, split_by_with_unit)[weighted_var].sum()
      bucket_sum = utils.adjust_slices_for_loo(bucket_sum, original_split_by)
      bucket_weight = self.group(data, split_by_with_unit)[self.weight].sum()
      bucket_weight = utils.adjust_slices_for_loo(bucket_weight,
                                                  original_split_by)
      loo_sum = total_sum - bucket_sum
      loo_weight = total_weight - bucket_weight
      if split_by:
        # total - bucket_sum might put the unit as the innermost level, but we
        # want the unit as the outermost level.
        loo_sum = loo_sum.reorder_levels(split_by_with_unit)
        loo_weight = loo_weight.reorder_levels(split_by_with_unit)
      loo = loo_sum / loo_weight
      mean = total_sum / total_weight
    else:
      total_sum = self.group(data, split_by)[self.var].sum()
      bucket_sum = self.group(data, split_by_with_unit)[self.var].sum()
      bucket_sum = utils.adjust_slices_for_loo(bucket_sum, original_split_by)
      total_ct = self.group(data, split_by)[self.var].count()
      bucket_ct = self.group(data, split_by_with_unit)[self.var].count()
      bucket_ct = utils.adjust_slices_for_loo(bucket_ct, original_split_by)
      loo_sum = total_sum - bucket_sum
      loo_ct = total_ct - bucket_ct
      loo = loo_sum / loo_ct
      mean = total_sum / total_ct
      if split_by:
        loo = loo.reorder_levels(split_by_with_unit)

    buckets = loo.index.get_level_values(0).unique() if split_by else loo.index
    for bucket in buckets:
      key = utils.CacheKey(('_RESERVED', 'Jackknife', unit, bucket),
                           self.cache_key.where, split_by)
      self.save_to_cache(key, loo.loc[bucket])
      self.tmp_cache_keys.add(key)
    return mean

  return precompute_loo


def save_to_cache_for_jackknife(self, key, val, split_by=None):
  """Used to monkey patch the save_to_cache() during Jackknife.precompute().

  What cache_key to use for the point estimate of Jackknife is tricky because we
  want to support two use cases at the same time.
  1. We want sumx to be computed only once in
    MetricList([Jackknife(sumx), sumx]).compute_on(df, return_dataframe=False),
    so the key for point estimate should be the same sumx uses.
  2. But then it will fail when multiple Jackknifes are involved. For example,
    (Jackknife(unit1, sumx) - Jackknife(unit2, sumx)).compute_on(df)
  will fail because two Jackknifes share point estimate but not LOO estimates.
  When the 2nd Jackknife precomputes its point esitmate, as it uses the same key
  as the 1st one, it will mistakenly assume LOO has been cached, but
  unfortunately it's not true.
  The solution here is we use different keys for different Jackknifes, so LOO
  will always be precomputed. Additionally we cache the point estimate again
  with the key other Metrics like Sum would use so they can reuse it.

  Args:
    self: An instance of metrics.Metric.
    key: The cache key currently being used in computation.
    val: The value to cache.
    split_by: Something can be passed into df.group_by().
  """
  key = self.wrap_cache_key(key, split_by)
  if isinstance(key.key, tuple) and key.key[:2] == ('_RESERVED', 'jk'):
    val = val.copy() if isinstance(val, (pd.Series, pd.DataFrame)) else val
    base_key = key.key[2]
    base_key = utils.CacheKey(base_key, key.where, key.split_by, key.slice_val)
    self.cache[base_key] = val
    if utils.is_tmp_key(base_key):
      self.tmp_cache_keys.add(base_key)
  val = val.copy() if isinstance(val, (pd.Series, pd.DataFrame)) else val
  self.cache[key] = val


class Jackknife(MetricWithCI):
  """Class for Jackknife estimates of standard errors.

  Attributes:
    unit: The column whose levels define the jackknife buckets.
    confidence: The level of the confidence interval, must be in (0, 1). If
      specified, we return confidence interval range instead of standard error.
      Additionally, a display() function will be bound to the result so you can
      visualize the confidence interval nicely in Colab and Jupyter notebook.
    children: A tuple of a Metric whose result we jackknife on.
    can_precompute: If all leaf Metrics are Sum, Count, and Mean, then we can
      cut the corner to compute leave-one-out estimates.
    And all other attributes inherited from Operation.
  """

  def __init__(self,
               unit: Text,
               child: Optional[metrics.Metric] = None,
               confidence: Optional[float] = None,
               **kwargs):
    super(Jackknife, self).__init__(unit, child, confidence, '{} Jackknife',
                                    None, **kwargs)
    self.can_precompute = self.can_be_precomputed()

  def __call__(self, child: metrics.Metric):
    jk = super(Jackknife, self).__call__(child)
    jk.can_precompute = jk.can_be_precomputed()
    return jk

  def precompute(self, df, split_by=None):
    """Caches point estimate and leave-one-out (LOO) results for Sum/Count/Mean.

    For Sum, Count and Mean, it's possible to compute the LOO estimates in a
    vectorized way. For Sum and Count, we can get the LOO estimates
    by subtracting the sum/count of each bucket from the total. For Mean, LOO
    mean is LOO sum / LOO count. So we can monkey patch the compute_slices() of
    the Metrics to cache the LOO results under certain keys when we precompute
    the point estimate.

    Args:
      df: The DataFrame passed from compute_on().
      split_by: The split_by passed from compute_on().

    Returns:
      The input df. All we do here is saving precomputed stuff to cache.
    """
    original_sum_compute_slices = metrics.Sum.compute_slices
    original_ct_compute_slices = metrics.Count.compute_slices
    original_mean_compute_slices = metrics.Mean.compute_slices
    original_save_to_cache = metrics.Metric.save_to_cache
    try:
      metrics.Sum.compute_slices = get_sum_ct_monkey_patch_fn(
          self.unit, split_by, original_sum_compute_slices)
      metrics.Count.compute_slices = get_sum_ct_monkey_patch_fn(
          self.unit, split_by, original_ct_compute_slices)
      metrics.Mean.compute_slices = get_mean_monkey_patch_fn(
          self.unit, split_by)
      metrics.Metric.save_to_cache = save_to_cache_for_jackknife
      cache_key = self.cache_key or self.RESERVED_KEY
      cache_key = ('_RESERVED', 'jk', cache_key, self.unit)
      self.compute_child(df, split_by, cache_key=cache_key)
    finally:
      metrics.Sum.compute_slices = original_sum_compute_slices
      metrics.Count.compute_slices = original_ct_compute_slices
      metrics.Mean.compute_slices = original_mean_compute_slices
      metrics.Metric.save_to_cache = original_save_to_cache
    return df

  def get_samples(self, df, split_by=None):
    """Yields leave-one-out (LOO) DataFrame with level value.

    This step is the bottleneck of Jackknife so we have some tricks here.
    1. If all leaf Metrics are Sum or Count, whose LOO results have already been
    calculated, then we don't bother to get the right DataFrame. All we need is
    the right cache_key to retrive the results. This saves lots of time.
    2. If split_by is True, some slices may be missing buckets, so we only keep
    the slices that appear in that bucket. In other words, if a slice doesn't
    have bucket i, then the leave-i-out sample won't have the slice.
    3. We yield the cache_key for bucket i together with the leave-i-out
    DataFrame because we need the cache_key to retrieve results.

    Args:
      df: The DataFrame to compute on.
      split_by: Something can be passed into df.group_by().

    Yields:
      ('_RESERVED', 'Jackknife', unit, i) and the leave-i-out DataFrame.
    """
    levels = df[self.unit].unique()
    if len(levels) < 2:
      raise ValueError('Too few %s to jackknife.' % self.unit)

    if self.can_precompute:
      for lvl in levels:
        yield ('_RESERVED', 'Jackknife', self.unit, lvl), None
    else:
      if not split_by:
        for lvl in levels:
          yield ('_RESERVED', 'Jackknife', self.unit,
                 lvl), df[df[self.unit] != lvl]
      else:
        df = df.set_index(split_by)
        max_slices = len(df.index.unique())
        for lvl, idx in df.groupby(self.unit).groups.items():
          df_rest = df[df[self.unit] != lvl]
          unique_slice_val = idx.unique()
          if len(unique_slice_val) != max_slices:
            # Keep only the slices that appeared in the dropped bucket.
            df_rest = df_rest[df_rest.index.isin(unique_slice_val)]
          yield ('_RESERVED', 'Jackknife', self.unit,
                 lvl), df_rest.reset_index()

  def compute_slices(self, df, split_by=None):
    """Tries to compute stderr in a vectorized way as much as possible.

    For the slices that have all the units, jackknife and groupby are
    interchangeable. We find those slices and compute them in a vectorized way.
    Then for the slices missing some units, we recursively apply such method to
    maximize the vectorization. When none of the slices are full, we fall back
    to computing slice by slice.

    Args:
      df: The DataFrame to compute on.
      split_by: A list of column names to be passed to df.group_by().

    Returns:
      A melted DataFrame of stderrs for all Metrics in self.children[0].
    """
    samples = self.get_samples(df, split_by)
    estimates = self.compute_on_samples(samples, split_by)
    return self.get_stderrs_or_ci_half_width(estimates)

  @staticmethod
  def get_stderrs(replicates):
    bucket_estimates = pd.concat(replicates, axis=1, sort=False)
    means = bucket_estimates.mean(axis=1)
    # Some slices may be missing buckets so we can't just use len(replicates).
    num_buckets = bucket_estimates.count(axis=1)
    rss = (bucket_estimates.subtract(means, axis=0)**2).sum(axis=1, min_count=1)
    return np.sqrt(rss * (1. - 1. / num_buckets)), num_buckets - 1

  def can_be_precomputed(self):
    """If all leaf Metrics are Sum, Count or Mean, LOO can be precomputed."""
    for m in self.traverse():
      if isinstance(m, Operation) and not m.precomputable_in_jk:
        return False
      if not m.children and not isinstance(
          m, (metrics.Sum, metrics.Count, metrics.Mean)):
        return False
      if isinstance(m, metrics.Count) and m.distinct:
        return False
    return True


class Bootstrap(MetricWithCI):
  """Class for Bootstrap estimates of standard errors.

  Attributes:
    unit: The column representing the level to be resampled. If sample the
      slices in unit column, otherwise we sample rows.
    n_replicates: The number of bootstrap replicates. In "What Teachers Should
      Know About the Bootstrap" Tim Hesterberg recommends 10000 for routine use
      https://amstat.tandfonline.com/doi/full/10.1080/00031305.2015.1089789.
    confidence: The level of the confidence interval, must be in (0, 1). If
      specified, we return confidence interval range instead of standard error.
      Additionally, a display() function will be bound to the result so you can
      visualize the confidence interval nicely in Colab and Jupyter notebook.
    children: A tuple of a Metric whose result we bootstrap on.
    And all other attributes inherited from Operation.
  """

  def __init__(self,
               unit: Optional[Text] = None,
               child: Optional[metrics.Metric] = None,
               n_replicates: int = 10000,
               confidence: Optional[float] = None,
               **kwargs):
    super(Bootstrap, self).__init__(unit, child, confidence, '{} Bootstrap',
                                    None, **kwargs)
    self.n_replicates = n_replicates

  def get_samples(self, df):
    if self.unit:
      slices = df[self.unit].unique()
      buckets = range(len(slices))
      data_slices = [df[df[self.unit] == s] for s in slices]
    else:
      buckets = range(len(df))
    for _ in range(self.n_replicates):
      buckets_sampled = np.random.choice(buckets, size=len(buckets))
      if self.unit is None:
        yield ('_RESERVED', 'Bootstrap', self.unit), df.iloc[buckets_sampled]
      else:
        yield ('_RESERVED', 'Bootstrap',
               self.unit), pd.concat(data_slices[i] for i in buckets_sampled)
