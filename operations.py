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
from typing import Any, Iterable, List, Optional, Sequence, Text, Tuple, Union
import warnings

from meterstick import confidence_interval_display
from meterstick import metrics
from meterstick import sql
from meterstick import utils
import numpy as np
import pandas as pd
from scipy import stats


class Operation(metrics.Metric):
  """A special kind of Metric that operates on other Metric instance(s).

  The differences between Metric and Operation are
  1. Operation must take other Metric(s) as the children to operate on.
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
    extra_split_by: Many Operations rely on adding extra split_by columns to
      child Metric. For example,
      PercentChange('condition', base_value, Sum('X')).compute_on(df, 'grp')
      would compute Sum('X').compute_on(df, ['grp', 'condition']) then get the
      change. As the result, the CacheKey used in PercentChange is different to
      that used in Sum('X'). The latter has more columns in the split_by.
      extra_split_by records what columns need to be added to children Metrics
      so we can flush the cache correctly. The convention is extra_split_by
      comes after split_by.
    extra_index: Not every extra_split_by show up in the result. For example,
      the group_by columns in Models don't show up in the final output.
      extra_index stores the columns that will show up and should be a subset of
      extra_split_by. If not given, it's same as extra_split_by.
    precomputable_in_jk: Indicates whether it is possible to cut corners to
      obtain leave-one-out (LOO) estimates for the Jackknife. This attribute is
      True if the input df is only used in compute_on() and compute_child().
      This is necessary because Jackknife emits None as the input df for LOO
      estimation when cutting corners. The compute_on() and compute_child()
      functions know to read from cache but other functions may not know what to
      do. If your Operation uses df outside the compute_on() and compute_child()
      functions, you have either to
      1. ensure that your computation doesn't break when df is None.
      2. set attribute 'precomputable_in_jk' to False (which will force the
         jackknife to be computed the manual way, which is slower).
    precompute_child: Many Operations first compute the child Metric with extra
      split_by columns, then perform computation on the child Metric' result. If
      precompute_child is True, in the precompute(), we return
      self.children[0].compute_on(df, split_by + self.extra_index). Otherwise
      the original input data is returned.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    cache_key: What key to use to cache the df. You can use anything that can be
      a key of a dict except '_RESERVED' and tuples like ('_RESERVED', ...).
    And all other attributes inherited from Metric.
  """

  def __init__(self,
               child: Optional[metrics.Metric] = None,
               name_tmpl: Optional[Text] = None,
               extra_split_by: Optional[Union[Text, Iterable[Text]]] = None,
               extra_index: Optional[Union[Text, Iterable[Text]]] = None,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None,
               additional_fingerprint_attrs: Optional[List[str]] = None,
               **kwargs):
    if name_tmpl and not name:
      name = name_tmpl.format(utils.get_name(child))
    super(Operation,
          self).__init__(name, child or (), where, name_tmpl, extra_split_by,
                         extra_index, additional_fingerprint_attrs, **kwargs)
    self.precomputable_in_jk = True
    self.precompute_child = True
    self.apply_name_tmpl = True

  def split_data(self, df, split_by=None):
    """Splits the DataFrame returned by the children if it's computed."""
    if not self.precompute_child or not split_by:
      for i in super(Operation, self).split_data(df, split_by):
        yield i
    else:
      keys, indices = zip(*df.groupby(split_by).groups.items())
      for k, idx in zip(keys, indices):
        yield df.loc[idx.unique()].droplevel(split_by), k

  def compute_slices(self, df, split_by: Optional[List[Text]] = None):
    try:
      children = self.compute_children(df, split_by + self.extra_split_by)
      res = self.compute_on_children(children, split_by)
      if isinstance(res, pd.Series):
        return pd.DataFrame([res], columns=children.columns)
      return res
    except NotImplementedError:
      return super(Operation, self).compute_slices(df, split_by)

  def compute_children(self,
                       df: pd.DataFrame,
                       split_by=None,
                       melted=False,
                       return_dataframe=True,
                       cache_key=None):
    return self.compute_child(df, split_by, melted, return_dataframe, cache_key)

  def compute_child(self,
                    df: pd.DataFrame,
                    split_by=None,
                    melted=False,
                    return_dataframe=True,
                    cache_key=None):
    child = self.children[0]
    return self.compute_util_metric_on(child, df, split_by, melted,
                                       return_dataframe, cache_key)

  def compute_child_sql(self,
                        table,
                        split_by,
                        execute,
                        melted=False,
                        mode=None,
                        cache_key=None):
    child = self.children[0]
    cache_key = self.wrap_cache_key(cache_key, split_by)
    return self.compute_util_metric_on_sql(child, table, split_by, execute,
                                           melted, mode, cache_key)

  def compute_on_sql_mixed_mode(self, table, split_by, execute, mode=None):
    res = super(Operation,
                self).compute_on_sql_mixed_mode(table, split_by, execute, mode)
    return utils.apply_name_tmpl(self.name_tmpl, res)

  def __call__(self, child: metrics.Metric):
    op = copy.deepcopy(self) if self.children else self
    op.name = op.name_tmpl.format(utils.get_name(child))
    op.children = (child,)
    return op


class Distribution(Operation):
  """Computes the normalized values of a Metric over column(s).

  Attributes:
    extra_split_by: A list of column(s) to normalize over.
    children: A tuple of a Metric whose result we normalize on.
    And all other attributes inherited from Operation.
  """

  def __init__(self,
               over: Union[Text, List[Text]],
               child: Optional[metrics.Metric] = None,
               **kwargs):
    super(Distribution, self).__init__(child, 'Distribution of {}', over,
                                       **kwargs)

  def compute_on_children(self, children, split_by):
    total = children.groupby(
        level=split_by).sum() if split_by else children.sum()
    return children / total

  def get_sql_and_with_clause(self, table, split_by, global_filter, indexes,
                              local_filter, with_data):
    """Gets the SQL query and WITH clause.

    The query is constructed by
    1. Get the query for the child metric.
    2. Keep all indexing/groupby columns unchanged.
    3. For all value columns, get
      value / SUM(value) OVER (PARTITION BY split_by).

    Args:
      table: The table we want to query from.
      split_by: The columns that we use to split the data.
      global_filter: The sql.Filters that can be applied to the whole Metric
        tree.
      indexes: The columns that we shouldn't apply any arithmetic operation.
      local_filter: The sql.Filters that have been accumulated so far.
      with_data: A global variable that contains all the WITH clauses we need.

    Returns:
      The SQL instance for metric, without the WITH clause component.
      The global with_data which holds all datasources we need in the WITH
        clause.
    """
    local_filter = sql.Filters([self.where, local_filter]).remove(global_filter)
    child_sql, with_data = self.children[0].get_sql_and_with_clause(
        table, indexes, global_filter, indexes, local_filter, with_data)
    child_table = sql.Datasource(child_sql, 'DistributionRaw')
    child_table_alias, rename = with_data.merge(child_table)
    groupby = sql.Columns(indexes.aliases, distinct=True)
    columns = sql.Columns()
    for c in child_sql.columns:
      if c.alias in groupby:
        continue
      alias = rename.get(c.alias, c.alias)
      col = sql.Column(alias) / sql.Column(
          alias, 'SUM({})', partition=split_by.aliases)
      col.set_alias('Distribution of %s' % c.alias_raw)
      columns.add(col)
    return sql.Sql(groupby.add(columns), child_table_alias), with_data


Normalize = Distribution  # An alias.


class CumulativeDistribution(Operation):
  """Computes the normalized cumulative sum.

  Attributes:
    extra_split_by: A list of column(s) to normalize over.
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
    super(CumulativeDistribution, self).__init__(
        child,
        'Cumulative Distribution of {}',
        over,
        additional_fingerprint_attrs=['order', 'ascending'],
        **kwargs)

  def compute(self, df):
    if self.order:
      df = pd.concat((
          df.loc[[o]] for o in self.order if o in df.index.get_level_values(0)))
    else:
      df.sort_values(self.extra_index, ascending=self.ascending, inplace=True)
    dist = df.cumsum()
    dist /= df.sum()
    return dist

  def get_sql_and_with_clause(self, table, split_by, global_filter, indexes,
                              local_filter, with_data):
    """Gets the SQL query and WITH clause.

    The query is constructed by
    1. Get the query for the Distribution of the child Metric.
    2. Keep all indexing/groupby columns unchanged.
    3. For all value columns, get the cumulative sum by summing over
      'ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW'.

    Args:
      table: The table we want to query from.
      split_by: The columns that we use to split the data.
      global_filter: The sql.Filters that can be applied to the whole Metric
        tree.
      indexes: The columns that we shouldn't apply any arithmetic operation.
      local_filter: The sql.Filters that have been accumulated so far.
      with_data: A global variable that contains all the WITH clauses we need.

    Returns:
      The SQL instance for metric, without the WITH clause component.
      The global with_data which holds all datasources we need in the WITH
        clause.
    """
    local_filter = sql.Filters([self.where, local_filter]).remove(global_filter)
    util_metric = Distribution(self.extra_split_by, self.children[0])
    child_sql, with_data = util_metric.get_sql_and_with_clause(
        table, split_by, global_filter, indexes, local_filter, with_data)
    child_table = sql.Datasource(child_sql, 'CumulativeDistributionRaw')
    child_table_alias, rename = with_data.merge(child_table)
    columns = sql.Columns(indexes.aliases)
    order = list(utils.get_extra_idx(self))
    order[0] = sql.Column(
        _get_order_for_cum_dist(sql.Column(order[0]).alias, self),
        auto_alias=False)
    for c in child_sql.columns:
      if c in columns:
        continue

      col = sql.Column(
          rename.get(c.alias, c.alias),
          'SUM({})',
          partition=split_by.aliases,
          order=order,
          window_frame='ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW')
      col.set_alias('Cumulative %s' % c.alias_raw)
      columns.add(col)
    return sql.Sql(columns, child_table_alias), with_data


def _get_order_for_cum_dist(over, metric):
  if metric.order:
    over = 'CASE %s\n' % over
    tmpl = 'WHEN %s THEN %s'
    over += '\n'.join(
        tmpl % (_format_to_condition(o), i) for i, o in enumerate(metric.order))
    over += '\nELSE %s\nEND' % len(metric.order)
  return over if metric.ascending else over + ' DESC'


def _format_to_condition(val):
  if isinstance(val, str) and not val.startswith('$'):
    return '"%s"' % val
  return '%s' % val


class Comparison(Operation):
  """Base class for comparisons like percent/absolute change."""

  def __init__(self,
               condition_column,
               baseline_key,
               child: Optional[metrics.Metric] = None,
               include_base: bool = False,
               name_tmpl: Optional[Text] = None,
               additional_fingerprint_attrs=None,
               **kwargs):
    self.baseline_key = baseline_key
    self.include_base = include_base
    additional_fingerprint_attrs = additional_fingerprint_attrs or []
    super(Comparison, self).__init__(
        child,
        name_tmpl,
        extra_split_by=condition_column,
        additional_fingerprint_attrs=['baseline_key', 'include_base'] +
        additional_fingerprint_attrs,
        **kwargs)

  def get_sql_and_with_clause(self, table, split_by, global_filter, indexes,
                              local_filter, with_data):
    """Gets the SQL for PercentChange or AbsoluteChange.

    The query is constructed by
    1. Get the query for the child metric and add it to with_data, we call it
      raw_value_table.
    2. Query the rows that only has the base value from raw_value_table, add it
      to with_data too. We call it base_value_table.
    3. sql.Join the two tables and computes the change for all value columns.

    For example, the query for
    AbsoluteChange('condition', 'base_value', metrics.Mean('click'))
    will look like this:

    WITH
    ChangeRaw AS (SELECT
      split_by,
      condition,
      AVG(click) AS `mean(click)`
    FROM $DATA
    GROUP BY split_by, condition),
    ChangeBase AS (SELECT
      split_by,
      `mean(click)`
    FROM ChangeRaw
    WHERE
    condition = "base_value")
    SELECT
      split_by,
      condition,
      ChangeRaw.`mean(click)` - ChangeBase.`mean(click)`
        AS `mean(click) Absolute Change`
    FROM ChangeRaw
    JOIN
    ChangeBase
    USING (split_by)
    WHERE
    condition != "base_value"

    Args:
      table: The table we want to query from.
      split_by: The columns that we use to split the data.
      global_filter: The sql.Filters that can be applied to the whole Metric
        tree.
      indexes: The columns that we shouldn't apply any arithmetic operation.
      local_filter: The sql.Filters that have been accumulated so far.
      with_data: A global variable that contains all the WITH clauses we need.

    Returns:
      The SQL instance for metric, without the WITH clause component.
      The global with_data which holds all datasources we need in the WITH
        clause.
    """
    if not isinstance(self, (PercentChange, AbsoluteChange)):
      raise ValueError('Not a PercentChange nor AbsoluteChange!')
    local_filter = sql.Filters([self.where, local_filter]).remove(global_filter)

    child = self.children[0]
    cond_cols = sql.Columns(self.extra_split_by)
    groupby = sql.Columns(split_by).add(cond_cols)
    alias_tmpl = self.name_tmpl
    raw_table_sql, with_data = child.get_sql_and_with_clause(
        table, groupby, global_filter, indexes, local_filter, with_data)
    raw_table = sql.Datasource(raw_table_sql, 'ChangeRaw')
    raw_table_alias, rename = with_data.merge(raw_table)

    base = self.baseline_key if isinstance(self.baseline_key,
                                           tuple) else [self.baseline_key]
    base_cond = ('%s = %s' % (c, _format_to_condition(b))
                 for c, b in zip(cond_cols.aliases, base))
    base_cond = ' AND '.join(base_cond)
    cols = sql.Columns(raw_table_sql.groupby.aliases)
    cols.add((rename.get(a, a) for a in raw_table_sql.columns.aliases))
    base_value = sql.Sql(
        cols.difference(cond_cols.aliases), raw_table_alias, base_cond)
    base_table = sql.Datasource(base_value, 'ChangeBase')
    base_table_alias, rename = with_data.merge(base_table)

    exclude_base_condition = ('%s != %s' % (c, _format_to_condition(b))
                              for c, b in zip(cond_cols.aliases, base))
    exclude_base_condition = ' OR '.join(exclude_base_condition)
    cond = None if self.include_base else sql.Filters([exclude_base_condition])
    col_tmp = '%s.{r} - %s.{b}' if isinstance(
        self, AbsoluteChange) else 'SAFE_DIVIDE(%s.{r}, (%s.{b})) * 100 - 100'
    columns = sql.Columns()
    for c in raw_table_sql.columns.difference(indexes.aliases):
      raw_table_col = rename.get(c.alias, c.alias)
      base_table_col = rename.get(c.alias, c.alias)
      col = sql.Column(
          col_tmp.format(r=raw_table_col, b=base_table_col) %
          (raw_table_alias, base_table_alias),
          alias=alias_tmpl.format(c.alias_raw))
      columns.add(col)
    using = indexes.difference(cond_cols)
    join = '' if using else 'CROSS'
    return sql.Sql(
        sql.Columns(indexes.aliases).add(columns),
        sql.Join(raw_table_alias, base_table_alias, join=join, using=using),
        cond), with_data


class PercentChange(Comparison):
  """Percent change estimator on a Metric.

  Attributes:
    extra_split_by: The column(s) that contains the conditions.
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
               name_tmpl=None,
               **kwargs):
    name_tmpl = name_tmpl or '{} Percent Change'
    super(PercentChange, self).__init__(condition_column, baseline_key, child,
                                        include_base, name_tmpl, **kwargs)

  def compute_on_children(self, children, split_by):
    level = None
    if split_by:
      level = self.extra_index[0] if len(
          self.extra_index) == 1 else self.extra_index
    res = (children / children.xs(self.baseline_key, level=level) - 1) * 100
    if len(children.index.names) > 1:  # xs might mess up the level order.
      res = res.reorder_levels(children.index.names)
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
               name_tmpl=None,
               **kwargs):
    name_tmpl = name_tmpl or '{} Absolute Change'
    super(AbsoluteChange, self).__init__(condition_column, baseline_key, child,
                                         include_base, name_tmpl, **kwargs)

  def compute_on_children(self, children, split_by):
    level = None
    if split_by:
      level = self.extra_index[0] if len(
          self.extra_index) == 1 else self.extra_index
    # Don't use "-=". For multiindex it might go wrong. The reason is DataFrame
    # has different implementations for __sub__ and __isub__. ___isub__ tries
    # to reindex to update in place which sometimes lead to lots of NAs.
    res = children - children.xs(self.baseline_key, level=level)
    if len(children.index.names) > 1:  # xs might mess up the level order.
      res = res.reorder_levels(children.index.names)
    if not self.include_base:
      to_drop = [i for i in res.index.names if i not in self.extra_index]
      idx_to_match = res.index.droplevel(to_drop) if to_drop else res.index
      res = res[~idx_to_match.isin([self.baseline_key])]
    return res


class PrePostChange(PercentChange):
  """PrePost Percent change estimator on a Metric.

  Computes the percent change after controlling for preperiod metrics.
  Essentially, if the data only has a baseline and a treatment slice, PrePost
  1. centers the covariates
  2. fit child ~ intercept + was_treated * covariate.
  As covariate is centered, the intercept is the mean value for the baseline.
  The coefficient for the was_treated term is the mean effect of treatment.
  PrePostChange returns the latter / the former * 100.
  See https://arxiv.org/pdf/1711.00562.pdf for more details.
  For data with multiple treatments, the result is same as applying the method
  to every pair of baseline and treatment.
  If child returns multiple columns, the result is same as applying the method
  to every column in it.

  Attributes:
    extra_split_by: The column(s) that contains the conditions.
    baseline_key: The value of the condition that represents the baseline (e.g.,
      "Control"). All conditions will be compared to this baseline. If
      condition_column contains multiple columns, then baseline_key should be a
      tuple.
    children: A MetricList whose first element is the Metric we want to compute
      change on and the rest is the covariates for adjustment.
    include_base: A boolean for whether the baseline condition should be
      included in the output.
    And all other attributes inherited from Operation.
  """

  def __init__(self,
               condition_column,
               baseline_key,
               child=None,
               covariates=None,
               stratified_by=None,
               include_base=False,
               **kwargs):
    if isinstance(covariates, (List, Tuple)):
      covariates = metrics.MetricList(covariates)
    if child and covariates:
      child = metrics.MetricList((child, covariates)) if child else covariates
    else:
      child = None
    stratified_by = [stratified_by] if isinstance(stratified_by,
                                                  str) else stratified_by or []
    condition_column = [condition_column] if isinstance(
        condition_column, str) else condition_column
    super(PrePostChange,
          self).__init__(stratified_by + condition_column, baseline_key, child,
                         include_base, '{} PrePost Percent Change', **kwargs)
    self.extra_index = condition_column
    self.computable_in_pure_sql = False

  def compute_children(self,
                       df,
                       split_by,
                       melted=False,
                       return_dataframe=True,
                       cache_key=None):
    child, covariates = super(PrePostChange, self).compute_children(
        df, split_by, return_dataframe=False, cache_key=cache_key)
    original_split_by = [s for s in split_by if s not in self.extra_split_by]
    return self.adjust_value(child, covariates, original_split_by)

  def adjust_value(self, child, covariates, split_by):
    """Adjust the raw value by controlling for Pre-metrics.

    As described in the class doc, PrePost fits a linear model,
    child ~ β0 + β1 * treated + β2 * covariate + β3 * treated * covariate,
    to adjust the effect, where β0 is the average effect of the control while
    β0 + β1 is that of the treatment group. Note that we center covariate first
    so in practice β0 and β1 can be achieved by fitting small models. β0_c in
    child ~ β0_c + β1_c * covariate,
    when fitted on control data only, would be equal to β0. And β0_t in
    child ~ β0_t + β1_t * covariate, when fitted on treatment data only, would
    equal to β0 + β1. The principle holds for multiple treatments. Here we fit
    child ~ 1 + covariate
    on every slice of data instead of fitting a large model on the whole data.

    Args:
      child: A pandas DataFrame. The result of the child Metric.
      covariates: A pandas DataFrame. The result of the covariates Metric.
      split_by: The split_by passed to self.compute_on().

    Returns:
      The adjusted values of the child (post metrics).
    """
    from sklearn import linear_model  # pylint: disable=g-import-not-at-top
    # Don't use "-=". For multiindex it might go wrong. The reason is DataFrame
    # has different implementations for __sub__ and __isub__. ___isub__ tries
    # to reindex to update in place which sometimes lead to lots of NAs.
    if split_by:
      covariates = covariates - covariates.groupby(split_by).mean()
    else:
      covariates = covariates - covariates.mean()
    # Align child with covariates in case there is any missing slices.
    covariates = covariates.reorder_levels(child.index.names)
    aligned = pd.concat([child, covariates], axis=1)
    len_child = child.shape[1]
    lm = linear_model.LinearRegression()

    def adjust(df_slice):
      child_slice = df_slice.iloc[:, :len_child]
      cov = df_slice.iloc[:, len_child:]
      adjusted = [lm.fit(cov, child_slice[c]).intercept_ for c in child_slice]
      return pd.DataFrame([adjusted], columns=child_slice.columns)

    adjusted = metrics.Metric('Adjusted', compute=adjust)
    return adjusted.compute_on(aligned, split_by + self.extra_index)

  def compute_children_sql(self, table, split_by, execute, mode=None):
    child = super(PrePostChange,
                  self).compute_children_sql(table, split_by, execute, mode)
    covariates = child.iloc[:, 1:]
    child = child.iloc[:, :1]
    return self.adjust_value(child, covariates, split_by)


class CUPED(AbsoluteChange):
  """CUPED change estimator on a Metric.

  Computes the absolute change after controlling for preperiod metrics.
  Essentially, if the data only has a baseline and a treatment slice, CUPED
  1. centers the covariates
  2. fit child ~ intercept + covariate.
  And the intercept is the adjusted effect and has a smaller variance than
  child. See https://exp-platform.com/cuped for more details.
  If child returns multiple columns, the result is same as applying the method
  to every column in it.

  Attributes:
    extra_split_by: The column(s) that contains the conditions.
    baseline_key: The value of the condition that represents the baseline (e.g.,
      "Control"). All conditions will be compared to this baseline. If
      condition_column contains multiple columns, then baseline_key should be a
      tuple.
    children: A MetricList whose first element is the Metric we want to compute
      change on and the rest is the covariates for adjustment.
    include_base: A boolean for whether the baseline condition should be
      included in the output.
    And all other attributes inherited from Operation.
  """

  def __init__(self,
               condition_column,
               baseline_key,
               child=None,
               covariates=None,
               stratified_by=None,
               include_base=False,
               **kwargs):
    if isinstance(covariates, (List, Tuple)):
      covariates = metrics.MetricList(covariates)
    if child and covariates:
      child = metrics.MetricList((child, covariates)) if child else covariates
    else:
      child = None
    stratified_by = [stratified_by] if isinstance(stratified_by,
                                                  str) else stratified_by or []
    condition_column = [condition_column] if isinstance(
        condition_column, str) else condition_column
    super(CUPED,
          self).__init__(stratified_by + condition_column, baseline_key, child,
                         include_base, '{} CUPED Change', **kwargs)
    self.extra_index = condition_column
    self.computable_in_pure_sql = False

  def compute_children(self,
                       df,
                       split_by,
                       melted=False,
                       return_dataframe=True,
                       cache_key=None):
    child, covariates = super(CUPED, self).compute_children(
        df, split_by, return_dataframe=False, cache_key=cache_key)
    original_split_by = [s for s in split_by if s not in self.extra_split_by]
    return self.adjust_value(child, covariates, original_split_by)

  def adjust_value(self, child, covariates, split_by):
    """Adjust the raw value by controlling for Pre-metrics.

    Args:
      child: A pandas DataFrame. The result of the child Metric.
      covariates: A pandas DataFrame. The result of the covariates Metric.
      split_by: The split_by passed to self.compute_on().

    Returns:
      The adjusted values of the child (post metrics).
    """
    from sklearn import linear_model  # pylint: disable=g-import-not-at-top
    # Don't use "-=". For multiindex it might go wrong. The reason is DataFrame
    # has different implementations for __sub__ and __isub__. ___isub__ tries
    # to reindex to update in place which sometimes lead to lots of NAs.
    if split_by:
      covariates = covariates - covariates.groupby(split_by).mean()
    else:
      covariates = covariates - covariates.mean()
    # Align child with covariates in case there is any missing slices.
    covariates = covariates.reorder_levels(child.index.names)
    aligned = pd.concat([child, covariates], axis=1)
    len_child = child.shape[1]
    lm = linear_model.LinearRegression()

    def adjust(df_slice):
      child_slice = df_slice.iloc[:, :len_child]
      cov = df_slice.iloc[:, len_child:]
      adjusted = df_slice.groupby(self.extra_index).mean()
      for c in aligned.iloc[:, :len_child]:
        theta = lm.fit(cov, child_slice[c]).coef_
        adjusted[c] = adjusted[c] - adjusted.iloc[:, len_child:].dot(theta)
      return adjusted.iloc[:, :len_child]

    adjusted = metrics.Metric('Adjusted', compute=adjust)
    return adjusted.compute_on(aligned, split_by)

  def compute_children_sql(self, table, split_by, execute, mode=None):
    child = super(CUPED, self).compute_children_sql(table, split_by, execute,
                                                    mode)
    covariates = child.iloc[:, 1:]
    child = child.iloc[:, :1]
    return self.adjust_value(child, covariates, split_by)


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
    condition_column = [condition_column] if isinstance(
        condition_column, str) else condition_column
    super(MH, self).__init__(
        condition_column + self.stratified_by,
        baseline_key,
        metric,
        include_base,
        '{} MH Ratio',
        extra_index=condition_column,
        **kwargs)

  def check_is_ratio(self, metric, allow_metriclist=True):
    if isinstance(metric, metrics.MetricList) and allow_metriclist:
      for m in metric:
        self.check_is_ratio(m, False)
    else:
      if not isinstance(
          metric,
          (metrics.CompositeMetric, metrics.Ratio)) or metric.op(2.0, 2) != 1:
        raise ValueError(
            'MH only makes sense on ratio Metrics or a MetricList of ratios. Got %s.'
            % metric)

  def compute_children(self,
                       df: pd.DataFrame,
                       split_by=None,
                       melted=False,
                       return_dataframe=True,
                       cache_key=None):
    child = self.children[0]
    self.check_is_ratio(child)
    if isinstance(child, metrics.MetricList):
      children = []
      for m in child.children:
        util_metric = metrics.MetricList(
            [metrics.MetricList(m.children, where=m.where_raw)],
            where=child.where_raw)
        children.append(
            self.compute_util_metric_on(
                util_metric, df, split_by, cache_key=cache_key))
      return children
    util_metric = metrics.MetricList(child.children, where=child.where_raw)
    return self.compute_util_metric_on(
        util_metric, df, split_by, cache_key=cache_key)

  def compute_on_children(self, children, split_by):
    child = self.children[0]
    if isinstance(child, metrics.MetricList):
      res = [
          self.compute_mh(c, d, split_by)
          for c, d in zip(child.children, children)
      ]
      return pd.concat(res, 1, sort=False)
    return self.compute_mh(child, children, split_by)

  def compute_mh(self, child, df_all, split_by):
    """Computes MH statistics for one Metric."""
    level = self.extra_index[0] if len(
        self.extra_index) == 1 else self.extra_index
    df_baseline = df_all.xs(self.baseline_key, level=level)
    suffix = '_base'
    numer = child.children[0].name
    denom = child.children[1].name
    df_mh = df_all.join(df_baseline, rsuffix=suffix)
    ka, na = df_mh[numer], df_mh[denom]
    kb, nb = df_mh[numer + suffix], df_mh[denom + suffix]
    weights = 1. / (na + nb)
    to_split = [i for i in ka.index.names if i not in self.stratified_by]
    res = ((ka * nb * weights).groupby(to_split).sum() /
           (kb * na * weights).groupby(to_split).sum() - 1) * 100
    res.name = child.name
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

  def compute_children_sql(self, table, split_by, execute, mode=None):
    child = self.children[0]
    self.check_is_ratio(child)
    if isinstance(child, metrics.MetricList):
      children = []
      for m in child.children:
        util_metric = metrics.MetricList(
            [metrics.MetricList(m.children, where=m.where_raw)],
            where=child.where_raw)
        c = self.compute_util_metric_on_sql(
            util_metric,
            table,
            split_by + self.extra_split_by,
            execute,
            mode=mode)
        children.append(c)
      return children
    util_metric = metrics.MetricList(child.children, where=child.where_raw)
    return self.compute_util_metric_on_sql(
        util_metric, table, split_by + self.extra_split_by, execute, mode=mode)

  def get_sql_and_with_clause(self, table, split_by, global_filter, indexes,
                              local_filter, with_data):
    """Gets the SQL query and WITH clause.

    The query is constructed in a similar way to AbsoluteChange except that we
    apply weights to adjust the change.

    For example, the query for
    MH('condition', 'base_value', 'stratified',
                  metrics.Ratio('click', 'impression', 'ctr'))
    will look like this:

    WITH
    MHRaw AS (SELECT
      split_by,
      condition,
      stratified,
      SUM(click) AS `sum(click)`,
      SUM(impression) AS `sum(impression)`
    FROM $DATA
    GROUP BY split_by, condition, stratified),
    MHBase AS (SELECT
      split_by,
      stratified,
      `sum(click)`,
      `sum(impression)`
    FROM MHRaw
    WHERE
    condition = "base_value")
    SELECT
      split_by,
      condition,
      100 * SAFE_DIVIDE(
        COALESCE(SUM(SAFE_DIVIDE(MHRaw.`sum(click)` * MHBase.`sum(impression)`,
            MHBase.`sum(impression)` + MHRaw.`sum(impression)`)), 0),
        COALESCE(SUM(SAFE_DIVIDE(MHBase.`sum(click)` * MHRaw.`sum(impression)`,
            MHBase.`sum(impression)` + MHRaw.`sum(impression)`)), 0)) - 100
        AS `ctr MH Ratio`
    FROM MHRaw
    JOIN
    MHBase
    USING (split_by, stratified)
    WHERE
    condition != "base_value"
    GROUP BY split_by, condition

    Args:
      table: The table we want to query from.
      split_by: The columns that we use to split the data.
      global_filter: The sql.Filters that can be applied to the whole Metric
        tree.
      indexes: The columns that we shouldn't apply any arithmetic operation.
      local_filter: The sql.Filters that have been accumulated so far.
      with_data: A global variable that contains all the WITH clauses we need.

    Returns:
      The SQL instance for metric, without the WITH clause component.
      The global with_data which holds all datasources we need in the WITH
        clause.
    """
    child = self.children[0]
    self.check_is_ratio(child)
    local_filter = sql.Filters([self.where, local_filter]).remove(global_filter)

    grandchildren = []
    if isinstance(child, metrics.MetricList):
      grandchildren = []
      for m in child:
        grandchildren.append(metrics.MetricList(m.children, where=m.where_raw))
      util_metric = metrics.MetricList(grandchildren, where=child.where_raw)
    else:
      util_metric = metrics.MetricList(child.children, where=child.where_raw)

    cond_cols = sql.Columns(self.extra_index)
    groupby = sql.Columns(split_by).add(self.extra_split_by)
    util_indexes = sql.Columns(indexes).add(self.stratified_by)
    raw_table_sql, with_data = util_metric.get_sql_and_with_clause(
        table, groupby, global_filter, util_indexes, local_filter, with_data)

    raw_table = sql.Datasource(raw_table_sql, 'MHRaw')
    raw_table_alias, _ = with_data.merge(raw_table)

    base = self.baseline_key if isinstance(self.baseline_key,
                                           tuple) else [self.baseline_key]
    base_cond = ('%s = %s' % (c, _format_to_condition(b))
                 for c, b in zip(cond_cols.aliases, base))
    base_cond = ' AND '.join(base_cond)
    base_value = sql.Sql(
        sql.Columns(raw_table_sql.groupby.aliases).add(
            raw_table_sql.columns.aliases).difference(cond_cols.aliases),
        raw_table_alias, base_cond)
    base_table = sql.Datasource(base_value, 'MHBase')
    base_table_alias, _ = with_data.merge(base_table)

    exclude_base_condition = ('%s != %s' % (c, _format_to_condition(b))
                              for c, b in zip(cond_cols.aliases, base))
    exclude_base_condition = ' OR '.join(exclude_base_condition)
    cond = None if self.include_base else sql.Filters([exclude_base_condition])
    col_tmpl = """100 * SAFE_DIVIDE(
      COALESCE(SUM(SAFE_DIVIDE(
        {raw}.%(numer)s * {base}.%(denom)s,
        {base}.%(denom)s + {raw}.%(denom)s)), 0),
      COALESCE(SUM(SAFE_DIVIDE(
        {base}.%(numer)s * {raw}.%(denom)s,
        {base}.%(denom)s + {raw}.%(denom)s)), 0)) - 100"""
    col_tmpl = col_tmpl.format(raw=raw_table_alias, base=base_table_alias)
    columns = sql.Columns()
    alias_tmpl = self.name_tmpl
    # The columns might get consolidated and have different aliases. We need to
    # find them by reconstruction. Reusing the with_data in reconstruction will
    # make sure the columns get renamed the same way as in raw_table_sql.
    if isinstance(child, metrics.MetricList):
      for c in child:
        with_data2 = copy.deepcopy(with_data)
        util = metrics.MetricList(c.children[:1], where=c.where_raw)
        numer_sql, with_data2 = util.get_sql_and_with_clause(
            table, groupby, global_filter, util_indexes, local_filter,
            with_data2)
        with_data2.merge(sql.Datasource(numer_sql))
        numer = numer_sql.columns[-1].alias
        with_data2 = copy.deepcopy(with_data)
        util = metrics.MetricList(c.children[1:], where=c.where_raw)
        denom_sql, with_data2 = util.get_sql_and_with_clause(
            table, groupby, global_filter, util_indexes, local_filter,
            with_data2)
        with_data2.merge(sql.Datasource(denom_sql))
        denom = denom_sql.columns[-1].alias
        columns.add(
            sql.Column(
                col_tmpl % {
                    'numer': numer,
                    'denom': denom
                },
                alias=alias_tmpl.format(c.name)))
    else:
      with_data2 = copy.deepcopy(with_data)
      util = metrics.MetricList(child.children[:1], where=child.where_raw)
      numer_sql, with_data2 = util.get_sql_and_with_clause(
          table, groupby, global_filter, util_indexes, local_filter, with_data2)
      with_data2.merge(sql.Datasource(numer_sql))
      numer = numer_sql.columns[-1].alias
      with_data2 = copy.deepcopy(with_data)
      util = metrics.MetricList(child.children[1:], where=child.where_raw)
      denom_sql, with_data2 = util.get_sql_and_with_clause(
          table, groupby, global_filter, util_indexes, local_filter, with_data2)
      with_data2.merge(sql.Datasource(denom_sql))
      denom = denom_sql.columns[-1].alias
      columns = sql.Column(
          col_tmpl % {
              'numer': numer,
              'denom': denom,
          },
          alias=alias_tmpl.format(child.name))

    using = indexes.difference(cond_cols).add(self.stratified_by)
    return sql.Sql(columns,
                   sql.Join(raw_table_alias, base_table_alias, using=using),
                   cond, indexes.aliases), with_data


def get_display_fn(name,
                   split_by=None,
                   melted=False,
                   value='Value',
                   condition_column: Optional[List[Text]] = None,
                   ctrl_id=None,
                   default_metric_formats=None):
  """Returns a function that displays confidence interval nicely.

  Args:
    name: 'Jackknife' or 'Bootstrap'.
    split_by: The split_by passed to Jackknife().compute_on().
    melted: Whether the input res is in long format.
    value: The name of the value column.
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
        [{'column': ('CI_Lower', 'Metric Foo'), 'ascending': False},
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
    base = res.meterstick_change_base
    if not melted:
      res = utils.melt(res)
    if base is not None:
      # base always has the baseline so needs to be at left.
      res = base.join(res)
      comparison_suffix = [
          AbsoluteChange('', '').name_tmpl.format(''),
          PercentChange('', '').name_tmpl.format('')
      ]
      comparison_suffix = '(%s)$' % '|'.join(comparison_suffix)
      # Don't use inplace=True. It will change the index of 'base' too.
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
  bounds. Then we compute the point estimates and combine it with stderr or CI.
  Similar to how you derive Metric, if you don't need vectorization, overwrite
  compute(). If you need vectorization, overwrite compute_slices(), or even
  simpler, get_samples(). See Jackknife and Bootstrap for examples.

  Attributes:
    unit: The column to go over (jackknife/bootstrap over) to get stderr.
    confidence: The level of the confidence interval, must be in (0, 1). If
      specified, we return confidence interval range instead of standard error.
      Additionally, a display() function will be bound to the result so you can
      visualize the confidence interval nicely in Colab and Jupyter notebook.
    prefix: In the result, the column names will be like "{prefix} SE",
      "{prefix} CI-upper".
    sql_batch_size: The number of resamples to compute in one SQL run. It only
      has effect in the 'mixed' mode of compute_on_sql(). Note that you can also
      specify batch_size in compute_on_sql() directly, which precedes this one.
    And all other attributes inherited from Operation.
  """

  def __init__(self,
               unit: Optional[Text],
               child: Optional[metrics.Metric] = None,
               confidence: Optional[float] = None,
               name_tmpl: Optional[Text] = None,
               prefix: Optional[Text] = None,
               additional_fingerprint_attrs=None,
               sql_batch_size=None,
               **kwargs):
    if confidence and not 0 < confidence < 1:
      raise ValueError('Confidence must be in (0, 1).')
    self.unit = unit
    self.confidence = confidence
    additional_fingerprint_attrs = additional_fingerprint_attrs or []
    additional_fingerprint_attrs += ['unit', 'confidence']
    super(MetricWithCI, self).__init__(
        child,
        name_tmpl,
        additional_fingerprint_attrs=additional_fingerprint_attrs,
        **kwargs)
    self.apply_name_tmpl = False
    self.prefix = prefix
    self.sql_batch_size = sql_batch_size
    if not self.prefix and self.name_tmpl:
      self.prefix = prefix or self.name_tmpl.format('').strip()
    self.precomputable_in_jk = False

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
        if cache_key is None:
          # If samples are unlikely to repeat, don't save res to self.cache.
          res = self.children[0].compute_on(sample, split_by, melted=True)
        else:
          res = self.compute_child(
              sample, split_by, melted=True, cache_key=cache_key)
        estimates.append(res)
      except Exception as e:  # pylint: disable=broad-except
        print(
            'Warning: Failed on %s sample data for reason %s. If you see many '
            'such failures, your data might be too sparse.' %
            (self.name_tmpl.format(''), repr(e)))
    return estimates

  def compute_children(self,
                       df: pd.DataFrame,
                       split_by=None,
                       melted=False,
                       return_dataframe=True,
                       cache_key=None):
    del melted, return_dataframe, cache_key  # unused
    return self.compute_on_samples(self.get_samples(df, split_by), split_by)

  def compute_on_children(self, children, split_by):
    del split_by  # unused
    bucket_estimates = pd.concat(children, axis=1, sort=False)
    return self.get_stderrs_or_ci_half_width(bucket_estimates)

  def manipulate(self,
                 res,
                 melted=False,
                 return_dataframe=True,
                 apply_name_tmpl=False):
    """Saves and restores the base in addition when has confidence."""
    if self.confidence:
      key = self.wrap_cache_key(self.cache_key)
      key.add_extra_info('base')
      if hasattr(res, 'meterstick_change_base'):
        # If res is computed from input data, it will have the attribute.
        base = res.meterstick_change_base
        self.save_to_cache(key, base)
      else:
        # If res is read from cache, it won't have the attribute, but it must
        # have been computed already so base has been saved in cache.
        base = self.find_all_in_cache(key)
    # Don't add suffix like "Jackknife" because point_est won't have it.
    res = super(MetricWithCI, self).manipulate(res, melted, return_dataframe,
                                               apply_name_tmpl)
    return self.add_base_to_res(res, base) if self.confidence else res

  def compute_slices(self, df, split_by):
    std = super(MetricWithCI, self).compute_slices(df, split_by)
    point_est = self.compute_child(df, split_by, melted=True)
    res = point_est.join(utils.melt(std))
    if self.confidence:
      res[self.prefix +
          ' CI-lower'] = res.iloc[:, 0] - res[self.prefix + ' CI-lower']
      res[self.prefix + ' CI-upper'] += res.iloc[:, 0]
    res = utils.unmelt(res)
    if not self.confidence:
      return res
    base = self.compute_change_base(df, split_by)
    return self.add_base_to_res(res, base)

  def compute_change_base(self,
                          df,
                          split_by,
                          execute=None,
                          mode=None,
                          cache_key=None):
    """Computes the base values for Change. It's used in res.display()."""
    if not self.confidence:
      return None
    if len(self.children) != 1 or not isinstance(
        self.children[0], (PercentChange, AbsoluteChange)):
      return None
    base = None
    change = self.children[0]
    util_metric = change.children[0]
    if isinstance(self.children[0], (PrePostChange, CUPED)):
      util_metric = metrics.MetricList([util_metric.children[0]],
                                       where=util_metric.where_raw)
    util_metric = metrics.MetricList([util_metric], where=change.where_raw)
    to_split = (
        split_by + change.extra_index if split_by else change.extra_index)
    if execute is None:
      base = self.compute_util_metric_on(
          util_metric, df, to_split, cache_key=cache_key)
    else:
      base = self.compute_util_metric_on_sql(
          util_metric, df, to_split, execute, mode=mode, cache_key=cache_key)
    base.columns = [change.name_tmpl.format(c) for c in base.columns]
    base = utils.melt(base)
    base.columns = ['_base_value']
    return base

  @staticmethod
  def add_base_to_res(res, base):
    with warnings.catch_warnings():
      warnings.simplefilter(action='ignore', category=UserWarning)
      res.meterstick_change_base = base
    return res

  def final_compute(self,
                    res,
                    melted: bool = False,
                    return_dataframe: bool = True,
                    split_by: Optional[List[Text]] = None,
                    df=None):
    """Add a display function if confidence is specified."""
    del return_dataframe  # unused
    if self.confidence:
      indexes = list(res.index.names)
      if melted:
        indexes = indexes[1:]
      if len(self.children) == 1 and isinstance(
          self.children[0], (PercentChange, AbsoluteChange)):
        change = self.children[0]
        indexes = [i for i in indexes if i not in change.extra_index]
      res = self.add_display_fn(res, indexes, melted)
    return res

  def add_display_fn(self, res, split_by, melted, base=None):
    """Bounds a display function to res so res.display() works."""
    if base is not None or not hasattr(res, 'meterstick_change_base'):
      self.add_base_to_res(res, base)
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

    fn = get_display_fn(self.prefix, split_by, melted, value, condition_col,
                        ctrl_id, metric_formats)
    # pylint: disable=no-value-for-parameter
    res.display = fn.__get__(res)  # pytype: disable=attribute-error
    # pylint: enable=no-value-for-parameter
    return res

  @staticmethod
  def get_stderrs(bucket_estimates):
    dof = bucket_estimates.count(axis=1) - 1
    return bucket_estimates.std(1), dof

  def get_ci_width(self, stderrs, dof):
    """You can return asymmetrical confidence interval."""
    dof = dof.fillna(0).astype(int)  # Scipy might not recognize the Int64 type.
    half_width = stderrs * stats.t.ppf((1 + self.confidence) / 2, dof)
    return half_width, half_width

  def get_stderrs_or_ci_half_width(self, bucket_estimates):
    """Returns confidence interval infomation in an unmelted DataFrame."""
    stderrs, dof = self.get_stderrs(bucket_estimates)
    if self.confidence:
      res = pd.DataFrame(self.get_ci_width(stderrs, dof)).T
      res.columns = [self.prefix + ' CI-lower', self.prefix + ' CI-upper']
    else:
      res = pd.DataFrame(stderrs, columns=[self.prefix + ' SE'])
    res = utils.unmelt(res)
    return res

  def get_samples(self, df, split_by=None):
    raise NotImplementedError

  def get_sql_and_with_clause(self, table, split_by, global_filter, indexes,
                              local_filter, with_data):
    """Gets the SQL for Jackknife or Bootstrap.

    The query is constructed by
    1. Resample the table.
    2. Compute the child Metric on the resampled data.
    3. Compute the standard error from #2.
    4. Compute the point estimate from original table.
    5. sql.Join #3 and #4.
    6. If metric has confidence level specified, we also get the degrees of
      freedom so we can later compute the critical value of t distribution in
      Python.
    7. If metric only has one child and it's PercentChange or AbsoluteChange, we
      also get the base values for comparison. They will be used in the
      res.display().

    Args:
      table: The table we want to query from.
      split_by: The columns that we use to split the data.
      global_filter: The sql.Filters that can be applied to the whole Metric
        tree.
      indexes: The columns that we shouldn't apply any arithmetic operation.
      local_filter: The sql.Filters that have been accumulated so far.
      with_data: A global variable that contains all the WITH clauses we need.

    Returns:
      The SQL instance for metric, without the WITH clause component.
      The global with_data which holds all datasources we need in the WITH
        clause.
    """
    if not isinstance(self, (Jackknife, Bootstrap)):
      raise ValueError('Not a Jackknife or Bootstrap!')
    local_filter = sql.Filters([self.where, local_filter]).remove(global_filter)

    name = 'Jackknife' if isinstance(self, Jackknife) else 'Bootstrap'
    self_copy = copy.deepcopy(self)
    se, with_data = get_se(self_copy, table, split_by, global_filter, indexes,
                           local_filter, with_data)
    se_alias, se_rename = with_data.merge(sql.Datasource(se, name + 'SE'))

    pt_est, with_data = self.children[0].get_sql_and_with_clause(
        table, split_by, global_filter, indexes, local_filter, with_data)
    pt_est_alias, pt_est_rename = with_data.merge(
        sql.Datasource(pt_est, name + 'PointEstimate'))

    columns = sql.Columns()
    using = sql.Columns(se.groupby)
    pt_est_col = []
    for c in pt_est.columns:
      if c in indexes.aliases:
        using.add(c)
      else:
        pt_est_col.append(
            sql.Column(
                '%s.%s' % (pt_est_alias, pt_est_rename.get(c.alias, c.alias)),
                alias=c.alias_raw))
    se_cols = []
    for c in se.columns:
      if c not in indexes.aliases:
        se_cols.append(
            sql.Column(
                '%s.%s' % (se_alias, se_rename.get(c.alias, c.alias)),
                alias=c.alias_raw))
    if self.confidence:
      dof_cols = se_cols[1::2]
      se_cols = se_cols[::2]
      cols = zip(pt_est_col, se_cols, dof_cols)
    else:
      cols = zip(pt_est_col, se_cols)
    columns.add(cols)

    has_base_vals = False
    if self.confidence:
      child = self.children[0]
      if len(self.children) == 1 and isinstance(
          child, (PercentChange, AbsoluteChange)):
        has_base_vals = True
        base, with_data = child.children[0].get_sql_and_with_clause(
            table,
            sql.Columns(split_by).add(child.extra_index), global_filter,
            indexes, local_filter, with_data)
        base_alias, base_rename = with_data.merge(
            sql.Datasource(base, '_ShouldAlreadyExists'))
        columns.add(
            sql.Column(
                '%s.%s' % (base_alias, base_rename.get(c.alias, c.alias)),
                alias=c.alias_raw) for c in base.columns.difference(indexes))

    join = 'LEFT' if using else 'CROSS'
    from_data = sql.Join(pt_est_alias, se_alias, join=join, using=using)
    if has_base_vals:
      from_data = from_data.join(base_alias, join=join, using=using)
    return sql.Sql(using.add(columns), from_data), with_data

  def compute_on_sql(
      self,
      table,
      split_by=None,
      execute=None,
      melted=False,
      mode=None,
      cache_key=None,
      cache=None,
      batch_size=None,
  ):
    """Computes self in pure SQL or a mixed of SQL and Python.

    Args:
      table: The table we want to query from.
      split_by: The columns that we use to split the data.
      execute: A function that can executes a SQL query and returns a DataFrame.
      melted: Whether to transform the result to long format.
      mode: For Operations, there are two ways to compute the result in SQL, one
        is computing everything in SQL, the other is computing the children
        in SQL then the rest in Python. We call them 'sql' and 'mixed' modes. If
        self has grandchildren, then we can compute the chilren in two modes
        too. We can call them light `mixed` mode and recursive `mixed` mode.
        If `mode` is 'sql', it computes everything in SQL.
        If `mode` is 'mixed', it computes everything recursively in the `mixed`
        mode.
        We recommend `mode` to be None. This mode tries the `sql` mode first, if
        not implemented, switch to light `mixed` mode. The logic is applied
        recursively from the root to leaf Metrics, so a Metric tree could have
        top 3 layeres computed in Python and the bottom in SQL. In summary,
        everything can be computed in SQL is computed in SQL.
      cache_key: What key to use to cache the result. You can use anything that
        can be a key of a dict except '_RESERVED' and tuples like
        ('_RESERVED', ..).
      cache: The global cache the whole Metric tree shares. If it's None, we
        initiate an empty dict.
      batch_size: The number of resamples to compute in one SQL run. It only has
        effect in the 'mixed' mode. It precedes self.batch_size.

    Returns:
      A pandas DataFrame. It's the computeation of self in SQL.
    """
    self._runtime_batch_size = batch_size
    try:
      return super(MetricWithCI,
                   self).compute_on_sql(table, split_by, execute, melted, mode,
                                        cache_key, cache)
    finally:
      self._runtime_batch_size = None

  def compute_through_sql(self, table, split_by, execute, mode):
    if mode not in (None, 'sql', 'mixed', 'magic'):
      raise ValueError('Mode %s is not supported!' % mode)
    if mode in (None, 'sql'):
      if self.all_computable_in_pure_sql(False):
        try:
          return self.compute_on_sql_sql_mode(table, split_by, execute)
        except Exception as e:  # pylint: disable=broad-except
          raise utils.MaybeBadSqlModeError(use_batch_size=True) from e
      elif mode == 'sql':
        raise ValueError('%s is not computable in pure SQL.' % self.name)
    if self.where:
      table = sql.Sql(sql.Column('*', auto_alias=False), table, self.where)
    return self.compute_on_sql_mixed_mode(table, split_by, execute, mode)

  def compute_on_sql_sql_mode(self, table, split_by, execute):
    """Computes self in a SQL query and process the result."""
    res = super(MetricWithCI,
                self).compute_on_sql_sql_mode(table, split_by, execute)
    sub_dfs = []
    base = None
    if self.confidence:
      if len(self.children) == 1 and isinstance(
          self.children[0], (PercentChange, AbsoluteChange)):
        # The first 3n columns are Value, SE, dof for n Metrics. The
        # last n columns are the base values of Change.
        if len(res.columns) % 4:
          raise ValueError('Wrong shape for a MetricWithCI with confidence!')
        n_metrics = len(res.columns) // 4
        base = res.iloc[:, -n_metrics:]
        res = res.iloc[:, :3 * n_metrics]
        change = self.children[0]
        base.columns = [change.name_tmpl.format(c) for c in base.columns]
        base = utils.melt(base)
        base.columns = ['_base_value']

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
                ci_upper: pt_est + half_width[1]
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
    return self.add_base_to_res(res, base)

  def compute_on_sql_mixed_mode(self, table, split_by, execute, mode=None):
    batch_size = self._runtime_batch_size or self.sql_batch_size
    try:
      replicates = self.compute_children_sql(table, split_by, execute, mode,
                                             batch_size)
    except utils.MaybeBadSqlModeError:
      raise
    except Exception as e:  # pylint: disable=broad-except
      raise utils.MaybeBadSqlModeError(batch_size=batch_size) from e
    std = self.compute_on_children(replicates, split_by)
    point_est = self.compute_child_sql(
        table, split_by, execute, True, mode=mode)
    res = point_est.join(utils.melt(std))
    if self.confidence:
      res[self.prefix +
          ' CI-lower'] = res.iloc[:, 0] - res[self.prefix + ' CI-lower']
      res[self.prefix + ' CI-upper'] += res.iloc[:, 0]
    res = utils.unmelt(res)
    base = self.compute_change_base(table, split_by, execute, mode)
    return self.add_base_to_res(res, base)

  def compute_children_sql(self,
                           table,
                           split_by,
                           execute,
                           mode=None,
                           batch_size=None):
    """The return should be similar to compute_children()."""
    raise NotImplementedError


class Jackknife(MetricWithCI):
  """Class for Jackknife estimates of standard errors.

  Attributes:
    unit: The column whose levels define the jackknife buckets.
    confidence: The level of the confidence interval, must be in (0, 1). If
      specified, we return confidence interval range instead of standard error.
      Additionally, a display() function will be bound to the result so you can
      visualize the confidence interval nicely in Colab and Jupyter notebook.
    children: A tuple of a Metric whose result we jackknife on.
    enable_optimization: If all leaf Metrics are Sum and/or Count, or can be
      expressed equivalently by Sum and/or Count, then we can cut the corner to
      compute leave-one-out estimates. See compute_slices() for more details.
    And all other attributes inherited from Operation.
  """

  def __init__(self,
               unit: Text,
               child: Optional[metrics.Metric] = None,
               confidence: Optional[float] = None,
               enable_optimization=True,
               **kwargs):
    super(Jackknife, self).__init__(unit, child, confidence, '{} Jackknife',
                                    **kwargs)
    self.enable_optimization = enable_optimization
    if confidence:
      self.computable_in_pure_sql = False

  def compute_slices(self, df, split_by=None):
    """Computes Jackknife with precomputation when possible.

    For Sum, Count, it's possible to compute the LOO estimates in a vectorized
    way. The leave-one-out (LOO) estimates are the differences between the
    sum/count of each bucket and the total. The trick applies to other Metrics
    that can be expressed by Sum and Count too, for example, Mean(x) which is
    equivalent to Sum(x) / Count(x). self.can_precompute() means all the leaf
    Metrics in self can be expressed by Sum or Count so we apply the trick by
    1. replace self with an equivalent tree whose leaf Metrics are all Sum or
    Count.
    2. computes it on self.unit + split_by.
    3. loop through the cache and find all Sum and Count results in #2. Use them
    to compute the LOOs and save to cache.
    4. call super().compute_slices() which will just hits the cached results.

    Args:
      df: The DataFrame to compute on.
      split_by: The columns that we use to split the data.

    Returns:
      The unmelted result.
    """
    if not self.can_precompute():
      return super(Jackknife, self).compute_slices(df, split_by)
    self.compute_child(df, split_by + [self.unit])
    precomputed = self.find_all_in_cache(metric=metrics.Sum)
    precomputed.update(self.find_all_in_cache(metric=metrics.Count))
    precomputed = {
        k: v for k, v in precomputed.items() if k.key == self.cache_key.key
    }
    for key, each_bucket in precomputed.items():
      self.precompute_sum_or_count_for_jackknife(key, each_bucket, split_by, df)
    return super(Jackknife, self).compute_slices(df, split_by)

  def precompute_sum_or_count_for_jackknife(self, cache_key, each_bucket,
                                            original_split_by, df):
    """Caches point estimate and leave-one-out (LOO) results for Sum and Count.

    For Sum, Count, it's possible to compute the LOO estimates in a vectorized
    way. The LOO estimates are the differences between the sum/count of each
    bucket and the total. We have computed the results sliced by self.unit in
    compute_slices() and here we compute the LOOs and save them to the cache
    under certain keys which will be used in compute_children() too.

    Args:
      cache_key: The key in the cache for a sum or count result that is split by
        self.unit.
      each_bucket: The result saved under cache_key.
      original_split_by: The split_by passed to self.compute_on().
      df: The df Jackknife computes on.

    Returns:
      None. Two additional results are saved to the cache.
      1. The total sum/count, which is each_bucket summed over self.unit.
      2. The LOO estimates, which is saved under key
        ('_RESERVED', 'Jackknife', self.unit).
    """
    if not cache_key.split_by:
      return
    if cache_key.split_by[:len(original_split_by) +
                          1] != tuple(original_split_by + [self.unit]):
      return
    split_by_with_unit = list(cache_key.split_by)
    split_by = [i for i in split_by_with_unit if i != self.unit]
    if split_by:
      total = each_bucket.groupby(level=split_by).sum()
    else:
      total = each_bucket.sum()
    key = copy.deepcopy(cache_key)
    key.split_by = tuple(split_by)
    self.save_to_cache(key, total)

    key = cache_key.replace_key(('_RESERVED', 'Jackknife', self.unit))
    if not self.in_cache(key):
      each_bucket = utils.adjust_slices_for_loo(each_bucket, original_split_by,
                                                df)
      loo = total - each_bucket
      if split_by:
        # The levels might get messed up.
        loo = loo.reorder_levels(split_by_with_unit)
      self.save_to_cache(key, loo)

  def get_samples(self, df, split_by=None):
    """Yields leave-one-out (LOO) DataFrame with level value.

    If self.can_precompute(), this function will not get triggered because the
    results have been precomputed and saved in cache.

    Args:
      df: The DataFrame to compute on.
      split_by: The columns that we use to split the data.

    Yields:
      ('_RESERVED', 'Jackknife', unit, i) and the leave-i-out DataFrame.
    """
    levels = df[self.unit].unique()
    if len(levels) < 2:
      raise ValueError('Too few %s to jackknife.' % self.unit)

    if not split_by:
      for lvl in levels:
        yield ('_RESERVED', 'Jackknife', self.unit, lvl), df[
            df[self.unit] != lvl
        ]
    else:
      df = df.set_index(split_by)
      max_slices = len(df.index.unique())
      for lvl, idx in df.groupby(self.unit).groups.items():
        df_rest = df[df[self.unit] != lvl]
        unique_slice_val = idx.unique()
        if len(unique_slice_val) != max_slices:
          # Keep only the slices that appeared in the dropped bucket.
          df_rest = df_rest[df_rest.index.isin(unique_slice_val)]
        yield ('_RESERVED', 'Jackknife', self.unit, lvl), df_rest.reset_index()

  def compute_children(self,
                       df: pd.DataFrame,
                       split_by=None,
                       melted=False,
                       return_dataframe=True,
                       cache_key=None):
    if not self.can_precompute():
      return super(Jackknife,
                   self).compute_children(df, split_by, melted,
                                          return_dataframe, cache_key)
    replicates = self.compute_child(
        df,
        split_by + [self.unit],
        True,
        cache_key=('_RESERVED', 'Jackknife', self.unit))
    return [replicates.unstack(self.unit)]

  @staticmethod
  def get_stderrs(bucket_estimates):
    stderrs, dof = super(Jackknife, Jackknife).get_stderrs(bucket_estimates)
    return stderrs * dof / np.sqrt(dof + 1), dof

  def can_precompute(self):
    """If all leafs can be expressed as Sum or Count, LOO can be precomputed."""
    if not self.enable_optimization:
      return False

    def precomputable(root):
      for m in root.traverse(include_self=False):
        if isinstance(m, Operation) and not m.precomputable_in_jk:
          return False
        if m.custom_fns:
          return False
        if isinstance(m, metrics.Count) and m.distinct:
          return False
        if isinstance(m, (metrics.Sum, metrics.Count)):
          continue
        if not m.children:
          return False
      return True

    return precomputable(self)

  def compute_children_sql(self, table, split_by, execute, mode, batch_size):
    """Compute the children on leave-one-out data in SQL."""
    batch_size = batch_size or 1
    slice_and_units = sql.Sql(
        sql.Columns(split_by + [self.unit], distinct=True), table, self.where)
    slice_and_units = execute(str(slice_and_units))
    if split_by:
      slice_and_units.set_index(split_by, inplace=True)
    replicates = []
    unique_units = slice_and_units[self.unit].unique()
    if batch_size == 1:
      loo_sql = sql.Sql(sql.Column('*', auto_alias=False), table)
      for unit in unique_units:
        loo_where = '%s != "%s"' % (self.unit, unit)
        if pd.api.types.is_numeric_dtype(slice_and_units[self.unit]):
          loo_where = '%s != %s' % (self.unit, unit)
        loo_sql.where = sql.Filters((self.where, loo_where))
        key = ('_RESERVED', 'Jackknife', self.unit, unit)
        loo = self.compute_child_sql(loo_sql, split_by, execute, False, mode,
                                     key)
        # If a slice doesn't have the unit in the input data, we should exclude
        # the slice in the loo.
        if split_by:
          loo = slice_and_units[slice_and_units[self.unit] == unit].join(loo)
          loo.drop(self.unit, 1, inplace=True)
        replicates.append(utils.melt(loo))
    else:
      if split_by:
        slice_and_units.set_index(self.unit, append=True, inplace=True)
      for i in range(int(np.ceil(len(unique_units) / batch_size))):
        units = list(unique_units[i * batch_size:(i + 1) * batch_size])
        loo = sql.Sql(
            sql.Column('*', auto_alias=False),
            sql.Datasource('UNNEST(%s)' % units, '_resample_idx').join(
                table, on='_resample_idx != %s' % self.unit), self.where)
        key = ('_RESERVED', 'Jackknife', self.unit, tuple(units))
        loo = self.compute_child_sql(loo, split_by + ['_resample_idx'], execute,
                                     True, mode, key)
        # If a slice doesn't have the unit in the input data, we should exclude
        # the slice in the loo.
        if split_by:
          loo.index.set_names(self.unit, level='_resample_idx', inplace=True)
          loo = slice_and_units.join(utils.unmelt(loo), how='inner')
          loo = utils.melt(loo).unstack(self.unit)
        else:
          loo = loo.unstack('_resample_idx')
        replicates.append(loo)
    return replicates


class Bootstrap(MetricWithCI):
  """Class for Bootstrap estimates of standard errors.

  Attributes:
    unit: The column representing the blocks to be resampled in block bootstrap.
      If specified we sample the unique blocks in the `unit` column, otherwise
      we sample rows.
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
    super(Bootstrap, self).__init__(
        unit,
        child,
        confidence,
        '{} Bootstrap',
        additional_fingerprint_attrs=['n_replicates'],
        **kwargs)
    self.n_replicates = n_replicates

  def get_samples(self, df, split_by=None):
    split_by = [split_by] if isinstance(split_by, str) else split_by or []
    if self.unit is None:
      to_sample = self.group(df, split_by)
      for _ in range(self.n_replicates):
        yield None, to_sample.sample(frac=1, replace=True)
    else:
      grp_by = split_by + [self.unit] if split_by else self.unit
      grped = df.groupby(grp_by)
      idx = grped.indices
      units_split_by = df[grp_by].drop_duplicates()
      for _ in range(self.n_replicates):
        resampled = self.group(units_split_by, split_by).sample(
            frac=1, replace=True).values
        if not split_by:
          yield None, df.iloc[np.concatenate([idx[i] for i in resampled])]
        else:
          yield None, df.iloc[np.concatenate([idx[tuple(i)] for i in resampled
                                             ])]

  def compute_children_sql(self, table, split_by, execute, mode, batch_size):
    """Compute the children on resampled data in SQL."""
    batch_size = batch_size or 1000
    global_filter = metrics.get_global_filter(self)
    util_metric = copy.deepcopy(self)
    util_metric.n_replicates = batch_size
    util_metric.confidence = None
    with_data = sql.Datasources()
    if not sql.Datasource(table).is_table:
      table = with_data.add(sql.Datasource(table, 'BootstrapData'))
    with_data2 = copy.deepcopy(with_data)
    _, with_data = get_bootstrap_data(util_metric, table, sql.Columns(split_by),
                                      global_filter, sql.Filters(), with_data)
    resampled = with_data.children.popitem()[1]
    resampled.with_data = with_data
    replicates = []
    for _ in range(self.n_replicates // batch_size):
      bst = self.children[0].compute_on_sql(resampled,
                                            ['_resample_idx'] + split_by,
                                            execute, True, mode)
      replicates.append(bst.unstack('_resample_idx'))
    util_metric.n_replicates = self.n_replicates % batch_size
    if util_metric.n_replicates:
      _, with_data2 = get_bootstrap_data(util_metric, table,
                                         sql.Columns(split_by), global_filter,
                                         sql.Filters(), with_data2)
      resampled = with_data2.children.popitem()[1]
      resampled.with_data = with_data2
      bst = self.children[0].compute_on_sql(resampled,
                                            ['_resample_idx'] + split_by,
                                            execute, True, mode)
      replicates.append(bst.unstack('_resample_idx'))
    return replicates


def get_se(metric, table, split_by, global_filter, indexes, local_filter,
           with_data):
  """Gets the SQL query that computes the standard error and dof if needed."""
  global_filter = sql.Filters([global_filter, local_filter]).add(metric.where)
  local_filter = sql.Filters()
  metric, table, split_by, global_filter, indexes, with_data = preaggregate_if_possible(
      metric, table, split_by, global_filter, indexes, with_data)

  if isinstance(metric, Jackknife):
    if metric.can_precompute():
      metric = copy.deepcopy(metric)  # We'll modify the metric tree in-place.
    table, with_data = get_jackknife_data(metric, table, split_by,
                                          global_filter, indexes, local_filter,
                                          with_data)
  else:
    table, with_data = get_bootstrap_data(metric, table, split_by,
                                          global_filter, local_filter,
                                          with_data)

  if isinstance(metric, Jackknife) and metric.can_precompute():
    split_by = adjust_indexes_for_jk_fast(split_by)
    indexes = adjust_indexes_for_jk_fast(indexes)
    # global_filter has been removed from all Metrics when precomputeing LOO.
    global_filter = sql.Filters(None)

  samples, with_data = metric.children[0].get_sql_and_with_clause(
      table,
      sql.Columns(split_by).add('_resample_idx'), global_filter,
      sql.Columns(indexes).add('_resample_idx'), local_filter, with_data)
  samples_alias, rename = with_data.merge(
      sql.Datasource(samples, 'ResampledResults'))

  columns = sql.Columns()
  groupby = sql.Columns(
      (c.alias for c in samples.groupby if c != '_resample_idx'))
  for c in samples.columns:
    if c == '_resample_idx':
      continue
    elif c in indexes.aliases:
      groupby.add(c.alias)
    else:
      alias = rename.get(c.alias, c.alias)
      se = sql.Column(c.alias, 'STDDEV_SAMP({})',
                      '%s Bootstrap SE' % c.alias_raw)
      if isinstance(metric, Jackknife):
        adjustment = sql.Column(
            'SAFE_DIVIDE((COUNT({c}) - 1), SQRT(COUNT({c})))'.format(c=alias))
        se = (se * adjustment).set_alias('%s Jackknife SE' % c.alias_raw)
      columns.add(se)
      if metric.confidence:
        columns.add(sql.Column(alias, 'COUNT({}) - 1', '%s dof' % c.alias_raw))
  return sql.Sql(columns, samples_alias, groupby=groupby), with_data


def preaggregate_if_possible(metric, table, split_by, global_filter, indexes,
                             with_data):
  """Preaggregates data to make the resampled table small.

  For Jackknife and Bootstrap over group, we may preaggegate the data to make
  the query more efficient, though there are some requirements.
  1. All leaf Metrics need to be Sum or Count. Namely, metric.can_precompute()
    is True.
  2. There cannot be any local filter in the metric tree, otherwise the filter
    might need access to original rows. Technically it's still possible to
    preaggregate when there are local filters, but we impose this requirement
    for code simplicity.

  If preaggregatable, we precompute the sum and count over all split_bys used by
  the metric tree, clear all the local filters, and replace Count(x) with
  Sum(preaggregated_x).

  Args:
    metric: An instance of Jackknife or Bootstrap.
    table: The table we want to query from.
    split_by: The columns that we use to split the data.
    global_filter: The filters that applied to the data for resampling..
    indexes: The columns that we shouldn't apply any arithmetic operation.
    with_data: A global variable that contains all the WITH clauses we need.

  Returns:
    If preaggregation is possible, we return
      Modified metric tree.
      The alias of the preaggregated table in the with_data.
      Modified split_by.
      Modified indexes.
      with_data, with preaggregated added.
    Otherwise, they are returned untouched.
  """
  if not isinstance(metric, Jackknife) and not (isinstance(metric, Bootstrap)
                                                and metric.unit):
    return metric, table, split_by, global_filter, indexes, with_data
  if isinstance(metric, Jackknife) and not metric.can_precompute():
    return metric, table, split_by, global_filter, indexes, with_data

  all_split_by = sql.Columns(indexes).add(utils.get_extra_split_by(metric)).add(
      metric.unit)
  cols = sql.Columns()
  for m in metric.traverse():
    if sql.Filters(m.where).remove(global_filter):
      return metric, table, split_by, global_filter, indexes, with_data
    if isinstance(m, metrics.Sum):
      cols.add(sql.Column(m.var, 'SUM({})'))
    elif isinstance(m, metrics.Count):
      cols.add(sql.Column(m.var, 'COUNT({})'))

  metric = copy.deepcopy(metric)
  metric.unit = sql.Column(metric.unit).alias
  todo = [metric]
  while todo:
    curr = todo.pop()
    curr.where = None
    new_children = list(curr.children)
    for i, m in enumerate(curr.children):
      if not isinstance(m, metrics.Metric):
        continue
      todo.append(m)
      if isinstance(m, Operation):
        m.extra_index = [sql.Column(i, alias=i).alias for i in m.extra_index]
        m.extra_split_by = [
            sql.Column(i, alias=i).alias for i in m.extra_split_by
        ]
        if isinstance(m, MH):
          m.stratified_by = sql.Column(m.stratified_by).alias
      elif isinstance(m, metrics.Sum):
        m.var = sql.Column(m.var, 'SUM({})').alias
      elif isinstance(m, metrics.Count):
        new_children[i] = metrics.Sum(sql.Column(m.var, 'COUNT({})').alias)
    curr.children = new_children

  preagg = sql.Sql(cols, table, global_filter, all_split_by)
  preagg_alias = with_data.add(sql.Datasource(preagg, 'Preaggregated'))

  return metric, preagg_alias, sql.Columns(
      split_by.aliases), sql.Filters(), sql.Columns(indexes.aliases), with_data


def adjust_indexes_for_jk_fast(indexes):
  """For the indexes that get renamed, only keep the alias.

  For a Jackknife that only has Sum and Count as leaf Metrics, we cut the corner
  by getting the LOO table first and select everything from it. See
  get_jackknife_data_fast() for how the LOO is constructed. If any of the index
  is renamed in LOO, for example, $Platform AS platform, then all the following
  selections from LOO should select 'platform' instead of $Platform. Here we
  adjust all these columns.

  Args:
    indexes: The columns that we shouldn't apply any arithmetic operation.

  Returns:
    Adjusted indexes. The new indexes won't have any column that needs an alias.
  """
  ind = list(indexes.children)
  for i, s in enumerate(indexes):
    if s != s.alias:
      ind[i] = sql.Column(s.alias)
  return sql.Columns(ind)


def get_jackknife_data(metric, table, split_by, global_filter, indexes,
                       local_filter, with_data):
  table = sql.Datasource(table)
  if not table.is_table:
    table.alias = table.alias or 'RawData'
    table = with_data.add(table)
  if metric.can_precompute():
    return get_jackknife_data_fast(metric, table, split_by, global_filter,
                                   indexes, local_filter, with_data)
  return get_jackknife_data_general(metric, table, split_by, global_filter,
                                    local_filter, with_data)


def get_jackknife_data_general(metric, table, split_by, global_filter,
                               local_filter, with_data):
  """Gets jackknife samples.

  If the leave-one-out estimates can be precomputed, see the doc of
  get_jackknife_data_fast().
  Otherwise for general cases, the SQL is constructed as
  1. if split_by is None:
    WITH
    Buckets AS (SELECT DISTINCT unit AS _resample_idx
    FROM $DATA
    WHERE filter),
    JackknifeResammpledData AS (SELECT
      *
    FROM Buckets
    CROSS JOIN
    $DATA
    WHERE
    _resample_idx != unit AND filter)

  2. if split_by is not None:
    WITH
    Buckets AS (SELECT DISTINCT
      split_by AS _jk_split_by,
      unit AS _resample_idx
    FROM $DATA
    WHERE filter
    GROUP BY _jk_split_by),
    JackknifeResammpledData AS (SELECT
      *
    FROM Buckets
    JOIN
    $DATA
    ON _jk_split_by = split_by AND _resample_idx != unit
    WHERE filter)

  Args:
    metric: An instance of Jackknife.
    table: The table we want to query from.
    split_by: The columns that we use to split the data.
    global_filter: The filters that can be applied to the whole Metric tree.
    local_filter: The filters that have been accumulated so far.
    with_data: A global variable that contains all the WITH clauses we need.

  Returns:
    The alias of the table in the WITH clause that has all resampled data.
    The global with_data which holds all datasources we need in the WITH clause.
  """
  unit = metric.unit
  unique_units = sql.Columns((sql.Column(unit, alias='_resample_idx')),
                             distinct=True)
  where = sql.Filters(global_filter).add(local_filter)
  if split_by:
    groupby = sql.Columns(
        (sql.Column(c.expression, alias='_jk_%s' % c.alias) for c in split_by))
    cols = sql.Columns(groupby.add(unique_units), distinct=True)
    buckets = sql.Sql(cols, table, where)
    buckets_alias = with_data.add(sql.Datasource(buckets, 'Buckets'))
    on = sql.Filters(('%s.%s = %s' % (buckets_alias, c.alias, s.expression)
                      for c, s in zip(groupby, split_by)))
    on.add('_resample_idx != %s' % unit)
    jk_from = sql.Join(buckets_alias, table, on)
    jk_data_table = sql.Sql(
        sql.Columns(sql.Column('*', auto_alias=False)), jk_from, where=where)
    jk_data_table = sql.Datasource(jk_data_table, 'JackknifeResammpledData')
    jk_data_table_alias = with_data.add(jk_data_table)
  else:
    buckets = sql.Sql(unique_units, table, where=where)
    buckets_alias = with_data.add(sql.Datasource(buckets, 'Buckets'))
    jk_from = sql.Join(buckets_alias, table, join='CROSS')
    jk_data_table = sql.Sql(
        sql.Column('*', auto_alias=False),
        jk_from,
        where=sql.Filters('_resample_idx != %s' % unit).add(where))
    jk_data_table = sql.Datasource(jk_data_table, 'JackknifeResammpledData')
    jk_data_table_alias = with_data.add(jk_data_table)

  return jk_data_table_alias, with_data


def get_jackknife_data_fast(metric, table, split_by, global_filter, indexes,
                            local_filter, with_data):
  # When there is any index added by Operation, we need adjustment.
  if split_by == sql.Columns(indexes):
    return get_jackknife_data_fast_no_adjustment(metric, table, global_filter,
                                                 indexes, local_filter,
                                                 with_data)
  return get_jackknife_data_fast_with_adjustment(metric, table, split_by,
                                                 global_filter, indexes,
                                                 local_filter, with_data)


def get_jackknife_data_fast_no_adjustment(metric, table, global_filter, indexes,
                                          local_filter, with_data):
  """Gets jackknife samples in a fast way for precomputable Jackknife.

  If all the leaf Metrics are Sum and/or Count, we can compute the
  leave-one-out (LOO) estimates faster. If there is no index added by Operation,
  then no adjustment for slices is needed. The query is just like
  WITH
  LOO AS (SELECT DISTINCT
    unrenamed_split_by,
    $RenamedSplitByIfAny AS renamed_split_by,
    unit AS _resample_idx,
    SUM(X) OVER (PARTITION BY split_by) -
      SUM(X) OVER (PARTITION BY split_by, unit) AS `sum(X)`,
    SAFE_DIVIDE(
      SUM(X) OVER (PARTITION BY split_by) -
        SUM(X) OVER (PARTITION BY split_by, unit),
      COUNT(X) OVER (PARTITION BY split_by) -
        COUNT(X) OVER (PARTITION BY split_by, unit)) AS `mean(X)`
  FROM $DATA
  WHERE
  filters),
  ResampledResults AS (SELECT
    split_by,
    _resample_idx,
    SUM(`sum(X)`) AS `sum(X)`,
    SUM(`mean(X)`) AS `mean(X)`
  FROM LOO
  GROUP BY unrenamed_split_by, renamed_split_by, _resample_idx)

  Args:
    metric: An instance of Jackknife.
    table: The table we want to query from.
    global_filter: The filters that can be applied to the whole Metric tree.
    indexes: The columns that we shouldn't apply any arithmetic operation.
    local_filter: The filters that have been accumulated so far.
    with_data: A global variable that contains all the WITH clauses we need.

  Returns:
    The alias of the table in the WITH clause that has all resampled result.
    The global with_data which holds all datasources we need in the WITH clause.
  """
  all_indexes = sql.Columns(indexes).add(utils.get_extra_split_by(metric))
  indexes_and_unit = sql.Columns(all_indexes).add(metric.unit)
  where = sql.Filters(global_filter).add(local_filter)
  columns = sql.Columns()
  # columns is filled in-place in modify_descendants_for_jackknife_fast.
  modified_jk = modify_descendants_for_jackknife_fast_no_adjustment(
      metric, columns,
      sql.Filters(global_filter).add(local_filter), sql.Filters(), all_indexes,
      indexes_and_unit)
  metric.children = modified_jk.children

  bucket = sql.Column(metric.unit, alias='_resample_idx')
  columns = sql.Columns(all_indexes).add(bucket).add(columns)
  columns.distinct = True
  loo_table = with_data.add(
      sql.Datasource(sql.Sql(columns, table, where=where), 'LOO'))
  return loo_table, with_data


def get_jackknife_data_fast_with_adjustment(metric, table, split_by,
                                            global_filter, indexes,
                                            local_filter, with_data):
  """Gets jackknife samples in a fast way for precomputable Jackknife.

  If all the leaf Metrics are Sum and/or Count, we can compute the
  leave-one-out (LOO) estimates faster. If there is any index added by Operation
  then we need to adjust the slices. See utils.adjust_slices_for_loo() for more
  discussions. The query will look like
  WITH
  UnitSliceCount AS (SELECT
    split_by,
    extra_index,
    unit,
    COUNT(*) AS ct,
    SUM(X) AS `sum(X)`
  FROM $DATA
  GROUP BY split_by, extra_index, unit),
  TotalCount AS (SELECT
    split_by,
    extra_index,
    SUM(ct) AS ct,
    SUM(`sum(X)`) AS `sum(X)`
  FROM UnitSliceCount
  GROUP BY split_by, extra_index),
  LOO AS (SELECT
    split_by,
    extra_index,
    unit AS _resample_idx,
    total.`sum(X)` - COALESCE(unit.`sum(X)`, 0) AS `sum(X)`
  FROM (SELECT DISTINCT
    split_by,
    unit
  FROM UnitSliceCount)
  LEFT JOIN  # Or (SELECT DISTINCT unit FROM UnitSliceCount) CROSS JOIN
  TotalCount AS total
  USING (split_by)
  LEFT JOIN
  UnitSliceCount AS unit
  USING (split_by, extra_index, unit)
  WHERE
  total.ct - COALESCE(unit.ct, 0) > 0)

  Args:
    metric: An instance of Jackknife.
    table: The table we want to query from.
    split_by: The columns that we use to split the data.
    global_filter: The filters that can be applied to the whole Metric tree.
    indexes: The columns that we shouldn't apply any arithmetic operation.
    local_filter: The filters that have been accumulated so far.
    with_data: A global variable that contains all the WITH clauses we need.

  Returns:
    The alias of the table in the WITH clause that has all resampled result.
    The global with_data which holds all datasources we need in the WITH clause.
  """
  all_indexes = sql.Columns(indexes).add(utils.get_extra_split_by(metric))
  indexes_and_unit = sql.Columns(all_indexes).add(metric.unit)
  where = sql.Filters(global_filter).add(local_filter)

  columns_to_preagg = sql.Columns(sql.Column('COUNT(*)', alias='ct'))
  columns_in_loo = sql.Columns()
  # columns_to_preagg and columns_in_loo are filled in-place.
  modified_jk = modify_descendants_for_jackknife_fast_with_adjustment(
      metric, columns_to_preagg, columns_in_loo,
      sql.Filters(global_filter).add(local_filter), sql.Filters())
  metric.children = modified_jk.children

  unit_slice_ct_table = sql.Sql(columns_to_preagg, table, where,
                                indexes_and_unit)
  unit_slice_ct_alias = with_data.add(
      sql.Datasource(unit_slice_ct_table, 'UnitSliceCount'))

  total_ct_table = sql.Sql(
      sql.Columns(
          [sql.Column(c, 'SUM({})', c) for c in columns_to_preagg.aliases]),
      unit_slice_ct_alias,
      groupby=indexes_and_unit.difference(metric.unit).aliases)
  total_ct_alias = with_data.add(sql.Datasource(total_ct_table, 'TotalCount'))

  split_by_and_unit = sql.Columns(split_by).add(metric.unit)
  all_slices = sql.Datasource(
      sql.Sql(
          sql.Columns(split_by_and_unit.aliases, distinct=True),
          unit_slice_ct_alias))
  if split_by:
    loo_from = all_slices.join(
        sql.Datasource(total_ct_alias, 'total_table'),
        using=split_by,
        join='LEFT')
  else:
    loo_from = all_slices.join(
        sql.Datasource(total_ct_alias, 'total_table'), join='CROSS')
  loo_from = loo_from.join(
      sql.Datasource(unit_slice_ct_alias, 'unit_slice_table'),
      using=indexes_and_unit.aliases,
      join='LEFT')
  loo = sql.Sql(
      sql.Columns(all_indexes.aliases).add(
          sql.Column(sql.Column(metric.unit).alias,
                     alias='_resample_idx')).add(columns_in_loo), loo_from,
      'total_table.ct - COALESCE(unit_slice_table.ct, 0) > 0')
  loo_table = with_data.add(sql.Datasource(loo, 'LOO'))
  return loo_table, with_data


def modify_descendants_for_jackknife_fast_no_adjustment(metric, columns,
                                                        global_filter,
                                                        local_filter,
                                                        all_indexes,
                                                        indexes_and_unit):
  """Gets the columns for leaf Metrics and modify them for fast Jackknife SQL.

  See the doc of get_jackknife_data_fast_no_adjustment() first. Here we
  1. collects the LOO columns for all Sum and Count.
  2. Modify metric in-place so when we generate SQL later, they know what column
    to use. For example, Sum('X') would generate a SQL column 'SUM(X) AS sum_x',
    but as we precompute LOO estimates, we need to query from a table that has
    "SUM(X) OVER (PARTITION BY split_by) -
      SUM(X) OVER (PARTITION BY split_by, unit) AS `sum(X)`". So the expression
    of Sum('X') should now become 'SUM(`sum(X)`) AS `sum(X)`'. So we replace the
    metric with Sum('sum(X)', metric.name) so it could handle it correctly.
  3. Removes filters as they have already been applied in the LOO table. Note
    that we made a copy in get_se for metric so the removal won't affect the
    metric used in point estimate computation.

  We need to make a copy for the Metric or in
  sumx = metrics.Sum('X')
  m1 = metrics.MetricList(sumx, where='X>1')
  m2 = metrics.MetricList(sumx, where='X>10')
  jk = Jackknife(metrics.MetricList((m1, m2)))
  sumx will be modified twice and the second one will overwrite the first one.

  Args:
    metric: An instance of metrics.Metric or a child of one.
    columns: A global container for all columns we need in LOO table. It's being
      added in-place.
    global_filter: The filters that can be applied to the whole Metric tree.
    local_filter: The filters that have been accumulated so far.
    all_indexes: All columns that we need to used as the group by columns in the
      LOO table. It includes split_by, all extra_idx added by Operations, and
      the stratified_by column of MH, if exists.
    indexes_and_unit: all_indexes plus the unit of Jackknife.

  Returns:
    The modified metric tree.
  """
  if not isinstance(metric, metrics.Metric):
    return metric

  metric = copy.deepcopy(metric)
  local_filter = sql.Filters(local_filter).add(metric.where)
  metric.where = None
  if isinstance(metric, (metrics.Sum, metrics.Count)):
    filters = sql.Filters(local_filter).remove(global_filter)
    c = metric.var
    op = 'COUNT({})' if isinstance(metric, metrics.Count) else 'SUM({})'
    total = sql.Column(c, op, filters=filters, partition=all_indexes)
    unit_sum = sql.Column(c, op, filters=filters, partition=indexes_and_unit)
    loo = (total - unit_sum)
    loo.set_alias(metric.name)
    columns.add(loo)
    return metrics.Sum(loo.alias, metric.name)

  new_children = []
  for m in metric.children:
    modified = modify_descendants_for_jackknife_fast_no_adjustment(
        m, columns, global_filter, local_filter, all_indexes, indexes_and_unit)
    new_children.append(modified)
  metric.children = new_children
  return metric


def modify_descendants_for_jackknife_fast_with_adjustment(
    metric, columns_to_preagg, columns_in_loo, global_filter, local_filter):
  """Gets the columns for leaf Metrics and modify them for fast Jackknife SQL.

  See the doc of get_jackknife_data_fast_with_adjustment() first. Here we
  1. collects the LOO columns for all Sum and Count.
  2. Modify them in-place so when we generate SQL later, they know what column
    to use. For example, Sum('X') would generate a SQL column 'SUM(X) AS sum_x',
    but as we precompute LOO estimates, we need to query from a table that has
    "total_table.`sum(X)` - COALESCE(unit_slice_table.`sum(X)`, 0) AS `sum(X)`".
    So the expression of Sum('X') should now become 'SUM(`sum(X)`) AS `sum(X)`'.
    Here we will replace the metric with Sum('sum(X)', metric.name).
  3. Removes filters as they have already been applied in the LOO table. Note
    that we made a copy in get_se for metric so the removal won't affect the
    metric used in point estimate computation.
  4. For Operations, their extra_index columns appear in indexes. If any of them
    has forbidden character in the name, it will be renamed in LOO so we have to
    change extra_index. For example, Distribution('$Foo') will generate a column
    $Foo AS macro_Foo in LOO so we need to replace '$Foo' with 'macro_Foo'.

  We need to make a copy for the Metric or in
  sumx = metrics.Sum('X')
  m1 = metrics.MetricList(sumx, where='X>1')
  m2 = metrics.MetricList(sumx, where='X>10')
  jk = Jackknife(metrics.MetricList((m1, m2)))
  sumx will be modified twice and the second one will overwrite the first one.

  Args:
    metric: An instance of metrics.Metric or a child of one.
    columns_to_preagg: A global container for all metric columns we need in
      UnitSliceCount and TotalCount tables. It's being added in-place.
    columns_in_loo: A global container for all metric columns we need in LOO
      table. It's being added in-place.
    global_filter: The filters that can be applied to the whole Metric tree.
    local_filter: The filters that have been accumulated so far.

  Returns:
    The modified metric tree.
  """
  if not isinstance(metric, metrics.Metric):
    return metric

  metric = copy.deepcopy(metric)
  local_filter = sql.Filters(local_filter).add(metric.where)
  metric.where = None
  tmpl = 'total_table.%s - COALESCE(unit_slice_table.%s, 0)'
  if isinstance(metric, (metrics.Sum, metrics.Count)):
    filters = sql.Filters(local_filter).remove(global_filter)
    c = metric.var
    op = 'COUNT({})' if isinstance(metric, metrics.Count) else 'SUM({})'
    col = sql.Column(c, op, filters=filters)
    columns_to_preagg.add(col)
    loo = sql.Column(tmpl % (col.alias, col.alias), alias=metric.name)
    columns_in_loo.add(loo)
    return metrics.Sum(loo.alias, metric.name)

  if isinstance(metric, Operation):
    metric.extra_index = sql.Columns(metric.extra_index).aliases
    metric.extra_split_by = sql.Columns(metric.extra_split_by).aliases
    if isinstance(metric, MH):
      metric.stratified_by = sql.Column(metric.stratified_by).alias

  new_children = []
  for m in metric.children:
    modified = modify_descendants_for_jackknife_fast_with_adjustment(
        m, columns_to_preagg, columns_in_loo, global_filter, local_filter)
    new_children.append(modified)
  metric.children = new_children
  return metric


def get_bootstrap_data(metric, table, split_by, global_filter, local_filter,
                       with_data):
  """Gets metric.n_replicates bootstrap resamples.

  The SQL is constructed as
  1. if metric.unit is None:
    WITH
    BootstrapRandomRows AS (SELECT
      *,
      filter AS _bs_filter,
      ROW_NUMBER() OVER (PARTITION BY _resample_idx, filter) AS _bs_row_number,
      CEILING(RAND() * COUNT(*) OVER (PARTITION BY _resample_idx, filter))
        AS _bs_random_row_number,
      $RenamedSplitByIfAny AS renamed_split_by,
    FROM table
    JOIN
    UNNEST(GENERATE_ARRAY(1, metric.n_replicates)) AS _resample_idx),
    BootstrapRandomChoices AS (SELECT
      b.*
    FROM (SELECT
      split_by,
      _resample_idx,
      _bs_filter,
      _bs_random_row_number AS _bs_row_number
    FROM BootstrapRandomRows) AS a
    JOIN
    BootstrapRandomRows AS b
    USING (split_by, _resample_idx, _bs_row_number, _bs_filter)
    WHERE
    _bs_filter),
  The filter parts are optional.

  2. if metric.unit is not None:
    WITH
    Candidates AS (SELECT
      split_by,
      ARRAY_AGG(DISTINCT unit) AS unit,
      COUNT(DISTINCT unit) AS _bs_length
    FROM table
    WHERE global_filter
    GROUP BY split_by),
    BootstrapRandomChoices AS (SELECT
      split_by,
      _resample_idx,
      unit[ORDINAL(CAST(CEILING(RAND() * _bs_length) AS INT64))] AS unit
    FROM Candidates
    JOIN
    UNNEST(unit)
    JOIN
    UNNEST(GENERATE_ARRAY(1, metric.n_replicates)) AS _resample_idx),
    BootstrapResammpledData AS (SELECT
      *
    FROM BootstrapRandomChoices
    LEFT JOIN
    table
    USING (split_by, unit)
    WHERE global_filter)

  Args:
    metric: An instance of Bootstrap.
    table: The table we want to query from.
    split_by: The columns that we use to split the data.
    global_filter: The filters that can be applied to the whole Metric tree.
    local_filter: The filters that have been accumulated so far.
    with_data: A global variable that contains all the WITH clauses we need.

  Returns:
    The alias of the table in the WITH clause that has all resampled data.
    The global with_data which holds all datasources we need in the WITH clause.
  """
  original_table = table
  table = sql.Datasource(table)
  if not table.is_table:
    table.alias = table.alias or 'RawData'
    table = with_data.add(table)
  replicates = sql.Datasource(
      'UNNEST(GENERATE_ARRAY(1, %s))' % metric.n_replicates, '_resample_idx')
  where = sql.Filters(global_filter).add(local_filter)
  if metric.unit is None:
    columns = sql.Columns(sql.Column('*', auto_alias=False))
    partition = split_by.expressions + ['_resample_idx']
    if where:
      columns.add(sql.Column(str(where), alias='_bs_filter'))
      partition.append(str(where))
    row_number = sql.Column(
        'ROW_NUMBER()', alias='_bs_row_number', partition=partition)
    length = sql.Column('COUNT(*)', partition=partition)
    random_row_number = sql.Column('RAND()') * length
    random_row_number = sql.Column(
        'CEILING(%s)' % random_row_number.expression,
        alias='_bs_random_row_number')
    columns.add((row_number, random_row_number))
    columns.add((i for i in split_by if i != i.alias))
    random_choice_table = sql.Sql(columns, sql.Join(table, replicates))
    random_choice_table_alias = with_data.add(
        sql.Datasource(random_choice_table, 'BootstrapRandomRows'))

    using = sql.Columns(partition).add('_bs_row_number').difference(str(where))
    if where:
      using.add('_bs_filter')
    random_rows = sql.Sql(
        sql.Columns(using).difference('_bs_row_number').add(
            sql.Column('_bs_random_row_number', alias='_bs_row_number')),
        random_choice_table_alias)
    random_rows = sql.Datasource(random_rows, 'a')
    resampled = random_rows.join(
        sql.Datasource(random_choice_table_alias, 'b'), using=using)
    table = sql.Sql(
        sql.Column('b.*', auto_alias=False),
        resampled,
        where='_bs_filter' if where else None)
    table = with_data.add(sql.Datasource(table, 'BootstrapRandomChoices'))
  else:
    unit = metric.unit
    unit_alias = sql.Column(unit).alias
    columns = (sql.Column('ARRAY_AGG(DISTINCT %s)' % unit, alias=unit),
               sql.Column('COUNT(DISTINCT %s)' % unit, alias='_bs_length'))
    units = sql.Sql(columns, table, where, split_by)
    units_alias = with_data.add(sql.Datasource(units, 'Candidates'))
    rand_samples = sql.Column(
        '%s[ORDINAL(CAST(CEILING(RAND() * _bs_length) AS INT64))]' % unit_alias,
        alias=unit_alias)

    sample_table = sql.Sql(
        sql.Columns(split_by.aliases).add('_resample_idx').add(rand_samples),
        sql.Join(units_alias,
                 sql.Datasource('UNNEST(%s)' % unit_alias)).join(replicates))
    sample_table_alias = with_data.add(
        sql.Datasource(sample_table, 'BootstrapRandomChoices'))

    table = original_table
    renamed = [i for i in sql.Columns(split_by).add(unit) if i != i.alias]
    if renamed or unit != unit_alias:
      table = sql.Sql(
          sql.Columns(sql.Column('*', auto_alias=False)).add(renamed),
          table,
          where=where)
    bs_data = sql.Sql(
        sql.Column('*', auto_alias=False),
        sql.Join(
            sample_table_alias,
            table,
            join='LEFT',
            using=sql.Columns(split_by.aliases).add(unit)),
        where=where)
    bs_data = sql.Datasource(bs_data, 'BootstrapResammpledData')
    table = with_data.add(bs_data)

  return table, with_data
