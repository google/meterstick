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
    child_table_alias = with_data.add(child_table)
    groupby = sql.Columns(indexes.aliases, distinct=True)
    columns = sql.Columns()
    for c in child_sql.columns:
      if c.alias in groupby:
        continue
      col = sql.Column(c.alias) / sql.Column(
          c.alias, 'SUM({})', partition=split_by.aliases)
      col.set_alias('Distribution of %s' % c.alias_raw)
      columns.add(col)
    return sql.Sql(groupby.add(columns), child_table_alias), with_data


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
    util_metric = Distribution(self.extra_index, self.children[0])
    child_sql, with_data = util_metric.get_sql_and_with_clause(
        table, split_by, global_filter, indexes, local_filter, with_data)
    child_table = sql.Datasource(child_sql, 'CumulativeDistributionRaw')
    child_table_alias = with_data.add(child_table)
    columns = sql.Columns(indexes.aliases)
    order = list(metrics.get_extra_idx(self))
    order[0] = sql.Column(
        _get_order_for_cum_dist(sql.Column(order[0]).alias, self),
        auto_alias=False)
    for c in child_sql.columns:
      if c in columns:
        continue

      col = sql.Column(
          c.alias,
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
               **kwargs):
    self.baseline_key = baseline_key
    self.include_base = include_base
    super(Comparison, self).__init__(child, name_tmpl, condition_column,
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
    cond_cols = sql.Columns(self.extra_index)
    groupby = sql.Columns(split_by).add(cond_cols)
    alias_tmpl = self.name_tmpl
    raw_table_sql, with_data = child.get_sql_and_with_clause(
        table, groupby, global_filter, indexes, local_filter, with_data)
    raw_table = sql.Datasource(raw_table_sql, 'ChangeRaw')
    raw_table_alias = with_data.add(raw_table)

    base = self.baseline_key if isinstance(self.baseline_key,
                                           tuple) else [self.baseline_key]
    base_cond = ('%s = %s' % (c, _format_to_condition(b))
                 for c, b in zip(cond_cols.aliases, base))
    base_cond = ' AND '.join(base_cond)
    base_value = sql.Sql(
        sql.Columns(raw_table_sql.groupby.aliases).add(
            raw_table_sql.columns.aliases).difference(cond_cols.aliases),
        raw_table_alias, base_cond)
    base_table = sql.Datasource(base_value, 'ChangeBase')
    base_table_alias = with_data.add(base_table)

    exclude_base_condition = ('%s != %s' % (c, _format_to_condition(b))
                              for c, b in zip(cond_cols.aliases, base))
    exclude_base_condition = ' OR '.join(exclude_base_condition)
    cond = None if self.include_base else sql.Filters([exclude_base_condition])
    col_tmp = '%s.{c} - %s.{c}' if isinstance(
        self, AbsoluteChange) else 'SAFE_DIVIDE(%s.{c}, (%s.{c})) * 100 - 100'
    columns = sql.Columns()
    for c in raw_table_sql.columns.difference(indexes.aliases):
      col = sql.Column(
          col_tmp.format(c=c.alias) % (raw_table_alias, base_table_alias),
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
    self.check_is_ratio()
    local_filter = sql.Filters([self.where, local_filter]).remove(global_filter)

    child = self.children[0]
    grandchildren = []
    if isinstance(child, metrics.MetricList):
      for m in child:
        grandchildren += list(m.children)
    else:
      grandchildren = child.children

    cond_cols = sql.Columns(self.extra_index)
    groupby = sql.Columns(split_by).add(cond_cols).add(self.stratified_by)
    util_metric = metrics.MetricList(grandchildren)
    util_indexes = sql.Columns(indexes).add(self.stratified_by)
    raw_table_sql, with_data = util_metric.get_sql_and_with_clause(
        table, groupby, global_filter, util_indexes, local_filter, with_data)

    raw_table = sql.Datasource(raw_table_sql, 'MHRaw')
    raw_table_alias = with_data.add(raw_table)

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
    base_table_alias = with_data.add(base_table)

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
    if isinstance(child, metrics.MetricList):
      for c in child:
        columns.add(
            sql.Column(
                col_tmpl % {
                    'numer': sql.Column(c.children[0].name).alias,
                    'denom': sql.Column(c.children[1].name).alias
                },
                alias=alias_tmpl.format(c.name)))
    else:
      columns = sql.Column(
          col_tmpl % {
              'numer': sql.Column(child.children[0].name).alias,
              'denom': sql.Column(child.children[1].name).alias
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
    # There are funtions outside meterstick directly call this, so don't change.
    return estimates

  def compute(self, df):
    replicates = self.compute_on_samples(self.get_samples(df))
    bucket_estimates = pd.concat(replicates, axis=1, sort=False)
    return self.get_stderrs_or_ci_half_width(bucket_estimates)

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
  def get_stderrs(bucket_estimates):
    dof = bucket_estimates.count(axis=1) - 1
    return bucket_estimates.std(1), dof

  def get_ci_width(self, stderrs, dof):
    """You can return asymmetrical confidence interval."""
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
    se, with_data = get_se(self, table, split_by, global_filter, indexes,
                           local_filter, with_data)
    se_alias = with_data.add(sql.Datasource(se, name + 'SE'))

    pt_est, with_data = self.children[0].get_sql_and_with_clause(
        table, split_by, global_filter, indexes, local_filter, with_data)
    pt_est_alias = with_data.add(sql.Datasource(pt_est, name + 'PointEstimate'))

    columns = sql.Columns()
    using = sql.Columns(se.groupby)
    for c in pt_est.columns:
      if c in indexes.aliases:
        using.add(c)
      else:
        pt_est_col = sql.Column(
            '%s.%s' % (pt_est_alias, c.alias), alias=c.alias_raw)
        alias = '%s %s SE' % (c.alias_raw, name)
        se_col = sql.Column(
            '%s.%s' % (se_alias, sql.escape_alias(alias)), alias=alias)
        columns.add(pt_est_col)
        columns.add(se_col)
        if self.confidence:
          dof = '%s dof' % c.alias_raw
          columns.add(
              sql.Column(
                  '%s.%s' % (se_alias, sql.escape_alias(dof)), alias=dof))

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
        base_alias = with_data.add(sql.Datasource(base, '_ShouldAlreadyExists'))
        columns.add(
            sql.Column('%s.%s' % (base_alias, c.alias), alias=c.alias)
            for c in base.columns.difference(indexes))

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
    key = utils.CacheKey(('_RESERVED', 'Jackknife', unit), self.cache_key.where,
                         [unit] + split_by)
    self.save_to_cache(key, loo)
    self.tmp_cache_keys.add(key)
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
    key = utils.CacheKey(('_RESERVED', 'Jackknife', unit), self.cache_key.where,
                         [unit] + split_by)
    self.save_to_cache(key, loo)
    self.tmp_cache_keys.add(key)
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

    If the Jackknife is precomputable, we try to compute the leave-one-out (loo)
    estimates on the whole df first. If it fails, we compute slice by slice
    instead because we don't want to return NAs for valid slices just because an
    irrelevant slice raises an error.

    Args:
      df: The DataFrame to compute on.
      split_by: A list of column names to be passed to df.group_by().

    Returns:
      A melted DataFrame of stderrs for all Metrics in self.children[0].
    """
    estimates = None
    if self.can_precompute:
      try:
        estimates = self.compute_child(
            None, [self.unit] + split_by,
            True,
            cache_key=('_RESERVED', 'Jackknife', self.unit))
        estimates = estimates.unstack(self.unit)
      except Exception:  # pylint: disable=broad-except
        pass  # Fall back to computing slice by slice to salvage good slices.
    if estimates is None:
      samples = self.get_samples(df, split_by)
      replicates = self.compute_on_samples(samples, split_by)
      estimates = pd.concat(replicates, axis=1, sort=False)
    return self.get_stderrs_or_ci_half_width(estimates)

  @staticmethod
  def get_stderrs(bucket_estimates):
    stderrs, dof = super(Jackknife, Jackknife).get_stderrs(bucket_estimates)
    return stderrs * dof / np.sqrt(dof + 1), dof

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


def get_se(metric, table, split_by, global_filter, indexes, local_filter,
           with_data):
  """Gets the SQL query that computes the standard error and dof if needed."""
  global_filter = sql.Filters([global_filter, local_filter]).add(metric.where)
  local_filter = sql.Filters()
  metric, table, split_by, global_filter, indexes, with_data = preaggregate_if_possible(
      metric, table, split_by, global_filter, indexes, with_data)

  if isinstance(metric, Jackknife):
    if metric.can_precompute:
      metric = copy.deepcopy(metric)  # We'll modify the metric tree in-place.
    table, with_data = get_jackknife_data(metric, table, split_by,
                                          global_filter, indexes, local_filter,
                                          with_data)
  else:
    table, with_data = get_bootstrap_data(metric, table, split_by,
                                          global_filter, local_filter,
                                          with_data)

  if isinstance(metric, Jackknife) and metric.can_precompute:
    split_by = adjust_indexes_for_jk_fast(split_by)
    indexes = adjust_indexes_for_jk_fast(indexes)
    # global_filter has been removed from all Metrics when precomputeing LOO.
    global_filter = sql.Filters(None)

  samples, with_data = metric.children[0].get_sql_and_with_clause(
      table,
      sql.Columns(split_by).add('_resample_idx'), global_filter,
      sql.Columns(indexes).add('_resample_idx'), local_filter, with_data)
  samples_alias = with_data.add(sql.Datasource(samples, 'ResampledResults'))

  columns = sql.Columns()
  groupby = sql.Columns(
      (c.alias for c in samples.groupby if c != '_resample_idx'))
  for c in samples.columns:
    if c == '_resample_idx':
      continue
    elif c in indexes.aliases:
      groupby.add(c.alias)
    else:
      se = sql.Column(c.alias, 'STDDEV_SAMP({})',
                      '%s Bootstrap SE' % c.alias_raw)
      if isinstance(metric, Jackknife):
        adjustment = sql.Column(
            'SAFE_DIVIDE((COUNT({c}) - 1), SQRT(COUNT({c})))'.format(c=c.alias))
        se = (se * adjustment).set_alias('%s Jackknife SE' % c.alias_raw)
      columns.add(se)
      if metric.confidence:
        columns.add(
            sql.Column(c.alias, 'COUNT({}) - 1', '%s dof' % c.alias_raw))
  return sql.Sql(columns, samples_alias, groupby=groupby), with_data


def preaggregate_if_possible(metric, table, split_by, global_filter, indexes,
                             with_data):
  """Preaggregates data to make the resampled table small.

  For Jackknife and Bootstrap over group, we may preaggegate the data to make
  the query more efficient, though there are some requirements.
  1. There cannot be any local filter in the metric tree, otherwise the filter
    might need access to original rows.
  2. All leaf Metrics need to be Sum. Techanically other Metrics could be
    supported but Sum is the most common one in use and the easiest to handle.

  If preaggregatable, we sum over all split_bys used by the metric tree, and
  clear all the local filters.

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

  all_split_by = sql.Columns([indexes, metric.unit])
  sums = sql.Columns()
  for m in metric.traverse():
    if sql.Filters(m.where).remove(global_filter):
      return metric, table, split_by, global_filter, indexes, with_data
    if isinstance(m, metrics.SimpleMetric):
      if not isinstance(m, metrics.Sum):
        return metric, table, split_by, global_filter, indexes, with_data
      else:
        sums.add(
            sql.Column(m.var, 'SUM({})',
                       sql.Column('', alias=m.var).alias))
    if isinstance(m, MH):
      all_split_by.add(m.stratified_by)

  metric = copy.deepcopy(metric)
  metric.unit = sql.Column(metric.unit).alias
  for m in metric.traverse():
    m.where = None
    if isinstance(m, Operation):
      m.extra_index = [sql.Column(i, alias=i).alias for i in m.extra_index]
      if isinstance(m, MH):
        m.stratified_by = sql.Column(m.stratified_by).alias
    if isinstance(m, metrics.Sum):
      m.var = sql.Column('', alias=m.var).alias

  preagg = sql.Sql(sums, table, global_filter, all_split_by)
  preagg_alias = with_data.add(sql.Datasource(preagg, 'Preaggregated'))

  return metric, preagg_alias, sql.Columns(
      split_by.aliases), sql.Filters(), sql.Columns(indexes.aliases), with_data


def adjust_indexes_for_jk_fast(indexes):
  """For the indexes that get renamed, only keep the alias.

  For a Jaccknife that only has Sum, Count and Mean as leaf Metrics, we cut the
  corner by getting the LOO table first and select everything from it. See
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
  if metric.can_precompute:
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
    Buckets AS (SELECT
      ARRAY_AGG(DISTINCT unit) AS _jk_buckets
    FROM $DATA
    WHERE filter),
    JackknifeResammpledData AS (SELECT
      * EXCEPT (_jk_buckets)
    FROM Buckets
    JOIN
    UNNEST(_jk_buckets) AS _resample_idx
    CROSS JOIN
    $DATA
    WHERE
    _resample_idx != unit AND filter)

  2. if split_by is not None:
    WITH
    Buckets AS (SELECT
      split_by AS _jk_split_by,
      ARRAY_AGG(DISTINCT unit) AS _jk_buckets
    FROM $DATA
    WHERE filter
    GROUP BY _jk_split_by),
    JackknifeResammpledData AS (SELECT
      * EXCEPT (_jk_split_by, _jk_buckets)
    FROM Buckets
    JOIN
    UNNEST(_jk_buckets) AS _resample_idx
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
  unique_units = sql.Columns(
      (sql.Column('ARRAY_AGG(DISTINCT %s)' % unit, alias='_jk_buckets')))
  where = sql.Filters(global_filter).add(local_filter)
  if split_by:
    groupby = sql.Columns(
        (sql.Column(c.expression, alias='_jk_%s' % c.alias) for c in split_by))
    buckets = sql.Sql(unique_units, table, where, groupby)
    buckets_alias = with_data.add(sql.Datasource(buckets, 'Buckets'))
    jk_from = sql.Join(buckets_alias,
                       sql.Datasource('UNNEST(_jk_buckets)', '_resample_idx'))
    on = sql.Filters(('%s.%s = %s' % (buckets_alias, c.alias, s.expression)
                      for c, s in zip(groupby, split_by)))
    on.add('_resample_idx != %s' % unit)
    jk_from = jk_from.join(table, on=on)
    exclude = groupby.as_groupby() + ', _jk_buckets'
    jk_data_table = sql.Sql(
        sql.Columns(sql.Column('* EXCEPT (%s)' % exclude, auto_alias=False)),
        jk_from,
        where=where)
    jk_data_table = sql.Datasource(jk_data_table, 'JackknifeResammpledData')
    jk_data_table_alias = with_data.add(jk_data_table)
  else:
    buckets = sql.Sql(unique_units, table, where=where)
    buckets_alias = with_data.add(sql.Datasource(buckets, 'Buckets'))

    jk_from = sql.Join(buckets_alias,
                       sql.Datasource('UNNEST(_jk_buckets)', '_resample_idx'))
    jk_from = jk_from.join(table, join='CROSS')
    jk_data_table = sql.Sql(
        sql.Column('* EXCEPT (_jk_buckets)', auto_alias=False),
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

  If all the leaf Metrics are Sum, Count or Mean, we can compute the
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
  all_indexes = sql.Columns(indexes)
  for m in metric.traverse():
    if isinstance(m, MH):
      all_indexes.add(m.stratified_by)
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

  If all the leaf Metrics are Sum, Count or Mean, we can compute the
  leave-one-out (LOO) estimates faster. If there is any index added by
  Operation, then we need to adjust the slices. See
  utils.adjust_slices_for_loo() for more discussions. The query will look like
  WITH
  AllSlices AS (SELECT
    split_by,
    ARRAY_AGG(DISTINCT extra_index) AS extra_index,
    ARRAY_AGG(DISTINCT unit) AS unit
  FROM $DATA
  GROUP BY split_by),
  UnitSliceCount AS (SELECT
    split_by,
    extra_index,
    unit,
    COUNT(*) AS ct,
    SUM(X) AS `sum(X)`
  FROM $DATA
  GROUP BY split_by, condition, unit),
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
  FROM AllSlices
  JOIN
  UNNEST (extra_index) AS extra_index
  JOIN
  UNNEST (unit) AS unit
  LEFT JOIN
  UnitSliceCount AS unit
  USING (split_by, extra_index, unit)
  JOIN
  TotalCount AS total
  USING (split_by, extra_index)
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
  all_indexes = sql.Columns(indexes)
  for m in metric.traverse():
    if isinstance(m, MH):
      all_indexes.add(m.stratified_by)
  indexes_and_unit = sql.Columns(all_indexes).add(metric.unit)
  where = sql.Filters(global_filter).add(local_filter)

  extra_idx = indexes_and_unit.difference(split_by)
  unique_idx = sql.Columns(
      (sql.Column('ARRAY_AGG(DISTINCT %s)' % c.expression, alias=c.alias)
       for c in extra_idx))
  all_slices_table = sql.Sql(unique_idx, table, where, split_by)
  all_slices_alias = with_data.add(
      sql.Datasource(all_slices_table, 'AllSlices'))

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

  slice_in_other_units = sql.Datasource(all_slices_alias)
  for c in extra_idx:
    slice_in_other_units = slice_in_other_units.join(
        'UNNEST ({c}) AS {c}'.format(c=c.alias))
  slice_in_other_units = slice_in_other_units.join(
      sql.Datasource(unit_slice_ct_alias, 'unit'),
      using=indexes_and_unit,
      join='LEFT')
  slice_in_other_units = slice_in_other_units.join(
      sql.Datasource(total_ct_alias, 'total'), using=all_indexes)
  slice_in_other_units_table = sql.Sql(
      sql.Columns(all_indexes.aliases).add(
          sql.Column(sql.Column(metric.unit).alias,
                     alias='_resample_idx')).add(columns_in_loo),
      slice_in_other_units, 'total.ct - COALESCE(unit.ct, 0) > 0')
  loo_table = with_data.add(sql.Datasource(slice_in_other_units_table, 'LOO'))
  return loo_table, with_data


def modify_descendants_for_jackknife_fast_no_adjustment(metric, columns,
                                                        global_filter,
                                                        local_filter,
                                                        all_indexes,
                                                        indexes_and_unit):
  """Gets the columns for leaf Metrics and modify them for fast Jackknife SQL.

  See the doc of get_jackknife_data_fast_no_adjustment() first. Here we
  1. collects the LOO columns for all Sum, Count and Mean Metrics.
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
  if isinstance(metric, (metrics.Sum, metrics.Count, metrics.Mean)):
    filters = sql.Filters(local_filter).remove(global_filter)
    if isinstance(metric, (metrics.Sum, metrics.Count)):
      c = metric.var
      op = 'COUNT({})' if isinstance(metric, metrics.Count) else 'SUM({})'
      total = sql.Column(c, op, filters=filters, partition=all_indexes)
      unit_sum = sql.Column(c, op, filters=filters, partition=indexes_and_unit)
      loo = (total - unit_sum).set_alias(metric.name)
      columns.add(loo)
    elif isinstance(metric, metrics.Mean):
      if metric.weight:
        op = 'SUM({} * {})'
        total_sum = sql.Column((metric.var, metric.weight),
                               op,
                               filters=filters,
                               partition=all_indexes)
        unit_sum = sql.Column((metric.var, metric.weight),
                              op,
                              filters=filters,
                              partition=indexes_and_unit)
        total_weight = sql.Column(
            metric.weight, 'SUM({})', filters=filters, partition=all_indexes)
        unit_weight = sql.Column(
            metric.weight,
            'SUM({})',
            filters=filters,
            partition=indexes_and_unit)
      else:
        total_sum = sql.Column(
            metric.var, 'SUM({})', filters=filters, partition=all_indexes)
        unit_sum = sql.Column(
            metric.var, 'SUM({})', filters=filters, partition=indexes_and_unit)
        total_weight = sql.Column(
            metric.var, 'COUNT({})', filters=filters, partition=all_indexes)
        unit_weight = sql.Column(
            metric.var,
            'COUNT({})',
            filters=filters,
            partition=indexes_and_unit)
      loo = (total_sum - unit_sum) / (total_weight - unit_weight)
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
  1. collects the LOO columns for all Sum, Count and Mean Metrics.
  2. Modify them in-place so when we generate SQL later, they know what column
    to use. For example, Sum('X') would generate a SQL column 'SUM(X) AS sum_x',
    but as we precompute LOO estimates, we need to query from a table that has
    "total.`sum(X)` - COALESCE(unit.`sum(X)`, 0) AS `sum(X)`". So the expression
    of Sum('X') should now become 'SUM(`sum(X)`) AS `sum(X)`'. So we replace the
    metric with Sum('sum(X)', metric.name) so it could handle it correctly.
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
  tmpl = 'total.%s - COALESCE(unit.%s, 0)'
  if isinstance(metric, (metrics.Sum, metrics.Count, metrics.Mean)):
    filters = sql.Filters(local_filter).remove(global_filter)
    if isinstance(metric, (metrics.Sum, metrics.Count)):
      c = metric.var
      op = 'COUNT({})' if isinstance(metric, metrics.Count) else 'SUM({})'
      col = sql.Column(c, op, filters=filters)
      columns_to_preagg.add(col)
      loo = sql.Column(tmpl % (col.alias, col.alias), alias=metric.name)
    elif isinstance(metric, metrics.Mean):
      if metric.weight:
        sum_col = sql.Column((metric.var, metric.weight),
                             'SUM({} * {})',
                             filters=filters)
        weight_col = sql.Column(metric.weight, 'SUM({})', filters=filters)
      else:
        sum_col = sql.Column(metric.var, 'SUM({})', filters=filters)
        weight_col = sql.Column(metric.var, 'COUNT({})', filters=filters)

      columns_to_preagg.add([sum_col, weight_col])
      loo_sum = sql.Column(tmpl % (sum_col.alias, sum_col.alias))
      loo_weight = sql.Column(tmpl % (weight_col.alias, weight_col.alias))
      loo = (loo_sum / loo_weight).set_alias(metric.name)
    columns_in_loo.add(loo)
    return metrics.Sum(loo.alias, metric.name)

  if isinstance(metric, Operation):
    metric.extra_index = sql.Columns(metric.extra_index).aliases
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
      b.* EXCEPT (_bs_row_number, _bs_filter)
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
      _bs_idx,
      unit[ORDINAL(CAST(CEILING(RAND() * _bs_length) AS INT64))] AS unit
    FROM Candidates
    JOIN
    UNNEST(unit)
    JOIN
    UNNEST(GENERATE_ARRAY(1, metric.n_replicates)) AS _bs_idx),
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
    excludes = ['_bs_row_number']
    if where:
      excludes.append('_bs_filter')
      using.add('_bs_filter')
    random_rows = sql.Sql(
        sql.Columns(using).difference('_bs_row_number').add(
            sql.Column('_bs_random_row_number', alias='_bs_row_number')),
        random_choice_table_alias)
    random_rows = sql.Datasource(random_rows, 'a')
    resampled = random_rows.join(
        sql.Datasource(random_choice_table_alias, 'b'), using=using)
    table = sql.Sql(
        sql.Column('b.* EXCEPT (%s)' % ', '.join(excludes), auto_alias=False),
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
