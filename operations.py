# Copyright 2023 Google LLC
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


def count_features(m: metrics.Metric):
  """Gets the width of the result of m.compute_on()."""
  if not m:
    return 0
  if isinstance(m, metrics.MetricList):
    return sum([count_features(i) for i in m])
  if isinstance(m, MetricWithCI):
    return (
        count_features(m.children[0]) * 3
        if m.confidence
        else count_features(m.children[0]) * 2
    )
  if isinstance(m, (CUPED, PrePostChange)):
    return count_features(m.children[0][0])
  if isinstance(m, Operation):
    return count_features(m.children[0])
  if isinstance(m, metrics.CompositeMetric):
    return max([count_features(i) for i in m.children])
  if isinstance(m, metrics.Quantile):
    if m.one_quantile:
      return 1
    return len(m.quantile)
  return 1


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
      child Metric. For example, PercentChange('condition', base_value,
      Sum('X')).compute_on(df, 'grp') would compute Sum('X').compute_on(df,
      ['grp', 'condition']) then get the change. As the result, the CacheKey
      used in PercentChange is different to that used in Sum('X'). The latter
      has more columns in the split_by. extra_split_by records what columns need
      to be added to children Metrics so we can flush the cache correctly. The
      convention is extra_split_by comes after split_by.
    extra_index: Not every extra_split_by show up in the result. For example,
      the group_by columns in Models don't show up in the final output.
      extra_index stores the columns that will show up and should be a subset of
      extra_split_by. If not given, it's same as extra_split_by.
    precomputable_in_jk_bs: Indicates whether it is possible to cut corners in
      Jackknife and Bootstrap with unit. During the precomputation the leaf
      Metrics might get modified. If the computation of the Operation won't get
      impacted by that, it's precomputable. More precisely, the default
      compute_children() returns children.compute_on(df, split_by +
      extra_split_by). If all the Operation need from the descendants are
      included in the result of compute_children(), the intermediate results it
      cached during the computation, and the name of the descendants, then it's
      precomputable. For examplem PercentChange(unit, 0, Sum(x)).compute_on(df)
      needs Sum(x).compute_on(df, unit) and nothing more from the Sum, so it's
      precomputable. Everything MH(unit, 0, Sum(x), grp).compute_on(df) needs
      from Sum have been computed and cached during the computation of
      Sum(x).compute_on(df, unit + grp) so MH is precomputable. The easiest way
      to check if an Operation is precomputable is that you just set the
      attribute to True and try Metrics like Jackknife(..., Operation(Dot('x',
      'y', where='x>2'))) and Jackknife(..., Operation(Dot('x', 'y',
      where='x>2')), enable_optimization=False). If the first one computes and
      gives the same result to the second one, the Operation is precomputable.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    cache_key: What key to use to cache the df. You can use anything that can be
      a key of a dict except '_RESERVED' and tuples like ('_RESERVED', ...). And
      all other attributes inherited from Metric.
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
    self.precomputable_in_jk_bs = True
    self.is_operation = True

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

  def split_data(self, df, split_by=None):
    """Splits the DataFrame returned by the children."""
    for k, idx in df.groupby(split_by, observed=True).indices.items():
      # split_by will be added back later during the concatenation.
      # Use iloc rather than loc because indexes can have duplicates.
      yield df.iloc[idx].droplevel(split_by), k

  def manipulate(
      self,
      res,
      melted: bool = False,
      return_dataframe: bool = True,
      apply_name_tmpl=None,
  ):
    apply_name_tmpl = True if apply_name_tmpl is None else apply_name_tmpl
    return super(Operation, self).manipulate(
        res, melted, return_dataframe, apply_name_tmpl
    )

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
               name_tmpl: Text = 'Distribution of {}',
               **kwargs):
    super(Distribution, self).__init__(child, name_tmpl, over, **kwargs)

  def compute_on_children(self, children, split_by):
    total = (
        children.groupby(level=split_by, observed=True).sum()
        if split_by
        else children.sum()
    )
    res = children / total
    # The order might get messed up for MultiIndex.
    if len(children.index.names) > 1:
      return res.reorder_levels(children.index.names)
    return res

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
    local_filter = (
        sql.Filters(self.where_).add(local_filter).remove(global_filter)
    )
    all_split_by = sql.Columns(split_by).add(self.extra_split_by)
    child_sql, with_data = self.children[0].get_sql_and_with_clause(
        table, all_split_by, global_filter, indexes, local_filter, with_data)
    child_table = sql.Datasource(child_sql, 'DistributionRaw')
    child_table_alias = with_data.merge(child_table)
    groupby = sql.Columns(all_split_by.aliases)
    columns = sql.Columns()
    for c in child_sql.columns:
      if c.alias in groupby:
        continue
      col = sql.Column(c.alias) / sql.Column(
          c.alias, 'SUM({})', partition=split_by.aliases
      )
      col.set_alias('Distribution of %s' % c.alias_raw)
      columns.add(col)
    return sql.Sql(groupby.add(columns), child_table_alias), with_data


Normalize = Distribution  # An alias.


class CumulativeDistribution(Distribution):
  """Computes the normalized cumulative sum.

  Attributes:
    extra_split_by: A list of column(s) to normalize over.
    children: A tuple of a Metric whose result we compute the cumulative
      distribution on.
    order: An iterable. The over column will be ordered by it before computing
      cumsum.
    ascending: Sort ascending or descending.
    sort_by_values: Boolean that indicates whether or not to sort by the
      computed distribution values instead of the `over` column. It works with
      `ascending` but not `order`.
    And all other attributes inherited from Distribution.
  """

  def __init__(
      self,
      over: Text,
      child: Optional[metrics.Metric] = None,
      order=None,
      ascending: bool = True,
      sort_by_values: bool = False,
      name_tmpl: Text = 'Cumulative Distribution of {}',
      additional_fingerprint_attrs=None,
      **kwargs,
  ):
    self.order = order
    self.ascending = ascending
    self.sort_by_values = sort_by_values
    super(CumulativeDistribution, self).__init__(
        over,
        child,
        name_tmpl,
        additional_fingerprint_attrs=['order', 'ascending', 'sort_by_values']
        + (additional_fingerprint_attrs or []),
        **kwargs,
    )
    if order and len(self.extra_index) > 1:
      raise ValueError(
          'Only one column is supported when "order" is specified.'
      )
    if order and sort_by_values:
      raise ValueError('Custom order is not allowed when sorting by values!')

  def compute_on_children(self, children, split_by):
    dist = super(CumulativeDistribution, self).compute_on_children(
        children, split_by
    )
    if self.order:
      order = self.order if self.ascending else reversed(self.order)
      level = None if len(dist.index.names) == 1 else self.extra_index[0]
      dist = dist.reindex(order, level=level).dropna()
      res = self.group(dist, split_by).cumsum()
    elif not self.sort_by_values:
      dist.sort_values(self.extra_index, ascending=self.ascending, inplace=True)
      res = self.group(dist, split_by).cumsum()
    else:
      cumsum = []
      for col in dist:
        cumsum.append(
            self.group(
                dist[col].sort_values(ascending=self.ascending), split_by
            ).cumsum()
        )
      res = pd.concat(cumsum, axis=1)
    if split_by:
      res.sort_index(level=split_by, sort_remaining=False, inplace=True)
    return res

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
    dist_sql, with_data = super(
        CumulativeDistribution, self
    ).get_sql_and_with_clause(
        table, split_by, global_filter, indexes, local_filter, with_data
    )
    child_table = sql.Datasource(dist_sql, 'CumulativeDistributionRaw')
    child_table_alias = with_data.merge(child_table)
    columns = sql.Columns(indexes.aliases)
    order = list(self.get_extra_idx(self))
    order = [
        sql.Column(self.get_ordered_col(sql.Column(o).alias), auto_alias=False)
        for o in order
    ]
    for c in dist_sql.columns:
      if c in columns:
        continue

      col = sql.Column(
          c.alias,
          'SUM({})',
          partition=split_by.aliases,
          order=self.get_ordered_col(c.alias) if self.sort_by_values else order,
          window_frame='ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW',
      )
      col.set_alias('Cumulative %s' % c.alias_raw)
      columns.add(col)
    return sql.Sql(columns, child_table_alias), with_data

  def get_ordered_col(self, over):
    if self.order:
      over = 'CASE %s\n' % over
      tmpl = 'WHEN %s THEN %s'
      over += '\n'.join(
          tmpl % (_format_to_condition(o), i) for i, o in enumerate(self.order)
      )
      over += '\nELSE %s\nEND' % len(self.order)
    return over if self.ascending else over + ' DESC'


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

  @property
  def stratified_by(self):
    return self.extra_split_by[len(self.extra_index):]

  @stratified_by.setter
  def stratified_by(self, stratified_by):
    stratified_by = (
        stratified_by if isinstance(stratified_by, list) else [stratified_by]
    )
    self.extra_split_by[len(self.extra_index):] = stratified_by

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
    cond_cols = sql.Columns(self.extra_index)
    raw_table_sql, with_data = self.get_change_raw_sql(
        table, split_by, global_filter, indexes, local_filter, with_data
    )
    raw_table = sql.Datasource(raw_table_sql, 'ChangeRaw')
    raw_table_alias = with_data.merge(raw_table)

    base = self.baseline_key if isinstance(self.baseline_key,
                                           tuple) else [self.baseline_key]
    base_cond = ('%s = %s' % (c, _format_to_condition(b))
                 for c, b in zip(cond_cols.aliases, base))
    base_cond = ' AND '.join(base_cond)
    cols = sql.Columns(raw_table_sql.groupby.aliases)
    cols.add(raw_table_sql.columns.aliases)
    base_value = sql.Sql(
        cols.difference(cond_cols.aliases), raw_table_alias, base_cond)
    base_table = sql.Datasource(base_value, 'ChangeBase')
    base_table_alias = with_data.merge(base_table)

    cond = None if self.include_base else sql.Filters([f'NOT ({base_cond})'])
    sql_template_for_comparison = self.get_sql_template_for_comparison(
        raw_table_alias, base_table_alias
    )
    columns = sql.Columns()
    val_col_len = len(raw_table_sql.all_columns) - len(indexes)
    for r, b in zip(
        raw_table_sql.all_columns[-val_col_len:],
        base_value.columns[-val_col_len:],
    ):
      col = sql.Column(
          sql_template_for_comparison % {'r': r.alias, 'b': b.alias},
          alias=self.name_tmpl.format(r.alias_raw),
      )
      columns.add(col)
    using = indexes.difference(cond_cols)
    join = '' if using else 'CROSS'
    return sql.Sql(
        sql.Columns(indexes.aliases).add(columns),
        sql.Join(raw_table_alias, base_table_alias, join=join, using=using),
        cond), with_data

  def get_change_raw_sql(
      self, table, split_by, global_filter, indexes, local_filter, with_data
  ):
    """Gets the query where the comparison will be carried out."""
    local_filter = (
        sql.Filters(self.where_).add(local_filter).remove(global_filter)
    )
    groupby = sql.Columns(split_by).add(self.extra_split_by)
    raw_table_sql, with_data = self.children[0].get_sql_and_with_clause(
        table, groupby, global_filter, indexes, local_filter, with_data
    )
    return raw_table_sql, with_data

  def get_sql_template_for_comparison(self, raw_table_alias, base_table_alias):
    """Gets a string template to compute the comparison between columns.

    The template needs to use "%(r)s" to represent the column from
    raw_table_alias and "%(b)s" to represent that from base_table_alias.
    For example, AbsoluteChange returns
    f'{raw_table_alias}.%(r)s - {base_table_alias}.%(b)s'.

    Args:
      raw_table_alias: The alias of the raw table for comparison.
      base_table_alias: The alias of the base table for comparison.

    Returns:
      A string template to compute the comparison between two columns.
    """
    raise NotImplementedError


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
               name_tmpl: Text = '{} Percent Change',
               **kwargs):
    super(PercentChange, self).__init__(condition_column, baseline_key, child,
                                        include_base, name_tmpl, **kwargs)

  def compute_on_children(self, children, split_by):
    level = None
    if split_by:
      level = self.extra_index[0] if len(
          self.extra_index) == 1 else self.extra_index
    # Avoid ZeroDivisionError when input is object dytpe.
    children = children.astype(float)
    res = (children / children.xs(self.baseline_key, level=level) - 1) * 100
    if len(children.index.names) > 1:  # xs might mess up the level order.
      res = res.reorder_levels(children.index.names)
    if not self.include_base:
      to_drop = [i for i in res.index.names if i not in self.extra_index]
      idx_to_match = res.index.droplevel(to_drop) if to_drop else res.index
      res = res[~idx_to_match.isin([self.baseline_key])]
    return res

  def get_sql_template_for_comparison(self, raw_table_alias, base_table_alias):
    return (
        sql.SAFE_DIVIDE.format(
            numer=f'{raw_table_alias}.%(r)s',
            denom=f'{base_table_alias}.%(b)s',
        )
        + ' * 100 - 100'
    )


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
               name_tmpl: Text = '{} Absolute Change',
               **kwargs):
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

  def get_sql_template_for_comparison(self, raw_table_alias, base_table_alias):
    return f'{raw_table_alias}.%(r)s - {base_table_alias}.%(b)s'


def _check_covariates_match_base(base, cov):
  len_base = len(base) if isinstance(base, metrics.MetricList) else 1
  len_cov = len(cov) if isinstance(cov, metrics.MetricList) else 1
  if len_cov != len_base:
    raise ValueError(
        'Covariates and base metric must have the same length. Got'
        f' {len_cov} and {len_base}'
    )


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
    child: A Metric(List) we want to compute change on. If it returns multiple
      columns, the result is same as applying the method to every column in it.
    covariates: A MetricList of the covariates for adjustment.
    children: MetricList([child, covariates]).
    include_base: A boolean for whether the baseline condition should be
      included in the output.
    multiple_covariates: If True, all covariates are used together as in the
      adjustment. If False, we zip the child and covariates and create a list of
      one-covariate PrePostChange. Namely,
      PrePostChange(child=[x1, x2], covariates=[y1, y2],
                    multiple_covariates=False) is equivalent to
      MetricList([PrePostChange(x1, y1), PrePostChange(x2, y2)]).
    k_covariates: The length of covariates.
    And all other attributes inherited from Operation.
  """

  def __init__(self,
               condition_column,
               baseline_key,
               child=None,
               covariates=None,
               stratified_by=None,
               include_base=False,
               multiple_covariates=True,
               name_tmpl: Text = '{} PrePost Percent Change',
               **kwargs):
    if isinstance(child, (List, Tuple)):
      child = metrics.MetricList(child)
    if isinstance(covariates, (List, Tuple)):
      covariates = metrics.MetricList(covariates)
    if child and covariates:
      if not multiple_covariates:
        _check_covariates_match_base(child, covariates)
      child = metrics.MetricList((child, covariates))
    else:
      child = None
    self.multiple_covariates = multiple_covariates
    stratified_by = [stratified_by] if isinstance(stratified_by,
                                                  str) else stratified_by or []
    condition_column = [condition_column] if isinstance(
        condition_column, str) else condition_column
    additional_fingerprint_attrs = kwargs.pop(
        'additional_fingerprint_attrs', []
    )
    additional_fingerprint_attrs += ['multiple_covariates']
    super(PrePostChange, self).__init__(
        condition_column + stratified_by,
        baseline_key,
        child,
        include_base,
        name_tmpl,
        additional_fingerprint_attrs=additional_fingerprint_attrs,
        **kwargs,
    )
    self.extra_index = condition_column

  @property
  def child(self):
    return self.children[0][0] if self.children else None

  @property
  def covariates(self):
    return self.children[0][1] if self.children else None

  @property
  def k_covariates(self) -> int:
    return count_features(self.covariates)

  def compute_slices(self, df, split_by=None):
    if self.multiple_covariates:
      return super(PrePostChange, self).compute_slices(df, split_by)
    equiv, _ = utils.get_equivalent_metric(self)
    res = self.compute_util_metric_on(equiv, df, split_by)
    tmpl_len = len(self.name_tmpl.format(''))
    res.columns = [c[:-tmpl_len] for c in res.columns]
    return res

  def compute_children(
      self,
      df,
      split_by=None,
      melted=False,
      return_dataframe=True,
      cache_key=None,
  ):
    if not self.multiple_covariates:
      raise NotImplementedError  # shouldn't be called.
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
      covariates = (
          covariates - covariates.groupby(split_by, observed=True).mean()
      )
    else:
      covariates = covariates - covariates.mean()
    # Align child with covariates in case there is any missing slices.
    covariates = covariates.reorder_levels(child.index.names)
    aligned = pd.concat([child, covariates], axis=1)
    len_child = child.shape[1]
    lm = linear_model.LinearRegression()

    # Define a custom Metric instead of using df.groupby().apply() because
    # 1. It's faster. See the comments in Metric.compute_slices().
    # 2. It ensures that the result is formatted correctly.
    class Adjust(metrics.Metric):
      """Adjusts the value by fitting controlling for the covariates.

      See the class doc for adjustment details. Essentially for every slice for
      comparison, we fit a linear regression child = c + θ * covariate and use c
      as the adjusted value for PercentChange computation later.
      Because we center covariate first, when there is only one covariate, θ can
      be computed as Covariance(child, covariate) / Var(covariate) and
      c = avg(child) - θ * avg(covariate).
      """

      def compute_slices(self, df, split_by: Optional[List[Text]] = None):
        child = df.iloc[:, :len_child]
        prefix = utils.get_unique_prefix(child)
        df.columns = list(child.columns) + [
            prefix + c for c in df.columns[len_child:]
        ]
        covariate = df.iloc[:, len_child:]
        if len(covariate.columns) > 1:
          return super(Adjust, self).compute_slices(df, split_by)
        adjusted = df.groupby(split_by, observed=True).mean()
        covariate_col = covariate.columns[0]
        covariate_adjusted = adjusted.iloc[:, -1]
        for c in child:
          theta = (
              metrics.Cov(c, covariate_col) / metrics.Variance(covariate_col)
          ).compute_on(df, split_by, return_dataframe=False)
          adjusted[c] = adjusted[c] - covariate_adjusted * theta
        return adjusted.iloc[:, :-1]

      def compute(self, df_slice):
        child_slice = df_slice.iloc[:, :len_child]
        covariate = df_slice.iloc[:, len_child:]
        adjusted = [
            lm.fit(covariate, child_slice[c]).intercept_ for c in child_slice
        ]
        return pd.DataFrame([adjusted], columns=child_slice.columns)

    return Adjust('').compute_on(aligned, split_by + self.extra_index)

  def compute_through_sql(self, table, split_by, execute, mode):
    if self.multiple_covariates:
      return super(PrePostChange, self).compute_through_sql(
          table, split_by, execute, mode
      )
    equiv, _ = utils.get_equivalent_metric(self)
    res = self.compute_util_metric_on_sql(
        equiv, table, split_by, execute, False, mode
    )
    # The column name got messed up when there is only one base metric because
    # we squeeze the dataframe to a series.
    if len(res.columns) == 1:
      res.columns = [self.name_tmpl.format(self.children[0][0].name)]
    return res

  def compute_children_sql(self, table, split_by, execute, mode=None):
    if not self.multiple_covariates:
      raise NotImplementedError  # shouldn't be called.
    child = super(PrePostChange,
                  self).compute_children_sql(table, split_by, execute, mode)
    covariates = child.iloc[:, -self.k_covariates:]
    child = child.iloc[:, :-self.k_covariates]
    return self.adjust_value(child, covariates, split_by)

  def get_sql_and_with_clause(self, table, split_by, global_filter, indexes,
                              local_filter, with_data):
    if self.multiple_covariates:
      return super(PrePostChange, self).get_sql_and_with_clause(
          table, split_by, global_filter, indexes, local_filter, with_data
      )
    equiv, _ = utils.get_equivalent_metric(self)
    return equiv.get_sql_and_with_clause(
        table, split_by, global_filter, indexes, local_filter, with_data
    )

  def get_change_raw_sql(
      self, table, split_by, global_filter, indexes, local_filter, with_data
  ):
    """Generates PrePost-adjusted values for PercentChange computation.

    This function generates subqueries like
    WITH PrePostRaw AS (SELECT
      split_by,
      stratified_by,
      condition_column,
      child_metric,
      covariate
    FROM T
    GROUP BY split_by, stratified_by, condition_column),
    PrePostcovariateCentered AS (SELECT
      split_by,
      stratified_by,
      condition_column,
      child_metric,
      covariate - AVG(covariate) OVER (PARTITION BY split_by) AS covariate
    FROM PrePostRaw),
    ChangeRaw AS (SELECT
      split_by,
      condition_column,
      AVG(child_metric) - SAFE_DIVIDE(AVG(covariate) * COVAR_SAMP(child_metric,
        covariate), VAR_SAMP(covariate)) AS child_metric
    FROM PrePostcovariateCentered
    GROUP BY split_by, condition_column)

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
    if count_features(self.children[0][1]) > 1:
      raise NotImplementedError
    local_filter = (
        sql.Filters(self.where_).add(local_filter).remove(global_filter)
    )
    all_split_by = sql.Columns(split_by).add(self.extra_split_by)
    all_indexes = sql.Columns(split_by).add(self.extra_index)
    child_sql, with_data = self.children[0].get_sql_and_with_clause(
        table, all_split_by, global_filter, indexes, local_filter, with_data)
    child_table = sql.Datasource(child_sql, 'PrePostRaw')
    child_table_alias = with_data.merge(child_table)

    split_by = split_by.aliases
    all_split_by = all_split_by.aliases
    all_indexes = all_indexes.aliases
    cols = [
        sql.Column(c.alias, alias=c.alias_raw)
        for c in child_sql.all_columns[:-1]
    ]
    covariate = child_sql.all_columns[-1].alias
    covariate_mean = sql.Column(covariate, 'AVG({})', partition=split_by)
    covariate_centered = (sql.Column(covariate) - covariate_mean).set_alias(
        covariate
    )
    cols.append(covariate_centered)
    covariate_centered_sql = sql.Sql(cols, child_table_alias)
    covariate_centered_table = sql.Datasource(
        covariate_centered_sql, 'PrePostcovariateCentered'
    )
    covariate_centered_table_alias = with_data.merge(covariate_centered_table)

    to_adjust = []
    for c in child_sql.all_columns[:-1]:
      if c.alias in all_split_by:
        continue
      adjusted = metrics.Mean(c.alias) - metrics.Mean(covariate) * metrics.Cov(
          c.alias, covariate
      ) / metrics.Variance(covariate)
      to_adjust.append(adjusted.set_name(c.alias_raw))
    return metrics.MetricList(to_adjust).get_sql_and_with_clause(
        covariate_centered_table_alias,
        all_indexes,
        None,
        all_indexes,
        None,
        with_data,
    )

  def get_equivalent_without_filter(self, *auxiliary_cols):
    del auxiliary_cols  # unused
    if self.multiple_covariates:
      return
    _check_covariates_match_base(self.child, self.covariates)
    if (
        not isinstance(self.covariates, metrics.MetricList)
        or len(self.covariates) == 1
    ):
      res = copy.deepcopy(self)
      res.multiple_covariates = True
      return metrics.MetricList([res])
    if not isinstance(self.child, metrics.MetricList) or not isinstance(
        self.covariates, metrics.MetricList
    ):
      raise ValueError(
          "child and covariates are not MetricList. This shouldn't happen."
      )
    return metrics.MetricList([
        PrePostChange(
            self.extra_index,
            self.baseline_key,
            metrics.MetricList([b], where=self.child.where_),
            metrics.MetricList([c], where=self.covariates.where_),
            self.stratified_by,
            self.include_base,
            False,
            self.name_tmpl,
        )
        for b, c in zip(self.child, self.covariates)
    ], where=self.children[0].where_)


class CUPED(AbsoluteChange):
  """CUPED change estimator on a Metric.

  Computes the absolute change after controlling for preperiod metrics.
  Essentially, if the data only has a baseline and a treatment slice, CUPED
  1. centers the covariates (skipping it doesn't affect the result but we do it
    for better numerical stability).
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
    child: A Metric we want to compute change on. If it returns multiple
      columns, the result is same as applying the method to every column in it.
    covariates: A MetricList of the covariates for adjustment.
    children: MetricList([child, covariates]).
    include_base: A boolean for whether the baseline condition should be
      included in the output.
    multiple_covariates: If True, all covariates are used together as in the
      adjustment. If False, we zip the child and covariates and create a list of
      one-covariate CUPED. Namely,
      CUPED(child=[x1, x2], covariates=[y1, y2], multiple_covariates=False) is
      equivalent to MetricList([CUPED(x1, y1), CUPED(x2, y2)]).
    k_covariates: The length of covariates.
    And all other attributes inherited from Operation.
  """

  def __init__(self,
               condition_column,
               baseline_key,
               child=None,
               covariates=None,
               stratified_by=None,
               include_base=False,
               multiple_covariates=True,
               name_tmpl: Text = '{} CUPED Change',
               **kwargs):
    if isinstance(child, (List, Tuple)):
      child = metrics.MetricList(child)
    if isinstance(covariates, (List, Tuple)):
      covariates = metrics.MetricList(covariates)
    if child and covariates:
      if not multiple_covariates:
        _check_covariates_match_base(child, covariates)
      child = metrics.MetricList((child, covariates))
    else:
      child = None
    self.multiple_covariates = multiple_covariates
    stratified_by = [stratified_by] if isinstance(stratified_by,
                                                  str) else stratified_by or []
    condition_column = [condition_column] if isinstance(
        condition_column, str) else condition_column
    additional_fingerprint_attrs = kwargs.pop(
        'additional_fingerprint_attrs', []
    )
    additional_fingerprint_attrs += ['multiple_covariates']
    super(CUPED, self).__init__(
        condition_column + stratified_by,
        baseline_key,
        child,
        include_base,
        name_tmpl,
        additional_fingerprint_attrs=additional_fingerprint_attrs,
        **kwargs,
    )
    self.extra_index = condition_column

  @property
  def child(self):
    return self.children[0][0] if self.children else None

  @property
  def covariates(self):
    return self.children[0][1] if self.children else None

  @property
  def k_covariates(self) -> int:
    return count_features(self.covariates)

  def compute_slices(self, df, split_by=None):
    if self.multiple_covariates:
      return super(CUPED, self).compute_slices(df, split_by)
    equiv, _ = utils.get_equivalent_metric(self)
    res = self.compute_util_metric_on(equiv, df, split_by)
    tmpl_len = len(self.name_tmpl.format(''))
    res.columns = [c[:-tmpl_len] for c in res.columns]
    return res

  def compute_children(
      self,
      df,
      split_by=None,
      melted=False,
      return_dataframe=True,
      cache_key=None,
  ):
    if not self.multiple_covariates:
      raise NotImplementedError  # shouldn't be called.
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
      covariates = (
          covariates - covariates.groupby(split_by, observed=True).mean()
      )
    else:
      covariates = covariates - covariates.mean()
    # Align child with covariates in case there is any missing slices.
    covariates = covariates.reorder_levels(child.index.names)
    aligned = pd.concat([child, covariates], axis=1)
    len_child = child.shape[1]
    lm = linear_model.LinearRegression()
    extra_index = self.extra_index

    # Define a custom Metric instead of using df.groupby().apply() because
    # 1. It's faster. See the comments in Metric.compute_slices().
    # 2. It ensures that the result is formatted correctly.
    class Adjust(metrics.Metric):
      """Adjusts the value by fitting controlling for the covariates.

      Essentially we fit a linear regression child = c + θ * covariate.
      and use child - θ * covariate as the adjusted value. When there is only
      one covariate, θ can be computed as
      Covariance(child, covariate) / Var(covariate)
      """

      def compute_slices(self, df, split_by: Optional[List[Text]] = None):
        child = df.iloc[:, :len_child]
        prefix = utils.get_unique_prefix(child)
        df.columns = list(child.columns) + [
            prefix + c for c in df.columns[len_child:]
        ]
        covariate = df.iloc[:, len_child:]
        if len(covariate.columns) > 1:
          return super(Adjust, self).compute_slices(df, split_by)
        adjusted = df.groupby(split_by + extra_index, observed=True).mean()
        covariate_col = covariate.columns[0]
        covariate_adjusted = adjusted.iloc[:, -1]
        for c in child:
          theta = (
              metrics.Cov(c, covariate_col) / metrics.Variance(covariate_col)
          ).compute_on(df, split_by, return_dataframe=False)
          adjusted[c] = adjusted[c] - covariate_adjusted * theta
        return adjusted.iloc[:, :-1]

      def compute(self, df_slice):
        child_slice = df_slice.iloc[:, :len_child]
        covariate = df_slice.iloc[:, len_child:]
        adjusted = df_slice.groupby(extra_index, observed=True).mean()
        for c in child_slice:
          theta = lm.fit(covariate, child_slice[c]).coef_
          adjusted[c] = adjusted[c] - adjusted.iloc[:, len_child:].dot(theta)
        return adjusted.iloc[:, :len_child]

    return Adjust('').compute_on(aligned, split_by)

  def compute_through_sql(self, table, split_by, execute, mode):
    if self.multiple_covariates:
      return super(CUPED, self).compute_through_sql(
          table, split_by, execute, mode
      )
    equiv, _ = utils.get_equivalent_metric(self)
    res = self.compute_util_metric_on_sql(
        equiv, table, split_by, execute, False, mode
    )
    # The column name got messed up when there is only one base metric because
    # we squeeze the dataframe to a series.
    if len(res.columns) == 1:
      res.columns = [self.name_tmpl.format(self.children[0][0].name)]
    return res

  def compute_children_sql(self, table, split_by, execute, mode=None):
    if not self.multiple_covariates:
      raise NotImplementedError  # shouldn't be called.
    child = super(CUPED, self).compute_children_sql(table, split_by, execute,
                                                    mode)
    covariates = child.iloc[:, -self.k_covariates:]
    child = child.iloc[:, :-self.k_covariates]
    return self.adjust_value(child, covariates, split_by)

  def get_sql_and_with_clause(self, table, split_by, global_filter, indexes,
                              local_filter, with_data):
    if self.multiple_covariates:
      return super(CUPED, self).get_sql_and_with_clause(
          table, split_by, global_filter, indexes, local_filter, with_data
      )
    equiv, _ = utils.get_equivalent_metric(self)
    return equiv.get_sql_and_with_clause(
        table, split_by, global_filter, indexes, local_filter, with_data
    )

  def get_change_raw_sql(
      self, table, split_by, global_filter, indexes, local_filter, with_data
  ):
    """Generates CUPED-adjusted values for AbsoluteChange computation.

    This function generates subqueries like
    WITH CUPEDRaw AS (SELECT
      split_by,
      stratified_by,
      condition_column,
      child_metric,
      covariate
    FROM T
    GROUP BY split_by, stratified_by, condition_column),
    CUPEDTheta AS (SELECT
      split_by,
      SAFE_DIVIDE(COVAR_SAMP(child_metric, covariate), VAR_SAMP(covariate))
        AS child_metric_theta
    FROM CUPEDRaw
    GROUP BY split_by),
    ChangeRaw AS (SELECT
      split_by,
      condition_column,
      AVG(child_metric) - AVG(child_metric_theta * covariate) AS child_metric
    FROM CUPEDRaw
    FULL JOIN
    CUPEDTheta
    USING (split_by)
    GROUP BY split_by, condition_column)

    Note that we don't center the covariate like we do in PrePostChange because
    it doesn't affect the results here.

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
    if count_features(self.children[0][1]) > 1:
      raise NotImplementedError
    local_filter = (
        sql.Filters(self.where_).add(local_filter).remove(global_filter)
    )
    all_split_by = sql.Columns(split_by).add(self.extra_split_by)
    all_indexes = sql.Columns(split_by).add(self.extra_index)
    child_sql, with_data = self.children[0].get_sql_and_with_clause(
        table, all_split_by, global_filter, indexes, local_filter, with_data)
    child_table = sql.Datasource(child_sql, 'CUPEDRaw')
    child_table_alias = with_data.merge(child_table)

    split_by = split_by.aliases
    all_split_by = all_split_by.aliases
    all_indexes = all_indexes.aliases
    cols = []
    for c in child_sql.columns:
      if c.alias not in all_split_by:
        cols.append(c)
    covariate = cols.pop().alias
    theta = metrics.MetricList(
        [
            (
                metrics.Cov(c.alias, covariate) / metrics.Variance(covariate)
            ).set_name(f'{c.alias}_theta')
            for c in cols
        ]
    )
    theta_sql, with_data = theta.get_sql_and_with_clause(
        child_table_alias, split_by, None, all_indexes, None, with_data
    )
    theta_table = sql.Datasource(theta_sql, 'CUPEDTheta')
    theta_table_alias = with_data.merge(theta_table)

    to_adjust = sql.Columns(
        [
            (
                sql.Column(c.alias, 'AVG({})')
                - sql.Column((c.alias, covariate), 'AVG({}_theta * {})')
            ).set_alias(c.alias_raw)
            for c in cols
        ]
    )
    join = 'FULL' if split_by else 'CROSS'
    adjusted_sql = sql.Sql(
        to_adjust,
        sql.Join(
            child_table_alias, theta_table_alias, using=split_by, join=join
        ),
        groupby=all_indexes,
    )
    return adjusted_sql, with_data

  def get_equivalent_without_filter(self, *auxiliary_cols):
    del auxiliary_cols  # unused
    if self.multiple_covariates:
      return
    _check_covariates_match_base(self.child, self.covariates)
    if (
        not isinstance(self.covariates, metrics.MetricList)
        or len(self.covariates) == 1
    ):
      res = copy.deepcopy(self)
      res.multiple_covariates = True
      return metrics.MetricList([res])
    if not isinstance(self.child, metrics.MetricList) or not isinstance(
        self.covariates, metrics.MetricList
    ):
      raise ValueError(
          "child and covariates are not MetricList. This shouldn't happen."
      )
    return metrics.MetricList([
        CUPED(
            self.extra_index,
            self.baseline_key,
            metrics.MetricList([b], where=self.child.where_),
            metrics.MetricList([c], where=self.covariates.where_),
            self.stratified_by,
            self.include_base,
            False,
            self.name_tmpl,
        )
        for b, c in zip(self.child, self.covariates)
    ], where=self.children[0].where_)


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
               name_tmpl: Text = '{} MH Ratio',
               **kwargs):
    stratified_by = (
        stratified_by if isinstance(stratified_by, list) else [stratified_by]
    )
    condition_column = [condition_column] if isinstance(
        condition_column, str) else condition_column
    super(MH, self).__init__(
        condition_column + stratified_by,
        baseline_key,
        metric,
        include_base,
        name_tmpl,
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
            'MH only makes sense on ratio Metrics or a MetricList of ratios.'
            ' Got %s.' % metric
        )

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
            [metrics.MetricList(m.children, where=m.where_)], where=child.where_
        )
        children.append(
            self.compute_util_metric_on(
                util_metric, df, split_by, cache_key=cache_key))
      return children
    util_metric = metrics.MetricList(child.children, where=child.where_)
    return self.compute_util_metric_on(
        util_metric, df, split_by, cache_key=cache_key)

  def compute_on_children(self, children, split_by):
    child = self.children[0]
    if isinstance(child, metrics.MetricList):
      res = [
          self.compute_mh(c, d, split_by)
          for c, d in zip(child.children, children)
      ]
      return pd.concat(res, axis=1, sort=False)
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
           (kb * na * weights).groupby(to_split, observed=True).sum() - 1) * 100
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

  def compute_children_sql(self, table, split_by=None, execute=None, mode=None):
    child = self.children[0]
    self.check_is_ratio(child)
    if isinstance(child, metrics.MetricList):
      children = []
      for m in child.children:
        util_metric = metrics.MetricList(
            [metrics.MetricList(m.children, where=m.where_)], where=child.where_
        )
        c = self.compute_util_metric_on_sql(
            util_metric,
            table,
            split_by + self.extra_split_by,
            execute,
            mode=mode)
        children.append(c)
      return children
    util_metric = metrics.MetricList(child.children, where=child.where_)
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
    LEFT JOIN
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
    local_filter = (
        sql.Filters(self.where_).add(local_filter).remove(global_filter)
    )

    if isinstance(child, metrics.MetricList):
      grandchildren = []
      for m in child:
        grandchildren.append(metrics.MetricList(m.children, where=m.where_))
      util_metric = metrics.MetricList(grandchildren, where=child.where_)
    else:
      util_metric = metrics.MetricList(child.children, where=child.where_)

    cond_cols = sql.Columns(self.extra_index)
    groupby = sql.Columns(split_by).add(self.extra_split_by)
    util_indexes = sql.Columns(indexes).add(self.stratified_by)
    raw_table_sql, with_data = util_metric.get_sql_and_with_clause(
        table, groupby, global_filter, util_indexes, local_filter, with_data)

    raw_table = sql.Datasource(raw_table_sql, 'MHRaw')
    raw_table_alias = with_data.merge(raw_table)

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
    base_table_alias = with_data.merge(base_table)

    exclude_base_condition = ('%s != %s' % (c, _format_to_condition(b))
                              for c, b in zip(cond_cols.aliases, base))
    exclude_base_condition = ' OR '.join(exclude_base_condition)
    cond = None if self.include_base else sql.Filters([exclude_base_condition])
    numerator = sql.SAFE_DIVIDE.format(
        numer=f'{raw_table_alias}.%(numer)s * {base_table_alias}.%(denom)s',
        denom=f'{base_table_alias}.%(denom)s + {raw_table_alias}.%(denom)s',
    )
    denominator = sql.SAFE_DIVIDE.format(
        numer=f'{base_table_alias}.%(numer)s * {raw_table_alias}.%(denom)s',
        denom=f'{base_table_alias}.%(denom)s + {raw_table_alias}.%(denom)s',
    )
    col_tmpl = f"""100 * {sql.SAFE_DIVIDE.format(
        numer=f'COALESCE(SUM({numerator}), 0)',
        denom=f'COALESCE(SUM({denominator}), 0)')} - 100"""
    columns = sql.Columns()
    alias_tmpl = self.name_tmpl
    # The columns might get consolidated and have different aliases. We need to
    # find them by reconstruction. Reusing the with_data in reconstruction will
    # make sure the columns get renamed the same way as in raw_table_sql.
    if isinstance(child, metrics.MetricList):
      for c in child:
        with_data2 = copy.deepcopy(with_data)
        util = metrics.MetricList(c.children[:1], where=c.where_)
        numer_sql, with_data2 = util.get_sql_and_with_clause(
            table, groupby, global_filter, util_indexes, local_filter,
            with_data2)
        with_data2.merge(sql.Datasource(numer_sql))
        numer = numer_sql.columns[-1].alias
        with_data2 = copy.deepcopy(with_data)
        util = metrics.MetricList(c.children[1:], where=c.where_)
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
      util = metrics.MetricList(child.children[:1], where=child.where_)
      numer_sql, with_data2 = util.get_sql_and_with_clause(
          table, groupby, global_filter, util_indexes, local_filter, with_data2)
      with_data2.merge(sql.Datasource(numer_sql))
      numer = numer_sql.columns[-1].alias
      with_data2 = copy.deepcopy(with_data)
      util = metrics.MetricList(child.children[1:], where=child.where_)
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
    return (
        sql.Sql(
            columns,
            sql.Join(
                raw_table_alias, base_table_alias, join='LEFT', using=using
            ),
            cond,
            indexes.aliases,
        ),
        with_data,
    )


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

  def display(
      res,
      aggregate_dimensions=True,
      show_control=None,
      metric_formats=None,
      sort_by=None,
      metric_order=None,
      flip_color=None,
      hide_null_ctrl=True,
      display_expr_info=False,
      auto_add_description=False,
      show_metric_value_when_control_hidden=False,
      return_pre_agg_df=False,
      return_formatted_df=False,
  ):
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
      sort_by: In the form of [{'column': ('CI_Lower', 'Metric Foo'),
        'ascending': False}, {'column': 'Dim Bar': 'order': ['Logged-in',
        'Logged-out']}]. 'column' is the column to sort by. If you want to sort
        by a metric, use (field, metric name) where field could be 'Ratio',
        'Value', 'CI_Lower' and 'CI_Upper'. 'order' is optional and for
        categorical column. 'ascending' is optional and default True. The result
        will be displayed in the order specified by sort_by from top to bottom.
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
      show_metric_value_when_control_hidden: Only has effect when show_control
        is False. If True, we also display the raw metric value, otherwise only
        the change and confidence interval are displayed.
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
          res.index.levels[0].str.replace(comparison_suffix, '', regex=True),
          level=0,
      )
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
        res['_expr_id'] = res[condition_column].agg(tuple, axis=1).astype(str)
        control = str(control)
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
        show_metric_value_when_control_hidden=show_metric_value_when_control_hidden,
        return_pre_agg_df=return_pre_agg_df,
    )
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
    enable_optimization: If all leaf Metrics are Sum and/or Count, or can be
      expressed equivalently by Sum and/or Count, then we can preaggregate the
      data for faster computation. In addition, for Jackknife we can further cut
      corners to compute leave-one-out-estimates. This attribute controls
      whether to use these optimizations.
    has_been_preaggregated: If the Metric and data has already been
      preaggregated, this will be set to True.
    _is_root_node: If the instance is a root Metric. And all other attributes
      inherited from Operation.
  """

  def __init__(
      self,
      unit: Optional[Text],
      child: Optional[metrics.Metric] = None,
      confidence: Optional[float] = None,
      name_tmpl: Optional[Text] = None,
      prefix: Optional[Text] = None,
      additional_fingerprint_attrs=None,
      sql_batch_size=None,
      enable_optimization=True,
      **kwargs,
  ):
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
    self.prefix = prefix
    self.sql_batch_size = sql_batch_size
    if not self.prefix and self.name_tmpl:
      self.prefix = prefix or self.name_tmpl.format('').strip()
    self.precomputable_in_jk_bs = False
    self.enable_optimization = enable_optimization
    self.has_been_preaggregated = False
    self._is_root_node = None

  def compute_slices(self, df, split_by=None):
    std = super(MetricWithCI, self).compute_slices(df, split_by)
    point_est = self.compute_point_estimate(df, split_by)
    res = point_est.join(utils.melt(std))
    if self.confidence:
      res = self.compute_ci(res)
    res = utils.unmelt(res)
    if not self.confidence:
      return res
    base = self.compute_change_base(df, split_by)
    return self.add_base_to_res(res, base)

  def compute_point_estimate(self, df, split_by):
    return self.compute_child(df, split_by, melted=True)

  def compute_ci(self, res):
    """Constructs the confidence interval.

    Args:
      res: A three-column DataFrame. The 1st column are the point estimates
        returned by compute_point_estimate. The 2nd and 3rd columns are called
        `{self.prefix} CI-lower` and `{self.prefix} CI-upper`, but actually what
        they stored are half CI widths from get_ci_width().

    Returns:
      The input res with the 2nd and 3rd columns modified in-place. The columns
      now actually contain CI bounds. By default we add/minus CI half width from
      the point estimate to get the bounds. If you want to construct CI without
      using the point estimates, for example, using percentiles from bootstrap
      instead, you can overwrite get_ci_width() to directly store the bounds
      then make this function a no-op.
    """
    res[self.prefix + ' CI-lower'] = (
        res.iloc[:, 0] - res[self.prefix + ' CI-lower']
    )
    res[self.prefix + ' CI-upper'] += res.iloc[:, 0]
    return res

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
    change = self.children[0]
    util_metric = change.children[0]
    if isinstance(self.children[0], (PrePostChange, CUPED)):
      util_metric = metrics.MetricList(
          [util_metric.children[0]], where=util_metric.where_
      )
    util_metric = metrics.MetricList([util_metric], where=change.where_)
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

  def compute_children(
      self,
      df: pd.DataFrame,
      split_by=None,
      melted=False,
      return_dataframe=True,
      cache_key=None,
  ):
    del melted, return_dataframe, cache_key  # unused
    return self.compute_on_samples(self.get_samples(df, split_by), split_by)

  def get_samples(self, df, split_by=None):
    raise NotImplementedError

  def compute_on_samples(
      self, keyed_samples: Iterable[Tuple[Any, pd.DataFrame]], split_by=None
  ):
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
              sample, split_by, melted=True, cache_key=cache_key
          )
        estimates.append(res)
      except Exception as e:  # pylint: disable=broad-except
        print(
            'Warning: Failed on%s sample data for reason %s. If you see many '
            'such failures, your data might be too sparse.'
            % (self.name_tmpl.format(''), repr(e))
        )
    return estimates

  def compute_on_children(self, children, split_by):
    del split_by  # unused
    bucket_estimates = pd.concat(children, axis=1, sort=False)
    return self.get_stderrs_or_ci_half_width(bucket_estimates)

  def get_stderrs_or_ci_half_width(self, bucket_estimates):
    """Returns confidence interval information in an unmelted DataFrame."""
    stderrs, dof = self.get_stderrs(bucket_estimates)
    if self.confidence:
      res = pd.DataFrame(self.get_ci_width(stderrs, dof)).T
      res.columns = [self.prefix + ' CI-lower', self.prefix + ' CI-upper']
    else:
      res = pd.DataFrame(stderrs, columns=[self.prefix + ' SE'])
    res = utils.unmelt(res)
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

  def manipulate(
      self, res, melted=False, return_dataframe=True, apply_name_tmpl=None
  ):
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
        base = self.get_cached(key)
    # Don't add suffix like "Jackknife" because point_est won't have it.
    res = super(MetricWithCI, self).manipulate(
        res, melted, return_dataframe, apply_name_tmpl or False
    )
    return self.add_base_to_res(res, base) if self.confidence else res

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
    else:
      msg = (
          'You need to specify a confidence level in order to use `.display()`'
      )
      warn = lambda _: print(msg)
      res.display = warn.__get__(res)  # pytype: disable=attribute-error
    return res

  def add_display_fn(self, res, split_by, melted):
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

    fn = get_display_fn(self.prefix, split_by, melted, value, condition_col,
                        ctrl_id, metric_formats)
    # pylint: disable=no-value-for-parameter
    res.display = fn.__get__(res)  # pytype: disable=attribute-error
    # pylint: enable=no-value-for-parameter
    return res

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
      return_dataframe=True,
  ):
    """Computes self in pure SQL or a mixed of SQL and Python.

    Args:
      table: The table we want to query from.
      split_by: The columns that we use to split the data.
      execute: A function that can executes a SQL query and returns a DataFrame.
      melted: Whether to transform the result to long format.
      mode: For Metrics with children, there are different ways to split the
        computation into SQL and Python. For example, we can compute everything
        in SQL, or the children in SQL and the parent in Python, or
        grandchildren in SQL and the rest in Python. Here we support two modes.
        The default mode where `mode` is None is recommend. This mode computes
        maximizes the SQL usage, namely, everything can be computed in SQL is
        computed in SQL. The opposite mode is called `mixed` where the SQL usage
        is minimized, namely, only leaf Metrics are computed in SQL. There is
        another `magic` mode which only applies to Models. The mode computes
        sufficient statistics in SQL then use them to solve the coefficients in
        Python. It's faster then the regular mode when fitting Models on large
        data.
      cache_key: What key to use to cache the result. You can use anything that
        can be a key of a dict except '_RESERVED' and tuples like ('_RESERVED',
        ..).
      cache: The global cache the whole Metric tree shares. If it's None, we
        initiate an empty dict.
      batch_size: The number of resamples to compute in one SQL run. It only has
        effect in the 'mixed' mode. It precedes self.batch_size.
      return_dataframe: Not used. MetricWithCI always returns a DataFrame.

    Returns:
      A pandas DataFrame. It's the computeation of self in SQL.
    """
    del return_dataframe  # not used
    self._runtime_batch_size = batch_size
    try:
      return super(MetricWithCI,
                   self).compute_on_sql(table, split_by, execute, melted, mode,
                                        cache_key, cache)
    finally:
      self._runtime_batch_size = None

  def compute_through_sql(self, table, split_by, execute, mode):
    try:
      return super(MetricWithCI, self).compute_through_sql(
          table, split_by, execute, mode
      )
    except NotImplementedError:
      raise
    except Exception as e:  # pylint: disable=broad-except
      batch_size = self._runtime_batch_size or self.sql_batch_size
      if batch_size:
        msg = 'reducing the batch_size. Current batch_size is %s.' % batch_size
      else:
        msg = "compute_on_sql(..., mode='mixed', batch_size=an integer)."
      raise ValueError(
          "Please see the root cause of the failure above. If it's caused by "
          'the query being too large/complex, you can try %s' % msg
      ) from e

  def compute_on_sql_sql_mode(self, table, split_by=None, execute=None):
    """Computes self in a SQL query and process the result.

    We first execute the SQL query then process the result.
    When confidence is not specified, for each child Metric, the SQL query
    returns two columns. The result columns are like
      metric1, metric1 jackknife SE, metric2, metric2 jackknife SE, ...
    When confidence is specified, for each child Metric, the SQL query
    returns four columns. The result columns are like
      metric1, metric1 CI lower, metric1 CI upper,
      metric2, metric2 CI lower, metric2 CI upper,
      ...
      metricN, metricN CI lower, metricN CI upper,
      metric 1 base value, metric 2 base value, ..., metricN base value.
    The base value columns only exist when the child is an instance of
    PercentChange, AbsoluteChange, or their derived classes. Base values are the
    raw value the comparison is carried out.

    Args:
      table: The table we want to query from.
      split_by: The columns that we use to split the data.
      execute: A function that can executes a SQL query and returns a DataFrame.

    Returns:
      The result DataFrame of Jackknife/Bootstrap.
    """
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

    res = pd.concat((sub_dfs), axis=1, keys=metric_names, names=['Metric'])
    return self.add_base_to_res(res, base)

  def compute_on_sql_mixed_mode(self, table, split_by, execute, mode=None):
    """Computes the child in SQL and the rest in Python.

    There are two parts. First we compute the standard errors. Then we join it
    with point estimate. When the child is a Comparison, we also compute the
    base value for the display().
    For the first part, we preaggregate the data when possible. See the docs of
    Bootstrap.compute_slices about the details of the preaggregation. The
    structure of this part is similar to to_sql(). Note that we apply
    preaggregation to both Jackknife and Bootstrap even though
    Jackknife.compute_slices doesn't have preaggregation. We don't do
    preaggregation in Jackknife.compute_slices because it already cuts the
    corner to get leave-one-out-estimates. Adding preaggregation actually slow
    things down. Here in 'mixed' mode we don't cut the corner to get LOO so
    preaggregation makes sense.
    Then we compute the point estimate, join it with the standard error, and do
    some data manipulations.

    Args:
      table: The table we want to query from.
      split_by: The columns that we use to split the data.
      execute: A function that can executes a SQL query and returns a DataFrame.
      mode: It's always 'mixed' or 'magic' otherwise we won't be here.

    Returns:
      The result DataFrame of Jackknife/Bootstrap.
    """
    batch_size = self._runtime_batch_size or self.sql_batch_size
    if self.has_been_preaggregated or not self.can_precompute():
      if self.where:
        table = sql.Sql(None, table, self.where)
        self_no_filter = copy.deepcopy(self)
        self_no_filter.where = None
        return self_no_filter.compute_on_sql_mixed_mode(
            table, split_by, execute, mode
        )

      replicates = self.compute_children_sql(
          table, split_by, execute, mode, batch_size
      )
      std = self.compute_on_children(replicates, split_by)
      point_est = self.compute_child_sql(
          table, split_by, execute, True, mode=mode
      )
      res = point_est.join(utils.melt(std))
      if self.confidence:
        res[self.prefix + ' CI-lower'] = (
            res.iloc[:, 0] - res[self.prefix + ' CI-lower']
        )
        res[self.prefix + ' CI-upper'] += res.iloc[:, 0]
      res = utils.unmelt(res)
      base = self.compute_change_base(table, split_by, execute, mode)
      return self.add_base_to_res(res, base)

    expanded, _ = utils.get_fully_expanded_equivalent_metric_tree(self)
    if self != expanded:
      return expanded.compute_on_sql_mixed_mode(table, split_by, execute, mode)

    # The filter has been taken care of in preaggregation.
    expanded.where = None
    expanded = utils.push_filters_to_leaf(expanded)
    all_split_by = (
        split_by
        + list(utils.get_extra_split_by(expanded, return_superset=True))
        + [expanded.unit]
    )
    leaf = utils.get_leaf_metrics(expanded)
    for m in leaf:
      m.where = sql.Filters(m.where_).remove(self.where_)
    cols = [
        l.get_sql_columns(l.where_).set_alias(get_preaggregated_metric_var(l))
        for l in leaf
    ]
    preagg = sql.Sql(cols, table, self.where_, all_split_by)
    equiv = get_preaggregated_metric_tree(expanded)
    equiv.unit = sql.Column(equiv.unit).alias
    split_by = sql.Columns(split_by).aliases
    for m in equiv.traverse():
      if isinstance(m, metrics.Metric):
        m.extra_index = sql.Columns(m.extra_index).aliases
        m.extra_split_by = sql.Columns(m.extra_split_by).aliases
    if isinstance(equiv, Bootstrap):
      # When each unit only has one row after preaggregation, we sample by
      # rows.
      if not utils.get_extra_split_by(equiv, return_superset=True):
        equiv.unit = None
    else:
      equiv.has_local_filter = any([l.where for l in leaf])
    return equiv.compute_on_sql_mixed_mode(preagg, split_by, execute, mode)

  def compute_children_sql(self,
                           table,
                           split_by,
                           execute,
                           mode=None,
                           batch_size=None):
    """The return should be similar to compute_children()."""
    raise NotImplementedError

  def to_sql(self, table, split_by=None, create_tmp_table_for_volatile_fn=None):
    """Generates SQL query for the metric.

    The SQL generation is actually delegated to get_sql_and_with_clause(). This
    function does the preaggregation when possible. See the docs of
    Bootstrap.compute_slices() about the details of the preaggregation. The
    structure of this function is similar to compute_on_sql_mixed_mode().

    Args:
      table: The table we want to query from.
      split_by: The columns that we use to split the data.
      create_tmp_table_for_volatile_fn: If to put subqueries that contain
      volatile functions, namely, RAND(), into a CREATE TEMP TABLE, or leave
      them in the WITH clause.

    Returns:
      The query that does Jackknife/Bootstrap.
    """
    if not isinstance(self, (Jackknife, Bootstrap)):
      raise NotImplementedError
    split_by = [split_by] if isinstance(split_by, str) else list(split_by or [])
    # If self is not root, this function won't be called.
    self._is_root_node = True
    if self.has_been_preaggregated or not self.can_precompute():
      if not self.where:
        return super(MetricWithCI, self).to_sql(
            table, split_by, create_tmp_table_for_volatile_fn
        )
      table = sql.Sql(None, table, self.where)
      self_no_filter = copy.deepcopy(self)
      self_no_filter.where = None
      return self_no_filter.to_sql(
          table, split_by, create_tmp_table_for_volatile_fn
      )

    expanded, _ = utils.get_fully_expanded_equivalent_metric_tree(self)
    if self != expanded:
      return expanded.to_sql(table, split_by, create_tmp_table_for_volatile_fn)

    expanded.where = None  # The filter has been taken care of in preaggregation
    expanded = utils.push_filters_to_leaf(expanded)
    split_by = [split_by] if isinstance(split_by, str) else list(split_by or [])
    all_split_by = (
        split_by
        + list(utils.get_extra_split_by(expanded, return_superset=True))
        + [expanded.unit]
    )
    leaf = utils.get_leaf_metrics(expanded)
    # The root filter has already been applied in preaggregation.
    for m in leaf:
      m.where = sql.Filters(m.where_).remove(self.where_)
    cols = [
        l.get_sql_columns(l.where_).set_alias(get_preaggregated_metric_var(l))
        for l in leaf
    ]
    preagg = sql.Sql(cols, table, self.where_, all_split_by)
    equiv = get_preaggregated_metric_tree(expanded)
    equiv.unit = sql.Column(equiv.unit).alias
    split_by = sql.Columns(split_by).aliases
    for m in equiv.traverse():
      if isinstance(m, metrics.Metric):
        m.extra_index = sql.Columns(m.extra_index).aliases
        m.extra_split_by = sql.Columns(m.extra_split_by).aliases
    if isinstance(equiv, Bootstrap):
      # When each unit only has one row after preaggregation, we sample by rows.
      if not utils.get_extra_split_by(equiv, return_superset=True):
        equiv.unit = None
    else:
      equiv.has_local_filter = any([l.where for l in leaf])
    return equiv.to_sql(preagg, split_by, create_tmp_table_for_volatile_fn)

  def get_sql_and_with_clause(
      self, table, split_by, global_filter, indexes, local_filter, with_data
  ):
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
    # Confidence interval cannot be computed in SQL completely so the SQL
    # generated below doesn't work correctly if self is not a root node.
    if self.confidence and not self._is_root_node:
      self._is_root_node = None
      raise NotImplementedError
    self._is_root_node = None

    local_filter = (
        sql.Filters(self.where_).add(local_filter).remove(global_filter)
    )
    # global_filter has been applied in preaggregated data.
    filters = (
        sql.Filters(None) if self.has_been_preaggregated else global_filter
    )

    name = self.name_tmpl.format('').strip()
    se, with_data = self.get_se_sql(
        table,
        split_by,
        filters,
        indexes,
        with_data,
    )
    se_alias = with_data.merge(sql.Datasource(se, name + 'SE'))

    pt_est, with_data = self.children[0].get_sql_and_with_clause(
        table, split_by, global_filter, indexes, local_filter, with_data
    )
    pt_est_alias = with_data.merge(
        sql.Datasource(pt_est, name + 'PointEstimate')
    )

    columns = sql.Columns()
    using = sql.Columns(se.groupby)
    pt_est_col = []
    for c in pt_est.columns:
      if c in indexes.aliases:
        using.add(c)
      else:
        pt_est_col.append(
            sql.Column(f'{pt_est_alias}.{c.alias}', alias=c.alias_raw)
        )
    se_cols = []
    for c in se.columns:
      if c not in indexes.aliases:
        se_cols.append(sql.Column(f'{se_alias}.{c.alias}', alias=c.alias_raw))
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
          child, (PercentChange, AbsoluteChange)
      ):
        has_base_vals = True
        base_metric = copy.deepcopy(child.children[0])
        if isinstance(child, (CUPED, PrePostChange)):
          base_metric = base_metric[0]
        if child.where:
          base_metric.add_where(child.where_)
        base, with_data = base_metric.get_sql_and_with_clause(
            table,
            sql.Columns(split_by).add(child.extra_index),
            global_filter,
            indexes,
            local_filter,
            with_data,
        )
        base_alias = with_data.merge(
            sql.Datasource(base, 'BaseValues')
        )
        columns.add(
            sql.Column(f'{base_alias}.{c.alias}', alias=c.alias_raw)
            for c in base.columns.difference(indexes)
        )

    join = 'LEFT' if using else 'CROSS'
    from_data = sql.Join(pt_est_alias, se_alias, join=join, using=using)
    if has_base_vals:
      from_data = from_data.join(base_alias, join=join, using=using)
    return sql.Sql(using.add(columns), from_data), with_data

  def get_se_sql(
      self,
      table,
      split_by,
      global_filter,
      indexes,
      with_data,
  ):
    """Gets the SQL query that computes the standard error and dof if needed."""
    global_filter = sql.Filters(global_filter).add(self.where_)
    self_copy = copy.deepcopy(self)  # self_copy might get modified in-place.
    table = sql.Datasource(table)
    if not table.is_table:
      table.alias = table.alias or 'RawData'
      table = with_data.add(table)
    table, with_data = self_copy.get_resampled_data_sql(
        table,
        split_by,
        global_filter,
        indexes,
        with_data,
    )
    return get_se_sql(
        self_copy,
        table,
        split_by,
        global_filter,
        indexes,
        with_data,
    )

  def get_resampled_data_sql(
      self,
      table,
      split_by,
      global_filter,
      indexes,
      with_data,
  ):
    raise NotImplementedError

  def can_precompute(self):
    return False


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
               name_tmpl: Text = '{} Jackknife',
               **kwargs):
    super(Jackknife, self).__init__(
        unit,
        child,
        confidence,
        name_tmpl,
        enable_optimization=enable_optimization,
        **kwargs,
    )

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
    if self != utils.get_fully_expanded_equivalent_metric_tree(self)[0]:
      util, df = utils.get_fully_expanded_equivalent_metric_tree(self, df)
      return self.compute_util_metric_on(util, df, split_by)

    self.compute_child(df, split_by + [self.unit])
    precomputed = self.find_all_in_cache_by_metric_type(metric=metrics.Sum)
    precomputed.update(
        self.find_all_in_cache_by_metric_type(metric=metrics.Count)
    )
    precomputed = {
        k: v for k, v in precomputed.items() if k.key == self.cache_key.key
    }
    all_split_by = (
        split_by
        + [self.unit]
        + list(utils.get_extra_split_by(self, return_superset=True))
    )
    df_slices = df.groupby(all_split_by, observed=True).first().iloc[:, [0]]
    for key, each_bucket in precomputed.items():
      self.precompute_sum_or_count_for_jackknife(
          key, each_bucket, split_by, df_slices
      )
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
      df: A dataframe that has the same slices as the df that Jackknife computes
        on.

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
      total = each_bucket.groupby(level=split_by, observed=True).sum()
    else:
      total = each_bucket.sum()
    key = cache_key.replace_split_by(split_by)
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

  def compute_children(
      self,
      df: pd.DataFrame,
      split_by=None,
      melted=False,
      return_dataframe=True,
      cache_key=None,
  ):
    if not self.can_precompute():
      return super(Jackknife, self).compute_children(
          df, split_by, melted, return_dataframe, cache_key
      )
    replicates = self.compute_child(
        df,
        split_by + [self.unit],
        True,
        cache_key=('_RESERVED', 'Jackknife', self.unit),
    )
    return [replicates.unstack(self.unit)]

  def get_samples(self, df, split_by=None, return_cache_key=False):
    """Yields leave-one-out (LOO) DataFrame with level value.

    If self.can_precompute(), this function will not get triggered because the
    results have been precomputed and saved in cache.

    Args:
      df: The DataFrame to compute on.
      split_by: The columns that we use to split the data.
      return_cache_key: If to return a cache key.

    Yields:
      ('_RESERVED', 'Jackknife', unit, i) if return_cache_key else None, and the
      leave-i-out DataFrame.
    """
    levels = df[self.unit].unique()
    if len(levels) < 2:
      raise ValueError('Too few %s to jackknife.' % self.unit)

    if not split_by:
      for lvl in levels:
        key = ('_RESERVED', 'Jackknife', self.unit, lvl)
        yield key if return_cache_key else None, df[df[self.unit] != lvl]
    else:
      df = df.set_index(split_by)
      max_slices = len(df.index.unique())
      for lvl, idx in df.groupby(self.unit, observed=True).groups.items():
        df_rest = df[df[self.unit] != lvl]
        unique_slice_val = idx.unique()
        if len(unique_slice_val) != max_slices:
          # Keep only the slices that appeared in the dropped bucket.
          df_rest = df_rest[df_rest.index.isin(unique_slice_val)]
        key = ('_RESERVED', 'Jackknife', self.unit, lvl)
        yield key if return_cache_key else None, df_rest.reset_index()

  @staticmethod
  def get_stderrs(bucket_estimates):
    stderrs, dof = super(Jackknife, Jackknife).get_stderrs(bucket_estimates)
    return stderrs * dof / np.sqrt(dof + 1), dof

  def compute_children_sql(
      self, table, split_by, execute, mode=None, batch_size=None
  ):
    """Compute the children on leave-one-out data in SQL.

    When
    1. the data have been preaggregated, which means all the leaf Metrics are
    Sum and Count,
    2. batch_size is None,
    we compute all the leaf nodes on the preaggregated date, grouped by all the
    split_by columns we ever need, including self.unit. Then we cut the corner
    to get the leave-one-out estimates. See the doc of compute_slices() for more
    details.
    Otherwise, if batch_size is None or 1, we iterate unique units in the data.
    In iteration k, we compute the child on
    'SELECT * FROM table WHERE unit != k' to get the leave-k-out estimate.
    If batch_size is larger than 1, in each iteration, we compute the child on
    SELECT
      *
    FROM UNNEST([1, 2, ..., batch_size]) AS meterstick_resample_idx
    JOIN
    table
    ON meterstick_resample_idx != unit), split by meterstick_resample_idx in
    addition.
    At last we concat the estimates.

    Args:
      table: The table we want to query from.
      split_by: The columns that we use to split the data.
      execute: A function that can executes a SQL query and returns a DataFrame.
      mode: It's always 'mixed' or 'magic' otherwise we won't be here.
      batch_size: The number of units we handle in one iteration.

    Returns:
      A DataFrame contains all the leave-one-out estimates. Each row is a child
      metric * split_by slice and each column is an estimate.
    """
    if self.has_been_preaggregated and not batch_size:
      all_split_by = (
          split_by
          + [self.unit]
          + list(utils.get_extra_split_by(self, return_superset=True))
      )
      all_split_by_no_unit = split_by + list(
          utils.get_extra_split_by(self, return_superset=True)
      )
      filter_in_leaf = utils.push_filters_to_leaf(self)
      leafs = utils.get_leaf_metrics(filter_in_leaf)
      for m in leafs:  # filters have been handled in preaggregation
        m.where = None
      # Make sure the output column names are the same as the var so we can
      # compute_child on it later. has_been_preaggregated being True means that
      # all leaf metrics are Sum or Count.
      leafs = copy.deepcopy(leafs)
      for m in leafs:
        m.name = m.var
      leafs = metrics.MetricList(tuple(set(leafs)))
      if len(leafs) == 1:
        leafs.name = leafs.children[0].name
      bucket_res = self.compute_util_metric_on_sql(
          leafs, table, all_split_by, execute, mode=mode
      )

      if all_split_by_no_unit:
        total = bucket_res.groupby(
            level=all_split_by_no_unit, observed=True
        ).sum()
      else:
        total = bucket_res.sum()
      bucket_res = bucket_res.fillna(0)
      bucket_res = utils.adjust_slices_for_loo(bucket_res, split_by, bucket_res)
      loo = total - bucket_res
      if all_split_by_no_unit:
        # The levels might get messed up.
        loo = loo.reorder_levels(all_split_by)
      res = filter_in_leaf.children[0].compute_on(
          loo, split_by + [self.unit], melted=True
      )
      return [res.unstack(self.unit)]

    batch_size = batch_size or 1
    slice_and_units = sql.Sql(
        sql.Columns(split_by + [self.unit], distinct=True),
        table,
        self.where_,
    )
    slice_and_units = execute(str(slice_and_units))
    # Columns got sanitized in SQL generation if they have special characters.
    slice_and_units.columns = split_by + [self.unit]
    if split_by:
      slice_and_units.set_index(split_by, inplace=True)
    replicates = []
    unique_units = slice_and_units[self.unit].unique().tolist()
    if batch_size == 1:
      loo_sql = sql.Sql(None, table, where=self.where_)
      where = copy.deepcopy(loo_sql.where)
      for unit in unique_units:
        loo_where = '%s != "%s"' % (self.unit, unit)
        if pd.api.types.is_numeric_dtype(slice_and_units[self.unit]):
          loo_where = '%s != %s' % (self.unit, unit)
        loo_sql.where = sql.Filters(where).add(loo_where)
        key = ('_RESERVED', 'Jackknife', self.unit, unit)
        loo = self.compute_child_sql(loo_sql, split_by, execute, False, mode,
                                     key)
        # If a slice doesn't have the unit in the input data, we should exclude
        # the slice in the loo.
        if split_by:
          loo = slice_and_units[slice_and_units[self.unit] == unit].join(loo)
          loo.drop(self.unit, axis=1, inplace=True)
        replicates.append(utils.melt(loo))
    else:
      if split_by:
        slice_and_units.set_index(self.unit, append=True, inplace=True)
      for i in range(int(np.ceil(len(unique_units) / batch_size))):
        units = unique_units[i * batch_size:(i + 1) * batch_size]
        loo = sql.Sql(
            sql.Column('*', auto_alias=False),
            sql.Datasource(
                'UNNEST(%s)' % units, 'meterstick_resample_idx'
            ).join(table, on='meterstick_resample_idx != %s' % self.unit),
            self.where_,
        )
        key = ('_RESERVED', 'Jackknife', self.unit, tuple(units))
        loo = self.compute_child_sql(
            loo,
            split_by + ['meterstick_resample_idx'],
            execute,
            True,
            mode,
            key,
        )
        # If a slice doesn't have the unit in the input data, we should exclude
        # the slice in the loo.
        if split_by:
          loo.index.set_names(
              self.unit, level='meterstick_resample_idx', inplace=True
          )
          loo = slice_and_units.join(utils.unmelt(loo), how='inner')
          loo = utils.melt(loo).unstack(self.unit)
        else:
          loo = loo.unstack('meterstick_resample_idx')
        replicates.append(loo)
    return replicates

  def get_resampled_data_sql(
      self,
      table,
      split_by,
      global_filter,
      indexes,
      with_data,
  ):
    """Gets the SQL query that resamples the original data."""
    if self.has_been_preaggregated:
      return get_jackknife_data_fast(
          self, table, split_by, global_filter, indexes, with_data
      )
    return get_jackknife_data_general(
        self, table, split_by, global_filter, with_data
    )

  def can_precompute(self):
    """LOO can be precomputed if all leaves can be expressed as Sum or Count."""
    return self.enable_optimization and is_metric_precomputable(self)


def is_metric_precomputable(metric: MetricWithCI) -> bool:
  """If metric is precomputable in Jackknife or Bootstrap."""
  for m in metric.traverse(include_self=False):
    if isinstance(m, Operation) and not m.precomputable_in_jk_bs:
      return False
    if isinstance(m, metrics.Count) and m.distinct:
      return False
    precomputable_leaf_types = (metrics.Sum, metrics.Count)
    if not isinstance(metric, Jackknife):
      precomputable_leaf_types = tuple(
          list(precomputable_leaf_types) + [metrics.Max, metrics.Min]
      )
    if isinstance(m, precomputable_leaf_types):
      continue
    if not m.children:
      equiv, _ = utils.get_equivalent_metric(m)
      if not equiv:
        return False
      if not is_metric_precomputable(equiv):
        return False
  return True


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
    enable_optimization: If all leaf Metrics are Sum and/or Count, or can be
      expressed equivalently by Sum and/or Count, then we can preaggregate the
      data for faster computation. See compute_slices() for more details.
    has_been_preaggregated: If the Metric and data has already been
      preaggregated, this will be set to True. And all other attributes
      inherited from Operation.
  """

  def __init__(
      self,
      unit: Optional[Text] = None,
      child: Optional[metrics.Metric] = None,
      n_replicates: int = 10000,
      confidence: Optional[float] = None,
      enable_optimization=True,
      name_tmpl: Text = '{} Bootstrap',
      **kwargs,
  ):
    super(Bootstrap, self).__init__(
        unit,
        child,
        confidence,
        name_tmpl,
        additional_fingerprint_attrs=['n_replicates'],
        enable_optimization=enable_optimization,
        **kwargs,
    )
    self.n_replicates = n_replicates

  def compute_slices(self, df, split_by=None):
    """Computes Bootstrap with unit with precomputation when possible.

    For Bootstrap with unit, if all leafs can be expressed as Sum or Count, we
    can preaggregate the data for faster computation. For example,
    Bootstrap(unit, Sum('x')).compute_on(df) equals to
    Bootstrap(unit, Sum('sum(x)')).compute_on(preaggregated)
    where preaggregated is computed as
      preaggregated = df.groupby(unit)[['x']].sum()
      preaggregated.columns = ['sum(x)']
    1. We can also preaggregate for Count, Max and Min.
    2. We can apply the trick to Metrics that can be expressed by Sum and Count,
      which includes Mean, Dot, Variance, StandardDeviation, CV, Correlation and
      Cov.
    3. The same trick applies when there are Operations, for example,
      Bootstrap(unit, PercentChange(Sum('x'))).compute_on(df)
      but now we cannot further drop the unit.

    Here we specifically handle such situation by
    1. replace all leaf Metrics with Sum and/or Count.
    2. push all filters down to the leaf Metrics.
    3. Compute the child using split_by + [self.unit].
    4. Find all the results of Sum/Count/Max/Min in the cache then
      4.1 Concat them into the preaggregated dataframe. A Sum('x') will have its
        result in the preaggregated df under column "sum('x') where {filter}".
        If there is no filter the 'where' part is skipped. A Count('y') will
        have its result under column "count('x') where {filter}".
      4.2 Replace all the leaf Metrics to use the columns in the preaggregated
        df. For example, Sum('x') will be replaced by Sum("sum('x')") and
        Count('x', where='foo') will become Sum("count('x') where foo").
    5. A tricky part is units might get dropped by the filters in the children.
      We adjust it in get_preaggregated_data().

    Args:
      df: The DataFrame to compute on.
      split_by: The columns that we use to split the data.

    Returns:
      The unmelted result.
    """
    if self.has_been_preaggregated or not self.can_precompute():
      return super(Bootstrap, self).compute_slices(df, split_by)
    if self != utils.get_fully_expanded_equivalent_metric_tree(self)[0]:
      util, df = utils.get_fully_expanded_equivalent_metric_tree(self, df)
      return self.compute_util_metric_on(util, df, split_by)
    preagg, preagg_df = get_preaggregated_data(self, df, split_by)
    return self.compute_util_metric_on(preagg, preagg_df, split_by)

  def get_samples(self, df, split_by=None):
    """Resamples for Bootstrap. When samples are likely to repeat, cache."""
    # If there is no extra split_by added, each unit will correspond to one row
    # in the preaggregated data so we can just sample by rows.
    if self.unit is None or (
        self.has_been_preaggregated and not utils.get_extra_split_by(self, True)
    ):
      to_sample = self.group(df, split_by)
      for _ in range(self.n_replicates):
        yield None, to_sample.sample(frac=1, replace=True)
    else:
      grp_by = split_by + [self.unit] if split_by else self.unit
      df = df.set_index(grp_by)
      grped = df.groupby(grp_by, observed=True)
      idx = grped.indices
      n_units = len(idx)
      units_df = grped.first().iloc[:, [0]]
      units_grped = self.group(units_df, split_by)
      use_cache = np.log(self.n_replicates) / n_units > np.log(n_units)
      sampled = set()
      for _ in range(self.n_replicates):
        resampled = units_grped.sample(frac=1, replace=True).index
        if use_cache:
          cache_key = tuple(resampled)
          if cache_key in sampled:
            yield cache_key, None
          else:
            sampled.add(cache_key)
            yield cache_key, df.loc[resampled].reset_index()
        else:
          yield None, df.loc[resampled].reset_index()

  def compute_children_sql(
      self, table, split_by, execute, mode=None, batch_size=None
  ):
    """Compute the children on resampled data in SQL.

    We compute the child on bootstrapped data in a batched way. We bootstrap for
    batch_size in one iteration. Namely, it's equivalent to compute self but
    setting n_replicates to batch_size.

    Args:
      table: The table we want to query from.
      split_by: The columns that we use to split the data.
      execute: A function that can executes a SQL query and returns a DataFrame.
      mode: It's always 'mixed' or 'magic' otherwise we won't be here.
      batch_size: The number of units we handle in one iteration.

    Returns:
      A DataFrame contains all the bootstrap estimates. Each row is a child
      metric * split_by slice and each column is an estimate.
    """
    batch_size = batch_size or 1000
    global_filter = utils.get_global_filter(self)
    util_metric = copy.deepcopy(self)
    util_metric.n_replicates = batch_size
    util_metric.confidence = None
    with_data = sql.Datasources()
    if not sql.Datasource(table).is_table:
      table = with_data.add(sql.Datasource(table, 'BootstrapData'))
    with_data2 = copy.deepcopy(with_data)
    _, with_data = util_metric.get_resampled_data_sql(
        table,
        sql.Columns(split_by),
        global_filter,
        None,
        with_data,
    )
    resampled = with_data.children.popitem()[1]
    resampled.with_data = with_data
    replicates = []
    for _ in range(self.n_replicates // batch_size):
      bst = self.children[0].compute_on_sql(
          resampled, ['meterstick_resample_idx'] + split_by, execute, True, mode
      )
      replicates.append(bst.unstack('meterstick_resample_idx'))
    util_metric.n_replicates = self.n_replicates % batch_size
    if util_metric.n_replicates:
      _, with_data2 = util_metric.get_resampled_data_sql(
          table,
          sql.Columns(split_by),
          global_filter,
          None,
          with_data2,
      )
      resampled = with_data2.children.popitem()[1]
      resampled.with_data = with_data2
      bst = self.children[0].compute_on_sql(
          resampled, ['meterstick_resample_idx'] + split_by, execute, True, mode
      )
      replicates.append(bst.unstack('meterstick_resample_idx'))
    return replicates

  def get_resampled_data_sql(
      self, table, split_by, global_filter, indexes, with_data
  ):
    """Gets self.n_replicates bootstrap resamples."""
    del indexes  # not used
    if not self.unit:
      return bootstrap_by_row(self, table, split_by, global_filter, with_data)
    return bootstrap_by_unit(
        self,
        table,
        split_by,
        global_filter,
        with_data,
    )

  def can_precompute(self):
    return (
        self.unit and self.enable_optimization and is_metric_precomputable(self)
    )


class PoissonBootstrap(Bootstrap):
  """Class for PoissonBootstrap estimates of standard errors.

  The only difference to Bootstrap is that PoissonBootstrap uses Poisson(1)
  instead of multinomial distribution in resampling. See
  https://www.unofficialgoogledatascience.com/2015/08/an-introduction-to-poisson-bootstrap26.html
  for an introduction.
  """

  def __init__(
      self,
      unit: Optional[Text] = None,
      child: Optional[metrics.Metric] = None,
      n_replicates: int = 10000,
      confidence: Optional[float] = None,
      enable_optimization=True,
      name_tmpl: Text = '{} Poisson Bootstrap',
      **kwargs,
  ):
    super(PoissonBootstrap, self).__init__(
        unit,
        child,
        n_replicates,
        confidence,
        enable_optimization,
        name_tmpl,
        **kwargs,
    )

  def get_samples(self, df, split_by=None):
    """Resamples for PoissonBootstrap.

    There are three cases here.
    1. When no unit, then by the definition of can_precompute(), optimization is
    off. We simply get sample weights for each row than duplicate the rows by
    the weight.
    2. When there is unit and optimization is disabled, then we get the sample
    weight for each split_by + unit slice, then duplicate the slice by the
    weight.
    3. If there is optimization, which means df is a preaggregated data and self
    only has Sum/Count/Max/Min as leaf nodes. The columns names in preaggregated
    data starts with 'sum_', 'count_', 'max_' and 'min_', indicating what type
    of Metric will consume them. For columns that starts with 'sum_' or
    'count_', we simply multiply them by the weight, which will give us the sum
    we need. For columns that starts with 'max_' and 'min_', no action needed.
    If the number of unique units are less than 7, we enable caching for samples
    we generate.

    Args:
      df: The DataFrame to compute on.
      split_by: The columns that we use to split the data.

    Yields:
      A cache_key if it makes sense to cache otherwise None, and resampled data.
    """
    n_split_by = (
        len(split_by)
        + len([self.unit] if self.unit else [])
        + len(utils.get_extra_split_by(self, True))
    )
    var_cols = df.columns[n_split_by:]
    sum_or_ct_cols = [
        c for c in var_cols if c.startswith('sum_') or c.startswith('count_')
    ]
    if self.unit:
      grp_by = split_by + [self.unit]
      grped = df.groupby(grp_by, observed=True)
      idx_rows = np.array([*grped.indices.values()], dtype=object)
      idx_vals = grped.first().index
      n = len(grped.indices)
      weight_col = utils.get_unique_prefix(df)
      # Poisson(1) generates a number under 6 with >99.9% probability. The
      # default n_replicates is 10000. We cache when n < 7 because 7^5 > 10000
      # while 6^5 < 10000.
      use_cache = n < 7
      sampled = set()
    else:
      n = len(df)
    row = np.arange(n)
    for _ in range(self.n_replicates):
      # If there is no extra split_by added, each unit will correspond to one
      # row in the preaggregated data so we can just sample by rows.
      if self.unit is None or (
          self.has_been_preaggregated
          and not utils.get_extra_split_by(self, True)
      ):
        weights = self.get_sample_weight(n)
        yield None, df.iloc[row.repeat(weights)]
      else:
        weights = self.get_sample_weight(n)
        cache_key = None
        if use_cache:
          cache_key = tuple(weights)
          if cache_key in sampled:
            yielded = True
            yield cache_key, None
          else:
            sampled.add(cache_key)
            yielded = False
        if not use_cache or not yielded:
          if not self.has_been_preaggregated:
            sampled_rows = (
                np.concatenate(idx_rows.repeat(weights, 0))
                if weights.any()
                else []
            )
            yield cache_key, df.iloc[sampled_rows]
          else:
            weights = pd.Series(weights, index=idx_vals, name=weight_col)
            selected = weights > 0
            sampled_rows = (
                np.concatenate(idx_rows[selected]) if selected.any() else []
            )
            weights = weights[selected]
            resampled = df.iloc[sampled_rows].set_index(grp_by)
            if not resampled.empty:
              resampled = resampled.join(weights)
              resampled[sum_or_ct_cols] = resampled[sum_or_ct_cols].multiply(
                  resampled[weight_col], axis=0
              )
            yield cache_key, resampled.reset_index()

  def get_sample_weight(self, n):
    return np.random.poisson(size=n)

  def get_resampled_data_sql(
      self, table, split_by, global_filter, indexes, with_data
  ):
    """Gets self.n_replicates Poisson bootstrap resamples.

    The function makes three or four subqueries. The first one adds a uniformly
    distributed random variable to the data. The second one uses the variable to
    get sample weights from Poisson(1) distribution. The rest uses the weights
    to resample the original table.
    The first subquery looks like
      PoissonBootstrapDataWithUniformVar AS (SELECT
        *,
        RAND() AS poisson_bootstrap_uniform_var
      FROM T
      JOIN
      UNNEST(GENERATE_ARRAY(1, n_replicates)) AS meterstick_resample_idx).
    There are two variations. First, when we know what columns are in the table
    because the table has been preaggregated, then we explicitly SELECT those
    columns instead of using '*'. Second, RAND() is used when sampling by row.
    When we need to sample by groups, we use
    FARM_FINGERPRINT(CONCAT(CAST(grp AS STRING),
      CAST(meterstick_resample_idx AS STRING)))
      / 0xFFFFFFFFFFFFFFFF + 0.5
    to get the uniformly distributed random variable. The hashing makes sure
    same group gets the same weight.

    The second query looks like
      PoissonBootstrapDataWithPoissonWeight AS (SELECT
        * EXCEPT(poisson_bootstrap_uniform_var),
        CASE
          WHEN poisson_bootstrap_uniform_var <= 0.7357588823428847 THEN 1
          WHEN poisson_bootstrap_uniform_var <= 0.9196986029286058 THEN 2
          WHEN poisson_bootstrap_uniform_var <= 0.9810118431238462 THEN 3
          WHEN poisson_bootstrap_uniform_var <= 0.9963401531726563 THEN 4
          WHEN poisson_bootstrap_uniform_var <= 0.9994058151824183 THEN 5
          WHEN poisson_bootstrap_uniform_var <= 0.999916758850712 THEN 6
          WHEN poisson_bootstrap_uniform_var <= 0.9999897508033253 THEN 7
          WHEN poisson_bootstrap_uniform_var <= 0.999998874797402 THEN 8
          WHEN poisson_bootstrap_uniform_var <= 0.9999998885745217 THEN 9
          WHEN poisson_bootstrap_uniform_var <= 0.9999999899522336 THEN 10
          WHEN poisson_bootstrap_uniform_var <= 0.9999999991683892 THEN 11
          WHEN poisson_bootstrap_uniform_var <= 0.9999999999364022 THEN 12
          WHEN poisson_bootstrap_uniform_var <= 0.9999999999954802 THEN 13
          WHEN poisson_bootstrap_uniform_var <= 0.9999999999997 THEN 14
          WHEN poisson_bootstrap_uniform_var <= 0.9999999999999813 THEN 15
          WHEN poisson_bootstrap_uniform_var <= 0.9999999999999989 THEN 16
          WHEN poisson_bootstrap_uniform_var <= 0.9999999999999999 THEN 17
          ELSE 18
        END AS poisson_bootstrap_weight
      FROM PoissonBootstrapDataWithUniformVar
      WHERE
      poisson_bootstrap_uniform_var > 0.36787944117144245)
    Again when we know column names, we will explicitly SELECT them instead of
    using '*'. The cutoff values are obtained from
    scipy.stats.poisson.cdf(range(0, 19), 1) / scipy.stats.poisson.cdf(19, 1).
    The cutoff value for 0 is directly used in the WHERE clause.

    The rest subqueries depend on if the Metric has been preaggregated. If not,
    we use
      PoissonBootstrapResampledData AS (SELECT
        * EXCEPT(poisson_bootstrap_weight, poisson_bootstrap_weight_unnested)
      FROM PoissonBootstrapDataWithPoissonWeight
      JOIN
      UNNEST(GENERATE_ARRAY(1, poisson_bootstrap_weight))
        AS poisson_bootstrap_weight_unnested),
      ResampledResults AS (SELECT
        meterstick_resample_idx,
        SUM(x) AS sum_x
      FROM PoissonBootstrapResampledData
      GROUP BY meterstick_resample_idx)
    to get the resampled data.
    If the data has been preaggregated, it means all leaf Metrics are one of
    Sum/Count/Max/Min and the columns in the table, except for split_by, all
    start with 'sum_', 'count_', 'max_' or 'min_'. The prefix indicates how the
    column will be consumed. For columns starting with 'sum_' or 'count_',
    their values will be summed so we can directly multiply the weights to them.
    For columns starting with 'max_' or 'min_' we don't need to do anything. So
    the query looks like
      PoissonBootstrapResampledData AS (SELECT
        split_by,
        unit,
        max_x,
        min_y,
        sum_x * poisson_bootstrap_weight AS sum_x,
        count_x * poisson_bootstrap_weight AS count_x,
        meterstick_resample_idx
      FROM PoissonBootstrapDataWithPoissonWeight).

    Args:
      table: The table we want to resample.
      split_by: The columns that we use to split the data.
      global_filter: All the filters that applied to the PoissonBootstrap.
      indexes: Unused.
      with_data: A global variable that contains all the WITH clauses we need.

    Returns:
      The alias of the table in the WITH clause that has all resampled data.
      The global with_data which holds all datasources we need in the WITH
        clause.
    """
    del indexes  # unused
    if self.has_been_preaggregated:
      uniform_columns = sql.Columns(with_data[table].all_columns.aliases).add(
          'meterstick_resample_idx'
      )
    else:
      table = sql.Datasource(table)
      if not table.is_table:
        table.alias = table.alias or 'RawData'
        table = with_data.add(table)
      uniform_columns = sql.Columns(sql.Column('*', auto_alias=False))
    global_filter = sql.Filters(global_filter).add(self.where_)
    uniform_var = sql.Column('RAND()', alias='poisson_bootstrap_uniform_var')
    split_by_cols = (
        split_by.aliases
        if self.has_been_preaggregated
        else split_by.original_columns
    )
    if self.unit:
      cols = ', '.join(
          map(
              'CAST({} AS STRING)'.format,
              (split_by_cols or []) + [self.unit, 'meterstick_resample_idx'],
          )
      )
      uniform_var = sql.Column(
          f'FARM_FINGERPRINT(CONCAT({cols})) / 0xFFFFFFFFFFFFFFFF + 0.5',
          alias='poisson_bootstrap_uniform_var',
      )
    uniform_columns.add(uniform_var)
    replicates = sql.Datasource(
        'UNNEST(GENERATE_ARRAY(1, %s))' % self.n_replicates,
        'meterstick_resample_idx',
    )
    table_with_uniform_var = sql.Sql(
        uniform_columns, sql.Join(table, replicates), global_filter
    )
    table_with_uniform_var_alias = with_data.add(
        sql.Datasource(
            table_with_uniform_var, 'PoissonBootstrapDataWithUniformVar'
        )
    )

    if self.has_been_preaggregated:
      poisson_weight_columns = sql.Columns(uniform_columns.aliases).difference(
          'poisson_bootstrap_uniform_var'
      )
    else:
      poisson_weight_columns = sql.Columns(
          sql.Column(
              '* EXCEPT(poisson_bootstrap_uniform_var)', auto_alias=False
          )
      )
    # The cutoff values are obtained from
    # scipy.stats.poisson.cdf(range(0, 19), 1) / scipy.stats.poisson.cdf(19, 1).
    # The cutoff value for 0 is not used here but used in the filter of
    # table_with_poisson_weight below.
    poisson_weight = sql.Column(
        """CASE
    WHEN poisson_bootstrap_uniform_var <= 0.7357588823428847 THEN 1
    WHEN poisson_bootstrap_uniform_var <= 0.9196986029286058 THEN 2
    WHEN poisson_bootstrap_uniform_var <= 0.9810118431238462 THEN 3
    WHEN poisson_bootstrap_uniform_var <= 0.9963401531726563 THEN 4
    WHEN poisson_bootstrap_uniform_var <= 0.9994058151824183 THEN 5
    WHEN poisson_bootstrap_uniform_var <= 0.999916758850712 THEN 6
    WHEN poisson_bootstrap_uniform_var <= 0.9999897508033253 THEN 7
    WHEN poisson_bootstrap_uniform_var <= 0.999998874797402 THEN 8
    WHEN poisson_bootstrap_uniform_var <= 0.9999998885745217 THEN 9
    WHEN poisson_bootstrap_uniform_var <= 0.9999999899522336 THEN 10
    WHEN poisson_bootstrap_uniform_var <= 0.9999999991683892 THEN 11
    WHEN poisson_bootstrap_uniform_var <= 0.9999999999364022 THEN 12
    WHEN poisson_bootstrap_uniform_var <= 0.9999999999954802 THEN 13
    WHEN poisson_bootstrap_uniform_var <= 0.9999999999997 THEN 14
    WHEN poisson_bootstrap_uniform_var <= 0.9999999999999813 THEN 15
    WHEN poisson_bootstrap_uniform_var <= 0.9999999999999989 THEN 16
    WHEN poisson_bootstrap_uniform_var <= 0.9999999999999999 THEN 17
    ELSE 18
  END""",
        alias='poisson_bootstrap_weight',
    )
    poisson_weight_columns.add(poisson_weight)
    table_with_poisson_weight = sql.Sql(
        poisson_weight_columns,
        table_with_uniform_var_alias,
        sql.Filter('poisson_bootstrap_uniform_var > 0.36787944117144245'),
    )
    table_with_poisson_weight_alias = with_data.add(
        sql.Datasource(
            table_with_poisson_weight, 'PoissonBootstrapDataWithPoissonWeight'
        )
    )

    if self.has_been_preaggregated:
      poisson_sampled_columns = sql.Columns()
      for c in poisson_weight_columns:
        if c.alias == 'poisson_bootstrap_weight':
          continue
        elif c.alias in split_by.aliases or not (
            c.alias.startswith('sum_') or c.alias.startswith('count_')
        ):
          poisson_sampled_columns.add(c.alias)
        else:
          col = c * sql.Column('poisson_bootstrap_weight')
          poisson_sampled_columns.add(col.set_alias(c.alias))
        poisson_sampled_table = sql.Sql(
            poisson_sampled_columns, table_with_poisson_weight_alias
        )
    else:
      poisson_sampled_columns = sql.Columns(
          sql.Column(
              (
                  '* EXCEPT(poisson_bootstrap_weight,'
                  ' poisson_bootstrap_weight_unnested)'
              ),
              auto_alias=False,
          )
      )
      replicates = sql.Datasource(
          'UNNEST(GENERATE_ARRAY(1, poisson_bootstrap_weight))',
          'poisson_bootstrap_weight_unnested',
      )
      poisson_sampled_table = sql.Sql(
          poisson_sampled_columns,
          sql.Join(table_with_poisson_weight_alias, replicates),
      )

    poisson_sampled_table_alias = with_data.add(
        sql.Datasource(poisson_sampled_table, 'PoissonBootstrapResampledData')
    )
    return poisson_sampled_table_alias, with_data


def get_preaggregated_data(m, df, split_by):
  """Gets the preaggegated Metric and data.

  Read the doc of Bootstrap.compute_slices() first.
  Here we construct the preaggegated Metric and data so that
  preagg.compute_on(preagg_df, split_by) == m.compute_on(df, split_by).
  It's achieved by
  1. push all filters to leaf nodes.
  2. collect all unique leaf nodes to a MetricList, leafs.
  3. collect all split_by dimensions we need to keep in the preaggegated data.
  4. The preaggegated data is then
    leafs.compute_on(df, all_split_by).reset_index().
  A subtle thing is that slices might get dropped by filters in the children
  during the computation. For example, the preaggegated data we got for
  Bootstrap('unit', Sum(x, where='unit != 0)) will not have the rows whose unit
  is 0. We need to recover the dropped slices fill in NA.

  Args:
    m: A Bootstrap with unit.
    df: The data passed to Jackknife/Bootstrap.compute_on().
    split_by: The split_by passed to Jackknife/Bootstrap.compute_on().

  Returns:
    The equivalent Metric that computes on the preaggregated data;
    The preaggregated dataframe.
  """
  all_split_by = (
      split_by
      + ([m.unit] if m.unit else [])
      + list(utils.get_extra_split_by(m, return_superset=True))
  )
  original_idx = df.groupby(all_split_by, observed=True).first().iloc[:, [0]]
  filter_in_leaf = utils.push_filters_to_leaf(m)
  leafs = metrics.MetricList(tuple(set(utils.get_leaf_metrics(filter_in_leaf))))
  preagg_df = m.compute_util_metric_on(leafs, df, all_split_by)
  preagg_leafs = get_preaggregated_metric_tree(leafs)
  preagg_df.columns = [c.var for c in preagg_leafs]
  preagg_df = preagg_df.loc[:, ~preagg_df.columns.duplicated()].copy()
  preagg_df = preagg_df.reindex(original_idx.index)
  if all_split_by:
    preagg_df.reset_index(all_split_by, inplace=True)
  for l, p in zip(leafs, preagg_leafs):
    key = utils.CacheKey(l, m.cache_key, l.where_, all_split_by)
    res = m.get_cached(key)
    key = key.replace_metric(p).replace_where(m.cache_key.where)
    m.save_to_cache(key, res)
  preagg = get_preaggregated_metric_tree(filter_in_leaf)
  return preagg, preagg_df


def get_preaggregated_metric_tree(m):
  """Gets the equivalent Metric of m on the preaggregated data."""
  if not isinstance(m, metrics.Metric):
    return m
  if not m.children:
    return get_preaggregated_metric(m)
  m = copy.copy(m)
  m.children = [get_preaggregated_metric_tree(c) for c in m.children]
  m.has_been_preaggregated = True
  return m


def get_preaggregated_metric(m):
  """Gets the equivalent metric of on the preaggregated data if m is a leaf."""
  var = get_preaggregated_metric_var(m)
  if isinstance(m, metrics.Max):
    return metrics.Max(var, name=m.name)
  if isinstance(m, metrics.Min):
    return metrics.Min(var, name=m.name)
  return metrics.Sum(var, name=m.name)


def get_preaggregated_metric_var(m: metrics.Metric):
  """Gets the new column name for leaf metric m in the preaggregated data."""
  if not isinstance(m, (metrics.Sum, metrics.Count, metrics.Max, metrics.Min)):
    raise ValueError(
        f'Expecting Sum/Count/Max/Min but got f{m.name} is f{type(m)}.'
    )
  tmpl_lookup = {
      metrics.Sum: 'sum(%s)',
      metrics.Count: 'count(%s)',
      metrics.Max: 'max(%s)',
      metrics.Min: 'min(%s)',
  }
  name = tmpl_lookup[type(m)] % m.var
  name = f'{name} where {m.where}' if m.where else name
  return sql.Column(name).alias


def get_se_sql(
    metric, table, split_by, global_filter, indexes, with_data
):
  """Gets the SQL query that computes the standard error and dof if needed."""
  samples, with_data = metric.children[0].get_sql_and_with_clause(
      table,
      sql.Columns(split_by).add('meterstick_resample_idx'),
      global_filter,
      sql.Columns(indexes).add('meterstick_resample_idx'),
      sql.Filters(),
      with_data,
  )
  samples_alias = with_data.merge(sql.Datasource(samples, 'ResampledResults'))

  columns = sql.Columns()
  groupby = sql.Columns(
      (c.alias for c in samples.groupby if c != 'meterstick_resample_idx')
  )
  for c in samples.columns:
    if c == 'meterstick_resample_idx':
      continue
    elif c in indexes.aliases:
      groupby.add(c.alias)
    else:
      alias = c.alias
      se = sql.Column(c.alias, 'STDDEV_SAMP({})',
                      '%s Bootstrap SE' % c.alias_raw)
      if isinstance(metric, Jackknife):
        adjustment = sql.Column(
            sql.SAFE_DIVIDE.format(
                numer='COUNT({c}) - 1', denom='SQRT(COUNT({c}))'
            ).format(c=alias)
        )
        se = (se * adjustment).set_alias('%s Jackknife SE' % c.alias_raw)
      columns.add(se)
      if metric.confidence:
        columns.add(sql.Column(alias, 'COUNT({}) - 1', '%s dof' % c.alias_raw))
  return sql.Sql(columns, samples_alias, groupby=groupby), with_data


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


def get_jackknife_data_general(
    metric, table, split_by, global_filter, with_data
):
  """Gets jackknife samples.

  If the leave-one-out estimates can be precomputed, see the doc of
  get_jackknife_data_fast().
  Otherwise for general cases, the SQL is constructed as
  1. if split_by is None:
    WITH
    Buckets AS (SELECT DISTINCT unit AS meterstick_resample_idx
    FROM $DATA
    WHERE global_filter),
    JackknifeResampledData AS (SELECT
      *
    FROM $DATA
    CROSS JOIN
    Buckets
    WHERE
    meterstick_resample_idx != unit AND global_filter)

  2. if split_by is not None:
    WITH
    Buckets AS (SELECT DISTINCT
      split_by AS jk_split_by,
      unit AS meterstick_resample_idx
    FROM $DATA
    WHERE global_filter
    GROUP BY jk_split_by),
    JackknifeResampledData AS (SELECT
      *
    FROM $DATA
    JOIN
    Buckets
    ON jk_split_by = split_by AND meterstick_resample_idx != unit
    WHERE global_filter)

  Args:
    metric: An instance of Jackknife.
    table: The table we want to query from.
    split_by: The columns that we use to split the data.
    global_filter: The filters that can be applied to the whole Metric tree.
    with_data: A global variable that contains all the WITH clauses we need.

  Returns:
    The alias of the table in the WITH clause that has all resampled data.
    The global with_data which holds all datasources we need in the WITH clause.
  """
  unit = metric.unit
  unique_units = sql.Columns(
      (sql.Column(unit, alias='meterstick_resample_idx')), distinct=True
  )
  if split_by:
    groupby = sql.Columns(
        (sql.Column(c.expression, alias='jk_%s' % c.alias) for c in split_by)
    )
    cols = sql.Columns(groupby.add(unique_units), distinct=True)
    buckets = sql.Sql(cols, table, global_filter)
    buckets_alias = with_data.add(sql.Datasource(buckets, 'Buckets'))
    on = sql.Filters(('%s.%s = %s' % (buckets_alias, c.alias, s.expression)
                      for c, s in zip(groupby, split_by)))
    on.add('meterstick_resample_idx != %s' % unit)
    jk_from = sql.Join(table, buckets_alias, on)
    jk_data_table = sql.Sql(
        sql.Columns(sql.Column('*', auto_alias=False)),
        jk_from,
        where=global_filter,
    )
    jk_data_table = sql.Datasource(jk_data_table, 'JackknifeResampledData')
    jk_data_table_alias = with_data.add(jk_data_table)
  else:
    buckets = sql.Sql(unique_units, table, where=global_filter)
    buckets_alias = with_data.add(sql.Datasource(buckets, 'Buckets'))
    jk_from = sql.Join(table, buckets_alias, join='CROSS')
    jk_data_table = sql.Sql(
        sql.Column('*', auto_alias=False),
        jk_from,
        where=sql.Filters('meterstick_resample_idx != %s' % unit).add(
            global_filter
        ),
    )
    jk_data_table = sql.Datasource(jk_data_table, 'JackknifeResampledData')
    jk_data_table_alias = with_data.add(jk_data_table)

  return jk_data_table_alias, with_data


def get_jackknife_data_fast(
    metric, table, split_by, global_filter, indexes, with_data
):
  """Gets jackknife samples in a fast way for precomputable Jackknife.

  If all the leaf Metrics are Sum and/or Count, we can compute the
  leave-one-out (LOO) estimates faster. The query will look like
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
    unit AS meterstick_resample_idx,
    # if needs_adjustment
    total.`sum(X)` - COALESCE(unit.`sum(X)`, 0) AS `sum(X)`
    # else
    total.`sum(X)` - unit.`sum(X)` AS `sum(X)`
  FROM
  TotalCount AS total
  RIGHT JOIN  # Or CROSS JOIN (SELECT DISTINCT unit FROM UnitSliceCount)
  (SELECT DISTINCT
    split_by,
    unit
  FROM UnitSliceCount)
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
    with_data: A global variable that contains all the WITH clauses we need.

  Returns:
    The alias of the table in the WITH clause that has all resampled result.
    The global with_data which holds all datasources we need in the WITH clause.
  """
  all_indexes = sql.Columns(indexes).add(
      utils.get_extra_split_by(metric, return_superset=True)
  )
  indexes_and_unit = sql.Columns(all_indexes).add(metric.unit)

  columns_to_preagg = sql.Columns(sql.Column('COUNT(*)', alias='ct'))
  columns_in_loo = sql.Columns()
  # When there is any filter or Operation inside Jackknife, we need adjustments.
  needs_adjustment = getattr(
      metric, 'has_local_filter', False
  ) or split_by != sql.Columns(indexes)
  # columns_to_preagg and columns_in_loo are filled in-place.
  modified_jk = modify_descendants_for_jackknife_fast(
      metric, columns_to_preagg, columns_in_loo, global_filter, needs_adjustment
  )
  metric.children = modified_jk.children

  unit_slice_ct_table = sql.Sql(
      columns_to_preagg, table, global_filter, indexes_and_unit
  )
  unit_slice_ct_alias = with_data.add(
      sql.Datasource(unit_slice_ct_table, 'UnitSliceCount'))

  total_ct_table = sql.Sql(
      sql.Columns(
          [sql.Column(c, 'SUM({})', c) for c in columns_to_preagg.aliases]),
      unit_slice_ct_alias,
      groupby=indexes_and_unit.difference(metric.unit).aliases)
  total_ct_alias = with_data.add(sql.Datasource(total_ct_table, 'TotalCount'))
  total_ct = sql.Datasource(total_ct_alias, 'total_table')

  split_by_and_unit = sql.Columns(split_by).add(metric.unit)
  all_slices = sql.Datasource(
      sql.Sql(
          sql.Columns(split_by_and_unit.aliases, distinct=True),
          unit_slice_ct_alias))
  if split_by:
    loo_from = total_ct.join(all_slices, using=split_by, join='RIGHT')
  else:
    loo_from = total_ct.join(all_slices, join='CROSS')
  loo_from = loo_from.join(
      sql.Datasource(unit_slice_ct_alias, 'unit_slice_table'),
      using=indexes_and_unit.aliases,
      join='LEFT')
  loo = sql.Sql(
      sql.Columns(all_indexes.aliases)
      .add(
          sql.Column(
              sql.Column(metric.unit).alias, alias='meterstick_resample_idx'
          )
      )
      .add(columns_in_loo),
      loo_from,
      'total_table.ct - COALESCE(unit_slice_table.ct, 0) > 0',
  )
  loo_table = with_data.add(sql.Datasource(loo, 'LOO'))
  return loo_table, with_data


def modify_descendants_for_jackknife_fast(
    metric,
    columns_to_preagg,
    columns_in_loo,
    global_filter,
    needs_adjustment,
):
  """Gets the columns for leaf Metrics and modify them for fast Jackknife SQL.

  See the doc of get_jackknife_data_fast() first. Here we
  1. collects the LOO columns for all Sum and Count.
  2. Modify them in-place so when we generate SQL later, they know what column
    to use. For example, Sum('X') would generate a SQL column 'SUM(X) AS sum_x',
    but as we precompute LOO estimates, we need to query from a table that has
    "total_table.`sum(X)` - COALESCE(unit_slice_table.`sum(X)`, 0) AS `sum(X)`".
    So the expression of Sum('X') should now become 'SUM(`sum(X)`) AS `sum(X)`'.
    Here we will replace the metric with Sum('sum(X)', metric.name).
  3. Removes filters as they have already been applied in the LOO table. Note
    that we made a copy in get_se_sql for metric so the removal won't affect the
    metric used in point estimate computation.
  4. For Operations, their extra_index columns appear in indexes. Any forbidden
    character in the name will be replaced/dropped in LOO so we have to change
    extra_index. For example, Distribution('$Foo') will generate a column
    '$Foo AS macro_Foo' in LOO so we need to replace '$Foo' with 'macro_Foo'.

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
    needs_adjustment: If we need to adjust the slices. See
      utils.adjust_slices_for_loo() for more discussions.

  Returns:
    The modified metric tree.
  """
  if not isinstance(metric, metrics.Metric):
    return metric

  metric = copy.deepcopy(metric)
  local_filter = sql.Filters(metric.where_)
  metric.where = None
  if needs_adjustment:
    tmpl = 'total_table.%s - COALESCE(unit_slice_table.%s, 0)'
  else:
    tmpl = 'total_table.%s - unit_slice_table.%s'
  if isinstance(metric, (metrics.Sum, metrics.Count)):
    filters = sql.Filters(local_filter).remove(global_filter)
    c = sql.Column(metric.var).alias
    op = 'COUNT({})' if isinstance(metric, metrics.Count) else 'SUM({})'
    col = sql.Column(c, op, filters=filters)
    columns_to_preagg.add(col)
    loo = sql.Column(tmpl % (col.alias, col.alias), alias=metric.name)
    columns_in_loo.add(loo)
    return metrics.Sum(loo.alias, metric.name)

  if isinstance(metric, Operation):
    metric.extra_index = sql.Columns(metric.extra_index).aliases
    metric.extra_split_by = sql.Columns(metric.extra_split_by).aliases

  new_children = []
  for m in metric.children:
    modified = modify_descendants_for_jackknife_fast(
        m, columns_to_preagg, columns_in_loo, global_filter, needs_adjustment
    )
    new_children.append(modified)
  metric.children = new_children
  return metric


def bootstrap_by_row(
    metric, table, split_by, global_filter, with_data, columns_in_table=None
):
  """Gets metric.n_replicates bootstrap resamples for Bootstrap without unit.

  The SQL is constructed as
    WITH
    BootstrapRandomRows AS (SELECT
      *,  # or columns_in_table and meterstick_resample_idx if columns_in_table
      global_filter AS meterstick_bs_filter,
      ROW_NUMBER() OVER (PARTITION BY meterstick_resample_idx, global_filter)
        AS meterstick_bs_row_number,
      CEILING(RAND() * COUNT(*)
        OVER (PARTITION BY meterstick_resample_idx, global_filter))
        AS meterstick_bs_random_row_number,
      $RenamedSplitByIfAny AS renamed_split_by,
    FROM table
    JOIN
    UNNEST(GENERATE_ARRAY(1, metric.n_replicates)) AS meterstick_resample_idx),
    BootstrapRandomChoices AS (SELECT
      b.*
    FROM (SELECT
      split_by,
      meterstick_resample_idx,
      meterstick_bs_filter,
      meterstick_bs_random_row_number AS meterstick_bs_row_number
    FROM BootstrapRandomRows) AS a
    JOIN
    BootstrapRandomRows AS b
    USING (split_by, meterstick_resample_idx, meterstick_bs_row_number,
      meterstick_bs_filter)
    WHERE
    meterstick_bs_filter),
  The filter parts are optional.

  Args:
    metric: An instance of Bootstrap.
    table: The table we want to resample.
    split_by: The columns that we use to split the data.
    global_filter: All the filters that applied to the Bootstrap.
    with_data: A global variable that contains all the WITH clauses we need.
    columns_in_table: All the columns we want to SELECT from `table`. If None,
      we do SELECT * FROM table.

  Returns:
    The alias of the table in the WITH clause that has all resampled data.
    The global with_data which holds all datasources we need in the WITH clause.
  """
  if columns_in_table:
    columns = sql.Columns(columns_in_table).add('meterstick_resample_idx')
  else:
    columns = sql.Columns(sql.Column('*', auto_alias=False))
  partition = split_by.expressions + ['meterstick_resample_idx']
  if global_filter:
    columns.add(sql.Column(str(global_filter), alias='meterstick_bs_filter'))
    partition.append(str(global_filter))
  row_number = sql.Column(
      'ROW_NUMBER()', alias='meterstick_bs_row_number', partition=partition
  )
  length = sql.Column('COUNT(*)', partition=partition)
  random_row_number = sql.Column('RAND()') * length
  random_row_number = sql.Column(
      'CEILING(%s)' % random_row_number.expression,
      alias='meterstick_bs_random_row_number',
  )
  columns.add((row_number, random_row_number))
  columns.add((i for i in split_by if i != i.alias))
  replicates = sql.Datasource(
      'UNNEST(GENERATE_ARRAY(1, %s))' % metric.n_replicates,
      'meterstick_resample_idx',
  )
  random_choice_table = sql.Sql(columns, sql.Join(table, replicates))
  random_choice_table_alias = with_data.add(
      sql.Datasource(random_choice_table, 'BootstrapRandomRows'))

  using = (
      sql.Columns(partition)
      .add('meterstick_bs_row_number')
      .difference(str(global_filter))
  )
  if global_filter:
    using.add('meterstick_bs_filter')
  random_rows = sql.Sql(
      sql.Columns(using)
      .difference('meterstick_bs_row_number')
      .add(
          sql.Column(
              'meterstick_bs_random_row_number',
              alias='meterstick_bs_row_number',
          )
      ),
      random_choice_table_alias,
  )
  random_rows = sql.Datasource(random_rows, 'a')
  resampled = random_rows.join(
      sql.Datasource(random_choice_table_alias, 'b'), using=using)
  table = sql.Sql(
      sql.Column('b.*', auto_alias=False),
      resampled,
      where='meterstick_bs_filter' if global_filter else None,
  )
  table = with_data.add(sql.Datasource(table, 'BootstrapRandomChoices'))
  return table, with_data


def bootstrap_by_unit(metric, table, split_by, global_filter, with_data):
  """Gets metric.n_replicates bootstrap resamples.

  The SQL is constructed as
    WITH
    Candidates AS (SELECT
      split_by,
      unit,
    FROM table
    WHERE global_filter
    GROUP BY split_by, unit),
    <bootstrap Candidates by rows and save to BootstrapRandomChoices>
    BootstrapResampledData AS (SELECT
      *
    FROM BootstrapRandomChoices
    LEFT JOIN
    table
    USING (split_by, unit)
    WHERE global_filter)

  Args:
    metric: An instance of Bootstrap.
    table: The table we want to resample.
    split_by: The columns that we use to split the data.
    global_filter: All the filters that applied to the Bootstrap.
    with_data: A global variable that contains all the WITH clauses we need.

  Returns:
    The alias of the table in the WITH clause that has all resampled data.
    The global with_data which holds all datasources we need in the WITH clause.
  """
  columns = sql.Columns(split_by).add(metric.unit)
  units = sql.Sql(columns, table, global_filter, columns)
  units_alias = with_data.add(sql.Datasource(units, 'Candidates'))
  resampled_grps, with_data = bootstrap_by_row(
      metric,
      units_alias,
      sql.Columns(split_by.aliases),
      sql.Filters(),
      with_data,
      columns.aliases,
  )

  renamed = [i for i in sql.Columns(split_by).add(metric.unit) if i != i.alias]
  if renamed:
    table = sql.Sql(
        sql.Columns(sql.Column('*', auto_alias=False)).add(renamed),
        table,
        where=global_filter)
  bs_data = sql.Sql(
      sql.Column('*', auto_alias=False),
      sql.Join(
          resampled_grps,
          table,
          join='LEFT',
          using=sql.Columns(split_by.aliases).add(metric.unit)),
      where=global_filter)
  bs_data = sql.Datasource(bs_data, 'BootstrapResampledData')
  table = with_data.add(bs_data)

  return table, with_data
