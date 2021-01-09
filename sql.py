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
"""Module to generate SQL scripts for Metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from collections import abc
import copy
import functools
import itertools
import re
from typing import Iterable, Optional, Text, Union

from meterstick import metrics
from meterstick import operations


def to_sql(table, split_by=None):
  return lambda metric: metric.to_sql(table, split_by)


def compute_on_sql(table,
                   split_by=None,
                   execute=None,
                   melted=False):
  # pylint: disable=g-long-lambda
  return lambda m: m.compute_on_sql(table,
                                    split_by,
                                    execute,
                                    melted)
  # pylint: enable=g-long-lambda


def get_sql(metric, table: Text, split_by=None):
  """The entry point function to get SQL for a Metric.

  Args:
    metric: An instance of metrics.Metric.
    table: The name of the table you want to query, like 'sql_table'.
    split_by: The columns you want to split by. It's the one you use in
      metric.compute_on(df, split_by).

  Returns:
    A Sql instance whose string representation is the SQL query that gives you
    the same result as metric.compute_on(df, split_by), where df is a DataFrame
    that has the same data as table.
  """
  global_filter = get_global_filter(metric)
  indexes = Columns(split_by).add(metrics.get_extra_idx(metric))
  with_data = Datasources()
  if not Datasource(table).is_table:
    table = with_data.add(Datasource(table, 'Data'))
  sql, with_data = get_sql_for_metric(metric, Datasource(table),
                                      Columns(split_by), global_filter, indexes,
                                      None, with_data)
  sql.with_data = with_data
  return sql


def get_sql_for_metric(metric, table, split_by, global_filter, indexes,
                       local_filter, with_data):
  """Gets the SQL query for metric.

  Args:
    metric: An instance of metrics.Metric.
    table: The table we want to query from, like 'babyname'.
    split_by: The columns that we use to split the data. Note it could be
      different to the split_by passed to the root Metric. For example, in the
      call of get_sql(AbsoluteChange('platform', 'tablet', Distribution(...)),
      'country') the split_by Distribution gets will be ('country', 'platform')
      because AbsoluteChange adds an extra index.
    global_filter: The filters that can be applied to the whole Metric tree.It
      will be passed down all the way to the leaf Metrics and become the WHERE
      clause in the query of root table.
    indexes: The columns that we shouldn't apply any arithmetic operation. For
      most of the time they are the indexes you would see in the result of
      metric.compute_on(df).
    local_filter: The filters that have been accumulated as we walk down the
      metric tree. It's the collection of all filters of the ancestor Metrics
      along the path so far. More filters might be added as we walk down to the
      leaf Metrics. It's used there as inline filters like IF(local_filter,
      value, NULL).
    with_data: A global variable that contains all the WITH clauses we need.
      It's being passed around and Metrics add the datasources they need to it.
      It's added to the SQL instance eventually in get_sql() once we have walked
      through the whole metric tree.

  Returns:
    The SQL instance for metric, without the WITH clause component.
    The global with_data which holds all datasources we need in the WITH clause.
  """
  local_filter = Filters(local_filter).add(metric.where)
  indexes = Columns(indexes).add(split_by)
  if isinstance(
      metric, (metrics.Variance, metrics.StandardDeviation)) and metric.weight:
    return get_sql_for_weighted_var_and_sd(metric, table, split_by,
                                           global_filter, local_filter,
                                           with_data)
  if isinstance(metric, metrics.SimpleMetric):
    columns = get_columns(metric, global_filter, local_filter)
    return Sql(columns, table, global_filter, split_by), with_data
  if isinstance(metric, metrics.CompositeMetric):
    return get_sql_for_composite_metric(metric, table, split_by, global_filter,
                                        indexes, local_filter, with_data)
  if isinstance(metric, metrics.MetricList):
    return get_sql_for_metriclist(metric, table, split_by, global_filter,
                                  indexes, local_filter, with_data)
  if isinstance(metric, operations.Distribution):
    return get_sql_for_distribution(metric, table, split_by, global_filter,
                                    indexes, local_filter, with_data)
  if isinstance(metric, operations.CumulativeDistribution):
    return get_sql_for_cum_dist(metric, table, split_by, global_filter, indexes,
                                local_filter, with_data)
  if isinstance(metric, operations.MH):
    return get_sql_for_mh(metric, table, split_by, global_filter, indexes,
                          local_filter, with_data)
  if isinstance(metric, (operations.PercentChange, operations.AbsoluteChange)):
    return get_sql_for_change(metric, table, split_by, global_filter, indexes,
                              local_filter, with_data)
  if isinstance(metric, (operations.Jackknife, operations.Bootstrap)):
    return get_sql_for_jackknife_or_bootstrap(metric, table, split_by,
                                              global_filter, indexes,
                                              local_filter, with_data)
  raise ValueError('SQL for %s is not supported!' % metric.name)


def get_columns(metric, global_filter=None, local_filter=None):
  """Collects the columns for all the leaf Metrics."""
  columns = Columns(get_column(metric, global_filter, local_filter))
  for m in metric.children:
    if isinstance(m, metrics.Metric):
      columns.add(
          get_columns(m, global_filter,
                      Filters(local_filter).add(metric.where)))
  return columns


def get_column(metric, global_filter=None, local_filter=None):
  """Gets the value column for metric, with filters adjusted."""
  if not isinstance(metric, metrics.SimpleMetric):
    return

  where = Filters([metric.where, local_filter]).remove(global_filter)

  if hasattr(metric, 'jackknife_fast_col'):
    # See modify_descendants_for_jackknife_fast() for context.
    return Column(metric.jackknife_fast_col, 'SUM({})', metric.name)
  if isinstance(metric, metrics.Sum):
    return Column(metric.var, 'SUM({})', metric.name, where)
  if isinstance(metric, metrics.Max):
    return Column(metric.var, 'MAX({})', metric.name, where)
  if isinstance(metric, metrics.Min):
    return Column(metric.var, 'MIN({})', metric.name, where)
  if isinstance(metric, metrics.Mean):
    if not metric.weight:
      return Column(metric.var, 'AVG({})', metric.name, where)
    else:
      res = Column('%s * %s' % (metric.weight, metric.var), 'SUM({})', 'total',
                   where)
      res /= Column(metric.weight, 'SUM({})', 'total_weight', where)
      return res.set_alias(metric.name)
  if isinstance(metric, metrics.Count):
    if metric.distinct:
      return Column(metric.var, 'COUNT(DISTINCT {})', metric.name, where)
    else:
      return Column(metric.var, 'COUNT({})', metric.name, where)
  if isinstance(metric, metrics.Quantile):
    if metric.weight:
      raise ValueError('SQL for weighted quantile is not supported!')
    if metric.one_quantile:
      alias = 'quantile(%s, %s)' % (metric.var, metric.quantile)
      return Column(
          metric.var,
          'APPROX_QUANTILES({}, 100)[OFFSET(%s)]' % int(100 * metric.quantile),
          alias, where)

    query = 'APPROX_QUANTILES({}, 100)[OFFSET(%s)]'
    quantiles = []
    for q in metric.quantile:
      alias = 'quantile(%s, %s)' % (metric.var, q)
      if alias.startswith('0.'):
        alias = 'point_' + alias[2:]
      quantiles.append(Column(metric.var, query % int(100 * q), alias, where))
    return Columns(quantiles)
  if isinstance(metric, metrics.Variance):
    if not metric.weight:
      if metric.ddof == 1:
        return Column(metric.var, 'VAR_SAMP({})', metric.name, where)
      else:
        return Column(metric.var, 'VAR_POP({})', metric.name, where)
    else:
      raise ValueError(
          "Shouldn't be here! It's handled in get_sql_for_weighted_var_and_sd.")
  if isinstance(metric, metrics.StandardDeviation):
    if not metric.weight:
      if metric.ddof == 1:
        return Column(metric.var, 'STDDEV_SAMP({})', metric.name, where)
      else:
        return Column(metric.var, 'STDDEV_POP({})', metric.name, where)
    else:
      raise ValueError(
          "Shouldn't be here! It's handled in get_sql_for_weighted_var_and_sd.")
  if isinstance(metric, metrics.CV):
    if metric.ddof == 1:
      res = Column(metric.var, 'STDDEV_SAMP({})', metric.name, where) / Column(
          metric.var, 'AVG({})', metric.name, where)
    else:
      res = Column(metric.var, 'STDDEV_POP({})', metric.name, where) / Column(
          metric.var, 'AVG({})', metric.name, where)
    return res.set_alias(metric.name)
  if isinstance(metric, metrics.Correlation):
    if not metric.weight:
      if metric.method != 'pearson':
        raise ValueError('Only Pearson correlation is supported!')
      return Column((metric.var, metric.var2), 'CORR({}, {})', metric.name,
                    where)
    else:
      raise ValueError('SQL for weighted correlation is not supported!')
  if isinstance(metric, metrics.Cov):
    if metric.weight:
      raise ValueError('SQL for weighted covariance is not supported!')
    ddof = metric.ddof
    if ddof is None:
      ddof = 0 if metric.bias else 1
    if ddof == 1:
      return Column((metric.var, metric.var2), 'COVAR_SAMP({}, {})',
                    metric.name, where)
    elif ddof == 0:
      return Column((metric.var, metric.var2), 'COVAR_POP({}, {})', metric.name,
                    where)
    else:
      raise ValueError('Only ddof being 0 or 1 is supported!')
  raise ValueError('SQL for %s is not supported!' % metric.name)


def get_sql_for_distribution(metric, table, split_by, global_filter, indexes,
                             local_filter, with_data):
  """Gets the SQL for operations.Distribution.

  The query is constructed by
  1. Get the query for the child metric.
  2. Keep all indexing/groupby columns unchanged.
  3. For all value columns, get value / SUM(value) OVER (PARTITION BY split_by).

  Args:
    metric: An instance of operations.Distribution.
    table: The table we want to query from.
    split_by: The columns that we use to split the data.
    global_filter: The filters that can be applied to the whole Metric tree.
    indexes: The columns that we shouldn't apply any arithmetic operation.
    local_filter: The filters that have been accumulated so far.
    with_data: A global variable that contains all the WITH clauses we need.

  Returns:
    The SQL instance for metric, without the WITH clause component.
    The global with_data which holds all datasources we need in the WITH clause.
  """
  if not isinstance(metric, operations.Distribution):
    raise ValueError('Not a Distribution!')

  child_sql, with_data = get_sql_for_metric(metric.children[0], table, indexes,
                                            global_filter, indexes,
                                            local_filter, with_data)
  child_table = Datasource(child_sql, 'DistributionRaw')
  child_table_alias = with_data.add(child_table)
  groupby = Columns(indexes.aliases, distinct=True)
  columns = Columns()
  for c in child_sql.columns:
    if c.alias in groupby:
      continue
    col = Column(c.alias) / Column(
        c.alias, 'SUM({})', partition=split_by.aliases)
    col.set_alias('Distribution of %s' % c.alias_raw)
    columns.add(col)
  return Sql(groupby.add(columns), child_table_alias), with_data


def get_sql_for_cum_dist(metric, table, split_by, global_filter, indexes,
                         local_filter, with_data):
  """Gets the SQL for operations.CumulativeDistribution.

  The query is constructed by
  1. Get the query for the Distribution of the child Metric.
  2. Keep all indexing/groupby columns unchanged.
  3. For all value columns, get the cumulative sum by summing over
    'ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW'.

  Args:
    metric: An instance of operations.CumulativeDistribution.
    table: The table we want to query from.
    split_by: The columns that we use to split the data.
    global_filter: The filters that can be applied to the whole Metric tree.
    indexes: The columns that we shouldn't apply any arithmetic operation.
    local_filter: The filters that have been accumulated so far.
    with_data: A global variable that contains all the WITH clauses we need.

  Returns:
    The SQL instance for metric, without the WITH clause component.
    The global with_data which holds all datasources we need in the WITH clause.
  """
  if not isinstance(metric, operations.CumulativeDistribution):
    raise ValueError('Not a CumulativeDistribution!')

  child_sql, with_data = get_sql_for_metric(
      operations.Distribution(metric.extra_index, metric.children[0]), table,
      split_by, global_filter, indexes, local_filter, with_data)
  child_table = Datasource(child_sql, 'CumulativeDistributionRaw')
  child_table_alias = with_data.add(child_table)
  columns = Columns(indexes.aliases)
  order = list(metrics.get_extra_idx(metric))
  order[0] = Column(
      get_order_for_cum_dist(Column(order[0]).alias, metric), auto_alias=False)
  for c in child_sql.columns:
    if c in columns:
      continue

    col = Column(
        c.alias,
        'SUM({})',
        partition=split_by.aliases,
        order=order,
        window_frame='ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW')
    col.set_alias('Cumulative %s' % c.alias_raw)
    columns.add(col)
  return Sql(columns, child_table_alias), with_data


def get_order_for_cum_dist(over, metric):
  if metric.order:
    over = 'CASE %s\n' % over
    tmpl = 'WHEN %s THEN %s'
    over += '\n'.join(
        tmpl % (format_to_condition(o), i) for i, o in enumerate(metric.order))
    over += '\nELSE %s\nEND' % len(metric.order)
  return over if metric.ascending else over + ' DESC'


def get_sql_for_weighted_var_and_sd(metric, table, split_by, global_filter,
                                    local_filter, with_data):
  """Gets the SQL for weighted metrics.Variance or metrics.StandardDeviation.

  For Variance the query is like
  WITH
  WeightedBase AS (SELECT
    split_by,
    weight,
    weight * POWER(var - SAFE_DIVIDE(
      SUM(weight * var) OVER (PARTITION BY split_by),
      SUM(weight) OVER (PARTITION BY split_by)), 2) AS weighted_squared_diff
  FROM table)
  SELECT
    split_by,
    SAFE_DIVIDE(SUM(weighted_squared_diff), SUM(weight) - 1)
      AS weighted_metric
  FROM WeightedBase
  GROUP BY split_by.

  For StandardDeviation we take the square root of weighted_metric.

  Args:
    metric: An instance of metrics.Variance or metrics.StandardDeviation, with
      weight.
    table: The table we want to query from.
    split_by: The columns that we use to split the data.
    global_filter: The filters that can be applied to the whole Metric tree.
    local_filter: The filters that have been accumulated so far.
    with_data: A global variable that contains all the WITH clauses we need.

  Returns:
    The SQL instance for metric, without the WITH clause component.
    The global with_data which holds all datasources we need in the WITH clause.
  """
  if not (isinstance(metric, (metrics.Variance, metrics.StandardDeviation)) and
          metric.weight):
    raise ValueError('Not a Metric with weight!')

  where = Filters([metric.where, local_filter]).remove(global_filter)
  weight = metric.weight
  var = metric.var
  columns = Columns(split_by).add(Column(weight, alias=weight, filters=where))
  total_sum = Column(
      '%s * %s' % (weight, var), 'SUM({})', filters=where, partition=split_by)
  total_weight = Column(weight, 'SUM({})', filters=where, partition=split_by)
  weighted_mean = total_sum / total_weight
  weighted_squared_diff = Column(
      '%s * POWER(%s - %s, 2)' % (weight, var, weighted_mean.expression),
      alias='weighted_squared_diff',
      filters=where)
  weighted_base_table = Sql(
      columns.add(weighted_squared_diff), table, global_filter)
  weighted_base_table_alias = with_data.add(
      Datasource(weighted_base_table, 'WeightedBase'))

  weighted_var = Column('weighted_squared_diff', 'SUM({})') / Column(
      Column(weight).alias, 'SUM({}) - 1')
  if isinstance(metric, metrics.StandardDeviation):
    weighted_var = weighted_var**0.5
  weighted_var.set_alias(metric.name)
  return Sql(
      weighted_var, weighted_base_table_alias,
      groupby=split_by.aliases), with_data


def get_sql_for_composite_metric(metric, table, split_by, global_filter,
                                 indexes, local_filter, with_data):
  """Gets the SQL for metrics.CompositeMetric.

  A CompositeMetric has two children and at least one of them is a Metric. The
  query is constructed as
  1. Get the queries for the two children.
  2. If one child is not a Metric, which means it's a constant number, then we
    apply the computation to the columns in the other child's SQL.
  3. If both are Metrics and their SQLs are compatible, we zip their columns
    and apply the computation on the column pairs.
  4. If both are Metrics and their SQLs are incompatible, we put children SQLs
    to with_data. Then apply the computation on column pairs and SELECT them
    from the JOIN of the two children SQLs.

  Args:
    metric: An instance of metrics.CompositeMetric.
    table: The table we want to query from.
    split_by: The columns that we use to split the data.
    global_filter: The filters that can be applied to the whole Metric tree.
    indexes: The columns that we shouldn't apply any arithmetic operation.
    local_filter: The filters that have been accumulated so far.
    with_data: A global variable that contains all the WITH clauses we need.

  Returns:
    The SQL instance for metric, without the WITH clause component.
    The global with_data which holds all datasources we need in the WITH clause.
  """
  if not isinstance(metric, metrics.CompositeMetric):
    raise ValueError('Not a CompositeMetric!')

  op = metric.op

  if not isinstance(metric.children[0], metrics.Metric):
    constant = metric.children[0]
    sql, with_data = get_sql_for_metric(metric.children[1], table, split_by,
                                        global_filter, indexes, local_filter,
                                        with_data)
    sql.columns = Columns(
        (c if c in indexes else op(constant, c) for c in sql.columns))
  elif not isinstance(metric.children[1], metrics.Metric):
    constant = metric.children[1]
    sql, with_data = get_sql_for_metric(metric.children[0], table, split_by,
                                        global_filter, indexes, local_filter,
                                        with_data)
    sql.columns = Columns(
        (c if c in indexes else op(c, constant) for c in sql.columns))
  else:
    sql0, with_data = get_sql_for_metric(metric.children[0], table, split_by,
                                         global_filter, indexes, local_filter,
                                         with_data)
    sql1, with_data = get_sql_for_metric(metric.children[1], table, split_by,
                                         global_filter, indexes, local_filter,
                                         with_data)
    if len(sql0.columns) != 1 and len(sql1.columns) != 1 and len(
        sql0.columns) != len(sql1.columns):
      raise ValueError('Children Metrics have different shapes!')

    compatible, larger_from = is_compatible(sql0, sql1)
    if compatible:
      col0_col1 = zip(itertools.cycle(sql0.columns), sql1.columns)
      if len(sql1.columns) == 1:
        col0_col1 = zip(sql0.columns, itertools.cycle(sql1.columns))
      columns = Columns()
      for c0, c1 in col0_col1:
        if c0 in indexes.aliases:
          columns.add(c0)
        else:
          alias = metric.name_tmpl.format(c0.alias_raw, c1.alias_raw)
          columns.add(op(c0, c1).set_alias(alias))
      sql = sql0
      sql.columns = columns
      sql.from_data = larger_from
    else:
      tbl0 = with_data.add(Datasource(sql0, 'CompositeMetricTable0'))
      tbl1 = with_data.add(Datasource(sql1, 'CompositeMetricTable1'))
      join = 'FULL' if indexes else 'CROSS'
      from_data = Join(tbl0, tbl1, join=join, using=indexes)
      columns = Columns()
      col0_col1 = zip(itertools.cycle(sql0.columns), sql1.columns)
      if len(sql1.columns) == 1:
        col0_col1 = zip(sql0.columns, itertools.cycle(sql1.columns))
      for c0, c1 in col0_col1:
        if c0 not in indexes.aliases:
          col = op(
              Column('%s.%s' % (tbl0, c0.alias), alias=c0.alias_raw),
              Column('%s.%s' % (tbl1, c1.alias), alias=c1.alias_raw))
          columns.add(col)
      sql = Sql(Columns(indexes.aliases).add(columns), from_data)

  if not isinstance(
      metric.children[0], operations.Operation) and not isinstance(
          metric.children[1], operations.Operation) and len(sql.columns) == 1:
    sql.columns[0].set_alias(metric.name)

  if metric.columns:
    columns = sql.columns.difference(indexes)
    if len(metric.columns) != len(columns):
      raise ValueError('The length of the renaming columns is wrong!')
    for col, rename in zip(columns, metric.columns):
      col.set_alias(rename)  # Modify in-place.

  return sql, with_data


def get_sql_for_metriclist(metric, table, split_by, global_filter, indexes,
                           local_filter, with_data):
  """Gets the SQL for metrics.MetricList.

  The query is constructed by
  1. Get the query for every children metric.
  2. If all children queries are compatible, we just collect all the columns
    from the children and use the WHERE and GROUP BY clauses from any chldren.
    The FROM clause is more complex. We use the largest FROM clause in children.
    See the doc of is_compatible() for its meaning.
    If any pair of children queries are incompatible, we merge the compatible
    children as much as possible then add the merged SQLs to with_data, join
    them on indexes, and SELECT *.

  Args:
    metric: An instance of metrics.MetricList.
    table: The table we want to query from.
    split_by: The columns that we use to split the data.
    global_filter: The filters that can be applied to the whole Metric tree.
    indexes: The columns that we shouldn't apply any arithmetic operation.
    local_filter: The filters that have been accumulated so far.
    with_data: A global variable that contains all the WITH clauses we need.

  Returns:
    The SQL instance for metric, without the WITH clause component.
    The global with_data which holds all datasources we need in the WITH clause.
  """
  if not isinstance(metric, metrics.MetricList):
    raise ValueError('Not a MetricList!')

  children_sql = [
      get_sql_for_metric(c, table, split_by, global_filter, indexes,
                         local_filter, with_data)[0] for c in metric.children
  ]
  incompatible_sqls = []
  # It's O(n^2). We can do better but I don't expect the metric tree to be big.
  for child_sql in children_sql:
    found = False
    for target in incompatible_sqls:
      can_merge, larger_from = is_compatible(child_sql, target)
      if can_merge:
        target.add('columns', child_sql.columns)
        target.from_data = larger_from
        found = True
        break
    if not found:
      incompatible_sqls.append(child_sql)

  if len(incompatible_sqls) == 1:
    return incompatible_sqls[0], with_data

  columns = Columns(indexes.aliases)
  for i, table in enumerate(incompatible_sqls):
    data = Datasource(table, 'MetricListChildTable')
    alias = with_data.add(data)
    for c in table.columns:
      if c not in columns:
        columns.add(Column('%s.%s' % (alias, c.alias), alias=c.alias_raw))
    if i == 0:
      from_data = Datasource(alias)
    else:
      join = 'FULL' if indexes else 'CROSS'
      from_data = from_data.join(Datasource(alias), join=join, using=indexes)
  return Sql(columns, from_data), with_data


def get_sql_for_mh(metric, table, split_by, global_filter, indexes,
                   local_filter, with_data):
  """Gets the SQL for operations.MH.

  The query is constructed in a similar way to AbsoluteChange except that we
  apply weights to adjust the change.

  For example, the query for
  operations.MH('condition', 'base_value', 'stratified',
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
      SUM(SAFE_DIVIDE(MHRaw.`sum(click)` * MHBase.`sum(impression)`,
          MHBase.`sum(impression)` + MHRaw.`sum(impression)`)),
      SUM(SAFE_DIVIDE(MHBase.`sum(click)` * MHRaw.`sum(impression)`,
          MHBase.`sum(impression)` + MHRaw.`sum(impression)`))) - 100
      AS `ctr MH Ratio`
  FROM MHRaw
  JOIN
  MHBase
  USING (split_by, stratified)
  WHERE
  condition != "base_value"
  GROUP BY split_by, condition

  Args:
    metric: An instance of operations.PercentChange or
      operations.AbsoluteChange.
    table: The table we want to query from.
    split_by: The columns that we use to split the data.
    global_filter: The filters that can be applied to the whole Metric tree.
    indexes: The columns that we shouldn't apply any arithmetic operation.
    local_filter: The filters that have been accumulated so far.
    with_data: A global variable that contains all the WITH clauses we need.

  Returns:
    The SQL instance for metric, without the WITH clause component.
    The global with_data which holds all datasources we need in the WITH clause.
  """
  if not isinstance(metric, operations.MH):
    raise ValueError('Not a MH!')
  metric.check_is_ratio()

  child = metric.children[0]
  grandchildren = []
  if isinstance(child, metrics.MetricList):
    for m in child:
      grandchildren += list(m.children)
  else:
    grandchildren = child.children

  cond_cols = Columns(metric.extra_index)
  groupby = Columns(split_by).add(cond_cols).add(metric.stratified_by)
  alias_tmpl = metric.name_tmpl
  raw_table_sql, with_data = get_sql_for_metric(
      metrics.MetricList(grandchildren), table, groupby, global_filter, indexes,
      local_filter, with_data)

  raw_table = Datasource(raw_table_sql, 'MHRaw')
  raw_table_alias = with_data.add(raw_table)

  base = metric.baseline_key if isinstance(metric.baseline_key,
                                           tuple) else [metric.baseline_key]
  base_cond = ('%s = %s' % (c, format_to_condition(b))
               for c, b in zip(cond_cols.aliases, base))
  base_cond = ' AND '.join(base_cond)
  base_value = Sql(
      Columns(raw_table_sql.groupby.aliases).add(
          raw_table_sql.columns.aliases).difference(cond_cols.aliases),
      raw_table_alias, base_cond)
  base_table = Datasource(base_value, 'MHBase')
  base_table_alias = with_data.add(base_table)

  exclude_base_condition = ('%s != %s' % (c, format_to_condition(b))
                            for c, b in zip(cond_cols.aliases, base))
  exclude_base_condition = ' OR '.join(exclude_base_condition)
  cond = None if metric.include_base else Filters([exclude_base_condition])
  col_tmpl = """100 * SAFE_DIVIDE(
    SUM(SAFE_DIVIDE(
      {raw}.%(numer)s * {base}.%(denom)s,
      {base}.%(denom)s + {raw}.%(denom)s)),
    SUM(SAFE_DIVIDE(
      {base}.%(numer)s * {raw}.%(denom)s,
      {base}.%(denom)s + {raw}.%(denom)s))) - 100"""
  col_tmpl = col_tmpl.format(raw=raw_table_alias, base=base_table_alias)
  columns = Columns()
  if isinstance(child, metrics.MetricList):
    for c in child:
      columns.add(
          Column(
              col_tmpl % {
                  'numer': Column(c.children[0].name).alias,
                  'denom': Column(c.children[1].name).alias
              },
              alias=alias_tmpl.format(c.name)))
  else:
    columns = Column(
        col_tmpl % {
            'numer': Column(child.children[0].name).alias,
            'denom': Column(child.children[1].name).alias
        },
        alias=alias_tmpl.format(child.name))

  using = indexes.difference(cond_cols).add(metric.stratified_by)
  return Sql(columns, Join(raw_table_alias, base_table_alias, using=using),
             cond, indexes.aliases), with_data


def get_sql_for_change(metric, table, split_by, global_filter, indexes,
                       local_filter, with_data):
  """Gets the SQL for operations.PercentChange or operations.AbsoluteChange.

  The query is constructed by
  1. Get the query for the child metric and add it to with_data, we call it
    raw_value_table.
  2. Query the rows that only has the base value from raw_value_table, add it to
    with_data too. We call it base_value_table.
  3. Join the two tables and computes the change for all value columns.

  For example, the query for
  operations.AbsoluteChange('condition', 'base_value', metrics.Mean('click'))
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
    metric: An instance of operations.PercentChange or
      operations.AbsoluteChange.
    table: The table we want to query from.
    split_by: The columns that we use to split the data.
    global_filter: The filters that can be applied to the whole Metric tree.
    indexes: The columns that we shouldn't apply any arithmetic operation.
    local_filter: The filters that have been accumulated so far.
    with_data: A global variable that contains all the WITH clauses we need.

  Returns:
    The SQL instance for metric, without the WITH clause component.
    The global with_data which holds all datasources we need in the WITH clause.
  """
  if not isinstance(metric,
                    (operations.PercentChange, operations.AbsoluteChange)):
    raise ValueError('Not a PercentChange nor AbsoluteChange!')

  child = metric.children[0]
  cond_cols = Columns(metric.extra_index)
  groupby = Columns(split_by).add(cond_cols)
  alias_tmpl = metric.name_tmpl
  raw_table_sql, with_data = get_sql_for_metric(child, table, groupby,
                                                global_filter, indexes,
                                                local_filter, with_data)
  raw_table = Datasource(raw_table_sql, 'ChangeRaw')
  raw_table_alias = with_data.add(raw_table)

  base = metric.baseline_key if isinstance(metric.baseline_key,
                                           tuple) else [metric.baseline_key]
  base_cond = ('%s = %s' % (c, format_to_condition(b))
               for c, b in zip(cond_cols.aliases, base))
  base_cond = ' AND '.join(base_cond)
  base_value = Sql(
      Columns(raw_table_sql.groupby.aliases).add(
          raw_table_sql.columns.aliases).difference(cond_cols.aliases),
      raw_table_alias, base_cond)
  base_table = Datasource(base_value, 'ChangeBase')
  base_table_alias = with_data.add(base_table)

  exclude_base_condition = ('%s != %s' % (c, format_to_condition(b))
                            for c, b in zip(cond_cols.aliases, base))
  exclude_base_condition = ' OR '.join(exclude_base_condition)
  cond = None if metric.include_base else Filters([exclude_base_condition])
  col_tmp = '%s.{c} - %s.{c}' if isinstance(
      metric, operations.AbsoluteChange
  ) else 'SAFE_DIVIDE(%s.{c}, (%s.{c})) * 100 - 100'
  columns = Columns()
  for c in raw_table_sql.columns.difference(indexes.aliases):
    col = Column(
        col_tmp.format(c=c.alias) % (raw_table_alias, base_table_alias),
        alias=alias_tmpl.format(c.alias_raw))
    columns.add(col)
  using = indexes.difference(cond_cols)
  join = '' if using else 'CROSS'
  return Sql(
      Columns(indexes.aliases).add(columns),
      Join(raw_table_alias, base_table_alias, join=join, using=using),
      cond), with_data


def get_sql_for_jackknife_or_bootstrap(metric, table, split_by, global_filter,
                                       indexes, local_filter, with_data):
  """Gets the SQL for operations.Jackknife or operations.Bootstrap.

  The query is constructed by
  1. Resample the table.
  2. Compute the child Metric on the resampled data.
  3. Compute the standard error from #2.
  4. Compute the point estimate from original table.
  5. Join #3 and #4.
  6. If metric has confidence level specified, we also get the degrees of
    freedom so we can later compute the critical value of t distribution in
    Python.
  7. If metric only has one child and it's PercentChange or AbsoluteChange, we
    also get the base values for comparison. They will be used in the
    res.display().

  Args:
    metric: An instance of operations.Jackknife or operations.Bootstrap.
    table: The table we want to query from.
    split_by: The columns that we use to split the data.
    global_filter: The filters that can be applied to the whole Metric tree.
    indexes: The columns that we shouldn't apply any arithmetic operation.
    local_filter: The filters that have been accumulated so far.
    with_data: A global variable that contains all the WITH clauses we need.

  Returns:
    The SQL instance for metric, without the WITH clause component.
    The global with_data which holds all datasources we need in the WITH clause.
  """
  if not isinstance(metric, (operations.Jackknife, operations.Bootstrap)):
    raise ValueError('Not a Jackknife or Bootstrap!')

  name = 'Jackknife' if isinstance(metric,
                                   operations.Jackknife) else 'Bootstrap'
  se, with_data = get_se(metric, table, split_by, global_filter, indexes,
                         local_filter, with_data)
  se_alias = with_data.add(Datasource(se, name + 'SE'))

  pt_est, with_data = get_sql_for_metric(metric.children[0], table,
                                         split_by, global_filter, indexes,
                                         local_filter, with_data)
  pt_est_alias = with_data.add(Datasource(pt_est, name + 'PointEstimate'))

  columns = Columns()
  using = Columns(se.groupby)
  for c in pt_est.columns:
    if c in indexes.aliases:
      using.add(c)
    else:
      pt_est_col = Column('%s.%s' % (pt_est_alias, c.alias), alias=c.alias_raw)
      alias = '%s %s SE' % (c.alias_raw, name)
      se_col = Column('%s.%s' % (se_alias, escape_alias(alias)), alias=alias)
      columns.add(pt_est_col)
      columns.add(se_col)
      if metric.confidence:
        dof = '%s dof' % c.alias_raw
        columns.add(Column('%s.%s' % (se_alias, escape_alias(dof)), alias=dof))

  has_base_vals = False
  if metric.confidence:
    child = metric.children[0]
    if len(metric.children) == 1 and isinstance(
        child, (operations.PercentChange, operations.AbsoluteChange)):
      has_base_vals = True
      base, with_data = get_sql_for_metric(
          child.children[0], table,
          Columns(split_by).add(child.extra_index), global_filter, indexes,
          local_filter, with_data)
      base_alias = with_data.add(Datasource(base, '_ShouldAlreadyExists'))
      columns.add(
          Column('%s.%s' % (base_alias, c.alias), alias=c.alias)
          for c in base.columns.difference(indexes))

  join = 'LEFT' if using else 'CROSS'
  from_data = Join(pt_est_alias, se_alias, join=join, using=using)
  if has_base_vals:
    from_data = from_data.join(base_alias, join=join, using=using)
  return Sql(using.add(columns), from_data), with_data


def get_se(metric, table, split_by, global_filter, indexes, local_filter,
           with_data):
  """Gets the SQL query that computes the standard error and dof if needed."""
  global_filter = Filters([global_filter, local_filter]).add(metric.where)
  local_filter = Filters()
  metric, table, split_by, global_filter, indexes, with_data = preaggregate_if_possible(
      metric, table, split_by, global_filter, indexes, with_data)

  if isinstance(metric, operations.Jackknife):
    if metric.can_precompute:
      metric = copy.deepcopy(metric)  # We'll modify the metric tree in-place.
    table, with_data = get_jackknife_data(metric, table, split_by,
                                          global_filter, indexes, local_filter,
                                          with_data)
  else:
    table, with_data = get_bootstrap_data(
        metric, table, split_by, global_filter, local_filter, with_data)

  if isinstance(metric, operations.Jackknife) and metric.can_precompute:
    split_by = adjust_indexes_for_jk_fast(split_by)
    indexes = adjust_indexes_for_jk_fast(indexes)
    # global_filter has been removed from all Metrics when precomputeing LOO.
    global_filter = Filters(None)

  samples, with_data = get_sql_for_metric(
      metric.children[0], table,
      Columns(split_by).add('_resample_idx'), global_filter, Columns(indexes),
      local_filter, with_data)
  samples_alias = with_data.add(Datasource(samples, 'ResampledResults'))

  columns = Columns()
  groupby = Columns((c.alias for c in samples.groupby if c != '_resample_idx'))
  for c in samples.columns:
    if c == '_resample_idx':
      continue
    elif c in indexes.aliases:
      groupby.add(c.alias)
    else:
      se = Column(c.alias, 'STDDEV_SAMP({})', '%s Bootstrap SE' % c.alias_raw)
      if isinstance(metric, operations.Jackknife):
        adjustment = Column(
            'SAFE_DIVIDE((COUNT({c}) - 1), SQRT(COUNT({c})))'.format(c=c.alias))
        se = (se * adjustment).set_alias('%s Jackknife SE' % c.alias_raw)
      columns.add(se)
      if metric.confidence:
        columns.add(Column(c.alias, 'COUNT({}) - 1', '%s dof' % c.alias_raw))
  return Sql(
      columns, samples_alias, groupby=groupby), with_data


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
    metric: An instance of operations.Jackknife or operations.Bootstrap.
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
  if not isinstance(metric, operations.Jackknife) and not (isinstance(
      metric, operations.Bootstrap) and metric.unit):
    return metric, table, split_by, global_filter, indexes, with_data

  all_split_by = Columns([indexes, metric.unit])
  sums = Columns()
  for m in metric.traverse():
    if Filters(m.where).remove(global_filter):
      return metric, table, split_by, global_filter, indexes, with_data
    if isinstance(m, metrics.SimpleMetric):
      if not isinstance(m, metrics.Sum):
        return metric, table, split_by, global_filter, indexes, with_data
      else:
        sums.add(Column(m.var, 'SUM({})', Column('', alias=m.var).alias))
    if isinstance(m, operations.MH):
      all_split_by.add(m.stratified_by)

  metric = copy.deepcopy(metric)
  metric.unit = Column(metric.unit).alias
  for m in metric.traverse():
    m.where = None
    if isinstance(m, operations.Operation):
      m.extra_index = [Column(i, alias=i).alias for i in m.extra_index]
      if isinstance(m, operations.MH):
        m.stratified_by = Column(m.stratified_by).alias
    if isinstance(m, metrics.Sum):
      m.var = Column('', alias=m.var).alias

  preagg = Sql(sums, table, global_filter, all_split_by)
  preagg_alias = with_data.add(Datasource(preagg, 'Preaggregated'))

  return metric, preagg_alias, Columns(split_by.aliases), Filters(), Columns(
      indexes.aliases), with_data


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
      ind[i] = Column(s.alias)
  return Columns(ind)


def get_jackknife_data(metric, table, split_by, global_filter, indexes,
                       local_filter, with_data):
  table = Datasource(table)
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
    metric: An instance of operations.Jackknife.
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
  unique_units = Columns(
      (Column('ARRAY_AGG(DISTINCT %s)' % unit, alias='_jk_buckets')))
  where = Filters(global_filter).add(local_filter)
  if split_by:
    groupby = Columns(
        (Column(c.expression, alias='_jk_%s' % c.alias) for c in split_by))
    buckets = Sql(unique_units, table, where, groupby)
    buckets_alias = with_data.add(Datasource(buckets, 'Buckets'))
    jk_from = Join(buckets_alias,
                   Datasource('UNNEST(_jk_buckets)', '_resample_idx'))
    on = Filters(('%s.%s = %s' % (buckets_alias, c.alias, s.expression)
                  for c, s in zip(groupby, split_by)))
    on.add('_resample_idx != %s' % unit)
    jk_from = jk_from.join(table, on=on)
    exclude = groupby.as_groupby() + ', _jk_buckets'
    jk_data_table = Sql(
        Columns(Column('* EXCEPT (%s)' % exclude, auto_alias=False)),
        jk_from,
        where=where)
    jk_data_table = Datasource(jk_data_table, 'JackknifeResammpledData')
    jk_data_table_alias = with_data.add(jk_data_table)
  else:
    buckets = Sql(unique_units, table, where=where)
    buckets_alias = with_data.add(Datasource(buckets, 'Buckets'))

    jk_from = Join(buckets_alias,
                   Datasource('UNNEST(_jk_buckets)', '_resample_idx'))
    jk_from = jk_from.join(table, join='CROSS')
    jk_data_table = Sql(
        Column('* EXCEPT (_jk_buckets)', auto_alias=False),
        jk_from,
        where=Filters('_resample_idx != %s' % unit).add(where))
    jk_data_table = Datasource(jk_data_table, 'JackknifeResammpledData')
    jk_data_table_alias = with_data.add(jk_data_table)

  return jk_data_table_alias, with_data


def get_jackknife_data_fast(metric, table, split_by, global_filter, indexes,
                            local_filter, with_data):
  """Gets jackknife samples in a fast way for precomputable Jackknife.

  If all the leaf Metrics are Sum, Count or Mean, we can compute the
  leave-one-out (LOO) estimates faster. There are two situations.
  1. If Jackknife doesn't operate on any other Operation. Then the SQL is like
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

  2. If Jackknife operates on any Operation, then that Operation might add extra
  indexes and we have to adjust the slices. See the doc for
  utils.adjust_slices_for_loo() for more discussions. The SQL will be like
  WITH
  AllSlices AS (SELECT
    unrenamed_split_by,
    $RenamedSplitByIfAny AS renamed_split_by,
    ARRAY_AGG(DISTINCT extra_index) AS extra_index,
    ARRAY_AGG(DISTINCT cookie) AS cookie
  FROM table
  WHERE filter
  GROUP BY split_by),
  LOO AS (SELECT DISTINCT
    split_by,
    unit AS _resample_idx,
    SUM(click) OVER (PARTITION BY split_by, platform) -
      COALESCE(SUM(click) OVER (PARTITION BY split_by, platform, unit), 0)
    AS `sum(click)`,
    COUNT(click) OVER (PARTITION BY split_by, platform) -
      COUNT(click) OVER (PARTITION BY split_by, platform, unit)
    AS `count(click)`
  FROM AllSlices
  JOIN
  UNNEST (extra_index) AS extra_index
  JOIN
  UNNEST (unit) AS unit
  LEFT JOIN
  (SELECT *, renamed_index FROM table WHERE filter)
  USING (split_by, extra_index, unit))

  If there is no filter and no index column needs to be renamed, the
  (SELECT *, renamed_index FROM table WHERE filter) will become just 'table'.
  Also note that we have a COALESCE around SUM. We don't need it in case #1
  because we don't need to adjust for slices there. If the sum of a unit is NULL
  than it shouldn't be in the LOO. We also don't need it for COUNT because the
  COUNT(NULL) is just 0.

  Args:
    metric: An instance of operations.Jackknife.
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
  all_indexes = Columns(indexes)
  need_slice_filling = (split_by != all_indexes)
  for m in metric.traverse():
    if isinstance(m, operations.MH):
      all_indexes.add(m.stratified_by)
  indexes_and_unit = Columns(all_indexes).add(metric.unit)
  where = Filters(global_filter).add(local_filter)
  if need_slice_filling:
    extra_idx = indexes_and_unit.difference(split_by)
    unique_idx = Columns(
        (Column('ARRAY_AGG(DISTINCT %s)' % c.expression, alias=c.alias)
         for c in extra_idx))
    all_slices_table = Sql(unique_idx, table, where, split_by)
    all_slices_alias = with_data.add(Datasource(all_slices_table, 'AllSlices'))

    loo_from = Datasource(all_slices_alias)
    for c in extra_idx:
      loo_from = loo_from.join('UNNEST ({c}) AS {c}'.format(c=c.alias))
    renamed = [c for c in indexes_and_unit if c != c.alias]
    if where or renamed:
      table = Sql(
          Columns(Column('*', auto_alias=False)).add(renamed),
          table,
          where=where)
    table = loo_from.join(table, join='LEFT', using=indexes_and_unit)
    where = None

  columns = Columns()
  if need_slice_filling:
    all_indexes = all_indexes.aliases
    indexes_and_unit = indexes_and_unit.aliases
  # columns is filled in-place in modify_descendants_for_jackknife_fast.
  modified_jk = modify_descendants_for_jackknife_fast(
      metric, columns, need_slice_filling,
      Filters(global_filter).add(local_filter), Filters(), all_indexes,
      indexes_and_unit)
  metric.children = modified_jk.children

  bucket = Column(metric.unit).alias if need_slice_filling else metric.unit
  bucket = Column(bucket, alias='_resample_idx')
  columns = Columns(all_indexes).add(bucket).add(columns)
  columns.distinct = True
  loo_table = with_data.add(Datasource(Sql(columns, table, where=where), 'LOO'))
  return loo_table, with_data


def modify_descendants_for_jackknife_fast(metric, columns, need_slice_filling,
                                          global_filter, local_filter,
                                          all_indexes, indexes_and_unit):
  """Gets the columns for leaf Metrics and modify them for fast Jackknife SQL.

  See the doc of get_jackknife_data_fast() first. Here we
  1. collects the LOO columns for all Sum, Count and Mean Metrics.
  2. Modify them in-place so when we generate SQL later, they know what column
    to use. For example, Sum('X') would generate a SQL column 'SUM(X) AS sum_x',
    but as we precompute LOO estimates, we need to query from a table that has
    "SUM(X) OVER (PARTITION BY split_by) -
      SUM(X) OVER (PARTITION BY split_by, unit) AS `sum(X)`". So the expression
    of Sum('X') should now become 'SUM(`sum(X)`) AS `sum(X)`'. We add the new
    column we need to query from as an attribute "jackknife_fast_col" to the
    metric so get_column() could handle it correctly.
  3. Removes filters that have already been applied in the LOO table from leaf
    Metrics. Note that we made a copy in get_se for metric so the removal won't
    affect the metric used in point estimate computation.
  4. For Operations, their extra_index columns appear in indexes. If any of them
    has forbidden character in the name, it will be renamed in LOO so we have to
    change extra_index. For example, Distribution('$Foo') will generate a column
    $Foo AS macro_Foo in LOO so we need to replace '$Foo' with 'macro_Foo'.

  We need to make a copy for the Metric or in
  sumx = metrics.Sum('X')
  m1 = metrics.MetricList(sumx, where='X>1')
  m2 = metrics.MetricList(sumx, where='X>10')
  jk = operations.Jackknife(metrics.MetricList((m1, m2)))
  sumx will be modified twice and the second one will overwrite the first one.

  Args:
    metric: An instance of metrics.Metric or a child of one.
    columns: A global container for all columns we need in LOO table. It's being
      added in-place.
    need_slice_filling: If we need to adjust for the slices. See the doc of
      get_jackknife_data_fast().
    global_filter: The filters that can be applied to the whole Metric tree.
    local_filter: The filters that have been accumulated so far.
    all_indexes: All columns that we need to used as the group by columns in the
      LOO table. It includes split_by, all extra_idx added by Operations, and
      the stratified_by column of operations.MH, if exists.
    indexes_and_unit: all_indexes plus the unit of Jackknife.

  Returns:
    The modified metric tree.
  """
  if not isinstance(metric, metrics.Metric):
    return metric

  metric = copy.deepcopy(metric)
  local_filter = Filters(local_filter).add(metric.where)
  if isinstance(metric, (metrics.Sum, metrics.Count, metrics.Mean)):
    filters = Filters(local_filter).remove(global_filter)
    metric.where = filters
    if isinstance(metric, (metrics.Sum, metrics.Count)):
      c = metric.var
      op = 'COUNT({})' if isinstance(metric, metrics.Count) else 'SUM({})'
      total = Column(c, op, filters=filters, partition=all_indexes)
      unit_sum = Column(c, op, filters=filters, partition=indexes_and_unit)
      if need_slice_filling and isinstance(metric, metrics.Sum):
        unit_sum = Column(
            'COALESCE(%s, 0)' % unit_sum.expression, auto_alias=False)
      loo = (total - unit_sum).set_alias(metric.name)
      columns.add(loo)
      metric.jackknife_fast_col = loo.alias
    elif isinstance(metric, metrics.Mean):
      if metric.weight:
        op = 'SUM({} * {})'
        total_sum = Column((metric.var, metric.weight),
                           op,
                           filters=filters,
                           partition=all_indexes)
        unit_sum = Column((metric.var, metric.weight),
                          op,
                          filters=filters,
                          partition=indexes_and_unit)
        total_weight = Column(
            metric.weight, 'SUM({})', filters=filters, partition=all_indexes)
        unit_weight = Column(
            metric.weight,
            'SUM({})',
            filters=filters,
            partition=indexes_and_unit)
        if need_slice_filling:
          unit_sum = Column(
              'COALESCE(%s, 0)' % unit_sum.expression, auto_alias=False)
          unit_weight = Column(
              'COALESCE(%s, 0)' % unit_weight.expression, auto_alias=False)
      else:
        total_sum = Column(
            metric.var, 'SUM({})', filters=filters, partition=all_indexes)
        unit_sum = Column(
            metric.var, 'SUM({})', filters=filters, partition=indexes_and_unit)
        if need_slice_filling:
          unit_sum = Column(
              'COALESCE(%s, 0)' % unit_sum.expression, auto_alias=False)
        total_weight = Column(
            metric.var, 'COUNT({})', filters=filters, partition=all_indexes)
        unit_weight = Column(
            metric.var,
            'COUNT({})',
            filters=filters,
            partition=indexes_and_unit)

      loo = (total_sum - unit_sum) / (total_weight - unit_weight)
      loo.set_alias(metric.name)
      columns.add(loo)
      metric.jackknife_fast_col = loo.alias
    return metric

  if isinstance(metric, operations.Operation):
    metric.extra_index = Columns(metric.extra_index).aliases
    if isinstance(metric, operations.MH):
      metric.stratified_by = Column(metric.stratified_by).alias

  new_children = []
  for m in metric.children:
    modified = modify_descendants_for_jackknife_fast(m, columns,
                                                     need_slice_filling,
                                                     global_filter,
                                                     local_filter, all_indexes,
                                                     indexes_and_unit)
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
    metric: An instance of operations.Bootstrap.
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
  table = Datasource(table)
  if not table.is_table:
    table.alias = table.alias or 'RawData'
    table = with_data.add(table)
  replicates = Datasource('UNNEST(GENERATE_ARRAY(1, %s))' % metric.n_replicates,
                          '_resample_idx')
  where = Filters(global_filter).add(local_filter)
  if metric.unit is None:
    columns = Columns(Column('*', auto_alias=False))
    partition = split_by.expressions + ['_resample_idx']
    if where:
      columns.add(Column(str(where), alias='_bs_filter'))
      partition.append(str(where))
    row_number = Column(
        'ROW_NUMBER()', alias='_bs_row_number', partition=partition)
    length = Column('COUNT(*)', partition=partition)
    random_row_number = Column('RAND()') * length
    random_row_number = Column(
        'CEILING(%s)' % random_row_number.expression,
        alias='_bs_random_row_number')
    columns.add((row_number, random_row_number))
    columns.add((i for i in split_by if i != i.alias))
    random_choice_table = Sql(columns, Join(table, replicates))
    random_choice_table_alias = with_data.add(
        Datasource(random_choice_table, 'BootstrapRandomRows'))

    using = Columns(partition).add('_bs_row_number').difference(str(where))
    excludes = ['_bs_row_number']
    if where:
      excludes.append('_bs_filter')
      using.add('_bs_filter')
    random_rows = Sql(
        Columns(using).difference('_bs_row_number').add(
            Column('_bs_random_row_number', alias='_bs_row_number')),
        random_choice_table_alias)
    random_rows = Datasource(random_rows, 'a')
    resampled = random_rows.join(
        Datasource(random_choice_table_alias, 'b'), using=using)
    table = Sql(
        Column('b.* EXCEPT (%s)' % ', '.join(excludes), auto_alias=False),
        resampled,
        where='_bs_filter' if where else None)
    table = with_data.add(Datasource(table, 'BootstrapRandomChoices'))
  else:
    unit = metric.unit
    unit_alias = Column(unit).alias
    columns = (Column('ARRAY_AGG(DISTINCT %s)' % unit, alias=unit),
               Column('COUNT(DISTINCT %s)' % unit, alias='_bs_length'))
    units = Sql(columns, table, where, split_by)
    units_alias = with_data.add(Datasource(units, 'Candidates'))
    rand_samples = Column(
        '%s[ORDINAL(CAST(CEILING(RAND() * _bs_length) AS INT64))]' % unit_alias,
        alias=unit_alias)

    sample_table = Sql(
        Columns(split_by.aliases).add('_resample_idx').add(rand_samples),
        Join(units_alias,
             Datasource('UNNEST(%s)' % unit_alias)).join(replicates))
    sample_table_alias = with_data.add(
        Datasource(sample_table, 'BootstrapRandomChoices'))

    table = original_table
    renamed = [i for i in Columns(split_by).add(unit) if i != i.alias]
    if renamed or unit != unit_alias:
      table = Sql(
          Columns(Column('*', auto_alias=False)).add(renamed),
          table,
          where=where)
    bs_data = Sql(
        Column('*', auto_alias=False),
        Join(
            sample_table_alias,
            table,
            join='LEFT',
            using=Columns(split_by.aliases).add(unit)),
        where=where)
    bs_data = Datasource(bs_data, 'BootstrapResammpledData')
    table = with_data.add(bs_data)

  return table, with_data


def is_compatible(sql0, sql1):
  """Checks if two datasources are compatible so their columns can be merged.

  Being compatible means datasources have same FROM and GROUP BY clauses. The
  FROM clause is more complex. If a FROM clause is a substing of another one,
  which means the latter is a JOIN of the former and some other datasources, we
  still consider them compatible. You might wonder why because the JOIN add or
  drop rows from the original data. The reason is all the JOINs we generate are
  1. CROSS JOIN with a single value, which is just adding a constant column.
  2. INNER JOIN The JOIN is done on all GROUP BY columns on both tables and they
    both have all slices, even with NULLs, so again the effect is same as adding
    columns.
  3. The LEFT JOIN. The left side is the original table. This might add rows
    with NULL values as some children SQL might miss some slices. But all the
    SimpleMetrics ignore NULL so it won't affect the result.
  As the result, as long as one datasource contains another one, they are
  considered compatible. If in the future we have a rule that generates JOIN
  that isn't compatible with original data, we need to change this function.

  Args:
    sql0: A Sql instance.
    sql1: A Sql instance.

  Returns:
    If sql0 and sql1 are compatible.
    The larger FROM clause if compatible.
  """
  if not isinstance(sql0, Sql) or not isinstance(sql1, Sql):
    raise ValueError('Both inputs must be a Sql instance!')
  if sql0.where != sql1.where or sql0.groupby != sql1.groupby:
    return False, None
  if sql0.from_data == sql1.from_data:
    return True, sql1.from_data
  # Exclude cases where two differ on suffix.
  if (str(sql0.from_data) + '\n' in str(sql1.from_data) or
      str(sql0.from_data) + ' ' in str(sql1.from_data)):
    return True, sql1.from_data
  if (str(sql1.from_data) + '\n' in str(sql0.from_data) or
      str(sql1.from_data) + ' ' in str(sql0.from_data)):
    return True, sql0.from_data
  return False, None


def get_global_filter(metric):
  """Collects the filters that can be applied globally to the Metric tree."""
  global_filter = Filters()
  if metric.where:
    global_filter.add(metric.where)
  children_filters = [
      set(get_global_filter(c))
      for c in metric.children
      if isinstance(c, metrics.Metric)
  ]
  if children_filters:
    shared_filter = set.intersection(*children_filters)
    global_filter.add(shared_filter)
  return global_filter


def add_suffix(alias):
  """Adds an int suffix to alias."""
  m = re.search(r'([0-9]+)$', alias)
  if m:
    suffix = m.group(1)
    alias = alias[:-len(suffix)] + str(int(suffix) + 1)
    return alias
  else:
    return alias + '_1'


def get_alias(c):
  return getattr(c, 'alias_raw', c)


def escape_alias(alias):
  # Macro still gets parsed inside backquotes.
  if alias and '$' in alias:
    alias = alias.replace('$', 'macro_')
  # Don't escape if alias is already escaped.
  if alias and set(r""" `~!@#$%^&*()-=+[]{}\|;:'",.<>/?""").intersection(
      alias) and not (alias.startswith('`') and alias.endswith('`')):
    return '`%s`' % alias.replace('\\', '\\\\')
  return alias


def format_to_condition(val):
  if isinstance(val, str) and not val.startswith('$'):
    return '"%s"' % val
  return '%s' % val


@functools.total_ordering
class SqlComponent():
  """Base class for a SQL component like column, tabel and filter."""

  def __eq__(self, other):
    return str(self) == str(other)

  def __lt__(self, other):
    return str(self) < other

  def __repr__(self):
    return str(self)

  def __hash__(self):
    return hash(str(self))

  def __bool__(self):
    return bool(str(self))

  def __nonzero__(self):
    return bool(str(self))

  def __add__(self, other):
    return str.__add__(str(self), other)

  def __mul__(self, other):
    return str.__mul__(str(self), other)

  def __rmul__(self, other):
    return str.__rmul__(str(self), other)

  def __getitem__(self, idx):
    return str(self)[idx]


class SqlComponents(SqlComponent):
  """Base class for a bunch of SQL components like columns and filters."""

  def __init__(self, children=None):
    super(SqlComponents, self).__init__()
    self.children = []
    self.add(children)

  def add(self, children):
    if not isinstance(children, str) and isinstance(children, abc.Iterable):
      for c in children:
        self.add(c)
    else:
      if children and children not in self.children:
        self.children.append(children)
    return self

  def __iter__(self):
    for c in self.children:
      yield c

  def __len__(self):
    return len(self.children)

  def __getitem__(self, key):
    return self.children[key]

  def __setitem__(self, key, value):
    self.children[key] = value


class Filter(SqlComponent):
  """Represents single condition in SQL WHERE clause."""

  def __init__(self, cond: Text):
    super(Filter, self).__init__()
    if isinstance(cond, Filter):
      self.cond = cond.cond
    else:
      self.cond = cond.replace('==', '=') or ''

  def __str__(self):
    return '(%s)' % self.cond if ' OR ' in self.cond.upper() else self.cond


class Filters(SqlComponents):
  """Represents a bunch of SQL conditions."""

  @property
  def where(self):
    return sorted((str(Filter(f)) for f in self.children))

  def remove(self, filters):
    if not filters:
      return self
    self.children = [c for c in self.children if c not in Filters(filters)]
    return self

  def __str__(self):
    return ' AND '.join(self.where)


class Column(SqlComponent):
  """Represents a SQL column.

  Generates a single row in the SELECT clause in SQL. Here are some examples of
  the input and representation.

  Input => Representation
  Column('click', 'SUM({})') => SUM(click) AS `sum(click)`
  Column('click * weight', 'SUM({})', 'foo') => SUM(click * weight) AS foo
  Column('click', 'SUM({})', auto_alias=False) => SUM(click)
  Column('click', 'SUM({})', filters='region = "US"') =>
    SUM(IF(region = "US", click, NULL)) AS `sum(click)`
  Column('region') => region  # No alias because it's same as the column.
  Column('* EXCEPT (click)', auto_alias=False) => * EXCEPT (click)
  Column(('click', 'impression'), 'SAFE_DIVIDE({}, {})', 'ctr') =>
    SAFE_DIVIDE(click, impression) AS ctr.
  Column(('click', 'impr'), 'SAFE_DIVIDE({}, {})', 'ctr', 'click > 5') =>
    SAFE_DIVIDE(IF(click > 5, click, NULL), IF(click > 5, impr, NULL)) AS ctr.
  Column('click', 'SUM({})', partition='region', auto_alias=False) =>
    SUM(click) OVER (PARTITION BY region)

  The representation is generated by applying the self.fn to self.column, then
  adding optional OVER clause and renaming. The advantange of using Column
  instead of raw string is
  1. It handles filters nicely.
  2. Even you don't need filters you can still pass the raw string, for exmaple,
    '* EXCEPT (click)', in and it'd equivalent to a string, but can be used
    with other Columns.
  3. It supports arithmetic operations.
    Column('click') * 2 is same as Column('click * 2') and
    Column('click')  Column('impression') is same as
    Column(('click', 'impression'), 'SAFE_DIVIDE({}, {})') except for the
    auto-generated aliases. This makes constructing complex SQL column easy.
  4. Alias will be sanitized and auto-added if necessary.
  """

  def __init__(self,
               column=None,
               fn: Text = '{}',
               alias: Optional[Text] = None,
               filters=None,
               partition=None,
               order=None,
               window_frame=None,
               auto_alias=True):
    super(Column, self).__init__()
    self.column = [column] if isinstance(column, str) else column or []
    self.fn = fn
    self.filters = Filters(filters)
    self.alias_raw = alias
    if not alias and auto_alias:
      self.alias_raw = fn.lower().format(*self.column)
    self.partition = partition
    self.order = order
    self.window_frame = window_frame
    self.auto_alias = auto_alias

  @property
  def alias(self):
    return escape_alias(self.alias_raw)

  @alias.setter
  def alias(self, alias):
    self.alias_raw = alias

  def set_alias(self, alias):
    self.alias = alias
    return self

  @property
  def expression(self):
    """Genereates the representation without the 'AS ...' part."""
    over = None
    if not (self.partition is None and self.order is None and
            self.window_frame is None):
      partition = 'PARTITION BY %s' % ', '.join(
          Columns(self.partition).expressions) if self.partition else ''
      order = 'ORDER BY %s' % ', '.join(Columns(
          self.order).expressions) if self.order else ''
      frame = self.window_frame
      window_clause = ' '.join(c for c in (partition, order, frame) if c)
      over = ' OVER (%s)' % window_clause
    column = ('IF(%s, %s, NULL)' % (self.filters, c) if self.filters else c
              for c in self.column)
    res = self.fn.format(*column)
    return res + over if over else res

  def __str__(self):
    if not self.expression:
      return ''
    res = self.expression
    if (not self.alias_raw and not self.auto_alias) or res == self.alias:
      return res
    return '%s AS %s' % (res, self.alias)

  def __add__(self, other):
    return Column(
        '{} + {}'.format(*add_parenthesis_if_needed(self, other)),
        alias='%s + %s' % (self.alias_raw, get_alias(other)))

  def __radd__(self, other):
    alias = '%s + %s' % (get_alias(other), self.alias_raw)
    return Column(
        '{} + {}'.format(*add_parenthesis_if_needed(other, self)), alias=alias)

  def __sub__(self, other):
    return Column(
        '{} - {}'.format(*add_parenthesis_if_needed(self, other)),
        alias='%s - %s' % (self.alias_raw, get_alias(other)))

  def __rsub__(self, other):
    alias = '%s - %s' % (get_alias(other), self.alias_raw)
    return Column(
        '{} - {}'.format(*add_parenthesis_if_needed(other, self)), alias=alias)

  def __mul__(self, other):
    return Column(
        '{} * {}'.format(*add_parenthesis_if_needed(self, other)),
        alias='%s * %s' % (self.alias_raw, get_alias(other)))

  def __rmul__(self, other):
    alias = '%s * %s' % (get_alias(other), self.alias_raw)
    return Column(
        '{} * {}'.format(*add_parenthesis_if_needed(other, self)), alias=alias)

  def __neg__(self):
    return Column(
        '-{}'.format(*add_parenthesis_if_needed(self)),
        alias='-%s' % self.alias_raw)

  def __div__(self, other):
    return Column(
        'SAFE_DIVIDE({}, {})'.format(self.expression,
                                     getattr(other, 'expression', other)),
        alias='%s / %s' % (self.alias_raw, get_alias(other)))

  def __truediv__(self, other):
    return self.__div__(other)

  def __rdiv__(self, other):
    alias = '%s / %s' % (get_alias(other), self.alias_raw)
    return Column(
        'SAFE_DIVIDE({}, {})'.format(
            getattr(other, 'expression', other), self.expression),
        alias=alias)

  def __rtruediv__(self, other):
    return self.__rdiv__(other)

  def __pow__(self, other):
    return Column(
        'POWER({}, {})'.format(self.expression,
                               getattr(other, 'expression', other)),
        alias='%s ^ %s' % (self.alias_raw, get_alias(other)))

  def __rpow__(self, other):
    alias = '%s ^ %s' % (get_alias(other), self.alias_raw)
    return Column(
        'POWER({}, {})'.format(
            getattr(other, 'expression', other), self.expression),
        alias=alias)


def add_parenthesis_if_needed(*columns):
  for column in columns:
    if not isinstance(column, Column):
      yield column
      continue
    expression = column.expression
    if '+' in expression or '-' in expression:
      yield '(%s)' % expression
      continue
    yield expression


class Columns(SqlComponents):
  """Represents a bunch of SQL columns."""

  def __init__(self, columns=None, distinct=None):  # pylint: disable=super-init-not-called
    self.columns = collections.OrderedDict()
    self.add(columns)
    self.distinct = distinct
    if distinct is None and isinstance(columns, Columns):
      self.distinct = columns.distinct

  @property
  def children(self):
    return tuple(self.columns.values())

  @property
  def aliases(self):
    return [escape_alias(c.alias) for c in self]

  def add(self, children):
    """Adds a Column if not existing.

    Renames it when necessary.

    If the Column already exists with the same alias. Do nothing.
    If neither the Column nor the alias exist. Add it.
    If the Column exists but with a different alias. Add it.
    If the Column doesn't exists but the alias exists. Give the Column
    a new alias by adding a unique suffix. Then add it under the new alias.

    Args:
      children: A string or a Column or an iterable of them.

    Returns:
      self.
    """
    if not isinstance(children, str) and isinstance(children, abc.Iterable):
      for c in children:
        self.add(c)
      return self
    if not children:
      return self
    if isinstance(children, str):
      return self.add(Column(children))
    alias = children.alias_raw
    if alias not in self.columns:
      self.columns[alias] = children
      return self
    else:
      if children == self.columns[alias]:
        return self
      children.alias = add_suffix(alias)
      return self.add(children)

  def difference(self, columns):
    return Columns((c for c in self if c not in Columns(columns)))

  @property
  def expression(self):
    return list(map(str, self))

  @property
  def expressions(self):
    return [c.expression for c in self]

  def get_columns(self, break_line=False, indent=True):
    delimiter = ',\n' if break_line else ', '
    if indent:
      res = delimiter.join(('  %s' % e for e in self.expression))
      return '  DISTINCT\n' + res if self.distinct else res
    res = delimiter.join(self.expression)
    return 'DISTINCT ' + res if self.distinct else res

  def as_groupby(self):
    return ', '.join(self.aliases)

  def __str__(self):
    return self.get_columns(True)


class Datasource(SqlComponent):
  """Represents a SQL datasource, could be a table name or a SQL query."""

  def __init__(self, table, alias=None):
    super(Datasource, self).__init__()
    if isinstance(table, Datasource):
      self.table = table.table
      self.alias = table.alias
    else:
      self.table = str(table).strip()
      self.alias = alias
    self.alias = escape_alias(self.alias)
    self.is_table = not self.table.upper().startswith('SELECT')

  def get_expression(self, form='FROM'):
    """Gets the expression that can be used in a FROM or WITH clause."""
    if form.upper() not in ('FROM', 'WITH'):
      raise ValueError('Unrecognized form for datasource!')
    if form.upper() == 'WITH':
      if not self.alias:
        raise ValueError('Datasource in a WITH clause must have an alias!')
      if self.is_table:
        raise ValueError('Datasource in a WITH clause must be a SQL query!')
      return '%s AS (%s)' % (self.alias, self.table)
    else:
      return str(self)

  def join(self, other, on=None, using=None, join='', alias=None):
    return Join(self, other, on, using, join, alias)

  def __str__(self):
    table = self.table if self.is_table else '(%s)' % self.table
    return '%s AS %s' % (table, self.alias) if self.alias else str(table)


class Join(Datasource):
  """Represents a JOIN of two Datasources."""

  def __init__(self,
               datasource1,
               datasource2,
               on=None,
               using=None,
               join='',
               alias=None):
    if on and using:
      raise ValueError('A JOIN cannot have both ON and USING condition!')
    if join.upper() not in ('', 'INNER', 'FULL', 'FULL OUTER', 'LEFT',
                            'LEFT OUTER', 'RIGHT', 'RIGHT OUTER', 'CROSS'):
      raise ValueError('Unrecognized JOIN type!')
    self.ds1 = Datasource(datasource1)
    self.ds2 = Datasource(datasource2)
    self.join_type = join.upper()
    self.on = Filters(on)
    self.using = Columns(using)
    super(Join, self).__init__(str(self), alias)

  def __str__(self):
    join = '%s JOIN' % self.join_type if self.join_type else 'JOIN'
    sql = '\n'.join(map(str, (self.ds1, join, self.ds2)))
    if self.on:
      return '%s\nON %s' % (sql, self.on)
    if self.using:
      return '%s\nUSING (%s)' % (sql, ', '.join(self.using.aliases))
    return sql


class Datasources(SqlComponents):
  """Represents a bunch of SQL datasources in a WITH clause."""

  def __init__(self, datasources=None):
    super(Datasources, self).__init__()
    self.children = collections.OrderedDict()
    self.add(datasources)

  @property
  def datasources(self):
    return (Datasource(v, k) for k, v in self.children.items())

  def add(self, children: Union[Datasource, Iterable[Datasource]]):
    """Adds a datasource if not existing.

    Renames it when necessary.

    If the Datasource already exists with the same alias. Do nothing.
    If neither the Datasource nor the alias exist. Add it.
    If the Datasource exists but with a different alias. Don't add it, and
    change its alias to the one already exists in-place.
    If the Datasource doesn't exists but the alias exists. Give the Datasource
    a new alias by adding a unique suffix. Then add it under the new alias.

    Args:
      children: A Datasource instance or an iterable of Datasource(s).

    Returns:
      The alias of the Datasource we eventually add.
    """
    if not isinstance(children, str) and isinstance(children, abc.Iterable):
      for c in children:
        self.add(c)
      return
    if not children:
      return
    if not isinstance(children, Datasource):
      raise ValueError('Not a Datasource!')
    alias, table = children.alias, children.table
    if alias not in self.children:
      if table not in self.children.values():
        self.children[alias] = table
        return alias
      children.alias = [k for k, v in self.children.items() if v == table][0]
      return children.alias
    else:
      if table == self.children[alias]:
        return alias
      children.alias = add_suffix(alias)
      return self.add(children)

  def __str__(self):
    return ',\n'.join((d.get_expression('WITH') for d in self.datasources if d))


class Sql(SqlComponent):
  """Represents a SQL query."""

  def __init__(self,
               columns,
               from_data,
               where=None,
               groupby=None,
               with_data=None):
    super(Sql, self).__init__()
    self.columns = Columns(columns)
    self.from_data = Datasource(from_data)
    self.where = Filters(where)
    self.groupby = Columns(groupby)
    self.with_data = Datasources(with_data)

  def add(self, attr, values):
    getattr(self, attr).add(values)
    return self

  def __str__(self):
    with_clause = 'WITH\n%s' % self.with_data if self.with_data else None
    tmpl = 'SELECT DISTINCT\n%s' if self.columns.distinct else 'SELECT\n%s'
    select_clause = tmpl % Columns(self.groupby).add(self.columns)
    from_clause = ('FROM %s'
                   if self.from_data.is_table else 'FROM\n%s') % self.from_data
    where_clause = 'WHERE\n%s' % self.where if self.where else None
    groupby_clause = 'GROUP BY %s' % self.groupby.as_groupby(
    ) if self.groupby else None
    clauses = [
        c for c in (with_clause, select_clause, from_clause, where_clause,
                    groupby_clause) if c is not None
    ]
    return '\n'.join(clauses)


# Because this module depends on metrics.py so there we cannot depend on sql.py
# We have to monkey patch the method like this so Metric can have to_sql().
metrics.Metric.to_sql = get_sql
