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
"""Operations to measure diversity."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from meterstick import operations
from meterstick import sql
import numpy as np
import pandas as pd


class DiversityBase(operations.Distribution):
  """Base class that captures shared logic of diversity Operations."""

  def __init__(self, over, child, name_tmpl, additional_fingerprint_attrs=None):
    super(DiversityBase, self).__init__(
        over,
        child,
        name_tmpl,
        additional_fingerprint_attrs=additional_fingerprint_attrs,
    )
    self.extra_index = []

  def get_distribution_sql_and_columns(
      self, table, split_by, global_filter, indexes, local_filter, with_data
  ):
    dist_sql, with_data = super(DiversityBase, self).get_sql_and_with_clause(
        table, split_by, global_filter, indexes, local_filter, with_data
    )
    # Remove 'Distribution of ' in column names.
    for c in dist_sql.columns:
      alias = c.alias_raw
      if alias.startswith('Distribution of '):
        c.set_alias(alias[len('Distribution of ') :])
    child_table = sql.Datasource(dist_sql, 'Distribution')
    child_table_alias = with_data.merge(child_table)
    return child_table_alias, dist_sql.columns, with_data

  def to_dataframe(self, res):
    if isinstance(res, pd.Series):
      return res.to_frame().T
    return super(DiversityBase, self).to_dataframe(res)


class HHI(DiversityBase):
  """Herfindahl–Hirschman index of metric distribution."""

  def __init__(self, over, child=None):
    super(HHI, self).__init__(over, child, 'HHI of {}')

  def compute_on_children(self, child, split_by):
    dist = super(HHI, self).compute_on_children(child, split_by)
    res = self.group(dist**2, split_by).sum()
    return self.to_dataframe(res)

  def get_sql_and_with_clause(
      self, table, split_by, global_filter, indexes, local_filter, with_data
  ):
    # First get the queries for Distribution(over, child).
    child_table_alias, dist_columns, with_data = (
        self.get_distribution_sql_and_columns(
            table, split_by, global_filter, indexes, local_filter, with_data
        )
    )
    columns = sql.Columns()
    all_split_by = (
        sql.Columns(split_by.aliases).add(self.extra_split_by).aliases
    )
    # For every value column, compute SUM(POWER(val, 2)) which is the HHI.
    for c in dist_columns:
      if c.alias in all_split_by:
        continue

      col = sql.Column(
          c.alias,
          'SUM(POWER({}, 2))',
      )
      col.set_alias(self.name_tmpl.format(c.alias_raw))
      columns.add(col)
    return (
        sql.Sql(columns, child_table_alias, groupby=indexes.aliases),
        with_data,
    )


class Entropy(DiversityBase):
  """Entropy of metric distribution."""

  def __init__(self, over, child=None):
    super(Entropy, self).__init__(over, child, 'Entropy of {}')

  def compute_on_children(self, child, split_by):
    dist = super(Entropy, self).compute_on_children(child, split_by)
    res = self.group(-dist * np.log(dist), split_by).sum()
    return self.to_dataframe(res)

  def get_sql_and_with_clause(
      self, table, split_by, global_filter, indexes, local_filter, with_data
  ):
    # First get the queries for Distribution(over, child).
    child_table_alias, dist_columns, with_data = (
        self.get_distribution_sql_and_columns(
            table, split_by, global_filter, indexes, local_filter, with_data
        )
    )
    all_split_by = (
        sql.Columns(split_by.aliases).add(self.extra_split_by).aliases
    )
    columns = sql.Columns()
    # For every value column, compute -SUM(val * LOG(val)) which is the entropy.
    for c in dist_columns:
      if c.alias in all_split_by:
        continue

      col = sql.Column(
          (c.alias, c.alias),
          '-SUM({} * LOG({}))',
      )
      col.set_alias(self.name_tmpl.format(c.alias_raw))
      columns.add(col)
    return (
        sql.Sql(columns, child_table_alias, groupby=indexes.aliases),
        with_data,
    )


class TopK(DiversityBase):
  """The total share of the largest k contributors."""

  def __init__(self, over, k, child=None, additional_fingerprint_attrs=None):
    if not isinstance(k, int):
      raise ValueError('k must be an integer!')
    super(TopK, self).__init__(
        over,
        child,
        "Top-%s's share of {}" % k,
        ['k'] + (additional_fingerprint_attrs or []),
    )
    self.k = k

  def compute_on_children(self, child, split_by):
    dist = super(TopK, self).compute_on_children(child, split_by)
    top_k = []
    grouped = self.group(dist, split_by)
    # groupby().nlargest() only works on Series but not DataFrame, so we need to
    # iterate the columns.
    for col in dist:
      top_k.append(self.group(grouped[col].nlargest(self.k), split_by).sum())
    if split_by:
      return pd.concat(top_k, axis=1)
    return pd.DataFrame([top_k], columns=dist.columns)

  def get_sql_and_with_clause(
      self, table, split_by, global_filter, indexes, local_filter, with_data
  ):
    """Gets the SQL query and WITH clause.

    The query is constructed in 4 steps.
    1. Get the query for the Distribution of the child Metric.
    2. Keep all indexing/groupby columns unchanged.
    3. For all value columns, collect the top-k values into an array by
       ARRAY_AGG(val_col ORDER BY val_col DESC LIMIT k) AS val_arr.
    4. For all value columns, do 'SELECT SUM(x) FROM UNNEST(val_arr) AS x' to
       get the sum of the top-k values.

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
    child_table_alias, dist_columns, with_data = (
        self.get_distribution_sql_and_columns(
            table, split_by, global_filter, indexes, local_filter, with_data
        )
    )
    all_split_by = (
        sql.Columns(split_by.aliases).add(self.extra_split_by).aliases
    )
    top_k_array_columns = sql.Columns()
    top_k_sum_columns = sql.Columns(indexes.aliases)
    for c in dist_columns:
      if c.alias in all_split_by:
        continue

      top_k_array_col = sql.Column(
          (c.alias, c.alias),
          'ARRAY_AGG({} ORDER BY {} DESC LIMIT %s)' % self.k,
      )
      top_k_array_col.set_alias(c.alias_raw)
      top_k_array_columns.add(top_k_array_col)
      top_k_sum_col = sql.Column(
          top_k_array_col.alias,
          '(SELECT SUM(x) FROM UNNEST({}) AS x)',
      )
      top_k_sum_col.set_alias(self.name_tmpl.format(c.alias_raw))
      top_k_sum_columns.add(top_k_sum_col)
    top_k_sql = sql.Sql(
        top_k_array_columns, child_table_alias, groupby=indexes.aliases
    )
    top_k_table = sql.Datasource(top_k_sql, 'TopKArrays')
    top_k_table_alias = with_data.merge(top_k_table)
    return sql.Sql(top_k_sum_columns, top_k_table_alias), with_data


class Nxx(DiversityBase):
  """The minimum number of contributors to achieve certain share."""

  def __init__(
      self, over, share, child=None, additional_fingerprint_attrs=None
  ):
    if not 0 < share <= 1:
      raise ValueError('Share must be in (0, 1]!')
    super(Nxx, self).__init__(
        over,
        child,
        'N(%s) of {}'
        % (int(100 * share) if (100 * share).is_integer() else 100 * share),
        ['share'] + (additional_fingerprint_attrs or []),
    )
    self.share = share

  def compute_on_children(self, child, split_by):
    dist = super(Nxx, self).compute_on_children(child, split_by)
    return pd.concat(
        [self.nxx_for_one_col(dist[[c]], split_by) for c in dist], axis=1
    )

  def nxx_for_one_col(self, col, split_by):
    sorted_col = col.sort_values(split_by + list(col.columns), ascending=False)
    cumsum = self.group(sorted_col, split_by).cumsum()
    res = self.group(cumsum < self.share, split_by).sum() + 1
    return self.to_dataframe(res)

  def get_sql_and_with_clause(
      self, table, split_by, global_filter, indexes, local_filter, with_data
  ):
    """Gets the SQL query and WITH clause.

    The query is constructed in 4 steps.
    1. Get the query for the Distribution of the child Metric.
    2. Keep all indexing/groupby columns unchanged.
    3. For all value columns, order the values in descending order and compute
       the cumulative sum by SELECT
       SUM(val_col) OVER
        (ORDER BY val_col DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    4. Get the minimum number of players to achieve the share by SELECT
       COUNTIF(cumulative_sum < share) + 1.

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
    child_table_alias, dist_columns, with_data = (
        self.get_distribution_sql_and_columns(
            table, split_by, global_filter, indexes, local_filter, with_data
        )
    )
    all_split_by = (
        sql.Columns(split_by.aliases).add(self.extra_split_by).aliases
    )
    cumsum_cols = sql.Columns(indexes.aliases)
    nxx_cols = sql.Columns()
    for c in dist_columns:
      if c.alias in all_split_by:
        continue

      cumsum_col = sql.Column(
          c.alias,
          'SUM({})',
          partition=split_by.aliases,
          order=f'{c.alias} DESC',
          window_frame='ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW',
      )
      cumsum_col.set_alias('Cumulative %s' % c.alias_raw)
      cumsum_cols.add(cumsum_col)

      nxx_col = sql.Column(
          cumsum_col.alias,
          'COUNTIF({} < %s) + 1' % self.share,
      )
      nxx_col.set_alias(self.name_tmpl.format(c.alias_raw))
      nxx_cols.add(nxx_col)
    cumsum_sql = sql.Sql(cumsum_cols, child_table_alias)
    cumsum_table = sql.Datasource(cumsum_sql, 'CumulativeDistribution')
    cumsum_alias = with_data.merge(cumsum_table)
    return sql.Sql(nxx_cols, cumsum_alias, groupby=indexes.aliases), with_data
