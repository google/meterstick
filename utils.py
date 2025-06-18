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
"""Utils functions for things like DataFrame manipulation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import datetime
import glob
import os
from typing import Iterable, List, Optional, Text, Union

from meterstick import sql
import pandas as pd


def get_name(obj):
  return getattr(obj, 'name', str(obj))


def is_metric(m):
  return hasattr(m, 'compute_on')


def melt(df):
  """Stacks the outermost comlumn level to the outermost index level.

  Similar to pd.stack(0) except
  1. It stacks to the leftmost level.
  2. Doesn's sort.
  3. If the result is a pd.Series, always convert it to a DataFrame.
  4. The stacked level will be named 'Metric'. The new column will be named
    'Value'.

  Args:
    df: An unmelted DataFrame in Meterstick's context, namely, the outermost
      column level is Metrics' names.

  Returns:
    A melted DataFrame in Meterstick's context, namely, the outermost index
    level is Metrics' names.
  """
  if not isinstance(df, pd.DataFrame):
    return df

  if isinstance(df.columns, pd.MultiIndex):
    flat_idx = False
    names = df.columns.get_level_values(0).unique()
  else:
    flat_idx = True
    names = df.columns
  df = pd.concat([df[n] for n in names], keys=names, names=['Metric'])
  df = pd.DataFrame(df)
  if flat_idx:
    if len(df.columns) > 1:
      raise ValueError(
          f'Duplicate columns found: {df.columns}. You likely have'
          ' duplicate names in your Metrics. This is a common error when you'
          ' have different filters on the same Metric. Please give Metrics'
          ' different names.'
      )
    df.columns = ['Value']
  return remove_empty_level(df)


def unmelt(df):
  """Unstacks the outermost index level to the outermost column level.

  Similar to pd.unstack(0) except
  1. It unstacks to the outermost level.
  2. Doesn's sort.
  3. If the result is a pd.Series, always convert it to a DataFrame.
  4. The unstacked level will be named 'Metric'.
  5. If the original DataFrame only has one column and it's named Value. The
    level will be dropped.

  Args:
    df: A melted DataFrame in Meterstick's context, namely, the outermost index
      level is Metrics' names.

  Returns:
    An unmelted DataFrame in Meterstick's context, namely, the outermost column
    level is Metrics' names.
  """
  if not isinstance(df, pd.DataFrame):
    return df

  if isinstance(df.index, pd.MultiIndex):
    names = df.index.get_level_values(0).unique()
  else:
    names = df.index
  single_value_col = len(df.columns) == 1 and df.columns[0] == 'Value'
  if len(df.index.names) == 1:
    # This is for pandas 2.1.0
    # https://pandas.pydata.org/docs/whatsnew/v2.1.0.html#new-implementation-of-dataframe-stack
    # It should be removed when future_stack becomes the default in pandas 3.0.
    df = pd.DataFrame(df.stack(0, future_stack=True)).T
  else:
    df = pd.concat([df.loc[n] for n in names],
                   axis=1,
                   keys=names,
                   names=['Metric'])
  if single_value_col:
    return df.droplevel(1, axis=1)
  return df


def remove_empty_level(df):
  """Drops redundant levels in the index of df."""
  if not isinstance(df, pd.DataFrame) or not isinstance(df.index,
                                                        pd.MultiIndex):
    return df

  drop = []
  for i, level in enumerate(df.index.levels):
    if not level.name and len(level.values) == 1 and not level.values[0]:
      drop.append(i)
  return df.droplevel(drop)


def apply_name_tmpl(name_tmpl, res, melted=False):
  """Applies name_tmpl to all columns or pd.Series.name."""
  if not name_tmpl:
    return res
  if isinstance(res, pd.Series):
    res.name = name_tmpl.format(res.name)
  elif isinstance(res, pd.DataFrame):
    if melted:
      if len(res.index.names) > 1:
        res.index = res.index.set_levels(
            map(name_tmpl.format, res.index.levels[0]), level=0)
      else:
        res.index = pd.Index(
            map(name_tmpl.format, res.index), name=res.index.name)
    else:
      if len(res.columns.names) > 1:
        res.columns = res.columns.set_levels(
            map(name_tmpl.format, res.columns.levels[0]), level=0
        )
      else:
        res.columns = map(name_tmpl.format, res.columns)
  return res


def get_extra_split_by(metric, return_superset=False):
  """Collects the extra split_by added by Operations for the metric tree.

  Args:
    metric: A Metric instance.
    return_superset: If to return the superset of extra split_by if metric has
      incompatible split_by.

  Returns:
    A tuple of all columns used to split the df in metric.compute_on(df).
  """
  extra_split_by = metric.extra_split_by[:]
  children_idx = [
      get_extra_split_by(c, return_superset)
      for c in metric.children
      if is_metric(c)
  ]
  if len(set(children_idx)) > 1:
    if not return_superset:
      raise ValueError('Incompatible split_by!')
    children_idx_superset = set()
    children_idx_superset.update(*children_idx)
    children_idx = [list(children_idx_superset)]
  if children_idx:
    extra_split_by += list(children_idx[0])
  return tuple(extra_split_by)


def get_leaf_metrics(metric, include_constants=False):
  leaf = []
  for m in metric.traverse(include_constants=include_constants):
    if not getattr(m, 'children', []):
      leaf.append(m)
  return leaf


def get_global_filter(metric) -> sql.Filters:
  """Collects the filters that can be applied globally to the Metric tree."""
  global_filter = sql.Filters()
  if metric.where:
    global_filter.add(metric.where_)
  # Filters inside resampling Operations shound not be considered global.
  if type(metric).__name__ in ('Jackknife', 'Bootstrap'):
    return global_filter
  children_filters = [
      set(get_global_filter(c)) for c in metric.children if is_metric(c)
  ]
  if children_filters:
    shared_filter = set.intersection(*children_filters)
    global_filter.add(shared_filter)
  return global_filter


def push_filters_to_leaf(metric, is_root=True):
  """Returns a Metric that all filters have been pushed to leaf nodes.

  Note that the return can differ subtly to the original metric when computing
  on the same data. For example, the result of these two metrics won't be same
    m1 = Jackknife('unit', Sum(x), where='unit !=1')
    m2 = Jackknife('unit', Sum(x, where='unit !=1'))
  m1.compute_on(df) == Jackknife('unit', Sum(x)).compute_on(df[df.unit != 1])
  while m2.compute_on(df) doesn't. For m1, the unit-1 slice is never used while
  for m2, it's used in resampling and only get dropped during the computation.
  In other words, suppose there are 5 units in the data, m1 won't generate
  leave-1-out data while m2 will.
  This situation is rare and for built-in stuff it only happens to Jackknife and
  Bootstrap with unit when the filter filters out unit slice(s). Nevertheless,
  it'd be good to keep the subtlety in mind.

  Args:
    metric: A Metric instance.
    is_root: If metric is the root node of the metric tree. For root node, if it
      has a filter encoded in its cache_key, we pass it down as well.

  Returns:
    A Metric instance 'equivalent' to the input metric but only have filters in
    leaf nodes. Please see above for the subtlety of 'equivalent'.
  """
  if not is_metric(metric):
    return metric
  if is_root and metric.cache_key and metric.cache_key.where:
    metric = copy.deepcopy(metric)
    metric.add_where(metric.cache_key.where)
  if not metric.children:
    return copy.deepcopy(metric)
  metric = copy.copy(metric)
  where = metric.where_
  children = metric.children
  if where:
    children = [
        copy.copy(c).add_where(where) if is_metric(c) else c for c in children
    ]
  metric.children = [push_filters_to_leaf(c, False) for c in children]
  metric.where = None
  return metric


class CacheKey:
  """Represents a cache_key used in the computation of Metrics.

  During the computation of a Metric, we often use a key to cache results. It
  either comes from users or a default value. For caching to be valid, one cache
  key needs to correspond to the same DataFrame AND split_by, but it can still
  lead to unintuitive behavior. For example,
  PercentChange(condition, base, Sum('X')).compute_on(df, cache_key='foo')
  would cache the percent change in PercentChange under key 'foo' and
  the result of Sum('X')).compute_on(df, condition)
  in Sum under 'foo'. Note it's NOT the result of Sum('X')).compute_on(df) gets
  cached because we don't need it to compute the PercentChange. As a result, if
  one calls
  PercentChange(condition, base, Sum('X')).compute_on(df, cache_key='foo')
  then
  Sum('X')).compute_on(df, cache_key='foo'),
  there will be a cache miss, but
  Sum('X')).compute_on(df, condition, cache_key='foo')
  will hit the cache. So internally the key used in cache have to encode
  split_by information. It cannot just be 'foo'.
  Similarly, internal cache key needs to encode the 'where' arg in Metric too or
  PercentChange(sumx, where='grp == 1') - PercentChange(sumx, where='grp == 0')
  would always return 0.
  The last piece is the slice information. It matters when a Metric is not
  vectorized. We need to store which slice the result is for.

  Attributes:
    metric: A Metric instance whose results are cached under the CacheKey.
    key: The raw cache_key user provides, or the default key.
    where: The filters to apply to the input DataFrame.
    split_by: The columns to split by.
    slice_val: When a Metric is not vectorized, it computes on a slice of data.
      Such computation shoulnd't use the same cache key with the vecoreized
      Metrics. slice_val is a dict of the value of the split_by columns of the
      data slice with the keys being columns in split_by.
    all_filters: The merge of all 'where' conditions that can be passed to
      df.query().
    extra_info: Extra information to distinguish CacheKeys.
    fingerprint: The unique identifier of CacheKey. Used to hash.
  """

  def __init__(self,
               metric,
               key,
               where: Optional[Union[Text, Iterable[Text]]] = None,
               split_by: Optional[Union[Text, List[Text]]] = None,
               slice_val=None,
               extra_info=()):
    """Wraps cache_key, split_by, filters and slice information.

    Args:
      metric: A Metric instance.
      key: A raw key or a CacheKey. If it's a CacheKey, we unwrap it and extend
        its split_by and where.
      where: The filters to apply to the input DataFrame.
      split_by: The columns to split by.
      slice_val: An ordered tuple of key, value pair in slice_val.
      extra_info: Extra information to distinguish CacheKeys.
    """
    self.metric = copy.deepcopy(metric)
    # `where` accumulates the filters so far and already includes metric.where.
    self.metric.where = None
    split_by = (split_by,) if isinstance(split_by, str) else split_by or ()
    where = [where] if isinstance(where, str) else where or []
    if isinstance(key, CacheKey):
      self.key = key.key
      self.where = key.where.copy()
      if where is not None:
        self.where.update(where)
      self.split_by = tuple(split_by) if split_by else key.split_by[:]
      self.slice_val = slice_val or {}
      self.extra_info = key.extra_info
      for k, v in key.slice_val:
        if k in self.slice_val and self.slice_val[k] != v:
          raise ValueError('Incompatible data slice values!')
        self.slice_val[k] = v
    else:
      self.key = key
      self.where = set(where)
      self.split_by = tuple(split_by)
      self.slice_val = slice_val or {}
      self.extra_info = extra_info
    self.slice_val = tuple(sorted(self.slice_val.items()))
    # `where` accumulates the filters so far and already includes metric.where.
    # self.metric doesn't have `where` but we cannot use it here because
    # get_fingerprint might collect the id of metric so the original metric is
    # needed.
    metric_fingerprint = [
        i for i in metric.get_fingerprint() if i[0] != 'where'
    ]
    self.fingerprint = {
        'metric': tuple(metric_fingerprint),
        'key': self.key,
        'split_by': self.split_by,
        'slice_val': self.slice_val,
        'extra_info': self.extra_info,
        'where': tuple(sorted(tuple(self.where))),
    }

  def add_extra_info(self, extra_info: str):
    self.extra_info = tuple(list(self.extra_info) + [extra_info])
    self.fingerprint['extra_info'] = self.extra_info

  def replace_key(self, key):
    new_key = copy.deepcopy(self)
    new_key.key = key
    new_key.fingerprint['key'] = key
    return new_key

  def replace_metric(self, new_metric):
    new_key = copy.deepcopy(self)
    new_key.metric = new_metric
    new_key.fingerprint['metric'] = new_metric.get_fingerprint()
    return new_key

  def replace_split_by(self, split_by):
    split_by = split_by or ()
    split_by = (split_by,) if isinstance(split_by, str) else tuple(split_by)
    new_key = copy.deepcopy(self)
    new_key.split_by = split_by
    new_key.fingerprint['split_by'] = split_by
    return new_key

  def replace_where(self, where):
    where = (where,) if isinstance(where, str) else tuple(sorted(where)) or ()
    new_key = copy.deepcopy(self)
    new_key.where = where
    new_key.fingerprint['where'] = where
    return new_key

  def __eq__(self, other):
    return isinstance(other, CacheKey) and hash(self) == hash(other)

  def __hash__(self):
    return hash((
        self.fingerprint['metric'],
        self.fingerprint['key'],
        self.fingerprint['split_by'],
        self.fingerprint['slice_val'],
        self.fingerprint['where'],
        self.fingerprint['extra_info'],
    ))

  def __repr__(self):
    return (
        'metric: %s, key: %s, split_by: %s, slice: %s, where: %s, '
        'extra_info: %s'
        % (
            self.metric.name,
            self.key,
            self.split_by,
            self.slice_val,
            self.where,
            self.extra_info,
        )
    )


def adjust_slices_for_loo(
    bucket_res: pd.Series, split_by: Optional[List[Text]] = None, df=None
):
  """Corrects the slices in the bucketized result.

  Jackknife has a precomputation step where we precompute leave-one-out (LOO)
  results for Sum and/or Count. The idea is that, for example, for Sum, we
  can get the LOO result by subtracting df.groupby([unit, groupby columns])
  from df.groupby([groupby columns]) where unit is the column to Jackknife on
  and groupby columns are optional and have two parts, both are optional too.
  The first part comes from Operations. For example,
  PercentChange(cond, base, Sum('x')).compute_on(df) internally computes
  Sum('x').compute_on(df, cond). The second part is the split_by passed to
  Jackknife(...).compute_on(df, split_by). In other words, the groupby
  columns are split_by + [cond] for
  Jackknife(PercentChange(cond, base, Sum('x'))).compute_on(df, split_by).
  The issue is that the index of the df.groupby([groupby columns]), the
  bucket_res here, can be different to that of the correct LOO result for
  two reasons.
  1. Sparse data: If one slice only appears in one unit, then it will appear in
  the bucket_res but shouldn't be included in the LOO.
  2. If descendants of the Jackknife have filters that filter out certain
  slices so bucket_res doesn't have them, we might need to add them back.
  For example, for data
    unit  grp  cond  X
     1     A   foo   10
     2     A   foo   20
     2     A   bar   30
     2     B   bar   40,
  the bucket_res of
  Jackknife('unit', PercentChange('cond', 'foo', Sum('X'))
    ).compute_on(data, 'grp') won't have slice 1 * A * bar but we should have it
  in the leave-unit-1-out result.
  The procedure to get the correct indexes for LOO is that
  1. For each split_by group, find all the unique Jackknife unit values.
  2. For each unit, i, find all the slices remained in the levels added by
  Operations, if any, in the leave-unit-i-out bucket_res. The
  split_by slice * i * operation slice are what we need to include in the LOO.

  Args:
    bucket_res: The first level of its indexes is the unit to Jackknife on,
      followed by split_by, then levels added by Operations, if any.
    split_by: The list of column(s) from Jackknife().compute_on(df, split_by).
    df: A dataframe that has the same slices as the df that Jackknife computes
      on.

  Returns:
    A pd.Series that has the same index names as bucket_res, but with some
    levels removed and/or added.
  """
  indexes = bucket_res.index.names
  unit_and_operation_lvl = indexes[len(split_by) :]
  operation_lvl = unit_and_operation_lvl[1:]
  split_by_and_unit = indexes[: len(split_by) + 1]
  unit = split_by_and_unit[-1]
  expected_units = (
      df.groupby(split_by_and_unit, observed=True).first().iloc[:, [0]]
  )
  if not operation_lvl:
    return bucket_res.reindex(expected_units.index, fill_value=0)

  expected_units = expected_units.reset_index(unit)[[unit]]
  b = bucket_res.reset_index(unit_and_operation_lvl)
  suffix = '_meterstick'
  while any((c.endswith(suffix) for c in b.columns)) or suffix == unit:
    suffix += '_'
  on = split_by
  if not on:
    expected_units[suffix] = 1
    b[suffix] = 1
    on = suffix
  cross_joined = expected_units.merge(
      b, on=on, how='outer', suffixes=('', suffix)
  )
  expected_slices = (
      cross_joined[cross_joined[unit] != cross_joined[f'{unit}{suffix}']]
      .set_index(unit_and_operation_lvl, append=split_by)
      .index.drop_duplicates()
  )
  return bucket_res.reindex(expected_slices, fill_value=0)


def get_fully_expanded_equivalent_metric_tree(m, df=None):
  """Gets a Metric that is equivalent to m, and cannot be further expanded.

  Some Metrics can be expressed by simpler Metrics like Sum and Count. Sum and
  Count are easy to compute and Jackknife knows how to cut the corner to compute
  the leave-one-out estimates for them. If we can replace complex Metrics with
  Sum and Count then Jackknife can cut the corner for them too. For example,
  Dot(x, y) is equivalent to Sum(x * y) so instead of computing
  Jackknife(Dot(x, y)), we can compute Jackknife(Sum(x * y)). However, column
  `x * y` doesn't necessarily exist in df so we need to create it. A Metric's
  direct equivalent form could still not be simple enough. For example, weighted
  Mean(x, y) is equivalent to Dot(x, y) / Sum(y) where the Dot can be further
  simplified. Here we returns the equivalent Metric that cannot be simplified
  anymore and add the intermediate columns, `x * y` in the Dot example, to a
  copy of df.

  Args:
    m: A Metric.
    df: The dataframe we compute on. If None, we skip the computation part.

  Returns:
    A Metric equivalent to m with leaf Metrics replaced by Sum and Count. df is
    modified in-place.
    A copy of the original dataframe with auxiliary columns added.
  """
  df = copy.copy(df)
  prev = m
  prefix = get_unique_prefix(df) if df is not None else ''
  curr = get_equivalent_metric_tree(m, df, prefix)
  while prev != curr:
    prev, curr = curr, get_equivalent_metric_tree(curr, df, prefix)
  return curr, df


def get_equivalent_metric_tree(m, df=None, prefix=''):
  """Replaces Metrics in the tree of m with equivalent Metrics."""
  if not is_metric(m):
    return m
  if m.children:
    res = copy.deepcopy(m)
    res.children = [
        get_equivalent_metric_tree(c, df, prefix) for c in res.children
    ]
    return res
  if not m.get_equivalent(*m.get_auxiliary_cols()):
    return m
  equiv, df = get_equivalent_metric(m, df, prefix)
  return equiv


def get_equivalent_metric(m, df=None, prefix=''):
  """Gets the equivalent Metric of m and adds auxiliary columns to df."""
  if df is not None and not prefix:
    prefix = get_unique_prefix(df)
  df, auxiliary_cols = add_auxiliary_cols(m.get_auxiliary_cols(), df, prefix)
  equiv = m.get_equivalent(*auxiliary_cols)
  return equiv, df


def get_unique_prefix(df):
  prefix = 'meterstick_tmp:'
  while any(str(c).startswith(prefix) for c in df.columns):
    prefix += ':'
  return prefix


def add_auxiliary_cols(auxiliary_cols,
                       df: Optional[pd.DataFrame] = None,
                       prefix: str = ''):
  """Parses auxiliary_cols from Metric.get_auxiliary_cols and adds them to df.

  Some Metrics can be expressed by simpler Metrics. For example, Dot(x, y) is
  equivalent to Sum(x * y). However, column `x * y` doesn't necessarily exist in
  df so we need to create it. Here we compute the column `x * y` and add it to
  df in-place.

  Args:
    auxiliary_cols: A list of tuples. Each tuple represents an auxiliary column
      that needs to be added. The tuple must have three elements. The second
      element stands for the operator while the rest are the inputs. For
      example, ('x', '*', 'y') means we need to add an auxiliary column that
      equals to df.x * df.y. The inputs can also be constants. ('x', '/', 2)
      stands for an auxiliary column that equals to df.x / 2. The inputs can
      also be another tuple that stands for an auxiliary column. For example,
      (('x', '+', 'y'), '-', 'z') stands for a column equals df.x + df.y - df.z.
      The nesting can be indefinite. The operator can be one of ('+', '-', '*' ,
      '/', '**') or a function that takes two args and returns one column.
    df: The dataframe we compute on. We adds the auxiliary columns to it
      in-place. If it's None, then we skip the computation.
    prefix: The prefix added to the names of auxiliary columns so they won't
      collide with existing columns.

  Returns:
    df with auxiliary columns added.
    The names of the auxiliary columns added.
  """
  auxiliary_col_names = []
  for c in auxiliary_cols or ():
    name, res = parse_auxiliary_col(c, df)
    name = prefix + name
    if df is not None:
      if name == prefix + 'lambda':
        while name in df:
          name += '_'
      if name not in df:
        df[name] = res
    auxiliary_col_names.append(name)
  return df, auxiliary_col_names


def parse_auxiliary_col(auxiliary_col, df: Optional[pd.DataFrame] = None):
  """Parses an auxiliary_col and computes it.

  Args:
    auxiliary_col: One element of the auxiliary_cols in add_auxiliary_cols().
    df: The same df in add_auxiliary_cols().

  Returns:
    The generated name of the auxiliary column. Note that the name is also a
    valid SQL expression. When called in compute_on(), the name is used as the
    column name of the auxiliary column while when called in compute_on_sql(),
    it's directly used to construct the SQL query.
    The result of the auxiliary column.
  """
  if isinstance(auxiliary_col, str):
    return auxiliary_col, df[auxiliary_col] if df is not None else None
  if isinstance(auxiliary_col, (float, int)):
    return str(auxiliary_col), auxiliary_col
  if not isinstance(auxiliary_col, tuple):
    raise ValueError('auxiliary_col must be a tuple/str/number but got %s.' %
                     auxiliary_col)
  if len(auxiliary_col) != 3:
    raise ValueError('auxiliary_col must be length-3 but got %s.' %
                     auxiliary_col)
  col0, fn, col1 = auxiliary_col
  name0, col0 = parse_auxiliary_col(col0, df)
  name1, col1 = parse_auxiliary_col(col1, df)
  if fn in ('+', '*'):
    (name0, col0), (name1, col1) = sorted(((name0, col0), (name1, col1)))
  if callable(fn):
    name = 'lambda'
    res = fn(col0, col1) if df is not None else None
  elif fn == '+':
    name = '(%s + %s)' % (name0, name1)
    res = col0 + col1 if df is not None else None
  elif fn == '-':
    name = '(%s - %s)' % (name0, name1)
    res = col0 - col1 if df is not None else None
  elif fn == '*':
    name = '(%s * %s)' % (name0, name1)
    res = col0 * col1 if df is not None else None
  elif fn == '/':
    name = '(%s / %s)' % (name0, name1)
    res = col0 / col1 if df is not None else None
  elif fn == '**':
    name = 'POWER(%s, %s)' % (name0, name1)
    res = col0**col1 if df is not None else None
  return name, res


def pcollection_to_df_via_file_io(
    pcol, pipeline, output_dir: str, cleanup=False
) -> pd.DataFrame:
  """Evaluates a PCollection, saves result, reads back to a DataFrame.

  Args:
    pcol: A PCollection instance for evaluation.
    pipeline: An Apache Beam pipeline that holds pcol. We assume pipeline.run()
      will execute the pipeline.
    output_dir: A folder where we saves the result of pcol.
    cleanup: If to delete the files we create when done.

  Returns:
    The result of pcol.
  """
  # pylint: disable=g-import-not-at-top
  from apache_beam.dataframe import convert
  from apache_beam.dataframe import io
  # pylint: enable=g-import-not-at-top

  now = datetime.datetime.now()
  filename = f'Meterstick pcollection_to_df_via_file_io at {now}'
  output_path = os.path.join(output_dir, filename)
  pcol = convert.to_dataframe(pcol, label=f'to df at {now}')
  pcol = io.to_csv(pcol, output_path, f'to csv at {now}', index=False)
  pipeline.run()

  res = []
  for f in glob.glob(f'{output_path}-*'):  # output is sharded
    res.append(pd.read_csv(f))
    if cleanup:
      os.remove(f)
  return pd.concat(res, ignore_index=True)
