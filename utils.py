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
"""Utils functions for things like DataFrame manipulation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Iterable, List, Optional, Text, Union
import pandas as pd


def get_name(obj):
  return getattr(obj, 'name', str(obj))


class CacheKey():
  """Represents a cache_key used in the computation of Metrics.

  During the computation of a Metric, we often use a key to cache results. It
  either comes from users or a default value. For caching to be valid, one cache
  key needs to correpond to the same DataFrame AND split_by, but it can still
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
  split_by infomation. It cannot just be 'foo'.
  Similarly, internal cache key needs to encode the 'where' arg in Metric too or
  PercentChange(sumx, where='grp == 1') - PercentChange(sumx, where='grp == 0')
  would always return 0.
  The last piece is the slice infomation. It matters when a Metric is not
  vectorized.
  CacheKey is the internal key we use in cache. It encodes split_by and where.
  A cache_key users provide is first converted to a CacheKey.

  Attributes:
    key: The raw cache_key user provides, or the default key.
    where: The filters to apply to the input DataFrame.
    split_by: The columns to split by.
    slice_val: When a Metric is not vectorized, it computes on a slice of data.
      Such computation shoulnd't use the same cache key with the vecoreized
      Metrics. slice_val is a dict of the value of the split_by columns of the
      data slice with the keys being columns in split_by.
    all_filters: The merge of all 'where' conditions that can be passed to
      df.query().
  """

  def __init__(self,
               key,
               where: Optional[Union[Text, Iterable[Text]]] = None,
               split_by: Optional[Union[Text, List[Text]]] = None,
               slice_val=None):
    """Wraps cache_key, split_by, filters and slice infomation.

    Args:
      key: A raw key or a CacheKey. If it's a CacheKey, we unwrap it and extend
        its split_by and where.
      where: The filters to apply to the input DataFrame.
      split_by: The columns to split by.
      slice_val: An ordered tuple of key, value pair in slice_val.
    """
    split_by = [split_by] if isinstance(split_by, str) else split_by or ()
    where = [where] if isinstance(where, str) else where or []
    if isinstance(key, CacheKey):
      self.key = key.key
      self.where = key.where.copy()
      if where is not None:
        self.where.update(where)
      self.split_by = key.split_by[:]
      self.extend(split_by)
      self.slice_val = slice_val or {}
      for k, v in key.slice_val:
        if k in self.slice_val and self.slice_val[k] != v:
          raise ValueError('Incompatible data slice values!')
        self.slice_val[k] = v
    else:
      self.key = key
      self.where = set(where)
      self.split_by = tuple(split_by)
      self.slice_val = slice_val or {}
    self.slice_val = tuple(sorted(self.slice_val.items()))

  def extend(self, split_by: Union[Text, List[Text]]):
    split_by = [split_by] if isinstance(split_by, str) else split_by
    new_split_by = [s for s in split_by if s not in self.split_by]
    self.split_by = tuple(list(self.split_by) + new_split_by)
    return self

  def add_filters(self, filters):
    self.where.update(filters)
    return self

  @property
  def all_filters(self):
    return ' & '.join('(%s)' % c for c in sorted(tuple(self.where))) or None

  def includes(self, other):
    """Decides if self is extended from other."""
    return (isinstance(other, CacheKey) and self.key == other.key and
            self.where == other.where and self.slice_val == other.slice_val and
            self.split_by[:len(other.split_by)] == other.split_by)

  def __eq__(self, other):
    return self.includes(other) and self.split_by == other.split_by

  def __hash__(self):
    return hash((self.key, self.split_by, self.slice_val,
                 tuple(sorted(tuple(self.where)))))

  def __repr__(self):
    return 'key: %s, split_by: %s, slice: %s, where: %s' % (
        self.key, self.split_by, self.slice_val, self.where)


def is_tmp_key(key):
  if isinstance(key, str):
    return key == '_RESERVED'
  if isinstance(key, tuple) and len(key) > 1 and isinstance(key[0], str):
    return key[0] == '_RESERVED'
  if isinstance(key, CacheKey):
    return is_tmp_key(key.key)
  return False


def adjust_slices_for_loo(bucket_res: pd.Series,
                          split_by: Optional[List[Text]] = None):
  """Corrects the slices in the bucketized result.

  Jackknife has a precomputation step where we precompute leave-one-out (LOO)
  results for Sum, Count and Mean. The idea is that, for example, for Sum, we
  can get the LOO result by subtracting df.groupby([unit, groupby columns])
  from df.groupby([groupby columns]) where unit is the column to Jackknife on
  and groupby columns are optional and have two parts, both are optional too.
  The first part comes from Operations. For example,
  PercentChange(cond, base, Sum('x')).compute_on(df) internally computes
  Sum('x').compute_on(df, cond). The second part is just the split_by passed
  to Jackknife(...).compute_on(df, split_by). In other words, the groupby
  columns are [cond] + split_by for
  Jackknife(PercentChange(cond, base, Sum('x'))).compute_on(df, split_by).
  The issue is for the bucketized sum/count, some slices will be missing for
  sparse data and hence missing when we subtract it from the total, while some
  slices should not be in the final LOO result because they are only present
  in this bucket. For example, for data
    unit  grp  cond  X
     1     A   foo   10
     2     A   foo   20
     2     A   bar   30
     2     B   bar   40,
  to compute
  Jackknife('unit', PercentChange('cond', 'foo', Sum('X'))
    ).compute_on(data, 'grp'),
  the leave-unit-1-out result should only have slices A * foo and A * bar
  because
  1. grp B never appears in unit 1 so we shouldn't calculate leave-unit-1-out
    for it.
  2. slice A * bar should be added because it's in the sum of leave-unit-1-out
    data.
  And leave-unit-2-out result should have slices A * foo, A * bar and B * bar
  because
  1. A * foo is in the leave-unit-2-out data and grp A is in the unit-2 data.
  2. A * bar and B * bar are in the unit-2 data, though their jackknife standard
    errors will be NA because they are not in other units of data.
  In summary, to get the slices that should be present for unit i,
  1. Find all the slices in the leave-unit-i-out data.
  2. Discard the slices whose split_by part is not in the unit-i data.
  3. All slices in unit-i data need to be counted too.
  In practice, we only enforce #1 and #2, namely, if a slice is expected ONLY
  because it's in unit-i, we might not include it. The reason is that it only
  happens when the slice appears in only one unit so its degrees of freedom is
  1. In operations.py we make sure the standard error is always NA when dof is
  1, so what we return doesn't matter. The slice won't be lost in the final
  result either because we will join the standard error to the point estimate
  and the latter has all slices. Nonetheless, for cases that no adjustment is
  needed, the original data is returned, so #3 still holds.

  Args:
    bucket_res: The first level of its indexes is the unit to Jackknife on,
      followed by split_by, then levels added by Operations, if any.
    split_by: The list of column(s) from Jackknife().compute_on(df, split_by).

  Returns:
    A pd.Series that has the same index names as bucket_res, but with some
    levels removed and/or added.
  """
  slicing = bucket_res.index.names[1:]
  operation_lvl = slicing[len(split_by or ()):]
  if not operation_lvl:
    return bucket_res

  unit_slice_ct = bucket_res.groupby(bucket_res.index.names).size()
  total_ct = unit_slice_ct.groupby(slicing).sum()
  # We need all combinations of slices.
  unit_slice_ct = unit_slice_ct.reindex(
      pd.MultiIndex.from_product(unit_slice_ct.index.levels), fill_value=0)
  if len(unit_slice_ct) == len(bucket_res):
    return bucket_res  # No missing slice.
  slices_in_other_units = total_ct - unit_slice_ct
  slices_in_other_units = slices_in_other_units.index[slices_in_other_units > 0]
  if slices_in_other_units.names != bucket_res.index.names:
    slices_in_other_units = slices_in_other_units.reorder_levels(
        bucket_res.index.names)
  to_keep = slices_in_other_units.droplevel(
      operation_lvl).isin(
          bucket_res.index.droplevel(operation_lvl))
  return bucket_res.reindex(slices_in_other_units[to_keep], fill_value=0)


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
    df = pd.DataFrame(df.stack(0, dropna=False)).T
  else:
    df = pd.concat([df.loc[n] for n in names], 1, keys=names, names=['Metric'])
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
        res.index.set_levels(
            map(name_tmpl.format, res.index.levels[0]), 0, inplace=True)
      else:
        res.index = pd.Index(
            map(name_tmpl.format, res.index), name=res.index.name)
    else:
      if len(res.columns.names) > 1:
        res.columns.set_levels(
            map(name_tmpl.format, res.columns.levels[0]), 0, inplace=True)
      else:
        res.columns = map(name_tmpl.format, res.columns)
  return res


class MaybeBadSqlModeError(Exception):
  """An Error used in compute_on_sql(), when a better mode might exist."""

  def __init__(self, better_mode=None, use_batch_size=False, batch_size=None):
    self.better_mode = better_mode
    self.use_batch_size = use_batch_size
    self.batch_size = batch_size
    if batch_size:
      msg = 'reducing the batch_size. Current batch_size is %s.' % batch_size
    elif use_batch_size:
      msg = "compute_on_sql(..., mode='mixed', batch_size=an integer)."
    else:
      msg = "compute_on_sql(..., mode='%s')." % (better_mode or 'mixed')
    super(MaybeBadSqlModeError, self).__init__(
        "Please see the root cause of the failure above. If it's caused "
        'by the query being too large/complex, you can try %s' % msg)
