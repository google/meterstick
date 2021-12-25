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
"""Base classes for Meterstick."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import itertools
from typing import Any, Callable, List, Optional, Sequence, Text, Union

from meterstick import sql
from meterstick import utils
import numpy as np
import pandas as pd


def compute_on(df,
               split_by=None,
               melted=False,
               return_dataframe=True,
               cache_key=None):
  # pylint: disable=g-long-lambda
  return lambda x: x.compute_on(df, split_by, melted, return_dataframe,
                                cache_key)
  # pylint: enable=g-long-lambda


def compute_on_sql(
    table,
    split_by=None,
    execute=None,
    melted=False,
    mode=None,
    cache_key=None,
    **kwargs
):
  """A wrapper that metric | compute_on_sql() === metric.compute_on_sql()."""
  # pylint: disable=g-long-lambda
  return lambda m: m.compute_on_sql(
      table,
      split_by,
      execute,
      melted,
      mode,
      cache_key,
      **kwargs
  )
  # pylint: enable=g-long-lambda


def to_sql(table, split_by=None):
  return lambda metric: metric.to_sql(table, split_by)


def get_extra_idx(metric):
  """Collects the extra indexes added by Operations for the metric tree.

  Args:
    metric: A Metric instance.

  Returns:
    A tuple of column names which are just the index of metric.compute_on(df).
  """
  extra_idx = metric.extra_index[:]
  children_idx = [
      get_extra_idx(c) for c in metric.children if isinstance(c, Metric)
  ]
  if len(set(children_idx)) > 1:
    raise ValueError('Incompatible indexes!')
  if children_idx:
    extra_idx += list(children_idx[0])
  return tuple(extra_idx)


def get_global_filter(metric):
  """Collects the filters that can be applied globally to the Metric tree."""
  global_filter = sql.Filters()
  if metric.where:
    global_filter.add(metric.where)
  children_filters = [
      set(get_global_filter(c))
      for c in metric.children
      if isinstance(c, Metric)
  ]
  if children_filters:
    shared_filter = set.intersection(*children_filters)
    global_filter.add(shared_filter)
  return global_filter


def is_operation(m):
  """We can't use isinstance because of loop dependancy."""
  return isinstance(m, Metric) and m.children and not isinstance(
      m, (MetricList, CompositeMetric))


class Metric(object):
  """Core class of Meterstick.

  A Metric is defined broadly in Meterstick. It could be a routine metric like
  CTR, or an operation like Bootstrap. As long as it taks a DataFrame and
  returns a number or a pd.Series, it can be treated as a Metric.

  The relations of methods of Metric are
        <------------------------------------------------------compute_on-------------------------------------------------------->
        <------------------------------compute_through---------------------------->                                              |
        |                              <-------compute_slices------->             |                                              |
        |                              |-> slice1 -> compute |      |             |                                              |
  df -> df.query(where) -> precompute -|-> slice2 -> compute | -> concat  -> postcompute -> manipulate -> final_compute -> flush_tmp_cache  # pylint: disable=line-too-long
                                       |-> ...
  In summary, compute() operates on a slice of data. precompute(),
  postcompute(), compute_slices(), compute_through() and final_compute() operate
  on the whole data. manipulate() does common data manipulation like melting
  and cleaning. Caching is handled in compute_on().
  If Metric has children Metrics, then compute_slices is further decomposed to
  compute_children() -> compute_on_children(), if they are implemented. For such
  Metrics, the decomposition makes the 'mixed' mode of compute_on_sql() simple.
  The `mixed` mode computes children in SQL and the rest in Python, so as long
  as a compute_children_sql() is implemented and has a similar return to
  compute_children(), the compute_on_children() is reused and the `mixed` mode
  automatically works.

  Depending on your case, you can overwrite most of them, but we suggest you NOT
  to overwrite compute_on because it might mess up the caching mechanism. Here
  are some rules to help you to decide.
  1. If your Metric has no vectorization over slices, overwrite compute(). To
  overwrite, you can either create a new class inheriting from Metric or just
  pass a lambda function into Metric.
  2. If you have vectorization logic over slices, overwrite compute_slices().
  See Sum() for an example.
  3. As compute() operates on a slice of data, it doesn't have access to the
  columns to split_by and the index value of the slice. If you need them, check
  out compute_with_split_by(). See Jackknife for a real example.
  4. The data passed into manipulate() should be a number, a pd.Series, or a
    wide/unmelted pd.DataFrame.

  It's possible to cache your result. However, as DataFrame is mutable, it's
  slow to hash it (O(shape) complexity). To avoid hashing, for most cases you'd
  rely on our MetricList() and CompositeMetric() which we know in one round
  of their computation, the DataFrame doesn't change. Or if you have to run
  many rounds of computation on the same DataFrame, you can directly assign a
  cache_key in compute_on(), then it's your responsibility to ensure
  same key always corresponds to the same DataFrame and split_by.

  Your Metric shouldn't rely on the index of the input DataFrame. We might
  set/reset the index during the computation so put all the information you need
  in the columns.

  Attributes:
    name: Name of the Metric.
    children: An iterable of Metric(s) this Metric based upon.
    cache: A dict to store cached results.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    precompute: A function. See the workflow chart above for its behavior.
    compute: A function. See the workflow chart above for its behavior.
    postcompute: A function. See the workflow chart above for its behavior.
    compute_slices: A function. See the workflow chart above for its behavior.
    final_compute: A function. See the workflow chart above for its behavior.
    extra_split_by: Used by Operation. See the doc there.
    extra_index: Used by Operation. See the doc there.
    computable_in_pure_sql: If the Metric can be completely computed in SQL. For
      example, all Models can't.
    name_tmpl: Used by Metrics that have children. It's applied to children's
      names in the output.
    cache_key: The key currently being used in computation.
    tmp_cache_keys: The set to track what temporary cache_keys are used during
      computation when default caching is enabled. When computation is done, all
      the keys in tmp_cache_keys are flushed.
  """
  RESERVED_KEY = '_RESERVED'

  def __init__(self,
               name: Text,
               children: Optional[Union['Metric', Sequence[Union['Metric', int,
                                                                 float]]]] = (),
               where: Optional[Union[Text, Sequence[Text]]] = None,
               name_tmpl=None,
               precompute=None,
               compute: Optional[Callable[[pd.DataFrame], Any]] = None,
               postcompute=None,
               compute_slices=None,
               final_compute=None):
    self.name = name
    self.cache = {}
    self.cache_key = None
    self.children = [children] if isinstance(children,
                                             Metric) else children or []
    if isinstance(where, List):
      where = ' and '.join(where)
    self.where = where
    self.extra_split_by = []
    self.extra_index = []
    self.computable_in_pure_sql = True
    self.name_tmpl = name_tmpl
    if precompute:
      self.precompute = precompute
    if compute:
      self.compute = compute
    if postcompute:
      self.postcompute = postcompute
    if compute_slices:
      self.compute_slices = compute_slices
    if final_compute:
      self.final_compute = final_compute
    self.tmp_cache_keys = set()

  def compute_with_split_by(self,
                            df,
                            split_by: Optional[List[Text]] = None,
                            slice_value=None):
    del split_by, slice_value  # In case users need them in derived classes.
    return self.compute(df)

  def compute_slices(self, df, split_by: Optional[List[Text]] = None):
    """Applies compute() to all slices. Each slice needs a unique cache_key."""
    if self.children:
      try:
        df = self.compute_children(df, split_by + self.extra_split_by)
        return self.compute_on_children(df, split_by)
      except NotImplementedError:
        pass
    if split_by:
      # Adapted from http://esantorella.com/2016/06/16/groupby. This is faster
      # than df.groupby(split_by).apply(self.compute).
      slices = []
      result = []
      # Different DataFrames need to have different cache_keys. Here as we split
      # the df so each slice need to has its own key. And we need to make sure
      # the key is recovered so when we continue to compute other Metrics that
      # might be vectoriezed, the key we use is the one for the whole df.
      for df_slice, slice_i in self.split_data(df, split_by):
        cache_key = self.cache_key
        slice_i_iter = slice_i if isinstance(slice_i, tuple) else [slice_i]
        self.cache_key = self.wrap_cache_key(
            cache_key or self.RESERVED_KEY,
            slice_val=dict(zip(split_by, slice_i_iter)))
        try:
          result.append(self.compute_with_split_by(df_slice, split_by, slice_i))
          slices.append(slice_i)
        finally:
          self.cache_key = cache_key
      if isinstance(result[0], (pd.Series, pd.DataFrame)):
        try:
          return pd.concat(result, keys=slices, names=split_by, sort=False)
        except ValueError:
          if len(split_by) == 1:
            # slices are tuples so pd unpacked it then the lengths didn't match.
            split = split_by[0]
            for r, s in zip(result, slices):
              r[split] = [s] * len(r)
              r.set_index(split, append=True, inplace=True)
            res = pd.concat(result, sort=False)
            if len(res.index.names) > 1:
              res = res.reorder_levels(np.roll(res.index.names, 1))
            return res
      else:
        if len(split_by) == 1:
          ind = pd.Index(slices, name=split_by[0])
        else:
          ind = pd.MultiIndex.from_tuples(slices, names=split_by)
        return pd.Series(result, index=ind)
    else:
      # Derived Metrics might do something in split_data().
      df, _ = next(self.split_data(df, split_by))
      return self.compute_with_split_by(df)

  def compute_children(self, df, split_by):
    raise NotImplementedError

  def compute_on_children(self, children, split_by):
    """Computes the return using the result returned by children Metrics.

    Args:
      children: The return of compute_children() or compute_children_sql().
      split_by: Something can be passed into df.group_by().

    Returns:
      Almost the final result. Only some manipulations are still needed.
    """
    if len(self.children) != 1:
      raise ValueError('We can only handle one child Metric!')
    if not split_by:
      return self.compute(children)
    result = []
    slices = []
    for d, i in self.split_data(children, split_by):
      result.append(self.compute(d))
      slices.append(i)
    return pd.concat(result, keys=slices, names=split_by, sort=False)

  @staticmethod
  def split_data(df, split_by=None):
    if not split_by:
      yield df, None
    else:
      keys, indices = zip(*df.groupby(split_by).groups.items())
      for k, idx in zip(keys, indices):
        yield df.loc[idx.unique()], k

  def compute_through(self, df, split_by: Optional[List[Text]] = None):
    """Precomputes df -> split df and apply compute() -> postcompute."""
    df = df.query(self.where) if df is not None and self.where else df
    res = self.precompute(df, split_by)
    res = self.compute_slices(res, split_by)
    return self.postcompute(res, split_by)

  def compute_on(self,
                 df: pd.DataFrame,
                 split_by: Optional[Union[Text, List[Text]]] = None,
                 melted: bool = False,
                 return_dataframe: bool = True,
                 cache_key: Any = None):
    """Key API of Metric.

    Wraps computing logic with caching.

    This is what you should call to use Metric. It's compute_through +
    final_compute + caching. As caching is the shared part of Metric, we suggest
    you NOT to overwrite this method. Overwriting compute_slices and/or
    final_compute should be enough. If not, contact us with your use cases.

    Args:
      df: The DataFrame to compute on.
      split_by: Something can be passed into df.group_by().
      melted: Whether to transform the result to long format.
      return_dataframe: Whether to convert the result to DataFrame if it's not.
        If False, it could still return a DataFrame.
      cache_key: What key to use to cache the df. You can use anything that can
        be a key of a dict except '_RESERVED' and tuples like ('_RESERVED', ..).

    Returns:
      Final result returned to user. If split_by, it's a pd.Series or a
      pd.DataFrame, otherwise it could be a base type.
    """
    need_clean_up = True
    try:
      split_by = [split_by] if isinstance(split_by, str) else split_by or []
      if cache_key is not None:
        cache_key = self.wrap_cache_key(cache_key, split_by)
        if self.in_cache(cache_key):
          need_clean_up = False
          raw_res = self.get_cached(cache_key)
          res = self.manipulate(raw_res, melted, return_dataframe)
          res = self.final_compute(res, melted, return_dataframe, split_by, df)
          return res
        else:
          self.cache_key = cache_key
          raw_res = self.compute_through(df, split_by)
          self.save_to_cache(cache_key, raw_res)
          if utils.is_tmp_key(cache_key):
            self.tmp_cache_keys.add(cache_key)
      else:
        raw_res = self.compute_through(df, split_by)

      res = self.manipulate(raw_res, melted, return_dataframe)
      return self.final_compute(res, melted, return_dataframe, split_by, df)

    finally:
      if need_clean_up:
        self.flush_tmp_cache()

  def precompute(self, df, split_by):
    del split_by  # Useful in derived classes.
    return df

  def postcompute(self, df, split_by):
    del split_by  # Useful in derived classes.
    return df

  def final_compute(self, res, melted, return_dataframe, split_by, df):
    del melted, return_dataframe, split_by, df  # Useful in derived classes.
    return res

  def manipulate(self,
                 res: pd.Series,
                 melted: bool = False,
                 return_dataframe: bool = True,
                 apply_name_tmpl=None):
    """Common adhoc data manipulation.

    It does
    1. Converts res to a DataFrame if asked.
    2. Melts res to long format if asked.
    3. Removes redundant index levels in res.
    4. Apply self.name_tmpl to the output name or columns if asked.

    Args:
      res: Returned by compute_through(). Usually a DataFrame, but could be a
        pd.Series or a base type.
      melted: Whether to transform the result to long format.
      return_dataframe: Whether to convert the result to DataFrame if it's not.
        If False, it could still return a DataFrame if the input is already a
        DataFrame.
      apply_name_tmpl: If to apply name_tmpl to the result. For example, in
        Distribution('country', Sum('X')).compute_on(df),
        we first compute Sum('X').compute_on(df, 'country'), then normalize, and
        finally apply the name_tmpl 'Distribution of {}' to all column names.

    Returns:
      Final result returned to user. If split_by, it's a pd.Series or a
      pd.DataFrame, otherwise it could be a base type.
    """
    res = copy.copy(res)  # Avoid changing the result in cache.
    if isinstance(res, pd.Series):
      res.name = self.name
    if not isinstance(res, pd.DataFrame) and return_dataframe:
      res = self.to_dataframe(res)
    if melted:
      res = utils.melt(res)
    if apply_name_tmpl is None and is_operation(self):
      apply_name_tmpl = True
    if apply_name_tmpl:
      res = utils.apply_name_tmpl(self.name_tmpl, res, melted)
    return utils.remove_empty_level(res)

  @staticmethod
  def group(df, split_by=None):
    return df.groupby(split_by) if split_by else df

  def to_dataframe(self, res):
    if isinstance(res, pd.DataFrame):
      return res
    elif isinstance(res, pd.Series):
      return pd.DataFrame(res)
    return pd.DataFrame({self.name: [res]})

  def wrap_cache_key(self, key, split_by=None, where=None, slice_val=None):
    if key is None:
      return None
    if where is None and self.cache_key:
      where = self.cache_key.where
    return utils.CacheKey(key, where or self.where, split_by, slice_val)

  def save_to_cache(self, key, val, split_by=None):
    key = self.wrap_cache_key(key, split_by)
    val = val.copy() if isinstance(val, (pd.Series, pd.DataFrame)) else val
    self.cache[key] = val

  def in_cache(self, key, split_by=None, where=None, exact=True):
    key = self.wrap_cache_key(key, split_by, where)
    if exact:
      return key in self.cache
    else:
      return any(k.includes(key) for k in self.cache)

  def get_cached(self,
                 key=None,
                 split_by=None,
                 where=None,
                 exact=True,
                 return_key=False):
    """Retrieves result from cache if there is an unique one."""
    key = self.wrap_cache_key(key, split_by, where)
    if key in self.cache:
      return (key, self.cache[key]) if return_key else self.cache[key]
    elif not exact:
      matches = {k: v for k, v in self.cache.items() if k.includes(key)}
      if len(matches) > 1:
        raise ValueError('Muliple fuzzy matches found!')
      elif matches:
        return tuple(matches.items())[0] if return_key else tuple(
            matches.values())[0]

  def flush_cache(self,
                  key=None,
                  split_by=None,
                  where=None,
                  recursive=True,
                  prune=True):
    """If prune, stops when the cache seems to be flushed already."""
    key = self.wrap_cache_key(key, split_by, where)
    if prune:
      if (key is not None and not self.in_cache(key)) or not self.cache:
        return
    if key is None:
      self.cache = {}
    elif self.in_cache(key):
      del self.cache[key]
    if recursive:
      self.flush_children(key, split_by, where, recursive, prune)

  def flush_children(self,
                     key=None,
                     split_by=None,
                     where=None,
                     recursive=True,
                     prune=True):
    for m in self.children:
      if isinstance(m, Metric):
        m.flush_cache(key, split_by, where, recursive, prune)

  def flush_tmp_cache(self):
    """Flushes all the temporary caches when a Metric tree has been computed.

    A Metric and all the descendants form a tree. When a computation is started
    from a MetricList or CompositeMetric, we know the input DataFrame is not
    going to change in the computation. So even if user doesn't ask for caching,
    we still enable it, but we need to clean things up when done. As the results
    need to be cached until all Metrics in the tree have been computed, we
    should only clean up at the end of the computation of the entry/top Metric.
    We recognize the top Metric by looking at the cache_key. All descendants
    will have it assigned as RESERVED_KEY but the entry Metric's will be None.
    """
    if self.cache_key is None:  # Entry point of computation
      for m in self.traverse():
        while m.tmp_cache_keys:
          m.flush_cache(m.tmp_cache_keys.pop(), recursive=False)
    self.cache_key = None

  def traverse(self, include_self=True):
    ms = [self] if include_self else list(self.children)
    while ms:
      m = ms.pop(0)
      if isinstance(m, Metric):
        ms += list(m.children)
        yield m

  def compute_on_sql(
      self,
      table,
      split_by=None,
      execute=None,
      melted=False,
      mode=None,
      cache_key=None,
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

    Returns:
      A pandas DataFrame. It's the computeation of self in SQL.
    """
    split_by = [split_by] if isinstance(split_by, str) else split_by or []
    need_clean_up = True
    try:
      if cache_key is not None:
        cache_key = self.wrap_cache_key(cache_key, split_by)
        if self.in_cache(cache_key):
          need_clean_up = False
          res = self.get_cached(cache_key)
          res = self.manipulate(res, melted, apply_name_tmpl=False)
          return self.final_compute(res, melted, True, split_by, table)
        else:
          self.cache_key = cache_key
          raw_res = self.compute_through_sql(table, split_by, execute, mode)
          res = raw_res
          # For simple metrics we save a pd.Series to be consistent with
          # compute_on().
          if isinstance(raw_res, pd.DataFrame
                       ) and raw_res.shape[1] == 1 and not is_operation(self):
            raw_res = raw_res.iloc[:, 0]
          self.save_to_cache(cache_key, raw_res)
          if utils.is_tmp_key(cache_key):
            self.tmp_cache_keys.add(cache_key)
      else:
        res = self.compute_through_sql(table, split_by, execute, mode)
      res = self.manipulate(res, melted, apply_name_tmpl=False)
      return self.final_compute(res, melted, True, split_by, table)
    finally:
      if need_clean_up:
        self.flush_tmp_cache()

  def compute_through_sql(self, table, split_by, execute, mode):
    """Delegeates the computation to different modes."""
    if mode not in (None, 'sql', 'mixed', 'magic'):
      raise ValueError('Mode %s is not supported!' % mode)
    if not self.children and not self.computable_in_pure_sql:
      raise ValueError('%s is not computable in SQL.' % self.name)
    if mode in (None, 'sql') or not self.children:
      if self.all_computable_in_pure_sql():
        try:
          return self.compute_on_sql_sql_mode(table, split_by, execute)
        except Exception as e:  # pylint: disable=broad-except
          if not self.children:
            raise
          raise utils.MaybeBadSqlModeError('mixed') from e
      elif mode == 'sql':
        raise ValueError('%s is not computable in pure SQL.' % self.name)
    if self.where:
      table = sql.Sql(sql.Column('*', auto_alias=False), table, self.where)
    return self.compute_on_sql_mixed_mode(table, split_by, execute, mode)

  def all_computable_in_pure_sql(self, include_self=True):
    """If the whole Metric tree is computable in SQL."""
    for m in self.traverse(include_self):
      if not m.computable_in_pure_sql:
        return False
    return True

  def compute_on_sql_sql_mode(self,
                              table,
                              split_by=None,
                              execute=None):
    """Executes the query from to_sql() and process the result."""
    query = self.to_sql(table, split_by)
    res = execute(str(query))
    extra_idx = list(get_extra_idx(self))
    indexes = split_by + extra_idx if split_by else extra_idx
    columns = [a.alias_raw for a in query.groupby.add(query.columns)]
    columns[:len(indexes)] = indexes
    res.columns = columns
    # We replace '$' with 'macro_' in the generated SQL. To recover the names,
    # we cannot just replace 'macro_' with '$' because 'macro_' might be in the
    # orignal names. We can get the index names so we can recover them but
    # unfortunately it's not as easy to recover metric/column names, so for the
    # metrics we just replace 'macro_' with '$'.
    if indexes:
      res.set_index(indexes, inplace=True)
    res.columns = [c.replace('macro_', '$') for c in res.columns]
    if split_by:  # Use a stable sort.
      res.sort_values(split_by, kind='mergesort', inplace=True)
    return res

  def to_sql(self,
             table,
             split_by: Optional[Union[Text, List[Text]]] = None):
    """Generates SQL query for the metric."""
    global_filter = get_global_filter(self)
    indexes = sql.Columns(split_by).add(get_extra_idx(self))
    with_data = sql.Datasources()
    if isinstance(table, sql.Sql) and table.with_data:
      table = copy.deepcopy(table)
      with_data = table.with_data
      table.with_data = None
    if not sql.Datasource(table).is_table:
      table = with_data.add(sql.Datasource(table, 'Data'))
    query, with_data = self.get_sql_and_with_clause(table,
                                                    sql.Columns(split_by),
                                                    global_filter, indexes,
                                                    sql.Filters(), with_data)
    query.with_data = with_data
    return query

  def get_sql_and_with_clause(self, table: sql.Datasource,
                              split_by: sql.Columns, global_filter: sql.Filters,
                              indexes: sql.Columns, local_filter: sql.Filters,
                              with_data: sql.Datasources):
    """Gets the SQL query for metric and its WITH clause separately.

    Args:
      table: The table we want to query from.
      split_by: The columns that we use to split the data. Note it could be
        different to the split_by passed to the root Metric. For example, in the
        call of get_sql(AbsoluteChange('platform', 'tablet', Distribution(...)),
        'country') the split_by Distribution gets will be ('country',
        'platform') because AbsoluteChange adds an extra index.
      global_filter: The filters that can be applied to the whole Metric tree.It
        will be passed down all the way to the leaf Metrics and become the WHERE
        clause in the query of root table.
      indexes: The columns that we shouldn't apply any arithmetic operation. For
        most of the time they are the indexes you would see in the result of
        metric.compute_on(df).
      local_filter: The filters that have been accumulated as we walk down the
        metric tree. It's the collection of all filters of the ancestor Metrics
        along the path so far. More filters might be added as we walk down to
        the leaf Metrics. It's used there as inline filters like
        IF(local_filter, value, NULL).
      with_data: A global variable that contains all the WITH clauses we need.
        It's being passed around and Metrics add the datasources they need to
        it. It's added to the SQL instance eventually in get_sql() once we have
        walked through the whole metric tree.

    Returns:
      The SQL instance for metric, without the WITH clause component.
      The global with_data which holds all datasources we need in the WITH
        clause.
    """
    raise NotImplementedError('SQL generator is not implemented for %s.' %
                              type(self))

  def compute_on_sql_mixed_mode(self,
                                table,
                                split_by,
                                execute,
                                mode=None):
    """Computes the child in SQL and the rest in Python."""
    children = self.compute_children_sql(table, split_by, execute, mode)
    return self.compute_on_children(children, split_by)

  def compute_children_sql(self, table, split_by, execute, mode=None):
    """The return should be similar to compute_children()."""
    children = []
    for c in self.children:
      if not isinstance(c, Metric):
        children.append(c)
      else:
        children.append(
            c.compute_on_sql(table, split_by + self.extra_split_by, execute,
                             False, mode, self.cache_key or self.RESERVED_KEY))
    return children[0] if len(self.children) == 1 else children

  def __or__(self, fn):
    """Overwrites the '|' operator to enable pipeline chaining."""
    return fn(self)

  def __add__(self, other):
    return CompositeMetric(lambda x, y: x + y, '{} + {}', (self, other))

  def __radd__(self, other):
    return CompositeMetric(lambda x, y: x + y, '{} + {}', (other, self))

  def __sub__(self, other):
    return CompositeMetric(lambda x, y: x - y, '{} - {}', (self, other))

  def __rsub__(self, other):
    return CompositeMetric(lambda x, y: x - y, '{} - {}', (other, self))

  def __mul__(self, other):
    return CompositeMetric(lambda x, y: x * y, '{} * {}', (self, other))

  def __rmul__(self, other):
    return CompositeMetric(lambda x, y: x * y, '{} * {}', (other, self))

  def __neg__(self):
    return CompositeMetric(lambda x, _: -x, '-{}', (self, -1))

  def __div__(self, other):
    return CompositeMetric(lambda x, y: x / y, '{} / {}', (self, other))

  def __truediv__(self, other):
    return self.__div__(other)

  def __rdiv__(self, other):
    return CompositeMetric(lambda x, y: x / y, '{} / {}', (other, self))

  def __rtruediv__(self, other):
    return self.__rdiv__(other)

  def __pow__(self, other):
    if isinstance(other, float) and other == 0.5:
      return CompositeMetric(lambda x, y: x**y, 'sqrt({})', (self, other))
    return CompositeMetric(lambda x, y: x**y, '{} ^ {}', (self, other))

  def __rpow__(self, other):
    return CompositeMetric(lambda x, y: x**y, '{} ^ {}', (other, self))


class MetricList(Metric):
  """Wraps Metrics and compute them with caching.

  Attributes:
    name: Name of the Metric.
    children: An sequence of Metrics.
    names: A list of names of children.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    cache: A dict to store cached results.
    cache_key: The key currently being used in computation.
    children_return_dataframe: Whether to convert the result to a children
      Metrics to DataFrames if they are not.
    name_tmpl: A string template to format the columns in the result DataFrame.
    And all other attributes inherited from Metric.
  """

  def __init__(self,
               children: Sequence[Metric],
               where: Optional[Union[Text, Sequence[Text]]] = None,
               children_return_dataframe: bool = True,
               name_tmpl=None):
    for m in children:
      if not isinstance(m, Metric):
        raise ValueError('%s is not a Metric.' % m)
    tmpl = name_tmpl or 'MetricList({})'
    name = tmpl.format(', '.join(m.name for m in children))
    super(MetricList, self).__init__(name, children, where, name_tmpl)
    self.children_return_dataframe = children_return_dataframe
    self.names = [m.name for m in children]

  def compute_slices(self, df, split_by=None):
    """Computes all Metrics with caching.

    We know df is not going to change so we can safely enable caching with an
    arbitrary key.

    Args:
      df: The DataFrame to compute on.
      split_by: Something can be passed into df.group_by().

    Returns:
      A list of results.
    """
    res = []
    key = self.cache_key or self.RESERVED_KEY
    for m in self:
      try:
        child = m.compute_on(
            df,
            split_by,
            return_dataframe=self.children_return_dataframe,
            cache_key=key)
        if isinstance(child, pd.DataFrame):
          if self.name_tmpl:
            child.columns = [self.name_tmpl.format(c) for c in child.columns]
        if isinstance(child, pd.Series):
          if self.name_tmpl:
            child.name = self.name_tmpl.format(child.name)
        res.append(child)
      except Exception as e:  # pylint: disable=broad-except
        print('Warning: %s failed for reason %s.' % (m.name, repr(e)))
    return res

  def get_sql_and_with_clause(self, table, split_by, global_filter, indexes,
                              local_filter, with_data):
    """Gets the SQL query and WITH clause.

    The query is constructed by
    1. Get the query for every children metric.
    2. If all children queries are compatible, we just collect all the columns
      from the children and use the WHERE and GROUP BY clauses from any
      children. The FROM clause is more complex. We use the largest FROM clause
      in children.
      See the doc of is_compatible() for its meaning.
      If any pair of children queries are incompatible, we merge the compatible
      children as much as possible then add the merged SQLs to with_data, join
      them on indexes, and SELECT *.

    Args:
      table: The table we want to query from.
      split_by: The columns that we use to split the data.
      global_filter: The filters that can be applied to the whole Metric tree.
      indexes: The columns that we shouldn't apply any arithmetic operation.
      local_filter: The filters that have been accumulated so far.
      with_data: A global variable that contains all the WITH clauses we need.

    Returns:
      The SQL instance for metric, without the WITH clause component.
      The global with_data which holds all datasources we need in the WITH
        clause.
    """
    local_filter = sql.Filters([self.where, local_filter]).remove(global_filter)
    children_sql = [
        c.get_sql_and_with_clause(table, split_by, global_filter, indexes,
                                  local_filter, with_data)[0]
        for c in self.children
    ]
    incompatible_sqls = sql.Datasources()
    for child_sql in children_sql:
      incompatible_sqls.merge(sql.Datasource(child_sql, 'MetricListChildTable'))

    name_tmpl = self.name_tmpl or '{}'
    if len(incompatible_sqls) == 1:
      res = next(iter(incompatible_sqls.children.values()))
      for c in res.columns:
        if c not in indexes:
          c.alias_raw = name_tmpl.format(c.alias_raw)
      return res, with_data

    columns = sql.Columns(indexes.aliases)
    for i, (alias, table) in enumerate(incompatible_sqls.children.items()):
      data = sql.Datasource(table, alias)
      alias, rename = with_data.merge(data)
      for c in table.columns:
        if c not in columns:
          columns.add(
              sql.Column(
                  '%s.%s' % (alias, rename.get(c.alias, c.alias)),
                  alias=name_tmpl.format(c.alias_raw)))
      if i == 0:
        from_data = alias
      else:
        join = 'FULL' if indexes else 'CROSS'
        from_data = sql.Join(from_data, alias, join=join, using=indexes)
    return sql.Sql(columns, from_data), with_data

  def compute_on_children(self, child, split_by):
    if isinstance(child, list):
      return self.to_dataframe(child)
    return child

  def to_dataframe(self, res):
    res_all = pd.concat(res, axis=1, sort=False)
    # In PY2, if index order are different, the names might get dropped.
    res_all.index.names = res[0].index.names
    return res_all

  def __iter__(self):
    for m in self.children:
      yield m

  def __len__(self):
    return len(self.children)

  def __getitem__(self, key):
    return self.children[key]


class CompositeMetric(Metric):
  """Class for Metrics formed by composing two Metrics.

  Attributes:
    name: Name of the Metric.
    op: Binary operation that defines the composite metric.
    name_tmpl: The template to generate the name from child Metrics' names.
    children: A length-2 iterable of Metrics and/or constants that forms the
      CompositeMetric.
    columns: Used to rename the columns of all DataFrames returned by child
      Metrics, so could be a list of column names or a pd.Index. This is useful
      when further operations (e.g., subtraction) need to be performed between
      two multiple-column DataFrames, which require that the DataFrames have
      matching column names.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    cache: A dict to store cached results.
    cache_key: The key currently being used in computation.
    And all other attributes inherited from Metric.
  """

  def __init__(self,
               op,
               name_tmpl: Text,
               children: Sequence[Union[Metric, int, float]],
               rename_columns=None,
               where: Optional[Text] = None):
    if len(children) != 2:
      raise ValueError('CompositeMetric must take two children!')
    if not isinstance(children[0], (Metric, int, float)):
      raise ValueError('%s is not a Metric or a number!' %
                       utils.get_name(children[0]))
    if not isinstance(children[1], (Metric, int, float)):
      raise ValueError('%s is not a Metric or a number!' %
                       utils.get_name(children[1]))
    if not isinstance(children[0], Metric) and not isinstance(
        children[1], Metric):
      raise ValueError('MetricList must take at least one Metric!')

    name = name_tmpl.format(*map(utils.get_name, children))
    super(CompositeMetric, self).__init__(name, children, where, name_tmpl)
    self.op = op
    self.columns = rename_columns

  def rename_columns(self, rename_columns):
    self.columns = rename_columns
    return self

  def set_name(self, name):
    self.name = name
    return self

  def compute_children(self, df, split_by):
    if len(self.children) != 2:
      raise ValueError('CompositeMetric can only have two children.')
    if not any([isinstance(m, Metric) for m in self.children]):
      raise ValueError('Must have at least one Metric.')
    children = []
    for m in self.children:
      if isinstance(m, Metric):
        m = m.compute_on(
            df,
            split_by,
            # MetricList returns an undesired list when not return_dataframe.
            return_dataframe=isinstance(m, MetricList),
            cache_key=self.cache_key or self.RESERVED_KEY)
      children.append(m)
    return children

  def compute_on_children(self, child, split_by):
    """Computes the result based on the results from the children.

    Computations between two DataFrames require columns to match. It makes
    Metric as simple as Sum('X') - Sum('Y') infeasible because they don't have
    same column names. To overcome this and make CompositeMetric useful, we try
    our best to reduce DataFrames to pd.Series and scalars. The rule is,
      1. If no split_by, a 1*1 DataFrame will be converted to a scalar and a n*1
        DataFrame will be a pd.Series.
      2. If split_by, both cases will be converted to pd.Series.
      3. If both DataFrames have the same number of columns but names don't
        match, you can pass in columns or call set_columns() to unify the
        columns. Otherwise we will apply self.name_tmpl to the column names.

    Args:
      child: A length-2 list. The elements could be numbers, pd.Series or
        pd.DataFrames.
      split_by: Something can be passed into df.group_by().

    Returns:
      The result to be sent to final_compute().
    """
    a, b = child[0], child[1]
    m1, m2 = self.children[0], self.children[1]
    if isinstance(a, pd.DataFrame) and a.shape[1] == 1 and not is_operation(m1):
      a = a.iloc[:, 0]
    if isinstance(b, pd.DataFrame) and b.shape[1] == 1 and not is_operation(m2):
      b = b.iloc[:, 0]
    res = None
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
      if len(a.columns) == len(b.columns):
        columns = [
            self.name_tmpl.format(c1, c2)
            for c1, c2 in zip(a.columns, b.columns)
        ]
        # pytype is not smart enough http://github.com/google/pytype/issues/391.
        a.columns = columns  # pytype: disable=not-writable
        b.columns = columns  # pytype: disable=not-writable
    elif isinstance(a, pd.DataFrame):
      for col in a.columns:
        a[col] = self.op(a[col], b)
      res = a
      columns = [
          self.name_tmpl.format(c, getattr(m2, 'name', m2)) for c in res.columns
      ]
      res.columns = columns  # pytype: disable=not-writable
    elif isinstance(b, pd.DataFrame):
      for col in b.columns:
        b[col] = self.op(a, b[col])
      res = b
      columns = [
          self.name_tmpl.format(getattr(m1, 'name', m1), c) for c in res.columns
      ]
      res.columns = columns  # pytype: disable=not-writable

    if res is None:
      res = self.op(a, b)
    if isinstance(res, pd.Series):
      res.name = self.name  # pytype: disable=not-writable
      if self.columns:
        res = pd.DataFrame(res)
    if self.columns is not None and isinstance(res, pd.DataFrame):
      res.columns = self.columns  # pytype: disable=not-writable
    return res

  def get_sql_and_with_clause(self, table, split_by, global_filter, indexes,
                              local_filter, with_data):
    """Gets the SQL query and WITH clause.

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
      table: The table we want to query from.
      split_by: The columns that we use to split the data.
      global_filter: The filters that can be applied to the whole Metric tree.
      indexes: The columns that we shouldn't apply any arithmetic operation.
      local_filter: The filters that have been accumulated so far.
      with_data: A global variable that contains all the WITH clauses we need.

    Returns:
      The SQL instance for metric, without the WITH clause component.
      The global with_data which holds all datasources we need in the WITH
        clause.
    """
    local_filter = sql.Filters([self.where, local_filter]).remove(global_filter)
    op = self.op

    if not isinstance(self.children[0], Metric):
      constant = self.children[0]
      query, with_data = self.children[1].get_sql_and_with_clause(
          table, split_by, global_filter, indexes, local_filter, with_data)
      query.columns = sql.Columns(
          (c if c in indexes else op(constant, c) for c in query.columns))
    elif not isinstance(self.children[1], Metric):
      constant = self.children[1]
      query, with_data = self.children[0].get_sql_and_with_clause(
          table, split_by, global_filter, indexes, local_filter, with_data)
      query.columns = sql.Columns(
          (c if c in indexes else op(c, constant) for c in query.columns))
    else:
      query0, with_data = self.children[0].get_sql_and_with_clause(
          table, split_by, global_filter, indexes, local_filter, with_data)
      query1, with_data = self.children[1].get_sql_and_with_clause(
          table, split_by, global_filter, indexes, local_filter, with_data)
      if len(query0.columns) != 1 and len(query1.columns) != 1 and len(
          query0.columns) != len(query1.columns):
        raise ValueError('Children Metrics have different shapes!')

      compatible, larger_from = sql.is_compatible(query0, query1)
      if compatible:
        col0_col1 = zip(itertools.cycle(query0.columns), query1.columns)
        if len(query1.columns) == 1:
          col0_col1 = zip(query0.columns, itertools.cycle(query1.columns))
        columns = sql.Columns()
        for c0, c1 in col0_col1:
          if c0 in indexes.aliases:
            columns.add(c0)
          else:
            alias = self.name_tmpl.format(c0.alias_raw, c1.alias_raw)
            columns.add(op(c0, c1).set_alias(alias))
        query = query0
        query.columns = columns
        query.from_data = larger_from
      else:
        tbl0 = with_data.add(sql.Datasource(query0, 'CompositeMetricTable0'))
        tbl1 = with_data.add(sql.Datasource(query1, 'CompositeMetricTable1'))
        join = 'FULL' if indexes else 'CROSS'
        from_data = sql.Join(tbl0, tbl1, join=join, using=indexes)
        columns = sql.Columns()
        col0_col1 = zip(itertools.cycle(query0.columns), query1.columns)
        if len(query1.columns) == 1:
          col0_col1 = zip(query0.columns, itertools.cycle(query1.columns))
        for c0, c1 in col0_col1:
          if c0 not in indexes.aliases:
            col = op(
                sql.Column('%s.%s' % (tbl0, c0.alias), alias=c0.alias_raw),
                sql.Column('%s.%s' % (tbl1, c1.alias), alias=c1.alias_raw))
            columns.add(col)
        query = sql.Sql(sql.Columns(indexes.aliases).add(columns), from_data)

    if not is_operation(self.children[0]) and not is_operation(
        self.children[1]) and len(query.columns) == 1:
      query.columns[0].set_alias(self.name)

    if self.columns:
      columns = query.columns.difference(indexes)
      if len(self.columns) != len(columns):
        raise ValueError('The length of the renaming columns is wrong!')
      for col, rename in zip(columns, self.columns):
        col.set_alias(rename)  # Modify in-place.

    return query, with_data


class Ratio(CompositeMetric):
  """Syntactic sugar for Sum('A') / Sum('B')."""

  def __init__(self,
               numerator: Text,
               denominator: Text,
               name: Optional[Text] = None,
               where: Optional[Text] = None):
    super(Ratio, self).__init__(
        lambda x, y: x / y,
        '{} / {}', (Sum(numerator), Sum(denominator)),
        where=where)
    self.name = name or self.name


class SimpleMetric(Metric):
  """Base class for common built-in aggregate functions of df.group_by()."""

  def __init__(self,
               var: Text,
               name: Optional[Text] = None,
               name_tmpl=None,
               where: Optional[Union[Text, Sequence[Text]]] = None,
               **kwargs):
    name = name or name_tmpl.format(var)
    self.var = var
    precompute = kwargs.pop('precompute', None)
    postcompute = kwargs.pop('postcompute', None)
    final_compute = kwargs.pop('final_compute', None)
    self.kwargs = kwargs
    super(SimpleMetric, self).__init__(
        name,
        None,
        where,
        name_tmpl,
        precompute=precompute,
        postcompute=postcompute,
        final_compute=final_compute)

  def get_sql_and_with_clause(self, table, split_by, global_filter, indexes,
                              local_filter, with_data):
    del indexes  # unused
    local_filter = sql.Filters([self.where, local_filter]).remove(global_filter)
    return sql.Sql(
        self.get_sql_columns(local_filter), table, global_filter,
        split_by), with_data

  def get_sql_columns(self, local_filter):
    raise ValueError('get_sql_columns is not implemented for %s.' % type(self))


class Count(SimpleMetric):
  """Count estimator.

  Attributes:
    var: Column to compute on.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    distinct: Whether to count distinct values.
    kwargs: Other kwargs passed to pd.DataFrame.count() or nunique().
    And all other attributes inherited from Metric.
  """

  def __init__(self,
               var: Text,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None,
               distinct: bool = False,
               **kwargs):
    self.distinct = distinct
    if distinct:
      name = name or 'count(distinct %s)' % var
    super(Count, self).__init__(var, name, 'count({})', where, **kwargs)

  def compute_slices(self, df, split_by=None):
    grped = self.group(df, split_by)[self.var]
    return grped.nunique(**self.kwargs) if self.distinct else grped.count(
        **self.kwargs)

  def get_sql_columns(self, local_filter):
    if self.distinct:
      return sql.Column(self.var, 'COUNT(DISTINCT {})', self.name, local_filter)
    else:
      return sql.Column(self.var, 'COUNT({})', self.name, local_filter)


class Sum(SimpleMetric):
  """Sum estimator.

  Attributes:
    var: Column to compute on.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    kwargs: Other kwargs passed to pd.DataFrame.sum().
    And all other attributes inherited from SimpleMetric.
  """

  def __init__(self,
               var: Text,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None,
               **kwargs):
    super(Sum, self).__init__(var, name, 'sum({})', where, **kwargs)

  def compute_slices(self, df, split_by=None):
    return self.group(df, split_by)[self.var].sum(**self.kwargs)

  def get_sql_columns(self, local_filter):
    return sql.Column(self.var, 'SUM({})', self.name, local_filter)


class Dot(SimpleMetric):
  """Inner product estimator.

  Attributes:
    var1: The first column in the inner product.
    var2: The second column in the inner product.
    normalize: If to normalize by the length.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    kwargs: Other kwargs passed to pd.DataFrame.sum() or .mean(). And all other
      attributes inherited from SimpleMetric.
  """

  def __init__(self,
               var1: Text,
               var2: Text,
               normalize=False,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None,
               **kwargs):
    self.var2 = var2
    self.normalize = normalize
    name_tmpl = ('mean({} * %s)' if normalize else 'sum({} * %s)') % var2
    super(Dot, self).__init__(var1, name, name_tmpl, where, **kwargs)

  def compute_slices(self, df, split_by=None):
    if not split_by:
      prod = (df[self.var] * df[self.var2])
      return prod.mean(**self.kwargs) if self.normalize else prod.sum(
          **self.kwargs)
    df['_meterstick_dot_helper'] = df[self.var] * df[self.var2]
    grped = df.groupby(split_by)['_meterstick_dot_helper']
    res = grped.mean() if self.normalize else grped.sum()
    df.drop('_meterstick_dot_helper', axis=1, inplace=True)
    return res

  def get_sql_columns(self, local_filter):
    tmpl = 'AVG({} * {})' if self.normalize else 'SUM({} * {})'
    return sql.Column((self.var, self.var2), tmpl, self.name, local_filter)


class Mean(SimpleMetric):
  """Mean estimator.

  Attributes:
    var: Column to compute on.
    weight: The column of weights.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    kwargs: Other kwargs passed to pd.DataFrame.mean(). Only has effect if no
      weight is specified.
    And all other attributes inherited from SimpleMetric.
  """

  def __init__(self,
               var: Text,
               weight: Optional[Text] = None,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None,
               **kwargs):
    name_tmpl = '%s-weighted mean({})' % weight if weight else 'mean({})'
    super(Mean, self).__init__(var, name, name_tmpl, where, **kwargs)
    self.weight = weight

  def compute(self, df):
    if not self.weight:
      raise ValueError('Weight is missing in %s.' % self.name)
    return np.average(df[self.var], weights=df[self.weight])

  def compute_slices(self, df, split_by=None):
    if self.weight:
      # When there is weight, just loop through slices.
      return super(Mean, self).compute_slices(df, split_by)
    return self.group(df, split_by)[self.var].mean(**self.kwargs)

  def get_sql_columns(self, local_filter):
    if not self.weight:
      return sql.Column(self.var, 'AVG({})', self.name, local_filter)
    else:
      res = sql.Column('%s * %s' % (self.weight, self.var), 'SUM({})',
                       'total_sum', local_filter)
      res /= sql.Column(self.weight, 'SUM({})', 'total_weight', local_filter)
      return res.set_alias(self.name)


class Max(SimpleMetric):
  """Max estimator.

  Attributes:
    var: Column to compute on.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    kwargs: Other kwargs passed to pd.DataFrame.max().
    And all other attributes inherited from SimpleMetric.
  """

  def __init__(self,
               var: Text,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None,
               **kwargs):
    super(Max, self).__init__(var, name, 'max({})', where, **kwargs)

  def compute_slices(self, df, split_by=None):
    return self.group(df, split_by)[self.var].max(**self.kwargs)

  def get_sql_columns(self, local_filter):
    return sql.Column(self.var, 'MAX({})', self.name, local_filter)


class Min(SimpleMetric):
  """Min estimator.

  Attributes:
    var: Column to compute on.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    kwargs: Other kwargs passed to pd.DataFrame.min().
    And all other attributes inherited from SimpleMetric.
  """

  def __init__(self,
               var: Text,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None,
               **kwargs):
    super(Min, self).__init__(var, name, 'min({})', where, **kwargs)

  def compute_slices(self, df, split_by=None):
    return self.group(df, split_by)[self.var].min(**self.kwargs)

  def get_sql_columns(self, local_filter):
    return sql.Column(self.var, 'MIN({})', self.name, local_filter)


class Quantile(SimpleMetric):
  """Quantile estimator.

  Attributes:
    var: Column to compute on.
    quantile: Same as the arg "q" in np.quantile().
    weight: The column of weights. If specified, we always return a DataFrame.
    interpolation: As the same arg in pd.Series.quantile(). No effect if
      'weight' is specified.
    one_quantile: If quantile is a number or an iterable.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    kwargs: Other kwargs passed to pd.DataFrame.quantile().
    And all other attributes inherited from SimpleMetric.
  """

  def __init__(self,
               var: Text,
               quantile: Union[float, int] = 0.5,
               weight: Optional[Text] = None,
               interpolation='linear',
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None,
               **kwargs):
    self.one_quantile = isinstance(quantile, (int, float))
    if not self.one_quantile:
      quantile = np.array(quantile)
    if self.one_quantile and not 0 <= quantile <= 1:
      raise ValueError('quantiles must be in [0, 1].')
    if not self.one_quantile and not (np.all(quantile >= 0) and
                                      np.all(quantile <= 1)):
      raise ValueError('quantiles must be in [0, 1].')
    name_tmpl = 'quantile({}, {})'
    if weight:
      name_tmpl = '%s-weighted quantile({}, {})' % weight
    name = name or name_tmpl.format(var, str(quantile))
    self.quantile = quantile
    self.interpolation = interpolation
    self.weight = weight
    super(Quantile, self).__init__(var, name, name_tmpl, where, **kwargs)

  def compute(self, df):
    """Adapted from https://stackoverflow.com/a/29677616/12728137."""
    if not self.weight:
      raise ValueError('Weight is missing in %s.' % self.name)

    sample_weight = np.array(df[self.weight])
    values = np.array(df[self.var])
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    weighted_quantiles /= np.sum(sample_weight)
    res = np.interp(self.quantile, weighted_quantiles, values)
    if self.one_quantile:
      return res
    return pd.DataFrame(
        [res],
        columns=[self.name_tmpl.format(self.var, q) for q in self.quantile])

  def compute_slices(self, df, split_by=None):
    if self.weight:
      # When there is weight, just loop through slices.
      return super(Quantile, self).compute_slices(df, split_by)
    res = self.group(df, split_by)[self.var].quantile(
        self.quantile, interpolation=self.interpolation, **self.kwargs)
    if self.one_quantile:
      return res

    if split_by:
      res = res.unstack()
      res.columns = [self.name_tmpl.format(self.var, c) for c in res]
      return res
    res = utils.unmelt(pd.DataFrame(res))
    res.columns = [self.name_tmpl.format(self.var, c[0]) for c in res]
    return res

  def get_sql_columns(self, local_filter):
    """Get SQL columns."""
    if self.weight:
      raise ValueError('SQL for weighted quantile is not supported!')
    if self.one_quantile:
      alias = 'quantile(%s, %s)' % (self.var, self.quantile)
      return sql.Column(
          self.var,
          'APPROX_QUANTILES({}, 100)[OFFSET(%s)]' % int(100 * self.quantile),
          alias, local_filter)

    query = 'APPROX_QUANTILES({}, 100)[OFFSET(%s)]'
    quantiles = []
    for q in self.quantile:
      alias = 'quantile(%s, %s)' % (self.var, q)
      if alias.startswith('0.'):
        alias = 'point_' + alias[2:]
      quantiles.append(
          sql.Column(self.var, query % int(100 * q), alias, local_filter))
    return sql.Columns(quantiles)


class Variance(SimpleMetric):
  """Variance estimator.

  Attributes:
    var: Column to compute on.
    ddof: Degree of freedom to use in pd.DataFrame.var().
    weight: The column of weights.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    kwargs: Other kwargs passed to pd.DataFrame.var(). Only has effect if no
      weight is specified.
    And all other attributes inherited from SimpleMetric.
  """

  def __init__(self,
               var: Text,
               unbiased: bool = True,
               weight: Optional[Text] = None,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None,
               **kwargs):
    self.ddof = 1 if unbiased else 0
    self.weight = weight
    name_tmpl = '%s-weighted var({})' % weight if weight else 'var({})'
    super(Variance, self).__init__(var, name, name_tmpl, where, **kwargs)

  def compute(self, df):
    if not self.weight:
      raise ValueError('Weight is missing in %s.' % self.name)
    avg = np.average(df[self.var], weights=df[self.weight])
    total = (df[self.weight] * (df[self.var] - avg)**2).sum()
    total_weights = df[self.weight].sum()
    return total / (total_weights - self.ddof)

  def compute_slices(self, df, split_by=None):
    if self.weight:
      # When there is weight, just loop through slices.
      return super(Variance, self).compute_slices(df, split_by)
    return self.group(df, split_by)[self.var].var(ddof=self.ddof, **self.kwargs)

  def get_sql_and_with_clause(self, table, split_by, global_filter, indexes,
                              local_filter, with_data):
    if self.weight:
      return _get_sql_for_weighted_var_or_se(self, table, split_by,
                                             global_filter, local_filter,
                                             with_data)
    return super(Variance,
                 self).get_sql_and_with_clause(table, split_by, global_filter,
                                               indexes, local_filter, with_data)

  def get_sql_columns(self, local_filter):
    if not self.weight:
      if self.ddof == 1:
        return sql.Column(self.var, 'VAR_SAMP({})', self.name, local_filter)
      else:
        return sql.Column(self.var, 'VAR_POP({})', self.name, local_filter)


class StandardDeviation(SimpleMetric):
  """Standard Deviation estimator.

  Attributes:
    var: Column to compute on.
    ddof: Degree of freedom to use in pd.DataFrame.std().
    weight: The column of weights.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    kwargs: Other kwargs passed to pd.DataFrame.std(). Only has effect if no
      weight is specified.
    And all other attributes inherited from SimpleMetric.
  """

  def __init__(self,
               var: Text,
               unbiased: bool = True,
               weight: Optional[Text] = None,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None,
               **kwargs):
    self.ddof = 1 if unbiased else 0
    self.weight = weight
    name_tmpl = '%s-weighted sd({})' % weight if weight else 'sd({})'
    super(StandardDeviation, self).__init__(var, name, name_tmpl, where,
                                            **kwargs)

  def compute(self, df):
    if not self.weight:
      raise ValueError('Weight is missing in %s.' % self.name)
    avg = np.average(df[self.var], weights=df[self.weight])
    total = (df[self.weight] * (df[self.var] - avg)**2).sum()
    total_weights = df[self.weight].sum()
    return np.sqrt(total / (total_weights - self.ddof))

  def compute_slices(self, df, split_by=None):
    if self.weight:
      # When there is weight, just loop through slices.
      return super(StandardDeviation, self).compute_slices(df, split_by)
    return self.group(df, split_by)[self.var].std(ddof=self.ddof, **self.kwargs)

  def get_sql_and_with_clause(self, table, split_by, global_filter, indexes,
                              local_filter, with_data):
    if self.weight:
      query, with_data = _get_sql_for_weighted_var_or_se(
          self, table, split_by, global_filter, local_filter, with_data)
      return query, with_data
    return super(StandardDeviation,
                 self).get_sql_and_with_clause(table, split_by, global_filter,
                                               indexes, local_filter, with_data)

  def get_sql_columns(self, local_filter):
    if not self.weight:
      if self.ddof == 1:
        return sql.Column(self.var, 'STDDEV_SAMP({})', self.name, local_filter)
      else:
        return sql.Column(self.var, 'STDDEV_POP({})', self.name, local_filter)


def _get_sql_for_weighted_var_or_se(metric, table, split_by, global_filter,
                                    local_filter, with_data):
  """Gets the SQL for weighted Variance or StandardDeviation.

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
    metric: An instance of Variance or StandardDeviation, with weight.
    table: The table we want to query from.
    split_by: The columns that we use to split the data.
    global_filter: The filters that can be applied to the whole Metric tree.
    local_filter: The filters that have been accumulated so far.
    with_data: A global variable that contains all the WITH clauses we need.

  Returns:
    The SQL instance for metric, without the WITH clause component.
    The global with_data which holds all datasources we need in the WITH clause.
  """
  where = sql.Filters([metric.where, local_filter]).remove(global_filter)
  weight = metric.weight
  var = metric.var
  columns = sql.Columns(split_by).add(
      sql.Column(weight, alias=weight, filters=where))
  total_sum = sql.Column(
      '%s * %s' % (weight, var), 'SUM({})', filters=where, partition=split_by)
  total_weight = sql.Column(
      weight, 'SUM({})', filters=where, partition=split_by)
  weighted_mean = total_sum / total_weight
  weighted_squared_diff = sql.Column(
      '%s * POWER(%s - %s, 2)' % (weight, var, weighted_mean.expression),
      alias='weighted_squared_diff',
      filters=where)
  weighted_base_table = sql.Sql(
      columns.add(weighted_squared_diff), table, global_filter)
  weighted_base_table_alias, rename = with_data.merge(
      sql.Datasource(weighted_base_table, 'WeightedBase'))

  weight = sql.Column(weight).alias
  weighted_var = sql.Column(
      rename.get('weighted_squared_diff', 'weighted_squared_diff'),
      'SUM({})') / sql.Column(rename.get(weight, weight), 'SUM({}) - 1')
  if isinstance(metric, StandardDeviation):
    weighted_var = weighted_var**0.5
  weighted_var.set_alias(metric.name)
  return sql.Sql(
      weighted_var, weighted_base_table_alias,
      groupby=split_by.aliases), with_data


class CV(SimpleMetric):
  """Coefficient of variation estimator.

  Attributes:
    var: Column to compute on.
    ddof: Degree of freedom to use.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    kwargs: Other kwargs passed to both pd.DataFrame.std() and
      pd.DataFrame.mean().
    And all other attributes inherited from SimpleMetric.
  """

  def __init__(self,
               var: Text,
               unbiased: bool = True,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None,
               **kwargs):
    self.ddof = 1 if unbiased else 0
    super(CV, self).__init__(var, name, 'cv({})', where, **kwargs)

  def compute_slices(self, df, split_by=None):
    var_grouped = self.group(df, split_by)[self.var]
    return var_grouped.std(
        ddof=self.ddof, **self.kwargs) / var_grouped.mean(**self.kwargs)

  def get_sql_columns(self, local_filter):
    if self.ddof == 1:
      res = sql.Column(self.var, 'STDDEV_SAMP({})',
                       self.name, local_filter) / sql.Column(
                           self.var, 'AVG({})', self.name, local_filter)
    else:
      res = sql.Column(self.var, 'STDDEV_POP({})',
                       self.name, local_filter) / sql.Column(
                           self.var, 'AVG({})', self.name, local_filter)
    return res.set_alias(self.name)


class Correlation(SimpleMetric):
  """Correlation estimator.

  Attributes:
    var: Column of first variable.
    var2: Column of second variable.
    weight: The column of weights.
    name: Name of the Metric.
    method: Method of correlation. The same arg in pd.Series.corr(). Only has
      effect if no weight is specified.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    kwargs: Other kwargs passed to pd.DataFrame.corr(). Only has effect if no
      weight is specified.
    And all other attributes inherited from SimpleMetric.
  """

  def __init__(self,
               var1: Text,
               var2: Text,
               weight: Optional[Text] = None,
               name: Optional[Text] = None,
               method='pearson',
               where: Optional[Union[Text, Sequence[Text]]] = None,
               **kwargs):
    name_tmpl = 'corr({}, {})'
    if weight:
      name_tmpl = '%s-weighted corr({}, {})' % weight
    if name is None:
      name = name_tmpl.format(var1, var2)
    self.var2 = var2
    self.method = method
    self.weight = weight
    super(Correlation, self).__init__(var1, name, name_tmpl, where, **kwargs)

  def compute(self, df):
    if not self.weight:
      raise ValueError('Weight is missing in %s.' % self.name)
    cov = np.cov(
        df[[self.var, self.var2]],
        rowvar=False,
        bias=True,
        aweights=df[self.weight])
    return cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

  def compute_slices(self, df, split_by=None):
    # If there are duplicated index values and split_by is not None, the result
    # will be wrong. For example,
    # df = pd.DataFrame(
    #     {'idx': (1, 2, 1, 2), 'a': range(4), 'grp': [1, 1, 2, 2]}
    #   ).set_index('idx')
    # df.groupby('grp').a.corr(df.a) returns 0.447 for both groups.
    # It seems that pandas join the two series so if there are duplicated
    # indexes the series from the grouped by part gets duplicated.
    if isinstance(df, pd.DataFrame):
      df = df.reset_index(drop=True)
    if self.weight:
      # When there is weight, just loop through slices.
      return super(Correlation, self).compute_slices(df, split_by)
    return self.group(df, split_by)[self.var].corr(
        df[self.var2], method=self.method, **self.kwargs)

  def get_sql_columns(self, local_filter):
    if self.weight:
      raise ValueError('SQL for weighted correlation is not supported!')
    if self.method != 'pearson':
      raise ValueError('Only Pearson correlation is supported!')
    return sql.Column((self.var, self.var2), 'CORR({}, {})', self.name,
                      local_filter)


class Cov(SimpleMetric):
  """Covariance estimator.

  Attributes:
    var: Column of first variable.
    var2: Column of second variable.
    bias: The same arg passed to np.cov().
    ddof: The same arg passed to np.cov().
    weight: Column name of aweights passed to np.cov(). If you need fweights,
      pass it in kwargs.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    kwargs: Other kwargs passed to np.cov().
    And all other attributes inherited from SimpleMetric.
  """

  def __init__(self,
               var1: Text,
               var2: Text,
               bias: bool = False,
               ddof: Optional[int] = None,
               weight: Optional[Text] = None,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None,
               **kwargs):
    name_tmpl = 'cov({}, {})'
    if weight:
      name_tmpl = '%s-weighted %s' % (weight, name_tmpl)

    if name is None:
      name = name_tmpl.format(var1, var2)
    self.var2 = var2
    self.bias = bias
    self.ddof = ddof
    self.weight = weight
    super(Cov, self).__init__(var1, name, name_tmpl, where, **kwargs)

  def compute(self, df):
    return np.cov(
        df[[self.var, self.var2]],
        rowvar=False,
        bias=self.bias,
        ddof=self.ddof,
        aweights=df[self.weight] if self.weight else None,
        **self.kwargs)[0, 1]

  def get_sql_columns(self, local_filter):
    """Get SQL columns."""
    if self.weight:
      raise ValueError('SQL for weighted covariance is not supported!')
    ddof = self.ddof
    if ddof is None:
      ddof = 0 if self.bias else 1
    if ddof == 1:
      return sql.Column((self.var, self.var2), 'COVAR_SAMP({}, {})', self.name,
                        local_filter)
    elif ddof == 0:
      return sql.Column((self.var, self.var2), 'COVAR_POP({}, {})', self.name,
                        local_filter)
    else:
      raise ValueError('Only ddof being 0 or 1 is supported!')
