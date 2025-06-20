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
"""Base classes for Meterstick."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import datetime
import itertools
from typing import Any, Iterable, List, Optional, Sequence, Text, Union

from meterstick import sql
from meterstick import utils
import numpy as np
import pandas as pd


def compute_on(df,
               split_by=None,
               melted=False,
               return_dataframe=True,
               cache_key=None,
               cache=None,
               **kwargs):
  # pylint: disable=g-long-lambda
  return lambda x: x.compute_on(df, split_by, melted, return_dataframe,
                                cache_key, cache, **kwargs)
  # pylint: enable=g-long-lambda


# pylint: disable=g-long-lambda
def compute_on_sql(
    table,
    split_by=None,
    execute=None,
    melted=False,
    mode=None,
    cache_key=None,
    cache=None,
    return_dataframe=True,
    **kwargs):
  """A wrapper that metric | compute_on_sql() === metric.compute_on_sql()."""
  return lambda m: m.compute_on_sql(
      table,
      split_by,
      execute,
      melted,
      mode,
      cache_key,
      cache=cache,
      return_dataframe=return_dataframe,
      **kwargs)


def compute_on_beam(
    table,
    split_by=None,
    execute=None,
    melted=False,
    mode=None,
    cache_key=None,
    cache=None,
    sql_transform_kwargs=None,
    **kwargs,
):
  """A wrapper for metric.compute_on_beam()."""
  return lambda m: m.compute_on_beam(
      table,
      split_by,
      execute,
      melted,
      mode,
      cache_key,
      cache=cache,
      sql_transform_kwargs=sql_transform_kwargs,
      **kwargs,
  )


# pylint: enable=g-long-lambda


def to_sql(table, split_by=None, create_tmp_table_for_volatile_fn=None):
  return lambda metric: metric.to_sql(
      table, split_by, create_tmp_table_for_volatile_fn
  )


# Classes we built so caching across instances can be enabled with confidence.
BUILT_INS = [
    # Metrics
    'MetricList',
    'CompositeMetric',
    'Ratio',
    'Count',
    'Sum',
    'Dot',
    'Mean',
    'Max',
    'Min',
    'Nth',
    'Quantile',
    'Variance',
    'StandardDeviation',
    'CV',
    'Correlation',
    'Cov',
    # Operations
    'Distribution',
    'Normalize',
    'CumulativeDistribution',
    'PercentChange',
    'AbsoluteChange',
    'PrePostChange',
    'CUPED',
    'MH',
    'Jackknife',
    'Bootstrap',
    'PoissonBootstrap',
    # Diversity Operations
    'HHI',
    'Entropy',
    'TopK',
    'Nxx',
    # Models
    'LinearRegression',
    'Ridge',
    'Lasso',
    'ElasticNet',
    'LogisticRegression',
]


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
  df -> df.query(where) -> precompute -|-> slice2 -> compute | -> concat  -> postcompute -> manipulate -> final_compute -> clean_up_cache  # pylint: disable=line-too-long
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
    where_: A string or list/tuple of strings to be concatenated that will be
      passed to df.query() as a prefilter.
    where: A string that will be passed to df.query() as a prefilter. It's ' and
      '.join(where_).
    additional_fingerprint_attrs: Additional attributes to be encoded into the
      fingerprint. The attribute value must be hashable, or a list/dict of
      hashables. See get_fingerprint() for how it's used.
    extra_split_by: Used by Operation. See the doc there.
    extra_index: Used by Operation. See the doc there.
    name_tmpl: Used by Metrics that have children. It's applied to children's
      names in the output.
    is_operation: If this instance is an Operation.
    cache_across_instances: If this Metric class will be cached across
      instances, namely, different instances with same attributes that matter
      can share the same place in cache. All the classes listed in BUILT_INS
      have the feature enabled. For custom Metrics, by default different
      instances don't share the same place in cache, because we don't know what
      attributes matter. If you want to enable the feature for a custom Metric,
      make sure you read the 'Custom Metric' and 'Caching' sections in the demo
      notebook and understand the `additional_fingerprint_attrs` attribute
      before setting this attribute to True.
    cache: A dict to store the result. It's shared across the Metric tree.
    cache_key: The key currently being used in computation.
  """

  def __init__(self,
               name: Text,
               children: Optional[Union['Metric', Sequence[Union['Metric', int,
                                                                 float]]]] = (),
               where: Optional[Union[Text, Sequence[Text]]] = None,
               name_tmpl=None,
               extra_split_by: Optional[Union[Text, Iterable[Text]]] = None,
               extra_index: Optional[Union[Text, Iterable[Text]]] = None,
               additional_fingerprint_attrs: Optional[List[str]] = None):
    self.name = name
    self.cache = {}
    self.cache_key = None
    self.children = [children] if isinstance(children,
                                             Metric) else children or []
    self.where_ = None
    self.where = where
    self.extra_split_by = [extra_split_by] if isinstance(
        extra_split_by, str) else extra_split_by or []
    if extra_index is None:
      self.extra_index = self.extra_split_by
    else:
      self.extra_index = [extra_index] if isinstance(extra_index,
                                                     str) else extra_index
    self.additional_fingerprint_attrs = set(additional_fingerprint_attrs or ())
    self.name_tmpl = name_tmpl
    self.is_operation = False
    self.cache_across_instances = False
    self.cache_key = None

  @property
  def where(self):
    if isinstance(self.where_, (list, tuple)):
      where_ = self.where_
      if len(where_) > 1:
        where_ = (f'({i})' for i in sorted(where_))
      return ' and '.join(where_)
    return self.where_

  @where.setter
  def where(self, where):
    if where is None:
      self.where_ = None
    elif isinstance(where, str):
      self.where_ = (where,)
    else:
      self.where_ = tuple(where)

  def add_where(self, where):
    if where is None:
      return self
    where = [where] if isinstance(where, str) else list(where) or []
    if not self.where_:
      self.where = where
    else:
      self.where = tuple(set(list(self.where_) + where))
    return self

  def _compute_with_caching_and_postprocessing(
      self,
      compute_fn,
      df,
      split_by,
      melted,
      return_dataframe,
      apply_name_tmpl,
      cache_key,
      cache,
      *args,
      **kwargs,
  ):
    """Wraps computation logic with caching and common postprocessing.

    This function does:
    1. Initializes a cache if it doesn't eixst.
    2. Reads from cache if possible.
    3. Otherwise calls compute_fn(df, split_by, *args, **kwargs).
    4. Postprocesses the result like melting and converting to pandas DataFrame.
    5. Cleans up cache if needed.

    Args:
      compute_fn: A function that compute_fn(df, split_by, *args, **kwargs)
        returns a number, pd.Series or a melted DataFrame. See compute_through
        and compute_through_sql for examples.
      df: The DataFrame to compute on.
      split_by: The columns that we use to split the data.
      melted: Whether to transform the result to long format.
      return_dataframe: Whether to convert the result to DataFrame if it's not.
        If False, it could still return a DataFrame.
      apply_name_tmpl: If to apply name_tmpl to the result. For example, in
        Distribution('country', Sum('X')).compute_on(df), we first compute
        Sum('X').compute_on(df, 'country'), then normalize, and finally apply
        the name_tmpl 'Distribution of {}' to all column names.
      cache_key: What key to use to cache the df. You can use anything that can
        be a key of a dict except '_RESERVED' and tuples like ('_RESERVED', ..).
      cache: The cache the whole Metric tree shares during one round of
        computation. If it's None, we initiate an empty dict.
      *args: Args passed to compute_fn.
      **kwargs: Args passed to compute_fn.

    Returns:
      Final result returned to user. If split_by, it's a pd.Series or a
      pd.DataFrame, otherwise it could be a base type.
    """
    self.cache = {} if cache is None else cache
    split_by = [split_by] if isinstance(split_by, str) else list(split_by or [])
    try:
      key = self.wrap_cache_key(cache_key or self.cache_key, split_by)
      if self.in_cache(key):
        raw_res = self.get_cached(key)
      else:
        self.cache_key = key
        raw_res = compute_fn(df, split_by, *args, **kwargs)
        self.save_to_cache(key, raw_res)

      res = self.manipulate(raw_res, melted, return_dataframe, apply_name_tmpl)
      return self.final_compute(res, melted, return_dataframe, split_by, df)
    finally:
      if cache_key is None:  # Only root metric can have None as cache_key
        self.clean_up_cache()

  def wrap_cache_key(self, key, split_by=None, where=None, slice_val=None):
    if key and not isinstance(key, utils.CacheKey) and self.cache_key:
      key = self.cache_key.replace_key(key)
    key = key or self.cache_key
    return utils.CacheKey(self, key, where or self.where_, split_by, slice_val)

  def save_to_cache(self, key, val, split_by=None):
    if not isinstance(key, utils.CacheKey):
      key = self.wrap_cache_key(key, split_by)
    val = val.copy() if isinstance(val, (pd.Series, pd.DataFrame)) else val
    self.cache[key] = val

  def get_cached(self, key):
    key = key if isinstance(key, utils.CacheKey) else self.wrap_cache_key(key)
    return self.cache[key]

  def in_cache(self, key):
    key = key if isinstance(key, utils.CacheKey) else self.wrap_cache_key(key)
    return key in self.cache

  def find_all_in_cache_by_metric_type(self, metric):
    """Retrieves results from a certain type of metric from cache."""
    return {k: v for k, v in self.cache.items() if k.metric.__class__ == metric}

  def manipulate(
      self,
      res,
      melted: bool = False,
      return_dataframe: bool = True,
      apply_name_tmpl=None,
  ):
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
        Distribution('country', Sum('X')).compute_on(df), we first compute
        Sum('X').compute_on(df, 'country'), then normalize, and finally apply
        the name_tmpl 'Distribution of {}' to all column names.

    Returns:
      Final result returned to user. If split_by, it's a pd.Series or a
      pd.DataFrame, otherwise it could be a base type.
    """
    if isinstance(res, pd.Series):
      res.name = self.name
    if not isinstance(res, pd.DataFrame) and return_dataframe:
      res = self.to_dataframe(res)
    if melted:
      res = utils.melt(res)
    if apply_name_tmpl:
      res = utils.apply_name_tmpl(self.name_tmpl, res, melted)
    return utils.remove_empty_level(res)

  def to_dataframe(self, res):
    if isinstance(res, pd.DataFrame):
      return res
    elif isinstance(res, pd.Series):
      return pd.DataFrame(res)
    return pd.DataFrame({self.name: [res]})

  def final_compute(self, res, melted, return_dataframe, split_by, df):
    del melted, return_dataframe, split_by, df  # Useful in derived classes.
    return res

  def clean_up_cache(self):
    """Flushes the cache when a Metric tree has been computed.

    A Metric and all the descendants form a tree. When a computation is started
    from a MetricList or CompositeMetric, we know the input DataFrame is not
    going to change in the computation. So even if user doesn't ask for caching,
    we still enable it, but we need to clean things up when done. As the results
    need to be cached until all Metrics in the tree have been computed, we
    should only clean up at the end of the computation of the entry/top Metric.
    We recognize the top Metric by looking at the cache_key. All descendants
    will have it assigned as RESERVED_KEY but the entry Metric's will be None.
    """
    self.cache.clear()
    for m in self.traverse():
      m.cache_key = None

  def compute_on(
      self,
      df: pd.DataFrame,
      split_by: Optional[Union[Text, List[Text]]] = None,
      melted: bool = False,
      return_dataframe: bool = True,
      cache_key: Any = None,
      cache=None,
  ):
    """Key API of Metric.

    This is what you should call to use Metric. As caching is the shared part of
    Metric, we suggest you NOT to overwrite this method. Overwriting
    compute_slices and/or final_compute should be enough. If not, contact us
    with your use cases.

    Args:
      df: The DataFrame to compute on.
      split_by: The columns that we use to split the data.
      melted: Whether to transform the result to long format.
      return_dataframe: Whether to convert the result to DataFrame if it's not.
        If False, it could still return a DataFrame.
      cache_key: What key to use to cache the df. You can use anything that can
        be a key of a dict except '_RESERVED' and tuples like ('_RESERVED', ..).
      cache: The cache the whole Metric tree shares during one round of
        computation. If it's None, we initiate an empty dict.

    Returns:
      Final result returned to user. If split_by, it's a pd.Series or a
      pd.DataFrame, otherwise it could be a base type.
    """
    return self._compute_with_caching_and_postprocessing(
        self.compute_through,
        df,
        split_by,
        melted,
        return_dataframe,
        None,
        cache_key,
        cache,
    )

  def compute_through(self, df, split_by: Optional[List[Text]] = None):
    """Precomputes df -> split df and apply compute() -> postcompute."""
    df = df.query(self.where) if df is not None and self.where else df
    res = self.precompute(df, split_by)
    res = self.compute_slices(res, split_by)
    return self.postcompute(res, split_by)

  def precompute(self, df, split_by):
    del split_by  # Useful in derived classes.
    return df

  def postcompute(self, df, split_by):
    del split_by  # Useful in derived classes.
    return df

  def compute_slices(self, df, split_by: Optional[List[Text]] = None):
    """Applies compute() to all slices. Each slice needs a unique cache_key."""
    if self.children:
      try:
        children = self.compute_children(df, split_by + self.extra_split_by)
        return self.compute_on_children(children, split_by)
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
            cache_key, slice_val=dict(zip(split_by, slice_i_iter)))
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

  def compute_children(
      self, df, split_by, melted=False, return_dataframe=True, cache_key=None
  ):
    raise NotImplementedError

  def compute_on_children(self, children, split_by):
    """Computes the return using the result returned by children Metrics.

    Args:
      children: The return of compute_children() or compute_children_sql().
      split_by: The columns that we use to split the data.

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
      for k, idx in df.groupby(split_by, observed=True).indices.items():
        # Use iloc rather than loc because indexes can have duplicates.
        yield df.iloc[idx], k

  def compute_with_split_by(
      self, df, split_by: Optional[List[Text]] = None, slice_value=None
  ):
    del split_by, slice_value  # In case users need them in derived classes.
    return self.compute(df)

  def compute(self, df):
    raise NotImplementedError

  def compute_on_sql(
      self,
      table,
      split_by=None,
      execute=None,
      melted=False,
      mode=None,
      cache_key=None,
      cache=None,
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
      cache: The cache the whole Metric tree shares during one round of
        computation. If it's None, we initiate an empty dict.
      return_dataframe: If False, result of simple Metric will be a number or
        pd.Series (when has split_by).

    Returns:
      A pandas DataFrame. It's the computeation of self in SQL.
    """
    return self._compute_with_caching_and_postprocessing(
        self.compute_through_sql,
        table,
        split_by,
        melted,
        return_dataframe,
        False,
        cache_key,
        cache,
        execute,
        mode,
    )

  def compute_through_sql(self, table, split_by, execute, mode):
    """Delegeates the computation to different modes."""
    if mode not in (None, 'mixed', 'magic'):
      raise ValueError('Mode %s is not supported!' % mode)
    if not self.children:
      res = self.compute_on_sql_sql_mode(table, split_by, execute)
      return self.to_series_or_number_if_not_operation(res)
    if not mode:
      try:
        res = self.compute_on_sql_sql_mode(table, split_by, execute)
        return self.to_series_or_number_if_not_operation(res)
      except NotImplementedError:
        pass
    if self.where:
      table = sql.Sql(None, table, self.where)
    try:
      res = self.compute_on_sql_mixed_mode(table, split_by, execute, mode)
      return self.to_series_or_number_if_not_operation(res)
    except NotImplementedError:
      raise
    except Exception as e:  # pylint: disable=broad-except
      if mode:
        raise ValueError(
            'Please see the root cause of the failure above. You can try'
            ' `mode=None` to see if it helps.'
        ) from e
      else:
        raise

  def to_series_or_number_if_not_operation(self, df):
    return self.to_series_or_number(df) if not self.is_operation else df

  def to_series_or_number(self, df):
    if not isinstance(df, pd.DataFrame):
      return df
    df = df.squeeze(axis=1)  # squeeze to a Series if possible
    if (
        isinstance(df, pd.Series)
        and len(df.index.names) == 1
        and not df.index.name
    ):  # squeeze to a number if applicable
      df = df.squeeze()
    return df

  def compute_on_sql_sql_mode(self, table, split_by=None, execute=None):
    """Executes the query from to_sql() and process the result."""
    query = self.to_sql(table, split_by, False)
    # We try to avoid using CREATE TEMP TABLE when possible. It's only used when
    # - the query contains RAND();
    # - the execute doesn't evaluate RAND() only once in the WITH clause;
    # - ALLOW_TEMP_TABLE is True.
    if sql.ALLOW_TEMP_TABLE and 'RAND()' in str(query):
      query_with_tmp_table = self.to_sql(table, split_by, True)
      if str(query) != str(
          query_with_tmp_table
      ) and not sql.rand_run_only_once_in_with_clause(execute):
        try:
          execute('CREATE OR REPLACE TEMP TABLE T AS (SELECT 42 AS ans);')
          sql.TEMP_TABLE_SUPPORTED = True
          query = self.to_sql(table, split_by, True)
        except Exception as exc:
          sql.TEMP_TABLE_SUPPORTED = False
          raise NotImplementedError from exc
        finally:
          sql.TEMP_TABLE_SUPPORTED = None
    res = execute(str(query))
    extra_idx = list(self.get_extra_idx(return_superset=True))
    indexes = split_by + extra_idx if split_by else extra_idx
    columns = [a.alias_raw for a in query.groupby.add(query.columns)]
    columns[:len(indexes)] = indexes
    res.columns = columns
    if indexes:
      res.set_index(indexes, inplace=True)
    if split_by:  # Use a stable sort.
      res.sort_values(split_by, kind='mergesort', inplace=True)
    return res

  def to_sql(
      self,
      table,
      split_by: Optional[Union[Text, List[Text]]] = None,
      create_tmp_table_for_volatile_fn=None,
  ):
    """Generates SQL query for the metric.

    Args:
      table: The table or subquery we want to query from.
      split_by: The columns that we use to split the data.
      create_tmp_table_for_volatile_fn: When generating the query, we assume
        that volatile functions like RAND() in the WITH clause behave as if they
        are evaluated only once. Unfortunately, not all engines behave like
        that. In those cases, we need to CREATE TEMP TABLE to materialize the
        subqueries that have volatile functions, so that the same result is used
        in all places. An example is
          WITH T AS (SELECT RAND() AS r)
          SELECT t1.r - t2.r AS d
          FROM T t1 CROSS JOIN T t2.
        If it doesn't always evaluates to 0, then this arg should be True, and
        we will put all subqueries that
          1) have volatile functions and
          2) are referenced in the same query multiple times,
        into CREATE TEMP TABLE statements.
        Note that this arg has no effect if sql.ALLOW_TEMP_TABLE is False.
        When you use compute_on_sql or compute_on_beam, this arg is
        automatically decided based on your `execute` function.

    Returns:
      The SQL query for the metric as a SQL instance, which is similar to a str.
      Calling str() on it will get the query in string.
    """
    global_filter = utils.get_global_filter(self)
    indexes = sql.Columns(split_by).add(
        self.get_extra_idx(return_superset=True)
    )
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
    create_tmp_table = (
        sql.ALLOW_TEMP_TABLE
        if create_tmp_table_for_volatile_fn is None
        else create_tmp_table_for_volatile_fn
    )
    if not create_tmp_table:
      return query
    # None means we don't know yet so we only check for False.
    if sql.TEMP_TABLE_SUPPORTED is False:  # pylint: disable=g-bool-id-comparison
      raise NotImplementedError  # to fall back to the mixed mode
    with_data.temp_tables = sql.get_temp_tables(with_data)
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

  def compute_on_sql_mixed_mode(self, table, split_by, execute, mode=None):
    """Computes the child in SQL and the rest in Python."""
    children = self.compute_children_sql(table, split_by, execute, mode)
    return self.compute_on_children(children, split_by)

  def compute_children_sql(
      self, table, split_by, execute, mode, *args, **kwargs
  ):
    """The return should be similar to compute_children()."""
    del args, kwargs  # unused
    children = []
    for c in self.children:
      if not isinstance(c, Metric):
        children.append(c)
      else:
        children.append(
            self.compute_util_metric_on_sql(
                c, table, split_by + self.extra_split_by, execute, False, mode
            )
        )
    return children[0] if len(self.children) == 1 else children

  def compute_on_beam(
      self,
      pcol,
      split_by=None,
      execute=None,
      melted=False,
      mode=None,
      cache_key=None,
      cache=None,
      sql_transform_kwargs=None,
      **kwargs,
  ):
    """Computes on an Apache Beam PCollection input.

    Args:
      pcol: An apache_beam.pvalue.PCollection instance we want to compute on. It
        needs to have a schema so it's queryable.
      split_by: The columns that we use to split the data.
      execute: A function that can executes PCollection with a SqlTransform and
        returns a DataFrame.
      melted: Whether to transform the result to long format.
      mode: Similar to the one in compute_on_sql(). `None` maximizes Beam usage
        while 'mixed' minimizes the usage.
      cache_key: What key to use to cache the result. You can use anything that
        can be a key of a dict except '_RESERVED' and tuples like ('_RESERVED',
        ..).
      cache: The cache the whole Metric tree shares during one round of
        computation. If it's None, we initiate an empty dict.
      sql_transform_kwargs: A dict that holds the kwargs to be passed to
        SqlTransform defined in
        https://beam.apache.org/releases/pydoc/2.30.0/apache_beam.transforms.sql.html.
      **kwargs: Other kwargs passed to compute_on_sql.

    Returns:
      A pandas DataFrame.
    """
    # pylint: disable=g-import-not-at-top
    from apache_beam import pvalue
    from apache_beam.transforms import sql as beam_sql

    if not isinstance(pcol, pvalue.PCollection):
      raise ValueError(
          'The input must be a PCollection but got %s!' % type(pcol)
      )

    def e(q):
      label = f'Meterstick at {datetime.datetime.now()} runs {q}'
      res = execute(
          pcol
          | label
          >> beam_sql.SqlTransform(str(q), **(sql_transform_kwargs or {}))
      )
      return res

    # pylint: disable=g-import-not-at-top

    try:
      return self.compute_on_sql(
          'PCOLLECTION', split_by, e, melted, mode, cache_key, cache, **kwargs
      )
    except Exception as e:  # pylint: disable=broad-except
      if not mode:
        raise ValueError(
            "Please see the root cause of the failure above. If it's caused by "
            'the SQL query not being supported, try '
            "compute_on_beam(..., mode='mixed')."
        ) from e
      raise

  def compute_equivalent(self, df, split_by=None):
    equiv, df = utils.get_fully_expanded_equivalent_metric_tree(self, df)
    return self.compute_util_metric_on(
        equiv, df, split_by, return_dataframe=False
    )

  def compute_util_metric_on(
      self,
      metric,
      df,
      split_by,
      melted=False,
      return_dataframe=True,
      cache_key=None,
  ):
    """Computes a util metric with caching and filtering handled correctly."""
    cache_key = self.wrap_cache_key(cache_key, split_by)
    return metric.compute_on(
        df, split_by, melted, return_dataframe, cache_key, self.cache
    )

  def compute_util_metric_on_sql(
      self,
      metric,
      table,
      split_by=None,
      execute=None,
      melted=False,
      mode=None,
      cache_key=None,
      return_dataframe=True,
  ):
    """Computes a util metric with caching and filtering handled correctly."""
    cache_key = self.wrap_cache_key(cache_key, split_by)
    return metric.compute_on_sql(
        table,
        split_by,
        execute,
        melted,
        mode,
        cache_key,
        self.cache,
        return_dataframe,
    )

  def get_equivalent(self, *auxiliary_cols):
    """Gets a Metric that is equivalent to self."""
    res = self.get_equivalent_without_filter(*auxiliary_cols)  # pylint: disable=assignment-from-none
    if res:
      res.name = self.name
      res.add_where(self.where_)
    return res

  def get_equivalent_without_filter(self, *auxiliary_cols):
    """Gets a Metric that is equivalent to self but ignoring the filter."""
    del auxiliary_cols  # might be used in derived classes
    return

  def get_auxiliary_cols(self):
    """Returns the auxiliary columns required by the equivalent Metric.

    See utils.add_auxiliary_cols() for the format of the return.
    """
    return ()

  @staticmethod
  def group(df, split_by=None):
    return df.groupby(split_by, observed=True) if split_by else df

  def visualize_metric_tree(self, rendering_fn, strict=True):
    """Renders the Metric tree.

    Args:
      rendering_fn: A function that takes a string of DOT representation of the
        Metric and renders it as side effect.
      strict: If to make the DOT language graph strict. The strict mode will
        dedupe duplicated edges.
    """
    rendering_fn(self.to_dot(strict))

  def to_dot(self, strict=True):
    """Represents the Metric in DOT language.

    Args:
      strict: If to make the DOT language graph strict. The strict mode will
        dedupe duplicated edges.

    Returns:
      A string representing the Metric tree in DOT language.
    """
    import pydot  # pylint: disable=g-import-not-at-top

    dot = pydot.Dot(self.name, graph_type='graph', strict=strict)
    for m in self.traverse(include_constants=True):
      label = str(getattr(m, 'name', m))
      if getattr(m, 'where', ''):
        label += ' where %s' % m.where
      dot.add_node(pydot.Node(id(m), label=label))

    def add_edges(metric):
      if isinstance(metric, Metric):
        for c in metric.children:
          dot.add_edge(pydot.Edge(id(metric), id(c)))
          add_edges(c)

    add_edges(self)
    return dot.to_string()

  def get_extra_idx(self, return_superset=False):
    """Collects the extra indexes added by self and its descendants.

    Args:
      return_superset: If to return the superset of extra indexes if metric has
        incompatible indexes.

    Returns:
      A tuple of column names which are just the index of metric.compute_on(df).
    """
    extra_idx = self.extra_index[:]
    children_idx = [
        c.get_extra_idx(return_superset)
        for c in self.children
        if utils.is_metric(c)
    ]
    if len(set(children_idx)) > 1:
      if not return_superset:
        raise ValueError('Incompatible indexes!')
      children_idx_superset = set()
      children_idx_superset.update(*children_idx)
      children_idx = [list(children_idx_superset)]
    if children_idx:
      extra_idx += list(children_idx[0])
    return tuple(extra_idx)

  def traverse(self, include_self=True, include_constants=False):
    ms = [self] if include_self else list(self.children)
    while ms:
      m = ms.pop(0)
      if isinstance(m, Metric):
        ms += list(m.children)
        yield m
      elif include_constants:
        yield m

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

  def __eq__(self, other):
    if not isinstance(other, type(self)) or not isinstance(self, type(other)):
      return False
    if self.name != other.name:
      return False
    if self.get_fingerprint(['id']) != other.get_fingerprint(['id']):
      return False
    # Some Metrics share fingerprints. For example, Mean has the same
    # fingerprint as Sum(x) / Count(x) so we need to further check.
    if len(self.children) != len(other.children):
      return False
    for m1, m2 in zip(self.children, other.children):
      if m1 != m2:
        return False
    return True

  def __hash__(self):
    return hash((self.name, self.get_fingerprint()))

  def get_fingerprint(self, attr_to_exclude=()):
    """Returns attributes that uniquely identify the Metric.

    Metrics with the same fingerprint will compute to the same numbers on the
    same data. Note that name is not part of the fingerprint.

    Args:
      attr_to_exclude: Iterable of attributes to be excluded from fingerprint.

    Returns:
      A sorted tuple of (attribute, value) pairs that uniquely identify the
      Metric.
    """
    fingerprint = {'class': self.__class__}
    if self.where_:
      fingerprint['where'] = sorted(self.where_)
    # Caching across instances is tricky so only turned on for built-ins and
    # custom Metrics with cache_across_instances being True. Otherwise different
    # instances of the same class are always saved under different keys.
    if type(self).__name__ not in BUILT_INS and not self.cache_across_instances:
      fingerprint['id'] = id(self)
    if self.children:
      fingerprint['children'] = (
          c.get_fingerprint(attr_to_exclude) if isinstance(c, Metric) else c
          for c in self.children
      )
    if self.extra_split_by:
      fingerprint['extra_split_by'] = self.extra_split_by
    if self.extra_index != self.extra_split_by:
      fingerprint['extra_index'] = self.extra_index
    for k in self.additional_fingerprint_attrs:
      val = getattr(self, k, None)
      if isinstance(val, dict):
        for kw, arg in val.items():
          fingerprint['%s:%s' % (k, kw)] = arg
      elif isinstance(val, Metric):
        fingerprint[k] = val.get_fingerprint(attr_to_exclude)
      elif val is not None:
        fingerprint[k] = val
    fingerprint = {
        k: v for k, v in fingerprint.items() if k not in attr_to_exclude
    }
    for k, v in fingerprint.items():
      if not isinstance(v, str) and isinstance(v, Iterable):
        fingerprint[k] = tuple(list(v))
    return tuple(sorted(fingerprint.items()))

  def __str__(self):
    where = f' where {self.where}' if self.where else ''
    return self.name + where

  def __repr__(self):
    return self.__str__()

  def __deepcopy__(self, memo):
    # We don't copy self.cache, for two reasons.
    # 1. The copied Metric can share the same cache to maximize caching.
    # 2. When deepcopy a Metric, its cache refers to CacheKey and CacheKey
    # refers back. The loop leads to missing attributes error.
    cls = self.__class__
    obj = cls.__new__(cls)
    memo[id(self)] = obj
    for k, v in self.__dict__.items():
      if k == 'cache':
        obj.cache = v
      else:
        setattr(obj, k, copy.deepcopy(v, memo))
    return obj


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
    columns: The titles of the columns. If none will be automatically generated.
    And all other attributes inherited from Metric.
  """

  def __init__(self,
               children: Sequence[Metric],
               where: Optional[Union[Text, Sequence[Text]]] = None,
               children_return_dataframe: bool = True,
               name_tmpl=None,
               rename_columns=None):
    for m in children:
      if not isinstance(m, Metric):
        raise ValueError('%s is not a Metric.' % m)
    tmpl = name_tmpl or 'MetricList({})'
    if len(children) == 1:
      name = children[0].name
      name = name_tmpl.format(name) if name_tmpl else name
    else:
      name = tmpl.format(', '.join(m.name for m in children))
    super(MetricList, self).__init__(name, children, where, name_tmpl)
    self.children_return_dataframe = children_return_dataframe
    self.names = [m.name for m in children]
    self.columns = rename_columns

  def compute_slices(self, df, split_by=None):
    """Computes all Metrics with caching.

    We know df is not going to change so we can safely enable caching with an
    arbitrary key.

    Args:
      df: The DataFrame to compute on.
      split_by: The columns that we use to split the data.

    Returns:
      A list of results.
    """
    res = []
    for m in self:
      try:
        child = self.compute_util_metric_on(
            m, df, split_by, return_dataframe=self.children_return_dataframe)
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

  def compute_on_children(self, children, split_by):
    if isinstance(children, list):
      children = self.to_dataframe(children)
    if isinstance(children, pd.DataFrame):
      if self.name_tmpl:
        children.columns = [self.name_tmpl.format(c) for c in children.columns]
    elif not isinstance(children, pd.Series):
      children = pd.DataFrame({self.name: [children]})
    if self.columns:
      if len(children.columns) != len(self.columns):
        raise ValueError(
            'rename_columns has length %s but there are %s columns in '
            'the result!' % (len(self.columns), len(children.columns))
        )
      children.columns = self.columns
    return children

  def manipulate(  # pytype: disable=annotation-type-mismatch
      self,
      res: pd.Series,
      melted: bool = False,
      return_dataframe: bool = True,
      apply_name_tmpl: bool = None,
  ):
    """Rename columns if asked in addition to original manipulation."""
    res = super(MetricList, self).manipulate(
        res, melted, return_dataframe, apply_name_tmpl
    )
    if not isinstance(res, pd.DataFrame):
      return res
    if self.columns:
      if melted:
        res = utils.unmelt(res)
      if len(res.columns) != len(self.columns):
        raise ValueError(
            'rename_columns has length %s but there are %s columns in '
            'the result!' % (len(self.columns), len(res.columns))
        )
      res.columns = self.columns
      if melted:
        res = utils.melt(res)
    return res

  def to_dataframe(self, res):
    if not isinstance(res, (list, tuple)):
      return super(MetricList, self).to_dataframe(res)
    res_all = pd.concat(res, axis=1, sort=False)
    # In PY2, if index order are different, the names might get dropped.
    res_all.index.names = res[0].index.names
    return res_all

  def unwrap(self) -> List[Metric]:
    """Unwraps a MetricList and returns a list of all child Metrics.

    It recursively removes the MetricList wrapper and collects all children
    Metrics into a list.

    Returns:
      A list of Metric instances.
    """
    result = []
    for child in self.children:
      if not isinstance(child, Metric):
        raise ValueError('%s is not a Metric so cannot be unwrapped.' % child)
      if self.where_:
        child = copy.deepcopy(child)
        child.add_where(self.where_)
      if isinstance(child, MetricList):
        result.extend(child.unwrap())
      else:
        result.append(child)
    return result

  def rename_columns(self, rename_columns: List[Text]):
    """Rename the columns of the MetricList.

    Useful for instances where you have Metrics in the MetricList that are
    CompositeMetrics with undesirable names. Alters the name of the children
    inplace.

    Args:
      rename_columns: The names to rename the columns of the output dataframe
        to.

    Returns:
      None
    """
    self.columns = rename_columns

  def compute_on_sql(
      self,
      table,
      split_by=None,
      execute=None,
      melted=False,
      mode=None,
      cache_key=None,
      cache=None,
      return_dataframe=True,
  ):
    if return_dataframe:
      return super(MetricList, self).compute_on_sql(
          table, split_by, execute, melted, mode, cache_key, cache
      )
      # Returns a list of results without pd.concat.
    return self._compute_with_caching_and_postprocessing(
        self.compute_children_sql,
        table,
        split_by,
        melted,
        return_dataframe,
        False,
        cache_key,
        cache,
        execute,
        mode,
    )

  def compute_children_sql(self, table, split_by, execute, mode=None):
    """The return should be similar to compute_children()."""
    children = []
    for c in self.children:
      if not isinstance(c, Metric):
        children.append(c)
      else:
        children.append(
            self.compute_util_metric_on_sql(
                c,
                table,
                split_by + self.extra_split_by,
                execute,
                False,
                mode,
                return_dataframe=self.children_return_dataframe,
            )
        )
    return children

  def get_sql_and_with_clause(self, table, split_by, global_filter, indexes,
                              local_filter, with_data):
    """Gets the SQL query and WITH clause.

    The query is constructed by
    1. Get the query for every children metric.
    2. If all children queries are compatible, we just collect all the columns
      from the children and use the WHERE and GROUP BY clauses from any
      children.
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
    self.get_extra_idx()  # Check if indexes are compatible.
    local_filter = (
        sql.Filters(self.where_).add(local_filter).remove(global_filter)
    )
    children_sql = [
        c.get_sql_and_with_clause(table, split_by, global_filter, indexes,
                                  local_filter, with_data)[0]
        for c in self.children
    ]
    children_sql_copy = copy.deepcopy(children_sql)
    incompatible_sqls = sql.Datasources()
    child_table_aliases = []
    for i, child_sql in enumerate(children_sql):
      child_table_aliases.append(
          incompatible_sqls.merge(
              sql.Datasource(child_sql, 'MetricListChildTable')
          )
      )

    name_tmpl = self.name_tmpl or '{}'
    if len(incompatible_sqls) == 1:
      res = next(iter(incompatible_sqls.children.values()))
      for c in res.columns:
        if c not in indexes:
          c.alias_raw = name_tmpl.format(c.alias_raw)
      return res, with_data

    columns = sql.Columns(indexes.aliases)
    alias_lookup = {}
    from_data = None
    for i, child_sql in enumerate(children_sql_copy):
      child_table_alias = child_table_aliases[i]
      if child_table_alias in alias_lookup:
        alias = alias_lookup[child_table_alias]
      else:
        table = incompatible_sqls.children[child_table_alias]
        data = sql.Datasource(table, child_table_alias)
        alias = with_data.merge(data)
        alias_lookup[child_table_alias] = alias
        if i == 0:
          from_data = alias
        else:
          join = 'FULL' if indexes else 'CROSS'
          from_data = sql.Join(from_data, alias, join=join, using=indexes)
      for c in child_sql.columns:
        if c not in columns:
          columns.add(
              sql.Column(
                  '%s.%s' % (alias, c.alias),
                  alias=name_tmpl.format(c.alias_raw),
              )
          )

    query = sql.Sql(columns, from_data)
    if self.columns:
      columns = query.columns[len(indexes):]
      if len(columns) != len(self.columns):
        raise ValueError(
            'rename_columns has length %s but there are %s columns in '
            'the result!' % (len(self.columns), len(columns)))
      for col, rename in zip(columns, self.columns):
        col.set_alias(rename)  # Modify in-place.

    return query, with_data

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
    super(CompositeMetric, self).__init__(
        name,
        children,
        where,
        name_tmpl,
        additional_fingerprint_attrs=['name_tmpl'])
    self.op = op
    self.columns = rename_columns

  def rename_columns(self, rename_columns):
    self.columns = rename_columns
    return self

  def set_name(self, name):
    self.name = name
    return self

  def compute_children(
      self, df, split_by, melted=False, return_dataframe=True, cache_key=None
  ):
    del melted, return_dataframe, cache_key  # not used
    if len(self.children) != 2:
      raise ValueError('CompositeMetric can only have two children.')
    if not any([isinstance(m, Metric) for m in self.children]):
      raise ValueError('Must have at least one Metric.')
    children = []
    for m in self.children:
      if isinstance(m, Metric):
        m = self.compute_util_metric_on(
            m,
            df,
            split_by,
            # MetricList returns an undesired list when not return_dataframe.
            return_dataframe=isinstance(m, MetricList))
      children.append(m)
    return children

  def compute_on_children(self, children, split_by):
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
      children: A length-2 list. The elements could be numbers, pd.Series or
        pd.DataFrames.
      split_by: The columns that we use to split the data.

    Returns:
      The result to be sent to final_compute().
    """
    a, b = children[0], children[1]
    m1, m2 = self.children[0], self.children[1]
    if isinstance(m1, Metric):
      a = m1.to_series_or_number(a)
    if isinstance(m2, Metric):
      b = m2.to_series_or_number(b)
    res = None
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
      if len(a.columns) == len(b.columns):
        columns = [
            self.name_tmpl.format(c1, c2)
            for c1, c2 in zip(a.columns, b.columns)
        ]
        a.columns = columns
        b.columns = columns
    elif isinstance(a, pd.DataFrame):
      for i in range(len(a.columns)):
        a.iloc[:, i] = self.op(a.iloc[:, i], b)
      res = a
      columns = [
          self.name_tmpl.format(c, getattr(m2, 'name', m2)) for c in res.columns
      ]
      res.columns = columns
    elif isinstance(b, pd.DataFrame):
      for i in range(len(b.columns)):
        b.iloc[:, i] = self.op(a, b.iloc[:, i])
      res = b
      columns = [
          self.name_tmpl.format(getattr(m1, 'name', m1), c) for c in res.columns
      ]
      res.columns = columns

    if res is None:
      res = self.op(a, b)
    if isinstance(res, pd.Series):
      res.name = self.name
      if self.columns:
        res = pd.DataFrame(res)
    if self.columns is not None and isinstance(res, pd.DataFrame):
      res.columns = self.columns
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
    local_filter = (
        sql.Filters(self.where_).add(local_filter).remove(global_filter)
    )
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
      idx_aliases = sql.Columns(indexes).aliases
      val_cols0 = [c for c in query0.columns if c.alias not in idx_aliases]
      val_cols1 = [c for c in query1.columns if c.alias not in idx_aliases]
      if (
          len(val_cols0) != 1
          and len(val_cols1) != 1
          and len(val_cols0) != len(val_cols1)
      ):
        raise ValueError('Children Metrics have different shapes!')

      # Index columns can be in `groupby` or the first part of `columns`.
      idx0 = (
          sql.Columns(query0.groupby)
          .add(query0.columns[:-len(val_cols0)])
          .aliases
      )
      idx1 = (
          sql.Columns(query1.groupby)
          .add(query1.columns[:-len(val_cols1)])
          .aliases
      )
      has_same_idx = set(idx0) == set(idx1)
      if not has_same_idx:
        # If one index set is a subset of the other, we JOIN on the smaller set.
        shared_idx = idx0 if len(idx0) < len(idx1) else idx1
        if set(idx0).difference(idx1) and set(idx1).difference(idx0):
          raise ValueError(
              f'Indexes {idx0} and {idx1} are incompatible in'
              ' CompositeMetric!'
          )
      using = indexes if has_same_idx else shared_idx

      compatible = sql.is_compatible(query0, query1)
      # If two queries are compatible, merge them into one.
      if compatible and has_same_idx:
        col0_col1 = zip(itertools.cycle(query0.columns), query1.columns)
        if len(query1.columns) == 1:
          col0_col1 = zip(query0.columns, itertools.cycle(query1.columns))
        columns = sql.Columns()
        for c0, c1 in col0_col1:
          if c0.alias in idx_aliases:
            columns.add(c0)
          else:
            alias = self.name_tmpl.format(c0.alias_raw, c1.alias_raw)
            columns.add(op(c0, c1).set_alias(alias))
        query = copy.deepcopy(query0)
        query.columns = columns
      # If incompatible, add both queries to with_data and SELECT from the JOIN
      # of both.
      else:
        tbl0 = with_data.merge(sql.Datasource(query0, 'CompositeMetricTable0'))
        tbl1 = with_data.merge(sql.Datasource(query1, 'CompositeMetricTable1'))
        join = 'FULL' if using else 'CROSS'
        from_data = sql.Join(tbl0, tbl1, join=join, using=using)
        columns = sql.Columns(idx_aliases)
        col0_col1 = zip(itertools.cycle(val_cols0), val_cols1)
        if len(val_cols1) == 1:
          col0_col1 = zip(val_cols0, itertools.cycle(val_cols1))
        for c0, c1 in col0_col1:
          if c0.alias not in idx_aliases:
            col = op(
                sql.Column('%s.%s' % (tbl0, c0.alias), alias=c0.alias_raw),
                sql.Column('%s.%s' % (tbl1, c1.alias), alias=c1.alias_raw))
            columns.add(col)
        query = sql.Sql(columns, from_data)

    if len(query.columns.difference(indexes)) == 1:
      query.columns[-1].set_alias(self.name)

    if self.columns:
      columns = query.columns.difference(indexes)
      if len(self.columns) != len(columns):
        raise ValueError('The length of the renaming columns is wrong!')
      for col, rename in zip(columns, self.columns):
        col.set_alias(rename)  # Modify in-place.

    return query, with_data

  def get_fingerprint(self, attr_to_exclude=()):
    # Make Sum(x) / Count(x) indistinguishable to Mean(x) in cache.
    s = self.children[0]
    c = self.children[1]
    if isinstance(s, Sum) and isinstance(
        c, Count) and s.var == c.var and s.where == c.where and not c.distinct:
      return Mean(s.var, where=s.where_).get_fingerprint(attr_to_exclude)
    return super(CompositeMetric, self).get_fingerprint(attr_to_exclude)


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
    self.numerator = numerator
    self.denominator = denominator
    self.name = name or self.name

  def get_fingerprint(self, attr_to_exclude=()):
    # Make the fingerprint same as the equivalent CompositeMetric for caching.
    util = self.children[0] / self.children[1]
    util.where = self.where_  # pytype: disable=not-writable
    return util.get_fingerprint(attr_to_exclude)


class SimpleMetric(Metric):
  """Base class for common built-in aggregate functions of df.group_by()."""

  def __init__(self,
               var: Text,
               name: Optional[Text] = None,
               name_tmpl=None,
               where: Optional[Union[Text, Sequence[Text]]] = None,
               additional_fingerprint_attrs: Optional[List[str]] = None):
    name = name or name_tmpl.format(var)
    self.var = var
    additional_fingerprint_attrs = ['var', 'var2'] + (
        additional_fingerprint_attrs or [])
    super(SimpleMetric, self).__init__(
        name,
        None,
        where,
        name_tmpl,
        additional_fingerprint_attrs=additional_fingerprint_attrs)

  def get_sql_and_with_clause(self, table, split_by, global_filter, indexes,
                              local_filter, with_data):
    local_filter = (
        sql.Filters(self.where_).add(local_filter).remove(global_filter)
    )
    cols = self.get_sql_columns(local_filter)
    if cols:
      return sql.Sql(cols, table, global_filter, split_by), with_data
    equiv, _ = utils.get_fully_expanded_equivalent_metric_tree(self)
    return equiv.get_sql_and_with_clause(table, split_by, global_filter,
                                         indexes, local_filter, with_data)

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
    And all other attributes inherited from Metric.
  """

  def __init__(self,
               var: Text,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None,
               distinct: bool = False):
    self.distinct = distinct
    if distinct:
      name = name or 'count(distinct %s)' % str(var)
    super(Count, self).__init__(var, name, 'count({})', where, ['distinct'])

  def compute_slices(self, df, split_by=None):
    grped = self.group(df, split_by)[self.var]
    return grped.nunique() if self.distinct else grped.count()

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
    And all other attributes inherited from SimpleMetric.
  """

  def __init__(self,
               var: Text,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None):
    super(Sum, self).__init__(var, name, 'sum({})', where)

  def compute_slices(self, df, split_by=None):
    return self.group(df, split_by)[self.var].sum()

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
    And all other attributes inherited from SimpleMetric.
  """

  def __init__(self,
               var1: Text,
               var2: Text,
               normalize=False,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None):
    self.var2 = var2
    self.normalize = normalize
    name_tmpl = ('mean({} * %s)' if normalize else 'sum({} * %s)') % str(var2)
    super(Dot, self).__init__(var1, name, name_tmpl, where, ['normalize'])

  def compute_slices(self, df, split_by=None):
    if not split_by:
      prod = (df[self.var] * df[self.var2])
      return prod.mean() if self.normalize else prod.sum()
    if self.normalize:
      fn = lambda df: (df[self.var] * df[self.var2]).mean()
    else:
      fn = lambda df: (df[self.var] * df[self.var2]).sum()
    return df.groupby(split_by, observed=True).apply(fn)

  def get_equivalent_without_filter(self, *auxiliary_cols):
    if self.normalize:
      return Sum(auxiliary_cols[0]) / Count(auxiliary_cols[0])
    return Sum(auxiliary_cols[0])

  def get_auxiliary_cols(self):
    return ((self.var, '*', self.var2),)

  def get_sql_columns(self, local_filter):
    tmpl = 'AVG({} * {})' if self.normalize else 'SUM({} * {})'
    return sql.Column((self.var, self.var2), tmpl, self.name, local_filter)

  def get_fingerprint(self, attr_to_exclude=()):
    if str(self.var) > str(self.var2):
      util = copy.deepcopy(self)
      util.var = self.var2
      util.var2 = self.var
      return util.get_fingerprint(attr_to_exclude)
    return super(Dot, self).get_fingerprint(attr_to_exclude)


class Mean(SimpleMetric):
  """Mean estimator.

  Attributes:
    var: Column to compute on.
    weight: The column of weights.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    And all other attributes inherited from SimpleMetric.
  """

  def __init__(self,
               var: Text,
               weight: Optional[Text] = None,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None):
    name_tmpl = '%s-weighted mean({})' % str(weight) if weight else 'mean({})'
    super(Mean, self).__init__(var, name, name_tmpl, where, ['weight'])
    self.weight = weight

  def compute_slices(self, df, split_by=None):
    if self.weight:
      return self.compute_equivalent(df, split_by)
    return self.group(df, split_by)[self.var].mean()

  def get_sql_columns(self, local_filter):
    if not self.weight:
      return sql.Column(self.var, 'AVG({})', self.name, local_filter)
    else:
      res = sql.Column('%s * %s' % (self.weight, self.var), 'SUM({})',
                       'total_sum', local_filter)
      res /= sql.Column(self.weight, 'SUM({})', 'total_weight', local_filter)
      return res.set_alias(self.name)

  def get_equivalent_without_filter(self, *auxiliary_cols):
    del auxiliary_cols  # unused
    if not self.weight:
      return Sum(self.var) / Count(self.var)
    return Dot(self.var, self.weight) / Sum(self.weight)


class Max(SimpleMetric):
  """Max estimator.

  Attributes:
    var: Column to compute on.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    And all other attributes inherited from SimpleMetric.
  """

  def __init__(self,
               var: Text,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None):
    super(Max, self).__init__(var, name, 'max({})', where)

  def compute_slices(self, df, split_by=None):
    return self.group(df, split_by)[self.var].max()

  def get_sql_columns(self, local_filter):
    return sql.Column(self.var, 'MAX({})', self.name, local_filter)


class Min(SimpleMetric):
  """Min estimator.

  Attributes:
    var: Column to compute on.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    And all other attributes inherited from SimpleMetric.
  """

  def __init__(self,
               var: Text,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None):
    super(Min, self).__init__(var, name, 'min({})', where)

  def compute_slices(self, df, split_by=None):
    return self.group(df, split_by)[self.var].min()

  def get_sql_columns(self, local_filter):
    return sql.Column(self.var, 'MIN({})', self.name, local_filter)


class Nth(SimpleMetric):
  """The n-th (0-based indexing) value of var when sorting by sort_by.

  Attributes:
    var: Column to compute on.
    n: The `n`-th value to get.
    sort_by: Column to sort by.
    ascending: If to sort in ascending order.
    dropna: If to drop NA in var before counting.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    And all other attributes inherited from SimpleMetric.
  """

  def __init__(
      self,
      var: Text,
      n: int,
      sort_by: Text,
      ascending: bool = True,
      dropna: bool = False,
      name: Optional[Text] = None,
      where: Optional[Union[Text, Sequence[Text]]] = None,
      additional_fingerprint_attrs: Optional[List[str]] = None,
  ):
    if not isinstance(n, int):
      raise ValueError('n must be an integer.')
    if n < 0:
      n = -n - 1
      ascending = not ascending
    self.n = n
    self.ascending = ascending
    self.dropna = dropna
    self.sort_by = sort_by
    i = n + 1
    if i % 10 == 1 and i % 100 != 11:
      tmpl = f'{i}st'
    elif i % 10 == 2 and i % 100 != 12:
      tmpl = f'{i}nd'
    elif i % 10 == 3 and i % 100 != 13:
      tmpl = f'{i}rd'
    else:
      tmpl = f'{i}th'
    order = 'asc' if ascending else 'desc'
    name_tmpl = '%s({}) sort by %s %s' % (tmpl, sort_by, order)
    additional_fingerprint_attrs = (additional_fingerprint_attrs or []) + [
        'n',
        'sort_by',
        'dropna',
        'ascending',
    ]
    super(Nth, self).__init__(
        var,
        name,
        name_tmpl,
        where,
        additional_fingerprint_attrs=additional_fingerprint_attrs
    )

  def compute_slices(self, df, split_by=None):
    if self.dropna:
      df = df.dropna(subset=[self.var])
    df = df.sort_values(self.sort_by, ascending=self.ascending)
    if split_by:
      res = self.group(df[split_by + [self.var]], split_by).nth(self.n)
      return res.set_index(split_by)[self.var].sort_index()
    if self.n > len(df) - 1:
      return np.nan
    return df[self.var].values[self.n]

  def get_sql_and_with_clause(self, table, split_by, global_filter, indexes,
                              local_filter, with_data):
    """Gets the SQL query and WITH clause.

    If there is no local filter, the metric can be expressed in one line like
    ARRAY_AGG(var IGNORE NULLS ORDER BY sort_by LIMIT n + 1)[SAFE_OFFSET(n)]. In
    that case we will fall back to get_sql_columns().
    Otherwise the metric requires multiple subqueries. We will first add
    SELECT split_by, var, sort_by FROM table WHERE local_filter + global_filter
    to with_data
    then generate one line query like above on the subquery.

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
    if not local_filter:
      return super(Nth, self).get_sql_and_with_clause(
          table, split_by, global_filter, indexes, None, with_data
      )
    all_filters = sql.Filters(local_filter).add(global_filter)
    if self.dropna:
      all_filters.add(f'{self.var} IS NOT NULL')
    split_by = sql.Columns(split_by)
    var = sql.Column(self.var, alias=self.var)
    sort_by = sql.Column(self.sort_by, alias=self.sort_by)
    filtered_sql = sql.Sql(
        sql.Columns(split_by).add([var, sort_by]), table, all_filters
    )
    filtered_table = sql.Datasource(filtered_sql, 'WeightedQuantileFiltered')
    filtered_table_alias = with_data.merge(filtered_table)
    no_filter = Nth(
        var.alias,
        self.n,
        sort_by.alias,
        dropna=self.dropna,
        ascending=self.ascending,
        name=self.name,
    )
    return super(Nth, no_filter).get_sql_and_with_clause(
        filtered_table_alias, split_by.aliases, None, indexes, None, with_data
    )

  def get_sql_columns(self, local_filter):
    if local_filter:
      raise ValueError(
          'This case should be handled by get_sql_and_with_clause() already.'
      )
    order = '' if self.ascending else ' DESC'
    dropna = ' IGNORE NULLS' if self.dropna else ''
    sql_tmpl = 'ARRAY_AGG({}%s ORDER BY %s%s LIMIT %s)[SAFE_OFFSET(%s)]' % (
        dropna,
        self.sort_by,
        order,
        self.n + 1,
        self.n,
    )
    return sql.Column(
        self.var,
        sql_tmpl,
        self.name,
    )


class Quantile(SimpleMetric):
  """Quantile estimator.

  Attributes:
    var: Column to compute on. NA values will be dropped.
    quantile: Same as the arg "q" in np.quantile().
    weight: The column of weights. If specified, we always return a DataFrame.
      Weights that are NA will be dropped.
    interpolation: As the same arg in pd.Series.quantile(). No effect if
      'weight' is specified.
    one_quantile: If quantile is a number or an iterable.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter.
    And all other attributes inherited from SimpleMetric.
  """

  def __init__(self,
               var: Text,
               quantile: Union[float, int, Sequence[Union[float, int]]] = 0.5,
               weight: Optional[Text] = None,
               interpolation='linear',
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None):
    if isinstance(quantile, (int, float)):
      self.one_quantile = True
    else:
      self.one_quantile = False
      quantile = np.array(quantile)
      if len(quantile) == 1:
        quantile = quantile[0]
        self.one_quantile = True
    if self.one_quantile and not 0 <= quantile <= 1:
      raise ValueError('quantiles must be in [0, 1].')
    if not self.one_quantile and not (np.all(quantile >= 0) and
                                      np.all(quantile <= 1)):
      raise ValueError('quantiles must be in [0, 1].')
    name_tmpl = 'quantile({}, {})'
    if weight:
      name_tmpl = '%s-weighted quantile({}, {})' % str(weight)
    name = name or name_tmpl.format(var, str(quantile))
    self.quantile = quantile
    self.interpolation = interpolation
    self.weight = weight
    super(Quantile, self).__init__(var, name, name_tmpl, where,
                                   ['quantile', 'weight', 'interpolation'])

  def compute_slices(self, df, split_by=None):
    if self.weight:
      # Adapted from https://stackoverflow.com/a/29677616/12728137.
      def interp(d):
        res = np.interp(self.quantile, d[self.weight], d[self.var])
        if self.one_quantile:
          return res
        return pd.DataFrame(
            [res],
            columns=[self.name_tmpl.format(self.var, q) for q in self.quantile])

      aggregated_weight = df.groupby(split_by + [self.var])[self.weight].sum()
      # See https://en.wikipedia.org/wiki/Percentile#Weighted_percentile.
      weighted_quantiles = (
          self.group(aggregated_weight, split_by).cumsum()
          - 0.5 * aggregated_weight
      )
      weighted_quantiles /= self.group(aggregated_weight, split_by).sum()
      if split_by:
        weighted_quantiles = weighted_quantiles.reset_index(self.var)
        return self.group(weighted_quantiles, split_by).apply(interp)
      else:
        weighted_quantiles = weighted_quantiles.to_frame().reset_index()
        return interp(weighted_quantiles)

    res = self.group(df, split_by)[self.var].quantile(
        self.quantile, interpolation=self.interpolation)
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
      raise ValueError('SQL for weighted quantile should already be handled!')
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

  def get_sql_and_with_clause(self, table, split_by, global_filter, indexes,
                              local_filter, with_data):
    """Gets the SQL for weighted quantile.

    The query is constructed as following.
    1. Add three subqueries below to the WITH clause:
      AggregatedQuantileWeights AS (SELECT
        split_by,
        val,
        SUM(weight) AS weight
      FROM T
      WHERE val IS NOT NULL AND weight IS NOT NULL AND weight != 0
      GROUP BY split_by, val),
      QuantileWeights AS (SELECT
        split_by,
        val,
        SAFE_DIVIDE(SUM(weight) OVER (PARTITION BY split_by ORDER BY val
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
          - 0.5 * weight,
          SUM(weight) OVER (PARTITION BY split_by)) AS weight
      FROM AggregatedQuantileWeights
      WHERE weight IS NOT NULL AND weight != 0
      ORDER BY split_by, val),
      PairedQuantileWeights AS (SELECT
        split_by,
        val,
        weight,
        LAG(weight) OVER (PARTITION BY split_by ORDER BY val) AS prev_weight,
        LEAD(weight) OVER (PARTITION BY split_by ORDER BY val) AS next_weight,
        LEAD(val) OVER (PARTITION BY split_by ORDER BY val) AS next_value
      FROM QuantileWeights)
    2. For each quantile q, SELECT
    SUM(IF((prev_weight IS NULL AND q < weight) OR
             (next_weight IS NULL AND q > weight),
           val,
           IF(q BETWEEN weight AND next_weight,
              (next_value * (q - weight) + (next_weight - q) * val) /
                (next_weight - weight),
              0))).

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
    if not self.weight:  # Fall back to get_sql_columns().
      return super(Quantile, self).get_sql_and_with_clause(
          table, split_by, global_filter, indexes, local_filter, with_data
      )
    if self.interpolation != 'linear':
      raise NotImplementedError('Only linear interpolation is supported!')
    local_filter = (
        sql.Filters(self.where_).add(local_filter).remove(global_filter)
    )
    split_by_and_value = sql.Columns(split_by).add(self.var)
    weight = sql.Column(
        self.weight, 'SUM({})', filters=local_filter, alias=self.weight
    )
    cols = sql.Columns(split_by_and_value).add(weight)
    deduped_weight_sql = sql.Sql(
        cols,
        table,
        sql.Filters(global_filter).add((
            f'{self.var} IS NOT NULL',
            f'{self.weight} IS NOT NULL',
            f'{self.weight} != 0',
        )),
        split_by_and_value,
    )
    deduped_weight_alias = with_data.merge(
        sql.Datasource(deduped_weight_sql, 'AggregatedQuantileWeights')
    )

    v = split_by_and_value.aliases[-1]
    w = weight.alias
    split_by = sql.Columns(split_by.aliases)
    split_by_and_value = sql.Columns(split_by_and_value.aliases)
    total_weight = sql.Column(w, 'SUM({})', partition=split_by)
    cum_weight = sql.Column(
        w,
        'SUM({})',
        partition=split_by,
        order=v,
        window_frame='ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW',
    )
    normalized_weights = (cum_weight - 0.5 * sql.Column(w)) / total_weight
    cols = sql.Columns(split_by_and_value).add(normalized_weights.set_alias(w))
    normalized_weights_sql = sql.Sql(
        cols,
        deduped_weight_alias,
        where=(f'{w} IS NOT NULL', f'{w} != 0'),
        orderby=split_by_and_value,
    )
    normalized_weights_alias = with_data.merge(
        sql.Datasource(normalized_weights_sql, 'QuantileWeights')
    )

    prev_w = sql.Column(
        w,
        'LAG({})',
        'prev_weight',
        partition=split_by,
        order=v
    )
    next_w = sql.Column(
        w,
        'LEAD({})',
        'next_weight',
        partition=split_by,
        order=v
    )
    next_val = sql.Column(
        v,
        'LEAD({})',
        'next_value',
        partition=split_by,
        order=v
    )
    paired_weights_cols = sql.Columns(cols.aliases).add(
        (prev_w, next_w, next_val)
    )
    paired_weights_sql = sql.Sql(paired_weights_cols, normalized_weights_alias)
    paired_weights_alias = with_data.merge(
        sql.Datasource(paired_weights_sql, 'PairedQuantileWeights')
    )

    prev_w = prev_w.alias
    next_w = next_w.alias
    next_v = next_val.alias
    cols = sql.Columns(split_by)
    quantiles = [self.quantile] if self.one_quantile else self.quantile
    for q in quantiles:
      interp = (
          f'({next_v} * ({q} - {w}) + ({next_w} - {q}) * {v})'
          f' / ({next_w} - {w})'
      )
      cols.add(
          sql.Column(
              f"""SUM(IF(({prev_w} IS NULL AND {q} < {w}) OR ({next_w} IS NULL AND {q} > {w}), {v},
    IF({q} BETWEEN {w} AND {next_w}, {interp}, 0)))""",
              alias=f'{self.weight}-weighted quantile({self.var}, {q})',
          )
      )
    if self.one_quantile:
      cols[-1].set_alias(self.name)
    res_sql = sql.Sql(cols, paired_weights_alias, groupby=split_by)
    return res_sql, with_data


class Variance(SimpleMetric):
  """Variance estimator.

  Attributes:
    var: Column to compute on.
    ddof: Degree of freedom to use in pd.DataFrame.var(). If ddof is larger than
      the degree of freedom in the data, we return NA.
    weight: The column of weights.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter. And all other attributes inherited from
      SimpleMetric.
  """

  def __init__(self,
               var: Text,
               unbiased: bool = True,
               weight: Optional[Text] = None,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None):
    self.ddof = 1 if unbiased else 0
    self.weight = weight
    name_tmpl = '%s-weighted var({})' % str(weight) if weight else 'var({})'
    super(Variance, self).__init__(var, name, name_tmpl, where,
                                   ['ddof', 'weight'])

  def compute_slices(self, df, split_by=None):
    if self.weight:
      return self.compute_equivalent(df, split_by)
    return self.group(df, split_by)[self.var].var(ddof=self.ddof)

  def get_equivalent_without_filter(self, *auxiliary_cols):
    if not self.weight:
      return Cov(self.var, self.var, ddof=self.ddof)
    numer = Dot(auxiliary_cols[0],
                self.weight) - Dot(self.var, self.weight)**2 / Sum(self.weight)
    denom = Sum(self.weight) - self.ddof if self.ddof else Sum(self.weight)
    # ddof is invalid if it makes the denom negative so we use ((denom)^0.5)^2.
    return numer / (denom**0.5)**2

  def get_auxiliary_cols(self):
    if self.weight:
      return ((self.var, '**', 2),)
    return ()

  def get_sql_columns(self, local_filter):
    if self.weight:
      return
    if self.ddof == 1:
      return sql.Column(self.var, 'VAR_SAMP({})', self.name, local_filter)
    else:
      return sql.Column(self.var, 'VAR_POP({})', self.name, local_filter)


class StandardDeviation(SimpleMetric):
  """Standard Deviation estimator.

  Attributes:
    var: Column to compute on.
    ddof: Degree of freedom to use in pd.DataFrame.std(). If ddof is larger than
      the degree of freedom in the data, we return NA.
    weight: The column of weights.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter. And all other attributes inherited from
      SimpleMetric.
  """

  def __init__(self,
               var: Text,
               unbiased: bool = True,
               weight: Optional[Text] = None,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None):
    self.ddof = 1 if unbiased else 0
    self.weight = weight
    name_tmpl = '%s-weighted sd({})' % str(weight) if weight else 'sd({})'
    super(StandardDeviation, self).__init__(var, name, name_tmpl, where,
                                            ['ddof', 'weight'])

  def compute_slices(self, df, split_by=None):
    if self.weight:
      return self.compute_equivalent(df, split_by)
    return self.group(df, split_by)[self.var].std(ddof=self.ddof)

  def get_equivalent_without_filter(self, *auxiliary_cols):
    del auxiliary_cols  # unused
    return Variance(self.var, bool(self.ddof), self.weight) ** 0.5

  def get_sql_columns(self, local_filter):
    if self.weight:
      return
    if self.ddof == 1:
      return sql.Column(self.var, 'STDDEV_SAMP({})', self.name, local_filter)
    else:
      return sql.Column(self.var, 'STDDEV_POP({})', self.name, local_filter)


class CV(SimpleMetric):
  """Coefficient of variation estimator.

  Attributes:
    var: Column to compute on.
    ddof: Degree of freedom to use. If ddof is larger than the degree of freedom
      in the data, we return NA.
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter. And all other attributes inherited from
      SimpleMetric.
  """

  def __init__(self,
               var: Text,
               unbiased: bool = True,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None):
    self.ddof = 1 if unbiased else 0
    super(CV, self).__init__(var, name, 'cv({})', where, ['ddof'])

  def compute_slices(self, df, split_by=None):
    var_grouped = self.group(df, split_by)[self.var]
    return var_grouped.std(ddof=self.ddof) / var_grouped.mean()

  def get_equivalent_without_filter(self, *auxiliary_cols):
    del auxiliary_cols  # unused
    return StandardDeviation(self.var, bool(self.ddof)) / Mean(self.var)

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
    And all other attributes inherited from SimpleMetric.
  """

  def __init__(self,
               var1: Text,
               var2: Text,
               weight: Optional[Text] = None,
               name: Optional[Text] = None,
               method='pearson',
               where: Optional[Union[Text, Sequence[Text]]] = None):
    name_tmpl = 'corr({}, {})'
    if weight:
      name_tmpl = '%s-weighted corr({}, {})' % str(weight)
    if name is None:
      name = name_tmpl.format(var1, var2)
    self.var2 = var2
    self.method = method
    self.weight = weight
    super(Correlation, self).__init__(var1, name, name_tmpl, where,
                                      ['method', 'weight'])

  def compute_slices(self, df, split_by=None):
    if self.weight and self.method != 'pearson':
      raise NotImplementedError(
          'Only Pearson correlation is supported in weighted Correlation!'
      )
    if self.weight:
      return self.compute_equivalent(df, split_by)
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
    return self.group(df, split_by)[self.var].corr(
        df[self.var2], method=self.method)

  def get_equivalent_without_filter(self, *auxiliary_cols):
    del auxiliary_cols  # unused
    if self.method == 'pearson':
      return (
          Cov(self.var, self.var2, True, weight=self.weight)
          / StandardDeviation(self.var, False, self.weight)
          / StandardDeviation(self.var2, False, self.weight)
      )

  def get_sql_columns(self, local_filter):
    if self.weight:
      return
    if self.method != 'pearson':
      raise ValueError('Only Pearson correlation is supported!')
    return sql.Column((self.var, self.var2), 'CORR({}, {})', self.name,
                      local_filter)

  def get_fingerprint(self, attr_to_exclude=()):
    if str(self.var) > str(self.var2):
      util = copy.deepcopy(self)
      util.var = self.var2
      util.var2 = self.var
      return util.get_fingerprint(attr_to_exclude)
    return super(Correlation, self).get_fingerprint(attr_to_exclude)


class Cov(SimpleMetric):
  """Covariance estimator.

  Attributes:
    var: Column of first variable.
    var2: Column of second variable.
    bias: The same arg passed to np.cov().
    ddof: The same arg passed to np.cov().  If ddof is larger than the degree of
      freedom in the data, we return NA.
    weight: Column name of aweights.
    fweight: Column name of fweights. We will convert the values to integer
      because that's required by definition. For the definitions of aweights and
      fweight, see the cod of numpy.cov().
    name: Name of the Metric.
    where: A string or list of strings to be concatenated that will be passed to
      df.query() as a prefilter. And all other attributes inherited from
      SimpleMetric.
  """

  def __init__(self,
               var1: Text,
               var2: Text,
               bias: bool = False,
               ddof: Optional[int] = None,
               weight: Optional[Text] = None,
               fweight: Optional[Text] = None,
               name: Optional[Text] = None,
               where: Optional[Union[Text, Sequence[Text]]] = None):
    name_tmpl = 'cov({}, {})'
    if weight:
      name_tmpl = '%s-weighted %s' % (str(weight), name_tmpl)
    if fweight:
      name_tmpl = '%s-fweighted %s' % (str(fweight), name_tmpl)

    if name is None:
      name = name_tmpl.format(var1, var2)
    self.var2 = var2
    self.bias = bias
    self.ddof = ddof
    self.weight = weight
    self.fweight = fweight
    super(Cov, self).__init__(var1, name, name_tmpl, where,
                              ['bias', 'ddof', 'weight', 'fweight'])

  def compute_slices(self, df, split_by=None):
    return self.compute_equivalent(df, split_by)

  def get_equivalent_without_filter(self, *auxiliary_cols):
    """Gets the equivalent Metric for Cov."""
    # See https://numpy.org/doc/stable/reference/generated/numpy.cov.html.
    ddof = self.ddof if self.ddof is not None else int(not self.bias)
    if not self.weight:
      if not self.fweight:
        v1 = Count(self.var)
        v2 = v1
        res = Dot(self.var, self.var2, normalize=True) - Mean(self.var) * Mean(
            self.var2
        )
      else:
        v1 = Sum(self.fweight)
        v2 = v1
        res = Mean(auxiliary_cols[0], self.fweight) - Mean(
            self.var, self.fweight) * Mean(self.var2, self.fweight)
    elif not self.fweight:
      v1 = Sum(self.weight)
      v2 = Dot(self.weight, self.weight)
      res = Mean(auxiliary_cols[0], self.weight) - Mean(
          self.var, self.weight) * Mean(self.var2, self.weight)
    else:
      v1 = Dot(self.weight, self.fweight)
      v2 = Dot(auxiliary_cols[1], self.weight)
      res = Mean(auxiliary_cols[0], auxiliary_cols[1]) - Mean(
          self.var, auxiliary_cols[1]) * Mean(self.var2, auxiliary_cols[1])

    # ddof is invalid if it makes the denom negative so we use ((denom)^0.5)^2.
    if ddof:
      if v1 != v2:
        res /= ((1 - ddof * v2 / v1**2) ** 0.5) ** 2
      else:
        res /= ((1 - ddof / v1) ** 0.5) ** 2
    return res

  def get_auxiliary_cols(self):
    if not self.weight and not self.fweight:
      return ()
    if not self.weight or not self.fweight:
      return ((self.var, '*', self.var2),)
    return (
        (self.var, '*', self.var2),
        (self.fweight, '*', self.weight),
    )

  def get_sql_columns(self, local_filter):
    """Get SQL columns."""
    if self.weight or self.fweight:
      return
    ddof = self.ddof
    if ddof is None:
      ddof = 0 if self.bias else 1
    if ddof == 1:
      return sql.Column(
          (self.var, self.var2), 'COVAR_SAMP({}, {})', self.name, local_filter
      )
    elif ddof == 0:
      return sql.Column(
          (self.var, self.var2), 'COVAR_POP({}, {})', self.name, local_filter
      )
    return

  def get_fingerprint(self, attr_to_exclude=()):
    if str(self.var) > str(self.var2):
      util = copy.deepcopy(self)
      util.var = self.var2
      util.var2 = self.var
      return util.get_fingerprint(attr_to_exclude)
    return super(Cov, self).get_fingerprint(attr_to_exclude)
