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
"""Models that can be fitted in Meterstick."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import itertools
from typing import List, Optional, Sequence, Text, Union

from meterstick import metrics
from meterstick import operations
from meterstick import sql
from meterstick import utils
import numpy as np
import pandas as pd
from sklearn import linear_model


class Model(operations.Operation):
  """Base class for model fitting."""

  def __init__(
      self,
      y: Optional[metrics.Metric] = None,
      x: Optional[
          Union[metrics.Metric, Sequence[metrics.Metric], metrics.MetricList]
      ] = None,
      group_by: Optional[Union[Text, List[Text]]] = None,
      model=None,
      model_name=None,
      where=None,
      name=None,
      fit_intercept=True,
      normalize=False,
      additional_fingerprint_attrs: Optional[List[str]] = None,
  ):
    """Initialize the model.

    Args:
      y: The Metric whose result will be used as the response variable.
      x: The Metrics whose results will be used as the explanatory variables.
      group_by: The column(s) to aggregate and compute x and y. The model will
        be fit on MetricList([y, x]).compute_on(df, group_by).
      model: The model to fit. It's either a sklearn.linear_model or obeys the
        API convention, namely, has a method fit(X, y) and attributes
        model.coef_ and model.intercept_.
      model_name: The name of the model, will be used to auto-generate a name if
        name is not given.
      where: A string or list of strings to be concatenated that will be passed
        to df.query() as a prefilter.
      name: The name to use for the model.
      fit_intercept: If to include intercept in the model.
      normalize: This parameter is ignored when fit_intercept is False. If True,
        the regressors X will be normalized before regression by subtracting the
        mean and dividing by the l2-norm.
      additional_fingerprint_attrs: Additioinal attributes to be encoded into
        the fingerprint. See get_fingerprint() for how it's used.
    """
    if y and not isinstance(y, metrics.Metric):
      raise ValueError('y must be a Metric!')
    if y and operations.count_features(y) != 1:
      raise ValueError(
          'y must be a 1D array but is %iD!' % operations.count_features(y)
      )
    if isinstance(x, metrics.Metric):
      x = [x]
    child = None
    if x and y:
      child = metrics.MetricList([y] + x)
    self.model = model
    self.model_name = model_name
    additional_fingerprint_attrs = (
        [additional_fingerprint_attrs]
        if isinstance(additional_fingerprint_attrs, str)
        else list(additional_fingerprint_attrs or [])
    )
    self.name_ = None
    self.name_tmpl_ = None
    super(Model, self).__init__(
        child,
        None,
        group_by,
        [],
        name=name,
        where=where,
        additional_fingerprint_attrs=['fit_intercept', 'normalize']
        + additional_fingerprint_attrs,
    )
    self.fit_intercept = fit_intercept
    self.normalize = normalize

  def compute(self, df):
    x, y = df.iloc[:, 1:], df.iloc[:, 0]
    if self.normalize and self.fit_intercept:
      x_scaled = x - x.mean()
      norms = np.sqrt((x_scaled**2).sum())
      x = x_scaled / norms
    self.model.fit(x, y)
    coef = self.model.coef_
    if self.normalize and self.fit_intercept:
      coef = coef / norms.values
    names = list(df.columns[1:])
    if self.fit_intercept:
      if self.normalize:
        intercept = y.mean() - df.iloc[:, 1:].mean().dot(coef)
      else:
        intercept = self.model.intercept_
      coef = [intercept] + list(coef)
      names = ['intercept'] + names
    return pd.DataFrame([coef], columns=names)

  def compute_through_sql(self, table, split_by, execute, mode):
    try:
      if mode == 'magic':
        if self.where:
          table = sql.Sql(None, table, self.where_)
        res = self.compute_on_sql_magic_mode(table, split_by, execute)
        return utils.apply_name_tmpl(self.name_tmpl, res)
      return super(Model, self).compute_through_sql(
          table, split_by, execute, mode
      )
    except NotImplementedError:
      raise
    except Exception as e:  # pylint: disable=broad-except
      msg = (
          "Please see the root cause of the failure above. If it's caused by"
          ' the query being too large/complex, you can try '
          "compute_on_sql(..., mode='%s')."
      )
      if mode == 'magic':
        raise ValueError(msg % 'mixed') from e
      raise ValueError(msg % 'magic') from e

  def compute_on_sql_magic_mode(self, table, split_by, execute):
    raise NotImplementedError

  @property
  def y(self):
    if not self.children or not isinstance(
        self.children[0], metrics.MetricList
    ):
      raise ValueError('y must be a Metric!')
    return self.children[0][0]

  @property
  def x(self):
    if not self.children or not isinstance(
        self.children[0], metrics.MetricList
    ):
      raise ValueError('x must be a MetricList!')
    return metrics.MetricList(self.children[0][1:])

  @property
  def k(self):
    return operations.count_features(self.x)

  @property
  def name(self):
    if self.name_:
      return self.name_
    if not self.children:
      return self.model_name
    x_names = [m.name for m in self.x]
    return '%s(%s ~ %s)' % (
        self.model_name,
        self.y.name,
        ' + '.join(x_names),
    )

  @name.setter
  def name(self, name):
    self.name_ = name

  @property
  def name_tmpl(self):
    if self.name_tmpl_:
      return self.name_tmpl_
    return self.name + ' Coefficient: {}'

  @name_tmpl.setter
  def name_tmpl(self, name_tmpl):
    self.name_tmpl_ = name_tmpl

  @property
  def group_by(self):
    return self.extra_split_by

  def __call__(self, child: metrics.Metric):
    model = copy.deepcopy(self) if self.children else self
    model.children = (child,)
    return model

  def get_extra_idx(self, return_superset=False):
    # Model's extra indexes don't apply to descendants.
    return ()


class LinearRegression(Model):
  """A class that can fit a linear regression."""

  def __init__(
      self,
      y: Optional[metrics.Metric] = None,
      x: Optional[
          Union[metrics.Metric, Sequence[metrics.Metric], metrics.MetricList]
      ] = None,
      group_by: Optional[Union[Text, List[Text]]] = None,
      fit_intercept: bool = True,
      normalize: bool = False,
      where: Optional[str] = None,
      name: Optional[str] = None,
  ):
    """Initialize a sklearn.LinearRegression model."""
    model = linear_model.LinearRegression(fit_intercept=fit_intercept)
    super(LinearRegression, self).__init__(
        y, x, group_by, model, 'OLS', where, name, fit_intercept, normalize
    )

  def compute_on_sql_magic_mode(self, table, split_by, execute):
    return Ridge(
        self.y,
        self.x,
        self.group_by,
        0,
        self.fit_intercept,
        self.normalize,
        self.where_,
        self.name,
    ).compute_on_sql_magic_mode(table, split_by, execute)


class Ridge(Model):
  """A class that can fit a ridge regression."""

  def __init__(
      self,
      y: Optional[metrics.Metric] = None,
      x: Optional[
          Union[metrics.Metric, Sequence[metrics.Metric], metrics.MetricList]
      ] = None,
      group_by: Optional[Union[Text, List[Text]]] = None,
      alpha=1,
      fit_intercept: bool = True,
      normalize: bool = False,
      where: Optional[str] = None,
      name: Optional[str] = None,
      copy_X=True,
      max_iter=None,
      tol=0.001,
      solver='auto',
      random_state=None,
  ):
    """Initialize a sklearn.Ridge model."""
    model = linear_model.Ridge(
        alpha=alpha,
        fit_intercept=fit_intercept,
        copy_X=copy_X,
        max_iter=max_iter,
        tol=tol,
        solver=solver,
        random_state=random_state,
    )
    super(Ridge, self).__init__(
        y,
        x,
        group_by,
        model,
        'Ridge',
        where,
        name,
        fit_intercept,
        normalize,
        ['alpha'],
    )
    self.alpha = alpha

  def compute_on_sql_magic_mode(self, table, split_by, execute):
    # Never normalize for the sufficient_stats. Normalization is handled in
    # compute_ridge_coefs() instead.
    xs, sufficient_stats, _, _ = get_sufficient_stats_elements(
        self, table, split_by, execute, normalize=False, include_n_obs=True
    )
    return apply_algorithm_to_sufficient_stats_elements(
        sufficient_stats, split_by, compute_ridge_coefs, xs, self
    )


def get_sufficient_stats_elements(
    m,
    table,
    split_by,
    execute,
    normalize=None,
    include_n_obs=False,
):
  """Computes the elements of X'X and X'y.

  Args:
    m: A Model instance.
    table: The table we want to query from.
    split_by: The columns that we use to split the data.
    execute: A function that can executes a SQL query and returns a DataFrame.
    normalize: If to normalize the X. Note that only has effect when
      m.fit_intercept is True, which is consistent to sklearn.
    include_n_obs: If to include the number of observations in the return.

  Returns:
    xs: A list of the column names of x1, x2, ...
    sufficient_stats_elements: A DataFrame holding all unique elements of
      sufficient stats. Each row corresponds to one slice in split_by. The
      columns are
        split_by,
        avg(x0), avg(x1), ...,  # if fit_intercept
        avg(x0 * x0), avg(x0 * x1), avg(x0 * x2), avg(x1 * x2), ...,
        avg(y),  # if fit_intercept
        avg(x0 * y), avg(x1 * y), ...,
        n_observation  # if include_n_obs.
      The column are named as
        split_by, x0, x1,..., x0x0, x0x1,..., y, x0y, x1y,..., n_obs.
    avg_x: Nonempty only when normalize. A pd.DataFrame which holds the
      avg(x0), avg(x1), ... of the UNNORMALIZED x.
      Don't confuse it with the ones in the sufficient_stats_elements, which are
      the average of normalized x, which are just 0s.
    norms: Nonempty only when normalize. A pd.DataFrame which holds the l2-norm
      values of all centered-x columns.
  """
  if normalize is None:
    normalize = m.normalize and m.fit_intercept
  table, with_data, xs_cols, y, avg_x, norms = get_data(
      m, table, split_by, execute, normalize
  )
  xs = xs_cols.aliases
  x_t_x = []
  x_t_y = []
  if m.fit_intercept:
    if not normalize:
      x_t_x = [sql.Column(f'AVG({x})', alias=f'x{i}') for i, x in enumerate(xs)]
    x_t_y = [sql.Column(f'AVG({y})', alias='y')]
  for i, x1 in enumerate(xs):
    for j, x2 in enumerate(xs[i:]):
      x_t_x.append(sql.Column(f'AVG({x1} * {x2})', alias=f'x{i}x{i + j}'))
  x_t_y += [
      sql.Column(f'AVG({x} * {y})', alias=f'x{i}y') for i, x in enumerate(xs)
  ]
  cols = sql.Columns(x_t_x + x_t_y)
  if include_n_obs:
    cols.add(sql.Column('COUNT(*)', alias='n_obs'))
  sufficient_stats_elements = sql.Sql(
      cols, table, groupby=sql.Columns(split_by).aliases, with_data=with_data
  )
  sufficient_stats_elements = execute(str(sufficient_stats_elements))
  if normalize:
    col_names = list(sufficient_stats_elements.columns)
    avg_x_names = [f'x{i}' for i in range(len(xs))]
    sufficient_stats_elements[avg_x_names] = 0
    sufficient_stats_elements = sufficient_stats_elements[
        col_names[: len(split_by)] + avg_x_names + col_names[len(split_by) :]
    ]
  return xs_cols, sufficient_stats_elements, avg_x, norms


def get_data(m, table, split_by, execute, normalize=False):
  """Retrieves the data that the model will be fit on.

  We compute a Model by first computing its children, and then fitting
  the model on it. This function retrieves the necessary variables to compute
  the children.
  We first get the result of m.to_sql(table, split_by + m.group_by). If
  `normalize` is False, we already get what we need. Otherwise we center and
  normalize the columns for `x`s and returns the centered-and-normalized table,
  together with the average of `x`s and the norms of centered `x`s.

  Args:
    m: A Model instance.
    table: The table we want to query from.
    split_by: The columns that we use to split the data.
    execute: A function that can executes a SQL query and returns a DataFrame.
    normalize: If the Model normalizes x.

  Returns:
    table: A string representing the table name which we can query from. The
      table has columns `split_by`, y, x1, x2, .... If normalize is True, x
      columns are centered then normalized.
    with_data: The WITH clause that holds all necessary subqueries so we can
      query the `table`.
    xs_cols: A list of the sql.Columns of x1, x2, ...
    y: The column name of the y column.
    avgs: Nonempty only when normalize is True. A pd.DataFrame which holds the
      average of all x and y columns.
    norms: Nonempty only when normalize is True. A pd.DataFrame which holds the
      l2-norm values of all centered-x columns.
  """
  data = m.children[0].to_sql(table, split_by + m.group_by)
  with_data = data.with_data
  data.with_data = None
  table = with_data.merge(sql.Datasource(data, 'DataToFit'))
  y = data.columns[-m.k - 1].alias
  xs_cols = sql.Columns(data.columns[-m.k :])
  if not normalize:
    return table, with_data, xs_cols, y, pd.DataFrame(), pd.DataFrame()

  xs = xs_cols.aliases
  split_by = sql.Columns(split_by).aliases
  avg_x_and_y = sql.Columns([sql.Column(f'AVG({x})', alias=x) for x in xs])
  avg_x_and_y.add(sql.Column(f'AVG({y})', alias=y))
  cols = sql.Columns(split_by).add(avg_x_and_y)
  avgs = execute(
      str(sql.Sql(cols, table, groupby=split_by, with_data=with_data))
  )
  avg_table = sql.Sql(
      cols,
      table,
      groupby=split_by,
  )
  avg_table = with_data.merge(sql.Datasource(avg_table, 'AverageValueTable'))
  table_with_centered_x = sql.Columns(split_by)
  table_with_centered_x.add(sql.Column(y, '%s.{}' % table, y))
  for x in avg_x_and_y.aliases[:-1]:
    centered = sql.Column(x, '%s.{}' % table) - sql.Column(
        x, '%s.{}' % avg_table
    )
    centered.alias = x
    table_with_centered_x.add(centered)
  join = 'LEFT' if split_by else 'CROSS'
  table = with_data.merge(
      sql.Datasource(
          sql.Sql(
              table_with_centered_x,
              sql.Join(table, avg_table, using=split_by, join=join),
          ),
          'DataCentered',
      )
  )

  x_norms = [sql.Column(f'SQRT(SUM(POWER({x}, 2)))', alias=x) for x in xs]
  norms = sql.Sql(
      sql.Columns(split_by).add(x_norms),
      table,
      groupby=split_by,
      with_data=with_data,
  )
  norms = execute(str(norms))

  x_norm_squared = [sql.Column(f'SUM(POWER({x}, 2))', alias=x) for x in xs]
  norm_squared_table = sql.Sql(
      sql.Columns(split_by).add(x_norm_squared),
      table,
      groupby=split_by,
  )
  norm_squared_table = with_data.merge(
      sql.Datasource(norm_squared_table, 'NormSquaredValueTable')
  )

  table_with_x_norms = sql.Columns(split_by)
  table_with_x_norms.add(sql.Column(y, '%s.{}' % table, y))
  for x in sql.Columns(x_norms).aliases:
    norm = (
        sql.Column(x, '%s.{}' % table)
        / sql.Column(x, '%s.{}' % norm_squared_table) ** 0.5
    )
    norm.alias = x
    table_with_x_norms.add(norm)
  table = with_data.merge(
      sql.Datasource(
          sql.Sql(
              table_with_x_norms,
              sql.Join(table, norm_squared_table, using=split_by, join=join),
          ),
          'DataNormalized',
      )
  )

  return table, with_data, xs_cols, y, avgs, norms


def apply_algorithm_to_sufficient_stats_elements(
    sufficient_stats_elements, split_by, algorithm, *args, **kwargs
):
  """Applies algorithm to sufficient stats to get the coefficients of Models.

  Args:
    sufficient_stats_elements: Contains the elements to construct sufficient
      stats. It's one of the return of get_sufficient_stats_elements().
    split_by: The columns that we use to split the data.
    algorithm: A function that can take the sufficient_stats_elements of a slice
      of data and computes the coefficients of the Model.
    *args: Additional args passed to the algorithm.
    **kwargs: Additional kwargs passed to the algorithm.

  Returns:
    The coefficients of the Model.
  """
  fn = lambda row: algorithm(row, *args, **kwargs)
  if split_by:
    # Special characters in split_by got escaped during SQL execution.
    sufficient_stats_elements.columns = (
        split_by + list(sufficient_stats_elements.columns)[len(split_by) :]
    )
    return sufficient_stats_elements.groupby(split_by, observed=True).apply(fn)
  return fn(sufficient_stats_elements)


def compute_ridge_coefs(sufficient_stats, xs, m):
  """Computes coefficients of linear/ridge regression from sufficient_stats."""
  if isinstance(sufficient_stats, pd.DataFrame):
    sufficient_stats = sufficient_stats.iloc[0]
  fit_intercept = m.fit_intercept
  if fit_intercept and m.normalize:
    return compute_coef_for_normalize_ridge(sufficient_stats, xs, m)
  x_t_x, x_t_y = construct_matrix_from_elements(sufficient_stats, fit_intercept)
  if isinstance(m, Ridge):
    n_obs = sufficient_stats['n_obs']
    penalty = np.identity(len(x_t_y))
    if fit_intercept:
      penalty[0, 0] = 0
    # We use AVG() to compute x_t_x so the penalty needs to be scaled.
    x_t_x += m.alpha / n_obs * penalty
  cond = np.linalg.cond(x_t_x)
  if cond > 20:
    print(
        "WARNING: The condition number of X'X is %i, which might be too large."
        ' The model coefficients might be inaccurate.' % cond
    )
  coef = np.linalg.solve(x_t_x, x_t_y)
  xs = [x.alias_raw for x in xs]
  if fit_intercept:
    xs = ['intercept'] + xs
  return pd.DataFrame([coef], columns=xs)


def compute_coef_for_normalize_ridge(sufficient_stats, xs, m):
  """Computes the coefficient of OLS or Ridge with normalization."""
  n = len(xs)
  # Compute the elements of X_scaled^T * X_scaled. See
  # https://colab.research.google.com/drive/1wOWgdNzKGT_xl4A7Mrs_GbRKiVQACFfy#scrollTo=HrMCbB5SxS0A
  x_t_x_elements = []
  x_t_y = []
  for i in range(n):
    x_t_y.append(
        sufficient_stats[f'x{i}y']
        - sufficient_stats[f'x{i}'] * sufficient_stats['y']
    )
    for j in range(i, n):
      x_t_x_elements.append(
          sufficient_stats[f'x{i}x{j}']
          - sufficient_stats[f'x{i}'] * sufficient_stats[f'x{j}']
      )
  x_t_x = symmetrize_triangular(x_t_x_elements)
  if isinstance(m, Ridge):
    x_t_x += m.alpha * np.diag(x_t_x.diagonal())
  cond = np.linalg.cond(x_t_x)
  if cond > 20:
    print(
        "WARNING: The condition number of X'X is %i, which might be too large."
        ' The model coefficients might be inaccurate.' % cond
    )
  coef = np.linalg.solve(x_t_x, x_t_y)
  xs = [x.alias_raw for x in xs]
  intercept = sufficient_stats.y - coef.dot(
      [sufficient_stats[f'x{i}'] for i in range(n)]
  )
  coef = [intercept] + list(coef)
  xs = ['intercept'] + xs
  return pd.DataFrame([coef], columns=xs)


def symmetrize_triangular(tril_elements):
  """Converts a list of upper triangular matrix to a symmetric matrix.

  For example, [1, 2, 3] -> [[1, 2], [2, 3]].

  Args:
    tril_elements: A list that can form a triangular matrix.

  Returns:
    A symmetric matrix whose upper triangular part is formed from tril_elements.
  """
  n = int(np.floor((2 * len(tril_elements)) ** 0.5))
  if n * (n + 1) / 2 != len(tril_elements):
    raise ValueError('The elements cannot form a symmetric matrix!')
  sym = np.zeros([n, n])
  sym[np.triu_indices(n)] = tril_elements
  return sym + sym.T - np.diag(sym.diagonal())


def construct_matrix_from_elements(sufficient_stats_elements, fit_intercept):
  """Constructs matries X'X and X'y from the elements.

  Args:
    sufficient_stats_elements: A DataFrame holding all unique elements of
      sufficient stats. See the doc of get_sufficient_stats_elements() for its
      shape and content.
    fit_intercept: If the model includes an intercept.

  Returns:
    x_t_x: X'X / n_observations in a numpy array.
    x_t_y: X'y / n_observations in a numpy array.
  """
  if isinstance(sufficient_stats_elements, pd.DataFrame):
    if len(sufficient_stats_elements) > 1:
      raise ValueError('Only support 1D input!')
    sufficient_stats_elements = sufficient_stats_elements.iloc[0]
  elif not isinstance(sufficient_stats_elements, pd.Series):
    raise ValueError('The input must be a panda Series!')
  xny = (
      sufficient_stats_elements.index[-2]
      if sufficient_stats_elements.index[-1] == 'n_obs'
      else sufficient_stats_elements.index[-1]
  )
  n = int(xny[1:-1]) + 1
  x_t_x_cols = []
  x_t_y_cols = []
  if fit_intercept:
    x_t_x_cols = [f'x{i}' for i in range(n)]
    x_t_y_cols = ['y']
  for i in range(n):
    for j in range(i, n):
      x_t_x_cols.append(f'x{i}x{j}')
  x_t_y_cols += [f'x{i}y' for i in range(n)]
  x_t_x_elements = list(sufficient_stats_elements[x_t_x_cols])
  if fit_intercept:
    x_t_x_elements = [1] + x_t_x_elements
  x_t_y = sufficient_stats_elements[x_t_y_cols]
  x_t_x = symmetrize_triangular(x_t_x_elements)
  return x_t_x, np.array(x_t_y)


class Lasso(Model):
  """A class that can fit a Lasso regression."""

  def __init__(
      self,
      y: Optional[metrics.Metric] = None,
      x: Optional[
          Union[metrics.Metric, Sequence[metrics.Metric], metrics.MetricList]
      ] = None,
      group_by: Optional[Union[Text, List[Text]]] = None,
      alpha=1,
      fit_intercept: bool = True,
      normalize: bool = False,
      where: Optional[str] = None,
      name: Optional[str] = None,
      precompute=False,
      copy_X=True,
      max_iter=1000,
      tol=0.0001,
      warm_start=False,
      positive=False,
      random_state=None,
      selection='cyclic',
  ):
    """Initialize a sklearn.Lasso model."""
    model = linear_model.Lasso(
        alpha=alpha,
        fit_intercept=fit_intercept,
        copy_X=copy_X,
        max_iter=max_iter,
        tol=tol,
        warm_start=warm_start,
        positive=positive,
        random_state=random_state,
        selection=selection,
    )
    super(Lasso, self).__init__(
        y,
        x,
        group_by,
        model,
        'Lasso',
        where,
        name,
        fit_intercept,
        normalize,
        ['alpha', 'tol', 'max_iter', 'random_state'],
    )
    self.alpha = alpha
    self.tol = tol
    self.max_iter = max_iter
    self.random_state = random_state

  def compute_on_sql_magic_mode(self, table, split_by, execute):
    return ElasticNet(
        self.y,
        self.x,
        self.group_by,
        self.alpha,
        1,
        self.fit_intercept,
        self.normalize,
        self.where_,
        self.name,
        tol=self.tol,
        max_iter=self.max_iter,
    ).compute_on_sql_magic_mode(table, split_by, execute)


class ElasticNet(Model):
  """A class that can fit a ElasticNet regression."""

  def __init__(
      self,
      y: Optional[metrics.Metric] = None,
      x: Optional[
          Union[metrics.Metric, Sequence[metrics.Metric], metrics.MetricList]
      ] = None,
      group_by: Optional[Union[Text, List[Text]]] = None,
      alpha=1,
      l1_ratio=0.5,
      fit_intercept: bool = True,
      normalize: bool = False,
      where: Optional[str] = None,
      name: Optional[str] = None,
      precompute=False,
      copy_X=True,
      max_iter=1000,
      tol=0.0001,
      warm_start=False,
      positive=False,
      random_state=None,
      selection='cyclic',
  ):
    """Initialize a sklearn.ElasticNet model."""
    model = linear_model.ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        copy_X=copy_X,
        max_iter=max_iter,
        tol=tol,
        warm_start=warm_start,
        positive=positive,
        random_state=random_state,
        selection=selection,
    )
    super(ElasticNet, self).__init__(
        y,
        x,
        group_by,
        model,
        'ElasticNet',
        where,
        name,
        fit_intercept,
        normalize,
        ['alpha', 'tol', 'max_iter', 'l1_ratio', 'random_state'],
    )
    self.alpha = alpha
    self.tol = tol
    self.max_iter = max_iter
    self.l1_ratio = l1_ratio
    self.random_state = random_state

  def compute_on_sql_magic_mode(self, table, split_by, execute):
    if not self.l1_ratio or not 0 <= self.l1_ratio <= 1:
      raise ValueError(
          f'l1_ratio must be between 0 and 1; got (l1_ratio={self.l1_ratio})'
      )
    l1 = self.l1_ratio * self.alpha
    l2 = (1 - self.l1_ratio) * self.alpha
    xs, sufficient_stats_elements, avgs, norms = get_sufficient_stats_elements(
        self, table, split_by, execute
    )
    np.random.seed(
        self.random_state if isinstance(self.random_state, int) else 42
    )
    coef = apply_algorithm_to_sufficient_stats_elements(
        sufficient_stats_elements,
        split_by,
        compute_coef_for_elastic_net,
        xs,
        l1,
        l2,
        self.fit_intercept,
        self.tol,
        self.max_iter,
    )
    if self.fit_intercept and self.normalize:
      coef = compute_normalized_coef(coef, norms, avgs, split_by)
    columns = list(coef.columns)
    columns[-len(xs) :] = [x.alias_raw for x in xs]
    coef.columns = columns
    return coef


def compute_coef_for_elastic_net(
    sufficient_stats_elements, xs, l1, l2, fit_intercept, tol, max_iter
):
  """Computes the coefficients for ElasticNet. Lasso is just a special case."""
  if fit_intercept:
    sufficient_stats_elements, avg_xs, avg_y = center_x(
        sufficient_stats_elements, len(xs)
    )
  x_t_x, x_t_y = construct_matrix_from_elements(
      sufficient_stats_elements, fit_intercept
  )
  init_guess = np.random.random(*x_t_y.shape)
  coef = fista_for_elastic_net(
      init_guess, l1, l2, x_t_x, x_t_y, tol, max_iter, fit_intercept
  )
  columns = list(xs.aliases)
  if fit_intercept:
    # We centered x and y above so the intercept from optimization is not right.
    coef[0] = (avg_y - avg_xs @ coef[1:]).values[0]
    columns = ['intercept'] + columns
  return pd.DataFrame([list(coef)], columns=columns)


def center_x(sufficient_stats_elements, n):
  """Compute the sufficient_stats_elements of centered x for better convergence.

  sufficient_stats_elements has four types of elements. The ones under 'x{i}'
  are the average values of the i-th feature. The 'y' column is the average
  of the dependent variable. The 'x{i}x{j}' are the normalized elements of
  matrix X'X, namely, 'x{i}x{j}' has value of AVG(xi * xj). Similarly, 'x{i}y'
  contains the normalized elements of X'y.
  We want to center X and y for better numerical stability and convergence. As
  a result, we need to update 'x{i}x{j}'. The new value would be
  AVG((xi - xi_bar) * (xj - xj_bar)) = AVG(xi * xj) - xi_bar * xj_bar. 'x{i}y'
  can be updated similarly.

  Args:
    sufficient_stats_elements: The sufficient_stats_elements computed from
      original x.
    n: The number of x.

  Returns:
    The sufficient_stats_elements computed from centered x and y.
    Average of original x.
    Average of original y.
  """
  sufficient_stats_elements = sufficient_stats_elements.copy()
  avg_xs = sufficient_stats_elements[[f'x{i}' for i in range(n)]]
  avg_y = sufficient_stats_elements['y']
  for i in range(n):
    sufficient_stats_elements[f'x{i}y'] -= avg_xs[f'x{i}'] * avg_y
    for j in range(i, n):
      sufficient_stats_elements[f'x{i}x{j}'] -= (
          avg_xs[f'x{i}'] * avg_xs[f'x{j}']
      )
  sufficient_stats_elements[[f'x{i}' for i in range(n)]] = 0
  return sufficient_stats_elements, avg_xs, avg_y


def fista_for_elastic_net(
    coef, l1, l2, x_t_x, x_t_y, tol, max_iter, fit_intercept
):
  """Applies FISTA algorithm to elastic net.

  Lasso is just a special case.

  The algorithm is also called accelerated proximal gradient descent. The
  parameters are the same as those in proximal gradient descent. A derivation
  can be found in
  https://web.archive.org/save/https://yuxinchen2020.github.io/ele520_math_data/lectures/lasso_algorithm_extension.pdf.
  There are variants for the acceleration. Here we implemented the one in
  https://web.archive.org/web/20220616072055/http://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/prox-grad_2.pdf.
  The function is used for Lasso/ElasticNet, the differentiable g(x) in the loss
  function used by sklearn is (1 / (2 * n_samples)) * ||y - Xw||^2_2. Its
  Lipschitz constant, L, is the largest eigenvalue of X'X / n_samples, so the
  max step size is 1/L. For proof, see Example 2.2 in
  Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding
  algorithm for linear inverse problems.
  SIAM journal on imaging sciences, 2(1), 183-202.

  Args:
    coef: The initial guess of the coefficients.
    l1: L1 penalty strength. In terms of the args of ElasticNet in sklearn, it
      equals alpha * l1_ratio.
    l2: L2 penalty strength. In terms of the args of ElasticNet in sklearn, it
      equals alpha * (1 - l1_ratio).
    x_t_x: X'X / n_observations.
    x_t_y: X'y / n_observations.
    tol: The tolerance for the optimization.
    max_iter: The maximum number of iterations.
    fit_intercept: If the coefficients include an intercept.

  Returns:
    The converged coef, namely, the coefficients of the model.
  """
  delta = np.zeros_like(coef)
  # If we just use 1, depending on how a float is stored, maybe the step_size
  # will be slightly larger than the max allowed? I don't know if it will ever
  # happen but to be safe I use 1 - 1e-6.
  step_size = (1 - 1e-6) / np.linalg.eigvals(x_t_x).max()
  k = l2 * step_size + 1
  threshold = l1 * step_size / k

  for i in range(int(max_iter)):
    v = coef + (i - 1) / (i + 2) * delta
    coef_old = coef
    coef = v - step_size * (x_t_x @ v - x_t_y)
    if fit_intercept:
      coef[1:] = soft_thresholding(coef[1:] / k, threshold)
    else:
      coef = soft_thresholding(coef / k, threshold)
    delta = coef - coef_old
    if abs(delta).sum() < tol:
      return coef
  print("WARNING: Lasso/ElasticNet didn't converge! Try increasing `max_iter`.")
  return coef


def soft_thresholding(x, thresh):
  return np.maximum(x - thresh, 0) + np.minimum(x + thresh, 0)


def compute_normalized_coef(coef, norms, avgs, split_by):
  """Scale the coef by the norms of x to get the coef for normalized models.

  Compute the coeffients of model(normalized=True).fit(x, y) from the coeffients
  of model.fit(x_normalized, y). The former is the latter divided by the norms
  of centered x, except for the intercept. Once the coefficients are calculated,
  the intercept can be computed from coefficients and the average of original x
  and y.

  Args:
    coef: The coeffients of model.fit(x_normalized, y), including intercept as
      the first column.
    norms: The norms of centered x.
    avgs: The average of normalized x and y.
    split_by: The columns that we use to split the data.

  Returns:
    The coeffients of model(normalized=True).fit(x, y).
  """
  coef = utils.remove_empty_level(coef)
  cols = coef.columns
  coef.drop(columns='intercept', inplace=True)
  if split_by:
    norms = norms.set_index(coef.index.names).reindex(coef.index)
    avgs = avgs.set_index(coef.index.names).reindex(coef.index)
  coef /= norms
  avg_x, avg_y = avgs.iloc[:, :-1], avgs.iloc[:, -1]
  coef['intercept'] = avg_y - (avg_x.values * coef.values).sum(1)
  return coef[cols]


class LogisticRegression(Model):
  """A class that can fit a logistic regression."""

  def __init__(
      self,
      y: metrics.Metric,
      x: Union[metrics.Metric, Sequence[metrics.Metric], metrics.MetricList],
      group_by: Optional[Union[Text, List[Text]]] = None,
      fit_intercept: bool = True,
      where: Optional[str] = None,
      name: Optional[str] = None,
      penalty='l2',
      dual=False,
      tol=0.0001,
      C=1.0,
      intercept_scaling=1,
      class_weight=None,
      random_state=None,
      solver='lbfgs',
      max_iter=100,
      verbose=0,
      warm_start=False,
      n_jobs=None,
      l1_ratio=None,
  ):
    """Initialize a sklearn.LogisticRegression model."""
    if penalty not in (None, 'l1', 'l2', 'elasticnet'):
      raise ValueError(
          "Penalty must be one of (None, 'l1', 'l2', 'elasticnet') but is"
          f' {penalty}!'
      )
    if penalty == 'elasticnet' and (not l1_ratio or not 0 <= l1_ratio <= 1):
      raise ValueError(
          f'l1_ratio must be between 0 and 1; got (l1_ratio={l1_ratio})'
      )
    if l1_ratio is not None and penalty != 'elasticnet':
      raise ValueError(
          "l1_ratio parameter is only used when penalty is 'elasticnet'. Got"
          f' (penalty={penalty})'
      )
    model = linear_model.LogisticRegression(
        fit_intercept=fit_intercept,
        penalty=penalty,
        dual=dual,
        tol=tol,
        C=C,
        intercept_scaling=intercept_scaling,
        class_weight=class_weight,
        random_state=random_state,
        solver=solver,
        max_iter=max_iter,
        verbose=verbose,
        warm_start=warm_start,
        n_jobs=n_jobs,
        l1_ratio=l1_ratio,
    )
    super(LogisticRegression, self).__init__(
        y,
        x,
        group_by,
        model,
        'LogisticRegression',
        where,
        name,
        fit_intercept,
        False,
        [
            'penalty',
            'tol',
            'c',
            'intercept_scaling',
            'max_iter',
            'l1_ratio',
            'random_state',
        ],
    )
    self.penalty = penalty
    self.tol = tol
    self.c = C
    self.intercept_scaling = intercept_scaling or 1
    self.max_iter = max_iter
    self.l1_ratio = l1_ratio
    self.random_state = random_state

  def compute(self, df):
    self.model.fit(df.iloc[:, 1:], df.iloc[:, 0])
    coef = self.model.coef_
    names = list(df.columns[1:])
    if coef.shape[0] == 1:
      coef = coef[0]
      if self.fit_intercept:
        intercept = self.model.intercept_
        intercept = intercept[0]
        coef = [intercept] + list(coef)
        names = ['intercept'] + names
      return pd.DataFrame([coef], columns=names)
    else:
      # Multi class
      if self.fit_intercept:
        coef = np.hstack((self.model.intercept_.reshape(-1, 1), coef))
        names = ['intercept'] + names
      res = pd.DataFrame(
          coef.reshape(1, -1),
          columns=(
              f'{n} for class {c}'
              for c, n in itertools.product(self.model.classes_, names)
          ),
      )
      return res

  def compute_on_sql_magic_mode(self, table, split_by, execute):
    """Gets the coefficients by minimizing the cost function.

    We use iteratively reweighted least squares algorithm to solve the model.
    The algorithm is described in
    https://colab.research.google.com/drive/1Srfs4weM4LO9vt1HbOkGrD4kVbG8cso8.

    Args:
      table: The table we want to query from.
      split_by: The columns that we use to split the data.
      execute: A function that can executes a SQL query and returns a DataFrame.

    Returns:
      A pd.DataFrame holding model coefficients.
    """
    if self.model.class_weight:
      raise ValueError("Magic mode doesn't support class_weight!")
    if self.intercept_scaling != 1:
      raise ValueError('intercept_scaling is not supported in magic mode!')
    if not self.y:
      raise ValueError('y is not set!')

    y = self.y.to_sql(table, self.group_by + split_by)
    n_y = metrics.Count(y.columns[-1].alias, distinct=True)
    n_y = n_y.compute_on_sql(
        y, y.groupby.aliases[len(self.group_by):], execute
    )
    if (n_y.values != 2).any():
      raise ValueError(
          f'Magic mode only support two classes but got {n_y} distinct y'
          ' values!'
      )

    if self.penalty in ('l1', 'elasticnet'):
      _, sufficient_stats, _, _ = get_sufficient_stats_elements(
          self, table, split_by, execute, include_n_obs=True
      )

    table, with_data, xs_cols, y, _, _ = get_data(
        self, table, split_by, execute
    )
    xs = xs_cols.aliases
    if self.fit_intercept:
      xs.append('1')
    conds = []
    if split_by:
      slices = execute(
          str(sql.Sql(sql.Columns(split_by, True), table, with_data=with_data))
      )
      slices.sort_values(list(slices.columns), inplace=True)
      conds = slices.values
    self._hess = None

    def grads(coef, converged, ignore_hess=True):
      return compute_grads_and_hess(coef, converged, ignore_hess)

    def hess(*unused_args):
      return self._hess

    def compute_grads_and_hess(coef, converged, ignore_hess=True):
      """Computes the gradients and Hessian matrices for coef.

      The grads we computes here is a n*k list of gradients, where n is the
      number of slices and k is the number of features. It has the same shape
      with coef, and represents the gradients of the coefficients.
      Similarly, the Hessian matrices we return is a n*k*k array. Each k*k
      element is a Hessian matrix. It's saved to self._hess as a side effect.

      Args:
        coef: A n*k array of coefficients being optimized. n is the number of
          slices and k is the number of features.
        converged: A list of the length of the number of slices. Its values
          indicate whether the coefficients of the slice have converged. If
          converged, we skip the computation for that slice.
        ignore_hess: If True, we only compute gradients.

      Returns:
        A n*k numpy array of gradients / n_observations. We also save
        hessian / n_observations to self._hess as a side effect, if not
        ignore_hess.
      """
      k = len(coef[0])
      if not split_by:
        grads, hessian = get_grads_and_hess_query(
            coef[0], ignore_hess=ignore_hess
        )
      else:
        grads = []
        hessian = []
        split_cols = sql.Columns(split_by).aliases
        for cond, slice_coef, done in zip(conds, coef, converged):
          if not done:
            condition = [
                f'{c} = "{v}"' if isinstance(v, str) else f'{c} = {v}'
                for c, v in zip(split_cols, cond)
            ]
            condition = ' AND '.join(condition)
            j, h = get_grads_and_hess_query(slice_coef, condition, ignore_hess)
            grads += j
            hessian += h
      for i, c in enumerate(grads):
        c.set_alias(f'grads_{i}')
      for i, c in enumerate(hessian):
        c.set_alias(f'hess_{i}')
      grads_and_hess = sql.Sql(sql.Columns(grads + hessian), table)
      grads_and_hess.with_data = with_data
      grads_and_hess = execute(str(grads_and_hess)).iloc[0]
      grads = []
      for done in converged:
        if done:
          grads.append([0] * k)
        else:
          grads.append(grads_and_hess.values[:k])
          grads_and_hess = grads_and_hess[k:]
      if ignore_hess:
        return np.array(grads)
      hess_elements = list(grads_and_hess.values.reshape(-1, k * (k + 1) // 2))
      hess_arr = []
      zero_hess = np.zeros((k, k))
      for done in converged:
        if done:
          hess_arr.append(zero_hess)
        else:
          hess_arr.append(symmetrize_triangular(hess_elements.pop(0)))
      self._hess = np.array(hess_arr)
      return np.array(grads)

    def get_grads_and_hess_query(coef, condition=None, ignore_hess=True):
      """Get the SQL columns to compute the gradients and Hessian matrixes.

      The formula of gradients and Hessian matrixes can be found in
      https://colab.research.google.com/drive/1Srfs4weM4LO9vt1HbOkGrD4kVbG8cso8.
      As the Hessian matrix is symmetric, we only construct the columns for
      unique values.

      Args:
        coef: A n*k list of coefficients being optimized, where n is the number
          of slices and k is the number of features.
        condition: A condition that can be applied in the WHERE clause. The
          gradients and Hessian matrixes will be computed on the rows that
          satisfies the condition.
        ignore_hess: If True, we only return gradients.

      Returns:
        grads: A list of SQL columns which has the same shape with coef. It
          represents the gradients / n_observations.
        hess: A list of SQL columns that can be used to construct the Hessian
          matrix / n_observations. Its elements are the upper triangular part of
          the Hessian, from left to right, top to down.
      """
      # A numerically stable implemntation, adapted from
      # http://fa.bianp.net/blog/2019/evaluate_logistic.
      z = ' + '.join(f'{coef[i]} * {xs[i]}' for i in range(len(xs)))
      grads = [
          sql.Column(f'{x} * {sig_minus_b(z, y)}', 'AVG({})', filters=condition)
          for x in xs
      ]
      sig_z = """IF({z} < 0,
          EXP({z}) / (1 + EXP({z})),
          1 / (1 + EXP(-({z}))))""".format(z=z)
      w = f'-{sig_z} * {sig_minus_b(z, 1)}'
      hess = []
      if not ignore_hess:
        for i, x1 in enumerate(xs):
          for x2 in xs[i:]:
            hess.append(
                sql.Column(f'{x1} * {x2} * {w}', 'AVG({})', filters=condition)
            )
      hess = np.array(hess)
      # See here for the behavior of differnt penalties.
      # https://colab.research.google.com/drive/1Srfs4weM4LO9vt1HbOkGrD4kVbG8cso8
      # For 'l1' and 'elasticnet', we use FISTA, which uses unregularized
      # gradient.
      n = f'COUNTIF({condition})' if condition else 'COUNT(*)'
      if self.penalty == 'l2':
        for i in range(self.k):
          grads[i] += sql.Column(f'{coef[i]} / {n}') / self.c
        if not ignore_hess:
          hess_diag_idx = np.arange(len(xs), len(xs) - self.k + 1, -1).cumsum()
          hess_diag_idx = np.concatenate([[0], hess_diag_idx])
          hess[hess_diag_idx] += sql.Column(f'1 / ({n} * {self.c})')
      return grads, list(hess)

    np.random.seed(
        self.random_state if isinstance(self.random_state, int) else 42
    )
    init_guess = np.random.random((len(conds) or 1, len(xs)))
    if self.penalty in ('l1', 'elasticnet'):
      l1_ratio = self.l1_ratio if self.l1_ratio is not None else 1
      l1 = l1_ratio / self.c
      l2 = (1 - l1_ratio) / self.c
      res = fista_for_logistic_regression(
          init_guess,
          split_by,
          grads,
          self.tol,
          self.max_iter,
          conds,
          l1,
          l2,
          sufficient_stats,
          self.fit_intercept,
      )
    else:
      res = newtons_method(
          init_guess, grads, hess, self.tol, self.max_iter, conds
      )

    xs = [x.alias_raw for x in xs_cols]
    if split_by:
      df = pd.DataFrame(conds, columns=split_by)
      if len(split_by) == 1:
        idx = pd.Index(df[split_by[0]])
      else:
        idx = pd.MultiIndex.from_frame(df)
      res = pd.DataFrame(res, index=idx)
      if self.fit_intercept:
        res.columns = xs + ['intercept']
        # Make intercept the 1st column.
        xs = ['intercept'] + xs
        res = res[xs]
      else:
        res.columns = xs
      return res.sort_index()
    res = res[0]
    if self.fit_intercept:
      res = np.concatenate([[res[-1]], res[:-1]])
      xs = ['intercept'] + xs
    return pd.DataFrame([res], columns=xs)


def fista_for_logistic_regression(
    coef,
    split_by,
    grads,
    tol,
    max_iter,
    conds,
    l1,
    l2,
    sufficient_stats,
    fit_intercept,
):
  """Applies FISTA algorithm to logistic regression.

  The algorithm is also called accelerated proximal gradient descent. The
  parameters are the same as those in the proximal gradient descent for elastic
  net. A derivation can be found in
  https://web.archive.org/save/https://yuxinchen2020.github.io/ele520_math_data/lectures/lasso_algorithm_extension.pdf.
  There are variants for the acceleration. Here we implemented the one in
  https://web.archive.org/web/20220616072055/http://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/prox-grad_2.pdf.
  The conventional loss function is the sum of logit loss on all data points.
  For numerical stability we use the average of logit loss instead. As a result,
  step size, l1 and l2 are scaled accordingly.

  Args:
    coef: The initial guess of the coefficients.
    split_by: The columns that we use to split the data.
    grads: A function that returns the gradients for certain coef.
    tol: The tolerance for the optimization.
    max_iter: The maximum number of iterations.
    conds: None if no split_by. Otherwise it's the sorted unique values in the
      split_by columns.
    l1: L1 penalty strength. In terms of the args of LogisticRegression in
      sklearn, it equals l1_ratio / C.
    l2: L2 penalty strength. In terms of the args of LogisticRegression in
      sklearn, it equals (1 - l1_ratio) / C.
    sufficient_stats: A DataFrame holding all unique elements of sufficient
      stats.
    fit_intercept: If the coefficients include an intercept.

  Returns:
    The converged coef, namely, the coefficients of the model.
  """

  def get_step_size(stats):
    # For the value of step size, see p29 in
    # https://web.archive.org/web/20211223053411/https://www.cs.ubc.ca/~schmidtm/Courses/540-W18/L4.pdf.
    x_t_x, _ = construct_matrix_from_elements(stats, fit_intercept)
    # If we just use 4, depending on how a float is stored, maybe the step_size
    # will be slightly larger than the max allowed? I don't know if it will ever
    # happen but to be safe I use 3.999999.
    return 3.999999 / np.linalg.eigvals(x_t_x).max()

  if not split_by:
    step_size = np.array([get_step_size(sufficient_stats)])
  else:
    sufficient_stats.columns = (
        split_by + list(sufficient_stats.columns)[len(split_by) :]
    )
    step_size = sufficient_stats.groupby(split_by, observed=True).apply(
        get_step_size
    )
    # cond is sorted. We need to make sure step_size and sufficient_stats are in
    # the same order.
    step_size.sort_index(inplace=True)
    sufficient_stats.sort_values(split_by, inplace=True)
    if len(step_size) != len(conds):
      raise ValueError('Incomatiple step size!')
    step_size = step_size.values.reshape((-1, 1))

  n_obs = sufficient_stats.n_obs.values.reshape((-1, 1))
  # Adjust because the sufficient_stats are the real sufficient stats / n_obs.
  l1 /= n_obs
  l2 /= n_obs
  n_slice = len(coef)
  converged = np.array([False] * n_slice)
  k = l2 * step_size + 1
  threshold = l1 * step_size / k
  v = coef.copy()
  delta = np.zeros_like(coef)

  for i in range(int(max_iter)):
    pending = ~converged
    v[pending] = coef[pending] + (i - 1) / (i + 2) * delta[pending]
    coef_old = coef.copy()
    gradients = grads(v, converged, ignore_hess=True)
    coef[pending] = v[pending] - step_size[pending] * gradients[pending]
    if fit_intercept:
      coef[pending, :-1] = soft_thresholding(
          coef[pending, :-1] / k[pending], threshold[pending]
      )
    else:
      coef[pending] = soft_thresholding(
          coef[pending] / k[pending], threshold[pending]
      )
    delta[pending] = coef[pending] - coef_old[pending]
    converged[pending] = abs(delta[pending]).max(1) < tol
    if all(converged):
      return coef
  if n_slice == 1:
    print(
        "WARNING: Optimization of LogisticRegression didn't converge! Try"
        ' increasing `max_iter`.'
    )
  else:
    print(
        "WARNING: Optimization of LogisticRegression didn't converge for"
        ' slice:%s. Try increasing `max_iter`.'
        % np.array(conds)[~converged]
    )
  return coef


def sig_minus_b(z, b):
  """Computes sigmoid(z) - b in a numerically stable way in SQL."""
  # Adapted from http://fa.bianp.net/blog/2019/evaluate_logistic
  exp_z = f'EXP({z})'
  exp_nz = f'EXP(-({z}))'
  return (
      'IF({z} < 0, ((1 - {b}) * {exp_z} - {b}) / (1 + {exp_z}), ((1 - {b}) '
      '- {b} * {exp_nz}) / (1 + {exp_nz}))'
  ).format(z=z, b=b, exp_z=exp_z, exp_nz=exp_nz)


def newtons_method(coef, grads, hess, tol, max_iter, conds):
  """Uses Newton's method to optimize coef on n slices at the same time."""
  n_slice = len(coef)
  converged = np.array([False] * n_slice)
  for _ in range(int(max_iter)):
    j = grads(coef, converged, ignore_hess=False)
    h = hess(coef, converged)
    for i in range(n_slice):
      if not converged[i]:
        delta = np.linalg.solve(h[i], j[i])
        if abs(delta).max() < tol:
          converged[i] = True
        coef[i] -= delta
    if all(converged):
      return coef
  if n_slice == 1:
    print("WARNING: Optimization of LogisticRegression didn't converge!")
  else:
    print(
        "WARNING: Optimization of LogisticRegression didn't converge for"
        ' slice: ',
        np.array(conds)[~converged],
    )
  return coef
