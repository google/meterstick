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
"""Models that can be fitted in Meterstick."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

  def __init__(self,
               y: metrics.Metric,
               x: Union[metrics.Metric, Sequence[metrics.Metric],
                        metrics.MetricList],
               group_by: Optional[Union[Text, List[Text]]] = None,
               model=None,
               model_name=None,
               where=None,
               name=None,
               fit_intercept=True,
               normalize=False):
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
    """
    if not isinstance(y, metrics.Metric):
      raise ValueError('y must be a Metric!')
    if count_features(y) != 1:
      raise ValueError('y must be a 1D array but is %iD!' % count_features(y))
    self.group_by = [group_by] if isinstance(group_by, str) else group_by or []
    if isinstance(x, Sequence):
      x = metrics.MetricList(x)
    self.x = x
    self.y = y
    self.model = model
    self.k = count_features(x)
    if not name:
      x_names = [m.name for m in x] if isinstance(
          x, metrics.MetricList) else [x.name]
      name = '%s(%s ~ %s)' % (model_name, y.name, ' + '.join(x_names))
    name_tmpl = name + ' Coefficient: {}'
    super(Model, self).__init__(
        metrics.MetricList((y, x)),
        name_tmpl,
        group_by, [],
        name=name,
        where=where)
    self.computable_in_pure_sql = False
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
    if mode not in (None, 'sql', 'mixed', 'magic'):
      raise ValueError('Mode %s is not supported!' % mode)
    if mode == 'sql':
      raise ValueError('%s is not computable in pure SQL!' % self.name)
    if mode == 'magic' and not self.all_computable_in_pure_sql(False):
      raise ValueError(
          'The "magic" mode requires all descendants to be computable in SQL!' %
          self.name)

    if self.where:
      table = sql.Sql(sql.Column('*', auto_alias=False), table, self.where)
    if mode == 'mixed' or not mode:
      try:
        return self.compute_on_sql_mixed_mode(table, split_by, execute, mode)
      except utils.MaybeBadSqlModeError as e:
        raise
      except Exception as e:  # pylint: disable=broad-except
        raise utils.MaybeBadSqlModeError('magic') from e
    if self.all_computable_in_pure_sql(False):
      try:
        res = self.compute_on_sql_magic_mode(table, split_by, execute)
        return utils.apply_name_tmpl(self.name_tmpl, res)
      except Exception as e:  # pylint: disable=broad-except
        raise utils.MaybeBadSqlModeError('mixed') from e

  def compute_on_sql_magic_mode(self, table, split_by, execute):
    raise NotImplementedError


class LinearRegression(Model):
  """A class that can fit a linear regression."""

  def __init__(self,
               y: metrics.Metric,
               x: Union[metrics.Metric, Sequence[metrics.Metric],
                        metrics.MetricList],
               group_by: Optional[Union[Text, List[Text]]] = None,
               fit_intercept: bool = True,
               normalize: bool = False,
               where: Optional[str] = None,
               name: Optional[str] = None):
    """Initialize a sklearn.LinearRegression model."""
    model = linear_model.LinearRegression(fit_intercept=fit_intercept)
    super(LinearRegression, self).__init__(y, x, group_by, model, 'OLS', where,
                                           name, fit_intercept, normalize)

  def compute_on_sql_magic_mode(self, table, split_by, execute=None):
    return compute_linear_or_ridge_coefficiens(self, table, split_by, execute)


def compute_linear_or_ridge_coefficiens(m, table, split_by, execute=None):
  """Computes X'X, X'y then the coefficients for linear or ridge regressions."""
  data = m.children[0].to_sql(table, split_by + m.group_by)
  with_data = data.with_data
  data.with_data = None
  table, _ = with_data.merge(sql.Datasource(data, 'DataToFit'))
  y = data.columns[-m.k - 1].alias
  xs = data.columns.aliases[-m.k:]
  x_t_x = []
  x_t_y = []
  if m.fit_intercept:
    x_t_x = [
        sql.Column('AVG(%s)' % x, alias='x%s' % i) for i, x in enumerate(xs)
    ]
    x_t_y = [sql.Column('AVG(%s)' % y, alias='y')]
  for i, x1 in enumerate(xs):
    for j, x2 in enumerate(xs[i:]):
      x_t_x.append(
          sql.Column('AVG(%s * %s)' % (x1, x2), alias='x%sx%s' % (i, i + j)))
  x_t_y += [
      sql.Column('AVG(%s * %s)' % (x, y), alias='x%sy' % i)
      for i, x in enumerate(xs)
  ]
  cols = sql.Columns(x_t_x + x_t_y)
  if isinstance(m, Ridge):
    cols.add(sql.Column('COUNT(*)', alias='n_obs'))
  sufficient_stats = sql.Sql(
      cols, table, groupby=sql.Columns(split_by).aliases, with_data=with_data)
  sufficient_stats = execute(str(sufficient_stats))
  fn = lambda row: compute_coef_from_sufficient_stats(row, xs, m)
  if split_by:
    sufficient_stats.columns = split_by + list(
        sufficient_stats.columns)[len(split_by):]
    return sufficient_stats.groupby(split_by).apply(fn)
  return fn(sufficient_stats)


def compute_coef_from_sufficient_stats(sufficient_stats, xs, m):
  """Computes coefficiens of linear or ridge regression from X'X and X'y."""
  if isinstance(sufficient_stats, pd.DataFrame):
    sufficient_stats = sufficient_stats.iloc[0]
  fit_intercept = m.fit_intercept
  if isinstance(m, Ridge) and fit_intercept and m.normalize:
    return compute_coef_for_normalize_ridge(sufficient_stats, xs, m)
  n = len(xs)
  x_t_x_cols = []
  x_t_y_cols = []
  if fit_intercept:
    x_t_x_cols = ['x%s' % i for i in range(n)]
    x_t_y_cols = ['y']
  for i in range(n):
    for j in range(i, n):
      x_t_x_cols += ['x%sx%s' % (i, j)]
  x_t_y_cols += ['x%sy' % i for i in range(n)]
  x_t_x_elements = list(sufficient_stats[x_t_x_cols])
  if fit_intercept:
    x_t_x_elements = [1] + x_t_x_elements
  x_t_y = sufficient_stats[x_t_y_cols]
  if fit_intercept:
    n += 1
  x_t_x = symmetrize_triangular(x_t_x_elements)
  if isinstance(m, Ridge):
    n_obs = sufficient_stats['n_obs']
    penalty = np.identity(n)
    if fit_intercept:
      penalty[0, 0] = 0
    # We use AVG() to compute x_t_x so the penalty needs to be scaled.
    x_t_x += m.alpha / n_obs * penalty
  cond = np.linalg.cond(x_t_x)
  if cond > 20:
    print(
        "WARNING: The condition number of X'X is %i, which might be too large."
        ' The model coefficients might be inaccurate.' % cond)
  coef = np.linalg.solve(x_t_x, x_t_y)
  xs = [n.replace('macro_', '$').strip('`') for n in xs]
  if fit_intercept:
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
  n = int(np.ceil(len(tril_elements)**0.5))
  if n * (n + 1) / 2 != len(tril_elements):
    raise ValueError('The elements cannot form a symmetric matrix!')
  sym = np.zeros([n, n])
  sym[np.triu_indices(n)] = tril_elements
  return sym + sym.T - np.diag(sym.diagonal())


def compute_coef_for_normalize_ridge(sufficient_stats, xs, m):
  """Computes the coefficient of Ridge with normalization and intercept."""
  n = len(xs)
  # Compute the elements of X_scaled^T * X_scaled. See
  # https://colab.research.google.com/drive/1wOWgdNzKGT_xl4A7Mrs_GbRKiVQACFfy#scrollTo=HrMCbB5SxS0A
  x_t_x_elements = []
  x_t_y = []
  for i in range(n):
    x_t_y.append(sufficient_stats['x%sy' % i] -
                 sufficient_stats['x%s' % i] * sufficient_stats['y'])
    for j in range(i, n):
      x_t_x_elements.append(sufficient_stats['x%sx%s' % (i, j)] -
                            sufficient_stats['x%s' % i] *
                            sufficient_stats['x%s' % j])
  x_t_x = symmetrize_triangular(x_t_x_elements)
  x_t_x += m.alpha * np.diag(x_t_x.diagonal())
  cond = np.linalg.cond(x_t_x)
  if cond > 20:
    print(
        "WARNING: The condition number of X'X is %i, which might be too large."
        ' The model coefficients might be inaccurate.' % cond)
  coef = np.linalg.solve(x_t_x, x_t_y)
  xs = [n.replace('macro_', '$').strip('`') for n in xs]
  intercept = sufficient_stats.y - coef.dot(
      [sufficient_stats['x%s' % i] for i in range(n)])
  coef = [intercept] + list(coef)
  xs = ['intercept'] + xs
  return pd.DataFrame([coef], columns=xs)


class Ridge(Model):
  """A class that can fit a ridge regression."""

  def __init__(self,
               y: metrics.Metric,
               x: Union[metrics.Metric, Sequence[metrics.Metric],
                        metrics.MetricList],
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
               random_state=None):
    """Initialize a sklearn.Ridge model."""
    model = linear_model.Ridge(
        alpha=alpha,
        fit_intercept=fit_intercept,
        copy_X=copy_X,
        max_iter=max_iter,
        tol=tol,
        solver=solver,
        random_state=random_state)
    super(Ridge, self).__init__(y, x, group_by, model, 'Ridge', where, name,
                                fit_intercept, normalize)
    self.alpha = alpha

  def compute_on_sql_magic_mode(self, table, split_by, execute=None):
    return compute_linear_or_ridge_coefficiens(self, table, split_by, execute)


class Lasso(Model):
  """A class that can fit a Lasso regression."""

  def __init__(self,
               y: metrics.Metric,
               x: Union[metrics.Metric, Sequence[metrics.Metric],
                        metrics.MetricList],
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
               selection='cyclic'):
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
        selection=selection)
    super(Lasso, self).__init__(y, x, group_by, model, 'Lasso', where, name,
                                fit_intercept, normalize)


class ElasticNet(Model):
  """A class that can fit a ElasticNet regression."""

  def __init__(self,
               y: metrics.Metric,
               x: Union[metrics.Metric, Sequence[metrics.Metric],
                        metrics.MetricList],
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
               selection='cyclic'):
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
        selection=selection)
    super(ElasticNet, self).__init__(y, x, group_by, model, 'ElasticNet', where,
                                     name, fit_intercept, normalize)
    self.alpha = alpha
    self.tol = tol
    self.max_iter = max_iter
    self.l1_ratio = l1_ratio


class LogisticRegression(Model):
  """A class that can fit a logistic regression."""

  def __init__(self,
               y: metrics.Metric,
               x: Union[metrics.Metric, Sequence[metrics.Metric],
                        metrics.MetricList],
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
               multi_class='auto',
               verbose=0,
               warm_start=False,
               n_jobs=None,
               l1_ratio=None):
    """Initialize a sklearn.LogisticRegression model."""
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
        multi_class=multi_class,
        verbose=verbose,
        warm_start=warm_start,
        n_jobs=n_jobs,
        l1_ratio=l1_ratio)
    super(LogisticRegression,
          self).__init__(y, x, group_by, model, 'LogisticRegression', where,
                         name, fit_intercept)
    self.penalty = penalty
    self.tol = tol
    self.c = C
    self.intercept_scaling = intercept_scaling or 1
    self.max_iter = max_iter
    self.l1_ratio = l1_ratio

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
          columns=('%s for class %s' % (n, c)
                   for c, n in itertools.product(self.model.classes_, names)))
      return res

  def compute_on_sql_magic_mode(self, table, split_by, execute=None):
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
    if self.model.multi_class == 'multinomial':
      raise ValueError("Magic mode doesn't support multi_class!")
    if self.penalty == 'elasticnet' and (not self.l1_ratio or
                                         not 0 <= self.l1_ratio <= 1):
      raise ValueError('l1_ratio must be between 0 and 1; got (l1_ratio="%s")' %
                       self.l1_ratio)
    if self.intercept_scaling != 1:
      raise ValueError('intercept_scaling is not supported in magic mode!')
    if self.penalty in ('l1', 'elasticnet'):
      print("WARNING: Our solution for L1 and elasticnet penalty doesn't quite "
            'achieve sparsity. Please interprete the results with care.')

    y = self.y.compute_on_sql(table, self.group_by, execute)
    n_y = y.iloc[:, 0].nunique()
    if n_y != 2:
      raise ValueError(
          'Magic mode only support two classes but got %s distinct y values!' %
          n_y)

    data = self.children[0].to_sql(table, split_by + self.group_by)
    with_data = data.with_data
    data.with_data = None
    table, _ = with_data.merge(sql.Datasource(data, 'DataToFit'))
    y = data.columns[-self.k - 1].alias
    xs = data.columns.aliases[-self.k:]
    if self.fit_intercept:
      xs.append('1')
    conds = []
    if split_by:
      slices = execute(
          str(sql.Sql(sql.Columns(split_by, True), table, with_data=with_data)))
      conds = slices.values
    self._gradients = None

    def grads(*unused_args):
      return self._gradients

    def hess(coef, converged):
      return compute_grads_and_hess(coef, converged)

    def compute_grads_and_hess(coef, converged):
      """Computes the gradients and Hessian matrices for coef.

      The grads we computes here is a n*k list of gradients, where n is the
      number of slices and k is the number of features. It has the same shape
      with coef, and represents the gradients of the coefficients. The
      gradients are saved to self._gradients as a side effect.
      Similarly, the Hessian matrices we return is a n*k*k array. Each k*k
      element is a Hessian matrix.

      Args:
        coef: A n*k array of coefficients being optimized. n is the number of
          slices and k is the number of features.
        converged: A list of the length of the number of slices. Its values
          indicate whether the coefficients of the slice have converged. If
          converged, we skip the computation for that slice.

      Returns:
        A n*k*k Hessian matrices. We also save the gradients to self._gradients
        as a side effect.
      """
      k = len(coef[0])
      if not split_by:
        grads, hessian = get_grads_and_hess_query(coef[0])
      else:
        grads = []
        hessian = []
        split_cols = sql.Columns(split_by).aliases
        for cond, slice_coef, done in zip(conds, coef, converged):
          if not done:
            condition = [
                '%s = "%s"' % (c, v) if isinstance(v, str) else '%s = %s' %
                (c, v) for c, v in zip(split_cols, cond)
            ]
            condition = ' AND '.join(condition)
            j, h = get_grads_and_hess_query(slice_coef, condition)
            grads += j
            hessian += h
      for i, c in enumerate(grads):
        c.set_alias('grads_%s' % i)
      for i, c in enumerate(hessian):
        c.set_alias('hess_%s' % i)
      grads_and_hess = sql.Sql(sql.Columns(grads + hessian), table)
      grads_and_hess.with_data = with_data
      grads_and_hess = execute(str(grads_and_hess)).iloc[0]
      self._gradients = []
      for done in converged:
        if done:
          self._gradients.append(None)
        else:
          self._gradients.append(grads_and_hess.values[:k])
          grads_and_hess = grads_and_hess[k:]
      hess_elements = list(grads_and_hess.values.reshape(-1, k * (k + 1) // 2))
      hess_arr = []
      for done in converged:
        if done:
          hess_arr.append(None)
        else:
          hess_arr.append(symmetrize_triangular(hess_elements.pop(0)))
      return hess_arr

    def get_grads_and_hess_query(coef, condition=None):
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

      Returns:
        grads: A list of SQL columns which has the same shape with coef. It
          represents the gradients of the corresponding coefficients.
        hess: A list of SQL columns that can be used to construct the Hessian
          matrix. Its elements are the upper triangular part of the Hessian,
          from left to right, top to down.
      """
      # A numerically stable implemntation, adapted from
      # http://fa.bianp.net/blog/2019/evaluate_logistic.
      z = ' + '.join('%s * %s' % (coef[i], xs[i]) for i in range(len(xs)))
      grads = [
          sql.Column(
              '%s * %s' % (x, sig_minus_b(z, y)), 'AVG({})', filters=condition)
          for x in xs
      ]
      sig_z = """IF({z} < 0,
          EXP({z}) / (1 + EXP({z})),
          1 / (1 + EXP(-({z}))))""".format(z=z)
      w = '-%s * %s' % (sig_z, sig_minus_b(z, 1))
      hess = []
      for i, x1 in enumerate(xs):
        for x2 in xs[i:]:
          hess.append(
              sql.Column(
                  '%s * %s * %s' % (x1, x2, w), 'AVG({})', filters=condition))
      hess = np.array(hess)
      # See here for the behavior of differnt penalties.
      # https://colab.research.google.com/drive/1Srfs4weM4LO9vt1HbOkGrD4kVbG8cso8
      n = 'COUNTIF(%s)' % condition if condition else 'COUNT(*)'
      if self.penalty == 'l1':
        for i in range(self.k):
          grads[i] += sql.Column('SIGN(%s) / %s' % (coef[i], n)) / self.c
      elif self.penalty == 'l2':
        for i in range(self.k):
          grads[i] += sql.Column('%s / %s' % (coef[i], n)) / self.c
        hess_diag_idx = np.arange(len(xs), len(xs) - self.k + 1, -1).cumsum()
        hess_diag_idx = np.concatenate([[0], hess_diag_idx])
        hess[hess_diag_idx] += sql.Column('1 / (%s * %s)' % (n, self.c))
      elif self.penalty == 'elasticnet':
        l1 = self.l1_ratio / self.c
        l2 = (1 - self.l1_ratio) / self.c
        for i in range(self.k):
          grads[i] += sql.Column('(%s * SIGN(%s) + %s * %s) / %s' %
                                 (l1, coef[i], l2, coef[i], n))
        hess_diag_idx = np.arange(len(xs), len(xs) - self.k + 1, -1).cumsum()
        hess_diag_idx = np.concatenate([[0], hess_diag_idx])
        hess[hess_diag_idx] += sql.Column('%s / %s' % (l2, n))
      elif self.penalty != 'none':
        raise ValueError('LogisticRegression supports only penalties in '
                         "['l1', 'l2', 'elasticnet', 'none'], got %s." %
                         self.penalty)
      return grads, list(hess)

    res = newtons_method(
        np.zeros((len(conds) or 1, len(xs))), grads, hess, self.tol,
        self.max_iter, conds)
    xs = [n.replace('macro_', '$').strip('`') for n in xs]
    if split_by:
      df = pd.DataFrame(conds, columns=split_by)
      if len(split_by) == 1:
        idx = pd.Index(df[split_by[0]])
      else:
        idx = pd.MultiIndex.from_frame((df))
      res = pd.DataFrame(res, columns=xs, index=idx)
      if self.fit_intercept:
        res.columns = list(res.columns[:-1]) + ['intercept']
        # Make intercept the 1st column.
        xs = ['intercept'] + xs[:-1]
        res = res[xs]
      res.sort_index(inplace=True)
      return res
    res = res[0]
    if self.fit_intercept:
      res = np.concatenate([[res[-1]], res[:-1]])
      xs = ['intercept'] + xs[:-1]
    return pd.DataFrame([res], columns=xs)


def sig_minus_b(z, b):
  """Computes sigmoid(z) - b in a numerically stable way in SQL."""
  # Adapted from http://fa.bianp.net/blog/2019/evaluate_logistic
  exp_z = 'EXP(%s)' % z
  exp_nz = 'EXP(-(%s))' % z
  return ('IF({z} < 0, ((1 - {b}) * {exp_z} - {b}) / (1 + {exp_z}), ((1 - {b}) '
          '- {b} * {exp_nz}) / (1 + {exp_nz}))').format(
              z=z, b=b, exp_z=exp_z, exp_nz=exp_nz)


def newtons_method(coef, grads, hess, tol, max_iter, conds, *args):
  """Uses Newton's method to optimize coef on n slices at the same time."""
  n_slice = len(coef)
  converged = np.array([False] * n_slice)
  for _ in range(max_iter):
    h = hess(coef, converged, *args)
    j = grads(coef, converged, *args)
    for i in range(n_slice):
      if not converged[i]:
        delta = np.linalg.solve(h[i], j[i])
        if abs(delta).max() < tol:
          converged[i] = True
        coef[i] -= delta
    if all(converged):
      return coef
  if n_slice == 1:
    print("WARNING: Optimization didn't converge!")
  else:
    print("WARNING: Optimization didn't converge for slice: ",
          np.array(conds)[~converged])
  return coef


def count_features(m: metrics.Metric):
  """Gets the width of the result of m.compute_on()."""
  if isinstance(m, Model):
    return m.k
  if isinstance(m, metrics.MetricList):
    return sum([count_features(i) for i in m])
  if isinstance(m, operations.MetricWithCI):
    return count_features(
        m.children[0]) * 3 if m.confidence else count_features(
            m.children[0]) * 2
  if isinstance(m, operations.Operation):
    return count_features(m.children[0])
  if isinstance(m, metrics.CompositeMetric):
    return max([count_features(i) for i in m.children])
  if isinstance(m, metrics.Quantile):
    if m.one_quantile:
      return 1
    return len(m.quantile)
  return 1
