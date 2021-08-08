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

from typing import List, Optional, Sequence, Text, Union

from meterstick import metrics
from meterstick import operations
from meterstick import sql
from meterstick import utils
import numpy as np
import pandas as pd
from sklearn import linear_model
_SCIPY_IMPORT_ERR = None
try:
  from scipy import optimize  # pylint: disable=g-import-not-at-top
except (ImportError, ModuleNotFoundError) as e:
  _SCIPY_IMPORT_ERR = e


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
      where: A string that will be passed to df.query() as a prefilter.
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
    self.name = name
    super(Model, self).__init__(
        metrics.MetricList((y, x)), name, group_by, where=where)
    self.computable_in_pure_sql = False
    self.fit_intercept = fit_intercept
    self.normalize = normalize

  def compute(self, df):
    self.model.fit(df.iloc[:, 1:], df.iloc[:, 0])
    coef = self.model.coef_
    names = list(df.columns[1:])
    if self.fit_intercept:
      intercept = self.model.intercept_
      coef = [intercept] + list(coef)
      names = ['intercept'] + names
    return pd.Series(coef, index=pd.Index(names, name='coefficient'))

  def manipulate(self,
                 res,
                 melted,
                 return_dataframe=True,
                 apply_name_tmpl=False):
    return super(operations.Operation,
                 self).manipulate(res, melted, return_dataframe,
                                  apply_name_tmpl)

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
        return self.compute_on_sql_magic_mode(table, split_by, execute)
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
    model = linear_model.LinearRegression(
        fit_intercept=fit_intercept, normalize=normalize)
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
      cols,
      table,
      groupby=sql.Columns(split_by).aliases,
      with_data=with_data)
  sufficient_stats = execute(str(sufficient_stats))
  fn = lambda row: compute_coef_from_sufficient_stats(row, xs, m)
  if split_by:
    sufficient_stats.columns = split_by + list(
        sufficient_stats.columns)[len(split_by):]
    return sufficient_stats.groupby(split_by).apply(fn).stack()
  return sufficient_stats.apply(fn, 1).stack()


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
  x_t_x = np.zeros([n, n])
  x_t_x[np.tril_indices(n)] = x_t_x_elements
  x_t_x = x_t_x + x_t_x.T - np.diag(x_t_x.diagonal())
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
        " The model coefficients might be inaccurate."
        % cond)
  coef = np.linalg.solve(x_t_x, x_t_y)
  xs = [n.replace('macro_', '$').strip('`') for n in xs]
  if fit_intercept:
    xs = ['intercept'] + xs
  return pd.Series(coef, index=pd.Index(xs, name='coefficient'))


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
  x_t_x = np.zeros([n, n])
  x_t_x[np.tril_indices(n)] = x_t_x_elements
  x_t_x = x_t_x + x_t_x.T + (m.alpha - 1) * np.diag(x_t_x.diagonal())
  cond = np.linalg.cond(x_t_x)
  if cond > 20:
    print(
        "WARNING: The condition number of X'X is %i, which might be too large."
        " The model coefficients might be inaccurate."
        % cond)
  coef = np.linalg.solve(x_t_x, x_t_y)
  xs = [n.replace('macro_', '$').strip('`') for n in xs]
  intercept = sufficient_stats.y - coef.dot(
      [sufficient_stats['x%s' % i] for i in range(n)])
  coef = [intercept] + list(coef)
  xs = ['intercept'] + xs
  return pd.Series(coef, index=pd.Index(xs, name='coefficient'))


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
        normalize=normalize,
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
        normalize=normalize,
        copy_X=copy_X,
        max_iter=max_iter,
        tol=tol,
        warm_start=warm_start,
        positive=positive,
        random_state=random_state,
        selection=selection)
    super(Lasso, self).__init__(y, x, group_by, model, 'Lasso', where, name,
                                fit_intercept, normalize)


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
      return pd.Series(coef, index=pd.Index(names, name='coefficient'))
    else:
      res = pd.DataFrame(
          coef,
          index=pd.Index(self.model.classes_, name='class'),
          columns=names)
      if self.fit_intercept:
        intercept = pd.DataFrame(
            self.model.intercept_,
            index=pd.Index(self.model.classes_, name='class'),
            columns=['intercept'])
        res = intercept.join(res)
      res = res.stack()
      res.index.set_names('coefficient', -1, inplace=True)
      return res

  def compute_on_sql_magic_mode(self, table, split_by, execute=None):
    """Gets the coefficients by minimizing the cost function."""
    if self.model.class_weight:
      raise ValueError("Magic mode doesn't support class_weight!")
    if self.model.multi_class == 'multinomial':
      raise ValueError("Magic mode doesn't support multi_class!")
    if self.penalty == 'elasticnet' and (not self.l1_ratio or
                                         not 0 <= self.l1_ratio <= 1):
      raise ValueError('l1_ratio must be between 0 and 1; got (l1_ratio="%s")' %
                       self.l1_ratio)
    if _SCIPY_IMPORT_ERR:
      raise _SCIPY_IMPORT_ERR
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
      xs.append(str(self.intercept_scaling))
    conds = []
    if split_by:
      slices = execute(
          str(sql.Sql(sql.Columns(split_by, True), table, with_data=with_data)))
      conds = slices.values
    self._curr_coef = None
    self._cost_and_jac = None

    def cost(coef):
      if any(coef != self._curr_coef):
        compute_cost_and_jac(coef)
      self._curr_coef = coef
      return self._cost_and_jac[0]

    def jac(coef):
      if any(coef != self._curr_coef):
        compute_cost_and_jac(coef)
      self._curr_coef = coef
      return self._cost_and_jac[1:]

    def compute_cost_and_jac(coef):
      """Computes the cost and Jacobian and stores them."""
      if not split_by:
        cost, jac = get_cost_and_jac_query(coef)
      else:
        # Combine cost and Jacobian from all slices to optimize in one run.
        cost = sql.Column('0')
        jac = []
        split_cols = sql.Columns(split_by).aliases
        for cond, slice_coef in zip(conds, coef.reshape(-1, len(xs))):
          condition = [
              '%s = "%s"' % (c, v) if isinstance(v, str) else '%s = %s' % (c, v)
              for c, v in zip(split_cols, cond)
          ]
          condition = ' AND '.join(condition)
          c, j = get_cost_and_jac_query(slice_coef, condition)
          cost += c
          jac += j
      cost.set_alias('cost')
      for i, c in enumerate(jac):
        c.set_alias('jac_%s' % i)
      cost_and_jac = sql.Sql(sql.Columns([cost] + jac), table)
      cost_and_jac.with_data = with_data
      self._cost_and_jac = execute(str(cost_and_jac)).iloc[0]

    def get_cost_and_jac_query(coef, condition=None):
      """Get the query to compute the cost function and Jacobian."""
      # A numerically stable implemntation, adapted from
      # http://fa.bianp.net/blog/2019/evaluate_logistic.
      z = ' + '.join('%s * %s' % (coef[i], xs[i]) for i in range(len(xs)))
      logsig_z = """CASE
          WHEN {z} < -33 THEN ({z})
          WHEN {z} < -18 THEN ({z} - EXP({z}))
          WHEN {z} < 37 THEN -LN(EXP(-({z})) + 1)
          ELSE -EXP(-({z}))
        END""".format(z=z)
      cost = sql.Column(
          '(1 - %s) * (%s) - %s' % (y, z, logsig_z),
          'AVG({})',
          filters=condition)
      non_intercept = coef
      if self.fit_intercept and self.penalty != 'none':
        non_intercept = coef[:-1]
      s = '''IF({z} < 0,
              ((1 - {y}) * EXP({z}) - {y}) / (1 + EXP({z})),
              ((1 - {y}) - {y} * EXP(-({z}))) / (1 + EXP(-({z})))
          )'''.format(y=y, z=z)
      jac = [
          sql.Column('%s * %s' % (x, s), 'AVG({})', filters=condition)
          for x in xs
      ]
      # See here for the behavior of differnt penalties.
      # https://colab.research.google.com/drive/1Srfs4weM4LO9vt1HbOkGrD4kVbG8cso8
      n = 'COUNT(*)'
      if condition:
        n = 'COUNTIF(%s)' % condition
      if self.penalty == 'l1':
        penalty = '+'.join(map('ABS({})'.format, non_intercept))
        penalty = sql.Column('(%s) / %s' % (penalty, n)) / self.c
        cost += penalty
        for i in range(self.k):
          jac[i] += sql.Column('SIGN(%s) / %s' % (coef[i], n)) / self.c
      elif self.penalty == 'l2':
        penalty = '+'.join(map('POW({}, 2)'.format, non_intercept))
        penalty = sql.Column('(%s) / %s' % (penalty, n)) / self.c / 2
        cost += penalty
        for i in range(self.k):
          jac[i] += sql.Column('%s / %s' % (coef[i], n)) / self.c
      elif self.penalty == 'elasticnet':
        l1_penalty = '+'.join(map('ABS({})'.format, non_intercept))
        l1_penalty = sql.Column('(%s) / %s' % (l1_penalty, n))
        l2_penalty = '+'.join(map('POW({}, 2)'.format, non_intercept))
        l2_penalty = sql.Column('(%s) / %s' % (l2_penalty, n)) / 2
        l1 = self.l1_ratio / self.c
        l2 = (1 - self.l1_ratio) / self.c
        cost += l1 * l1_penalty + l2 * l2_penalty
        for i in range(self.k):
          jac[i] += sql.Column('(%s * SIGN(%s) + %s * %s) / %s' %
                               (l1, coef[i], l2, coef[i], n))
      elif self.penalty != 'none':
        raise ValueError(
            "LogisticRegression supports only penalties in "
            "['l1', 'l2', 'elasticnet', 'none'], got ."
            % self.penalty)
      return cost, jac

    res = optimize.minimize(
        cost, [0] * len(xs) * (len(conds) or 1),
        jac=jac,
        tol=self.tol,
        options={'maxiter': self.max_iter})
    if not res.success:
      print('WARNING: %s' % res.message)
    xs = [n.replace('macro_', '$').strip('`') for n in xs]
    res = list(res.x)
    if split_by:
      df = pd.DataFrame(conds, columns=split_by)
      res = pd.DataFrame(
          np.reshape(res, (-1, len(xs))),
          columns=xs,
          index=pd.MultiIndex.from_frame((df)))
      if self.fit_intercept:
        res.columns = list(res.columns[:-1]) + ['intercept']
        res['intercept'] *= self.intercept_scaling
        # Make intercept the 1st column.
        xs = ['intercept'] + xs[:-1]
        res = res[xs]
      res.sort_index(inplace=True)
      res = res.stack(dropna=False)
      res.index.names = split_by + ['coefficient']
      return res
    if self.fit_intercept:
      res = [self.intercept_scaling * res[-1]] + res[:-1]
      xs = ['intercept'] + xs[:-1]
    return pd.Series(res, index=pd.Index(xs, name='coefficient'))


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
