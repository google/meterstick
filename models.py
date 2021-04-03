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
               name=None):
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
    """
    self.group_by = [group_by] if isinstance(group_by, str) else group_by or []
    if isinstance(x, Sequence):
      x = metrics.MetricList(x)
    self.model = model
    self.k = len(x) if isinstance(x, metrics.MetricList) else 1
    if not name:
      x_names = [m.name for m in x] if isinstance(
          x, metrics.MetricList) else [x.name]
      name = '%s(%s ~ %s)' % (model_name, y.name, ' + '.join(x_names))
    self.name = name
    super(Model, self).__init__(
        metrics.MetricList((y, x)), name, group_by, where=where)

  def compute(self, df):
    self.model.fit(df.iloc[:, 1:], df.iloc[:, 0])
    coef = self.model.coef_
    names = ['beta%s' % i for i in range(1, self.k + 1)]
    if self.model.fit_intercept:
      intercept = self.model.intercept_
      coef = [intercept] + list(coef)
      names = ['intercept'] + names
    return pd.Series(coef, index=pd.Index(names, name='coefficient'))

  def manipulate(self, res, melted, return_dataframe=True):
    return super(operations.Operation, self).manipulate(res, melted,
                                                        return_dataframe)


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
                                           name)


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
    super(Ridge, self).__init__(y, x, group_by, model, 'Ridge', where, name)


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
    super(Lasso, self).__init__(y, x, group_by, model, 'Lasso', where, name)


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
    super(LogisticRegression, self).__init__(y, x, group_by, model,
                                             'LogisticRegression', where, name)

  def compute(self, df):
    self.model.fit(df.iloc[:, 1:], df.iloc[:, 0])
    coef = self.model.coef_
    names = ['beta%s' % i for i in range(1, self.k + 1)]
    if coef.shape[0] == 1:
      coef = coef[0]
      if self.model.fit_intercept:
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
      if self.model.fit_intercept:
        intercept = pd.DataFrame(
            self.model.intercept_,
            index=pd.Index(self.model.classes_, name='class'),
            columns=['intercept'])
        res = intercept.join(res)
      res = res.stack()
      res.index.set_names('coefficient', -1, inplace=True)
      return res
