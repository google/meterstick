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
"""Tests for meterstick.v2.models."""

from meterstick import metrics
from meterstick import models
from meterstick import operations
from meterstick import utils
import pandas as pd
from sklearn import linear_model

import unittest


class ModelsTest(unittest.TestCase):

  df = pd.DataFrame({
      'X1': range(30),
      'X2': range(10, 40),
      'Y': range(20, 50),
      'grp1': ['A', 'B', 'C'] * 10,
      'grp2': ['foo', 'bar'] * 15
  })
  grped1 = df.groupby('grp1').sum()
  grped2 = df.groupby(['grp2', 'grp1']).sum()

  def test_linear_regression(self):
    m = models.LinearRegression(metrics.Sum('Y'), metrics.Sum('X1'), 'grp1')
    output = m.compute_on(self.df)
    model = linear_model.LinearRegression()
    model.fit(self.grped1[['X1']], self.grped1[['Y']])
    expected = pd.DataFrame(
        {'OLS(sum(Y) ~ sum(X1))': [model.intercept_[0], model.coef_[0][0]]},
        index=['intercept', 'sum(X1)'])
    expected.index.names = ['coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_linear_regression_melted(self):
    m = models.LinearRegression(metrics.Sum('Y'), metrics.Sum('X1'), 'grp1')
    output = m.compute_on(self.df, melted=True)
    expected = utils.melt(m.compute_on(self.df))
    pd.testing.assert_frame_equal(output, expected)

  def test_linear_regression_multi_var(self):
    m = models.LinearRegression(
        metrics.Sum('Y'),
        [metrics.Sum('X1'), metrics.Sum('X2')], 'grp1')
    output = m.compute_on(self.df)
    model = linear_model.LinearRegression()
    model.fit(self.grped1[['X1', 'X2']], self.grped1[['Y']])
    expected = pd.DataFrame(
        {
            'OLS(sum(Y) ~ sum(X1) + sum(X2))':
                [model.intercept_[0], model.coef_[0][0], model.coef_[0][1]]
        },
        index=['intercept', 'sum(X1)', 'sum(X2)'])
    expected.index.names = ['coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_linear_regression_split_by(self):
    m = models.LinearRegression(metrics.Sum('Y'), metrics.Sum('X1'), 'grp1')
    output = m.compute_on(self.df, 'grp2')
    model = linear_model.LinearRegression()
    model.fit(self.grped2.loc['bar'][['X1']], self.grped2.loc['bar'][['Y']])
    expected1 = pd.DataFrame(
        {'OLS(sum(Y) ~ sum(X1))': [model.intercept_[0], model.coef_[0][0]]},
        index=['intercept', 'sum(X1)'])
    model.fit(self.grped2.loc['foo'][['X1']], self.grped2.loc['foo'][['Y']])
    expected2 = pd.DataFrame(
        {'OLS(sum(Y) ~ sum(X1))': [model.intercept_[0], model.coef_[0][0]]},
        index=['intercept', 'sum(X1)'])
    expected = pd.concat([expected1, expected2],
                         keys=['bar', 'foo'],
                         names=['grp2'])
    expected.index.names = ['grp2', 'coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_linear_regression_no_intercept(self):
    m = models.LinearRegression(
        metrics.Sum('Y'), metrics.Sum('X1'), 'grp1', False)
    output = m.compute_on(self.df)
    model = linear_model.LinearRegression(fit_intercept=False)
    model.fit(self.grped1[['X1']], self.grped1[['Y']])
    expected = pd.DataFrame({'OLS(sum(Y) ~ sum(X1))': model.coef_[0]},
                            index=['sum(X1)'])
    expected.index.names = ['coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_operation_on_linear_regression(self):
    m = models.LinearRegression(metrics.Sum('Y'), metrics.Sum('X1'), 'grp1')
    output = operations.Distribution('grp2', m).compute_on(self.df)
    expected = m.compute_on(self.df, 'grp2') / m.compute_on(self.df,
                                                            'grp2').sum()
    expected.columns = ['Distribution of OLS(sum(Y) ~ sum(X1))']
    pd.testing.assert_frame_equal(output, expected)

  def test_ridge(self):
    m = models.Ridge(metrics.Sum('Y'), metrics.Sum('X1'), 'grp1')
    output = m.compute_on(self.df)
    model = linear_model.Ridge()
    model.fit(self.grped1[['X1']], self.grped1[['Y']])
    expected = pd.DataFrame(
        {'Ridge(sum(Y) ~ sum(X1))': [model.intercept_[0], model.coef_[0][0]]},
        index=['intercept', 'sum(X1)'])
    expected.index.names = ['coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_ridge_multi_var(self):
    m = models.Ridge(
        metrics.Sum('Y'),
        metrics.MetricList([metrics.Sum('X1'),
                            metrics.Sum('X2')]), 'grp1')
    output = m.compute_on(self.df)
    model = linear_model.Ridge()
    model.fit(self.grped1[['X1', 'X2']], self.grped1[['Y']])
    expected = pd.DataFrame(
        {
            'Ridge(sum(Y) ~ sum(X1) + sum(X2))':
                [model.intercept_[0], model.coef_[0][0], model.coef_[0][1]]
        },
        index=['intercept', 'sum(X1)', 'sum(X2)'])
    expected.index.names = ['coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_ridge_split_by(self):
    m = models.Ridge(metrics.Sum('Y'), metrics.Sum('X1'), 'grp1')
    output = m.compute_on(self.df, 'grp2')
    model = linear_model.Ridge()
    model.fit(self.grped2.loc['bar'][['X1']], self.grped2.loc['bar'][['Y']])
    expected1 = pd.DataFrame(
        {'Ridge(sum(Y) ~ sum(X1))': [model.intercept_[0], model.coef_[0][0]]},
        index=['intercept', 'sum(X1)'])
    model.fit(self.grped2.loc['foo'][['X1']], self.grped2.loc['foo'][['Y']])
    expected2 = pd.DataFrame(
        {'Ridge(sum(Y) ~ sum(X1))': [model.intercept_[0], model.coef_[0][0]]},
        index=['intercept', 'sum(X1)'])
    expected = pd.concat([expected1, expected2],
                         keys=['bar', 'foo'],
                         names=['grp2'])
    expected.index.names = ['grp2', 'coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_ridge_no_intercept(self):
    m = models.Ridge(
        metrics.Sum('Y'), metrics.Sum('X1'), 'grp1', fit_intercept=False)
    output = m.compute_on(self.df)
    model = linear_model.Ridge(fit_intercept=False)
    model.fit(self.grped1[['X1']], self.grped1[['Y']])
    expected = pd.DataFrame({'Ridge(sum(Y) ~ sum(X1))': model.coef_[0]},
                            index=['sum(X1)'])
    expected.index.names = ['coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_lasso(self):
    m = models.Lasso(metrics.Sum('Y'), metrics.Sum('X1'), 'grp1')
    output = m.compute_on(self.df)
    model = linear_model.Lasso()
    model.fit(self.grped1[['X1']], self.grped1[['Y']])
    expected = pd.DataFrame(
        {'Lasso(sum(Y) ~ sum(X1))': [model.intercept_[0], model.coef_[0]]},
        index=['intercept', 'sum(X1)'])
    expected.index.names = ['coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_lasso_multi_var(self):
    m = models.Lasso(
        metrics.Sum('Y'),
        [metrics.Sum('X1'), metrics.Sum('X2')], 'grp1')
    output = m.compute_on(self.df)
    model = linear_model.Lasso()
    model.fit(self.grped1[['X1', 'X2']], self.grped1[['Y']])
    expected = pd.DataFrame(
        {
            'Lasso(sum(Y) ~ sum(X1) + sum(X2))':
                [model.intercept_[0], model.coef_[0], model.coef_[1]]
        },
        index=['intercept', 'sum(X1)', 'sum(X2)'])
    expected.index.names = ['coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_lasso_split_by(self):
    m = models.Lasso(metrics.Sum('Y'), metrics.Sum('X1'), 'grp1')
    output = m.compute_on(self.df, 'grp2')
    model = linear_model.Lasso()
    model.fit(self.grped2.loc['bar'][['X1']], self.grped2.loc['bar'][['Y']])
    expected1 = pd.DataFrame(
        {'Lasso(sum(Y) ~ sum(X1))': [model.intercept_[0], model.coef_[0]]},
        index=['intercept', 'sum(X1)'])
    model.fit(self.grped2.loc['foo'][['X1']], self.grped2.loc['foo'][['Y']])
    expected2 = pd.DataFrame(
        {'Lasso(sum(Y) ~ sum(X1))': [model.intercept_[0], model.coef_[0]]},
        index=['intercept', 'sum(X1)'])
    expected = pd.concat([expected1, expected2],
                         keys=['bar', 'foo'],
                         names=['grp2'])
    expected.index.names = ['grp2', 'coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_lasso_no_intercept(self):
    m = models.Lasso(
        metrics.Sum('Y'), metrics.Sum('X1'), 'grp1', fit_intercept=False)
    output = m.compute_on(self.df)
    model = linear_model.Lasso(fit_intercept=False)
    model.fit(self.grped1[['X1']], self.grped1[['Y']])
    expected = pd.DataFrame({'Lasso(sum(Y) ~ sum(X1))': model.coef_[0]},
                            index=['sum(X1)'])
    expected.index.names = ['coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_elastic_net(self):
    m = models.ElasticNet(metrics.Sum('Y'), metrics.Sum('X1'), 'grp1')
    output = m.compute_on(self.df)
    model = linear_model.ElasticNet()
    model.fit(self.grped1[['X1']], self.grped1[['Y']])
    expected = pd.DataFrame(
        {'ElasticNet(sum(Y) ~ sum(X1))': [model.intercept_[0], model.coef_[0]]},
        index=['intercept', 'sum(X1)'])
    expected.index.names = ['coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_elastic_net_multi_var(self):
    m = models.ElasticNet(
        metrics.Sum('Y'),
        [metrics.Sum('X1'), metrics.Sum('X2')], 'grp1')
    output = m.compute_on(self.df)
    model = linear_model.ElasticNet()
    model.fit(self.grped1[['X1', 'X2']], self.grped1[['Y']])
    expected = pd.DataFrame(
        {
            'ElasticNet(sum(Y) ~ sum(X1) + sum(X2))':
                [model.intercept_[0], model.coef_[0], model.coef_[1]]
        },
        index=['intercept', 'sum(X1)', 'sum(X2)'])
    expected.index.names = ['coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_elastic_net_split_by(self):
    m = models.ElasticNet(metrics.Sum('Y'), metrics.Sum('X1'), 'grp1')
    output = m.compute_on(self.df, 'grp2')
    model = linear_model.ElasticNet()
    model.fit(self.grped2.loc['bar'][['X1']], self.grped2.loc['bar'][['Y']])
    expected1 = pd.DataFrame(
        {'ElasticNet(sum(Y) ~ sum(X1))': [model.intercept_[0], model.coef_[0]]},
        index=['intercept', 'sum(X1)'])
    model.fit(self.grped2.loc['foo'][['X1']], self.grped2.loc['foo'][['Y']])
    expected2 = pd.DataFrame(
        {'ElasticNet(sum(Y) ~ sum(X1))': [model.intercept_[0], model.coef_[0]]},
        index=['intercept', 'sum(X1)'])
    expected = pd.concat([expected1, expected2],
                         keys=['bar', 'foo'],
                         names=['grp2'])
    expected.index.names = ['grp2', 'coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_elastic_net_no_intercept(self):
    m = models.ElasticNet(
        metrics.Sum('Y'), metrics.Sum('X1'), 'grp1', fit_intercept=False)
    output = m.compute_on(self.df)
    model = linear_model.ElasticNet(fit_intercept=False)
    model.fit(self.grped1[['X1']], self.grped1[['Y']])
    expected = pd.DataFrame({'ElasticNet(sum(Y) ~ sum(X1))': model.coef_[0]},
                            index=['sum(X1)'])
    expected.index.names = ['coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_logistic_regression(self):
    m = models.LogisticRegression(metrics.Sum('grp2'), metrics.Sum('X1'), 'X1')
    output = m.compute_on(self.df)
    model = linear_model.LogisticRegression()
    model.fit(self.df[['X1']], self.df['grp2'])
    expected = pd.DataFrame(
        {
            'LogisticRegression(sum(grp2) ~ sum(X1))':
                [model.intercept_[0], model.coef_[0][0]]
        },
        index=['intercept', 'sum(X1)'])
    expected.index.names = ['coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_logistic_regression_multi_var(self):
    m = models.LogisticRegression(
        metrics.Sum('grp2'),
        [metrics.Sum('X1'), metrics.Sum('X2')], 'X1')
    output = m.compute_on(self.df)
    model = linear_model.LogisticRegression()
    model.fit(self.df[['X1', 'X2']], self.df['grp2'])
    expected = pd.DataFrame(
        {
            'LogisticRegression(sum(grp2) ~ sum(X1) + sum(X2))':
                [model.intercept_[0], model.coef_[0][0], model.coef_[0][1]]
        },
        index=['intercept', 'sum(X1)', 'sum(X2)'])
    expected.index.names = ['coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_logistic_regression_split_by(self):
    m = models.LogisticRegression(metrics.Sum('grp2'), metrics.Sum('X1'), 'X1')
    output = m.compute_on(self.df, 'grp1')
    model = linear_model.LogisticRegression()
    res = []
    grps = ['A', 'B', 'C']
    for g in grps:
      model.fit(self.df[self.df.grp1 == g][['X1']],
                self.df[self.df.grp1 == g]['grp2'])
      expected = pd.DataFrame(
          {
              'LogisticRegression(sum(grp2) ~ sum(X1))':
                  [model.intercept_[0], model.coef_[0][0]]
          },
          index=['intercept', 'sum(X1)'])
      res.append(expected)
    expected = pd.concat(res, keys=grps, names=['grp1'])
    expected.index.names = ['grp1', 'coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_logistic_regression_no_intercept(self):
    m = models.LogisticRegression(
        metrics.Sum('grp2'), metrics.Sum('X1'), 'X1', False)
    output = m.compute_on(self.df)
    model = linear_model.LogisticRegression(fit_intercept=False)
    model.fit(self.df[['X1']], self.df['grp2'])
    expected = pd.DataFrame(
        {'LogisticRegression(sum(grp2) ~ sum(X1))': model.coef_[0]},
        index=['sum(X1)'])
    expected.index.names = ['coefficient']
    pd.testing.assert_frame_equal(output, expected)

  def test_logistic_regression_multi_classes(self):
    m = models.LogisticRegression(metrics.Sum('Y'), metrics.Sum('X1'), 'grp1')
    output = m.compute_on(self.df)
    res = []
    model = linear_model.LogisticRegression()
    for i, c in enumerate(self.grped1.Y):
      model.fit(self.grped1[['X1']], self.grped1['Y'])
      expected = pd.DataFrame({
          'coefficient': ['intercept', 'sum(X1)'],
          'LogisticRegression(sum(Y) ~ sum(X1))': [
              model.intercept_[i], model.coef_[i][0]
          ]
      })
      expected['class'] = c
      res.append(expected)
    expected = pd.concat(res).set_index(['class', 'coefficient'])
    pd.testing.assert_frame_equal(output, expected)

  def test_logistic_regression_multi_classes_no_intercept(self):
    m = models.LogisticRegression(
        metrics.Sum('Y'), metrics.Sum('X1'), 'grp1', fit_intercept=False)
    output = m.compute_on(self.df)
    res = []
    model = linear_model.LogisticRegression(fit_intercept=False)
    for i, c in enumerate(self.grped1.Y):
      model.fit(self.grped1[['X1']], self.grped1['Y'])
      expected = pd.DataFrame({
          'coefficient': ['sum(X1)'],
          'LogisticRegression(sum(Y) ~ sum(X1))': model.coef_[i]
      })
      expected['class'] = c
      res.append(expected)
    expected = pd.concat(res).set_index(['class', 'coefficient'])
    pd.testing.assert_frame_equal(output, expected)

  def test_model_composition(self):
    lm = models.LinearRegression(metrics.Sum('Y'), metrics.Sum('X1'), 'grp1')
    ridge = models.Ridge(metrics.Sum('Y'), metrics.Sum('X1'), 'grp1')
    output = (lm - ridge).compute_on(self.df)
    a = lm.compute_on(self.df, return_dataframe=False)
    b = ridge.compute_on(self.df, return_dataframe=False)
    expected = pd.DataFrame(a - b)
    expected.columns = ['OLS(sum(Y) ~ sum(X1)) - Ridge(sum(Y) ~ sum(X1))']
    pd.testing.assert_frame_equal(output, expected)

  def test_count_features(self):
    s = metrics.Sum('x')
    self.assertEqual(models.count_features(metrics.Sum('x')), 1)
    self.assertEqual(models.count_features(metrics.MetricList([s, s])), 2)
    self.assertEqual(
        models.count_features(
            metrics.MetricList([metrics.Sum('x'),
                                metrics.MetricList([s])])), 2)
    self.assertEqual(
        models.count_features(operations.AbsoluteChange('a', 'b', s)), 1)
    self.assertEqual(
        models.count_features(
            operations.AbsoluteChange(
                'a', 'b', metrics.MetricList([s, metrics.MetricList([s])]))), 2)
    self.assertEqual(
        models.count_features(
            operations.AbsoluteChange(
                'a', 'b',
                metrics.MetricList([
                    operations.AbsoluteChange('a', 'b',
                                              metrics.MetricList([s, s]))
                ]))), 2)
    self.assertEqual(models.count_features(metrics.Ratio('x', 'y')), 1)
    self.assertEqual(models.count_features(metrics.MetricList([s, s]) / 2), 2)


if __name__ == '__main__':
  unittest.main()
