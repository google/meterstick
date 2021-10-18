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

from absl.testing import absltest
from absl.testing import parameterized
from meterstick import metrics
from meterstick import models
from meterstick import operations
from meterstick import utils
import numpy as np
import pandas as pd
from sklearn import linear_model

np.random.seed(42)
n = 40
DF = pd.DataFrame({
    'X1': np.random.random(n),
    'X2': np.random.random(n),
    'Y': np.random.randint(0, 100, n),
    'grp1': np.random.choice(['A', 'B', 'C'], n),
    'grp2': np.random.choice(('foo', 'bar'), n),
})
GRPED1 = DF.groupby('grp1').sum()
GRPED2 = DF.groupby(['grp2', 'grp1']).sum()


@parameterized.named_parameters(
    ('linear_regression', models.LinearRegression,
     linear_model.LinearRegression, 'OLS'),
    ('ridge', models.Ridge, linear_model.Ridge, 'Ridge'),
    ('lasso', models.Lasso, linear_model.Lasso, 'Lasso'),
    ('elastic_net', models.ElasticNet, linear_model.ElasticNet, 'ElasticNet'))
class ModelsTest(parameterized.TestCase):

  def test_model(self, model, sklearn_model, name):
    m = model(metrics.Sum('Y'), metrics.Sum('X1'), 'grp1')
    output = m.compute_on(DF)
    model = sklearn_model().fit(GRPED1[['X1']], GRPED1[['Y']])
    expected = pd.DataFrame(
        [[model.intercept_[0], model.coef_.flatten()[0]]],
        columns=[
            name + '(sum(Y) ~ sum(X1)) Coefficient: intercept',
            name + '(sum(Y) ~ sum(X1)) Coefficient: sum(X1)'
        ])
    pd.testing.assert_frame_equal(output, expected)

  def test_melted(self, model, sklearn_model, name):
    del sklearn_model, name  # unused
    m = model(metrics.Sum('Y'), metrics.Sum('X1'), 'grp1')
    output = m.compute_on(DF, melted=True)
    expected = utils.melt(m.compute_on(DF))
    pd.testing.assert_frame_equal(output, expected)

  def test_multi_var(self, model, sklearn_model, name):
    m = model(metrics.Sum('Y'), [metrics.Sum('X1'), metrics.Sum('X2')], 'grp1')
    output = m.compute_on(DF)
    model = sklearn_model().fit(GRPED1[['X1', 'X2']], GRPED1[['Y']])
    expected = pd.DataFrame(
        [[
            model.intercept_[0],
            model.coef_.flatten()[0],
            model.coef_.flatten()[1]
        ]],
        columns=[
            name + '(sum(Y) ~ sum(X1) + sum(X2)) Coefficient: intercept',
            name + '(sum(Y) ~ sum(X1) + sum(X2)) Coefficient: sum(X1)',
            name + '(sum(Y) ~ sum(X1) + sum(X2)) Coefficient: sum(X2)'
        ])
    pd.testing.assert_frame_equal(output, expected)

  def test_split_by(self, model, sklearn_model, name):
    m = model(metrics.Sum('Y'), metrics.Sum('X1'), 'grp1')
    output = m.compute_on(DF, 'grp2')
    model = sklearn_model()
    model.fit(GRPED2.loc['bar'][['X1']], GRPED2.loc['bar'][['Y']])
    expected1 = pd.DataFrame(
        [[model.intercept_[0], model.coef_.flatten()[0]]],
        columns=[
            name + '(sum(Y) ~ sum(X1)) Coefficient: intercept',
            name + '(sum(Y) ~ sum(X1)) Coefficient: sum(X1)'
        ])
    model.fit(GRPED2.loc['foo'][['X1']], GRPED2.loc['foo'][['Y']])
    expected2 = pd.DataFrame(
        [[model.intercept_[0], model.coef_.flatten()[0]]],
        columns=[
            name + '(sum(Y) ~ sum(X1)) Coefficient: intercept',
            name + '(sum(Y) ~ sum(X1)) Coefficient: sum(X1)'
        ])
    expected = pd.concat([expected1, expected2],
                         keys=['bar', 'foo'],
                         names=['grp2'])
    expected = expected.droplevel(-1)
    pd.testing.assert_frame_equal(output, expected)

  def test_no_intercept(self, model, sklearn_model, name):
    m = model(metrics.Sum('Y'), metrics.Sum('X1'), 'grp1', fit_intercept=False)
    output = m.compute_on(DF)
    model = sklearn_model(fit_intercept=False)
    model.fit(GRPED1[['X1']], GRPED1[['Y']])
    expected = pd.DataFrame(
        [[model.coef_.flatten()[0]]],
        columns=[name + '(sum(Y) ~ sum(X1)) Coefficient: sum(X1)'])
    pd.testing.assert_frame_equal(output, expected)

  def test_operation_on(self, model, sklearn_model, name):
    del sklearn_model  # unused
    m = model(metrics.Sum('Y'), metrics.Sum('X1'), 'grp1')
    output = operations.Distribution('grp2', m).compute_on(DF)
    expected = m.compute_on(DF, 'grp2') / m.compute_on(DF, 'grp2').sum()
    expected.columns = [
        'Distribution of %s(sum(Y) ~ sum(X1)) Coefficient: intercept' % name,
        'Distribution of %s(sum(Y) ~ sum(X1)) Coefficient: sum(X1)' % name
    ]
    pd.testing.assert_frame_equal(output, expected)


class LogisticRegressionTest(absltest.TestCase):

  def test_model(self):
    m = models.LogisticRegression(metrics.Sum('grp2'), metrics.Sum('X1'), 'X1')
    output = m.compute_on(DF)
    model = linear_model.LogisticRegression().fit(DF[['X1']], DF[['grp2']])
    expected = pd.DataFrame(
        [[model.intercept_[0], model.coef_.flatten()[0]]],
        columns=[
            'LogisticRegression(sum(grp2) ~ sum(X1)) Coefficient: intercept',
            'LogisticRegression(sum(grp2) ~ sum(X1)) Coefficient: sum(X1)'
        ])
    pd.testing.assert_frame_equal(output, expected)

  def test_melted(self):
    m = models.LogisticRegression(metrics.Sum('grp2'), metrics.Sum('X1'), 'X1')
    output = m.compute_on(DF, melted=True)
    expected = utils.melt(m.compute_on(DF))
    pd.testing.assert_frame_equal(output, expected)

  def test_multi_var(self):
    m = models.LogisticRegression(
        metrics.Sum('grp2'),
        [metrics.Sum('X1'), metrics.Sum('X2')], 'X1')
    output = m.compute_on(DF)
    model = linear_model.LogisticRegression().fit(DF[['X1', 'X2']],
                                                  DF[['grp2']])
    expected = pd.DataFrame(
        [[
            model.intercept_[0],
            model.coef_.flatten()[0],
            model.coef_.flatten()[1]
        ]],
        columns=[
            'LogisticRegression(sum(grp2) ~ sum(X1) + sum(X2)) Coefficient: intercept',
            'LogisticRegression(sum(grp2) ~ sum(X1) + sum(X2)) Coefficient: sum(X1)',
            'LogisticRegression(sum(grp2) ~ sum(X1) + sum(X2)) Coefficient: sum(X2)'
        ])
    pd.testing.assert_frame_equal(output, expected)

  def test_split_by(self):
    m = models.LogisticRegression(metrics.Sum('grp2'), metrics.Sum('X1'), 'X1')
    output = m.compute_on(DF, 'grp1')
    res = []
    grps = ['A', 'B', 'C']
    for g in grps:
      expected = m.compute_on(DF[DF.grp1 == g])
      res.append(expected)
    expected = pd.concat(res, keys=grps, names=['grp1'])
    expected = expected.droplevel(-1)
    pd.testing.assert_frame_equal(output, expected)

  def test_no_intercept(self):
    m = models.LogisticRegression(
        metrics.Sum('grp2'), metrics.Sum('X1'), 'X1', fit_intercept=False)
    output = m.compute_on(DF)
    model = linear_model.LogisticRegression(fit_intercept=False)
    model.fit(DF[['X1']], DF[['grp2']])
    expected = pd.DataFrame(
        [[model.coef_.flatten()[0]]],
        columns=[
            'LogisticRegression(sum(grp2) ~ sum(X1)) Coefficient: sum(X1)'
        ])
    pd.testing.assert_frame_equal(output, expected)

  def test_operation_on(self):
    m = models.LogisticRegression(
        metrics.Sum('grp2'), metrics.Sum('X1'), 'X1', name='LR')
    output = operations.Distribution('grp1', m).compute_on(DF)
    expected = m.compute_on(DF, 'grp1') / m.compute_on(DF, 'grp1').sum()
    expected.columns = [
        'Distribution of LR Coefficient: intercept',
        'Distribution of LR Coefficient: sum(X1)'
    ]
    pd.testing.assert_frame_equal(output, expected)

  def test_logistic_regression_multi_classes(self):
    m = models.LogisticRegression(
        metrics.Sum('Y'),
        [metrics.Sum('X1'), metrics.Sum('X2')], 'grp1')
    output = m.compute_on(DF)
    res = []
    model = linear_model.LogisticRegression()
    model.fit(GRPED1[['X1', 'X2']], GRPED1['Y'])
    prefix = 'LogisticRegression(sum(Y) ~ sum(X1) + sum(X2)) Coefficient: '
    for c, i, cl in zip(model.coef_, model.intercept_, model.classes_):
      expected = pd.DataFrame([[i, c[0], c[1]]],
                              columns=[
                                  prefix + 'intercept for class %s' % cl,
                                  prefix + 'sum(X1) for class %s' % cl,
                                  prefix + 'sum(X2) for class %s' % cl
                              ])
      res.append(expected)
    expected = pd.concat(res, 1)
    pd.testing.assert_frame_equal(output, expected)

  def test_logistic_regression_multi_classes_no_intercept(self):
    m = models.LogisticRegression(
        metrics.Sum('Y'), metrics.Sum('X1'), 'grp1', fit_intercept=False)
    output = m.compute_on(DF)
    res = []
    model = linear_model.LogisticRegression(fit_intercept=False)
    model.fit(GRPED1[['X1']], GRPED1['Y'])
    for c, cl in zip(model.coef_, model.classes_):
      expected = pd.DataFrame(
          [c],
          columns=[
              'LogisticRegression(sum(Y) ~ sum(X1)) Coefficient: '
              'sum(X1) for class %s' % cl
          ])
      res.append(expected)
    expected = pd.concat(res, 1)
    pd.testing.assert_frame_equal(output, expected)


class MiscellaneousTests(absltest.TestCase):

  def test_model_composition(self):
    lm = models.LinearRegression(
        metrics.Sum('Y'), metrics.Sum('X1'), 'grp1', name='lm')
    ridge = models.Ridge(
        metrics.Sum('Y'), metrics.Sum('X1'), 'grp1', name='ridge')
    output = (lm - ridge).compute_on(DF)
    a = lm.compute_on(DF, return_dataframe=False)
    b = ridge.compute_on(DF, return_dataframe=False)
    cols = [
        'lm Coefficient: intercept - ridge Coefficient: intercept',
        'lm Coefficient: sum(X1) - ridge Coefficient: sum(X1)'
    ]
    a.columns = cols
    b.columns = cols
    expected = a - b
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
  absltest.main()
