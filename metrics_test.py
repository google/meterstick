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
"""Tests for Metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from absl.testing import absltest
from absl.testing import parameterized
from meterstick import metrics
from meterstick import operations
from meterstick import utils
import mock
import numpy as np
import pandas as pd
from pandas import testing

# pylint: disable=g-long-lambda
METRICS_TO_TEST = metrics_to_test = [
    ('Ratio', metrics.Ratio('X', 'Y'), lambda d: d.X.sum() / d.Y.sum()),
    ('Sum', metrics.Sum('X'), lambda d: d.X.sum()),
    ('Count', metrics.Count('X'), lambda d: d.X.size),
    (
        'Count distinct',
        metrics.Count('X', distinct=True),
        lambda d: d.X.nunique(),
    ),
    ('Mean', metrics.Mean('X'), lambda d: d.X.mean()),
    (
        'Weighted Mean',
        metrics.Mean('X', 'Y'),
        lambda d: np.average(d.X, weights=d.Y),
    ),
    ('Dot', metrics.Dot('X', 'Y'), lambda d: (d.X * d.Y).sum()),
    (
        'Normalized Dot',
        metrics.Dot('X', 'Y', True),
        lambda d: (d.X * d.Y).mean(),
    ),
    ('Max', metrics.Max('X'), lambda d: d.X.max()),
    ('Min', metrics.Min('X'), lambda d: d.X.min()),
    ('Quantile', metrics.Quantile('X', 0.2), lambda d: d.X.quantile(0.2)),
    ('Variance', metrics.Variance('X', True), lambda d: d.X.var()),
    (
        'Biased Variance',
        metrics.Variance('X', False),
        lambda d: d.X.var(ddof=0),
    ),
    (
        'Weighted Variance',
        metrics.Variance('X', True, 'Y'),
        lambda d: np.dot(d.Y, (d.X - np.average(d.X, weights=d.Y)) ** 2)
        / (d.Y.sum() - 1),
    ),
    (
        'Biased Weighted Variance',
        metrics.Variance('X', False, 'Y'),
        lambda d: float(np.cov(d.X, bias=True, aweights=d.Y)),
    ),
    (
        'StandardDeviation',
        metrics.StandardDeviation('X', True),
        lambda d: d.X.std(),
    ),
    (
        'Biased StandardDeviation',
        metrics.StandardDeviation('X', False),
        lambda d: d.X.std(ddof=0),
    ),
    (
        'Weighted StandardDeviation',
        metrics.StandardDeviation('X', True, 'Y'),
        lambda d: metrics.Variance('X', True, 'Y').compute_on(d).iloc[0, 0]
        ** 0.5,
    ),
    (
        'Biased Weighted StandardDeviation',
        metrics.StandardDeviation('X', False, 'Y'),
        lambda d: metrics.Variance('X', False, 'Y').compute_on(d).iloc[0, 0]
        ** 0.5,
    ),
    ('CV', metrics.CV('X', True), lambda d: d.X.std() / d.X.mean()),
    (
        'Biased CV',
        metrics.CV('X', False),
        lambda d: d.X.std(ddof=0) / d.X.mean(),
    ),
    ('Cov', metrics.Cov('X', 'Y', False), lambda d: np.cov(d.X, d.Y)[0, 1]),
    (
        'Biased Cov',
        metrics.Cov('X', 'Y', True),
        lambda d: np.cov(d.X, d.Y, bias=True)[0, 1],
    ),
    (
        'Cov ddof',
        metrics.Cov('X', 'Y', False, 2),
        lambda d: np.cov(d.X, d.Y, bias=False, ddof=2)[0, 1],
    ),
    (
        'Biased Cov ddof',
        metrics.Cov('X', 'Y', True, 2),
        lambda d: np.cov(d.X, d.Y, bias=True, ddof=2)[0, 1],
    ),
    (
        'Weighted Cov',
        metrics.Cov('X', 'Y', False, weight='w'),
        lambda d: np.cov(d.X, d.Y, bias=False, aweights=d.w)[0, 1],
    ),
    (
        'Biased Weighted Cov',
        metrics.Cov('X', 'Y', True, weight='w'),
        lambda d: np.cov(d.X, d.Y, bias=True, aweights=d.w)[0, 1],
    ),
    (
        'Weighted Cov ddof',
        metrics.Cov('X', 'Y', False, 2, 'w'),
        lambda d: np.cov(d.X, d.Y, bias=False, ddof=2, aweights=d.w)[0, 1],
    ),
    (
        'Biased Weighted Cov ddof',
        metrics.Cov('X', 'Y', True, 2, 'w'),
        lambda d: np.cov(d.X, d.Y, bias=True, ddof=2, aweights=d.w)[0, 1],
    ),
    (
        'Fweighted Cov',
        metrics.Cov('X', 'Y', False, fweight='w2'),
        lambda d: np.cov(d.X, d.Y, bias=False, fweights=d.w2)[0, 1],
    ),
    (
        'Biased Fweighted Cov',
        metrics.Cov('X', 'Y', True, fweight='w2'),
        lambda d: np.cov(d.X, d.Y, bias=True, fweights=d.w2)[0, 1],
    ),
    (
        'Fweighted Cov ddof',
        metrics.Cov('X', 'Y', False, 2, fweight='w2'),
        lambda d: np.cov(d.X, d.Y, bias=False, ddof=2, fweights=d.w2)[0, 1],
    ),
    (
        'Biased Fweighted Cov ddof',
        metrics.Cov('X', 'Y', True, 2, fweight='w2'),
        lambda d: np.cov(d.X, d.Y, bias=True, ddof=2, fweights=d.w2)[0, 1],
    ),
    (
        'Weighted and fweighted Cov',
        metrics.Cov('X', 'Y', False, None, 'w', 'w2'),
        lambda d: np.cov(d.X, d.Y, bias=False, aweights=d.w, fweights=d.w2)[
            0, 1
        ],
    ),
    (
        'Biased Weighted and fweighted Cov',
        metrics.Cov('X', 'Y', True, None, 'w', 'w2'),
        lambda d: np.cov(d.X, d.Y, bias=True, aweights=d.w, fweights=d.w2)[
            0, 1
        ],
    ),
    (
        'Weighted and fweighted Cov ddof',
        metrics.Cov('X', 'Y', False, 2, 'w', 'w2'),
        lambda d: np.cov(
            d.X, d.Y, bias=False, ddof=2, aweights=d.w, fweights=d.w2
        )[0, 1],
    ),
    (
        'Biased Weighted and fweighted Cov ddof',
        metrics.Cov('X', 'Y', True, 2, 'w', 'w2'),
        lambda d: np.cov(
            d.X, d.Y, bias=True, ddof=2, aweights=d.w, fweights=d.w2
        )[0, 1],
    ),
    (
        'Correlation',
        metrics.Correlation('X', 'Y'),
        lambda d: np.corrcoef(d.X, d.Y)[0, 1],
    ),
    (
        'Correlation kendall method',
        metrics.Correlation('X', 'Y', method='kendall'),
        lambda d: d[['X', 'Y']].corr(method='kendall').iloc[0, 1],
    ),
    (
        'Correlation spearman method',
        metrics.Correlation('X', 'Y', method='spearman'),
        lambda d: d[['X', 'Y']].corr(method='spearman').iloc[0, 1],
    ),
    (
        'Weighted Correlation',
        metrics.Correlation('X', 'Y', 'w'),
        lambda d: np.cov(d.X, d.Y, bias=True, aweights=d.w)[0, 1]
        / metrics.StandardDeviation('X', False, 'w').compute_on(d).iloc[0, 0]
        / metrics.StandardDeviation('Y', False, 'w').compute_on(d).iloc[0, 0],
    ),
]
# pylint: enable=g-long-lambda


@parameterized.named_parameters(*METRICS_TO_TEST)
class TestSimpleMetric(parameterized.TestCase):
  df = pd.DataFrame({
      'X': np.random.rand(100) + 5,
      'Y': np.random.rand(100) + 5,
      'w': np.random.rand(100) + 5,
      'w2': np.random.randint(100, size=100) + 5,
      'g1': np.random.randint(3, size=100),
      'g2': np.random.choice(list('ab'), size=100),
  })

  def test_return_number(self, m, fn):
    output = m.compute_on(self.df, return_dataframe=False)
    expected = fn(self.df)
    self.assertEqual(round(output, 10), round(expected, 10))

  def test_return_dataframe(self, m, fn):
    output = m.compute_on(self.df)
    expected = pd.DataFrame([[fn(self.df)]], columns=[m.name])
    testing.assert_frame_equal(output, expected)

  def test_return_dataframe_melted(self, m, fn):
    output = m.compute_on(self.df, melted=True)
    expected = pd.DataFrame({'Value': [fn(self.df)]}, index=[m.name])
    expected.index.name = 'Metric'
    testing.assert_frame_equal(output, expected)

  def test_split_by_return_series(self, m, fn):
    output = m.compute_on(self.df, 'g1', return_dataframe=False)
    expected = self.df.groupby('g1').apply(fn)
    expected.name = m.name
    testing.assert_series_equal(output, expected)

  def test_split_by_return_dataframe(self, m, fn):
    output = m.compute_on(self.df, 'g1')
    expected = pd.DataFrame(self.df.groupby('g1').apply(fn))
    expected.columns = [m.name]
    testing.assert_frame_equal(output, expected)

  def test_split_by_melted(self, m, _):
    output = m.compute_on(self.df, 'g1', melted=True)
    expected = utils.melt(m.compute_on(self.df, 'g1'))
    testing.assert_frame_equal(output, expected)

  def test_split_by_multiple(self, m, fn):
    output = m.compute_on(self.df, ['g1', 'g2'])
    expected = pd.DataFrame(self.df.groupby(['g1', 'g2']).apply(fn))
    expected.columns = [m.name]
    testing.assert_frame_equal(output, expected)

  def test_split_by_multiple_melted(self, m, _):
    output = m.compute_on(self.df, ['g1', 'g2'], melted=True)
    expected = utils.melt(m.compute_on(self.df, ['g1', 'g2']))
    testing.assert_frame_equal(output, expected)

  def test_where(self, m, fn):
    m = copy.deepcopy(m)
    m.where = 'X > 0.2'
    output = m.compute_on(self.df)
    expected = pd.DataFrame([[fn(self.df[self.df.X > 0.2])]], columns=[m.name])
    testing.assert_frame_equal(output, expected)

  def test_pipeline_operator(self, m, _):
    output = m | metrics.compute_on(self.df)
    expected = m.compute_on(self.df)
    testing.assert_frame_equal(output, expected)

  def test_metriclist(self, m, fn):
    output = metrics.MetricList([m], where='g1 == 1').compute_on(self.df)
    expected = pd.DataFrame([[fn(self.df[self.df.g1 == 1])]], columns=[m.name])
    testing.assert_frame_equal(output, expected)


class TestMetricsMiscellaneous(absltest.TestCase):
  df = pd.DataFrame({'X': [0, 1, 2, 3], 'Y': [0, 1, 1, 2]})

  def test_quantile_raise(self):
    with self.assertRaises(ValueError) as cm:
      metrics.Quantile('X', 2)
    self.assertEqual(str(cm.exception), 'quantiles must be in [0, 1].')

  def test_quantile_multiple_quantiles_raise(self):
    with self.assertRaises(ValueError) as cm:
      metrics.Quantile('X', [0.1, 2])
    self.assertEqual(str(cm.exception), 'quantiles must be in [0, 1].')

  def test_quantile_interpolation(self):
    metric = metrics.Quantile('X', 0.5, interpolation='lower')
    output = metric.compute_on(
        pd.DataFrame({'X': [1, 2]}), return_dataframe=False)
    self.assertEqual(output, 1)

  def test_quantile_multiple_quantiles(self):
    df = pd.DataFrame({'X': [0, 1]})
    metric = metrics.MetricList(
        [metrics.Quantile('X', [0.1, 0.5]),
         metrics.Count('X')])
    output = metric.compute_on(df)
    expected = pd.DataFrame(
        [[0.1, 0.5, 2]],
        columns=['quantile(X, 0.1)', 'quantile(X, 0.5)', 'count(X)'])
    testing.assert_frame_equal(output, expected)

  def test_quantile_multiple_quantiles_melted(self):
    df = pd.DataFrame({'X': [0, 1]})
    metric = metrics.MetricList(
        [metrics.Quantile('X', [0.1, 0.5]),
         metrics.Count('X')])
    output = metric.compute_on(df, melted=True)
    expected = pd.DataFrame(
        {'Value': [0.1, 0.5, 2]},
        index=['quantile(X, 0.1)', 'quantile(X, 0.5)', 'count(X)'])
    expected.index.name = 'Metric'
    testing.assert_frame_equal(output, expected)

  def test_weighted_quantile_not_df(self):
    df = pd.DataFrame({'X': [1, 2], 'Y': [3, 1]})
    metric = metrics.Quantile('X', weight='Y')
    output = metric.compute_on(df, return_dataframe=False)
    self.assertEqual(output, 1.25)

  def test_weighted_quantile_df(self):
    df = pd.DataFrame({'X': [1, 2], 'Y': [3, 1]})
    metric = metrics.Quantile('X', weight='Y')
    output = metric.compute_on(df)
    expected = pd.DataFrame({'Y-weighted quantile(X, 0.5)': [1.25]})
    testing.assert_frame_equal(output, expected)

  def test_weighted_quantile_multiple_quantiles_split_by(self):
    df = pd.DataFrame({
        'X': [0, 1, 2, 1, 2, 3],
        'Y': [1, 2, 2, 1, 1, 1],
        'grp': ['B'] * 3 + ['A'] * 3
    })
    metric = metrics.MetricList(
        [metrics.Quantile('X', [0.25, 0.5], weight='Y'),
         metrics.Sum('X')])
    output = metric.compute_on(df, 'grp')
    output.sort_index(level='grp', inplace=True)  # For Py2
    expected = pd.DataFrame(
        {
            'Y-weighted quantile(X, 0.25)': [1.25, 0.5],
            'Y-weighted quantile(X, 0.5)': [2., 1.25],
            'sum(X)': [6, 3]
        },
        index=['A', 'B'])
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_weighted_quantile_multiple_quantiles_split_by_melted(self):
    df = pd.DataFrame({
        'X': [0, 1, 2, 1, 2, 3],
        'Y': [1, 2, 2, 1, 1, 1],
        'grp': ['B'] * 3 + ['A'] * 3
    })
    metric = metrics.MetricList(
        [metrics.Quantile('X', [0.25, 0.5], weight='Y'),
         metrics.Sum('X')])
    output = metric.compute_on(df, 'grp', melted=True)
    output.sort_index(level=['Metric', 'grp'], inplace=True)  # For Py2
    expected = pd.DataFrame({'Value': [1.25, 0.5, 2., 1.25, 6., 3.]},
                            index=pd.MultiIndex.from_product(
                                ([
                                    'Y-weighted quantile(X, 0.25)',
                                    'Y-weighted quantile(X, 0.5)', 'sum(X)'
                                ], ['A', 'B']),
                                names=['Metric', 'grp']))
    testing.assert_frame_equal(output, expected)

  def test_cov_invalid_ddof(self):
    df = pd.DataFrame({'X': np.random.rand(3), 'w': np.array([1, 1, 2])})
    m = metrics.Cov('X', 'X', ddof=5, fweight='w')
    self.assertTrue(pd.isnull(m.compute_on(df, return_dataframe=False)))


class TestCompositeMetric(absltest.TestCase):
  """Tests for composition of two metrics."""

  df = pd.DataFrame({'X': [0, 1, 2, 3], 'Y': [0, 1, 1, 2]})

  def test_add(self):
    df = pd.DataFrame({'X': [1, 2, 3], 'Y': ['a', 'b', 'c']})
    sumx = metrics.Sum('X')
    metric = sumx + sumx
    output = metric.compute_on(df, 'Y', return_dataframe=False)
    expected = pd.Series([2, 4, 6], index=['a', 'b', 'c'])
    expected.name = 'sum(X) + sum(X)'
    expected.index.name = 'Y'
    testing.assert_series_equal(output, expected)

  def test_sub(self):
    sumx = metrics.Sum('X')
    sumy = metrics.Sum('Y')
    metric = sumx - sumy
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, 2)

  def test_mul(self):
    metric = 2. * metrics.Sum('X') * metrics.Sum('Y')
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, 48)
    self.assertEqual(metric.name, '2.0 * sum(X) * sum(Y)')

  def test_div(self):
    metric = 6. / metrics.Sum('X') / metrics.Sum('Y')
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, 0.25)
    self.assertEqual(metric.name, '6.0 / sum(X) / sum(Y)')

  def test_neg(self):
    base = metrics.MetricList((metrics.Sum('X'), metrics.Sum('Y')))
    metric = -base
    output = metric.compute_on(self.df)
    expected = -base.compute_on(self.df)
    expected.columns = ['-sum(X)', '-sum(Y)']
    testing.assert_frame_equal(output, expected)

  def test_pow(self):
    metric = metrics.Sum('X')**metrics.Sum('Y')
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, 1296)
    self.assertEqual(metric.name, 'sum(X) ^ sum(Y)')

  def test_pow_with_scalar(self):
    metric = metrics.Sum('X')**2
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, 36)
    self.assertEqual(metric.name, 'sum(X) ^ 2')

  def test_sqrt(self):
    metric = metrics.Sum('Y')**0.5
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, 2)
    self.assertEqual(metric.name, 'sqrt(sum(Y))')

  def test_rpow(self):
    metric = 2**metrics.Sum('X')
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, 64)
    self.assertEqual(metric.name, '2 ^ sum(X)')

  def test_ratio(self):
    metric = metrics.Ratio('X', 'Y')
    output = metric.compute_on(self.df)
    expected = metrics.Sum('X') / metrics.Sum('Y')
    expected = expected.compute_on(self.df)
    testing.assert_frame_equal(output, expected)

  def test_to_dataframe(self):
    metric = 5 + metrics.Sum('X')
    output = metric.compute_on(self.df)
    expected = pd.DataFrame({'5 + sum(X)': [11]})
    testing.assert_frame_equal(output, expected)

  def test_where(self):
    metric = metrics.Count('X', 'f', 'Y == 1') * metrics.Sum('X', 'b', 'Y == 2')
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, 6)

  def test_between_operations(self):
    df = pd.DataFrame({
        'X': [1, 2, 3, 4, 5, 6],
        'Condition': [0, 0, 0, 1, 1, 1],
        'grp': ['A', 'A', 'B', 'A', 'B', 'C']
    })
    suma = metrics.Sum('X', where='grp == "A"')
    sumb = metrics.Sum('X', where='grp == "B"')
    pct = operations.PercentChange('Condition', 0)
    output = (pct(suma) - pct(sumb)).compute_on(df)
    expected = pct(suma).compute_on(df) - pct(sumb).compute_on(df)
    expected.columns = ['%s - %s' % (c, c) for c in expected.columns]
    testing.assert_frame_equal(output, expected)

  def test_between_operations_where(self):
    df = pd.DataFrame({
        'X': [1, 2, 3, 4, 5, 6],
        'Condition': [0, 0, 0, 1, 1, 1],
        'grp': ['A', 'A', 'B', 'A', 'B', 'C']
    })
    sumx = metrics.Sum('X')
    pcta = operations.PercentChange('Condition', 0, sumx, where='grp == "A"')
    pctb = operations.PercentChange('Condition', 0, sumx, where='grp == "B"')
    output = (pcta - pctb).compute_on(df)
    expected = pcta.compute_on(df) - pctb.compute_on(df)
    expected.columns = ['%s - %s' % (c, c) for c in expected.columns]
    testing.assert_frame_equal(output, expected)

  def test_between_stderr_operations_where(self):
    df = pd.DataFrame({
        'X': [1, 2, 3, 4, 5, 6] * 2,
        'Condition': [0, 0, 0, 1, 1, 1] * 2,
        'grp': ['A', 'A', 'B', 'A', 'B', 'C'] * 2,
        'cookie': [1, 2, 3] * 4
    })
    np.random.seed(42)
    sumx = metrics.Sum('X')
    pcta = operations.PercentChange('Condition', 0, sumx, where='grp == "A"')
    pctb = operations.PercentChange('Condition', 0, sumx)
    jk = operations.Jackknife('cookie', pcta)
    bst = operations.Bootstrap(None, pctb, 20, where='grp != "C"')
    m = (jk / bst).rename_columns(
        pd.MultiIndex.from_product((('sum(X)',), ('Value', 'SE'))))
    output = m.compute_on(df)

    np.random.seed(42)
    expected = jk.compute_on(df).values / bst.compute_on(df).values
    expected = pd.DataFrame(
        expected, index=output.index, columns=output.columns)
    testing.assert_frame_equal(output, expected)

  def test_rename_columns(self):
    df = pd.DataFrame({'X': [1, 2], 'Y': [3, 1]})
    unweightd_metric = metrics.MetricList(
        (metrics.Mean('X'), metrics.StandardDeviation('X')))
    weightd_metric = metrics.MetricList(
        (metrics.Mean('X', 'Y'), metrics.StandardDeviation('X', weight='Y')))
    columns = ['mean', 'sum']
    metric = (unweightd_metric / weightd_metric).rename_columns(columns)
    output = metric.compute_on(df)
    unweightd = unweightd_metric.compute_on(df)
    weightd = weightd_metric.compute_on(df)
    expected = pd.DataFrame(unweightd.values / weightd.values, columns=columns)
    testing.assert_frame_equal(output, expected)

  def test_set_name(self):
    df = pd.DataFrame({'click': [1, 2], 'impression': [3, 1]})
    metric = (metrics.Sum('click') / metrics.Sum('impression')).set_name('ctr')
    output = metric.compute_on(df)
    expected = pd.DataFrame([[0.75]], columns=['ctr'])
    testing.assert_frame_equal(output, expected)

  def test_operation_set_name(self):
    df = pd.DataFrame({'click': [1, 2], 'grp': [0, 1]})
    metric = operations.AbsoluteChange('grp', 0, metrics.Sum('click'))
    metric = (1 + metric).set_name('foo')
    output = metric.compute_on(df)
    expected = pd.DataFrame([[2]],
                            columns=['foo'],
                            index=pd.Index([1], name='grp'))
    testing.assert_frame_equal(output, expected)

  def test_columns_multiindex(self):
    df = pd.DataFrame({'X': [1, 2], 'Y': [3, 1]})
    unweightd_metric = metrics.MetricList(
        (metrics.Mean('X'), metrics.StandardDeviation('X')))
    weightd_metric = metrics.MetricList(
        (metrics.Mean('X', 'Y'), metrics.StandardDeviation('X', weight='Y')))
    columns = pd.MultiIndex.from_product((['foo'], ['mean', 'sum']))
    metric = (unweightd_metric / weightd_metric).rename_columns(columns)
    output = metric.compute_on(df)
    unweightd = unweightd_metric.compute_on(df)
    weightd = weightd_metric.compute_on(df)
    expected = pd.DataFrame(unweightd.values / weightd.values, columns=columns)
    testing.assert_frame_equal(output, expected)


class TestMetricList(absltest.TestCase):

  def test_return_list(self):
    df = pd.DataFrame({'X': [0, 1, 2, 3]})
    ms = [metrics.Sum('X'), metrics.Mean('X')]
    m = metrics.MetricList(ms)
    output = m.compute_on(df, return_dataframe=False)
    self.assertLen(output, 2)
    testing.assert_frame_equal(output[0], ms[0].compute_on(df))
    testing.assert_frame_equal(output[1], ms[1].compute_on(df))

  def test_children_return_dataframe(self):
    df = pd.DataFrame({'X': [0, 1, 2, 3]})
    ms = [metrics.Sum('X'), metrics.Mean('X')]
    m = metrics.MetricList(ms, children_return_dataframe=False)
    output = m.compute_on(df, return_dataframe=False)
    self.assertEqual(output, [6, 1.5])

  def test_return_df(self):
    df = pd.DataFrame({'X': [0, 1, 2, 3]})
    ms = [metrics.Sum('X'), metrics.Mean('X')]
    m = metrics.MetricList(ms)
    output = m.compute_on(df)
    expected = pd.DataFrame(
        data={
            'sum(X)': [6],
            'mean(X)': [1.5]
        }, columns=['sum(X)', 'mean(X)'])
    testing.assert_frame_equal(output, expected)

  def test_with_name_tmpl(self):
    df = pd.DataFrame({'X': [0, 1, 2, 3]})
    ms = [metrics.Sum('X'), metrics.Mean('X')]
    m = metrics.MetricList(ms, name_tmpl='a{}b')
    output = m.compute_on(df)
    expected = pd.DataFrame(
        data={
            'asum(X)b': [6],
            'amean(X)b': [1.5]
        },
        columns=['asum(X)b', 'amean(X)b'])
    testing.assert_frame_equal(output, expected)

  def test_operations(self):
    df = pd.DataFrame({'X': [1, 1, 1], 'Y': ['a', 'a', 'b']})
    sumx = metrics.Sum('X')
    pct = operations.PercentChange('Y', 'b', sumx, include_base=False)
    absl = operations.AbsoluteChange('Y', 'b', sumx, include_base=True)
    metric = metrics.MetricList((pct, absl))
    output = metric.compute_on(df)
    expected = pd.DataFrame(
        {
            'sum(X) Percent Change': [100., np.nan],
            'sum(X) Absolute Change': [1, 0]
        },
        index=['a', 'b'],
        columns=['sum(X) Percent Change', 'sum(X) Absolute Change'])
    expected.index.name = 'Y'
    testing.assert_frame_equal(output, expected)

  def test_len(self):
    ms = [metrics.Sum('X'), metrics.Mean('X')]
    m = metrics.MetricList(ms)
    self.assertLen(m, 2)

  def test_split_by_return_df(self):
    df = pd.DataFrame({'X': [0, 1, 2, 3], 'grp': ['A', 'A', 'B', 'B']})
    ms = [metrics.Sum('X'), metrics.Mean('X')]
    m = metrics.MetricList(ms)
    output = m.compute_on(df, 'grp')
    expected = pd.DataFrame(
        data={
            'sum(X)': [1, 5],
            'mean(X)': [0.5, 2.5]
        },
        columns=['sum(X)', 'mean(X)'],
        index=['A', 'B'])
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_rename_columns(self):
    df = pd.DataFrame({'X': [0, 1, 2, 3]})
    ms = [metrics.Sum('X'), metrics.Mean('X')]
    m = metrics.MetricList(ms)
    m.rename_columns(['A', 'B'])
    output = m.compute_on(df)
    expected = pd.DataFrame({'A': [6], 'B': [1.5]})
    testing.assert_frame_equal(output, expected)

  def test_rename_columns_melted(self):
    df = pd.DataFrame({'X': [0, 1, 2, 3]})
    ms = [metrics.Sum('X'), metrics.Mean('X')]
    m = metrics.MetricList(ms)
    m.rename_columns(['A', 'B'])
    output = m.compute_on(df, melted=True)
    expected = pd.DataFrame({'Value': [6, 1.5]},
                            index=pd.Index(['A', 'B'], name='Metric'))
    testing.assert_frame_equal(output, expected)

  def test_rename_columns_incorrect_length(self):
    df = pd.DataFrame({'X': [0, 1, 2, 3]})
    ms = metrics.MetricList([metrics.Sum('X'), metrics.Mean('X')])
    ms.rename_columns(['A'])
    with self.assertRaises(ValueError):
      ms.compute_on(df)

  def test_interaction_with_compositemetric(self):
    df = pd.DataFrame({'X': [0, 1, 2, 3]})
    sum_x = metrics.Sum('X')
    double_sum_x = 2 * metrics.Sum('X')
    comp = double_sum_x / sum_x
    comp.name = 'foo'
    m = sum_x / metrics.MetricList((sum_x, comp))
    output = m.compute_on(df)
    expected = pd.DataFrame(
        data={
            'sum(X) / sum(X)': [1.],
            'sum(X) / foo': [3.]
        },
        columns=['sum(X) / sum(X)', 'sum(X) / foo'])
    testing.assert_frame_equal(output, expected)

  def test_to_dot(self):
    m0 = 1
    m1 = metrics.Sum('x', where='foo')
    m2 = m0 + m1
    m3 = metrics.Sum('y')
    m4 = metrics.Count('z', where='bar')
    m5 = m3 / m4
    m6 = metrics.MetricList((m2, m5))
    m6.name = 'baz'
    output = m6.to_dot()
    s = """
{id6} [label=baz];
{id2} [label="1 + sum(x)"];
{id5} [label="sum(y) / count(z)"];
{id0} [label=1];
{id1} [label="sum(x) where foo"];
{id3} [label="sum(y)"];
{id4} [label="count(z) where bar"];
{id6} -- {id2};
{id2} -- {id0};
{id2} -- {id1};
{id6} -- {id5};
{id5} -- {id3};
{id5} -- {id4};
""".format(
        id0=id(m0),
        id1=id(m1),
        id2=id(m2),
        id3=id(m3),
        id4=id(m4),
        id5=id(m5),
        id6=id(m6))
    expected = 'strict graph baz {%s}\n' % s
    self.assertEqual(output, expected)


class TestCaching(parameterized.TestCase):

  df = pd.DataFrame({'X': [0, 1], 'Y': [0, 1], 'grp': ['A', 'B']})

  def test_simple_metric_caching(self):
    m1 = metrics.Sum('X')
    m2 = metrics.Sum('X', name='s') + metrics.Sum('X')
    m3 = metrics.Ratio('X', 'X')
    m = metrics.MetricList((m1, m2, m3))
    with mock.patch.object(
        metrics.Sum, 'compute_through', autospec=True) as mock_fn:
      m.compute_on(self.df)
    mock_fn.assert_called_once()
    self.assertIsNone(m1.cache_key)
    self.assertIsNone(m2.cache_key)
    self.assertIsNone(m3.cache_key)
    self.assertIsNone(m.cache_key)
    self.assertEmpty(m.cache)

  def test_cache_in_multiple_runs(self):
    m = metrics.Count('X')
    with mock.patch.object(m, 'compute_through', autospec=True) as mock_fn:
      m.compute_on(self.df)
      m.compute_on(self.df)
      self.assertEqual(2, mock_fn.call_count)
    m = metrics.Count('X')
    with mock.patch.object(
        m, 'compute_through', return_value='a', autospec=True) as mock_fn:
      m.compute_on(self.df, cache_key='foo')
      output = m.compute_on(
          self.df, cache_key='foo', return_dataframe=False, cache=m.cache)
      mock_fn.assert_called_once_with(self.df, [])
      self.assertEqual('a', output)

  def test_clear_cache_key_when_fail(self):
    sum_x = metrics.Sum('X')
    sum_fail = metrics.Sum('fail')
    try:
      metrics.MetricList((sum_x, sum_fail)).compute_on(self.df)
    except KeyError:
      pass
    self.assertIsNone(sum_x.cache_key)
    self.assertEmpty(sum_x.cache)
    self.assertIsNone(sum_fail.cache_key)
    self.assertEmpty(sum_fail.cache)

  def test_simple_metric_caching_split_by(self):
    m = metrics.Count('X')
    with mock.patch.object(m, 'compute_through', autospec=True) as mock_fn:
      m.compute_on(self.df, 'grp')
      m.compute_on(self.df, ['grp'])

    self.assertEqual(2, mock_fn.call_count)

    m = metrics.Count('X')
    with mock.patch.object(
        m, 'compute_through', return_value='a', autospec=True) as mock_fn:
      m.compute_on(self.df, 'grp', cache_key='a')
      output = m.compute_on(
          self.df, ['grp'], return_dataframe=False, cache_key='a', cache=m.cache
      )

    mock_fn.assert_called_once_with(self.df, ['grp'])
    self.assertEqual('a', output)

  equivalent_metrics = [
      ('Ratio', metrics.Ratio('X', 'Y'), metrics.Sum('X') / metrics.Sum('Y')),
      ('Mean', metrics.Mean('X'), metrics.Sum('X') / metrics.Count('X')),
      ('Dot', metrics.Dot('X', 'Y'), metrics.Dot('Y', 'X')),
      ('Correlation', metrics.Correlation('X',
                                          'Y'), metrics.Correlation('Y', 'X')),
      ('Cov', metrics.Cov('X', 'Y'), metrics.Cov('Y', 'X'))
  ]

  @parameterized.named_parameters(*equivalent_metrics)
  def test_equivalent_metrics(self, m1, m2):
    expected = m1.compute_on(self.df, cache_key='foo')
    expected.columns = [m2.name]
    output = m2.compute_on(None, cache_key='foo', cache=m1.cache)
    testing.assert_frame_equal(output, expected)

  def test_filter_has_effect(self):
    df = pd.DataFrame({'X': [1, 2]})
    output = metrics.MetricList(
        (metrics.Count('X'), metrics.Count('X', where='X > 1'))).compute_on(df)
    expected = pd.DataFrame([[2, 1]], columns=['count(X)'] * 2)
    testing.assert_frame_equal(output, expected)

  def test_different_var_has_effect(self):
    df = pd.DataFrame({'X': [2], 'Y': [1]})
    output = metrics.MetricList(
        (metrics.Sum('X'), metrics.Sum('Y'))).compute_on(df)
    expected = pd.DataFrame([[2, 1]], columns=['sum(X)', 'sum(Y)'])
    testing.assert_frame_equal(output, expected)

  def test_different_metrics_have_different_fingerprints(self):
    distinct_metrics = [
        metrics.Ratio('x', 'y'),
        metrics.Ratio('x', 'x'),
        metrics.Ratio('y', 'y'),
        metrics.MetricList([metrics.Sum('x')]),
        metrics.MetricList([metrics.Sum('y')]),
        metrics.Sum('x') + metrics.Sum('y'),
        metrics.Sum('x') + metrics.Sum('x'),
        metrics.Count('x'),
        metrics.Count('x', distinct=True),
        metrics.Sum('x'),
        metrics.Dot('x', 'y'),
        metrics.Mean('x'),
        metrics.Mean('x', 'y'),
        metrics.Max('x'),
        metrics.Min('x'),
        metrics.Quantile('x'),
        metrics.Quantile('x', 0.2),
        metrics.Variance('x', True),
        metrics.Variance('x', False),
        metrics.Variance('x', weight='y'),
        metrics.StandardDeviation('x', True),
        metrics.StandardDeviation('x', False),
        metrics.StandardDeviation('x', weight='y'),
        metrics.CV('x', True),
        metrics.CV('x', False),
        metrics.Correlation('x', 'y'),
        metrics.Correlation('x', 'y', 'w'),
        metrics.Correlation('x', 'y', method='kendall'),
        metrics.Cov('x', 'y', True),
        metrics.Cov('x', 'y', False),
        metrics.Cov('x', 'y', True, 3),
        metrics.Cov('x', 'y', True, weight='w'),
        metrics.Cov('x', 'y', True, fweight='w'),
        metrics.Cov('x', 'y', True, weight='w', fweight='w'),
    ]
    fingerprints = set([m.get_fingerprint() for m in distinct_metrics])
    self.assertLen(fingerprints, len(distinct_metrics))

  def test_class_level_caching_disabled_for_custom_metric_by_default(self):
    class Custom(metrics.Metric):
      def __init__(self, x):
        self.x = x
        super(Custom, self).__init__('Foo')

      def compute(self, data):
        return self.x

    output = metrics.MetricList(
        [metrics.MetricList([Custom(1)]), metrics.MetricList([Custom(2)])]
    ).compute_on(self.df)
    expected = pd.DataFrame([[1, 2]], columns=['Foo'] * 2)
    testing.assert_frame_equal(output, expected)

  def test_class_level_caching_enabled_for_custom_metric(self):
    class WrongImpl(metrics.Metric):
      def __init__(self, x):
        self.x = x
        super(WrongImpl, self).__init__('Foo')
        self.cache_across_instances = True

      def compute(self, data):
        return self.x

    output = metrics.MetricList([WrongImpl(1), WrongImpl(2)]).compute_on(
        self.df
    )
    expected = pd.DataFrame([[1, 1]], columns=['Foo'] * 2)
    testing.assert_frame_equal(output, expected)


if __name__ == '__main__':
  absltest.main()
