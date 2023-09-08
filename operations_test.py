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
"""Tests for Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from absl.testing import absltest
from absl.testing import parameterized
from meterstick import diversity
from meterstick import metrics
from meterstick import models
from meterstick import operations
from meterstick import utils
import mock
import numpy as np
import pandas as pd
from pandas import testing
from scipy import stats
from sklearn import linear_model


def spy_decorator(method_to_decorate):
  # Adapted from https://stackoverflow.com/a/41599695.
  m = mock.MagicMock()

  def wrapper(self, *args, **kwargs):
    m(*args, **kwargs)
    return method_to_decorate(self, *args, **kwargs)

  wrapper.mock = m
  return wrapper


class SimpleOperationTests(absltest.TestCase):
  df = pd.DataFrame({
      'x': [1, 1, 1, 5],
      'grp': ['B', 'B', 'A', 'A'],
      'grp2': [0, 1, 0, 1],
  })

  def test_distribution(self):
    output = operations.Distribution('grp', metrics.Sum('x')).compute_on(
        self.df
    )
    expected = pd.DataFrame(
        {'Distribution of sum(x)': [0.75, 0.25]}, index=['A', 'B']
    )
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_normalize(self):
    self.assertEqual(operations.Distribution, operations.Normalize)

  def test_cumulative_distribution(self):
    output = operations.CumulativeDistribution(
        'grp', metrics.Sum('x')
    ).compute_on(self.df)
    expected = pd.DataFrame(
        {'Cumulative Distribution of sum(x)': [0.75, 1.0]}, index=['A', 'B']
    )
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_cumulative_distribution_order(self):
    metric = operations.CumulativeDistribution(
        'grp', metrics.Sum('x'), ('B', 'A')
    )
    output = metric.compute_on(self.df)
    expected = pd.DataFrame(
        {'Cumulative Distribution of sum(x)': [0.25, 1.0]}, index=['B', 'A']
    )
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_cumulative_distribution_ascending(self):
    metric = operations.CumulativeDistribution(
        'grp', metrics.Sum('x'), ascending=False
    )
    output = metric.compute_on(self.df)
    expected = pd.DataFrame(
        {'Cumulative Distribution of sum(x)': [0.25, 1.0]}, index=['B', 'A']
    )
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_cumulative_distribution_order_descending(self):
    metric = operations.CumulativeDistribution(
        'grp', metrics.Sum('x'), ('B', 'A'), False
    )
    output = metric.compute_on(self.df)
    expected = operations.CumulativeDistribution(
        'grp', metrics.Sum('x'), ('A', 'B')
    ).compute_on(self.df)
    testing.assert_frame_equal(output, expected)

  def test_percent_change(self):
    metric = operations.PercentChange('grp', 'B', metrics.Sum('x'))
    output = metric.compute_on(self.df)
    expected = pd.DataFrame(
        [[200.0]],
        columns=['sum(x) Percent Change'],
        index=['A'],
    )
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_percent_change_include_baseline(self):
    metric = operations.PercentChange('grp', 'B', metrics.Sum('x'), True)
    output = metric.compute_on(self.df)
    expected = pd.DataFrame(
        {'sum(x) Percent Change': [200.0, 0]},
        index=['A', 'B'],
    )
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_absolute_change(self):
    metric = operations.AbsoluteChange('grp', 'B', metrics.Sum('x'))
    output = metric.compute_on(self.df)
    expected = pd.DataFrame(
        [[4]],
        columns=['sum(x) Absolute Change'],
        index=['A'],
    )
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_absolute_change_include_baseline(self):
    metric = operations.AbsoluteChange('grp', 'B', metrics.Sum('x'), True)
    output = metric.compute_on(self.df)
    expected = pd.DataFrame(
        {'sum(x) Absolute Change': [4, 0]},
        index=['A', 'B'],
    )
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_difference_in_differences(self):
    pct1 = operations.PercentChange(
        'grp', 'A', metrics.Sum('x'), where='grp2 == 0'
    )
    pct2 = operations.PercentChange(
        'grp', 'A', metrics.Sum('x'), where='grp2 == 1'
    )
    m = pct1 - pct2
    output = m.compute_on(self.df)
    expected = pct1.compute_on(self.df) - pct2.compute_on(self.df)
    expected.columns = output.columns
    testing.assert_frame_equal(output, expected)

  def test_chained_operation(self):
    m = (
        metrics.Sum('x')
        | operations.PercentChange('grp', 'A')
        | operations.AbsoluteChange('grp2', 0)
    )
    output = m.compute_on(self.df)
    expected = pd.DataFrame({
        'sum(x) Percent Change Absolute Change': [-80.0],
        'grp': ['B'],
        'grp2': [1],
    })
    expected.set_index(['grp2', 'grp'], inplace=True)
    testing.assert_frame_equal(output, expected)

  def test_hhi(self):
    output = diversity.HHI('grp', metrics.Sum('x')).compute_on(self.df)
    expected = pd.DataFrame({
        'HHI of sum(x)': [0.75 ** 2 + 0.25 ** 2],
    })
    testing.assert_frame_equal(output, expected)

  def test_entropy(self):
    output = diversity.Entropy('grp', metrics.Sum('x')).compute_on(self.df)
    expected = pd.DataFrame({
        'Entropy of sum(x)': [-(0.75 * np.log(0.75) + 0.25 * np.log(0.25))],
    })
    testing.assert_frame_equal(output, expected)

  def test_topk(self):
    s = metrics.Sum('x')
    output = metrics.MetricList(
        (diversity.TopK('grp', 1, s), diversity.TopK('grp', 2, s))
    ).compute_on(self.df)
    expected = pd.DataFrame(
        [[0.75, 1.0]],
        columns=["Top-1's share of sum(x)", "Top-2's share of sum(x)"],
    )
    testing.assert_frame_equal(output, expected)

  def test_nxx(self):
    s = metrics.Sum('x')
    output = metrics.MetricList(
        (diversity.Nxx('grp', 0.75, s), diversity.Nxx('grp', 0.751, s))
    ).compute_on(self.df)
    expected = pd.DataFrame(
        [[1, 2]], columns=['N(75) of sum(x)', 'N(75.1) of sum(x)']
    )
    testing.assert_frame_equal(output, expected)


class PrePostChangeTests(absltest.TestCase):
  n = 40
  df = pd.DataFrame({
      'x': np.random.choice(range(20), n),
      'y': np.random.choice(range(20), n),
      'pre_x': np.random.choice(range(20), n),
      'cookie': np.random.choice(range(4), n),
      'condition': np.random.choice(range(2), n),
      'grp': np.random.choice(('A', 'B', 'C'), n),
  })
  sum_x = metrics.Sum('x')
  sum_prex = metrics.Sum('pre_x')
  df_agg = (
      df.groupby(['cookie', 'condition']).sum(numeric_only=True).reset_index()
  )
  df_agg.pre_x = df_agg.pre_x - df_agg.pre_x.mean()
  df_agg['interaction'] = df_agg.pre_x * df_agg.condition
  x = df_agg[['condition', 'pre_x', 'interaction']]
  y = df_agg['x']

  def test_basic(self):
    metric = operations.PrePostChange(
        'condition', 0, self.sum_x, self.sum_prex, 'cookie'
    )
    output = metric.compute_on(self.df)
    lm = linear_model.LinearRegression()
    lm.fit(self.x, self.y)
    expected = pd.DataFrame(
        [[100 * lm.coef_[0] / lm.intercept_]],
        columns=['sum(x) PrePost Percent Change'],
        index=[1],
    )
    expected.index.name = 'condition'
    testing.assert_frame_equal(output, expected)

  def test_include_base(self):
    metric = operations.PrePostChange(
        'condition', 0, self.sum_x, self.sum_prex, 'cookie', True
    )
    output = metric.compute_on(self.df)
    lm = linear_model.LinearRegression()
    lm.fit(self.x, self.y)
    expected = pd.DataFrame(
        [0, 100 * lm.coef_[0] / lm.intercept_],
        columns=['sum(x) PrePost Percent Change'],
        index=[0, 1],
    )
    expected.index.name = 'condition'
    testing.assert_frame_equal(output, expected)

  def test_multiple_conditions(self):
    metric = operations.PrePostChange(
        ['condition', 'grp'], (0, 'C'), self.sum_x, self.sum_prex, 'cookie'
    )
    output = metric.compute_on(self.df)
    df = self.df.copy()
    df['condition_and_grp'] = df[['condition', 'grp']].apply(tuple, 1)
    expected_metric = operations.PrePostChange(
        'condition_and_grp', (0, 'C'), self.sum_x, self.sum_prex, 'cookie'
    )
    expected = expected_metric.compute_on(df)
    expected = pd.DataFrame(
        expected.values, index=output.index, columns=output.columns
    )
    testing.assert_frame_equal(output, expected)

  def test_multiple_stratified_by(self):
    metric = operations.PrePostChange(
        'condition', 0, self.sum_x, self.sum_prex, ['cookie', 'grp']
    )
    output = metric.compute_on(self.df)
    df_agg = (
        self.df.groupby(['cookie', 'grp', 'condition'])
        .sum(numeric_only=True)
        .reset_index()
    )
    df_agg.pre_x = df_agg.pre_x - df_agg.pre_x.mean()
    df_agg['interaction'] = df_agg.pre_x * df_agg.condition
    lm = linear_model.LinearRegression()
    lm.fit(df_agg[['condition', 'pre_x', 'interaction']], df_agg['x'])
    expected = pd.DataFrame(
        [100 * lm.coef_[0] / lm.intercept_],
        columns=['sum(x) PrePost Percent Change'],
        index=[1],
    )
    expected.index.name = 'condition'
    testing.assert_frame_equal(output, expected)

  def test_multiple_metrics(self):
    metric = operations.PrePostChange(
        'condition',
        0,
        metrics.MetricList([self.sum_x, metrics.Sum('y')]),
        self.sum_prex,
        'cookie',
    )
    output = metric.compute_on(self.df)
    expected1 = operations.PrePostChange(
        'condition', 0, self.sum_x, self.sum_prex, 'cookie'
    ).compute_on(self.df)
    expected2 = operations.PrePostChange(
        'condition', 0, metrics.Sum('y'), self.sum_prex, 'cookie'
    ).compute_on(self.df)
    expected = pd.concat((expected1, expected2), axis=1)
    testing.assert_frame_equal(output, expected)

  def test_multiple_covariates(self):
    metric = operations.PrePostChange(
        'condition', 0, self.sum_x, [self.sum_prex, metrics.Sum('y')], 'cookie'
    )
    output = metric.compute_on(self.df)
    df = self.df.copy()
    df_agg = (
        df.groupby(['cookie', 'condition']).sum(numeric_only=True).reset_index()
    )
    df_agg.pre_x = df_agg.pre_x - df_agg.pre_x.mean()
    df_agg.y = df_agg.y - df_agg.y.mean()
    df_agg['interaction1'] = df_agg.pre_x * df_agg.condition
    df_agg['interaction2'] = df_agg.y * df_agg.condition
    lm = linear_model.LinearRegression()
    lm.fit(
        df_agg[['condition', 'pre_x', 'y', 'interaction1', 'interaction2']],
        df_agg['x'],
    )
    expected = pd.DataFrame(
        [100 * lm.coef_[0] / lm.intercept_],
        columns=['sum(x) PrePost Percent Change'],
        index=[1],
    )
    expected.index.name = 'condition'
    testing.assert_frame_equal(output, expected)

  def test_complex(self):
    n = 50
    df = pd.DataFrame({
        'x': np.random.choice(range(20), n),
        'y': np.random.choice(range(20), n),
        'pre_x': np.random.choice(range(20), n),
        'cookie': np.random.choice(range(5), n),
        'condition1': np.random.choice(range(2), n),
        'condition2': np.random.choice(('A', 'B', 'C'), n),
        'grp1': np.random.choice(('foo', 'bar', 'baz'), n),
        'grp2': np.random.choice(('US', 'non-US'), n),
        'grp3': np.random.choice(('desktop', 'mobile', 'tablet'), n),
    })
    post = [self.sum_x, self.sum_x**2]
    pre = [self.sum_prex, metrics.Sum('y')]
    metric = operations.PrePostChange(
        ['condition1', 'condition2'],
        (1, 'C'),
        metrics.MetricList(post),
        pre,
        ['cookie', 'grp1'],
    )
    output = metric.compute_on(df, ['grp2', 'grp3'])
    df['condition'] = df[['condition1', 'condition2']].apply(tuple, 1)
    df['agg'] = df[['cookie', 'grp1']].apply(tuple, 1)

    expected = [
        operations.PrePostChange(
            'condition', (1, 'C'), m, pre, 'agg'
        ).compute_on(df, ['grp2', 'grp3'])
        for m in post
    ]
    expected = pd.concat(expected, axis=1)
    expected = expected.reset_index('condition')
    expected[['condition1', 'condition2']] = expected.condition.to_list()
    expected = expected.set_index(['condition1', 'condition2'], append=True)
    expected = expected.drop(columns=['condition'])
    testing.assert_frame_equal(output, expected)


class CUPEDTests(absltest.TestCase):
  n = 40
  df = pd.DataFrame({
      'x': np.random.choice(range(20), n),
      'y': np.random.choice(range(20), n),
      'pre_x': np.random.choice(range(20), n),
      'cookie': np.random.choice(range(4), n),
      'condition': np.random.choice(range(2), n),
      'grp': np.random.choice(('A', 'B', 'C'), n),
  })
  sum_x = metrics.Sum('x')
  sum_prex = metrics.Sum('pre_x')
  df_agg = (
      df.groupby(['cookie', 'condition']).sum(numeric_only=True).reset_index()
  )
  df_agg.pre_x = df_agg.pre_x - df_agg.pre_x.mean()
  df_agg.y = df_agg.y - df_agg.y.mean()

  def test_basic(self):
    metric = operations.CUPED(
        'condition', 0, self.sum_x, self.sum_prex, 'cookie'
    )
    output = metric.compute_on(self.df)
    theta = (
        self.df_agg[['x', 'pre_x']].cov().iloc[0, 1] / self.df_agg.pre_x.var()
    )
    adjusted = (
        self.df_agg.groupby('condition').x.mean()
        - theta * self.df_agg.groupby('condition').pre_x.mean()
    )
    expected = pd.DataFrame(
        [[adjusted[1] - adjusted[0]]],
        columns=['sum(x) CUPED Change'],
        index=[1],
    )
    expected.index.name = 'condition'
    testing.assert_frame_equal(output, expected)

  def test_include_base(self):
    metric = operations.CUPED(
        'condition', 0, self.sum_x, self.sum_prex, 'cookie', True
    )
    output = metric.compute_on(self.df)
    theta = (
        self.df_agg[['x', 'pre_x']].cov().iloc[0, 1] / self.df_agg.pre_x.var()
    )
    adjusted = (
        self.df_agg.groupby('condition').x.mean()
        - theta * self.df_agg.groupby('condition').pre_x.mean()
    )
    expected = pd.DataFrame(
        [0, adjusted[1] - adjusted[0]],
        columns=['sum(x) CUPED Change'],
        index=[0, 1],
    )
    expected.index.name = 'condition'
    testing.assert_frame_equal(output, expected)

  def test_multiple_conditions(self):
    metric = operations.CUPED(
        ['condition', 'grp'], (0, 'C'), self.sum_x, self.sum_prex, 'cookie'
    )
    output = metric.compute_on(self.df)
    df = self.df.copy()
    df['condition_and_grp'] = df[['condition', 'grp']].apply(tuple, 1)
    expected_metric = operations.CUPED(
        'condition_and_grp', (0, 'C'), self.sum_x, self.sum_prex, 'cookie'
    )
    expected = expected_metric.compute_on(df)
    expected = pd.DataFrame(
        expected.values, index=output.index, columns=output.columns
    )
    testing.assert_frame_equal(output, expected)

  def test_multiple_stratified_by(self):
    metric = operations.CUPED(
        'condition', 0, self.sum_x, self.sum_prex, ['cookie', 'grp']
    )
    output = metric.compute_on(self.df)
    df_agg = (
        self.df.groupby(['cookie', 'condition', 'grp'])
        .sum(numeric_only=True)
        .reset_index()
    )
    df_agg.pre_x = df_agg.pre_x - df_agg.pre_x.mean()
    df_agg.y = df_agg.y - df_agg.y.mean()
    theta = df_agg[['x', 'pre_x']].cov().iloc[0, 1] / df_agg.pre_x.var()
    adjusted = (
        df_agg.groupby('condition').x.mean()
        - theta * df_agg.groupby('condition').pre_x.mean()
    )
    expected = pd.DataFrame(
        [[adjusted[1] - adjusted[0]]],
        columns=['sum(x) CUPED Change'],
        index=[1],
    )
    expected.index.name = 'condition'
    testing.assert_frame_equal(output, expected)

  def test_multiple_metrics(self):
    metric = operations.CUPED(
        'condition',
        0,
        metrics.MetricList([self.sum_x, metrics.Sum('y')]),
        self.sum_prex,
        'cookie',
    )
    output = metric.compute_on(self.df)
    expected1 = operations.CUPED(
        'condition', 0, self.sum_x, self.sum_prex, 'cookie'
    ).compute_on(self.df)
    expected2 = operations.CUPED(
        'condition', 0, metrics.Sum('y'), self.sum_prex, 'cookie'
    ).compute_on(self.df)
    expected = pd.concat((expected1, expected2), axis=1)
    testing.assert_frame_equal(output, expected)

  def test_multiple_covariates(self):
    metric = operations.CUPED(
        'condition', 0, self.sum_x, [self.sum_prex, metrics.Sum('y')], 'cookie'
    )
    output = metric.compute_on(self.df)
    lm = linear_model.LinearRegression()
    lm.fit(self.df_agg[['pre_x', 'y']], self.df_agg['x'])
    theta = lm.coef_
    adjusted = (
        self.df_agg.groupby('condition').x.mean()
        - theta[0] * self.df_agg.groupby('condition').pre_x.mean()
        - theta[1] * self.df_agg.groupby('condition').y.mean()
    )
    expected = pd.DataFrame(
        adjusted[1] - adjusted[0], columns=['sum(x) CUPED Change'], index=[1]
    )
    expected.index.name = 'condition'
    testing.assert_frame_equal(output, expected)

  def test_complex(self):
    n = 50
    df = pd.DataFrame({
        'x': np.random.choice(range(20), n),
        'y': np.random.choice(range(20), n),
        'pre_x': np.random.choice(range(20), n),
        'cookie': np.random.choice(range(5), n),
        'condition1': np.random.choice(range(2), n),
        'condition2': np.random.choice(('A', 'B', 'C'), n),
        'grp1': np.random.choice(('foo', 'bar', 'baz'), n),
        'grp2': np.random.choice(('US', 'non-US'), n),
        'grp3': np.random.choice(('desktop', 'mobile', 'tablet'), n),
    })
    post = [self.sum_x, self.sum_x**2]
    pre = [self.sum_prex, metrics.Sum('y')]
    metric = operations.CUPED(
        ['condition1', 'condition2'],
        (1, 'C'),
        metrics.MetricList(post),
        pre,
        ['cookie', 'grp1'],
    )
    output = metric.compute_on(df, ['grp2', 'grp3'])
    df['condition'] = df[['condition1', 'condition2']].apply(tuple, 1)
    df['agg'] = df[['cookie', 'grp1']].apply(tuple, 1)

    expected = [
        operations.CUPED('condition', (1, 'C'), m, pre, 'agg').compute_on(
            df, ['grp2', 'grp3']
        )
        for m in post
    ]
    expected = pd.concat(expected, axis=1)
    expected = expected.reset_index('condition')
    expected[['condition1', 'condition2']] = expected.condition.to_list()
    expected = expected.set_index(['condition1', 'condition2'], append=True)
    expected = expected.drop(columns=['condition'])
    testing.assert_frame_equal(output, expected)


class MHTests(absltest.TestCase):
  df = pd.DataFrame({
      'x': [1, 3, 2, 3, 1, 2],
      'y': [1, 0, 1, 2, 1, 1],
      'Id': [1, 2, 3, 1, 2, 3],
      'Condition': [0, 0, 0, 1, 1, 1],
  })
  sum_x = metrics.Sum('x')
  sum_conv = metrics.Sum('y')
  cvr = metrics.Ratio('y', 'x', 'cvr')
  metric_lst = metrics.MetricList((sum_conv / sum_x, cvr))

  def test_mh(self):
    metric = operations.MH('Condition', 0, 'Id', self.cvr)
    output = metric.compute_on(self.df)
    expected = pd.DataFrame([[40.0]], columns=['cvr MH Ratio'], index=[1])
    expected.index.name = 'Condition'
    testing.assert_frame_equal(output, expected)

  def test_mh_include_baseline(self):
    metric = operations.MH('Condition', 0, 'Id', self.metric_lst, True)
    output = metric.compute_on(self.df)
    expected = pd.DataFrame(
        [[0.0, 0.0], [40.0, 40.0]],
        columns=['sum(y) / sum(x) MH Ratio', 'cvr MH Ratio'],
        index=[0, 1],
    )
    expected.index.name = 'Condition'
    testing.assert_frame_equal(output, expected)

  def test_mh_stratified_by_multiple(self):
    df = pd.DataFrame({
        'x': [1, 3, 2, 3, 1, 2, 12, 31, 22, 30, 15, 23],
        'y': [1, 0, 1, 2, 1, 1, 3, 2, 4, 6, 7, 1],
        'Id': [1, 2, 3, 1, 2, 3] * 2,
        'platform': ['Desktop'] * 6 + ['Mobile'] * 6,
        'Condition': [0, 0, 0, 1, 1, 1] * 2,
    })
    df['id_platform'] = df[['Id', 'platform']].apply(tuple, axis=1)
    cvr = metrics.Ratio('y', 'x', 'cvr')

    metric = operations.MH('Condition', 0, ['Id', 'platform'], cvr)
    output = metric.compute_on(df)
    expected = operations.MH('Condition', 0, 'id_platform', cvr).compute_on(df)
    testing.assert_frame_equal(output, expected)

  def test_mh_on_operations(self):
    df = pd.DataFrame({
        'x': np.random.random(24),
        'y': np.random.random(24),
        'Id': [1, 2, 1, 2] * 6,
        'Condition': [0, 0, 0, 1, 1, 1] * 4,
        'grp': list('AABBCCBC') * 3,
    })
    sum_x = metrics.Sum('x')
    ab = operations.AbsoluteChange('grp', 'A', sum_x)
    pct = operations.PercentChange('grp', 'A', sum_x)
    metric = operations.MH('Condition', 0, 'Id', ab / pct)
    output = metric.compute_on(df)
    d = (
        (metrics.MetricList((ab, pct)))
        .compute_on(df, ['Condition', 'Id'])
        .reset_index()
    )
    m = metric(metrics.Sum(ab.name) / metrics.Sum(pct.name))
    expected = m.compute_on(d, 'grp')
    expected.columns = output.columns
    expected = expected.reorder_levels(output.index.names)
    testing.assert_frame_equal(output, expected)

  def test_mh_fail_on_nonratio_metric(self):
    with self.assertRaisesRegex(
        ValueError, 'MH only makes sense on ratio Metrics.'
    ):
      operations.MH('Condition', 0, 'Id', self.sum_x).compute_on(self.df)


SIMPLE_OPERATIONS = [
    ('Distribution', operations.Distribution('grp')),
    ('CumulativeDistribution', operations.CumulativeDistribution('grp')),
    ('PercentChange', operations.PercentChange('grp', 0)),
    ('AbsoluteChange', operations.AbsoluteChange('grp', 0)),
    ('HHI', diversity.HHI('grp')),
    ('Entropy', diversity.Entropy('grp')),
    ('TopK', diversity.TopK('grp', 1)),
    ('Nxx', diversity.Nxx('grp', 0.4)),
]
PRECOMPUTABLE_OPERATIONS = SIMPLE_OPERATIONS + [
    ('MH', operations.MH('grp', 0, 'cookie')),
    (
        'PrePostChange',
        operations.PrePostChange(
            'grp', 0, metrics.Sum('x'), metrics.Sum('y'), 'cookie'
        ),
    ),
    (
        'CUPED',
        operations.CUPED(
            'grp', 0, metrics.Sum('x'), metrics.Sum('y'), 'cookie'
        ),
    ),
    (
        'LinearRegression',
        models.LinearRegression(metrics.Sum('x'), metrics.Mean('y'), 'grp'),
    ),
    ('Ridge', models.Ridge(metrics.Sum('x'), metrics.Mean('y'), 'grp')),
    ('Lasso', models.Lasso(metrics.Sum('x'), metrics.Mean('y'), 'grp')),
    (
        'ElasticNet',
        models.ElasticNet(metrics.Sum('x'), metrics.Mean('y'), 'grp'),
    ),
]
PRECOMPUTABLE_METRICS_JK = [
    ('Sum', metrics.Sum('x')),
    ('Count', metrics.Count('x')),
    ('Mean', metrics.Mean('x')),
    ('Weighted Mean', metrics.Mean('x', 'y')),
    ('Dot', metrics.Dot('x', 'y')),
    ('Normalized Dot', metrics.Dot('x', 'y', True)),
    ('Variance', metrics.Variance('x', True)),
    ('Biased Variance', metrics.Variance('x', False)),
    ('Weighted Variance', metrics.Variance('x', True, 'y')),
    ('Biased Weighted Variance', metrics.Variance('x', False, 'y')),
    ('StandardDeviation', metrics.StandardDeviation('x', True)),
    ('Biased StandardDeviation', metrics.StandardDeviation('x', False)),
    ('Weighted StandardDeviation', metrics.StandardDeviation('x', True, 'y')),
    (
        'Biased Weighted StandardDeviation',
        metrics.StandardDeviation('x', False, 'y'),
    ),
    ('CV', metrics.CV('x', True)),
    ('Biased CV', metrics.CV('x', False)),
    ('Cov', metrics.Cov('x', 'y', False)),
    ('Biased Cov', metrics.Cov('x', 'y', True)),
    ('Cov ddof', metrics.Cov('x', 'y', False, 2)),
    ('Biased Cov ddof', metrics.Cov('x', 'y', True, 2)),
    ('Weighted Cov', metrics.Cov('x', 'y', False, weight='w')),
    ('Biased Weighted Cov', metrics.Cov('x', 'y', True, weight='w')),
    ('Weighted Cov ddof', metrics.Cov('x', 'y', False, 2, 'w')),
    ('Biased Weighted Cov ddof', metrics.Cov('x', 'y', True, 2, 'w')),
    ('Fweighted Cov', metrics.Cov('x', 'y', False, fweight='w2')),
    ('Biased Fweighted Cov', metrics.Cov('x', 'y', True, fweight='w2')),
    ('Fweighted Cov ddof', metrics.Cov('x', 'y', False, 2, fweight='w2')),
    ('Biased Fweighted Cov ddof', metrics.Cov('x', 'y', True, 2, fweight='w2')),
    (
        'Weighted and fweighted Cov',
        metrics.Cov('x', 'y', False, None, 'w', 'w2'),
    ),
    (
        'Biased Weighted and fweighted Cov',
        metrics.Cov('x', 'y', True, None, 'w', 'w2'),
    ),
    (
        'Weighted and fweighted Cov ddof',
        metrics.Cov('x', 'y', False, 2, 'w', 'w2'),
    ),
    (
        'Biased Weighted and fweighted Cov ddof',
        metrics.Cov('x', 'y', True, 2, 'w', 'w2'),
    ),
    ('Correlation', metrics.Correlation('x', 'y')),
    ('Weighted Correlation', metrics.Correlation('x', 'y', 'w')),
]
PRECOMPUTABLE_METRICS_BS = PRECOMPUTABLE_METRICS_JK + [
    ('Max', metrics.Max('x')),
    ('Min', metrics.Min('x')),
]


class JackknifeTests(parameterized.TestCase):
  count_x0 = metrics.Sum('x') / metrics.Mean('x')
  count_x1 = metrics.Count('x')
  dot1 = metrics.Dot('x', 'x')
  dot2 = metrics.Dot('x', 'x', True)
  metric = metrics.MetricList((count_x0, count_x1, dot1, dot2))
  change = operations.AbsoluteChange('grp', 'foo', metric)
  jk = operations.Jackknife('cookie', metric)
  jk_change = operations.Jackknife('cookie', change)

  def test_jackknife(self):
    df = pd.DataFrame({'x': np.arange(0, 3, 0.5), 'cookie': [1, 2, 2, 1, 2, 2]})
    unmelted = self.jk.compute_on(df)
    expected = pd.DataFrame(
        [
            [6.0, 1.0] * 2
            + [(df.x**2).sum(), 4.625, (df.x**2).mean(), 0.875]
        ],
        columns=pd.MultiIndex.from_product(
            [
                ['sum(x) / mean(x)', 'count(x)', 'sum(x * x)', 'mean(x * x)'],
                ['Value', 'Jackknife SE'],
            ],
            names=['Metric', None],
        ),
    )
    testing.assert_frame_equal(unmelted, expected)

    melted = self.jk.compute_on(df, melted=True)
    expected = utils.melt(expected)
    testing.assert_frame_equal(melted, expected)

  def test_jackknife_splitby_partial_overlap(self):
    df = pd.DataFrame({
        'x': range(1, 7),
        'cookie': [1, 2, 2, 1, 2, 3],
        'grp': ['B'] * 3 + ['A'] * 3,
    })
    unmelted = self.jk.compute_on(df, 'grp')
    expected = []
    for g in ['A', 'B']:
      expected.append(self.jk.compute_on(df[df.grp == g]))
    expected = pd.concat(expected, keys=['A', 'B'], names=['grp'])
    expected = expected.droplevel(-1)
    testing.assert_frame_equal(unmelted, expected)

    melted = self.jk.compute_on(df, 'grp', melted=True)
    testing.assert_frame_equal(melted, utils.melt(expected))

  def test_jackknife_unequal_slice_length_with_confidence(self):
    df = pd.DataFrame({
        'x': range(1, 7),
        'cookie': [1, 2, 2, 1, 2, 3],
        'grp': ['B'] * 3 + ['A'] * 3,
    })
    jk = operations.Jackknife('cookie', self.metric, 0.9)
    output = jk.compute_on(df, 'grp')
    expected = []
    for g in ['A', 'B']:
      expected.append(jk.compute_on(df[df.grp == g]))
    expected = pd.concat(expected, keys=['A', 'B'], names=['grp'])
    expected = expected.droplevel(-1)
    testing.assert_frame_equal(output, expected)

  def test_jackknife_splitby_partial_no_overlap(self):
    df = pd.DataFrame({
        'x': range(1, 7),
        'cookie': [1, 2, 2, 3, 3, 4],  # No slice has full levels.
        'grp': ['B'] * 3 + ['A'] * 3,
    })
    unmelted = self.jk.compute_on(df, 'grp')
    expected = []
    for g in ['A', 'B']:
      expected.append(self.jk.compute_on(df[df.grp == g]))
    expected = pd.concat(expected, keys=['A', 'B'], names=['grp'])
    expected = expected.droplevel(-1)
    testing.assert_frame_equal(unmelted, expected, check_dtype=False)

    melted = self.jk.compute_on(df, 'grp', melted=True)
    testing.assert_frame_equal(melted, utils.melt(expected), check_dtype=False)

  def test_jackknife_splitby_multiple(self):
    df = pd.DataFrame({
        'x': np.arange(0, 6, 0.5),
        'cookie': [1, 2, 2, 3, 3, 4, 1, 2, 2, 1, 2, 3],
        'grp': list('BBBAAA') * 2,
        'region': ['US'] * 6 + ['non-US'] * 6,
    })
    unmelted = self.jk.compute_on(df, ['grp', 'region'])
    expected = []
    for g in ['A', 'B']:
      expected.append(self.jk.compute_on(df[df.grp == g], 'region'))
    expected = pd.concat(expected, keys=['A', 'B'], names=['grp'])
    testing.assert_frame_equal(unmelted, expected)

    melted = self.jk.compute_on(df, ['grp', 'region'], melted=True)
    testing.assert_frame_equal(melted, utils.melt(expected))

  def test_jackknife_one_dof(self):
    df = pd.DataFrame({
        'x': range(2),
        'cookie': [0, 0],
    })
    jk = operations.Jackknife('cookie', metrics.Sum('x'))
    output = jk.compute_on(df)
    expected = pd.DataFrame(
        [[1.0, np.nan]],
        columns=pd.MultiIndex.from_product(
            [['sum(x)'], ['Value', 'Jackknife SE']], names=['Metric', None]
        ),
    )
    testing.assert_frame_equal(output, expected)

  @parameterized.named_parameters(*PRECOMPUTABLE_METRICS_JK)
  def test_optimization(self, m):
    opt = operations.Jackknife('cookie', m, 0.9)
    no_opt = operations.Jackknife('cookie', m, 0.9, enable_optimization=False)
    df = pd.DataFrame({
        'x': np.random.rand(10),
        'y': np.random.rand(10),
        'w': np.random.rand(10),
        'w2': np.random.randint(1, 10, size=10),
        'cookie': range(10),
    })
    original_df = df.copy()

    with mock.patch.object(
        m.__class__, 'compute_slices', spy_decorator(m.__class__.compute_slices)
    ) as mock_fn:
      output = opt.compute_on(df).display(return_formatted_df=True)
      if isinstance(m, (metrics.Sum, metrics.Count)):
        mock_fn.mock.assert_called_once()
      else:
        mock_fn.mock.assert_not_called()
      mock_fn.mock.reset_mock()
      expected = no_opt.compute_on(df).display(return_formatted_df=True)

    self.assertEqual(11, mock_fn.mock.call_count)
    testing.assert_frame_equal(output, expected)
    testing.assert_frame_equal(df, original_df)

  @parameterized.named_parameters(*PRECOMPUTABLE_METRICS_JK)
  def test_optimization_root_filter(self, m):
    opt = operations.Jackknife('cookie', m, 0.9, where='cookie > 0')
    no_opt = operations.Jackknife(
        'cookie', m, 0.9, enable_optimization=False, where='cookie > 0'
    )
    df = pd.DataFrame({
        'x': np.random.rand(10),
        'y': np.random.rand(10),
        'w': np.random.rand(10),
        'w2': np.random.randint(1, 10, size=10),
        'cookie': range(10),
    })
    original_df = df.copy()

    with mock.patch.object(
        m.__class__, 'compute_slices', spy_decorator(m.__class__.compute_slices)
    ) as mock_fn:
      output1 = opt.compute_on(df).display(return_formatted_df=True)
      if isinstance(m, (metrics.Sum, metrics.Count)):
        mock_fn.mock.assert_called_once()
      else:
        mock_fn.mock.assert_not_called()
      mock_fn.mock.reset_mock()
      output2 = no_opt.compute_on(df).display(return_formatted_df=True)
    expected = no_opt.compute_on(df[df.cookie > 0]).display(
        return_formatted_df=True
    )

    self.assertEqual(10, mock_fn.mock.call_count)
    testing.assert_frame_equal(output1, expected)
    testing.assert_frame_equal(output2, expected)
    testing.assert_frame_equal(df, original_df)

  @parameterized.named_parameters(*PRECOMPUTABLE_METRICS_JK)
  def test_optimization_leaf_filter(self, m):
    m = copy.deepcopy(m)
    m.where = 'cookie > 0'
    opt = operations.Jackknife('cookie', m, 0.9)
    no_opt = operations.Jackknife('cookie', m, 0.9, enable_optimization=False)
    df = pd.DataFrame({
        'x': np.random.rand(10),
        'y': np.random.rand(10),
        'w': np.random.rand(10),
        'w2': np.random.randint(1, 10, size=10),
        'cookie': range(10),
    })
    original_df = df.copy()

    with mock.patch.object(
        m.__class__, 'compute_slices', spy_decorator(m.__class__.compute_slices)
    ) as mock_fn:
      output = opt.compute_on(df).display(return_formatted_df=True)
      if isinstance(m, (metrics.Sum, metrics.Count)):
        mock_fn.mock.assert_called_once()
      else:
        mock_fn.mock.assert_not_called()
      mock_fn.mock.reset_mock()
      expected = no_opt.compute_on(df).display(return_formatted_df=True)

    self.assertEqual(11, mock_fn.mock.call_count)
    testing.assert_frame_equal(output, expected)
    testing.assert_frame_equal(df, original_df)

  @parameterized.named_parameters(*PRECOMPUTABLE_OPERATIONS)
  def test_optimization_on_operation_filter(self, m):
    m = copy.deepcopy(m)
    if not m.children:
      m = m(metrics.Ratio('x', 'y'))
    m.where = 'unit != 0'
    n = 40
    opt = operations.Jackknife('unit', m, 0.9)
    no_opt = operations.Jackknife('unit', m, 0.9, enable_optimization=False)
    df = pd.DataFrame({
        'x': np.random.rand(n),
        'y': np.random.rand(n),
        'w': np.random.rand(n),
        'w2': np.random.randint(1, 10, size=n),
        'unit': np.random.choice(range(4), n),
        'grp': np.random.choice(range(4), n),
        'cookie': np.random.choice(range(5), n),
    })
    original_df = df.copy()
    expected_call_ct = (
        3 if isinstance(m, (operations.PrePostChange, operations.CUPED)) else 2
    )
    with mock.patch.object(
        metrics.Sum, 'compute_slices', spy_decorator(metrics.Sum.compute_slices)
    ) as mock_fn:
      output = opt.compute_on(df).display(return_formatted_df=True)
      self.assertEqual(mock_fn.mock.call_count, expected_call_ct)
    expected = no_opt.compute_on(df).display(return_formatted_df=True)

    testing.assert_frame_equal(output, expected)
    testing.assert_frame_equal(df, original_df)

  @parameterized.named_parameters(*PRECOMPUTABLE_OPERATIONS)
  def test_optimization_on_operation_leaf_filter(self, m):
    m = copy.deepcopy(m)
    if not m.children:
      m = m(metrics.Ratio('x', 'y'))
    m.children[0].where = 'unit != 0'
    n = 40
    opt = operations.Jackknife('unit', m, 0.9)
    no_opt = operations.Jackknife('unit', m, 0.9, enable_optimization=False)
    df = pd.DataFrame({
        'x': np.random.rand(n),
        'y': np.random.rand(n),
        'w': np.random.rand(n),
        'w2': np.random.randint(1, 10, size=n),
        'unit': np.random.choice(range(4), n),
        'grp': np.random.choice(range(4), n),
        'cookie': np.random.choice(range(5), n),
    })
    original_df = df.copy()
    expected_call_ct = (
        3 if isinstance(m, (operations.PrePostChange, operations.CUPED)) else 2
    )
    with mock.patch.object(
        metrics.Sum, 'compute_slices', spy_decorator(metrics.Sum.compute_slices)
    ) as mock_fn:
      output = opt.compute_on(df).display(return_formatted_df=True)
      self.assertEqual(mock_fn.mock.call_count, expected_call_ct)
    expected = no_opt.compute_on(df).display(return_formatted_df=True)

    testing.assert_frame_equal(output, expected)
    testing.assert_frame_equal(df, original_df)

  @parameterized.named_parameters(*PRECOMPUTABLE_METRICS_JK)
  def test_split_by(self, m):
    opt = operations.Jackknife('cookie', m, 0.9)
    n = 20
    df = pd.DataFrame({
        'x': np.random.rand(n),
        'y': np.random.rand(n),
        'grp': ['a'] * 10 + ['b'] * 10,
        'w': np.random.random(n),
        'w2': np.random.randint(1, 10, size=n),
        'cookie': list(range(5)) * 4,
    })
    original_df = df.copy()
    with mock.patch.object(
        m.__class__, 'compute_slices', spy_decorator(m.__class__.compute_slices)
    ) as mock_fn:
      output = opt.compute_on(df, 'grp').display(return_formatted_df=True)
      if isinstance(m, (metrics.Sum, metrics.Count)):
        mock_fn.mock.assert_called_once()
      else:
        mock_fn.mock.assert_not_called()
      no_opt = operations.Jackknife('cookie', m, 0.9, enable_optimization=False)
      mock_fn.mock.reset_mock()
      expected = no_opt.compute_on(df, 'grp').display(return_formatted_df=True)

    self.assertEqual(6, mock_fn.mock.call_count)
    testing.assert_frame_equal(output, expected)
    testing.assert_frame_equal(df, original_df)

  @parameterized.named_parameters(*PRECOMPUTABLE_METRICS_JK)
  def test_operation_optimization(self, m):
    p = operations.PercentChange('grp', 'a', m, True)
    opt = operations.Jackknife('cookie', p, 0.9)
    n = 20
    df = pd.DataFrame({
        'x': np.random.rand(n),
        'y': np.random.rand(n),
        'grp': ['a'] * 10 + ['b'] * 10,
        'w': np.random.random(n),
        'w2': np.random.randint(1, 10, size=n),
        'cookie': list(range(5)) * 4,
    })
    original_df = df.copy()
    with mock.patch.object(
        m.__class__, 'compute_slices', spy_decorator(m.__class__.compute_slices)
    ) as mock_fn:
      output = opt.compute_on(df).display(return_formatted_df=True)
      if isinstance(m, (metrics.Sum, metrics.Count)):
        mock_fn.mock.assert_called_once()
      else:
        mock_fn.mock.assert_not_called()
      no_opt = operations.Jackknife('cookie', p, 0.9, enable_optimization=False)
      mock_fn.mock.reset_mock()
      expected = no_opt.compute_on(df).display(return_formatted_df=True)

    self.assertEqual(6, mock_fn.mock.call_count)
    testing.assert_frame_equal(output, expected)
    testing.assert_frame_equal(df, original_df)

  def test_jackknife_with_count_distinct(self):
    df = pd.DataFrame({
        'x': [1, 2, 2],
        'cookie': [1, 2, 3],
    })
    m = operations.Jackknife('cookie', metrics.Count('x', distinct=True))
    output = m.compute_on(df)
    expected = pd.DataFrame({
        ('count(distinct x)', 'Value'): [2.0],
        ('count(distinct x)', 'Jackknife SE'): [2.0 / 3],
    })
    expected.columns.names = ['Metric', None]
    testing.assert_frame_equal(output, expected)

  def test_jackknife_with_operation(self):
    df = pd.DataFrame({
        'x': range(1, 6),
        'cookie': [1, 1, 2, 3, 4],
        'grp': ['foo', 'foo', 'bar', 'bar', 'bar'],
    })
    sum_x = metrics.Sum('x')
    mean_x = metrics.Mean('x')
    count_x = sum_x / mean_x
    metric = metrics.MetricList((sum_x, mean_x, count_x))
    change = operations.AbsoluteChange('grp', 'foo', metric)
    jk_change = operations.Jackknife('cookie', change)
    output = jk_change.compute_on(df, melted=True)
    sum_std = np.std((6, 5, 4), ddof=1) * 2 / np.sqrt(3)
    mean_std = np.std((3, 2.5, 2), ddof=1) * 2 / np.sqrt(3)
    expected = pd.DataFrame({
        'Value': [9, 2.5, 1],
        'Jackknife SE': [sum_std, mean_std, 0],
        'Metric': [
            'sum(x) Absolute Change',
            'mean(x) Absolute Change',
            'sum(x) / mean(x) Absolute Change',
        ],
    })
    expected['grp'] = 'bar'
    expected.set_index(['Metric', 'grp'], inplace=True)
    testing.assert_frame_equal(output, expected)

  def test_jackknife_with_operation_with_multiple_columns_display(self):
    df = pd.DataFrame({
        'x': range(1, 6),
        'cookie': [1, 1, 2, 3, 4],
        'grp': ['foo', 'foo', 'bar', 'bar', 'bar'],
        'grp2': ['A', 'A', 'A', 'B', 'B'],
    })
    sum_x = metrics.Sum('x')
    change = operations.AbsoluteChange(
        ['grp', 'grp2'], ('foo', 'A'), sum_x
    )
    jk_change = operations.Jackknife('cookie', change, 0.9)
    output = jk_change.compute_on(df)
    output.display()

  def test_operation_with_jackknife(self):
    df = pd.DataFrame({
        'x': range(1, 11),
        'cookie': [1, 1, 2, 3, 4] * 2,
        'grp': ['foo', 'foo', 'bar', 'bar', 'bar'] * 2,
        'grp2': ['A'] * 6 + ['B'] * 4,
    })
    ms = metrics.MetricList(
        [metrics.Sum('x', where='x > 3'), metrics.Mean('x', where='x > 5')]
    )
    jk = operations.Jackknife('cookie', ms, where='x > 4')
    m = operations.AbsoluteChange('grp2', 'A', jk, where='x > 2')
    output = m.compute_on(df)

    sumx = metrics.Sum('x')
    meanx = metrics.Mean('x')
    jk = operations.Jackknife('cookie')
    ab = operations.AbsoluteChange('grp2', 'A')
    expected_sum = ab(jk(sumx)).compute_on(df[df.x > 4])
    expected_mean = ab(jk(meanx)).compute_on(df[df.x > 5])
    expected = pd.concat((expected_sum, expected_mean), axis=1)
    testing.assert_frame_equal(output, expected)

  def test_jackknife_with_operation_splitby(self):
    df = pd.DataFrame({
        'x': range(1, 11),
        'cookie': [1, 1, 2, 3, 4] * 2,
        'grp': ['foo', 'foo', 'bar', 'bar', 'bar'] * 2,
        'grp2': ['A'] * 5 + ['B'] * 5,
    })
    output = self.jk_change.compute_on(df, 'grp2')
    expected = []
    for g in ['A', 'B']:
      expected.append(self.jk_change.compute_on(df[df.grp2 == g]))
    expected = pd.concat(expected, keys=['A', 'B'], names=['grp2'])
    testing.assert_frame_equal(output, expected)

  def test_unequal_index_broacasting(self):
    df = pd.DataFrame({
        'x': range(6),
        'grp': ['A'] * 3 + ['B'] * 3,
        'cookie': [1, 2, 3, 1, 2, 3],
    })
    s = metrics.Sum('x')
    pct = operations.PercentChange('grp', 'A', s)
    m = operations.Jackknife('cookie', s * pct, 0.9)
    m_no_opt = operations.Jackknife('cookie', s * pct, 0.9, False)

    output = m.compute_on(df, melted=True)
    output_pt_est = output['Value']
    expected_pt_est = pct.compute_on(df, melted=True).iloc[:, 0] * df.x.sum()
    expected_pt_est.index = expected_pt_est.index.set_levels(
        ['sum(x) * sum(x) Percent Change'], level=0
    )
    output_html = output.display(return_formatted_df=True)
    expected_html = m_no_opt.compute_on(df).display(return_formatted_df=True)

    testing.assert_series_equal(expected_pt_est, output_pt_est)
    testing.assert_frame_equal(output_html, expected_html)
    self.assertTrue(m.can_precompute())

  def test_custom_operation(self):
    df = pd.DataFrame({
        'x': range(6),
        'grp': ['A'] * 3 + ['B'] * 3,
        'unit': [1, 2, 3, 1, 2, 3],
    })

    class AggregateBy(operations.Operation):

      def __init__(self, agg, child):
        super(AggregateBy, self).__init__(child, '{}', agg)

      def compute_on_children(self, children, split_by):
        return children

    agg = AggregateBy('grp', metrics.Sum('x'))
    jk1 = operations.Jackknife('unit', agg)
    jk2 = operations.Jackknife('unit', agg, enable_optimization=False)

    output1 = jk1.compute_on(df)
    output2 = jk2.compute_on(df)
    expected = operations.Jackknife('unit', metrics.Sum('x')).compute_on(
        df, 'grp'
    )

    self.assertTrue(jk1.can_precompute())
    testing.assert_frame_equal(output1, expected)
    testing.assert_frame_equal(output2, expected)

  def test_integration(self):
    df = pd.DataFrame({
        'x': np.arange(0, 3, 0.5),
        'cookie': [1, 2, 3, 1, 2, 3],
        'grp': list('AABBAB'),
    })
    change = metrics.Sum('x') | operations.AbsoluteChange('grp', 'A')
    m = change | operations.Jackknife('cookie')
    output = m.compute_on(df, melted=True)
    std = np.std((1, 5, -1), ddof=1) * 2 / np.sqrt(3)

    expected = pd.DataFrame(
        [['sum(x) Absolute Change', 'B', 2.5, std]],
        columns=['Metric', 'grp', 'Value', 'Jackknife SE'],
    )
    expected.set_index(['Metric', 'grp'], inplace=True)
    testing.assert_frame_equal(output, expected)

  def test_display_simple_metric(self):
    df = pd.DataFrame({
        'x': np.arange(0, 3, 0.5),
        'cookie': [1, 2, 3, 1, 2, 3],
    })
    m = operations.Jackknife('cookie', metrics.Sum('x', where='x > 0.5'), 0.9)
    res = m.compute_on(df)
    output = res.display(return_formatted_df=True)
    expected = pd.DataFrame(
        {
            'sum(x)': (
                (
                    '<div class="ci-display-good-change'
                    ' ci-display-cell"><div><span'
                    ' class="ci-display-ratio">7.0000</span><div'
                    ' class="ci-display-flex-line-break"></div><span'
                    ' class="ci-display-ci-range">[3.4906,'
                    ' 10.5094]</span></div></div>'
                ),
            )
        }
    )
    expected.columns.name = 'Metric'
    testing.assert_frame_equal(output, expected)

  def test_display_simple_metric_split_by(self):
    df = pd.DataFrame({
        'x': np.arange(0, 3, 0.5),
        'cookie': [1, 2, 3, 1, 2, 3],
        'grp': list('AABBAB'),
    })
    m = operations.Jackknife('cookie', metrics.Sum('x'), 0.9)
    res = m.compute_on(df, 'grp')
    output = res.display(return_formatted_df=True)
    expected = pd.DataFrame(
        {
            'Dimensions': [
                (
                    '<div><div><span'
                    ' class="ci-display-dimension">A</span></div></div>'
                ),
                (
                    '<div><div><span'
                    ' class="ci-display-dimension">B</span></div></div>'
                ),
            ],
            'sum(x)': [
                (
                    '<div class="ci-display-cell"><div><span'
                    ' class="ci-display-ratio">2.5000</span><div'
                    ' class="ci-display-flex-line-break"></div><span'
                    ' class="ci-display-ci-range">[-5.3922,'
                    ' 10.3922]</span></div></div>'
                ),
                (
                    '<div class="ci-display-cell"><div><span'
                    ' class="ci-display-ratio">5.0000</span><div'
                    ' class="ci-display-flex-line-break"></div><span'
                    ' class="ci-display-ci-range">[-1.3138,'
                    ' 11.3138]</span></div></div>'
                ),
            ],
        },
    )
    expected.columns.name = 'Metric'
    testing.assert_frame_equal(output, expected)

  def test_display_change(self):
    df = pd.DataFrame({
        'x': [1, 100, 2, 100, 3, 100],
        'cookie': [1, 2, 3, 1, 2, 3],
        'grp': ['A', 'B'] * 3,
    })
    change = metrics.Sum('x') | operations.PercentChange('grp', 'A')
    m = operations.Jackknife('cookie', change, 0.9)
    res = m.compute_on(df)
    output = res.display(return_formatted_df=True)
    expected = pd.DataFrame(
        {
            'Dimensions': [
                (
                    '<div><div><span class="ci-display-experiment-id">A</span>'
                    '</div></div>'
                ),
                (
                    '<div><div><span class="ci-display-experiment-id">B</span>'
                    '</div></div>'
                ),
            ],
            'sum(x)': [
                '<div class="ci-display-cell">6.0000</div>',
                (
                    '<div class="ci-display-good-change'
                    ' ci-display-cell"><div>300.0000<div'
                    ' class="ci-display-flex-line-break"></div><span'
                    ' class="ci-display-ratio">4900.00%</span><div'
                    ' class="ci-display-flex-line-break"></div><span'
                    ' class="ci-display-ci-range">[357.80, 9442.20]'
                    ' %</span></div></div>'
                ),
            ],
        },
    )
    expected.columns.name = 'Metric'
    testing.assert_frame_equal(output, expected)

  def test_display_change_split_by(self):
    df = pd.DataFrame({
        'x': list(range(0, 5)) + list(range(1000, 1004)),
        'cookie': [1, 2, 3] * 3,
        'grp': list('AB') * 4 + ['B'],
        'expr': ['foo'] * 5 + ['bar'] * 4,
    })
    change = metrics.Sum('x') | operations.AbsoluteChange('expr', 'foo')
    m = operations.Jackknife('cookie', change, 0.9)
    res = m.compute_on(df, 'grp')
    output = res.display(return_formatted_df=True)
    expected = pd.DataFrame(
        {
            'Dimensions': [
                (
                    '<div><div><span'
                    ' class="ci-display-experiment-id">foo</span><div'
                    ' class="ci-display-flex-line-break"></div><span'
                    ' class="ci-display-dimension">A</span></div></div>'
                ),
                (
                    '<div><div><span'
                    ' class="ci-display-experiment-id">bar</span><div'
                    ' class="ci-display-flex-line-break"></div><span'
                    ' class="ci-display-dimension">A</span></div></div>'
                ),
                (
                    '<div><div><span'
                    ' class="ci-display-experiment-id">foo</span><div'
                    ' class="ci-display-flex-line-break"></div><span'
                    ' class="ci-display-dimension">B</span></div></div>'
                ),
                (
                    '<div><div><span'
                    ' class="ci-display-experiment-id">bar</span><div'
                    ' class="ci-display-flex-line-break"></div><span'
                    ' class="ci-display-dimension">B</span></div></div>'
                ),
            ],
            'sum(x)': [
                '<div class="ci-display-cell">6.0000</div>',
                (
                    '<div class="ci-display-good-change'
                    ' ci-display-cell"><div>1001.0000<div'
                    ' class="ci-display-flex-line-break"></div><span'
                    ' class="ci-display-ratio">995.0000</span><div'
                    ' class="ci-display-flex-line-break"></div><span'
                    ' class="ci-display-ci-range">[988.6862,'
                    ' 1001.3138]</span></div></div>'
                ),
                '<div class="ci-display-cell">4.0000</div>',
                (
                    '<div class="ci-display-cell">'
                    '<div>3005.0000<div class="ci-display-flex-line-break">'
                    '</div><span class="ci-display-ratio">3001.0000</span>'
                    '<div class="ci-display-flex-line-break"></div>'
                    '<span class="ci-display-ci-range">[-380.8246, 6382.8246]'
                    '</span></div></div>'
                ),
            ],
        },
    )
    expected.columns.name = 'Metric'
    testing.assert_frame_equal(output, expected)


class BootstrapTests(parameterized.TestCase):
  n = 100
  x = np.arange(0, 3, 0.5)
  df = pd.DataFrame({'x': x, 'grp': ['A'] * 3 + ['B'] * 3})
  metric = metrics.MetricList((metrics.Sum('x'), metrics.Count('x')))
  bootstrap_no_unit = operations.Bootstrap(None, metric, n)
  bootstrap_no_unit_no_opt = operations.Bootstrap(
      None, metric, n, enable_optimization=False
  )
  bootstrap_unit = operations.Bootstrap('unit', metric, n)
  bootstrap_unit_no_opt = operations.Bootstrap(
      'unit', metric, n, enable_optimization=False
  )

  def test_get_samples(self):
    m = operations.Bootstrap(None, metrics.Sum('x'), 2)
    output = [s[1] for s in m.get_samples(self.df, [])]
    self.assertLen(output, 2)
    for s in output:
      self.assertLen(s, len(self.df))

  def test_get_samples_splitby(self):
    m = operations.Bootstrap(None, metrics.Sum('x'), 2)
    output = [s[1] for s in m.get_samples(self.df, ['grp'])]
    self.assertLen(output, 2)
    expected = self.df.groupby('grp').size()
    for s in output:
      testing.assert_series_equal(s.groupby('grp').size(), expected)

  def test_get_samples_with_unit(self):
    m = operations.Bootstrap('grp', metrics.Sum('x'), 20)
    output = [s[1] for s in m.get_samples(self.df, [])]
    self.assertLen(output, 20)
    grp_cts = self.df.groupby('grp').size()
    for s in output:
      if s is not None:
        self.assertEqual(
            [2], (s.groupby('grp').size() / grp_cts).sum(numeric_only=True)
        )

  def test_get_samples_with_unit_splitby(self):
    df = pd.DataFrame({
        'x': range(10),
        'grp': ['A'] * 2 + ['B'] * 3 + ['C'] * 4 + ['D'],
        'grp2': ['foo'] * 5 + ['bar'] * 5,
    })
    m = operations.Bootstrap('grp', metrics.Sum('x'), 10)
    output = [s[1] for s in m.get_samples(df, ['grp2'])]
    self.assertLen(output, 10)
    grp_cts = df.groupby(['grp2', 'grp']).size()
    for s in output:
      s = s.groupby(['grp2', 'grp']).size()
      self.assertEqual(
          [2], (s / grp_cts).groupby(['grp2']).sum(numeric_only=True).unique()
      )

  def test_bootstrap_no_unit(self):
    np.random.seed(42)
    unmelted = self.bootstrap_no_unit.compute_on(self.df)

    np.random.seed(42)
    estimates = []
    for _ in range(self.n):
      buckets_sampled = np.random.choice(range(len(self.x)), size=len(self.x))
      sample = self.df.iloc[buckets_sampled]
      res = metrics.Sum('x').compute_on(sample, return_dataframe=False)
      estimates.append(res)
    std_sumx = np.std(estimates, ddof=1)

    expected = pd.DataFrame(
        [[7.5, std_sumx, 6.0, 0.0]],
        columns=pd.MultiIndex.from_product(
            [['sum(x)', 'count(x)'], ['Value', 'Bootstrap SE']],
            names=['Metric', None],
        ),
    )
    testing.assert_frame_equal(unmelted, expected)

    np.random.seed(42)
    melted = self.bootstrap_no_unit.compute_on(self.df, melted=True)
    expected = pd.DataFrame(
        data={'Value': [7.5, 6.0], 'Bootstrap SE': [std_sumx, 0.0]},
        columns=['Value', 'Bootstrap SE'],
        index=['sum(x)', 'count(x)'],
    )
    expected.index.name = 'Metric'
    testing.assert_frame_equal(melted, expected)

  @parameterized.named_parameters(*PRECOMPUTABLE_METRICS_BS)
  def test_bootstrap_unit(self, m):
    df = pd.DataFrame({
        'x': np.random.rand(6),
        'y': np.random.rand(6),
        'w': np.random.rand(6),
        'w2': np.random.randint(1, 10, size=6),
        'unit': ['A', 'A', 'B', 'C', 'C', 'C'],
    })
    bootstrap_unit = operations.Bootstrap('unit', m, 5)
    bootstrap_unit_no_opt = operations.Bootstrap(
        'unit', m, 5, enable_optimization=False
    )
    np.random.seed(42)
    with mock.patch.object(
        operations.Bootstrap,
        'get_samples',
        spy_decorator(operations.Bootstrap.get_samples),
    ) as mock_fn_opt:
      output1 = bootstrap_unit.compute_on(df)
    np.random.seed(42)
    with mock.patch.object(
        operations.Bootstrap,
        'get_samples',
        spy_decorator(operations.Bootstrap.get_samples),
    ) as mock_fn_no_opt:
      output2 = bootstrap_unit_no_opt.compute_on(df)

    np.random.seed(42)
    estimates = []
    for _ in range(5):
      buckets_sampled = np.random.choice(['A', 'B', 'C'], size=3)
      sample = pd.concat(df[df['unit'] == b] for b in buckets_sampled)
      res = m.compute_on(sample, return_dataframe=False)
      estimates.append(res)
    std = np.std(estimates, ddof=1)
    expected = pd.DataFrame(
        [[m.compute_on(df, return_dataframe=False), std]],
        columns=pd.MultiIndex.from_product(
            [[m.name], ['Value', 'Bootstrap SE']], names=['Metric', None]
        ),
    ).astype(float)

    testing.assert_frame_equal(output1, expected)
    mock_fn_opt.mock.assert_called_once()
    self.assertLen(mock_fn_opt.mock.call_args[0][0], 3)
    testing.assert_frame_equal(output2, expected)
    mock_fn_no_opt.mock.assert_called_once()
    self.assertIs(mock_fn_no_opt.mock.call_args[0][0], df)

  @parameterized.named_parameters(*PRECOMPUTABLE_METRICS_BS)
  def test_bootstrap_unit_opt_splitby(self, m):
    m = copy.deepcopy(m)
    m.where = 'unit != 0'
    n = 8
    df = pd.DataFrame({
        'x': np.random.rand(n),
        'y': np.random.rand(n),
        'w': np.random.rand(n),
        'w2': np.random.randint(1, 10, size=n),
        'unit': [0, 1, 1, 2, 2, 2, 2, 2],
        'grp': [1, 2] * 4,
    })
    bootstrap_unit = operations.Bootstrap('unit', m, 5)
    with mock.patch.object(
        operations.Bootstrap,
        'get_samples',
        spy_decorator(operations.Bootstrap.get_samples),
    ) as mock_fn_opt:
      bootstrap_unit.compute_on(df, 'grp')

    mock_fn_opt.mock.assert_called_once()
    self.assertLen(mock_fn_opt.mock.call_args[0][0], 5)

  @parameterized.named_parameters(*PRECOMPUTABLE_METRICS_BS)
  def test_bootstrap_unit_root_filter(self, m):
    n = 8
    df = pd.DataFrame({
        'x': np.random.rand(n),
        'y': np.random.rand(n),
        'w': np.random.rand(n),
        'w2': np.random.randint(1, 10, size=n),
        'unit': [0, 0, 1, 2, 2, 2, 2, 2],
        'grp': [1, 2] * 4,
    })
    bootstrap_unit = operations.Bootstrap('unit', m, 5, 0.9, where='unit!=0')
    bootstrap_unit_no_opt = operations.Bootstrap(
        'unit', m, 5, 0.9, False, where='unit!=0'
    )
    np.random.seed(42)
    with mock.patch.object(
        operations.Bootstrap,
        'get_samples',
        spy_decorator(operations.Bootstrap.get_samples),
    ) as mock_fn_opt:
      output = bootstrap_unit.compute_on(df, 'grp').display(
          return_formatted_df=True
      )
    np.random.seed(42)
    expected = bootstrap_unit_no_opt.compute_on(df, 'grp').display(
        return_formatted_df=True
    )

    mock_fn_opt.mock.assert_called_once()
    self.assertLen(mock_fn_opt.mock.call_args[0][0], 3)
    testing.assert_frame_equal(output, expected)

  @parameterized.named_parameters(*PRECOMPUTABLE_OPERATIONS)
  def test_bootstrap_unit_opt_on_operation_filter(self, m):
    m = copy.deepcopy(m)
    if not m.children:
      m = m(metrics.Ratio('x', 'y'))
    m.where = 'unit != 0'
    n = 40
    df = pd.DataFrame({
        'x': np.random.rand(n),
        'y': np.random.rand(n),
        'w': np.random.rand(n),
        'w2': np.random.randint(1, 10, size=n),
        'unit': np.random.choice(range(3), n),
        'grp': np.random.choice(range(4), n),
        'cookie': np.random.choice(range(5), n),
    })
    bootstrap_unit = operations.Bootstrap('unit', m, 5)
    with mock.patch.object(
        operations.Bootstrap,
        'get_samples',
        spy_decorator(operations.Bootstrap.get_samples),
    ) as mock_fn_opt:
      bootstrap_unit.compute_on(df)
    preaggregated = mock_fn_opt.mock.call_args[0][0]

    split_by = list(utils.get_extra_split_by(m)) + ['unit']
    self.assertLen(preaggregated, df[split_by].apply(tuple, 1).nunique())
    self.assertTrue(
        preaggregated[preaggregated.unit == 0]
        .drop(columns=split_by)
        .isnull()
        .all()
        .all()
    )

  @parameterized.named_parameters(*PRECOMPUTABLE_OPERATIONS)
  def test_bootstrap_unit_opt_on_operation_leaf_filter(self, m):
    m = copy.deepcopy(m)
    if not m.children:
      m = m(metrics.Ratio('x', 'y'))
    m.children[0].where = 'unit != 0'
    n = 40
    df = pd.DataFrame({
        'x': np.random.rand(n),
        'y': np.random.rand(n),
        'w': np.random.rand(n),
        'w2': np.random.randint(1, 10, size=n),
        'unit': np.random.choice(range(3), n),
        'grp': np.random.choice(range(4), n),
        'cookie': np.random.choice(range(5), n),
    })
    bootstrap_unit = operations.Bootstrap('unit', m, 5)
    with mock.patch.object(
        operations.Bootstrap,
        'get_samples',
        spy_decorator(operations.Bootstrap.get_samples),
    ) as mock_fn_opt:
      bootstrap_unit.compute_on(df)
    preaggregated = mock_fn_opt.mock.call_args[0][0]

    split_by = list(utils.get_extra_split_by(m)) + ['unit']
    self.assertLen(preaggregated, df[split_by].apply(tuple, 1).nunique())
    self.assertTrue(
        preaggregated[preaggregated.unit == 0]
        .drop(columns=split_by)
        .isnull()
        .all()
        .all()
    )

  def test_bootstrap_unit_cache_across_samples(self):
    df = pd.DataFrame({
        'x': range(4),
        'unit': [0, 0, 1, 1],
        'grp': [0, 1] * 2,
    })
    SUM_COMPUTE_THROUGH.mock.reset_mock()
    bootstrap_unit = operations.Bootstrap(
        'unit', operations.AbsoluteChange('grp', 0, metrics.Sum('x')), 5
    )
    with mock.patch.object(
        metrics.Sum,
        'compute_through',
        spy_decorator(metrics.Sum.compute_through),
    ) as mock_fn_opt:
      bootstrap_unit.compute_on(df)
    # The additional two calls are for precomputation and point estimate.
    self.assertLess(mock_fn_opt.mock.call_count, 7)

  def test_bootstrap_no_unit_splitby(self):
    np.random.seed(42)
    unmelted = self.bootstrap_no_unit.compute_on(self.df, 'grp')

    np.random.seed(42)
    expected = []
    grps = ['A', 'B']
    for g in grps:
      expected.append(
          self.bootstrap_no_unit.compute_on(self.df[self.df.grp == g])
      )
    expected = pd.concat(expected, keys=grps, names=['grp'])
    expected = expected.droplevel(-1)  # empty level
    testing.assert_frame_equal(unmelted, expected, rtol=0.04)

    np.random.seed(42)
    melted = self.bootstrap_no_unit.compute_on(self.df, 'grp', melted=True)
    testing.assert_frame_equal(melted, utils.melt(expected), rtol=0.04)

  def test_bootstrap_splitby_multiple(self):
    df = pd.concat([self.df, self.df], keys=['foo', 'bar'], names=['grp0'])
    output = operations.Bootstrap(None, self.metric, self.n, 0.9).compute_on(
        df, ['grp0', 'grp']
    )
    self.assertEqual(output.index.names, ['grp0', 'grp'])
    output.display()  # Check display() runs.

  def test_bootstrap_no_unit_where(self):
    df = pd.DataFrame({'x': range(1, 7), 'grp': ['B'] * 3 + ['A'] * 3})
    metric = operations.Bootstrap(
        None, metrics.Sum('x'), self.n, where='grp == "A"'
    )
    metric_no_filter = operations.Bootstrap(None, metrics.Sum('x'), self.n)
    np.random.seed(42)
    output = metric.compute_on(df)
    np.random.seed(42)
    expected = metric_no_filter.compute_on(df[df.grp == 'A'])
    testing.assert_frame_equal(output, expected)

  def test_confidence(self):
    np.random.seed(42)
    melted = operations.Bootstrap(None, self.metric, self.n, 0.9).compute_on(
        self.df, melted=True
    )
    np.random.seed(42)
    expected = self.bootstrap_no_unit.compute_on(self.df, melted=True)
    multiplier = stats.t.ppf((1 + 0.9) / 2, self.n - 1)
    expected['Bootstrap CI-lower'] = (
        expected['Value'] - multiplier * expected['Bootstrap SE']
    )
    expected['Bootstrap CI-upper'] = (
        expected['Value'] + multiplier * expected['Bootstrap SE']
    )
    expected.drop('Bootstrap SE', axis=1, inplace=True)
    testing.assert_frame_equal(melted, expected)
    melted.display()  # Check display() runs.

    np.random.seed(42)
    unmelted = operations.Bootstrap(None, self.metric, self.n, 0.9).compute_on(
        self.df
    )
    expected = pd.DataFrame(
        [
            list(melted.loc['sum(x)'].values)
            + list(melted.loc['count(x)'].values)
        ],
        columns=pd.MultiIndex.from_product(
            [
                ['sum(x)', 'count(x)'],
                ['Value', 'Bootstrap CI-lower', 'Bootstrap CI-upper'],
            ],
            names=['Metric', None],
        ),
    )
    testing.assert_frame_equal(unmelted, expected)
    unmelted.display()  # Check display() runs.

  def test_unequal_index_broacasting(self):
    df = pd.DataFrame({
        'x': range(6),
        'grp': ['A'] * 3 + ['B'] * 3,
        'cookie': [1, 2, 3, 1, 2, 3],
    })
    s = metrics.Sum('x')
    pct = operations.PercentChange('grp', 'A', s)
    m = operations.Bootstrap('cookie', s * pct, 10, 0.9)
    m_no_opt = operations.Bootstrap('cookie', s * pct, 10, 0.9, False)

    np.random.seed(0)
    output = m.compute_on(df, melted=True)
    output_pt_est = output['Value']
    expected_pt_est = pct.compute_on(df, melted=True).iloc[:, 0] * df.x.sum(
        numeric_only=True
    )
    expected_pt_est.index = expected_pt_est.index.set_levels(
        ['sum(x) * sum(x) Percent Change'], level=0
    )
    output_html = output.display(return_formatted_df=True)
    np.random.seed(0)
    expected_html = m_no_opt.compute_on(df).display(return_formatted_df=True)

    testing.assert_series_equal(expected_pt_est, output_pt_est)
    testing.assert_frame_equal(output_html, expected_html)
    self.assertTrue(m.can_precompute())

  def test_integration(self):
    change = metrics.Sum('x') | operations.AbsoluteChange('grp', 'A')
    m = change | operations.Bootstrap(None, n_replicates=self.n)
    np.random.seed(42)
    output = m.compute_on(self.df, melted=True)
    np.random.seed(42)
    estimates = []
    for _ in range(self.n):
      buckets_sampled = np.random.choice(range(len(self.x)), size=len(self.x))
      sample = self.df.iloc[buckets_sampled]
      try:  # Some sampls might not have any grp 'A' rows.
        res = change.compute_on(sample)
        estimates.append(res.iloc[0, 0])
      except:  # pylint: disable=bare-except
        continue
    std = np.std(estimates, ddof=1)

    expected = pd.DataFrame(
        [['sum(x) Absolute Change', 'B', 4.5, std]],
        columns=['Metric', 'grp', 'Value', 'Bootstrap SE'],
    )
    expected.set_index(['Metric', 'grp'], inplace=True)
    testing.assert_frame_equal(output, expected)


class PoissonBootstrapTests(parameterized.TestCase):

  @parameterized.parameters([([],), (['grp2'],), (('grp2', 'grp3'),)])
  def test_runs(self, split_by):
    df = pd.DataFrame({
        'x': range(6),
        'grp': range(6),
        'grp2': [0] * 3 + [1] * 3,
        'grp3': [0, 1, 2] * 2,
    })
    m = metrics.MetricList([
        metrics.Sum('x'),
        metrics.Count('x'),
        metrics.Max('x'),
        metrics.Min('x'),
    ])
    m1 = operations.PoissonBootstrap('grp', m, 5, 0.9)
    m2 = operations.PoissonBootstrap('grp', m, 5, 0.9, False)
    m3 = operations.PoissonBootstrap(None, m, 5, 0.9)
    np.random.seed(0)
    res1 = m1.compute_on(df, split_by).display(return_formatted_df=True)
    np.random.seed(0)
    res2 = m2.compute_on(df, split_by).display(return_formatted_df=True)
    np.random.seed(0)
    res3 = m3.compute_on(df, split_by).display(return_formatted_df=True)
    testing.assert_frame_equal(res1, res2)
    testing.assert_frame_equal(res2, res3)

  @parameterized.parameters([([],), (['grp2'],), (('grp2', 'grp3'),)])
  def test_each_unit_has_one_row(self, split_by):
    df = pd.DataFrame({
        'x': range(6),
        'grp': range(6),
        'grp2': [0] * 3 + [1] * 3,
        'grp3': [0, 1, 2] * 2,
        'grp4': [0, 1] * 3,
    })
    m = metrics.MetricList([
        metrics.Sum('x'),
        metrics.Count('x'),
        metrics.Max('x'),
        metrics.Min('x'),
    ])
    m = operations.AbsoluteChange('grp4', 1, m)
    m1 = operations.PoissonBootstrap('grp', m, 5, 0.9)
    m2 = operations.PoissonBootstrap(None, m, 5, 0.9)
    np.random.seed(0)
    res1 = m1.compute_on(df, split_by).display(return_formatted_df=True)
    np.random.seed(0)
    res2 = m2.compute_on(df, split_by).display(return_formatted_df=True)
    testing.assert_frame_equal(res1, res2)

  @parameterized.parameters([([],), (['grp2'],), (('grp2', 'grp3'),)])
  def test_each_unit_has_multiple_rows(self, split_by):
    df = pd.DataFrame({
        'x': range(6),
        'grp': [0] * 3 + [1] * 3,
        'grp2': [0, 1] * 3,
        'grp3': [0, 1] * 3,
        'grp4': [0, 1, 2] * 2,
    })
    m = metrics.MetricList([
        metrics.Sum('x'),
        metrics.Count('x'),
        metrics.Max('x'),
        metrics.Min('x'),
    ])
    m = operations.AbsoluteChange('grp4', 0, m)
    m1 = operations.PoissonBootstrap('grp', m, 5, 0.9)
    m2 = operations.PoissonBootstrap('grp', m, 5, 0.9, False)
    np.random.seed(0)
    res1 = m1.compute_on(df, split_by).display(return_formatted_df=True)
    np.random.seed(0)
    res2 = m2.compute_on(df, split_by).display(return_formatted_df=True)
    testing.assert_frame_equal(res1, res2)

  def test_get_samples_with_unit(self):
    x = range(6)
    df = pd.DataFrame({'x': x, 'grp': ['A'] * 3 + ['B'] * 3})
    m = operations.PoissonBootstrap(
        'grp', metrics.Sum('x'), 20, enable_optimization=False
    )
    output = [s[1] for s in m.get_samples(df, [])]
    grp_cts = df.groupby('grp').size()
    for s in output:
      if s is not None:
        self.assertTrue(
            (s.groupby('grp').size() / grp_cts)
            .sum(numeric_only=True)
            .is_integer()
        )

  @parameterized.parameters([True, False])
  def test_poissonbootstrap_unit_cache_across_samples(self, enable_opt):
    np.random.seed(0)
    df = pd.DataFrame({
        'x': range(4),
        'unit': [0, 0, 1, 1],
        'grp': [0, 1] * 2,
    })
    SUM_COMPUTE_THROUGH.mock.reset_mock()
    m = operations.PoissonBootstrap(
        'unit',
        operations.AbsoluteChange('grp', 0, metrics.Sum('x')),
        5,
        enable_optimization=enable_opt,
    )
    with mock.patch.object(
        metrics.Sum,
        'compute_through',
        spy_decorator(metrics.Sum.compute_through),
    ) as mock_fn_opt:
      m.compute_on(df)
    # The additional two calls are for precomputation and point estimate.
    self.assertLess(mock_fn_opt.mock.call_count, 7)


OPERATIONS = SIMPLE_OPERATIONS + [
    ('Distribution multiple over', operations.Distribution(['grp', 'grp4'])),
    (
        'CumulativeDistribution descending',
        operations.CumulativeDistribution('grp', ascending=False),
    ),
    (
        'CumulativeDistribution with order',
        operations.CumulativeDistribution(
            'grp', order=(1, 0, 2), ascending=False
        ),
    ),
    (
        'CumulativeDistribution multiple over',
        operations.CumulativeDistribution(['grp', 'grp4']),
    ),
    (
        'PercentChange include base',
        operations.PercentChange('grp', 1, include_base=True),
    ),
    (
        'PercentChange multiple conditions',
        operations.PercentChange(['grp', 'grp4'], (0, 'a')),
    ),
    (
        'AbsoluteChange include base',
        operations.AbsoluteChange('grp', 1, include_base=True),
    ),
    (
        'AbsoluteChange multiple conditions',
        operations.AbsoluteChange(['grp', 'grp4'], (0, 'a')),
    ),
    ('CUPED', operations.CUPED('grp', 0, stratified_by='grp4')),
    ('PrePost', operations.PrePostChange('grp', 0, stratified_by='grp4')),
    ('MH', operations.MH('grp', 0, 'grp4')),
    (
        'MH multiple conditions',
        operations.MH(['grp', 'grp4'], (0, 'b'), 'grp3'),
    ),
]
OPERATIONS_AND_JACKKNIFE = OPERATIONS + [
    ('Jackknife no confidence', operations.Jackknife('cookie')),
    (
        'Jackknife with confidence',
        operations.Jackknife('cookie', confidence=0.9),
    ),
]
ALL_OPERATIONS = OPERATIONS_AND_JACKKNIFE + [
    (
        'Bootstrap no unit no confidence',
        operations.Bootstrap(None, n_replicates=5),
    ),
    (
        'Bootstrap with unit no confidence',
        operations.Bootstrap('grp', n_replicates=5),
    ),
    (
        'Bootstrap no unit',
        operations.Bootstrap(None, confidence=0.9, n_replicates=5),
    ),
    (
        'Bootstrap with unit',
        operations.Bootstrap('grp', confidence=0.9, n_replicates=5),
    ),
    (
        'PoissonBootstrap no unit no confidence',
        operations.PoissonBootstrap(None, n_replicates=5),
    ),
    (
        'PoissonBootstrap with unit no confidence',
        operations.PoissonBootstrap('grp', n_replicates=5),
    ),
    (
        'PoissonBootstrap no unit',
        operations.PoissonBootstrap(None, confidence=0.9, n_replicates=5),
    ),
    (
        'PoissonBootstrap with unit',
        operations.PoissonBootstrap('grp', confidence=0.9, n_replicates=5),
    ),
]


def set_up_metric(m):
  m = copy.deepcopy(m)
  if not m.children:
    m = m(
        metrics.MetricList((metrics.Ratio('x', 'y'), metrics.Ratio('y', 'x')))
    )
  return m


SUM_COMPUTE_THROUGH = spy_decorator(metrics.Sum.compute_through)


class CommonTest(parameterized.TestCase):
  np.random.seed(0)
  n = 100
  df = pd.DataFrame({
      'x': np.random.random(n),
      'y': np.random.random(n),
      'cookie': np.random.choice(range(4), n),
      'grp': np.random.choice(range(3), n),
      'grp2': np.random.choice(list('ab'), n),
      'grp3': np.random.choice(range(2), n),
      'grp4': np.random.choice(list('abc'), n),
  })

  @parameterized.named_parameters(*OPERATIONS_AND_JACKKNIFE)
  def test_melted(self, op):
    op = set_up_metric(op)
    output = op.compute_on(self.df, melted=True)
    expected = utils.melt(op.compute_on(self.df))
    pd.testing.assert_frame_equal(output, expected)

  @parameterized.named_parameters(*OPERATIONS_AND_JACKKNIFE)
  def test_splitby_melted(self, op):
    op = set_up_metric(op)
    output = op.compute_on(self.df, 'grp2', True)
    expected = utils.melt(op.compute_on(self.df, 'grp2'))
    pd.testing.assert_frame_equal(output, expected)

  @parameterized.named_parameters(*OPERATIONS_AND_JACKKNIFE)
  def test_split_by(self, op):
    op = set_up_metric(op)
    output = op.compute_on(self.df, 'grp2')
    expected1 = op.compute_on(self.df[self.df.grp2 == 'a'])
    expected2 = op.compute_on(self.df[self.df.grp2 == 'b'])
    expected = pd.concat(
        [expected1, expected2], keys=['a', 'b'], names=['grp2']
    )
    if not expected.index.names[-1]:
      expected = expected.droplevel(-1)
    pd.testing.assert_frame_equal(output, expected)

  @parameterized.named_parameters(*OPERATIONS)
  def test_split_by_multiple(self, op):
    op = set_up_metric(op)
    output = op.compute_on(self.df, ['grp2', 'cookie'])
    expected1 = op.compute_on(self.df[self.df.grp2 == 'a'], 'cookie')
    expected2 = op.compute_on(self.df[self.df.grp2 == 'b'], 'cookie')
    expected = pd.concat(
        [expected1, expected2], keys=['a', 'b'], names=['grp2']
    )
    pd.testing.assert_frame_equal(output, expected)

  @parameterized.named_parameters(*OPERATIONS_AND_JACKKNIFE)
  def test_multiple_children(self, op):
    if isinstance(op, (operations.CUPED, operations.PrePostChange)):
      return
    op = copy.deepcopy(op)
    op1 = copy.deepcopy(op)
    op2 = copy.deepcopy(op)
    m1 = metrics.Ratio('y', 'x')
    m2 = metrics.Ratio('x', 'y')
    op = op(metrics.MetricList((m1, m2)))
    op1 = op1(m1)
    op2 = op2(m2)
    output = op.compute_on(self.df)
    expected1 = op1.compute_on(self.df)
    expected2 = op2.compute_on(self.df)
    expected = pd.concat([expected1, expected2], axis=1)
    pd.testing.assert_frame_equal(output, expected)

  @parameterized.named_parameters(*OPERATIONS_AND_JACKKNIFE)
  def test_pipeline(self, op):
    op1 = copy.deepcopy(op)
    op2 = copy.deepcopy(op)
    m = metrics.MetricList((metrics.Ratio('x', 'y'), metrics.Ratio('y', 'x')))
    output = m | op1 | metrics.compute_on(self.df)
    expected = op2(m).compute_on(self.df)
    testing.assert_frame_equal(output, expected)

  @parameterized.named_parameters(*OPERATIONS)
  def test_with_jackknife(self, op):
    op = set_up_metric(op)
    output = (
        operations.Jackknife('cookie', op, 0.9)
        .compute_on(self.df)
        .display(return_formatted_df=True)
    )
    expected = (
        operations.Jackknife('cookie', op, 0.9, False)
        .compute_on(self.df)
        .display(return_formatted_df=True)
    )
    pd.testing.assert_frame_equal(output, expected)

  @parameterized.named_parameters(*OPERATIONS_AND_JACKKNIFE)
  def test_parent_filter(self, op):
    op = set_up_metric(op)
    op.where = 'x > 0.2'
    output = op.compute_on(self.df)
    expected = op.compute_on(self.df[self.df.x > 0.2])
    testing.assert_frame_equal(output, expected)

  @parameterized.named_parameters(*OPERATIONS_AND_JACKKNIFE)
  def test_child_filter(self, op):
    op = set_up_metric(op)
    no_filter = copy.deepcopy(op)
    op.children[0].where = 'x > 0.2'
    output = op.compute_on(self.df)
    expected = no_filter.compute_on(self.df[self.df.x > 0.2])
    testing.assert_frame_equal(output, expected)

  @parameterized.named_parameters(*OPERATIONS_AND_JACKKNIFE)
  def test_children_filter(self, op):
    if isinstance(op, (operations.CUPED, operations.PrePostChange)):
      return
    op = copy.deepcopy(op)
    m1 = metrics.Ratio('x', 'y', where='x > 0.2')
    m2 = metrics.Ratio('y', 'x', where='x > 0.7')
    output = op(metrics.MetricList((m1, m2))).compute_on(self.df)
    expected1 = op(m1).compute_on(self.df)
    expected2 = op(m2).compute_on(self.df)
    expected = pd.concat((expected1, expected2), axis=1)
    testing.assert_frame_equal(output, expected)

  @parameterized.named_parameters(*OPERATIONS_AND_JACKKNIFE)
  def test_metriclist_filter(self, op):
    op = copy.deepcopy(op)
    m0 = metrics.MetricList((metrics.Ratio('y', 'x'), metrics.Ratio('x', 'y')))
    m = metrics.MetricList(
        (
            metrics.Ratio('y', 'x', where='x > 0.2'),
            metrics.Ratio('x', 'y', where='x > 0.2'),
        ),
        where='y > 0.2',
    )
    metric = op(m)
    output = metric.compute_on(self.df)
    expected = op(m0).compute_on(self.df[(self.df.x > 0.2) & (self.df.y > 0.2)])
    testing.assert_frame_equal(output, expected)

  @parameterized.named_parameters(*ALL_OPERATIONS)
  @mock.patch.object(metrics.Sum, 'compute_through', SUM_COMPUTE_THROUGH)
  def test_leaf_caching(self, op):
    op = set_up_metric(op)
    SUM_COMPUTE_THROUGH.mock.reset_mock()
    op.compute_on(self.df)
    actual_call_ct = SUM_COMPUTE_THROUGH.mock.call_count
    expected_call_ct = (
        2 * op.n_replicates + 2 if isinstance(op, operations.Bootstrap) else 2
    )
    if isinstance(op, operations.Bootstrap) and op.unit:
      expected_call_ct += 2  # for precomputation

    self.assertEqual(actual_call_ct, expected_call_ct)
    self.assertEmpty(op.cache)

  @parameterized.named_parameters(*OPERATIONS)
  @mock.patch.object(metrics.Sum, 'compute_through', SUM_COMPUTE_THROUGH)
  def test_filter_at_different_levels(self, op):
    SUM_COMPUTE_THROUGH.mock.reset_mock()
    op = set_up_metric(op)
    no_filter = copy.deepcopy(op)
    op1 = copy.deepcopy(op)
    op2 = copy.deepcopy(op)
    op3 = copy.deepcopy(op)
    op4 = copy.deepcopy(op)
    f1 = 'x > 0.2'
    f2 = 'y > 0.3'
    op1.where = f2
    op1.children[0].where = f1
    op2.where = f1
    op2.children[0].where = f2
    op3.where = (f1, f2)
    op4.children[0].where = (f2, f1)
    m = metrics.MetricList((op1, op2, op3, op4))
    output = m.compute_on(self.df)
    actual_call_ct = SUM_COMPUTE_THROUGH.mock.call_count
    expected = pd.concat(
        [no_filter.compute_on(self.df[(self.df.x > 0.2) & (self.df.y > 0.3)])]
        * 4,
        axis=1,
    )

    self.assertEqual(actual_call_ct, 2)
    testing.assert_frame_equal(output, expected)
    self.assertEmpty(m.cache)

  def test_different_metrics_have_different_fingerprints(self):
    distinct_ops = [
        operations.Distribution('x'),
        operations.Distribution('y'),
        operations.CumulativeDistribution('x'),
        operations.CumulativeDistribution('y'),
        operations.CumulativeDistribution('x', order='foo'),
        operations.CumulativeDistribution('x', ascending=False),
        operations.PercentChange('x', 'y'),
        operations.PercentChange('z', 'y'),
        operations.PercentChange('x', 'z'),
        operations.PercentChange('x', 'y', include_base=True),
        operations.AbsoluteChange('x', 'y'),
        operations.AbsoluteChange('z', 'y'),
        operations.AbsoluteChange('x', 'z'),
        operations.AbsoluteChange('x', 'y', include_base=True),
        operations.PrePostChange(
            'x', 'y', covariates=metrics.Count('x'), stratified_by='q'
        ),
        operations.PrePostChange(
            'x', 'y', covariates=metrics.Count('x'), stratified_by='w'
        ),
        operations.PrePostChange(
            'b', 'y', covariates=metrics.Count('x'), stratified_by='q'
        ),
        operations.PrePostChange(
            'x', 'a', covariates=metrics.Count('x'), stratified_by='q'
        ),
        operations.CUPED(
            'x', 'y', covariates=metrics.Count('x'), stratified_by='q'
        ),
        operations.CUPED(
            'x', 'y', covariates=metrics.Count('x'), stratified_by='w'
        ),
        operations.CUPED(
            'b', 'y', covariates=metrics.Count('x'), stratified_by='q'
        ),
        operations.CUPED(
            'x', 'a', covariates=metrics.Count('x'), stratified_by='q'
        ),
        operations.Jackknife('x'),
        operations.Jackknife('y'),
        operations.Jackknife('x', confidence=0.9),
        operations.Jackknife('x', confidence=0.95),
        operations.Bootstrap(None),
        operations.Bootstrap('x'),
        operations.Bootstrap('x', n_replicates=10),
        operations.Bootstrap('x', confidence=0.9),
        operations.Bootstrap('x', confidence=0.95),
        diversity.HHI('x'),
        diversity.HHI('y'),
        diversity.Entropy('x'),
        diversity.Entropy('y'),
        diversity.TopK('x', 1),
        diversity.TopK('x', 2),
        diversity.TopK('y', 1),
        diversity.Nxx('x', 0.1),
        diversity.Nxx('x', 0.2),
        diversity.Nxx('y', 0.1),
    ]
    fingerprints = set(
        [m(metrics.Ratio('x', 'y')).get_fingerprint() for m in distinct_ops]
    )
    self.assertLen(fingerprints, len(distinct_ops))


if __name__ == '__main__':
  absltest.main()
