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

from meterstick import metrics
from meterstick import operations
import mock
import numpy as np
import pandas as pd
from pandas import testing
import unittest


class MetricTest(unittest.TestCase):
  """Tests general features of Metric."""

  df = pd.DataFrame({'X': [0, 1, 2, 3], 'Y': [0, 1, 1, 2]})

  def test_precompute(self):
    metric = metrics.Metric(
        'foo',
        precompute=lambda df, split_by: df[split_by],
        compute=lambda x: x.sum().values[0])
    output = metric.compute_on(self.df, 'Y')
    expected = pd.DataFrame({'foo': [0, 2, 2]}, index=range(3))
    expected.index.name = 'Y'
    testing.assert_frame_equal(output, expected)

  def test_compute(self):
    metric = metrics.Metric('foo', compute=lambda x: x['X'].sum())
    output = metric.compute_on(self.df)
    expected = metrics.Sum('X', 'foo').compute_on(self.df)
    testing.assert_frame_equal(output, expected)

  def test_postcompute(self):
    def postcompute(values, split_by):
      del split_by
      return values / values.sum()

    output = metrics.Sum('X', postcompute=postcompute).compute_on(self.df, 'Y')
    expected = operations.Distribution('Y',
                                       metrics.Sum('X')).compute_on(self.df)
    expected.columns = ['sum(X)']
    testing.assert_frame_equal(output.astype(float), expected)

  def test_compute_slices(self):

    def _sum(df, split_by):
      if split_by:
        df = df.groupby(split_by)
      return df['X'].sum()

    metric = metrics.Metric('foo', compute_slices=_sum)
    output = metric.compute_on(self.df)
    expected = metrics.Sum('X', 'foo').compute_on(self.df)
    testing.assert_frame_equal(output, expected)

  def test_final_compute(self):
    metric = metrics.Metric(
        'foo', compute=lambda x: x, final_compute=lambda *_: 2)
    output = metric.compute_on(None)
    self.assertEqual(output, 2)

  def test_pipeline_operator(self):
    m = metrics.Count('X')
    testing.assert_frame_equal(
        m.compute_on(self.df), m | metrics.compute_on(self.df))


class SimpleMetricTest(unittest.TestCase):

  df = pd.DataFrame({
      'X': [1, 1, 1, 2, 2, 3, 4],
      'Y': [3, 1, 1, 4, 4, 3, 5],
      'grp': ['A'] * 3 + ['B'] * 4
  })

  def test_list_where(self):
    metric = metrics.Mean('X', where=['grp == "A"'])
    output = metric.compute_on(self.df, return_dataframe=False)
    expected = self.df.query('grp == "A"')['X'].mean()
    self.assertEqual(output, expected)

  def test_split_data(self):
    df = pd.DataFrame({'X': [1, 2], 'grp': ['a', 'b']}, index=[1, 1])
    output = sorted(metrics.Metric.split_data(df, 'grp'), key=lambda x: x[1])
    testing.assert_frame_equal(output[0][0], df[:1])
    self.assertEqual(output[0][1], 'a')
    testing.assert_frame_equal(output[1][0], df[1:])
    self.assertEqual(output[1][1], 'b')

  def test_single_list_where(self):
    metric = metrics.Mean('X', where=['grp == "A"', 'Y < 2'])
    output = metric.compute_on(self.df, return_dataframe=False)
    expected = self.df.query('grp == "A" and Y < 2')['X'].mean()
    self.assertEqual(output, expected)

  def test_count_not_df(self):
    metric = metrics.Count('X')
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, 7)

  def test_count_split_by_not_df(self):
    metric = metrics.Count('X')
    output = metric.compute_on(self.df, 'grp', return_dataframe=False)
    expected = self.df.groupby('grp')['X'].count()
    expected.name = 'count(X)'
    testing.assert_series_equal(output, expected)

  def test_count_where(self):
    metric = metrics.Count('X', where='grp == "A"')
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, 3)

  def test_count_with_nan(self):
    df = pd.DataFrame({'X': [1, 1, np.nan, 2, 2, 3, 4]})
    metric = metrics.Count('X')
    output = metric.compute_on(df, return_dataframe=False)
    self.assertEqual(output, 6)

  def test_count_unmelted(self):
    metric = metrics.Count('X')
    output = metric.compute_on(self.df)
    expected = pd.DataFrame({'count(X)': [7]})
    testing.assert_frame_equal(output, expected)

  def test_count_melted(self):
    metric = metrics.Count('X')
    output = metric.compute_on(self.df, melted=True)
    expected = pd.DataFrame({'Value': [7]}, index=['count(X)'])
    expected.index.name = 'Metric'
    testing.assert_frame_equal(output, expected)

  def test_count_split_by_unmelted(self):
    metric = metrics.Count('X')
    output = metric.compute_on(self.df, 'grp')
    expected = pd.DataFrame({'count(X)': [3, 4]}, index=['A', 'B'])
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_count_split_by_melted(self):
    metric = metrics.Count('X')
    output = metric.compute_on(self.df, 'grp', melted=True)
    expected = pd.DataFrame({
        'Value': [3, 4],
        'grp': ['A', 'B']
    },
                            index=['count(X)', 'count(X)'])
    expected.index.name = 'Metric'
    expected.set_index('grp', append=True, inplace=True)
    testing.assert_frame_equal(output, expected)

  def test_count_distinct(self):
    df = pd.DataFrame({'X': [1, 1, np.nan, 2, 2, 3]})
    metric = metrics.Count('X', distinct=True)
    output = metric.compute_on(df, return_dataframe=False)
    self.assertEqual(output, 3)

  def test_sum_not_df(self):
    metric = metrics.Sum('X')
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, 14)

  def test_sum_split_by_not_df(self):
    metric = metrics.Sum('X')
    output = metric.compute_on(self.df, 'grp', return_dataframe=False)
    expected = self.df.groupby('grp')['X'].sum()
    expected.name = 'sum(X)'
    testing.assert_series_equal(output, expected)

  def test_sum_where(self):
    metric = metrics.Sum('X', where='grp == "A"')
    output = metric.compute_on(self.df, return_dataframe=False)
    expected = self.df.query('grp == "A"')['X'].sum()
    self.assertEqual(output, expected)

  def test_sum_unmelted(self):
    metric = metrics.Sum('X')
    output = metric.compute_on(self.df)
    expected = pd.DataFrame({'sum(X)': [14]})
    testing.assert_frame_equal(output, expected)

  def test_sum_melted(self):
    metric = metrics.Sum('X')
    output = metric.compute_on(self.df, melted=True)
    expected = pd.DataFrame({'Value': [14]}, index=['sum(X)'])
    expected.index.name = 'Metric'
    testing.assert_frame_equal(output, expected)

  def test_sum_split_by_unmelted(self):
    metric = metrics.Sum('X')
    output = metric.compute_on(self.df, 'grp')
    expected = pd.DataFrame({'sum(X)': [3, 11]}, index=['A', 'B'])
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_sum_split_by_melted(self):
    metric = metrics.Sum('X')
    output = metric.compute_on(self.df, 'grp', melted=True)
    expected = pd.DataFrame({
        'Value': [3, 11],
        'grp': ['A', 'B']
    },
                            index=['sum(X)', 'sum(X)'])
    expected.index.name = 'Metric'
    expected.set_index('grp', append=True, inplace=True)
    testing.assert_frame_equal(output, expected)

  def test_dot_not_df(self):
    metric = metrics.Dot('X', 'Y')
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, sum(self.df.X * self.df.Y))

  def test_dot_split_by_not_df(self):
    metric = metrics.Dot('X', 'Y')
    output = metric.compute_on(self.df, 'grp', return_dataframe=False)
    self.df['X * Y'] = self.df.X * self.df.Y
    expected = self.df.groupby('grp')['X * Y'].sum()
    expected.name = 'sum(X * Y)'
    testing.assert_series_equal(output, expected)

  def test_dot_where(self):
    metric = metrics.Dot('X', 'Y', where='grp == "A"')
    output = metric.compute_on(self.df, return_dataframe=False)
    d = self.df.query('grp == "A"')
    self.assertEqual(output, sum(d.X * d.Y))

  def test_dot_unmelted(self):
    metric = metrics.Dot('X', 'Y')
    output = metric.compute_on(self.df)
    expected = pd.DataFrame({'sum(X * Y)': [sum(self.df.X * self.df.Y)]})
    testing.assert_frame_equal(output, expected)

  def test_dot_normalized(self):
    metric = metrics.Dot('X', 'Y', True)
    output = metric.compute_on(self.df)
    expected = pd.DataFrame({'mean(X * Y)': [(self.df.X * self.df.Y).mean()]})
    testing.assert_frame_equal(output, expected)

  def test_dot_melted(self):
    metric = metrics.Dot('X', 'Y')
    output = metric.compute_on(self.df, melted=True)
    expected = pd.DataFrame({'Value': [sum(self.df.X * self.df.Y)]},
                            index=['sum(X * Y)'])
    expected.index.name = 'Metric'
    testing.assert_frame_equal(output, expected)

  def test_dot_split_by_unmelted(self):
    metric = metrics.Dot('X', 'Y')
    output = metric.compute_on(self.df, 'grp')
    expected = pd.DataFrame({'sum(X * Y)': [5, 45]}, index=['A', 'B'])
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_dot_split_by_melted(self):
    metric = metrics.Dot('X', 'Y')
    output = metric.compute_on(self.df, 'grp', melted=True)
    expected = pd.DataFrame({
        'Value': [5, 45],
        'grp': ['A', 'B']
    },
                            index=['sum(X * Y)', 'sum(X * Y)'])
    expected.index.name = 'Metric'
    expected.set_index('grp', append=True, inplace=True)
    testing.assert_frame_equal(output, expected)

  def test_mean_not_df(self):
    metric = metrics.Mean('X')
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, 2)

  def test_mean_split_by_not_df(self):
    metric = metrics.Mean('X')
    output = metric.compute_on(self.df, 'grp', return_dataframe=False)
    expected = self.df.groupby('grp')['X'].mean()
    expected.name = 'mean(X)'
    testing.assert_series_equal(output, expected)

  def test_mean_where(self):
    metric = metrics.Mean('X', where='grp == "A"')
    output = metric.compute_on(self.df, return_dataframe=False)
    expected = self.df.query('grp == "A"')['X'].mean()
    self.assertEqual(output, expected)

  def test_mean_unmelted(self):
    metric = metrics.Mean('X')
    output = metric.compute_on(self.df)
    expected = pd.DataFrame({'mean(X)': [2.]})
    testing.assert_frame_equal(output, expected)

  def test_mean_melted(self):
    metric = metrics.Mean('X')
    output = metric.compute_on(self.df, melted=True)
    expected = pd.DataFrame({'Value': [2.]}, index=['mean(X)'])
    expected.index.name = 'Metric'
    testing.assert_frame_equal(output, expected)

  def test_mean_split_by_unmelted(self):
    metric = metrics.Mean('X')
    output = metric.compute_on(self.df, 'grp')
    expected = pd.DataFrame({'mean(X)': [1, 2.75]}, index=['A', 'B'])
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_mean_split_by_melted(self):
    metric = metrics.Mean('X')
    output = metric.compute_on(self.df, 'grp', melted=True)
    expected = pd.DataFrame({
        'Value': [1, 2.75],
        'grp': ['A', 'B']
    },
                            index=['mean(X)', 'mean(X)'])
    expected.index.name = 'Metric'
    expected.set_index('grp', append=True, inplace=True)
    testing.assert_frame_equal(output, expected)

  def test_max(self):
    metric = metrics.Max('X')
    output = metric.compute_on(self.df)
    expected = pd.DataFrame({'max(X)': [4]})
    testing.assert_frame_equal(output, expected)

  def test_min(self):
    metric = metrics.Min('X')
    output = metric.compute_on(self.df)
    expected = pd.DataFrame({'min(X)': [1]})
    testing.assert_frame_equal(output, expected)

  def test_weighted_mean_not_df(self):
    df = pd.DataFrame({'X': [1, 2], 'Y': [3, 1]})
    metric = metrics.Mean('X', 'Y')
    output = metric.compute_on(df, return_dataframe=False)
    self.assertEqual(output, 1.25)

  def test_weighted_mean_split_by_not_df(self):
    df = pd.DataFrame({
        'X': [1, 2, 1, 3],
        'Y': [3, 1, 0, 1],
        'grp': ['A', 'A', 'B', 'B']
    })
    metric = metrics.Mean('X', 'Y')
    output = metric.compute_on(df, 'grp', return_dataframe=False)
    output.sort_index(level='grp', inplace=True)  # For Py2
    expected = pd.Series((1.25, 3.), index=['A', 'B'])
    expected.index.name = 'grp'
    expected.name = 'Y-weighted mean(X)'
    testing.assert_series_equal(output, expected)

  def test_weighted_mean_unmelted(self):
    df = pd.DataFrame({'X': [1, 2], 'Y': [3, 1]})
    metric = metrics.Mean('X', 'Y')
    output = metric.compute_on(df)
    expected = pd.DataFrame({'Y-weighted mean(X)': [1.25]})
    testing.assert_frame_equal(output, expected)

  def test_weighted_mean_melted(self):
    df = pd.DataFrame({'X': [1, 2], 'Y': [3, 1]})
    metric = metrics.Mean('X', 'Y')
    output = metric.compute_on(df, melted=True)
    expected = pd.DataFrame({'Value': [1.25]}, index=['Y-weighted mean(X)'])
    expected.index.name = 'Metric'
    testing.assert_frame_equal(output, expected)

  def test_weighted_mean_split_by_unmelted(self):
    df = pd.DataFrame({
        'X': [1, 2, 1, 3],
        'Y': [3, 1, 0, 1],
        'grp': ['A', 'A', 'B', 'B']
    })
    metric = metrics.Mean('X', 'Y')
    output = metric.compute_on(df, 'grp')
    output.sort_index(level='grp', inplace=True)  # For Py2
    expected = pd.DataFrame({'Y-weighted mean(X)': [1.25, 3.]},
                            index=['A', 'B'])
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_weighted_mean_split_by_melted(self):
    df = pd.DataFrame({
        'X': [1, 2, 1, 3],
        'Y': [3, 1, 0, 1],
        'grp': ['A', 'A', 'B', 'B']
    })
    metric = metrics.Mean('X', 'Y')
    output = metric.compute_on(df, 'grp', melted=True)
    output.sort_index(level='grp', inplace=True)  # For Py2
    expected = pd.DataFrame({
        'Value': [1.25, 3.],
        'grp': ['A', 'B']
    },
                            index=['Y-weighted mean(X)', 'Y-weighted mean(X)'])
    expected.index.name = 'Metric'
    expected.set_index('grp', append=True, inplace=True)
    testing.assert_frame_equal(output, expected)

  def test_quantile_raise(self):
    with self.assertRaises(ValueError) as cm:
      metrics.Quantile('X', 2)
    self.assertEqual(str(cm.exception), 'quantiles must be in [0, 1].')

  def test_quantile_multiple_quantiles_raise(self):
    with self.assertRaises(ValueError) as cm:
      metrics.Quantile('X', [0.1, 2])
    self.assertEqual(str(cm.exception), 'quantiles must be in [0, 1].')

  def test_quantile_not_df(self):
    metric = metrics.Quantile('X', 0.5)
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, 2)

  def test_quantile_where(self):
    metric = metrics.Quantile('X', where='grp == "B"')
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, 2.5)

  def test_quantile_interpolation(self):
    metric = metrics.Quantile('X', 0.5, interpolation='lower')
    output = metric.compute_on(
        pd.DataFrame({'X': [1, 2]}), return_dataframe=False)
    self.assertEqual(output, 1)

  def test_quantile_split_by_not_df(self):
    metric = metrics.Quantile('X', 0.5)
    output = metric.compute_on(self.df, 'grp', return_dataframe=False)
    expected = self.df.groupby('grp')['X'].quantile(0.5)
    expected.name = 'quantile(X, 0.5)'
    testing.assert_series_equal(output, expected)

  def test_quantile_unmelted(self):
    metric = metrics.Quantile('X', 0.5)
    output = metric.compute_on(self.df)
    expected = pd.DataFrame({'quantile(X, 0.5)': [2.]})
    testing.assert_frame_equal(output, expected)

  def test_quantile_melted(self):
    metric = metrics.Quantile('X', 0.5)
    output = metric.compute_on(self.df, melted=True)
    expected = pd.DataFrame({'Value': [2.]}, index=['quantile(X, 0.5)'])
    expected.index.name = 'Metric'
    testing.assert_frame_equal(output, expected)

  def test_quantile_split_by_unmelted(self):
    metric = metrics.Quantile('X', 0.5)
    output = metric.compute_on(self.df, 'grp')
    expected = pd.DataFrame({'quantile(X, 0.5)': [1, 2.5]}, index=['A', 'B'])
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_quantile_split_by_melted(self):
    metric = metrics.Quantile('X', 0.5)
    output = metric.compute_on(self.df, 'grp', melted=True)
    expected = pd.DataFrame({
        'Value': [1, 2.5],
        'grp': ['A', 'B']
    },
                            index=['quantile(X, 0.5)'] * 2)
    expected.index.name = 'Metric'
    expected.set_index('grp', append=True, inplace=True)
    testing.assert_frame_equal(output, expected)

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

  def test_variance_not_df(self):
    metric = metrics.Variance('X')
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, self.df.X.var())

  def test_variance_biased(self):
    metric = metrics.Variance('X', False)
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, self.df.X.var(ddof=0))

  def test_variance_split_by_not_df(self):
    metric = metrics.Variance('X')
    output = metric.compute_on(self.df, 'grp', return_dataframe=False)
    expected = self.df.groupby('grp')['X'].var()
    expected.name = 'var(X)'
    testing.assert_series_equal(output, expected)

  def test_variance_where(self):
    metric = metrics.Variance('X', where='grp == "B"')
    output = metric.compute_on(self.df, return_dataframe=False)
    expected = self.df.query('grp == "B"')['X'].var()
    self.assertEqual(output, expected)

  def test_variance_unmelted(self):
    metric = metrics.Variance('X')
    output = metric.compute_on(self.df)
    expected = pd.DataFrame({'var(X)': [self.df.X.var()]})
    testing.assert_frame_equal(output, expected)

  def test_variance_melted(self):
    metric = metrics.Variance('X')
    output = metric.compute_on(self.df, melted=True)
    expected = pd.DataFrame({'Value': [self.df.X.var()]}, index=['var(X)'])
    expected.index.name = 'Metric'
    testing.assert_frame_equal(output, expected)

  def test_variance_split_by_unmelted(self):
    metric = metrics.Variance('X')
    output = metric.compute_on(self.df, 'grp')
    expected = pd.DataFrame({'var(X)': self.df.groupby('grp')['X'].var()},
                            index=['A', 'B'])
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_variance_split_by_melted(self):
    metric = metrics.Variance('X')
    output = metric.compute_on(self.df, 'grp', melted=True)
    expected = pd.DataFrame(
        {
            'Value': self.df.groupby('grp')['X'].var().values,
            'grp': ['A', 'B']
        },
        index=['var(X)', 'var(X)'])
    expected.index.name = 'Metric'
    expected.set_index('grp', append=True, inplace=True)
    testing.assert_frame_equal(output, expected)

  def test_weighted_variance_not_df(self):
    df = pd.DataFrame({'X': [0, 2], 'Y': [1, 3]})
    metric = metrics.Variance('X', weight='Y')
    output = metric.compute_on(df, return_dataframe=False)
    self.assertEqual(output, 1)

  def test_weighted_variance_not_df_biased(self):
    df = pd.DataFrame({'X': [0, 2], 'Y': [1, 3]})
    metric = metrics.Variance('X', False, 'Y')
    output = metric.compute_on(df, return_dataframe=False)
    self.assertEqual(output, 0.75)

  def test_weighted_variance_split_by_not_df(self):
    df = pd.DataFrame({
        'X': [0, 2, 1, 3],
        'Y': [1, 3, 1, 1],
        'grp': ['B', 'B', 'A', 'A']
    })
    metric = metrics.Variance('X', weight='Y')
    output = metric.compute_on(df, 'grp', return_dataframe=False)
    output.sort_index(level='grp', inplace=True)  # For Py2
    expected = pd.Series((2., 1), index=['A', 'B'])
    expected.index.name = 'grp'
    expected.name = 'Y-weighted var(X)'
    testing.assert_series_equal(output, expected)

  def test_weighted_variance_unmelted(self):
    df = pd.DataFrame({'X': [0, 2], 'Y': [1, 3]})
    metric = metrics.Variance('X', weight='Y')
    output = metric.compute_on(df)
    expected = pd.DataFrame({'Y-weighted var(X)': [1.]})
    testing.assert_frame_equal(output, expected)

  def test_weighted_variance_melted(self):
    df = pd.DataFrame({'X': [0, 2], 'Y': [1, 3]})
    metric = metrics.Variance('X', weight='Y')
    output = metric.compute_on(df, melted=True)
    expected = pd.DataFrame({'Value': [1.]}, index=['Y-weighted var(X)'])
    expected.index.name = 'Metric'
    testing.assert_frame_equal(output, expected)

  def test_weighted_variance_split_by_unmelted(self):
    df = pd.DataFrame({
        'X': [0, 2, 1, 3],
        'Y': [1, 3, 1, 1],
        'grp': ['B', 'B', 'A', 'A']
    })
    metric = metrics.Variance('X', weight='Y')
    output = metric.compute_on(df, 'grp')
    output.sort_index(level='grp', inplace=True)  # For Py2
    expected = pd.DataFrame({'Y-weighted var(X)': [2., 1]}, index=['A', 'B'])
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_weighted_variance_split_by_melted(self):
    df = pd.DataFrame({
        'X': [0, 2, 1, 3],
        'Y': [1, 3, 1, 1],
        'grp': ['B', 'B', 'A', 'A']
    })
    metric = metrics.Variance('X', weight='Y')
    output = metric.compute_on(df, 'grp', melted=True)
    output.sort_index(level='grp', inplace=True)  # For Py2
    expected = pd.DataFrame({
        'Value': [2., 1],
        'grp': ['A', 'B']
    },
                            index=['Y-weighted var(X)', 'Y-weighted var(X)'])
    expected.index.name = 'Metric'
    expected.set_index('grp', append=True, inplace=True)
    testing.assert_frame_equal(output, expected)

  def test_standard_deviation_not_df(self):
    metric = metrics.StandardDeviation('X')
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, self.df.X.std())

  def test_standard_deviation_biased(self):
    metric = metrics.StandardDeviation('X', False)
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, self.df.X.std(ddof=0))

  def test_standard_deviation_split_by_not_df(self):
    metric = metrics.StandardDeviation('X')
    output = metric.compute_on(self.df, 'grp', return_dataframe=False)
    expected = self.df.groupby('grp')['X'].std()
    expected.name = 'sd(X)'
    testing.assert_series_equal(output, expected)

  def test_standard_deviation_where(self):
    metric = metrics.StandardDeviation('X', where='grp == "B"')
    output = metric.compute_on(self.df, return_dataframe=False)
    expected = self.df.query('grp == "B"')['X'].std()
    self.assertEqual(output, expected)

  def test_standard_deviation_unmelted(self):
    metric = metrics.StandardDeviation('X')
    output = metric.compute_on(self.df)
    expected = pd.DataFrame({'sd(X)': [self.df.X.std()]})
    testing.assert_frame_equal(output, expected)

  def test_standard_deviation_melted(self):
    metric = metrics.StandardDeviation('X')
    output = metric.compute_on(self.df, melted=True)
    expected = pd.DataFrame({'Value': [self.df.X.std()]}, index=['sd(X)'])
    expected.index.name = 'Metric'
    testing.assert_frame_equal(output, expected)

  def test_standard_deviation_split_by_unmelted(self):
    metric = metrics.StandardDeviation('X')
    output = metric.compute_on(self.df, 'grp')
    expected = pd.DataFrame({'sd(X)': self.df.groupby('grp')['X'].std()},
                            index=['A', 'B'])
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_standard_deviation_split_by_melted(self):
    metric = metrics.StandardDeviation('X')
    output = metric.compute_on(self.df, 'grp', melted=True)
    expected = pd.DataFrame(
        {
            'Value': self.df.groupby('grp')['X'].std().values,
            'grp': ['A', 'B']
        },
        index=['sd(X)', 'sd(X)'])
    expected.index.name = 'Metric'
    expected.set_index('grp', append=True, inplace=True)
    testing.assert_frame_equal(output, expected)

  def test_weighted_standard_deviation_not_df(self):
    df = pd.DataFrame({'X': [0, 2], 'Y': [1, 3]})
    metric = metrics.StandardDeviation('X', weight='Y')
    output = metric.compute_on(df, return_dataframe=False)
    self.assertEqual(output, 1)

  def test_weighted_standard_deviation_not_df_biased(self):
    df = pd.DataFrame({'X': [0, 2], 'Y': [1, 3]})
    metric = metrics.StandardDeviation('X', False, 'Y')
    output = metric.compute_on(df, return_dataframe=False)
    self.assertEqual(output, np.sqrt(0.75))

  def test_weighted_standard_deviation_split_by_not_df(self):
    df = pd.DataFrame({
        'X': [0, 2, 1, 3],
        'Y': [1, 3, 1, 1],
        'grp': ['B', 'B', 'A', 'A']
    })
    metric = metrics.StandardDeviation('X', weight='Y')
    output = metric.compute_on(df, 'grp', return_dataframe=False)
    output.sort_index(level='grp', inplace=True)  # For Py2
    expected = pd.Series((np.sqrt(2), 1), index=['A', 'B'])
    expected.index.name = 'grp'
    expected.name = 'Y-weighted sd(X)'
    testing.assert_series_equal(output, expected)

  def test_weighted_standard_deviation_unmelted(self):
    df = pd.DataFrame({'X': [0, 2], 'Y': [1, 3]})
    metric = metrics.StandardDeviation('X', weight='Y')
    output = metric.compute_on(df)
    expected = pd.DataFrame({'Y-weighted sd(X)': [1.]})
    testing.assert_frame_equal(output, expected)

  def test_weighted_standard_deviation_melted(self):
    df = pd.DataFrame({'X': [0, 2], 'Y': [1, 3]})
    metric = metrics.StandardDeviation('X', weight='Y')
    output = metric.compute_on(df, melted=True)
    expected = pd.DataFrame({'Value': [1.]}, index=['Y-weighted sd(X)'])
    expected.index.name = 'Metric'
    testing.assert_frame_equal(output, expected)

  def test_weighted_standard_deviation_split_by_unmelted(self):
    df = pd.DataFrame({
        'X': [0, 2, 1, 3],
        'Y': [1, 3, 1, 1],
        'grp': ['B', 'B', 'A', 'A']
    })
    metric = metrics.StandardDeviation('X', weight='Y')
    output = metric.compute_on(df, 'grp')
    output.sort_index(level='grp', inplace=True)  # For Py2
    expected = pd.DataFrame({'Y-weighted sd(X)': [np.sqrt(2), 1]},
                            index=['A', 'B'])
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_weighted_standard_deviation_split_by_melted(self):
    df = pd.DataFrame({
        'X': [0, 2, 1, 3],
        'Y': [1, 3, 1, 1],
        'grp': ['B', 'B', 'A', 'A']
    })
    metric = metrics.StandardDeviation('X', weight='Y')
    output = metric.compute_on(df, 'grp', melted=True)
    output.sort_index(level='grp', inplace=True)  # For Py2
    expected = pd.DataFrame({
        'Value': [np.sqrt(2), 1],
        'grp': ['A', 'B']
    },
                            index=['Y-weighted sd(X)', 'Y-weighted sd(X)'])
    expected.index.name = 'Metric'
    expected.set_index('grp', append=True, inplace=True)
    testing.assert_frame_equal(output, expected)

  def test_cv_not_df(self):
    metric = metrics.CV('X')
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, np.sqrt(1 / 3.))

  def test_cv_biased(self):
    metric = metrics.CV('X', False)
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, self.df.X.std(ddof=0) / np.mean(self.df.X))

  def test_cv_split_by_not_df(self):
    metric = metrics.CV('X')
    output = metric.compute_on(self.df, 'grp', return_dataframe=False)
    expected = self.df.groupby('grp')['X'].std() / [1, 2.75]
    expected.name = 'cv(X)'
    testing.assert_series_equal(output, expected)

  def test_cv_where(self):
    metric = metrics.CV('X', where='grp == "B"')
    output = metric.compute_on(self.df, return_dataframe=False)
    expected = self.df.query('grp == "B"')['X'].std() / 2.75
    self.assertEqual(output, expected)

  def test_cv_unmelted(self):
    metric = metrics.CV('X')
    output = metric.compute_on(self.df)
    expected = pd.DataFrame({'cv(X)': [np.sqrt(1 / 3.)]})
    testing.assert_frame_equal(output, expected)

  def test_cv_melted(self):
    metric = metrics.CV('X')
    output = metric.compute_on(self.df, melted=True)
    expected = pd.DataFrame({'Value': [np.sqrt(1 / 3.)]}, index=['cv(X)'])
    expected.index.name = 'Metric'
    testing.assert_frame_equal(output, expected)

  def test_cv_split_by_unmelted(self):
    metric = metrics.CV('X')
    output = metric.compute_on(self.df, 'grp')
    expected = pd.DataFrame({'cv(X)': [0, np.sqrt(1 / 8.25)]}, index=['A', 'B'])
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_cv_split_by_melted(self):
    metric = metrics.CV('X')
    output = metric.compute_on(self.df, 'grp', melted=True)
    expected = pd.DataFrame(
        data={
            'Value': [0, np.sqrt(1 / 8.25)],
            'grp': ['A', 'B']
        },
        index=['cv(X)', 'cv(X)'])
    expected.index.name = 'Metric'
    expected.set_index('grp', append=True, inplace=True)
    testing.assert_frame_equal(output, expected)

  def test_correlation(self):
    metric = metrics.Correlation('X', 'Y')
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, np.corrcoef(self.df.X, self.df.Y)[0, 1])
    self.assertEqual(output, self.df.X.corr(self.df.Y))

  def test_weighted_correlation(self):
    metric = metrics.Correlation('X', 'Y', weight='Y')
    output = metric.compute_on(self.df)
    cov = np.cov(self.df.X, self.df.Y, aweights=self.df.Y)
    expected = pd.DataFrame(
        {'Y-weighted corr(X, Y)': [cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])]})
    testing.assert_frame_equal(output, expected)

  def test_correlation_method(self):
    metric = metrics.Correlation('X', 'Y', method='kendall')
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, self.df.X.corr(self.df.Y, method='kendall'))

  def test_correlation_kwargs(self):
    metric = metrics.Correlation('X', 'Y', min_periods=10)
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertTrue(pd.isnull(output))

  def test_correlation_split_by_not_df(self):
    metric = metrics.Correlation('X', 'Y')
    output = metric.compute_on(self.df, 'grp', return_dataframe=False)
    corr_a = metric.compute_on(
        self.df[self.df.grp == 'A'], return_dataframe=False)
    corr_b = metric.compute_on(
        self.df[self.df.grp == 'B'], return_dataframe=False)
    expected = pd.Series([corr_a, corr_b], index=['A', 'B'])
    expected.index.name = 'grp'
    expected.name = 'corr(X, Y)'
    testing.assert_series_equal(output, expected)

  def test_correlation_where(self):
    metric = metrics.Correlation('X', 'Y', where='grp == "B"')
    metric_no_filter = metrics.Correlation('X', 'Y')
    output = metric.compute_on(self.df)
    expected = metric_no_filter.compute_on(self.df[self.df.grp == 'B'])
    testing.assert_frame_equal(output, expected)

  def test_correlation_df(self):
    metric = metrics.Correlation('X', 'Y')
    output = metric.compute_on(self.df)
    expected = pd.DataFrame({'corr(X, Y)': [self.df.X.corr(self.df.Y)]})
    testing.assert_frame_equal(output, expected)

  def test_correlation_split_by_df(self):
    df = pd.DataFrame({
        'X': [1, 1, 1, 2, 2, 2, 3, 4],
        'Y': [3, 1, 1, 3, 4, 4, 3, 5],
        'grp': ['A'] * 4 + ['B'] * 4
    })
    metric = metrics.Correlation('X', 'Y')
    output = metric.compute_on(df, 'grp')
    corr_a = metric.compute_on(df[df.grp == 'A'], return_dataframe=False)
    corr_b = metric.compute_on(df[df.grp == 'B'], return_dataframe=False)
    expected = pd.DataFrame({'corr(X, Y)': [corr_a, corr_b]}, index=['A', 'B'])
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)

  def test_cov(self):
    metric = metrics.Cov('X', 'Y')
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, np.cov(self.df.X, self.df.Y)[0, 1])
    self.assertEqual(output, self.df.X.cov(self.df.Y))

  def test_cov_bias(self):
    metric = metrics.Cov('X', 'Y', bias=True)
    output = metric.compute_on(self.df, return_dataframe=False)
    expected = np.mean(
        (self.df.X - self.df.X.mean()) * (self.df.Y - self.df.Y.mean()))
    self.assertEqual(output, expected)

  def test_cov_ddof(self):
    metric = metrics.Cov('X', 'Y', ddof=0)
    output = metric.compute_on(self.df, return_dataframe=False)
    expected = np.mean(
        (self.df.X - self.df.X.mean()) * (self.df.Y - self.df.Y.mean()))
    self.assertEqual(output, expected)

  def test_cov_kwargs(self):
    metric = metrics.Cov('X', 'Y', fweights=self.df.Y)
    output = metric.compute_on(self.df)
    expected = np.cov(self.df.X, self.df.Y, fweights=self.df.Y)[0, 1]
    expected = pd.DataFrame({'cov(X, Y)': [expected]})
    testing.assert_frame_equal(output, expected)

  def test_weighted_cov(self):
    metric = metrics.Cov('X', 'Y', weight='Y')
    output = metric.compute_on(self.df)
    expected = np.cov(self.df.X, self.df.Y, aweights=self.df.Y)[0, 1]
    expected = pd.DataFrame({'Y-weighted cov(X, Y)': [expected]})
    testing.assert_frame_equal(output, expected)

  def test_cov_split_by_not_df(self):
    metric = metrics.Cov('X', 'Y')
    output = metric.compute_on(self.df, 'grp', return_dataframe=False)
    output.sort_index(level='grp', inplace=True)  # For Py2
    cov_a = metric.compute_on(
        self.df[self.df.grp == 'A'], return_dataframe=False)
    cov_b = metric.compute_on(
        self.df[self.df.grp == 'B'], return_dataframe=False)
    expected = pd.Series([cov_a, cov_b], index=['A', 'B'])
    expected.index.name = 'grp'
    expected.name = 'cov(X, Y)'
    testing.assert_series_equal(output, expected)

  def test_cov_where(self):
    metric = metrics.Cov('X', 'Y', where='grp == "B"')
    metric_no_filter = metrics.Cov('X', 'Y')
    output = metric.compute_on(self.df)
    expected = metric_no_filter.compute_on(self.df[self.df.grp == 'B'])
    testing.assert_frame_equal(output, expected)

  def test_cov_df(self):
    metric = metrics.Cov('X', 'Y')
    output = metric.compute_on(self.df)
    expected = pd.DataFrame({'cov(X, Y)': [self.df.X.cov(self.df.Y)]})
    testing.assert_frame_equal(output, expected)

  def test_cov_split_by_df(self):
    df = pd.DataFrame({
        'X': [1, 1, 1, 2, 2, 2, 3, 4],
        'Y': [3, 1, 1, 3, 4, 4, 3, 5],
        'grp': ['A'] * 4 + ['B'] * 4
    })
    metric = metrics.Cov('X', 'Y')
    output = metric.compute_on(df, 'grp')
    output.sort_index(level='grp', inplace=True)  # For Py2
    cov_a = metric.compute_on(df[df.grp == 'A'], return_dataframe=False)
    cov_b = metric.compute_on(df[df.grp == 'B'], return_dataframe=False)
    expected = pd.DataFrame({'cov(X, Y)': [cov_a, cov_b]}, index=['A', 'B'])
    expected.index.name = 'grp'
    testing.assert_frame_equal(output, expected)


class CompositeMetric(unittest.TestCase):
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


class TestRatio(unittest.TestCase):
  df = pd.DataFrame({
      'X': [1, 1, 2],
      'Y': [-1, 2, 0],
      'grp0': ['A', 'A', 'B'],
      'grp': ['A', 'A', 'B']
  })
  metric = metrics.Ratio('X', 'Y', 'foo')

  def test_ratio_not_df(self):
    output = self.metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(4., output)

  def test_ratio_split_by_not_df(self):
    output = self.metric.compute_on(self.df, 'grp', return_dataframe=False)
    expected = pd.Series((2.0, float('inf')),
                         index=pd.Index(('A', 'B'), name='grp'))
    expected.name = 'foo'
    testing.assert_series_equal(output, expected)

  def test_ratio_split_by_muliple(self):
    output = self.metric.compute_on(self.df, ['grp0', 'grp'])
    expected = pd.DataFrame({
        'foo': (2.0, float('inf')),
        'grp0': ('A', 'B'),
        'grp': ('A', 'B')
    })
    expected.set_index(['grp0', 'grp'], inplace=True)
    testing.assert_frame_equal(output, expected)

  def test_ratio_where(self):
    metric = metrics.Ratio('X', 'Y', where='grp == "A"')
    output = metric.compute_on(self.df, return_dataframe=False)
    self.assertEqual(output, 2.0)


class TestMetricList(unittest.TestCase):

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


class TestCaching(unittest.TestCase):

  df = pd.DataFrame({'X': [0, 1], 'grp': ['A', 'B']})

  def test_simple_metric_caching(self):
    m = metrics.Count('X')
    with mock.patch.object(m, 'compute_through', autospec=True) as mock_fn:
      m.compute_on(self.df)
      m.compute_on(self.df)
      self.assertEqual(2, mock_fn.call_count)
    m = metrics.Count('X')
    with mock.patch.object(
        m, 'compute_through', return_value='a', autospec=True) as mock_fn:
      m.compute_on(self.df, cache_key='foo')
      output = m.compute_on(self.df, cache_key='foo', return_dataframe=False)
      mock_fn.assert_called_once_with(self.df, [])
      self.assertEqual('a', output)
      self.assertTrue(m.in_cache('foo'))
      self.assertEqual('a', m.get_cached('foo'))

  def test_simple_metric_flush_cache(self):
    m = metrics.Count('X')
    m.compute_on(self.df, cache_key=42)
    m.flush_cache(42)
    self.assertFalse(m.in_cache(42))

  def test_flush_cache_when_fail(self):
    sum_x = metrics.Sum('X')
    sum_fail = metrics.Sum('fail')
    try:
      metrics.MetricList((sum_x, sum_fail)).compute_on(self.df)
    except KeyError:
      pass
    self.assertEqual(sum_x.cache, {})
    self.assertEqual(sum_x.tmp_cache_keys, set())
    self.assertIsNone(sum_x.cache_key)
    self.assertEqual(sum_fail.cache, {})
    self.assertEqual(sum_fail.tmp_cache_keys, set())
    self.assertIsNone(sum_fail.cache_key)

  def test_simple_metric_caching_split_by(self):
    m = metrics.Count('X')
    with mock.patch.object(m, 'compute_through', autospec=True) as mock_fn:
      m.compute_on(self.df, 'grp')
      m.compute_on(self.df, ['grp'])
      self.assertEqual(2, mock_fn.call_count)
    m = metrics.Count('X')
    with mock.patch.object(
        m, 'compute_through', return_value='a', autospec=True) as mock_fn:
      m.compute_on(self.df, 'grp', cache_key='foo')
      output = m.compute_on(
          self.df, ['grp'], cache_key='foo', return_dataframe=False)
      mock_fn.assert_called_once_with(self.df, ['grp'])
      self.assertEqual('a', output)
      self.assertTrue(m.in_cache('foo', 'grp'))
      self.assertEqual('a', m.get_cached('foo', 'grp'))

  def test_simple_metric_flush_cache_split_by(self):
    m = metrics.Count('X')
    m.compute_on(self.df, 'grp', cache_key=42)
    self.assertTrue(m.in_cache(42, 'grp'))
    m.flush_cache(42, 'grp')
    self.assertFalse(m.in_cache(42, 'grp'))

  def test_one_time_caching(self):
    m = metrics.Count('X')
    key = (42, 'foo')
    m.compute_on(self.df, cache_key=key)
    self.assertTrue(m.in_cache(key))
    self.assertEqual(2, m.get_cached(key))

  def test_simple_metric_cache_key(self):
    m = metrics.Count('X')
    m.compute_on(self.df, cache_key=42)
    self.assertTrue(m.in_cache(42))
    self.assertEqual(2, m.get_cached(42))
    output = m.compute_on(None, cache_key=42, return_dataframe=False)
    self.assertEqual(2, output)

  def test_metriclist_internal_caching(self):
    sum_x = metrics.Sum('X')
    double_sum_x = 2 * metrics.Sum('X')
    m = metrics.MetricList((sum_x, double_sum_x))
    with mock.patch.object(sum_x, 'compute_through', autospec=True) as mock_fn:
      m.compute_on(self.df, return_dataframe=False)
      mock_fn.assert_called_once_with(self.df, [])

  def test_metriclist_cache_key(self):
    sum_x = metrics.Sum('X')
    double_sum_x = 2 * metrics.Sum('X')
    m = metrics.MetricList((sum_x, double_sum_x))
    m.compute_on(self.df, cache_key=42)
    self.assertEqual(1, sum_x.get_cached(42))
    self.assertEqual(2, double_sum_x.get_cached(42))
    self.assertTrue(m.in_cache(42))

  def test_metriclist_internal_caching_cleaned_up(self):
    sum_x = metrics.Sum('X')
    double_sum_x = 2 * metrics.Sum('X')
    m = metrics.MetricList((sum_x, double_sum_x))
    m.compute_on(self.df)
    self.assertEqual(m.cache, {})
    self.assertEqual(sum_x.cache, {})
    self.assertEqual(double_sum_x.cache, {})
    self.assertIsNone(m.cache_key)
    self.assertIsNone(sum_x.cache_key)
    self.assertIsNone(double_sum_x.cache_key)

  def test_metrics_with_different_indexes(self):
    df = pd.DataFrame({'X': [0, 1, 2], 'grp': ['A', 'B', 'C']})
    sum_x = metrics.Sum('X')
    cum = operations.CumulativeDistribution('grp', sum_x)
    change = operations.PercentChange('grp', 'B', sum_x, where='grp != "C"')
    bst = operations.Bootstrap(None, change, 100)
    m = metrics.MetricList((cum, change, sum_x, bst))
    output = m.compute_on(df, return_dataframe=False)

    cum_expected = (
        metrics.Sum('X') | operations.CumulativeDistribution('grp')
        | metrics.compute_on(df))
    pct_expected = (
        metrics.Sum('X')
        | operations.PercentChange('grp', 'B', where='grp != "C"')
        | metrics.compute_on(df))
    bst_expected = (
        metrics.Sum('X')
        | operations.PercentChange('grp', 'B', where='grp != "C"')
        | operations.Bootstrap(None, n_replicates=100) | metrics.compute_on(df))
    testing.assert_frame_equal(output[0], cum_expected)
    testing.assert_frame_equal(output[1], pct_expected)
    testing.assert_frame_equal(output[2], metrics.Sum('X').compute_on(df))
    testing.assert_frame_equal(output[3], bst_expected)

  def test_compositemetric_internal_caching(self):
    sum_x = metrics.Sum('X')
    double_sum_x = 2 * metrics.Sum('X')
    m = double_sum_x / sum_x
    with mock.patch.object(
        sum_x, 'compute_through', return_value=1, autospec=True) as mock_fn:
      m.compute_on(self.df, return_dataframe=False)
      mock_fn.assert_called_once_with(self.df, [])

  def test_compositemetric_cache_key(self):
    sum_x = metrics.Sum('X')
    double_sum_x = 2 * metrics.Sum('X')
    m = double_sum_x / sum_x
    m.compute_on(self.df, cache_key=42)
    self.assertEqual(1, sum_x.get_cached(42))
    self.assertEqual(2, double_sum_x.get_cached(42))
    self.assertEqual(2, m.get_cached(42))

  def test_compositemetric_internal_caching_cleaned_up(self):
    sum_x = metrics.Sum('X')
    double_sum_x = 2 * metrics.Sum('X')
    m = double_sum_x / sum_x
    m.compute_on(self.df)
    self.assertEqual(m.cache, {})
    self.assertEqual(sum_x.cache, {})
    self.assertEqual(double_sum_x.cache, {})
    self.assertIsNone(m.cache_key)
    self.assertIsNone(sum_x.cache_key)
    self.assertIsNone(double_sum_x.cache_key)

  def test_complex_metric_internal_caching(self):
    sum_x = metrics.Sum('X')
    double_sum_x = 2 * metrics.Sum('X')
    m = metrics.MetricList((sum_x, double_sum_x / sum_x))
    with mock.patch.object(
        sum_x, 'compute_through', return_value=1, autospec=True) as mock_fn:
      m.compute_on(self.df)
      mock_fn.assert_called_once_with(self.df, [])

  def test_complex_metric_cache_key(self):
    sum_x = metrics.Sum('X')
    double_sum_x = 2 * metrics.Sum('X')
    comp = double_sum_x / sum_x
    m = metrics.MetricList((sum_x, comp))
    m.compute_on(self.df, cache_key=42)
    self.assertEqual(1, sum_x.get_cached(42))
    self.assertEqual(2, double_sum_x.get_cached(42))
    self.assertEqual(2, comp.get_cached(42))
    self.assertTrue(m.in_cache(42))

  def test_complex_metric_internal_caching_cleaned_up(self):
    sum_x = metrics.Sum('X')
    double_sum_x = 2 * metrics.Sum('X')
    comp = double_sum_x / sum_x
    m = metrics.MetricList((sum_x, comp))
    m.compute_on(self.df)
    self.assertEqual(m.cache, {})
    self.assertEqual(sum_x.cache, {})
    self.assertEqual(double_sum_x.cache, {})
    self.assertEqual(comp.cache, {})
    self.assertIsNone(m.cache_key)
    self.assertIsNone(sum_x.cache_key)
    self.assertIsNone(double_sum_x.cache_key)
    self.assertIsNone(comp.cache_key)

  def test_complex_metric_flush_cache_nonrecursive(self):
    sum_x = metrics.Sum('X')
    double_sum_x = 2 * metrics.Sum('X')
    comp = double_sum_x / sum_x
    m = metrics.MetricList((sum_x, comp))
    m.compute_on(self.df, cache_key=42)
    m.flush_cache(42, recursive=False)
    self.assertFalse(m.in_cache(42))
    self.assertTrue(sum_x.in_cache(42))
    self.assertTrue(double_sum_x.in_cache(42))
    self.assertTrue(comp.in_cache(42))

  def test_complex_metric_flush_cache_recursive(self):
    sum_x = metrics.Sum('X')
    double_sum_x = 2 * metrics.Sum('X')
    comp = double_sum_x / sum_x
    m = metrics.MetricList((sum_x, comp))
    m.compute_on(self.df, cache_key=42)
    m.flush_cache(42)
    self.assertFalse(m.in_cache(42))
    self.assertFalse(sum_x.in_cache(42))
    self.assertFalse(double_sum_x.in_cache(42))
    self.assertFalse(comp.in_cache(42))

  def test_flush_cache_prune(self):
    sum_x = metrics.Sum('X')
    double_sum_x = 2 * sum_x
    double_sum_x_plus_one = double_sum_x + 1
    comp = double_sum_x_plus_one - double_sum_x
    m = metrics.MetricList((double_sum_x, comp))
    with mock.patch.object(sum_x, 'flush_cache', autospec=True) as mock_fn:
      m.compute_on(self.df)
      mock_fn.assert_called_once_with(
          sum_x.wrap_cache_key('_RESERVED'), recursive=False)

  def test_flush_cache_no_prune(self):
    sum_x = metrics.Sum('X')
    double_sum_x = 2 * sum_x
    double_sum_x_plus_one = double_sum_x + 1
    comp = double_sum_x_plus_one - double_sum_x
    m = metrics.MetricList((double_sum_x, comp))
    with mock.patch.object(sum_x, 'flush_cache', autospec=True) as mock_fn:
      m.compute_on(self.df, cache_key=42)
      m.flush_cache(42, prune=False)
      self.assertEqual(3, mock_fn.call_count)


if __name__ == '__main__':
  unittest.main()
