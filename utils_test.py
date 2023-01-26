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
"""Tests for meterstick.v2.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from meterstick import metrics
from meterstick import operations
from meterstick import utils
import numpy as np
import pandas as pd
from pandas import testing


class UtilsTest(absltest.TestCase):

  def test_adjust_slices_for_loo_no_splitby_no_operation_unit_filled(self):
    df = pd.DataFrame({'unit': list('abc'), 'x': range(1, 4)})
    bucket_res = df[df.unit != 'a'].groupby('unit').sum()
    output = utils.adjust_slices_for_loo(bucket_res, [], df)
    expected = pd.DataFrame({'x': [0, 2, 3]},
                            index=pd.Index(list('abc'), name='unit'))
    testing.assert_frame_equal(output, expected)

  def test_adjust_slices_for_loo_no_splitby_operation(self):
    df = pd.DataFrame({
        'unit': list('abb'),
        'grp': list('bbc'),
        'x': range(1, 4)
    })
    bucket_res = df[df.unit != 'a'].groupby(['unit', 'grp']).sum()
    output = utils.adjust_slices_for_loo(bucket_res, [], df)
    expected = pd.DataFrame({'x': [0, 0]},
                            index=pd.MultiIndex.from_tuples(
                                (('a', 'b'), ('a', 'c')),
                                names=('unit', 'grp')))
    testing.assert_frame_equal(output, expected)

  def test_adjust_slices_for_loo_splitby_no_operation(self):
    df = pd.DataFrame({
        'unit': list('abc'),
        'grp': list('abb'),
        'x': range(1, 4)
    })
    bucket_res = df[df.grp != 'b'].groupby(['grp', 'unit']).sum()
    output = utils.adjust_slices_for_loo(bucket_res, ['grp'], df)
    expected = pd.DataFrame({'x': [1, 0, 0]},
                            index=pd.MultiIndex.from_tuples(
                                (('a', 'a'), ('b', 'b'), ('b', 'c')),
                                names=('grp', 'unit')))
    testing.assert_frame_equal(output, expected)

  def test_adjust_slices_for_loo_splitby_operation(self):
    df = pd.DataFrame({
        'grp': list('aaabbb'),
        'op': ['x'] * 2 + ['y'] * 2 + ['z'] * 2,
        'unit': [1, 2, 3, 2, 3, 2],
        'x': range(1, 7)
    })
    bucket_res = df[df.unit != 1].groupby(['grp', 'unit', 'op']).sum()
    output = utils.adjust_slices_for_loo(bucket_res, ['grp'], df)
    expected = pd.DataFrame({'x': [0, 0, 0, 0, 6, 0, 5]},
                            index=pd.MultiIndex.from_tuples(
                                (
                                    ('a', 1, 'x'),
                                    ('a', 1, 'y'),
                                    ('a', 2, 'y'),
                                    ('a', 3, 'x'),
                                    ('b', 2, 'z'),
                                    ('b', 3, 'y'),
                                    ('b', 3, 'z'),
                                ),
                                names=('grp', 'unit', 'op')))
    testing.assert_frame_equal(output, expected)

  def test_one_level_column_and_no_splitby_melt(self):
    unmelted = pd.DataFrame({'foo': [1], 'bar': [2]}, columns=['foo', 'bar'])
    expected = pd.DataFrame({'Value': [1, 2]}, index=['foo', 'bar'])
    expected.index.name = 'Metric'
    testing.assert_frame_equal(expected, utils.melt(unmelted))

  def test_one_level_value_column_and_no_splitby_unmelt(self):
    melted = pd.DataFrame({'Value': [1, 2]}, index=['foo', 'bar'])
    melted.index.name = 'Metric'
    expected = pd.DataFrame({'foo': [1], 'bar': [2]}, columns=['foo', 'bar'])
    expected.columns.names = ['Metric']
    testing.assert_frame_equal(expected, utils.unmelt(melted))

  def test_one_level_not_value_column_and_no_splitby_unmelt(self):
    melted = pd.DataFrame({'Baz': [1, 2]}, index=['foo', 'bar'])
    melted.index.name = 'Metric'
    expected = pd.DataFrame([[1, 2]],
                            columns=pd.MultiIndex.from_product(
                                [['foo', 'bar'], ['Baz']],
                                names=['Metric', None]))
    testing.assert_frame_equal(expected, utils.unmelt(melted))

  def test_one_level_column_and_single_splitby_melt(self):
    unmelted = pd.DataFrame(
        data={
            'foo': [0, 1],
            'bar': [2, 3]
        },
        columns=['foo', 'bar'],
        index=['B', 'A'])
    unmelted.index.name = 'grp'
    expected = pd.DataFrame({'Value': range(4)},
                            index=pd.MultiIndex.from_product(
                                (['foo', 'bar'], ['B', 'A']),
                                names=['Metric', 'grp']))
    expected.index.name = 'Metric'
    testing.assert_frame_equal(expected, utils.melt(unmelted))

  def test_one_level_column_and_single_splitby_unmelt(self):
    expected = pd.DataFrame(
        data={
            'foo': [0, 1],
            'bar': [2, 3]
        },
        columns=['foo', 'bar'],
        index=['B', 'A'])
    expected.index.name = 'grp'
    expected.columns.name = 'Metric'
    melted = pd.DataFrame({'Value': range(4)},
                          index=pd.MultiIndex.from_product(
                              (['foo', 'bar'], ['B', 'A']),
                              names=['Metric', 'grp']))
    melted.index.name = 'Metric'
    testing.assert_frame_equal(expected, utils.unmelt(melted))

  def test_one_level_column_and_multiple_splitby_melt(self):
    unmelted = pd.DataFrame(
        data={
            'foo': range(4),
            'bar': range(4, 8)
        },
        columns=['foo', 'bar'],
        index=pd.MultiIndex.from_product((['B', 'A'], ['US', 'non-US']),
                                         names=['grp', 'country']))
    expected = pd.DataFrame({'Value': range(8)},
                            index=pd.MultiIndex.from_product(
                                (['foo', 'bar'], ['B', 'A'], ['US', 'non-US']),
                                names=['Metric', 'grp', 'country']))
    expected.index.name = 'Metric'
    testing.assert_frame_equal(expected, utils.melt(unmelted))

  def test_one_level_column_and_multiple_splitby_unmelt(self):
    melted = pd.DataFrame({'Value': range(8)},
                          index=pd.MultiIndex.from_product(
                              (['foo', 'bar'], ['B', 'A'], ['US', 'non-US']),
                              names=['Metric', 'grp', 'country']))
    expected = pd.DataFrame(
        data={
            'foo': range(4),
            'bar': range(4, 8)
        },
        columns=['foo', 'bar'],
        index=pd.MultiIndex.from_product((['B', 'A'], ['US', 'non-US']),
                                         names=['grp', 'country']))
    expected.columns.name = 'Metric'
    testing.assert_frame_equal(expected, utils.unmelt(melted))

  def test_multiple_index_columns_and_no_splitby_melt(self):
    unmelted = pd.DataFrame([[1, 2, 3, 4]],
                            columns=pd.MultiIndex.from_product(
                                (['foo', 'bar'], ['Value', 'SE'])))
    expected = pd.DataFrame(
        data={
            'Value': [1, 3],
            'SE': [2, 4]
        },
        index=['foo', 'bar'],
        columns=['Value', 'SE'])
    expected.index.name = 'Metric'
    testing.assert_frame_equal(expected, utils.melt(unmelted))

  def test_multiple_index_columns_and_no_splitby_unmelt(self):
    melted = pd.DataFrame(
        data={
            'Value': [1, 3],
            'SE': [2, 4]
        },
        index=['foo', 'bar'],
        columns=['Value', 'SE'])
    melted.index.name = 'Metric'
    expected = pd.DataFrame([[1, 2, 3, 4]],
                            columns=pd.MultiIndex.from_product(
                                (['foo', 'bar'], ['Value', 'SE'])))
    expected.columns.names = ['Metric', None]
    testing.assert_frame_equal(expected, utils.unmelt(melted))

  def test_multiple_index_column_and_single_splitby_melt(self):
    unmelted = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]],
                            columns=pd.MultiIndex.from_product(
                                (['foo', 'bar'], ['Value', 'SE'])),
                            index=['B', 'A'])
    unmelted.index.name = 'grp'
    expected = pd.DataFrame(
        data={
            'Value': [1, 5, 3, 7],
            'SE': [2, 6, 4, 8]
        },
        index=pd.MultiIndex.from_product((['foo', 'bar'], ['B', 'A']),
                                         names=['Metric', 'grp']),
        columns=['Value', 'SE'])
    testing.assert_frame_equal(expected, utils.melt(unmelted))

  def test_multiple_index_column_and_single_splitby_unmelt(self):
    melted = pd.DataFrame(
        data={
            'Value': [1, 5, 3, 7],
            'SE': [2, 6, 4, 8]
        },
        index=pd.MultiIndex.from_product((['foo', 'bar'], ['B', 'A']),
                                         names=['Metric', 'grp']),
        columns=['Value', 'SE'])
    expected = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]],
                            columns=pd.MultiIndex.from_product(
                                (['foo', 'bar'], ['Value', 'SE'])),
                            index=['B', 'A'])
    expected.index.name = 'grp'
    expected.columns.names = ['Metric', None]
    testing.assert_frame_equal(expected, utils.unmelt(melted))

  def test_multiple_index_column_and_multiple_splitby_melt(self):
    unmelted = pd.DataFrame(
        [range(4), range(4, 8),
         range(8, 12), range(12, 16)],
        columns=pd.MultiIndex.from_product((['foo', 'bar'], ['Value', 'SE'])),
        index=pd.MultiIndex.from_product((['B', 'A'], ['US', 'non-US']),
                                         names=['grp', 'country']))
    expected = pd.DataFrame(
        data={
            'Value': [0, 4, 8, 12, 2, 6, 10, 14],
            'SE': [1, 5, 9, 13, 3, 7, 11, 15]
        },
        index=pd.MultiIndex.from_product(
            (['foo', 'bar'], ['B', 'A'], ['US', 'non-US']),
            names=['Metric', 'grp', 'country']),
        columns=['Value', 'SE'])
    testing.assert_frame_equal(expected, utils.melt(unmelted))

  def test_multiple_index_column_and_multiple_splitby_unmelt(self):
    melted = pd.DataFrame(
        data={
            'Value': [0, 4, 8, 12, 2, 6, 10, 14],
            'SE': [1, 5, 9, 13, 3, 7, 11, 15]
        },
        index=pd.MultiIndex.from_product(
            (['foo', 'bar'], ['B', 'A'], ['US', 'non-US']),
            names=['Metric', 'grp', 'country']),
        columns=['Value', 'SE'])
    expected = pd.DataFrame(
        [range(4), range(4, 8),
         range(8, 12), range(12, 16)],
        columns=pd.MultiIndex.from_product((['foo', 'bar'], ['Value', 'SE'])),
        index=pd.MultiIndex.from_product((['B', 'A'], ['US', 'non-US']),
                                         names=['grp', 'country']))
    expected.columns.names = ['Metric', None]
    testing.assert_frame_equal(expected, utils.unmelt(melted))

  def test_nan_melt_unmelt(self):
    df = pd.DataFrame({'sum(x)': [np.nan]})
    expected = pd.DataFrame({'Value': [np.nan]}, index=['sum(x)'])
    expected.index.name = 'Metric'
    testing.assert_frame_equal(expected, utils.melt(df))
    df.columns.name = 'Metric'
    testing.assert_frame_equal(df, utils.unmelt(expected))

  def test_remove_empty_level(self):
    df = pd.DataFrame([[0, 1, 2]], columns=['a', 'b', 'c'])
    df.set_index(['b', 'a'], append=True, inplace=True)
    expected = df.droplevel(0)
    actual = utils.remove_empty_level(df)
    testing.assert_frame_equal(expected, actual)

  def test_get_extra_idx(self):
    mh = operations.MH('foo', 'f', 'bar', metrics.Ratio('a', 'b'))
    ab = operations.AbsoluteChange('foo', 'f', metrics.Sum('c'))
    m = operations.Jackknife('unit', metrics.MetricList((mh, ab)))
    self.assertEqual(utils.get_extra_idx(m), ('foo',))

  def test_get_extra_idx_raises(self):
    mh = operations.MH('foo', 'f', 'bar', metrics.Ratio('a', 'b'))
    ab = operations.AbsoluteChange('baz', 'f', metrics.Sum('c'))
    m = operations.Jackknife('unit', metrics.MetricList((mh, ab)))
    with self.assertRaises(ValueError) as cm:
      utils.get_extra_idx(m)
    self.assertEqual(str(cm.exception), 'Incompatible indexes!')

  def test_get_extra_split_by(self):
    mh = operations.MH('foo', 'f', 'bar', metrics.Ratio('a', 'b'))
    m = operations.AbsoluteChange('unit', 'a', mh)
    self.assertEqual(utils.get_extra_split_by(m), ('unit', 'foo', 'bar'))

  def test_get_extra_split_by_raises(self):
    mh = operations.MH('foo', 'f', 'bar', metrics.Ratio('a', 'b'))
    ab = operations.AbsoluteChange('foo', 'f', metrics.Sum('c'))
    m = operations.Jackknife('unit', metrics.MetricList((mh, ab)))
    with self.assertRaises(ValueError) as cm:
      utils.get_extra_split_by(m)
    self.assertEqual(str(cm.exception), 'Incompatible split_by!')

  def test_get_unique_prefix(self):
    df = pd.DataFrame([[0, 1]], columns=['meterstick_tmp::', 'b'])
    self.assertEqual(utils.get_unique_prefix(df), 'meterstick_tmp:::')

  def test_get_equivalent_metric_no_df(self):
    m = metrics.Mean('x', where='a')
    output, _ = utils.get_equivalent_metric(m)
    expected = metrics.Sum('x') / metrics.Count('x')
    expected.where = 'a'
    expected.name = 'mean(x)'
    self.assertEqual(output, expected)

  def test_get_equivalent_metric_with_df(self):
    m = metrics.Dot('x', 'y', name='foo', where='a')
    df = pd.DataFrame({'x': [1, 2], 'y': [2, 3]})
    output, _ = utils.get_equivalent_metric(m, df)
    expected = metrics.Sum('meterstick_tmp:(x * y)')
    expected.where = 'a'
    expected.name = 'foo'
    expected_df = pd.DataFrame({
        'x': [1, 2],
        'y': [2, 3],
        'meterstick_tmp:(x * y)': [2, 6]
    })
    self.assertEqual(output, expected)
    testing.assert_frame_equal(df, expected_df)

  def test_get_stable_equivalent_metric_tree(self):
    m1 = metrics.Mean('x', where='a')
    m2 = metrics.Dot('x', 'y', where='b')
    m = 2 * (m1 - m2)
    output, _ = utils.get_stable_equivalent_metric_tree(m)
    expected = 2 * (
        utils.get_equivalent_metric(m1)[0] - utils.get_equivalent_metric(m2)[0])
    self.assertEqual(output, expected)

  def test_push_filters_to_leaf(self):
    s = metrics.Sum('x', where='a')
    m = s / metrics.Count('y')
    m.where = 'c'
    m = metrics.MetricList([s + 1, m], where='b')
    output = utils.push_filters_to_leaf(m)
    expected = metrics.MetricList([
        metrics.Sum('x', where=('a', 'b')) + 1,
        metrics.Sum('x', where=('a', 'b', 'c')) /
        metrics.Count('y', where=('b', 'c'))
    ])
    self.assertEqual(output, expected)

  def test_get_leaf_metrics(self):
    m = metrics.MetricList(
        (metrics.Ratio('x', 'y'), metrics.Sum('c', where='f') + 1))
    output = utils.get_leaf_metrics(m)
    expected = [metrics.Sum('x'), metrics.Sum('y'), metrics.Sum('c', where='f')]
    self.assertEqual(output, expected)

  def test_get_leaf_metrics_include_constants(self):
    m = metrics.MetricList((metrics.Ratio('x', 'y',
                                          where='f'), metrics.Sum('c') + 1))
    output = utils.get_leaf_metrics(m, True)
    expected = [metrics.Sum('x'), metrics.Sum('y'), metrics.Sum('c'), 1]
    self.assertEqual(output, expected)


if __name__ == '__main__':
  absltest.main()
