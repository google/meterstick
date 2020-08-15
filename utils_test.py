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
"""Tests for meterstick.v2.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from meterstick import utils
import numpy as np
import pandas as pd
import pandas.util.testing as testing
import unittest


class UtilsTest(unittest.TestCase):

  def test_cache_key_init_with_cache_key(self):
    expected = utils.CacheKey('foo', 'where', ['bar'])
    output = utils.CacheKey(expected)
    self.assertEqual(expected, output)

  def test_cache_key_extend(self):
    output = utils.CacheKey('foo', 'where', ['bar', 'baz'])
    output.extend(['baz', 'qux'])
    self.assertEqual(
        utils.CacheKey('foo', 'where', ['bar', 'baz', 'qux']), output)

  def test_cache_key_includes(self):
    derived = utils.CacheKey('foo', 'where', ['bar', 'baz'])
    base = utils.CacheKey('foo', 'where', ['bar'])
    self.assertTrue(derived.includes(base))

  def test_cache_key_where(self):
    output = utils.CacheKey('foo', 'where', ['bar', 'baz'])
    output = utils.CacheKey(output, 'where2')
    output.add_filters(['where', 'where3'])
    self.assertEqual(set(('where', 'where2', 'where3')), output.where)
    self.assertEqual('(where) & (where2) & (where3)', output.all_filters)

  def test_adjust_slices_for_loo_no_splitby_no_extra(self):
    s = pd.Series(
        range(1, 4),
        index=pd.Index(range(3), name='unit')
    )
    output = utils.adjust_slices_for_loo(s)
    testing.assert_series_equal(s, output)

  def test_adjust_slices_for_loo_no_splitby(self):
    s = pd.Series(
        range(1, 5),
        index=pd.MultiIndex.from_tuples(
            (('A', 'grp1'), ('A', 'grp2'), ('B', 'grp1'), ('C', 'grp1')),
            names=('unit', 'grp')))
    output = utils.adjust_slices_for_loo(s)
    expected = pd.Series((1, 3, 0, 4, 0),
                         index=pd.MultiIndex.from_tuples(
                             (('A', 'grp1'), ('B', 'grp1'), ('B', 'grp2'),
                              ('C', 'grp1'), ('C', 'grp2')),
                             names=('unit', 'grp')))
    testing.assert_series_equal(expected, output)

  def test_adjust_slices_for_loo_no_extra(self):
    s = pd.Series(
        range(1, 5),
        index=pd.MultiIndex.from_tuples(
            (('A', 'grp1'), ('A', 'grp2'), ('B', 'grp1'), ('C', 'grp1')),
            names=('unit', 'grp')))
    output = utils.adjust_slices_for_loo(s, ['grp'])
    testing.assert_series_equal(s.drop(('A', 'grp2')), output)

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

if __name__ == '__main__':
  unittest.main()
