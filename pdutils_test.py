# Copyright 2019 Google LLC
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

"""Tests for meterstick.core."""

from meterstick import pdutils
import pandas as pd
from google3.testing.pybase import googletest


class SelectTest(googletest.TestCase):
  """Tests for selecting from an index."""

  def testSelectForBooleanIndex(self):
    data = pd.DataFrame({
        "a": [True, True, False, True],
        "b": [0, 1, 0, 1],
        "c": [1, 2, 3, 4]
    }).set_index("a")

    output = pdutils.select_by_label(data, True)
    correct = pd.DataFrame({
        "b": [0, 1, 1],
        "c": [1, 2, 4]
    })

    pd.util.testing.assert_frame_equal(output, correct)

    output = pdutils.select_by_label(data, False)
    correct = pd.DataFrame({
        "b": [0],
        "c": [3]
    })

    pd.util.testing.assert_frame_equal(output, correct)


class IndexProductTest(googletest.TestCase):
  """Tests for index_product function."""

  def testTwoIndexes(self):
    index1 = pd.Index(["c", "a", "b"], name="letters")
    index2 = pd.Index([2, 1], name="numbers")

    output = pdutils.index_product(index1, index2)
    correct = pd.MultiIndex(
        levels=[["a", "b", "c"], [1, 2]],
        labels=[[2, 2, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0]],
        names=["letters", "numbers"])

    pd.util.testing.assert_index_equal(output, correct)

  def testIndexAndMultiIndex(self):
    index1 = pd.Index(["c", "a", "b"], name="abc")
    index2 = pd.MultiIndex(
        levels=[["x", "y"], [1, 2]],
        labels=[[0, 0, 1, 1], [1, 0, 1, 0]],
        names=["xy", "numbers"])

    output = pdutils.index_product(index1, index2)
    correct = pd.MultiIndex(
        levels=[["a", "b", "c"], ["x", "y"], [1, 2]],
        labels=[[2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]],
        names=["abc", "xy", "numbers"])

    pd.util.testing.assert_index_equal(output, correct)

    output = pdutils.index_product(index2, index1)
    correct = pd.MultiIndex(
        levels=[["x", "y"], [1, 2], ["a", "b", "c"]],
        labels=[[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]],
        names=["xy", "numbers", "abc"])

    pd.util.testing.assert_index_equal(output, correct)

  def testTwoMultiIndexes(self):
    index1 = pd.MultiIndex(
        levels=[["a", "b", "c"], [1, 2, 3]],
        labels=[[0, 1, 2, 0], [0, 1, 2, 2]],
        names=["abc", "123"])
    index2 = pd.MultiIndex(
        levels=[["x", "y"], [0, 1]],
        labels=[[0, 0, 1, 1], [1, 0, 1, 0]],
        names=["xy", "01"])

    output = pdutils.index_product(index1, index2)
    correct = pd.MultiIndex(
        levels=[["a", "b", "c"], [1, 2, 3],
                ["x", "y"], [0, 1]],
        labels=[[0, 0, 0, 0, 1, 1, 1, 1,
                 2, 2, 2, 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1,
                 2, 2, 2, 2, 2, 2, 2, 2],
                [0, 0, 1, 1, 0, 0, 1, 1,
                 0, 0, 1, 1, 0, 0, 1, 1],
                [1, 0, 1, 0, 1, 0, 1, 0,
                 1, 0, 1, 0, 1, 0, 1, 0]],
        names=["abc", "123", "xy", "01"])

    pd.util.testing.assert_index_equal(output, correct)


if __name__ == "__main__":
  googletest.main()
