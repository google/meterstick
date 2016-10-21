#!/usr/bin/python
#
# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for google3.ads.metrics.lib.meterstick.comparisons."""

from __future__ import division

import google3
import numpy as np
import pandas as pd

from google3.testing.pybase import googletest

from google3.ads.metrics.lib.meterstick import comparisons
from google3.ads.metrics.lib.meterstick import metrics


class ComparisonsTests(googletest.TestCase):
  """Tests for Comparisons objects."""

  def testAbsoluteDifference(self):
    data = pd.DataFrame({"X": [1, 3, 2, 3, 1, 4],
                         "Condition": [0, 0, 0, 1, 1, 1]})
    weights = np.ones(6)

    comparison = comparisons.AbsoluteDifference("Condition", 0)
    comparison.precalculate_factors(data)

    metric = metrics.Sum("X")
    output = comparison(data, weights, metric).values[0]

    self.assertEqual(2, output)

  def testPercentageDifference(self):
    data = pd.DataFrame({"X": [1, 3, 2, 3, 1, 4],
                         "Condition": [0, 0, 0, 1, 1, 1]})
    weights = np.ones(6)

    comparison = comparisons.PercentageDifference("Condition", 0)
    comparison.precalculate_factors(data)

    metric = metrics.Sum("X")
    output = comparison(data, weights, metric).values[0]

    self.assertEqual(100 * (8 - 6) / 6, output)

  def testIncludeBaseline(self):
    data = pd.DataFrame({"X": [1, 3, 2, 3, 1, 4],
                         "Condition": [0, 0, 0, 1, 1, 1]})
    weights = np.ones(6)

    comparison = comparisons.PercentageDifference(
        "Condition", 0, include_base=True)
    comparison.precalculate_factors(data)

    metric = metrics.Sum("X")
    output = comparison(data, weights, metric).values[0]

    self.assertEqual(0, output)

  def testMH(self):
    data = pd.DataFrame({"Clicks": [1, 3, 2, 3, 1, 2],
                         "Conversions": [1, 0, 1, 2, 1, 1],
                         "Id": [1, 2, 3, 1, 2, 3],
                         "Condition": [0, 0, 0, 1, 1, 1]})

    weights = np.ones(6)

    comparison = comparisons.MH("Condition", 0, "Id")
    comparison.precalculate_factors(data)

    metric = metrics.Ratio("Conversions", "Clicks")
    output = comparison(data, weights, metric).values[0]

    ka = np.array([2, 1, 1])
    kb = np.array([1, 0, 1])

    na = np.array([3, 1, 2])
    nb = np.array([1, 3, 2])

    w = 1 / (na + nb)

    correct = sum(ka * nb * w) / sum(kb * na * w)

    self.assertEqual(output, correct)

  def testMHMultipleIds(self):
    data = pd.DataFrame({"Clicks": [0, 1, 1, 2, 1, 1, 2, 1, 1, 0, 2, 0],
                         "Conversions": [1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
                         "Id": [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3],
                         "Condition": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]})

    weights = np.ones(12)

    comparison = comparisons.MH("Condition", 0, "Id")
    comparison.precalculate_factors(data)

    metric = metrics.Ratio("Conversions", "Clicks")
    output = comparison(data, weights, metric).values[0]

    ka = np.array([2, 1, 1])
    kb = np.array([1, 0, 1])

    na = np.array([3, 1, 2])
    nb = np.array([1, 3, 2])

    w = 1 / (na + nb)

    correct = sum(ka * nb * w) / sum(kb * na * w)

    self.assertEqual(output, correct)

  def testMHZeroRateRaisesError(self):
    data = pd.DataFrame({"Clicks": [2, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                         "Conversions": [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                         "Id": [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3],
                         "Condition": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]})

    weights = np.ones(12)

    comparison = comparisons.MH("Condition", 0, "Id")
    comparison.precalculate_factors(data)

    metric = metrics.Ratio("Conversions", "Clicks")

    with self.assertRaises(ValueError):
      comparison(data, weights, metric)


if __name__ == "__main__":
  googletest.main()
