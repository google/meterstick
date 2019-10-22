# Lint as: python2, python3
"""Tests for meterstick.comparisons."""

from __future__ import division

from meterstick import comparisons
from meterstick import metrics
import numpy as np
import pandas as pd
from google3.testing.pybase import googletest


class ComparisonsTests(googletest.TestCase):
  """Tests for Comparisons objects."""

  def testAbsoluteDifference(self):
    data = pd.DataFrame({"X": [1, 3, 2, 3, 1, 4],
                         "Condition": [0, 0, 0, 1, 1, 1]})

    comparison = comparisons.AbsoluteDifference("Condition", 0)
    comparison.precalculate(data)
    data.set_index("Condition", inplace=True)

    metric = metrics.Sum("X")
    metric.precalculate(data, split_index=None)
    output = comparison(data, metric).values[0]

    self.assertEqual(2, output)

  def testPercentageDifference(self):
    data = pd.DataFrame({"X": [1, 3, 2, 3, 1, 4],
                         "Condition": [0, 0, 0, 1, 1, 1]})

    comparison = comparisons.PercentageDifference("Condition", 0)
    comparison.precalculate(data)
    data.set_index("Condition", inplace=True)

    metric = metrics.Sum("X")
    metric.precalculate(data, split_index=None)
    output = comparison(data, metric).values[0]

    self.assertEqual(100 * (8 - 6) / 6, output)

  def testIncludeBaseline(self):
    data = pd.DataFrame({"X": [1, 3, 2, 3, 1, 4],
                         "Condition": [0, 0, 0, 1, 1, 1]})

    comparison = comparisons.PercentageDifference(
        "Condition", 0, include_base=True)
    comparison.precalculate(data)
    data.set_index("Condition", inplace=True)

    metric = metrics.Sum("X")
    metric.precalculate(data, split_index=None)
    output = comparison(data, metric).values[0]

    self.assertEqual(0, output)

  def testMH(self):
    data = pd.DataFrame({"Clicks": [1, 3, 2, 3, 1, 2],
                         "Conversions": [1, 0, 1, 2, 1, 1],
                         "Id": [1, 2, 3, 1, 2, 3],
                         "Condition": [0, 0, 0, 1, 1, 1]})

    comparison = comparisons.MH("Condition", 0, "Id")
    comparison.precalculate(data)
    data.set_index("Condition", inplace=True)

    metric = metrics.Ratio("Conversions", "Clicks")
    metric.precalculate(data, split_index=None)
    output = comparison(data, metric).values[0]

    ka = np.array([2, 1, 1])
    kb = np.array([1, 0, 1])
    na = np.array([3, 1, 2])
    nb = np.array([1, 3, 2])
    w = 1. / (na + nb)
    correct = (sum(ka * nb * w) / sum(kb * na * w) - 1) * 100

    self.assertAlmostEqual(output, correct)

  def testMHMultipleIds(self):
    data = pd.DataFrame({"Clicks": [0, 1, 1, 2, 1, 1, 2, 1, 1, 0, 2, 0],
                         "Conversions": [1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
                         "Id": [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3],
                         "Condition": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]})

    comparison = comparisons.MH("Condition", 0, "Id")
    comparison.precalculate(data)
    data.set_index("Condition", inplace=True)

    metric = metrics.Ratio("Conversions", "Clicks")
    metric.precalculate(data, split_index=None)
    output = comparison(data, metric).values[0]

    ka = np.array([2, 1, 1])
    kb = np.array([1, 0, 1])
    na = np.array([3, 1, 2])
    nb = np.array([1, 3, 2])
    w = 1. / (na + nb)
    correct = (sum(ka * nb * w) / sum(kb * na * w) - 1) * 100

    self.assertAlmostEqual(output, correct)

  def testMHRaisesErrorForMetricNotDefinedByRatio(self):
    data = pd.DataFrame({"Clicks": [2, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                         "Conversions": [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                         "Id": [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3],
                         "Condition": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]})

    comparison = comparisons.MH("Condition", 0, "Id")
    comparison.precalculate(data)
    metric = metrics.Sum("Conversions") / metrics.Sum("Clicks")
    metric.precalculate(data, split_index=None)
    with self.assertRaises(AttributeError):
      comparison(data, metric)


if __name__ == "__main__":
  googletest.main()
