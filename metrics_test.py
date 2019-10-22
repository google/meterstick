# Lint as: python2, python3
"""Tests for meterstick.metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import pandas as pd
from six.moves import range

from meterstick import core
from meterstick import metrics
from google3.testing.pybase import googletest


class DistributionTest(googletest.TestCase):
  """Tests for Distribution class."""

  def testDistribution(self):
    df = pd.DataFrame({"XX": [1, 1, 1, 2, 2, 3, 4],
                       "YY": [1, 2, 0, 1, 1, 1, 1]})
    # note that the "over" argument is specified as
    # a string, since we only have a single variable
    metric = metrics.Distribution("XX", "YY", sort=False)
    metric.precalculate(df, None)
    output = metric(df)
    correct = pd.Series(
        [6 / 7., 1 / 14., 1 / 14.],
        name="",
        index=pd.Index([1, 2, 0], name="YY"))

    pd.util.testing.assert_series_equal(output, correct)

  def testDistributionNormalize(self):
    df = pd.DataFrame({"XX": [1, 1, 1, 2, 2, 3, 4],
                       "YY": [1, 2, 0, 1, 1, 1, 1]})
    # note that the "over" argument is specified as
    # a string, since we only have a single variable
    metric = metrics.Distribution("XX", "YY", sort=False, normalize=False)
    metric.precalculate(df, None)
    output = metric(df)
    correct = pd.Series(
        [12, 1, 1],
        name="",
        index=pd.Index([1, 2, 0], name="YY"))

    pd.util.testing.assert_series_equal(output, correct)

  def testTwoDimensionalDistribution(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [1, 2, 0, 1, 1, 1, 1],
                       "Z": [1, 0, 0, 0, 0, 0, 0]})
    metric = metrics.Distribution("X", ["Y", "Z"], sort=False)
    metric.precalculate(df, None)
    output = metric(df)
    correct = pd.Series(
        np.array([1 / 14., 1 / 14., 1 / 14., 11 / 14.]),
        name="",
        index=pd.MultiIndex(levels=[[0, 1, 2], [0, 1]],
                            labels=[[1, 2, 0, 1], [1, 0, 0, 0]],
                            names=["Y", "Z"]))

    pd.util.testing.assert_series_equal(output, correct)

  def testTwoDimensionalDistributionNormalize(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [1, 2, 0, 1, 1, 1, 1],
                       "Z": [1, 0, 0, 0, 0, 0, 0]})
    metric = metrics.Distribution("X", ["Y", "Z"], sort=False, normalize=False)
    metric.precalculate(df, None)
    output = metric(df)
    correct = pd.Series(
        np.array([1, 1, 1, 11]),
        name="",
        index=pd.MultiIndex(levels=[[0, 1, 2], [0, 1]],
                            labels=[[1, 2, 0, 1], [1, 0, 0, 0]],
                            names=["Y", "Z"]))

    pd.util.testing.assert_series_equal(output, correct)

  def testTwoDimensionalDistributionExpand(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [1, 2, 0, 1, 1, 1, 1],
                       "Z": [1, 0, 0, 0, 0, 0, 0]})
    metric = metrics.Distribution("X", ["Y", "Z"], expand=True)
    metric.precalculate(df, None)
    output = metric(df)
    correct = pd.Series(
        np.array([1 / 14., 0.,
                  11 / 14., 1 / 14.,
                  1 / 14., 0.]),
        name="",
        index=pd.MultiIndex(levels=[[0, 1, 2], [0, 1]],
                            labels=[[0, 0, 1, 1, 2, 2],
                                    [0, 1, 0, 1, 0, 1]],
                            names=["Y", "Z"]))

    pd.util.testing.assert_series_equal(output, correct)


class CumulativeDistributionTest(googletest.TestCase):
  """Tests for Cumulative class."""

  def testCumulativeDistribution(self):
    df = pd.DataFrame({"XX": [1, 1, 1, 2, 2, 3, 4],
                       "YY": [1, 2, 0, 1, 1, 1, 1]})
    # note that the "over" argument is specified as
    # a string, since we only have a single variable
    metric = metrics.CumulativeDistribution("XX", "YY")
    metric.precalculate(df, None)
    output = metric(df)
    correct = pd.Series(
        np.array([1 / 14., 13 / 14., 1.]),
        name="",
        index=pd.Index([0, 1, 2], name="YY"))

    pd.util.testing.assert_series_equal(output, correct,
                                        check_exact=False)

  def testCumulativeDistributionNormalize(self):
    df = pd.DataFrame({"XX": [1, 1, 1, 2, 2, 3, 4],
                       "YY": [1, 2, 0, 1, 1, 1, 1]})
    # note that the "over" argument is specified as
    # a string, since we only have a single variable
    metric = metrics.CumulativeDistribution("XX", "YY", normalize=False)
    metric.precalculate(df, None)
    output = metric(df)
    correct = pd.Series(
        np.array([1, 13, 14]),
        name="",
        index=pd.Index([0, 1, 2], name="YY"))

    pd.util.testing.assert_series_equal(output, correct,
                                        check_exact=False)

  def testCumulativeDistributionSortDescending(self):
    df = pd.DataFrame({"XX": [1, 1, 1, 2, 2, 3, 4],
                       "YY": [1, 2, 0, 1, 1, 1, 1]})
    # note that the "over" argument is specified as
    # a string, since we only have a single variable
    metric = metrics.CumulativeDistribution("XX", "YY", ascending=False)
    metric.precalculate(df, None)
    output = metric(df)
    correct = pd.Series(
        np.array([1 / 14., 13 / 14., 1.]),
        name="",
        index=pd.Index([2, 1, 0], name="YY"))

    pd.util.testing.assert_series_equal(output, correct,
                                        check_exact=False)

  def testTwoDimensionalCumulativeDistribution(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [1, 2, 0, 1, 1, 1, 1],
                       "Z": [1, 0, 0, 0, 0, 0, 0]})
    metric = metrics.CumulativeDistribution("X", ["Y", "Z"],
                                            expand=True)
    metric.precalculate(df, None)
    output = metric(df)
    correct = pd.Series(
        np.array([1 / 14., 1 / 14.,
                  12 / 14., 13 / 14.,
                  1., 1.]),
        name="",
        index=pd.MultiIndex(levels=[[0, 1, 2], [0, 1]],
                            labels=[[0, 0, 1, 1, 2, 2],
                                    [0, 1, 0, 1, 0, 1]],
                            names=["Y", "Z"]))

    pd.util.testing.assert_series_equal(output, correct,
                                        check_exact=False)

  def testTwoDimensionalCumulativeDistributionNormalize(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [1, 2, 0, 1, 1, 1, 1],
                       "Z": [1, 0, 0, 0, 0, 0, 0]})
    metric = metrics.CumulativeDistribution("X", ["Y", "Z"],
                                            expand=True, normalize=False)
    metric.precalculate(df, None)
    output = metric(df)
    correct = pd.Series(
        np.array([1., 1., 12., 13., 14., 14.]),
        name="",
        index=pd.MultiIndex(levels=[[0, 1, 2], [0, 1]],
                            labels=[[0, 0, 1, 1, 2, 2],
                                    [0, 1, 0, 1, 0, 1]],
                            names=["Y", "Z"]))

    pd.util.testing.assert_series_equal(output, correct,
                                        check_exact=False)

  def testShuffledTwoDimensionalCumulativeDistribution(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [1, 2, 0, 1, 1, 1, 1],
                       "Z": [1, 0, 0, 0, 0, 0, 0]})
    metric = metrics.CumulativeDistribution("X", ["Y", "Z"],
                                            expand=True)
    metric.precalculate(df, None)
    output = metric(df.iloc[np.random.permutation(7)])
    correct = pd.Series(
        np.array([1 / 14., 1 / 14.,
                  12 / 14., 13 / 14.,
                  1., 1.]),
        name="",
        index=pd.MultiIndex(levels=[[0, 1, 2], [0, 1]],
                            labels=[[0, 0, 1, 1, 2, 2],
                                    [0, 1, 0, 1, 0, 1]],
                            names=["Y", "Z"]))

    pd.util.testing.assert_series_equal(output, correct,
                                        check_exact=False)


class RatioTest(googletest.TestCase):
  """Tests for Ratio class."""

  def testRatio(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [1, 2, 0, 1, 1, 1, 1]})

    metric = metrics.Ratio("X", "Y")
    metric.precalculate(df, None)

    output = metric(df)

    correct = 2.0

    self.assertEqual(output, correct)

  def testRatioPositiveDivideByZero(self):
    df = pd.DataFrame({"X": [1, 1], "Y": [0, 0]})

    metric = metrics.Ratio("X", "Y")
    metric.precalculate(df, None)

    output = metric(df)

    correct = np.inf

    self.assertEqual(output, correct)

  def testRatioZeroDivideByZero(self):
    df = pd.DataFrame({"X": [0, 0], "Y": [0, 0]})

    metric = metrics.Ratio("X", "Y")
    metric.precalculate(df, None)

    output = metric(df)

    self.assertTrue(np.isnan(output))


class CountTest(googletest.TestCase):
  """Tests for Count class."""

  def testCount(self):
    df = pd.DataFrame({"X": [1, 1, np.nan, 2, 2, 3, 4]})

    metric = metrics.Count("X")
    metric.precalculate(df, None)

    output = metric(df)

    correct = 6

    self.assertEqual(output, correct)


class SumTest(googletest.TestCase):
  """Tests for Sum class."""

  def testSum(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4]})

    metric = metrics.Sum("X")
    metric.precalculate(df, None)

    output = metric(df)

    correct = 14

    self.assertEqual(output, correct)

  def testLambdaSum(self):
    df = pd.DataFrame({"X": [0, 1, 2, 3]})
    built_in_sum = metrics.Sum("X")
    lambda_sum = metrics.Metric("lambda sum(X)", fn=lambda df: sum(df.X))
    built_in_sum.precalculate(df, None)
    lambda_sum.precalculate(df, None)
    expected = built_in_sum(df)
    actual = lambda_sum(df)
    self.assertEqual(expected, actual)
    self.assertEqual(lambda_sum.name, "lambda sum(X)")


class MeanTest(googletest.TestCase):
  """Tests for Sum class."""

  def testMean(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4]})

    metric = metrics.Mean("X")
    metric.precalculate(df, None)

    output = metric(df)

    correct = 2.0

    self.assertEqual(output, correct)


class WeightedMeanTest(googletest.TestCase):
  """Tests for WeightedMean class."""

  def testEqualWeightedMean(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [1, 1, 1, 1, 1, 1, 1]})

    metric = metrics.WeightedMean("X", "Y")
    metric.precalculate(df, None)

    output = metric(df)

    correct = 2.0

    self.assertEqual(output, correct)

  def testWeightedMean(self):
    df = pd.DataFrame({"X": [1, 1, 2, 2, 2, 3, 4],
                       "Y": [1, 2, 3, 4, 3, 1, 1]})

    metric = metrics.WeightedMean("X", "Y")
    metric.precalculate(df, None)

    output = metric(df)

    correct = 2.0

    self.assertEqual(output, correct)

  def testGroupedWeightedMean(self):
    df = pd.DataFrame({"X": [1, 1, 2, 2, 2, 3, 4, 5],
                       "Y": [1, 2, 3, 4, 3, 1, 1, 1],
                       "Z": [1, 1, 1, 1, 1, 1, 1, 2]})

    metric = metrics.WeightedMean("X", "Y")
    output = (core.Analyze(df).
              split_by("Z").
              calculate(metric)).run()

    correct = pd.DataFrame({
        "Y_weighted_mean(X)": [2.0, 5.0],
        "Z": [1, 2]
    })
    correct = correct.set_index("Z")

    pd.util.testing.assert_frame_equal(output, correct)


class QuantileTest(googletest.TestCase):
  """Tests for Quantile class."""

  def testQuantile(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4]})

    metric = metrics.Quantile("X", 0.5)
    metric.precalculate(df, None)

    correct = 2.0

    for idx in itertools.permutations(list(range(len(df)))):
      output = metric(df.iloc[list(idx)])
      self.assertEqual(output, correct)

  def testQuantileInterpolateSame(self):
    df = pd.DataFrame({"X": [1, 1, 2, 2, 3, 4]})

    metric = metrics.Quantile("X", 0.5)
    metric.precalculate(df, None)

    correct = 2.0

    for idx in itertools.permutations(list(range(len(df)))):
      output = metric(df.iloc[list(idx)])
      self.assertEqual(output, correct)

  def testQuantileInterpolateDifferent(self):
    df = pd.DataFrame({"X": [1, 1, 2, 3, 3, 4]})

    metric = metrics.Quantile("X", 0.5)
    metric.precalculate(df, None)

    correct = 2.5

    for idx in itertools.permutations(list(range(len(df)))):
      output = metric(df.iloc[list(idx)])
      self.assertEqual(output, correct)

  def testQuantileMin(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4]})

    metric = metrics.Quantile("X", 0.0)
    metric.precalculate(df, None)

    correct = 1.0

    for idx in itertools.permutations(list(range(len(df)))):
      output = metric(df.iloc[list(idx)])
      self.assertEqual(output, correct)

  def testQuantileMax(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4]})

    metric = metrics.Quantile("X", 1.0)
    metric.precalculate(df, None)

    correct = 4.0

    for idx in itertools.permutations(list(range(len(df)))):
      output = metric(df.iloc[list(idx)])
      self.assertEqual(output, correct)


class VarianceTest(googletest.TestCase):
  """Tests for Variance class."""

  def testVariance(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4]})

    metric = metrics.Variance("X")
    metric.precalculate(df, None)

    output = metric(df)

    correct = 4 / 3

    self.assertEqual(output, correct)


class StandardDeviationTest(googletest.TestCase):
  """Tests for Standard Deviation class."""

  def testStandardDeviation(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4]})

    metric = metrics.StandardDeviation("X")
    metric.precalculate(df, None)

    output = metric(df)

    correct = np.sqrt(4 / 3)

    self.assertEqual(output, correct)


class CVTest(googletest.TestCase):
  """Tests for Standard Deviation class."""

  def testCV(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4]})

    metric = metrics.CV("X")
    metric.precalculate(df, None)

    output = metric(df)

    correct = np.sqrt(1 / 3)

    self.assertEqual(output, correct)


class CorrelationTest(googletest.TestCase):
  """Tests for Standard Deviation class."""

  def testCorrelation(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [3, 1, 1, 4, 4, 3, 5]})

    metric = metrics.Correlation("X", "Y")
    metric.precalculate(df, None)

    output = metric(df)

    correct = 8 / np.sqrt(112)

    self.assertAlmostEqual(output, correct)

  def testWeightedCorrelation(self):
    df = pd.DataFrame({"X": [1, 1, 2, 3, 4],
                       "Y": [3, 1, 4, 3, 5],
                       "W": [1, 2, 2, 1, 1]})

    metric = metrics.WeightedCorrelation("X", "Y", "W")
    metric.precalculate(df, None)

    output = metric(df)

    correct = 8 / np.sqrt(112)

    self.assertEqual(output, correct)


class TestComposition(googletest.TestCase):
  """Tests for composition of two metrics."""

  df = pd.DataFrame({"X": [0, 1, 2, 3],
                     "Y": [0, 1, 1, 2]})

  def testAdd(self):
    metric = 5 + metrics.Sum("X") + metrics.Sum("Y")
    metric.precalculate(self.df, None)

    output = metric(self.df)
    correct = 15

    self.assertEqual(output, correct)
    self.assertEqual(metric.name, "5 + sum(X) + sum(Y)")

  def testSub(self):
    metric = 5 - metrics.Sum("X") - metrics.Sum("Y")
    metric.precalculate(self.df, None)

    output = metric(self.df)
    correct = -5

    self.assertEqual(output, correct)
    self.assertEqual(metric.name, "5 - sum(X) - sum(Y)")

  def testMul(self):
    metric = 2. * metrics.Sum("X") * metrics.Sum("Y")
    metric.precalculate(self.df, None)

    output = metric(self.df)
    correct = 48

    self.assertEqual(output, correct)
    self.assertEqual(metric.name, "2.0 * sum(X) * sum(Y)")

  def testDiv(self):
    metric = 6. / metrics.Sum("X") / metrics.Sum("Y")
    metric.precalculate(self.df, None)

    output = metric(self.df)
    correct = 0.25

    self.assertEqual(output, correct)
    self.assertEqual(metric.name, "6.0 / sum(X) / sum(Y)")

  def testNeg(self):
    metric = -metrics.Sum("X")
    metric.precalculate(self.df, None)

    output = metric(self.df)
    correct = -6

    self.assertEqual(output, correct)
    self.assertEqual(metric.name, "-sum(X)")

  def testPow(self):
    metric = metrics.Sum("X") ** metrics.Sum("Y")
    metric.precalculate(self.df, None)

    output = metric(self.df)
    correct = 1296

    self.assertEqual(output, correct)
    self.assertEqual(metric.name, "sum(X) ^ sum(Y)")

  def testPowWithScalar(self):
    metric = metrics.Sum("X") ** 2
    metric.precalculate(self.df, None)

    output = metric(self.df)
    correct = 36

    self.assertEqual(output, correct)
    self.assertEqual(metric.name, "sum(X) ^ 2")

  def testRpow(self):
    metric = 2 ** metrics.Sum("X")
    metric.precalculate(self.df, None)

    output = metric(self.df)
    correct = 64

    self.assertEqual(output, correct)
    self.assertEqual(metric.name, "2 ^ sum(X)")


if __name__ == "__main__":
  googletest.main()
