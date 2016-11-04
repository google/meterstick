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

"""Tests for meterstick.metrics."""

from __future__ import division

import itertools

import numpy as np
import pandas as pd

from google3.testing.pybase import googletest

from meterstick import metrics


class DistributionTest(googletest.TestCase):
  """Tests for Distribution class."""

  def testDistribution(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [1, 2, 0, 1, 1, 1, 1]})
    weights = np.array([1, 1, 1, 1, 1, 1, 1])
    metric = metrics.Distribution("X", ["Y"])
    output = metric(df, weights)
    correct = pd.DataFrame(
        np.array([6 / 7., 1 / 14., 1 / 14.]),
        columns=[""],
        index=pd.Index([1, 2, 0], name="Y"))
    self.assertTrue(output.equals(correct))

  def testWeightedDistribution(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [1, 2, 0, 1, 1, 1, 1]})
    weights = np.array([1, 7, 1, 1, 1, 1, 1])
    metric = metrics.Distribution("X", ["Y"])
    output = metric(df, weights)
    correct = pd.DataFrame(
        np.array([12 / 20., 7 / 20., 1 / 20.]),
        columns=[""],
        index=pd.Index([1, 2, 0], name="Y"))
    self.assertTrue(output.equals(correct))

  def testTwoDimensionalDistribution(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [1, 2, 0, 1, 1, 1, 1],
                       "Z": [1, 0, 0, 0, 0, 0, 0]})
    weights = np.array([1, 1, 1, 1, 1, 1, 1])
    metric = metrics.Distribution("X", ["Y", "Z"])
    output = metric(df, weights)
    correct = pd.DataFrame(
        np.array([1 / 14., 1 / 14., 1 / 14., 11 / 14.]),
        columns=[""],
        index=pd.MultiIndex(levels=[[0, 1, 2], [0, 1]],
                            labels=[[1, 2, 0, 1], [1, 0, 0, 0]],
                            names=["Y", "Z"]))
    self.assertTrue(output.equals(correct))


class CumulativeDistributionTest(googletest.TestCase):
  """Tests for Cumulative class."""

  def testCumulativeDistribution(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [1, 2, 0, 1, 1, 1, 1]})
    weights = np.array([1, 1, 1, 1, 1, 1, 1])
    metric = metrics.CumulativeDistribution("X", ["Y"])
    output = metric(df, weights)
    correct = pd.DataFrame(
        np.array([1 / 14., 13 / 14., 1.]),
        columns=[""],
        index=pd.Index([0, 1, 2], name="Y"))
    self.assertTrue(all(output.index == correct.index) and
                    all(output.columns == correct.columns) and
                    all(abs(output.values - correct.values) < 1e-10))

  def testWeightedCumulativeDistribution(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [1, 2, 0, 1, 1, 1, 1]})
    weights = np.array([1, 7, 1, 1, 1, 1, 1])
    metric = metrics.CumulativeDistribution("X", ["Y"])
    output = metric(df, weights)
    correct = pd.DataFrame(
        np.array([1 / 20., 13 / 20., 1.]),
        columns=[""],
        index=pd.Index([0, 1, 2], name="Y"))
    self.assertTrue(all(output.index == correct.index) and
                    all(output.columns == correct.columns) and
                    all(abs(output.values - correct.values) < 1e-10))

  def testTwoDimensionalCumulativeDistribution(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [1, 2, 0, 1, 1, 1, 1],
                       "Z": [1, 0, 0, 0, 0, 0, 0]})
    weights = np.array([1, 1, 1, 1, 1, 1, 1])
    metric = metrics.CumulativeDistribution("X", ["Y", "Z"])
    output = metric(df, weights)
    correct = pd.DataFrame(
        np.array([1 / 14., 12 / 14., 13 / 14., 1.]),
        columns=[""],
        index=pd.MultiIndex(levels=[[0, 1, 2], [0, 1]],
                            labels=[[0, 1, 1, 2], [0, 0, 1, 0]],
                            names=["Y", "Z"]))
    self.assertTrue(all(output.index == correct.index) and
                    all(output.columns == correct.columns) and
                    all(abs(output.values - correct.values) < 1e-10))

  def testShuffledTwoDimensionalCumulativeDistribution(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [1, 2, 0, 1, 1, 1, 1],
                       "Z": [1, 0, 0, 0, 0, 0, 0]})
    weights = np.array([1, 1, 1, 1, 1, 1, 1])
    metric = metrics.CumulativeDistribution("X", ["Y", "Z"])
    output = metric(df.iloc[np.random.permutation(7)], weights)
    correct = pd.DataFrame(
        np.array([1 / 14., 12 / 14., 13 / 14., 1.]),
        columns=[""],
        index=pd.MultiIndex(levels=[[0, 1, 2], [0, 1]],
                            labels=[[0, 1, 1, 2], [0, 0, 1, 0]],
                            names=["Y", "Z"]))
    self.assertTrue(all(output.index == correct.index) and
                    all(output.columns == correct.columns) and
                    all(abs(output.values - correct.values) < 1e-10))


class RatioTest(googletest.TestCase):
  """Tests for Ratio class."""

  def testRatio(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [1, 2, 0, 1, 1, 1, 1]})
    weights = np.array([1, 1, 1, 1, 1, 1, 1])

    metric = metrics.Ratio("X", "Y")

    output = metric(df, weights)

    correct = 2.0

    self.assertEqual(output, correct)

  def testRatioPositiveDivideByZero(self):
    df = pd.DataFrame({"X": [1, 1], "Y": [0, 0]})

    weights = np.array([1, 1])

    metric = metrics.Ratio("X", "Y")

    output = metric(df, weights)

    correct = np.inf

    self.assertEqual(output, correct)

  def testRatioZeroDivideByZero(self):
    df = pd.DataFrame({"X": [0, 0], "Y": [0, 0]})

    weights = np.array([1, 1])

    metric = metrics.Ratio("X", "Y")

    output = metric(df, weights)

    self.assertTrue(np.isnan(output))

  def testRatioWithWeights(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 3, 4], "Y": [1, 2, 0, 1, 1, 1]})
    weights = np.array([1, 1, 1, 2, 1, 1])

    metric = metrics.Ratio("X", "Y")

    output = metric(df, weights)
    correct = 2.0

    self.assertEqual(output, correct)


class SumTest(googletest.TestCase):
  """Tests for Sum class."""

  def testSum(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4]})
    weights = np.array([1, 1, 1, 1, 1, 1, 1])

    metric = metrics.Sum("X")

    output = metric(df, weights)

    correct = 14

    self.assertEqual(output, correct)

  def testSumWithWeights(self):
    df = pd.DataFrame({"X": [1, 2, 3, 4]})
    weights = np.array([3, 2, 1, 1])

    metric = metrics.Sum("X")

    output = metric(df, weights)

    correct = 14

    self.assertEqual(output, correct)


class MeanTest(googletest.TestCase):
  """Tests for Sum class."""

  def testMean(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4]})
    weights = np.array([1, 1, 1, 1, 1, 1, 1])

    metric = metrics.Mean("X")

    output = metric(df, weights)

    correct = 2.0

    self.assertEqual(output, correct)

  def testMeanWithWeights(self):
    df = pd.DataFrame({"X": [1, 2, 3, 4]})
    weights = np.array([3, 2, 1, 1])

    metric = metrics.Mean("X")

    output = metric(df, weights)

    correct = 2.0

    self.assertEqual(output, correct)


class WeightedMeanTest(googletest.TestCase):
  """Tests for WeightedMean class."""

  def testEqualWeightedMean(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [1, 1, 1, 1, 1, 1, 1]})
    weights = np.array([1, 1, 1, 1, 1, 1, 1])

    metric = metrics.WeightedMean("X", "Y")

    output = metric(df, weights)

    correct = 2.0

    self.assertEqual(output, correct)

  def testEqualWeightedMeanWithWeights(self):
    df = pd.DataFrame({"X": [1, 2, 3, 4], "Y": [1, 1, 1, 1]})
    weights = np.array([3, 2, 1, 1])

    metric = metrics.WeightedMean("X", "Y")

    output = metric(df, weights)

    correct = 2.0

    self.assertEqual(output, correct)

  def testWeightedMean(self):
    df = pd.DataFrame({"X": [1, 1, 2, 2, 2, 3, 4],
                       "Y": [1, 2, 3, 4, 3, 1, 1]})
    weights = np.array([1, 1, 1, 1, 1, 1, 1])

    metric = metrics.WeightedMean("X", "Y")

    output = metric(df, weights)

    correct = 2.0

    self.assertEqual(output, correct)

  def testWeightedMeanWithWeights(self):
    df = pd.DataFrame({"X": [1, 1, 2, 2, 3, 4], "Y": [1, 2, 3, 4, 1, 1]})
    weights = np.array([1, 1, 1, 2, 1, 1])

    metric = metrics.WeightedMean("X", "Y")

    output = metric(df, weights)

    correct = 2.0

    self.assertEqual(output, correct)


class QuantileTest(googletest.TestCase):
  """Tests for Quantile class."""

  def testQuantile(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4]})
    weights = np.array([1, 1, 1, 1, 1, 1, 1])

    metric = metrics.Quantile("X", 0.5)

    correct = 2.0

    for idx in itertools.permutations(range(len(weights))):
      output = metric(df.iloc[list(idx)], weights[list(idx)])
      self.assertEqual(output, correct)

  def testQuantileInterpolateSame(self):
    df = pd.DataFrame({"X": [1, 1, 2, 2, 3, 4]})
    weights = np.array([1, 1, 1, 1, 1, 1])

    metric = metrics.Quantile("X", 0.5)

    correct = 2.0

    for idx in itertools.permutations(range(len(weights))):
      output = metric(df.iloc[list(idx)], weights[list(idx)])
      self.assertEqual(output, correct)

  def testQuantileInterpolateSameWithWeights(self):
    df = pd.DataFrame({"X": [1, 2, 3, 4]})
    weights = np.array([2, 2, 1, 1])

    metric = metrics.Quantile("X", 0.5)

    correct = 2.0

    for idx in itertools.permutations(range(len(weights))):
      output = metric(df.iloc[list(idx)], weights[list(idx)])
      self.assertEqual(output, correct)

  def testQuantileInterpolateDifferent(self):
    df = pd.DataFrame({"X": [1, 1, 2, 3, 3, 4]})
    weights = np.array([1, 1, 1, 1, 1, 1])

    metric = metrics.Quantile("X", 0.5)

    correct = 2.5

    for idx in itertools.permutations(range(len(weights))):
      output = metric(df.iloc[list(idx)], weights[list(idx)])
      self.assertEqual(output, correct)

  def testQuantileInterpolateDifferentWithWeights(self):
    df = pd.DataFrame({"X": [1, 2, 3, 4]})
    weights = np.array([2, 1, 2, 1])

    metric = metrics.Quantile("X", 0.5)

    correct = 2.5

    for idx in itertools.permutations(range(len(weights))):
      output = metric(df.iloc[list(idx)], weights[list(idx)])
      self.assertEqual(output, correct)

  def testQuantileMin(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4]})
    weights = np.array([1, 1, 1, 1, 1, 1, 1])

    metric = metrics.Quantile("X", 0.0)

    correct = 1.0

    for idx in itertools.permutations(range(len(weights))):
      output = metric(df.iloc[list(idx)], weights[list(idx)])
      self.assertEqual(output, correct)

  def testQuantileMinWithWeights(self):
    df = pd.DataFrame({"X": [1, 2, 3, 4]})
    weights = np.array([3, 2, 1, 1])

    metric = metrics.Quantile("X", 0.0)

    correct = 1.0

    for idx in itertools.permutations(range(len(weights))):
      output = metric(df.iloc[list(idx)], weights[list(idx)])
      self.assertEqual(output, correct)

  def testQuantileMax(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4]})
    weights = np.array([1, 1, 1, 1, 1, 1, 1])

    metric = metrics.Quantile("X", 1.0)

    correct = 4.0

    for idx in itertools.permutations(range(len(weights))):
      output = metric(df.iloc[list(idx)], weights[list(idx)])
      self.assertEqual(output, correct)

  def testQuantileWithWeights(self):
    df = pd.DataFrame({"X": [1, 2, 3, 4]})
    weights = np.array([3, 2, 1, 1])

    metric = metrics.Quantile("X", 0.5)

    correct = 2.0

    for idx in itertools.permutations(range(len(weights))):
      output = metric(df.iloc[list(idx)], weights[list(idx)])
      self.assertEqual(output, correct)


class VarianceTest(googletest.TestCase):
  """Tests for Variance class."""

  def testVariance(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4]})
    weights = np.array([1, 1, 1, 1, 1, 1, 1])

    metric = metrics.Variance("X")

    output = metric(df, weights)

    correct = 4 / 3

    self.assertEqual(output, correct)

  def testVarianceWeighted(self):
    df = pd.DataFrame({"X": [1, 2, 3, 4]})
    weights = np.array([3, 2, 1, 1])

    metric = metrics.Variance("X")

    output = metric(df, weights)

    correct = 4 / 3

    self.assertEqual(output, correct)


class StandardDeviationTest(googletest.TestCase):
  """Tests for Standard Deviation class."""

  def testStandardDeviation(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4]})
    weights = np.array([1, 1, 1, 1, 1, 1, 1])

    metric = metrics.StandardDeviation("X")

    output = metric(df, weights)

    correct = np.sqrt(4 / 3)

    self.assertEqual(output, correct)

  def testStandardDeviationWeighted(self):
    df = pd.DataFrame({"X": [1, 2, 3, 4]})
    weights = np.array([3, 2, 1, 1])

    metric = metrics.StandardDeviation("X")

    output = metric(df, weights)

    correct = np.sqrt(4 / 3)

    self.assertEqual(output, correct)


class CVTest(googletest.TestCase):
  """Tests for Standard Deviation class."""

  def testCV(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4]})
    weights = np.array([1, 1, 1, 1, 1, 1, 1])

    metric = metrics.CV("X")

    output = metric(df, weights)

    correct = np.sqrt(1 / 3)

    self.assertEqual(output, correct)

  def testCVWeighted(self):
    df = pd.DataFrame({"X": [1, 2, 3, 4]})
    weights = np.array([3, 2, 1, 1])

    metric = metrics.CV("X")

    output = metric(df, weights)

    correct = np.sqrt(1 / 3)

    self.assertEqual(output, correct)


class CorrelationTest(googletest.TestCase):
  """Tests for Standard Deviation class."""

  def testCorrelation(self):
    df = pd.DataFrame({"X": [1, 1, 1, 2, 2, 3, 4],
                       "Y": [3, 1, 1, 4, 4, 3, 5]})
    weights = np.array([1, 1, 1, 1, 1, 1, 1])

    metric = metrics.Correlation("X", "Y")

    output = metric(df, weights)

    correct = 8 / np.sqrt(112)

    self.assertEqual(output, correct)

  def testCorrelationWeighted(self):
    df = pd.DataFrame({"X": [1, 1, 2, 3, 4], "Y": [3, 1, 4, 3, 5]})
    weights = np.array([1, 2, 2, 1, 1])

    metric = metrics.Correlation("X", "Y")

    output = metric(df, weights)

    correct = 8 / np.sqrt(112)

    self.assertEqual(output, correct)


if __name__ == "__main__":
  googletest.main()
