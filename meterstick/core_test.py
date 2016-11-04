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

"""Tests for core."""

from __future__ import division

import numpy as np
import pandas as pd
import scipy.stats

from google3.testing.pybase import googletest

from meterstick import comparisons
from meterstick import core
from meterstick import metrics
from meterstick import standard_errors


class AnalysisTest(googletest.TestCase):
  """Tests for Analysis class."""

  def testCalculate(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5]})

    metric = metrics.Sum("X")
    output = core.Analyze(data).calculate(metric).run()

    correct = pd.DataFrame(np.array([[15]]), columns=["sum(X)"])

    self.assertTrue(output.equals(correct))

  def testNamingCalculations(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5]})

    metric = metrics.Sum("X", name="X-Total")
    output = core.Analyze(data).calculate(metric).run()

    correct = pd.DataFrame(np.array([[15]]), columns=["X-Total"])

    self.assertTrue(output.equals(correct))

  def testWhere(self):
    data = pd.DataFrame({"X": (1, 2, 3, 10, 20, 30, 100, 200, 300),
                         "Y": (0, 1, 2, 3, 4, 5, 6, 7, 8)})

    metric = metrics.Sum("X")
    output = core.Analyze(data).where("Y >= 4").calculate(metric).run()

    correct = pd.DataFrame(np.array([[650]]), columns=["sum(X)"])

    self.assertTrue(output.equals(correct))

  def testMultipleWhere(self):
    data = pd.DataFrame({"X": (1, 2, 3, 10, 20, 30, 100, 200, 300),
                         "Y": (0, 1, 2, 3, 4, 5, 6, 7, 8)})

    metric = metrics.Sum("X")
    output = core.Analyze(data).where("Y >= 4").where("Y <= 6").calculate(
        metric).run()

    correct = pd.DataFrame(np.array([[150]]), columns=["sum(X)"])

    self.assertTrue(output.equals(correct))

  def testBadWhereRaisesError(self):
    data = pd.DataFrame({"X": (1, 2, 3, 10, 20, 30, 100, 200, 300),
                         "Y": (0, 1, 2, 3, 4, 5, 6, 7, 8)})

    metric = metrics.Sum("X")
    with self.assertRaises(ValueError):
      core.Analyze(data).where("X + Y").calculate(metric).run()

  def testMultipleCalculations(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5]})

    output = core.Analyze(data).calculate(
        [metrics.Sum("X"), metrics.Mean("X")]).run()

    correct = pd.DataFrame(np.array([[15, 3.0]]), columns=["sum(X)", "mean(X)"])
    correct[["sum(X)"]] = correct[["sum(X)"]].astype(int)

    self.assertTrue(output.equals(correct))

  def testMultipleCalculationsTuple(self):
    data = pd.DataFrame({"X": (1, 2, 3, 4, 5)})

    output = core.Analyze(data).calculate(
        (metrics.Sum("X"), metrics.Mean("X"))).run()

    correct = pd.DataFrame(np.array([[15, 3.0]]), columns=["sum(X)", "mean(X)"])
    correct[["sum(X)"]] = correct[["sum(X)"]].astype(int)

    self.assertTrue(output.equals(correct))

  def testMultipleCalculationsRelativeTo(self):
    data = pd.DataFrame({"X": (1, 2, 3, 10, 20, 30, 100, 200, 300),
                         "Y": (0, 1, 2, 3, 4, 5, 6, 7, 8),
                         "Experiment": ("Control", "Control", "Control", "Exp1",
                                        "Exp1", "Exp1", "Exp2", "Exp2",
                                        "Exp2")})

    comparison = comparisons.AbsoluteDifference("Experiment", "Control")
    output = core.Analyze(data).relative_to(comparison).calculate(
        (metrics.Sum("X"), metrics.Sum("Y"))).run()

    correct = pd.DataFrame(
        {"sum(X) Absolute Difference": (60 - 6, 600 - 6),
         "sum(Y) Absolute Difference": (12 - 3, 21 - 3)},
        index=pd.Index(
            ("Exp1", "Exp2"), name="Experiment"))

    self.assertTrue(output.equals(correct))

  def testSplitBy(self):
    data = pd.DataFrame({"X": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                         "Y": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]})

    metric = metrics.Sum("X")
    output = core.Analyze(data).calculate(metric).split_by("Y").run()

    correct = pd.DataFrame({"sum(X)": [1, 5, 9, 13, 17],
                            "Y": [1, 2, 3, 4, 5]})
    correct = correct.set_index("Y")

    self.assertTrue(output.equals(correct))

  def testMultipleSplitBy(self):
    data = pd.DataFrame({"X": [4, 5, 6, 7, 0, 1, 2, 3],
                         "Y": [1, 1, 1, 1, 0, 0, 0, 0],
                         "Z": [0, 0, 1, 1, 0, 0, 1, 1]})

    metric = metrics.Sum("X")
    output = core.Analyze(data).split_by(["Y", "Z"]).calculate(metric).run()

    correct = pd.DataFrame({"sum(X)": [1, 5, 9, 13],
                            "Y": [0, 0, 1, 1],
                            "Z": [0, 1, 0, 1]})
    correct = correct.set_index(["Y", "Z"])

    self.assertTrue(output.equals(correct))

  def testSortTrue(self):
    data = pd.DataFrame({"X": [6, 5, 4, 7, 0, 1, 2, 3],
                         "Y": [1, 1, 1, 1, 0, 0, 0, 0],
                         "Z": [1, 0, 0, 1, 0, 0, 1, 1]})

    metric = metrics.Sum("X")
    output = core.Analyze(data).split_by(["Y", "Z"]).calculate(
        metric).run(sort=True)

    correct = pd.DataFrame({"sum(X)": [1, 5, 9, 13],
                            "Y": [0, 0, 1, 1],
                            "Z": [0, 1, 0, 1]})
    correct = correct.set_index(["Y", "Z"])

    self.assertTrue(output.equals(correct))

  def testSortFalse(self):
    data = pd.DataFrame({"X": [6, 5, 4, 7, 0, 1, 2, 3],
                         "Y": [1, 1, 1, 1, 0, 0, 0, 0],
                         "Z": [1, 0, 0, 1, 0, 0, 1, 1]})

    metric = metrics.Sum("X")
    output = core.Analyze(data).split_by(["Y", "Z"]).calculate(
        metric).run(sort=False)

    correct = pd.DataFrame({"sum(X)": [13, 9, 1, 5],
                            "Y": [1, 1, 0, 0],
                            "Z": [1, 0, 0, 1]})
    correct = correct.set_index(["Y", "Z"])

    self.assertTrue(output.equals(correct))

  def testJackknife(self):
    data = pd.DataFrame({"X": range(11)})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife()
    output = core.Analyze(data).with_standard_errors(se_method).calculate(
        metric).run()

    correct = pd.DataFrame(
        np.array([[55.0, 10.0]]), columns=("sum(X)", "sum(X) Jackknife SE"))

    self.assertTrue(output.equals(correct))

  def testNinetyFiveCIs(self):
    data = pd.DataFrame({"X": range(11)})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife(confidence=0.95)
    output = core.Analyze(data).with_standard_errors(se_method).calculate(
        metric).run()

    multiplier = scipy.stats.t.ppf(0.975, 10)
    correct_sd = 10.0

    correct_mean = 55.0
    correct_lower = correct_mean - multiplier * correct_sd
    correct_upper = correct_mean + multiplier * correct_sd

    correct = pd.DataFrame(
        np.array([[correct_mean, correct_lower, correct_upper]]),
        columns=("sum(X)", "sum(X) Jackknife CI-lower",
                 "sum(X) Jackknife CI-upper"))

    self.assertTrue(output.equals(correct))

  def testNinetyFiveCIsWithComparison(self):
    data = pd.DataFrame({"X": range(11),
                         "Y": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 0)
    se_method = standard_errors.Jackknife(confidence=0.95)
    output = core.Analyze(data).with_standard_errors(se_method).relative_to(
        comparison).calculate(metric).run()

    multiplier = scipy.stats.t.ppf(0.975, 10)
    correct_mean = 25
    correct_buckets = [15., 16., 17., 18., 19., 25., 26., 27., 28., 29., 30.]
    m = sum(correct_buckets) / len(correct_buckets)
    r = sum([(b - m) ** 2 for b in correct_buckets])
    correct_sd = np.sqrt(r * (len(correct_buckets) - 1) / len(correct_buckets))

    correct_lower = correct_mean - multiplier * correct_sd
    correct_upper = correct_mean + multiplier * correct_sd

    rowindex = pd.Index([1], name="Y")
    correct = pd.DataFrame(
        {"sum(X) Absolute Difference": correct_mean,
         "sum(X) Absolute Difference Jackknife CI-lower": correct_lower,
         "sum(X) Absolute Difference Jackknife CI-upper": correct_upper},
        index=rowindex)

    self.assertTrue(output.equals(correct))

  def testFiftyCIs(self):
    data = pd.DataFrame({"X": range(11)})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife(confidence=0.50)
    output = core.Analyze(data).with_standard_errors(se_method).calculate(
        metric).run()

    multiplier = scipy.stats.t.ppf(0.75, 10)
    correct_sd = 10.0

    correct_mean = 55.0
    correct_lower = correct_mean - multiplier * correct_sd
    correct_upper = correct_mean + multiplier * correct_sd

    correct = pd.DataFrame(
        np.array([[correct_mean, correct_lower, correct_upper]]),
        columns=("sum(X)", "sum(X) Jackknife CI-lower",
                 "sum(X) Jackknife CI-upper"))

    self.assertTrue(output.equals(correct))

  def testBadConfidenceRaisesException(self):
    with self.assertRaises(ValueError):
      standard_errors.Jackknife(confidence=95)

  def testJackknifeRatio(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4], "Y": [4, 3, 2, 1]})

    metric = metrics.Ratio("X", "Y")
    se_method = standard_errors.Jackknife()
    output = core.Analyze(data).with_standard_errors(se_method).calculate(
        metric).run()

    estimates = np.array([9 / 6, 8 / 7, 7 / 8, 6 / 9])
    rss = ((estimates - estimates.mean())**2).sum()
    se = np.sqrt(rss * 3 / 4)

    correct = pd.DataFrame([[1.0, se]], columns=("X/Y", "X/Y Jackknife SE"))

    self.assertTrue(output.equals(correct))

  def testBootstrap(self):
    # The bootstrap depends upon random values to work; thus, we'll
    # only check that it's statistically close to the theoretical
    # value.

    # We set the seed to avoid flaky tests; this test will fail with
    # probability 0.05 otherwise.
    np.random.seed(12345)

    data = pd.DataFrame({"X": range(1, 101)})

    metric = metrics.Mean("X")
    se_method = standard_errors.Bootstrap(100)
    output = core.Analyze(data).with_standard_errors(se_method).calculate(
        metric).run()

    bootstrap_se = output["mean(X) Bootstrap SE"].values[0]

    # Parameters based on the following R simulation
    # set.seed(12345)
    # library(bootstrap)
    # x <- 1:100
    # estimates <- replicate(1000, sd(bootstrap(x, 100, mean)$thetastar))
    # mean(estimates)
    # sd(estimates)

    simulation_se = 2.88
    epsilon = 0.41  # Two standard errors based on simulation.

    self.assertAlmostEqual(simulation_se, bootstrap_se, delta=epsilon)

  def testUnitBootstrap(self):
    # The bootstrap depends upon random values to work; thus, we'll
    # only check that it's statistically close to a simulated value.

    # We set the seed to avoid flaky tests; this test will fail with
    # probability 0.05 otherwise.

    # Note this is an equivalent problem to the testBootstrap case,
    # we've just split some rows.

    np.random.seed(12345)

    x = []
    y = []
    for ii in range(1, 101):
      for _ in range(3):
        x.append(ii)
        y.append(ii)

    data = pd.DataFrame({"X": x, "Y": y})

    metric = metrics.Mean("X")
    se_method = standard_errors.Bootstrap(100, unit="Y")
    output = core.Analyze(data).with_standard_errors(se_method).calculate(
        metric).run()

    bootstrap_se = output["mean(X) Bootstrap SE"].values[0]

    simulation_se = 2.88
    epsilon = 0.41  # Two standard errors based on simulation.

    self.assertAlmostEqual(simulation_se, bootstrap_se, delta=epsilon)

  def testRelativeTo(self):
    data = pd.DataFrame({"X": [1, 2, 3, 10, 20, 30, 100, 200, 300],
                         "Y": [0, 0, 0, 1, 1, 1, 2, 2, 2]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 0)
    output = core.Analyze(data).relative_to(comparison).calculate(metric).run()

    correct = pd.DataFrame({"sum(X) Absolute Difference": [60 - 6, 600 - 6],
                            "Y": [1, 2]})

    correct = correct.set_index("Y")

    self.assertTrue(output.equals(correct))

  def testRelativeToSplit(self):
    data = pd.DataFrame(
        {"X": [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
         "Y": [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2],
         "Z": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 0)
    output = core.Analyze(data).split_by("Z").relative_to(comparison).calculate(
        metric).run()

    correct = pd.DataFrame(
        {"sum(X) Absolute Difference": [13 - 5, 23 - 5, 14 - 4, 22 - 4],
         "Z": [0, 0, 1, 1],
         "Y": [1, 2, 1, 2]})

    correct = correct.set_index(["Z", "Y"])

    self.assertTrue(output.equals(correct))

  def testRelativeToJackknife(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                         "Y": [0, 0, 0, 1, 1, 1, 2, 2, 2]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 0)
    se_method = standard_errors.Jackknife()
    output = core.Analyze(data).relative_to(comparison).with_standard_errors(
        se_method).calculate(metric).run()

    rowindex = pd.Index([1, 2], name="Y")
    correct = pd.DataFrame(
        np.array([[9.0, np.sqrt(5 * np.var([12, 11, 10, 5, 4, 3]))],
                  [18.0, np.sqrt(5 * np.var([21, 20, 19, 11, 10, 9]))]]),
        columns=("sum(X) Absolute Difference",
                 "sum(X) Absolute Difference Jackknife SE"),
        index=rowindex)

    self.assertTrue(output.equals(correct))

  def testRelativeToJackknifeIncludeBaseline(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                         "Y": [0, 0, 0, 1, 1, 1, 2, 2, 2]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 0, include_base=True)
    se_method = standard_errors.Jackknife()
    output = core.Analyze(data).relative_to(comparison).with_standard_errors(
        se_method).calculate(metric).run()

    rowindex = pd.Index([0, 1, 2], name="Y")
    correct = pd.DataFrame(
        np.array([[0.0, 0.0],
                  [9.0, np.sqrt(5 * np.var([12, 11, 10, 5, 4, 3]))],
                  [18.0, np.sqrt(5 * np.var([21, 20, 19, 11, 10, 9]))]]),
        columns=("sum(X) Absolute Difference",
                 "sum(X) Absolute Difference Jackknife SE"),
        index=rowindex)

    self.assertTrue(output.equals(correct))

  def testRelativeToJackknifeSingleComparisonBaselineFirst(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5, 6], "Y": [0, 0, 0, 1, 1, 1]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 0)
    se_method = standard_errors.Jackknife()
    output = core.Analyze(data).relative_to(comparison).with_standard_errors(
        se_method).calculate(metric).run()

    rowindex = pd.Index([1], name="Y")
    correct = pd.DataFrame(
        np.array([[9.0, np.sqrt(5 * np.var([12, 11, 10, 5, 4, 3]))]]),
        columns=("sum(X) Absolute Difference",
                 "sum(X) Absolute Difference Jackknife SE"),
        index=rowindex)

    self.assertTrue(output.equals(correct))

  def testRelativeToJackknifeSingleComparisonBaselineSecond(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5, 6], "Y": [0, 0, 0, 1, 1, 1]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 1)
    se_method = standard_errors.Jackknife()
    output = core.Analyze(data).relative_to(comparison).with_standard_errors(
        se_method).calculate(metric).run()

    rowindex = pd.Index([0], name="Y")
    correct = pd.DataFrame(
        np.array([[-9.0, np.sqrt(5 * np.var([12, 11, 10, 5, 4, 3]))]]),
        columns=("sum(X) Absolute Difference",
                 "sum(X) Absolute Difference Jackknife SE"),
        index=rowindex)

    self.assertTrue(output.equals(correct))

  def testSplitJackknife(self):
    data = pd.DataFrame({"X": np.array([range(11) + [5] * 10]).flatten(),
                         "Y": np.array([[0] * 11 + [1] * 10]).flatten()})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife()
    output = core.Analyze(data).split_by("Y").with_standard_errors(
        se_method).calculate(metric).run()

    rowindex = pd.Index([0, 1], name="Y")
    correct = pd.DataFrame(
        np.array([[55.0, 10.0], [50.0, 0.0]]),
        columns=("sum(X)", "sum(X) Jackknife SE"),
        index=rowindex)

    self.assertTrue(output.equals(correct))

  def testRelativeToSplitJackknife(self):
    data = pd.DataFrame(
        {"X": [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
         "Y": [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3],
         "Z": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Z", 0)
    se_method = standard_errors.Jackknife()
    output = core.Analyze(data).split_by("Y").relative_to(
        comparison).with_standard_errors(se_method).calculate(metric).run()

    rowindex = pd.MultiIndex(
        levels=[[1, 2, 3], [1]],
        labels=[[0, 1, 2], [0, 0, 0]],
        names=["Y", "Z"])
    correct = pd.DataFrame(
        np.array([[-3.0, np.sqrt(5 * np.var([0, -1, -2, -3, -4, -5]))],
                  [-3.0, np.sqrt(5 * np.var([3, 2, 1, -8, -7, -6]))],
                  [-3.0, np.sqrt(5 * np.var([6, 5, 4, -11, -10, -9]))]]),
        columns=("sum(X) Absolute Difference",
                 "sum(X) Absolute Difference Jackknife SE"),
        index=rowindex)

    self.assertTrue(output.equals(correct))

  def testDoubleComparisonDefinitionRaisesException(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5]})

    comparison = comparisons.AbsoluteDifference("X", 0)
    with self.assertRaises(ValueError):
      core.Analyze(data).relative_to(comparison).relative_to(comparison)

  def testDoubleSEMethodDefinitionRaisesException(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5]})

    se_method = standard_errors.Jackknife()
    with self.assertRaises(ValueError):
      core.Analyze(data).with_standard_errors(se_method).with_standard_errors(
          se_method)

  def testSingleJackknifeBucketRaisesException(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5], "Y": [1, 1, 1, 1, 1]})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife(unit="Y")

    with self.assertRaises(ValueError):
      core.Analyze(data).with_standard_errors(se_method).calculate(metric).run()

  def testJackknifeBadSample(self):
    data = pd.DataFrame({"X": range(22), "Y": ([0] * 11) + ([1] * 11)})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife()
    output = core.Analyze(data).split_by("Y").with_standard_errors(
        se_method).calculate(metric).run()

    correct = pd.DataFrame(
        np.array([[55.0, 10.0], [176.0, 10.0]]),
        columns=("sum(X)", "sum(X) Jackknife SE"))

    correct.index.name = "Y"

    self.assertTrue(output.equals(correct))

  def testJackknifeOutOfRangeBins(self):
    data = pd.DataFrame({"X": range(11) + range(11),
                         "Y": range(22),
                         "Z": ([0] * 11 + [1] * 11)})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife("Y")
    output = core.Analyze(data).split_by("Z").with_standard_errors(
        se_method).calculate(metric).run()

    correct = pd.DataFrame(
        np.array([[55.0, 10.0], [55.0, 10.0]]),
        columns=("sum(X)", "sum(X) Jackknife SE"))

    correct.index.name = "Z"

    self.assertTrue(output.equals(correct))

  def testComparisonBaslineGivesError(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5], "Y": [1, 1, 1, 1, 1]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 0)

    with self.assertRaises(ValueError):
      core.Analyze(data).relative_to(comparison).calculate(metric).run()

  def testRelativeToSplitsWithNoAlternativeGivesNaN(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4],
                         "Y": [0, 0, 0, 1],
                         "Z": [0, 0, 1, 1]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 0)
    output = core.Analyze(data).split_by("Z").relative_to(comparison).calculate(
        metric).run()

    correct = pd.DataFrame({"sum(X) Absolute Difference": [np.nan, 4 - 3],
                            "Z": [0, 1],
                            "Y": [1, 1]})
    correct = correct.set_index(["Z", "Y"])

    self.assertTrue(output.equals(correct))

  def testDataframeJackknife(self):
    df = pd.DataFrame({"X": range(11),
                       "Y": np.concatenate((np.zeros(6), np.ones(5))),
                       "Z": np.concatenate((np.zeros(3), np.ones(8)))})

    metric = metrics.Distribution("X", ["Z"])
    se_method = standard_errors.Jackknife("Y")
    output = core.Analyze(df).with_standard_errors(se_method).calculate(
        metric).run()

    correct = pd.DataFrame(
        np.array([[3 / 55., np.sqrt(((3 / 15. - 0.1)**2 + 0.1**2) / 2.)],
                  [52 / 55., np.sqrt(((12 / 15. - 0.9)**2 + 0.1**2) / 2.)]]),
        columns=("X Distribution", "X Distribution Jackknife SE"),
        index=pd.Index([0., 1.], name="Z"))

    self.assertTrue(all(output.index == correct.index) and
                    all(output.columns == correct.columns) and
                    np.all(abs(output.values - correct.values) < 1e-10))

  def testDataframeRelativeTo(self):
    df = pd.DataFrame({"X": range(11),
                       "Y": np.concatenate((np.zeros(6), np.ones(5))),
                       "Z": np.concatenate((np.zeros(3), np.ones(8)))})

    metric = metrics.Distribution("X", ["Z"])
    output = core.Analyze(df).relative_to(comparisons.AbsoluteDifference(
        "Y", 0)).calculate(metric).run()

    correct = pd.DataFrame(
        np.array([-0.2, 0.2]),
        columns=["X Distribution Absolute Difference"],
        index=pd.MultiIndex(levels=[[1.], [0., 1.]],
                            labels=[[0, 0], [0, 1]],
                            names=["Y", "Z"]))

    self.assertTrue(all(output.index == correct.index) and
                    all(output.columns == correct.columns) and
                    np.all(abs(output.values - correct.values) < 1e-10))

  def testDataframeRelativeToJackknife(self):
    df = pd.DataFrame({"X": range(11),
                       "Y": np.concatenate((np.zeros(6), np.ones(5))),
                       "Z": np.concatenate((np.zeros(3), np.ones(8)))})

    metric = metrics.Distribution("X", ["Z"])
    se_method = standard_errors.Jackknife()
    output = core.Analyze(df).relative_to(comparisons.AbsoluteDifference(
        "Y", 0)).with_standard_errors(se_method).calculate(metric).run()

    correct = pd.DataFrame(
        np.array([[-0.2, 0.18100283490],
                  [0.2, 0.18100283490]]),
        columns=["X Distribution Absolute Difference",
                 "X Distribution Absolute Difference Jackknife SE"],
        index=pd.MultiIndex(levels=[[1.], [0., 1.]],
                            labels=[[0, 0], [0, 1]],
                            names=["Y", "Z"]))

    self.assertTrue(all(output.index == correct.index) and
                    all(output.columns == correct.columns) and
                    np.all(abs(output.values - correct.values) < 1e-10))

  def testShuffledDataframeRelativeToJackknife(self):
    # Same as test above, but also testing that reordering the data doesn't
    # change results, up to order.
    df = pd.DataFrame({"X": range(11),
                       "Y": np.concatenate((np.zeros(6), np.ones(5))),
                       "Z": np.concatenate((np.zeros(3), np.ones(8)))})

    metric = metrics.Distribution("X", ["Z"])
    se_method = standard_errors.Jackknife()
    output = core.Analyze(df.iloc[np.random.permutation(11)]).relative_to(
        comparisons.AbsoluteDifference("Y", 0)).with_standard_errors(
            se_method).calculate(metric).run()
    output = (output.
              reset_index().
              sort_values(by=["Y", "Z"]).
              set_index(["Y", "Z"]))

    correct = pd.DataFrame(
        np.array([[-0.2, 0.18100283490],
                  [0.2, 0.18100283490]]),
        columns=["X Distribution Absolute Difference",
                 "X Distribution Absolute Difference Jackknife SE"],
        index=pd.MultiIndex(levels=[[1.], [0., 1.]],
                            labels=[[0, 0], [0, 1]],
                            names=["Y", "Z"]))
    correct = (correct.
               reset_index().
               sort_values(by=["Y", "Z"]).
               set_index(["Y", "Z"]))

    self.assertTrue(all(output.index == correct.index) and
                    all(output.columns == correct.columns) and
                    np.all(abs(output.values - correct.values) < 1e-10))

  def testSplitDataframe(self):
    df = pd.DataFrame({"X": range(11),
                       "Y": np.concatenate((np.zeros(6), np.ones(5))),
                       "Z": np.concatenate((np.zeros(3), np.ones(8)))})

    metric = metrics.Distribution("X", ["Z"])
    output = core.Analyze(df).split_by(["Y"]).calculate(metric).run()

    correct = pd.DataFrame(
        np.array([0.2, 0.8, 0.0, 1.0]),
        columns=["X Distribution"],
        index=pd.MultiIndex(levels=[[0.0, 1.0], [0.0, 1.0]],
                            labels=[[0, 0, 1, 1], [0, 1, 0, 1]],
                            names=["Y", "Z"]))

    self.assertTrue(all(output.index == correct.index) and
                    all(output.columns == correct.columns) and
                    np.all(abs(output.values - correct.values) < 1e-10))

if __name__ == "__main__":
  googletest.main()
