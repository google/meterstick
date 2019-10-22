# Lint as: python2, python3
"""Tests for meterstick.core."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import pandas as pd
import scipy.stats
from six.moves import range
from six.moves import zip

from meterstick import comparisons
from meterstick import core
from meterstick import metrics
from meterstick import pdutils
from meterstick import standard_errors
from google3.testing.pybase import googletest


class AnalysisTest(googletest.TestCase):
  """Tests for Analysis class."""

  def testCalculate(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5]})

    metric = metrics.Sum("X")
    output = core.Analyze(data).calculate(metric).run()

    correct = pd.DataFrame(np.array([[15]]), columns=["sum(X)"])

    pd.util.testing.assert_frame_equal(output, correct)

  def testNamingCalculations(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5]})

    metric = metrics.Sum("X", name="X-Total")
    output = core.Analyze(data).calculate(metric).run()

    correct = pd.DataFrame(np.array([[15]]), columns=["X-Total"])

    pd.util.testing.assert_frame_equal(output, correct)

  def testWhere(self):
    data = pd.DataFrame({"X": (1, 2, 3, 10, 20, 30, 100, 200, 300),
                         "Y": (0, 1, 2, 3, 4, 5, 6, 7, 8)})

    metric = metrics.Sum("X")
    output = core.Analyze(data).where("Y >= 4").calculate(metric).run()

    correct = pd.DataFrame(np.array([[650]]), columns=["sum(X)"])

    pd.util.testing.assert_frame_equal(output, correct)

  def testMultipleWhere(self):
    data = pd.DataFrame({"X": (1, 2, 3, 10, 20, 30, 100, 200, 300),
                         "Y": (0, 1, 2, 3, 4, 5, 6, 7, 8)})

    metric = metrics.Sum("X")
    output = core.Analyze(data).where("Y >= 4").where("Y <= 6").calculate(
        metric).run()

    correct = pd.DataFrame(np.array([[150]]), columns=["sum(X)"])

    pd.util.testing.assert_frame_equal(output, correct)

  def testMultipleCalculations(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5]})

    output = core.Analyze(data).calculate(
        [metrics.Sum("X"), metrics.Mean("X")]).run()

    correct = pd.DataFrame(np.array([[15, 3.0]]), columns=["sum(X)", "mean(X)"])
    correct[["sum(X)"]] = correct[["sum(X)"]].astype(int)

    pd.util.testing.assert_frame_equal(output, correct)

  def testMultipleCalculationsTuple(self):
    data = pd.DataFrame({"X": (1, 2, 3, 4, 5)})

    output = core.Analyze(data).calculate(
        (metrics.Sum("X"), metrics.Mean("X"))).run()

    correct = pd.DataFrame(np.array([[15, 3.0]]), columns=["sum(X)", "mean(X)"])
    correct[["sum(X)"]] = correct[["sum(X)"]].astype(int)

    pd.util.testing.assert_frame_equal(output, correct)

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

    pd.util.testing.assert_frame_equal(output, correct)

  def testSplitBy(self):
    data = pd.DataFrame({"X": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                         "Y": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]})

    metric = metrics.Sum("X")
    output = core.Analyze(data).calculate(metric).split_by("Y").run()

    correct = pd.DataFrame({"sum(X)": [1, 5, 9, 13, 17],
                            "Y": [1, 2, 3, 4, 5]})
    correct = correct.set_index("Y")

    pd.util.testing.assert_frame_equal(output, correct)

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

    pd.util.testing.assert_frame_equal(output, correct)

  def testSortTrue(self):
    data = pd.DataFrame({"X": [6, 5, 4, 7, 0, 1, 2, 3],
                         "Y": [1, 1, 1, 1, 0, 0, 0, 0],
                         "Z": [1, 0, 0, 1, 0, 0, 1, 1]})

    metric = metrics.Sum("X")
    output = (core.Analyze(data).split_by(["Y", "Z"], sort=True).
              calculate(metric).run())

    correct = pd.DataFrame({"sum(X)": [1, 5, 9, 13],
                            "Y": [0, 0, 1, 1],
                            "Z": [0, 1, 0, 1]})
    correct = correct.set_index(["Y", "Z"])

    pd.util.testing.assert_frame_equal(output, correct)

  def testSortFalse(self):
    data = pd.DataFrame({"X": [6, 5, 4, 7, 0, 1, 2, 3],
                         "Y": [1, 1, 1, 1, 0, 0, 0, 0],
                         "Z": [1, 0, 0, 1, 0, 0, 1, 1]})

    metric = metrics.Sum("X")
    output = (core.Analyze(data).split_by(["Y", "Z"], sort=False).
              calculate(metric).run())

    correct = pd.DataFrame({"sum(X)": [13, 9, 1, 5],
                            "Y": [1, 1, 0, 0],
                            "Z": [1, 0, 0, 1]})
    correct = correct.set_index(["Y", "Z"])

    pd.util.testing.assert_frame_equal(output, correct)

  def testJackknife(self):
    data = pd.DataFrame({"X": list(range(11))})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife()
    output = core.Analyze(data).with_standard_errors(se_method).calculate(
        metric).run()

    correct = pd.DataFrame(
        [[55, 10.0]], columns=("sum(X)", "sum(X) Jackknife SE"))

    pd.util.testing.assert_frame_equal(output, correct)

  def testNinetyFiveCIs(self):
    data = pd.DataFrame({"X": list(range(11))})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife(confidence=0.95)
    output = core.Analyze(data).with_standard_errors(se_method).calculate(
        metric).run()

    multiplier = scipy.stats.t.ppf(0.975, 10)
    correct_sd = 10.0

    correct_mean = 55
    correct_lower = correct_mean - multiplier * correct_sd
    correct_upper = correct_mean + multiplier * correct_sd

    correct = pd.DataFrame(
        [[correct_mean, correct_lower, correct_upper]],
        columns=("sum(X)", "sum(X) Jackknife CI-lower",
                 "sum(X) Jackknife CI-upper"))

    pd.util.testing.assert_frame_equal(output, correct)

  def testNinetyFiveCIsWithComparison(self):
    data = pd.DataFrame({
        "X": list(range(11)),
        "Y": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "U": list(range(11))
    })

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 0)
    se_method = standard_errors.Jackknife(unit="U", confidence=0.95)
    output = core.Analyze(data).with_standard_errors(se_method).relative_to(
        comparison).calculate(metric).run()

    correct_mean = 25
    correct_lower = np.nan
    correct_upper = np.nan

    rowindex = pd.Index([1], name="Y")
    correct = pd.DataFrame(
        {"sum(X) Absolute Difference": correct_mean,
         "sum(X) Absolute Difference Jackknife CI-lower": correct_lower,
         "sum(X) Absolute Difference Jackknife CI-upper": correct_upper},
        index=rowindex)

    pd.util.testing.assert_frame_equal(output, correct,
                                       check_exact=False)

  def testFiftyCIs(self):
    data = pd.DataFrame({"X": list(range(11))})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife(confidence=0.50)
    output = core.Analyze(data).with_standard_errors(se_method).calculate(
        metric).run()

    multiplier = scipy.stats.t.ppf(0.75, 10)
    correct_sd = 10.0

    correct_mean = 55
    correct_lower = correct_mean - multiplier * correct_sd
    correct_upper = correct_mean + multiplier * correct_sd

    correct = pd.DataFrame(
        [[correct_mean, correct_lower, correct_upper]],
        columns=("sum(X)", "sum(X) Jackknife CI-lower",
                 "sum(X) Jackknife CI-upper"))

    pd.util.testing.assert_frame_equal(output, correct)

  def testJackknifeRatio(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4], "Y": [4, 3, 2, 1]})

    metric = metrics.Ratio("X", "Y")
    se_method = standard_errors.Jackknife()
    output = core.Analyze(data).with_standard_errors(se_method).calculate(
        metric).run()

    estimates = np.array([9 / 6, 8 / 7, 7 / 8, 6 / 9])
    rss = ((estimates - estimates.mean())**2).sum()
    se = np.sqrt(rss * 3 / 4)

    correct = pd.DataFrame([[1.0, se]], columns=("X / Y", "X / Y Jackknife SE"))

    pd.util.testing.assert_frame_equal(output, correct)

  def testBootstrap(self):
    # The bootstrap depends upon random values to work; thus, we'll
    # only check that it's statistically close to the theoretical
    # value.

    # We set the seed to avoid flaky tests; this test will fail with
    # probability 0.05 otherwise.
    np.random.seed(12345)

    data = pd.DataFrame({"X": list(range(1, 101))})

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

  def testJackknifeNotFlatIndex(self):
    data = pd.DataFrame({"X": list(range(11))})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife(flat_index=False)
    output = core.Analyze(data).with_standard_errors(se_method).calculate(
        metric).run()

    correct = pd.DataFrame([[55, 10.0]], columns=("Value", "Jackknife SE"))
    correct.columns = pd.MultiIndex.from_product([["sum(X)"], correct.columns])

    pd.util.testing.assert_frame_equal(output, correct)

  def testNinetyFiveCIsNotFlatIndex(self):
    data = pd.DataFrame({"X": list(range(11))})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife(confidence=0.95, flat_index=False)
    output = core.Analyze(data).with_standard_errors(se_method).calculate(
        metric).run()

    multiplier = scipy.stats.t.ppf(0.975, 10)
    correct_sd = 10.0

    correct_mean = 55
    correct_lower = correct_mean - multiplier * correct_sd
    correct_upper = correct_mean + multiplier * correct_sd

    correct = pd.DataFrame([[correct_mean, correct_lower, correct_upper]],
                           columns=("Value", "Jackknife CI-lower",
                                    "Jackknife CI-upper"))
    correct.columns = pd.MultiIndex.from_product([["sum(X)"], correct.columns])

    pd.util.testing.assert_frame_equal(output, correct)

  def testNinetyFiveCIsWithComparisonNotFlatIndex(self):
    data = pd.DataFrame({
        "X": list(range(11)),
        "Y": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "U": list(range(11))
    })

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 0)
    se_method = standard_errors.Jackknife(
        unit="U", confidence=0.95, flat_index=False)
    output = core.Analyze(data).with_standard_errors(se_method).relative_to(
        comparison).calculate(metric).run()

    correct_mean = 25
    correct_lower = np.nan
    correct_upper = np.nan

    rowindex = pd.Index([1], name="Y")
    correct = pd.DataFrame(
        {
            "Absolute Difference": correct_mean,
            "Absolute Difference Jackknife CI-lower": correct_lower,
            "Absolute Difference Jackknife CI-upper": correct_upper
        },
        index=rowindex)
    correct.columns = pd.MultiIndex.from_product([["sum(X)"], correct.columns])

    pd.util.testing.assert_frame_equal(output, correct,
                                       check_exact=False)

  def testFiftyCIsNotFlatIndex(self):
    data = pd.DataFrame({"X": list(range(11))})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife(confidence=0.50, flat_index=False)
    output = core.Analyze(data).with_standard_errors(se_method).calculate(
        metric).run()

    multiplier = scipy.stats.t.ppf(0.75, 10)
    correct_sd = 10.0

    correct_mean = 55
    correct_lower = correct_mean - multiplier * correct_sd
    correct_upper = correct_mean + multiplier * correct_sd

    correct = pd.DataFrame([[correct_mean, correct_lower, correct_upper]],
                           columns=("Value", "Jackknife CI-lower",
                                    "Jackknife CI-upper"))
    correct.columns = pd.MultiIndex.from_product([["sum(X)"], correct.columns])

    pd.util.testing.assert_frame_equal(output, correct)

  def testJackknifeRatioNotFlatIndex(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4], "Y": [4, 3, 2, 1]})

    metric = metrics.Ratio("X", "Y")
    se_method = standard_errors.Jackknife(flat_index=False)
    output = core.Analyze(data).with_standard_errors(se_method).calculate(
        metric).run()

    estimates = np.array([9 / 6, 8 / 7, 7 / 8, 6 / 9])
    rss = ((estimates - estimates.mean())**2).sum()
    se = np.sqrt(rss * 3 / 4)

    correct = pd.DataFrame([[1.0, se]], columns=("Value", "Jackknife SE"))
    correct.columns = pd.MultiIndex.from_product([["X / Y"], correct.columns])

    pd.util.testing.assert_frame_equal(output, correct)

  def testBootstrapNotFlatIndex(self):
    # The bootstrap depends upon random values to work; thus, we'll
    # only check that it's statistically close to the theoretical
    # value.

    # We set the seed to avoid flaky tests; this test will fail with
    # probability 0.05 otherwise.
    np.random.seed(12345)

    data = pd.DataFrame({"X": list(range(1, 101))})

    metric = metrics.Mean("X")
    se_method = standard_errors.Bootstrap(100, flat_index=False)
    output = core.Analyze(data).with_standard_errors(se_method).calculate(
        metric).run()

    bootstrap_se = output[("mean(X)", "Bootstrap SE")].values[0]

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

  def testUnitBootstrapNotFlatIndex(self):
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
    se_method = standard_errors.Bootstrap(100, unit="Y", flat_index=False)
    output = core.Analyze(data).with_standard_errors(se_method).calculate(
        metric).run()

    bootstrap_se = output[("mean(X)", "Bootstrap SE")].values[0]

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

    pd.util.testing.assert_frame_equal(output, correct)

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

    pd.util.testing.assert_frame_equal(output, correct)

  def testRelativeToJackknife(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                         "Y": [0, 0, 0, 1, 1, 1, 2, 2, 2],
                         "U": [1, 2, 3, 2, 3, 1, 3, 1, 2]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 0)
    se_method = standard_errors.Jackknife(unit="U")
    output = core.Analyze(data).relative_to(comparison).with_standard_errors(
        se_method).calculate(metric).run()

    rowindex = pd.Index([1, 2], name="Y")
    correct = pd.DataFrame({
        "sum(X) Absolute Difference": [9, 18],
        "sum(X) Absolute Difference Jackknife SE": [
            np.sqrt(2 * np.var([4, 7, 7])),
            np.sqrt(2 * np.var([11, 11, 14]))
        ]
    }, index=rowindex)

    pd.util.testing.assert_frame_equal(output, correct)

  def testRelativeToJackknifeIncludeBaseline(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                         "Y": [0, 0, 0, 1, 1, 1, 2, 2, 2],
                         "U": [1, 2, 3, 2, 3, 1, 3, 1, 2]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 0, include_base=True)
    se_method = standard_errors.Jackknife(unit="U")
    output = core.Analyze(data).relative_to(comparison).with_standard_errors(
        se_method).calculate(metric).run()

    rowindex = pd.Index([0, 1, 2], name="Y")
    correct = pd.DataFrame({
        "sum(X) Absolute Difference": [0, 9, 18],
        "sum(X) Absolute Difference Jackknife SE": [
            0.0,
            np.sqrt(2 * np.var([4, 7, 7])),
            np.sqrt(2 * np.var([11, 11, 14]))
        ]
    }, index=rowindex)

    pd.util.testing.assert_frame_equal(output, correct)

  def testRelativeToJackknifeSingleComparisonBaselineFirst(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5, 6],
                         "Y": [0, 0, 0, 1, 1, 1],
                         "U": [1, 2, 3, 2, 3, 1]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 0)
    se_method = standard_errors.Jackknife(unit="U")
    output = core.Analyze(data).relative_to(comparison).with_standard_errors(
        se_method).calculate(metric).run()

    rowindex = pd.Index([1], name="Y")
    correct = pd.DataFrame({
        "sum(X) Absolute Difference": [9],
        "sum(X) Absolute Difference Jackknife SE": [
            np.sqrt(2 * np.var([4, 7, 7]))
        ]
    }, index=rowindex)

    pd.util.testing.assert_frame_equal(output, correct)

  def testRelativeToJackknifeSingleComparisonBaselineSecond(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5, 6],
                         "Y": [0, 0, 0, 1, 1, 1],
                         "U": [1, 2, 3, 2, 3, 1]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 1)
    se_method = standard_errors.Jackknife(unit="U")
    output = core.Analyze(data).relative_to(comparison).with_standard_errors(
        se_method).calculate(metric).run()

    rowindex = pd.Index([0], name="Y")
    correct = pd.DataFrame({
        "sum(X) Absolute Difference": [-9],
        "sum(X) Absolute Difference Jackknife SE": [
            np.sqrt(2 * np.var([4, 7, 7]))
        ]
    }, index=rowindex)

    pd.util.testing.assert_frame_equal(output, correct)

  def testSplitJackknife(self):
    data = pd.DataFrame({
        "X": np.array([list(range(11)) + [5] * 10]).flatten(),
        "Y": np.array([[0] * 11 + [1] * 10]).flatten()
    })

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife()
    output = core.Analyze(data).split_by("Y").with_standard_errors(
        se_method).calculate(metric).run()

    rowindex = pd.Index([0, 1], name="Y")
    correct = pd.DataFrame(
        [[55, 10.0], [50, 0.0]],
        columns=("sum(X)", "sum(X) Jackknife SE"),
        index=rowindex)

    pd.util.testing.assert_frame_equal(output, correct)

  def testRelativeToSplitJackknife(self):
    data = pd.DataFrame({
        "X": [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        "Y": [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        "Z": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "U": list(range(18))
    })

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Z", 0)
    se_method = standard_errors.Jackknife(unit="U")
    output = core.Analyze(data).split_by("Y").relative_to(
        comparison).with_standard_errors(se_method).calculate(metric).run()

    rowindex = pd.MultiIndex(
        levels=[[1, 2, 3], [1]],
        labels=[[0, 1, 2], [0, 0, 0]],
        names=["Y", "Z"])
    correct = pd.DataFrame({
        "sum(X) Absolute Difference": [-3, -3, -3],
        "sum(X) Absolute Difference Jackknife SE": [
            np.nan, np.nan, np.nan
        ]}, index=rowindex)

    pd.util.testing.assert_frame_equal(output, correct)

  def testSingleJackknifeBucket(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5], "Y": [1, 1, 1, 1, 1]})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife(unit="Y")
    correct = pd.DataFrame(
        [[15, np.nan]],
        columns=("sum(X)", "sum(X) Jackknife SE"))
    output = (core.Analyze(data).with_standard_errors(se_method).
              calculate(metric).run())
    pd.util.testing.assert_frame_equal(output, correct)

  def testNoJackknifeBuckets(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5, 6],
                         "Y": [1, 2, 1, 2, 1, 2],
                         "Z": [1, 1, 2, 2, 3, 3],
                         "W": [0, 0, 0, 0, 1, 1]})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife(unit="Y")
    comparison = comparisons.AbsoluteDifference("Z", 1)
    correct = pd.DataFrame(
        [[4.0, 0.0],
         [np.nan, np.nan],
         [np.nan, np.nan],
         [np.nan, np.nan]],
        columns=("sum(X) Absolute Difference",
                 "sum(X) Absolute Difference Jackknife SE"),
        index=pd.MultiIndex(levels=[[0, 1], [2, 3]],
                            labels=[[0, 0, 1, 1], [0, 1, 0, 1]],
                            names=["W", "Z"]))
    output = (core.Analyze(data).split_by("W").with_standard_errors(se_method).
              relative_to(comparison).calculate(metric).run())
    pd.util.testing.assert_frame_equal(output, correct)

  def testJackknifeBadSample(self):
    data = pd.DataFrame({"X": list(range(22)), "Y": ([0] * 11) + ([1] * 11)})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife()
    output = core.Analyze(data).split_by("Y").with_standard_errors(
        se_method).calculate(metric).run()

    correct = pd.DataFrame(
        [[55, 10.0], [176, 10.0]],
        columns=("sum(X)", "sum(X) Jackknife SE"))

    correct.index.name = "Y"

    pd.util.testing.assert_frame_equal(output, correct)

  def testJackknifeOutOfRangeBins(self):
    data = pd.DataFrame({
        "X": list(range(11)) + list(range(11)),
        "Y": list(range(22)),
        "Z": ([0] * 11 + [1] * 11)
    })

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife("Y")
    output = core.Analyze(data).split_by("Z").with_standard_errors(
        se_method).calculate(metric).run()

    correct = pd.DataFrame(
        [[55, 10.0], [55, 10.0]],
        columns=("sum(X)", "sum(X) Jackknife SE"))

    correct.index.name = "Z"

    pd.util.testing.assert_frame_equal(output, correct)

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

    pd.util.testing.assert_frame_equal(output, correct)

  def testDistributionJackknife(self):
    df = pd.DataFrame({
        "X": list(range(11)),
        "Y": np.concatenate((np.zeros(6), np.ones(5))),
        "Z": np.concatenate((np.zeros(3), np.ones(8)))
    })

    metric = metrics.Distribution("X", ["Z"])
    se_method = standard_errors.Jackknife("Y")
    output = core.Analyze(df).with_standard_errors(se_method).calculate(
        metric).run()

    correct = pd.DataFrame(
        np.array([[3 / 55., np.sqrt(((3 / 15. - 0.1)**2 + 0.1**2) / 2.)],
                  [52 / 55., np.sqrt(((12 / 15. - 0.9)**2 + 0.1**2) / 2.)]]),
        columns=("X Distribution", "X Distribution Jackknife SE"),
        index=pd.Index([0., 1.], name="Z"))

    pd.util.testing.assert_frame_equal(output, correct,
                                       check_exact=False)

  def testDistributionRelativeToJackknife(self):
    df = pd.DataFrame({
        "X": list(range(11)),
        "Y": np.concatenate((np.zeros(6), np.ones(5))),
        "Z": np.concatenate((np.zeros(3), np.ones(8))),
        "U": list(range(11))
    })

    metric = metrics.Distribution("X", ["Z"])
    se_method = standard_errors.Jackknife(unit="U")
    output = core.Analyze(df).relative_to(comparisons.AbsoluteDifference(
        "Y", 0)).with_standard_errors(se_method).calculate(metric).run()

    correct = pd.DataFrame(
        np.array([[-0.2, np.nan],
                  [0.2, np.nan]]),
        columns=["X Distribution Absolute Difference",
                 "X Distribution Absolute Difference Jackknife SE"],
        index=pd.MultiIndex(levels=[[1.], [0., 1.]],
                            labels=[[0, 0], [0, 1]],
                            names=["Y", "Z"]))

    pd.util.testing.assert_frame_equal(output, correct,
                                       check_exact=False)

  def testShuffledDistributionRelativeToJackknife(self):
    # Same as test above, but also testing that reordering the data doesn't
    # change results, up to order.
    df = pd.DataFrame({
        "X": list(range(11)),
        "Y": np.concatenate((np.zeros(6), np.ones(5))),
        "Z": np.concatenate((np.zeros(3), np.ones(8))),
        "U": list(range(11))
    })

    metric = metrics.Distribution("X", ["Z"])
    se_method = standard_errors.Jackknife(unit="U")
    output = core.Analyze(df.iloc[np.random.permutation(11)]).relative_to(
        comparisons.AbsoluteDifference("Y", 0)).with_standard_errors(
            se_method).calculate(metric).run()
    output = output.sort_index(level=["Y", "Z"])

    correct = pd.DataFrame(
        np.array([[-0.2, np.nan],
                  [0.2, np.nan]]),
        columns=["X Distribution Absolute Difference",
                 "X Distribution Absolute Difference Jackknife SE"],
        index=pd.MultiIndex(levels=[[1.], [0., 1.]],
                            labels=[[0, 0], [0, 1]],
                            names=["Y", "Z"]))
    correct = (correct.
               reset_index().
               sort_values(by=["Y", "Z"]).
               set_index(["Y", "Z"]))

    pd.util.testing.assert_frame_equal(output, correct,
                                       check_exact=False)

  def testRelativeToJackknifeNotFlatIndex(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                         "Y": [0, 0, 0, 1, 1, 1, 2, 2, 2],
                         "U": [1, 2, 3, 2, 3, 1, 3, 1, 2]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 0)
    se_method = standard_errors.Jackknife(unit="U", flat_index=False)
    output = core.Analyze(data).relative_to(comparison).with_standard_errors(
        se_method).calculate(metric).run()

    rowindex = pd.Index([1, 2], name="Y")
    correct = pd.DataFrame(
        {
            "Absolute Difference": [9, 18],
            "Absolute Difference Jackknife SE": [
                np.sqrt(2 * np.var([4, 7, 7])),
                np.sqrt(2 * np.var([11, 11, 14]))
            ]
        },
        index=rowindex)
    correct.columns = pd.MultiIndex.from_product([["sum(X)"], correct.columns])

    pd.util.testing.assert_frame_equal(output, correct)

  def testRelativeToJackknifeIncludeBaselineNotFlatIndex(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                         "Y": [0, 0, 0, 1, 1, 1, 2, 2, 2],
                         "U": [1, 2, 3, 2, 3, 1, 3, 1, 2]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 0, include_base=True)
    se_method = standard_errors.Jackknife(unit="U", flat_index=False)
    output = core.Analyze(data).relative_to(comparison).with_standard_errors(
        se_method).calculate(metric).run()

    rowindex = pd.Index([0, 1, 2], name="Y")
    correct = pd.DataFrame(
        {
            "Absolute Difference": [0, 9, 18],
            "Absolute Difference Jackknife SE": [
                0.0,
                np.sqrt(2 * np.var([4, 7, 7])),
                np.sqrt(2 * np.var([11, 11, 14]))
            ]
        },
        index=rowindex)
    correct.columns = pd.MultiIndex.from_product([["sum(X)"], correct.columns])

    pd.util.testing.assert_frame_equal(output, correct)

  def testRelativeToJackknifeSingleComparisonBaselineFirstNotFlatIndex(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5, 6],
                         "Y": [0, 0, 0, 1, 1, 1],
                         "U": [1, 2, 3, 2, 3, 1]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 0)
    se_method = standard_errors.Jackknife(unit="U", flat_index=False)
    output = core.Analyze(data).relative_to(comparison).with_standard_errors(
        se_method).calculate(metric).run()

    rowindex = pd.Index([1], name="Y")
    correct = pd.DataFrame(
        {
            "Absolute Difference": [9],
            "Absolute Difference Jackknife SE":
                [np.sqrt(2 * np.var([4, 7, 7]))]
        },
        index=rowindex)
    correct.columns = pd.MultiIndex.from_product([["sum(X)"], correct.columns])

    pd.util.testing.assert_frame_equal(output, correct)

  def testRelativeToJackknifeSingleComparisonBaselineSecondNotFlatIndex(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5, 6],
                         "Y": [0, 0, 0, 1, 1, 1],
                         "U": [1, 2, 3, 2, 3, 1]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 1)
    se_method = standard_errors.Jackknife(unit="U", flat_index=False)
    output = core.Analyze(data).relative_to(comparison).with_standard_errors(
        se_method).calculate(metric).run()

    rowindex = pd.Index([0], name="Y")
    correct = pd.DataFrame(
        {
            "Absolute Difference": [-9],
            "Absolute Difference Jackknife SE":
                [np.sqrt(2 * np.var([4, 7, 7]))]
        },
        index=rowindex)
    correct.columns = pd.MultiIndex.from_product([["sum(X)"], correct.columns])

    pd.util.testing.assert_frame_equal(output, correct)

  def testSplitJackknifeNotFlatIndex(self):
    data = pd.DataFrame({
        "X": np.array([list(range(11)) + [5] * 10]).flatten(),
        "Y": np.array([[0] * 11 + [1] * 10]).flatten()
    })

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife(flat_index=False)
    output = core.Analyze(data).split_by("Y").with_standard_errors(
        se_method).calculate(metric).run()

    rowindex = pd.Index([0, 1], name="Y")
    correct = pd.DataFrame([[55, 10.0], [50, 0.0]],
                           columns=("Value", "Jackknife SE"),
                           index=rowindex)
    correct.columns = pd.MultiIndex.from_product([["sum(X)"], correct.columns])

    pd.util.testing.assert_frame_equal(output, correct)

  def testRelativeToSplitJackknifeNotFlatIndex(self):
    data = pd.DataFrame({
        "X": [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        "Y": [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        "Z": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "U": list(range(18))
    })

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Z", 0)
    se_method = standard_errors.Jackknife(unit="U", flat_index=False)
    output = core.Analyze(data).split_by("Y").relative_to(
        comparison).with_standard_errors(se_method).calculate(metric).run()

    rowindex = pd.MultiIndex(
        levels=[[1, 2, 3], [1]],
        labels=[[0, 1, 2], [0, 0, 0]],
        names=["Y", "Z"])
    correct = pd.DataFrame(
        {
            "Absolute Difference": [-3, -3, -3],
            "Absolute Difference Jackknife SE": [np.nan, np.nan, np.nan]
        },
        index=rowindex)
    correct.columns = pd.MultiIndex.from_product([["sum(X)"], correct.columns])

    pd.util.testing.assert_frame_equal(output, correct)

  def testSingleJackknifeBucketNotFlatIndex(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5], "Y": [1, 1, 1, 1, 1]})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife(unit="Y", flat_index=False)
    correct = pd.DataFrame([[15, np.nan]], columns=("Value", "Jackknife SE"))
    correct.columns = pd.MultiIndex.from_product([["sum(X)"], correct.columns])
    output = (core.Analyze(data).with_standard_errors(se_method).
              calculate(metric).run())
    pd.util.testing.assert_frame_equal(output, correct)

  def testNoJackknifeBucketsNotFlatIndex(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5, 6],
                         "Y": [1, 2, 1, 2, 1, 2],
                         "Z": [1, 1, 2, 2, 3, 3],
                         "W": [0, 0, 0, 0, 1, 1]})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife(unit="Y", flat_index=False)
    comparison = comparisons.AbsoluteDifference("Z", 1)
    correct = pd.DataFrame(
        [[4.0, 0.0], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
        columns=("Absolute Difference", "Absolute Difference Jackknife SE"),
        index=pd.MultiIndex(
            levels=[[0, 1], [2, 3]],
            labels=[[0, 0, 1, 1], [0, 1, 0, 1]],
            names=["W", "Z"]))
    correct.columns = pd.MultiIndex.from_product([["sum(X)"], correct.columns])
    output = (core.Analyze(data).split_by("W").with_standard_errors(se_method).
              relative_to(comparison).calculate(metric).run())
    pd.util.testing.assert_frame_equal(output, correct)

  def testJackknifeBadSampleNotFlatIndex(self):
    data = pd.DataFrame({"X": list(range(22)), "Y": ([0] * 11) + ([1] * 11)})

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife(flat_index=False)
    output = core.Analyze(data).split_by("Y").with_standard_errors(
        se_method).calculate(metric).run()

    correct = pd.DataFrame([[55, 10.0], [176, 10.0]],
                           columns=("Value", "Jackknife SE"))
    correct.columns = pd.MultiIndex.from_product([["sum(X)"], correct.columns])

    correct.index.name = "Y"

    pd.util.testing.assert_frame_equal(output, correct)

  def testJackknifeOutOfRangeBinsNotFlatIndex(self):
    data = pd.DataFrame({
        "X": list(range(11)) + list(range(11)),
        "Y": list(range(22)),
        "Z": ([0] * 11 + [1] * 11)
    })

    metric = metrics.Sum("X")
    se_method = standard_errors.Jackknife("Y", flat_index=False)
    output = core.Analyze(data).split_by("Z").with_standard_errors(
        se_method).calculate(metric).run()

    correct = pd.DataFrame([[55, 10.0], [55, 10.0]],
                           columns=("Value", "Jackknife SE"))
    correct.columns = pd.MultiIndex.from_product([["sum(X)"], correct.columns])
    correct.index.name = "Z"

    pd.util.testing.assert_frame_equal(output, correct)

  def testRelativeToSplitsWithNoAlternativeGivesNaNNotFlatIndex(self):
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

    pd.util.testing.assert_frame_equal(output, correct)

  def testDistributionJackknifeNotFlatIndex(self):
    df = pd.DataFrame({
        "X": list(range(11)),
        "Y": np.concatenate((np.zeros(6), np.ones(5))),
        "Z": np.concatenate((np.zeros(3), np.ones(8)))
    })

    metric = metrics.Distribution("X", ["Z"])
    se_method = standard_errors.Jackknife("Y", flat_index=False)
    output = core.Analyze(df).with_standard_errors(se_method).calculate(
        metric).run()

    correct = pd.DataFrame(
        np.array([[3 / 55.,
                   np.sqrt(((3 / 15. - 0.1)**2 + 0.1**2) / 2.)],
                  [52 / 55.,
                   np.sqrt(((12 / 15. - 0.9)**2 + 0.1**2) / 2.)]]),
        columns=("Value", "Jackknife SE"),
        index=pd.Index([0., 1.], name="Z"))
    correct.columns = pd.MultiIndex.from_product([["X Distribution"],
                                                  correct.columns])

    pd.util.testing.assert_frame_equal(output, correct,
                                       check_exact=False)

  def testDistributionRelativeTo(self):
    df = pd.DataFrame({
        "X": list(range(11)),
        "Y": np.concatenate((np.zeros(6), np.ones(5))),
        "Z": np.concatenate((np.zeros(3), np.ones(8)))
    })

    metric = metrics.Distribution("X", ["Z"])
    output = core.Analyze(df).relative_to(comparisons.AbsoluteDifference(
        "Y", 0)).calculate(metric).run()

    correct = pd.DataFrame(
        np.array([-0.2, 0.2]),
        columns=["X Distribution Absolute Difference"],
        index=pd.MultiIndex(levels=[[1.], [0., 1.]],
                            labels=[[0, 0], [0, 1]],
                            names=["Y", "Z"]))

    pd.util.testing.assert_frame_equal(output, correct,
                                       check_exact=False)

  def testDistributionRelativeToJackknifeNotFlatIndex(self):
    df = pd.DataFrame({
        "X": list(range(11)),
        "Y": np.concatenate((np.zeros(6), np.ones(5))),
        "Z": np.concatenate((np.zeros(3), np.ones(8))),
        "U": list(range(11))
    })

    metric = metrics.Distribution("X", ["Z"])
    se_method = standard_errors.Jackknife(unit="U", flat_index=False)
    output = core.Analyze(df).relative_to(comparisons.AbsoluteDifference(
        "Y", 0)).with_standard_errors(se_method).calculate(metric).run()

    correct = pd.DataFrame(
        np.array([[-0.2, np.nan], [0.2, np.nan]]),
        columns=["Absolute Difference", "Absolute Difference Jackknife SE"],
        index=pd.MultiIndex(
            levels=[[1.], [0., 1.]], labels=[[0, 0], [0, 1]], names=["Y", "Z"]))
    correct.columns = pd.MultiIndex.from_product([["X Distribution"],
                                                  correct.columns])

    pd.util.testing.assert_frame_equal(output, correct,
                                       check_exact=False)

  def testShuffledDistributionRelativeToJackknifeNotFlatIndex(self):
    # Same as test above, but also testing that reordering the data doesn't
    # change results, up to order.
    df = pd.DataFrame({
        "X": list(range(11)),
        "Y": np.concatenate((np.zeros(6), np.ones(5))),
        "Z": np.concatenate((np.zeros(3), np.ones(8))),
        "U": list(range(11))
    })

    metric = metrics.Distribution("X", ["Z"])
    se_method = standard_errors.Jackknife(unit="U", flat_index=False)
    output = core.Analyze(df.iloc[np.random.permutation(11)]).relative_to(
        comparisons.AbsoluteDifference("Y", 0)).with_standard_errors(
            se_method).calculate(metric).run()
    output = output.sort_index(level=["Y", "Z"])

    correct = pd.DataFrame(
        np.array([[-0.2, np.nan], [0.2, np.nan]]),
        columns=["Absolute Difference", "Absolute Difference Jackknife SE"],
        index=pd.MultiIndex(
            levels=[[1.], [0., 1.]], labels=[[0, 0], [0, 1]], names=["Y", "Z"]))
    correct = (correct.
               reset_index().
               sort_values(by=["Y", "Z"]).
               set_index(["Y", "Z"]))
    correct.columns = pd.MultiIndex.from_product([["X Distribution"],
                                                  correct.columns])

    pd.util.testing.assert_frame_equal(output, correct,
                                       check_exact=False)

  def testDistributionsOverDifferentColumns(self):
    df = pd.DataFrame({
        "X": list(range(11)),
        "Y": np.concatenate((np.zeros(6), np.ones(5))),
        "Z": np.concatenate((np.zeros(3), np.ones(8)))
    })
    metric = [
        metrics.Distribution("X", ["Z"]),
        metrics.Distribution("Z", ["Y"])
    ]
    output = core.Analyze(df).calculate(metric).run()
    correct0 = core.Analyze(df).calculate(metric[0]).run()
    correct1 = core.Analyze(df).calculate(metric[1]).run()
    correct0.index = pd.MultiIndex.from_product([correct0.index, [""]])
    correct0.index.names = ["Z", "Y"]
    correct1.index = pd.MultiIndex.from_product([[""], correct1.index])
    correct1.index.names = ["Z", "Y"]
    correct = pd.concat([correct0, correct1])
    pd.util.testing.assert_frame_equal(output, correct)

  def testSplitDistribution(self):
    df = pd.DataFrame({
        "X": list(range(11)),
        "Y": np.concatenate((np.zeros(6), np.ones(5))),
        "Z": np.concatenate((np.zeros(3), np.ones(8)))
    })

    metric = metrics.Distribution("X", ["Z"])
    output = core.Analyze(df).split_by(["Y"]).calculate(metric).run()

    correct = pd.DataFrame(
        np.array([0.2, 0.8, 0.0, 1.0]),
        columns=["X Distribution"],
        index=pd.MultiIndex(levels=[[0.0, 1.0], [0.0, 1.0]],
                            labels=[[0, 0, 1, 1], [0, 1, 0, 1]],
                            names=["Y", "Z"]))

    pd.util.testing.assert_frame_equal(output, correct,
                                       check_exact=False)

  def testSplitDistributionInCustomFunction(self):
    # The output is screenshot/3KGVX9WQLep
    df = pd.DataFrame({
        "X": list(range(11)),
        "Y": np.concatenate((np.zeros(6), np.ones(5))),
        "Z": np.concatenate((np.zeros(3), np.ones(5), np.zeros(3)))
    })

    metric = metrics.Distribution("X", ["Z"], normalize=False)
    correct = core.Analyze(df).split_by(["Y"]).calculate(metric).run()
    correct.columns = pd.MultiIndex.from_product([correct.columns, ["X"]])
    correct.columns.names = [None, "MetricIndex0"]
    fn = lambda d: d.groupby("Z").sum(min_count=1)
    custom_dist = metrics.Metric("X Distribution", fn=fn)
    output = core.Analyze(df).split_by(["Y"]).calculate(custom_dist).run()
    pd.util.testing.assert_frame_equal(output, correct)

  def testDistributionWithNaNs(self):
    df = pd.DataFrame({"X": [1, 2, 1, 2, 1, 3, 1, 2, 2, 1, 1],
                       "Y": [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
                       "Z": [0, 1, 2, 4, 5, 6, 1, 3, 5, 1, 4],
                       "W": [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]})
    metric = metrics.Distribution("X", "Z")
    comparison = comparisons.AbsoluteDifference("Y", 0)

    output = core.Analyze(df).split_by("W").relative_to(
        comparison).calculate(metric).run()

    correct = pd.DataFrame(
        [-0.1, 0.0, -0.1, 0.4, -0.2, 0.3, -0.3,
         0., -1., 0., 0., 1., 0., 0.],
        columns=["X Distribution Absolute Difference"],
        index=pd.MultiIndex(
            levels=[[1, 2], [1], [0, 1, 2, 3, 4, 5, 6]],
            labels=[[0, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 2, 3, 4, 5, 6,
                     0, 1, 2, 3, 4, 5, 6]],
            names=["W", "Y", "Z"]))

    pd.util.testing.assert_frame_equal(output, correct,
                                       check_exact=False)

  def testCumulativeDistributionWithNaNs(self):
    df = pd.DataFrame({"X": [1, 2, 1, 2, 1, 3, 1, 2, 2, 1, 1],
                       "Y": [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
                       "Z": [0, 1, 2, 4, 5, 6, 1, 3, 5, 1, 4],
                       "W": [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]})
    metric = metrics.CumulativeDistribution("X", "Z")
    comparison = comparisons.AbsoluteDifference("Y", 0)

    output = core.Analyze(df).split_by("W").relative_to(
        comparison).calculate(metric).run()

    correct = pd.DataFrame(
        [-0.1, -0.1, -0.2, 0.2, 0., 0.3, 0.,
         0., -1., -1., -1., 0., 0., 0.],
        columns=["X Cumulative Distribution Absolute Difference"],
        index=pd.MultiIndex(
            levels=[[1, 2], [1], [0, 1, 2, 3, 4, 5, 6]],
            labels=[[0, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 2, 3, 4, 5, 6,
                     0, 1, 2, 3, 4, 5, 6]],
            names=["W", "Y", "Z"]))

    pd.util.testing.assert_frame_equal(output, correct,
                                       check_exact=False)


class CopyTest(googletest.TestCase):
  """Check that Meterstick is modifying copies of objects."""

  def testCopy(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5, 6],
                         "Y": [0, 0, 0, 1, 1, 1],
                         "U": [1, 2, 3, 2, 3, 1]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 1)
    se_method = standard_errors.Jackknife(unit="U")
    analysis = core.Analyze(data).relative_to(comparison).with_standard_errors(
        se_method).calculate(metric)

    self.assertIsNot(comparison, analysis.parameters.comparison)
    self.assertIsNot(se_method, analysis.parameters.se_method)
    self.assertIsNot(metric, analysis.parameters.metrics[0])

  def testCopyMultipleMetrics(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5, 6],
                         "Y": [0, 0, 0, 1, 1, 1],
                         "U": [1, 2, 3, 2, 3, 1]})

    metric = [metrics.Sum("X"), metrics.Mean("X")]
    comparison = comparisons.AbsoluteDifference("Y", 1)
    se_method = standard_errors.Jackknife(unit="U")
    analysis = core.Analyze(data).relative_to(comparison).with_standard_errors(
        se_method).calculate(metric)

    self.assertIsNot(comparison, analysis.parameters.comparison)
    self.assertIsNot(se_method, analysis.parameters.se_method)
    self.assertIsNot(metric, analysis.parameters.metrics)

  def testComposedMetric(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5, 6],
                         "Y": [0, 0, 0, 1, 1, 1],
                         "U": [1, 2, 3, 2, 3, 1]})

    metric = metrics.StandardDeviation("X") / metrics.Mean("X")
    comparison = comparisons.AbsoluteDifference("Y", 1)
    se_method = standard_errors.Jackknife(unit="U")
    analysis = core.Analyze(data).relative_to(comparison).with_standard_errors(
        se_method).calculate(metric)

    self.assertIsNot(comparison, analysis.parameters.comparison)
    self.assertIsNot(se_method, analysis.parameters.se_method)
    self.assertIsNot(metric, analysis.parameters.metrics[0])


class MeltTest(googletest.TestCase):
  """Check that Meterstick is returning melted data correctly."""

  def testMeltRelativeToSplit(self):
    data = pd.DataFrame(
        {"X": [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
         "Y": [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2],
         "Z": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})

    comparison = comparisons.AbsoluteDifference("Y", 0)
    analysis = (core.Analyze(data).
                split_by("Z").
                relative_to(comparison).
                calculate([metrics.Sum("X"), metrics.Mean("X")]))
    output = analysis.run(melted=True)

    correct = pd.DataFrame({
        "Absolute Difference": [
            13 - 5, 23 - 5, 14 - 4, 22 - 4,
            (13. - 5.) / 3.,
            (23. - 5.) / 3.,
            (14. - 4.) / 3.,
            (22. - 4.) / 3.
        ],
        "Metric": (["sum(X)"] * 4) + (["mean(X)"] * 4),
        "Z": [0, 0, 1, 1, 0, 0, 1, 1],
        "Y": [1, 2, 1, 2, 1, 2, 1, 2]
    })

    correct = correct.set_index(["Z", "Metric", "Y"])
    pd.util.testing.assert_frame_equal(output, correct)


class ExceptionTest(googletest.TestCase):
  """Check that Meterstick is raising exceptions appropriately."""

  def testEmptyDataFrame(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5]})

    metric = metrics.Sum("X")
    with self.assertRaisesRegex(ValueError, "empty"):
      core.Analyze(data.query("X == 0")).calculate(metric).run()

  def testBadWhereRaisesError(self):
    data = pd.DataFrame({"X": (1, 2, 3, 10, 20, 30, 100, 200, 300),
                         "Y": (0, 1, 2, 3, 4, 5, 6, 7, 8)})

    metric = metrics.Sum("X")
    with self.assertRaises(ValueError):
      core.Analyze(data).where("X + Y").calculate(metric).run()

  def testBadConfidenceRaisesException(self):
    with self.assertRaises(ValueError):
      standard_errors.Jackknife(confidence=95)

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

  def testComparisonBaselineGivesError(self):
    data = pd.DataFrame({"X": [1, 2, 3, 4, 5], "Y": [1, 1, 1, 1, 1]})

    metric = metrics.Sum("X")
    comparison = comparisons.AbsoluteDifference("Y", 0)

    with self.assertRaises(ValueError):
      core.Analyze(data).relative_to(comparison).calculate(metric).run()

  def testDuplicateMetrics(self):
    data = pd.DataFrame({"X": [0, 0, 1, 1]})

    metric = [metrics.Sum("X"), metrics.Sum("X")]
    with self.assertRaises(ValueError):
      core.Analyze(data).calculate(metric)


class CustomFunctionMetricTest(googletest.TestCase):
  """Test Metric instantiated by a custom function."""
  data = pd.DataFrame({
      "X": [1, 2, 3, 4, 5, 6, 7, 8, 9],
      "Y": [0, 0, 0, 1, 1, 1, 2, 2, 2],
      "U": [1, 2, 3, 2, 3, 1, 3, 1, 2]
  })
  n_grp = list("ABC")
  n_type = ("Desk", "Mob")
  n_cookie = list(range(5))
  x = np.arange(0, len(n_grp) * len(n_type) * len(n_cookie)) + 1
  data_complex = pd.DataFrame(
      list(
          zip(x, itertools.cycle(n_type), itertools.cycle(n_grp),
              itertools.cycle(n_cookie))),
      columns=("X", "type", "grp", "cookie"))
  data_complex["Y"] = 2 * data_complex.X

  def testFlatIndexSeriesMelted(self):
    ten_to_x = lambda df: pd.Series((10**(df.X)).values)
    metric = metrics.Metric("10^x", fn=ten_to_x)
    output = core.Analyze(self.data).calculate(metric).run(1)
    correct = pd.DataFrame({
        "Metric": ["10^x"] * len(self.data),
        "MetricIndex0": list(range(len(self.data))),
        "Value": list(10**self.data.X)
    })
    correct.set_index(["Metric", "MetricIndex0"], inplace=True)
    pd.util.testing.assert_frame_equal(correct, output)

  def testFlatIndexSeriesNotMelted(self):
    ten_to_x = lambda df: pd.Series((10**(df.X)).values)
    metric = metrics.Metric("10^x", fn=ten_to_x)
    output = core.Analyze(self.data).calculate(metric).run()
    correct = pd.DataFrame([list(10**self.data.X)],
                           columns=pd.MultiIndex.from_product(
                               [["10^x"], list(range(len(self.data)))]))
    correct.columns.names = [None, "MetricIndex0"]
    correct.index.name = ""
    pd.util.testing.assert_frame_equal(correct, output)

  def testFlatIndexSeriesMeltedWithSplitBy(self):
    ten_to_x = lambda df: pd.Series((10**(df.X)).values, index=list("abc"))
    metric = [metrics.Metric("10^x", fn=ten_to_x), metrics.Sum("X")]
    output = core.Analyze(self.data).split_by(["Y"]).calculate(metric).run(1)
    correct = pd.DataFrame({
        "Y": [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2],
        "Metric": ["10^x"] * len(self.data) + ["sum(X)"] * 3,
        "MetricIndex0": list("abc") * 3 + [""] * 3,
        "Value": list(10**self.data.X) + [6, 15, 24]
    })
    correct.set_index(["Y", "Metric", "MetricIndex0"], inplace=True)
    pd.util.testing.assert_frame_equal(correct, output)

  def testFlatIndexSeriesNotMeltedWithSplitBy(self):
    ten_to_x = lambda df: pd.Series((10**(df.X)).values, index=list("abc"))
    metric = [metrics.Metric("10^x", fn=ten_to_x), metrics.Sum("X")]
    output = core.Analyze(self.data).split_by(["Y"]).calculate(metric).run()
    correct = pd.DataFrame({
        ("10^x", "a"): [10, 10000, 10000000],
        ("10^x", "b"): [100, 100000, 100000000],
        ("10^x", "c"): [1000, 1000000, 1000000000],
        ("sum(X)", ""): [6, 15, 24],
    })
    correct.columns.names = [None, "MetricIndex0"]
    correct.index.name = "Y"
    pd.util.testing.assert_frame_equal(correct, output)

  def testMultiIndexSeriesWithFlatIndexSeriesMelted(self):
    data = pd.DataFrame({"X": [1, 2, 3]})
    multi_index = pd.MultiIndex.from_arrays([list("abc"), list("xyz")])
    ten_to_x = lambda df: pd.Series((10**(df.X)).values, index=multi_index)
    two_to_x = lambda df: pd.Series((2**(df.X)).values, index=list("def"))
    metric = [
        metrics.Metric("10^x", fn=ten_to_x),
        metrics.Sum("X"),
        metrics.Metric("2^x", fn=two_to_x)
    ]
    output = core.Analyze(data).calculate(metric).run(1)
    output = output.astype(int)
    correct = pd.DataFrame({
        "Metric": ["10^x"] * 3 + ["sum(X)"] + ["2^x"] * 3,
        "MetricIndex0": list("abc") + [""] + list("def"),
        "MetricIndex1": list("xyz") + [""] * 4,
        "Value": [10, 100, 1000, 6, 2, 4, 8]
    })
    correct.set_index(["Metric", "MetricIndex0", "MetricIndex1"], inplace=True)
    pd.util.testing.assert_frame_equal(correct, output)

  def testMultiIndexSeriesWithFlatIndexSeriesNotMelted(self):
    data = pd.DataFrame({"X": [1, 2, 3]})
    multi_index = pd.MultiIndex.from_arrays([list("abc"), list("xyz")])
    ten_to_x = lambda df: pd.Series((10**(df.X)).values, index=multi_index)
    two_to_x = lambda df: pd.Series((2**(df.X)).values, index=list("def"))
    metric = [
        metrics.Metric("10^x", fn=ten_to_x),
        metrics.Sum("X"),
        metrics.Metric("2^x", fn=two_to_x)
    ]
    output = core.Analyze(data).calculate(metric).run()
    correct = pd.DataFrame({
        ("10^x", "a", "x"): [10],
        ("10^x", "b", "y"): [100],
        ("10^x", "c", "z"): [1000],
        ("sum(X)", "", ""): [6],
        ("2^x", "d", ""): [2],
        ("2^x", "e", ""): [4],
        ("2^x", "f", ""): [8],
    })
    correct = correct[["10^x", "sum(X)", "2^x"]]
    correct.columns.names = [None, "MetricIndex0", "MetricIndex1"]
    correct.index.name = ""
    pd.util.testing.assert_frame_equal(correct, output)

  def testMultiIndexSeriesWithFlatIndexSeriesSplitByMelted(self):
    multi_index = pd.MultiIndex.from_arrays([list("abc"), list("xyz")])
    ten_to_x = lambda df: pd.Series((10**(df.X)).values, index=multi_index)
    two_to_x = lambda df: pd.Series((2**(df.X)).values, index=list("def"))
    metric = [
        metrics.Metric("10^x", fn=ten_to_x),
        metrics.Sum("X"),
        metrics.Metric("2^x", fn=two_to_x)
    ]
    output = core.Analyze(self.data).split_by(["Y"]).calculate(metric).run(1)
    output = output.astype(int)
    correct = pd.DataFrame({
        "Y": [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2],
        "Metric": ["10^x"] * 9 + ["sum(X)"] * 3 + ["2^x"] * 9,
        "MetricIndex0":
            list("abc") * 3 + [""] * 3 + list("def") * 3,
        "MetricIndex1":
            list("xyz") * 3 + [""] * 12,
        "Value": [
            10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000,
            1000000000, 6, 15, 24, 2, 4, 8, 16, 32, 64, 128, 256, 512
        ]
    })
    correct.set_index(["Y", "Metric", "MetricIndex0", "MetricIndex1"],
                      inplace=True)
    pd.util.testing.assert_frame_equal(correct, output)

  def testMultiIndexSeriesWithFlatIndexSeriesSplitByNotMelted(self):
    multi_index = pd.MultiIndex.from_arrays([list("abc"), list("xyz")])
    ten_to_x = lambda df: pd.Series((10**(df.X)).values, index=multi_index)
    two_to_x = lambda df: pd.Series((2**(df.X)).values, index=list("def"))
    metric = [
        metrics.Metric("10^x", fn=ten_to_x),
        metrics.Sum("X"),
        metrics.Metric("2^x", fn=two_to_x)
    ]
    output = core.Analyze(self.data).split_by(["Y"]).calculate(metric).run()
    output = output.astype(int)
    correct = pd.DataFrame({
        ("10^x", "a", "x"): [10, 10000, 10000000],
        ("10^x", "b", "y"): [100, 100000, 100000000],
        ("10^x", "c", "z"): [1000, 1000000, 1000000000],
        ("sum(X)", "", ""): [6, 15, 24],
        ("2^x", "d", ""): [2, 16, 128],
        ("2^x", "e", ""): [4, 32, 256],
        ("2^x", "f", ""): [8, 64, 512],
    })
    correct = correct[["10^x", "sum(X)", "2^x"]]
    correct.columns.names = [None, "MetricIndex0", "MetricIndex1"]
    correct.index.name = "Y"
    pd.util.testing.assert_frame_equal(correct, output)

  def testFlatIndexSeriesPercentageDifferenceJackknifeMelted(self):
    # Output is screenshot/xXqjxgZb6eH
    flat_idx = pd.Series(
        list(range(3)), index=pd.Index(list("abc"), name="foo"))
    metric = [
        metrics.Sum("X"),
        metrics.Metric("Constant", fn=lambda df: flat_idx)
    ]
    comparison = comparisons.PercentageDifference("U", 1)
    se_method = standard_errors.Jackknife("Y")
    output = core.Analyze(self.data).relative_to(
        comparison).with_standard_errors(se_method).calculate(metric).run(1)
    sum_x = core.Analyze(self.data).relative_to(
        comparison).with_standard_errors(se_method).calculate(metric[0]).run(1)
    sum_x["foo"] = ""
    sum_x.set_index("foo", append=True, inplace=True)
    sum_x = sum_x.reorder_levels(["Metric", "foo", "U"])
    constant = pd.DataFrame({
        "Metric": ["Constant"] * 6,
        "foo": list("abcabc"),
        "U": [2] * 3 + [3] * 3,
        "Percentage Difference": [np.NAN, 0, 0] * 2,
        "Percentage Difference Jackknife SE": [np.NAN, 0, 0] * 2,
    })
    constant.set_index(["Metric", "foo", "U"], inplace=True)
    correct = pd.concat([sum_x, constant])
    pd.util.testing.assert_frame_equal(output, correct)

  def testFlatIndexSeriesAbsoluteDifferenceJackknifeNotMelted(self):
    # Output is screenshot/gWShTZiQyui.
    flat_idx = pd.Series(
        list(range(3)), index=pd.Index(list("abc"), name="foo"))
    metric = [
        metrics.Sum("X"),
        metrics.Metric("Constant", fn=lambda df: flat_idx)
    ]
    comparison = comparisons.AbsoluteDifference("U", 1)
    se_method = standard_errors.Jackknife("Y")
    output = core.Analyze(self.data).relative_to(
        comparison).with_standard_errors(se_method).calculate(metric).run()
    sum_x = core.Analyze(self.data).relative_to(
        comparison).with_standard_errors(se_method).calculate(metric[0]).run()
    sum_x.columns = pd.MultiIndex.from_product([sum_x.columns, [""]])
    sum_x.columns.names = [None, "foo"]
    constant = (core.Analyze(self.data).relative_to(
        comparison).with_standard_errors(se_method).calculate(metric[1])).run()
    correct = pd.concat([sum_x, constant], axis=1)
    pd.util.testing.assert_frame_equal(output, correct)

  def testFlatIndexSeriesAbsoluteDifferenceJackknifeNotMeltedNotFlatIndex(self):
    # Output is screenshot/i20HM28Ldvc
    flat_idx = pd.Series(
        list(range(3)), index=pd.Index(list("abc"), name="foo"))
    metric = [
        metrics.Sum("X"),
        metrics.Metric("Constant", fn=lambda df: flat_idx)
    ]
    comparison = comparisons.AbsoluteDifference("U", 1)
    se_method = standard_errors.Jackknife("Y", flat_index=False)
    output = core.Analyze(self.data).relative_to(
        comparison).with_standard_errors(se_method).calculate(metric).run()
    sum_x = core.Analyze(self.data).relative_to(
        comparison).with_standard_errors(se_method).calculate(metric[0]).run()
    sum_x.columns = pdutils.index_product(sum_x.columns,
                                          pd.Index([""], name="foo"))
    constant = (core.Analyze(self.data).relative_to(
        comparison).with_standard_errors(se_method).calculate(metric[1])).run()
    correct = pd.concat([sum_x, constant], axis=1)
    pd.util.testing.assert_frame_equal(output, correct)

  def testFlatIndexSeriesPercentageDifferenceJackknifeSplitByMelted(self):
    # Output is screenshot/tA2FNgbmXRT
    flat_idx = pd.Series(
        list(range(3)), index=pd.Index(list("abc"), name="foo"))
    metric = [
        metrics.Sum("X"),
        metrics.Metric("Constant", fn=lambda df: flat_idx)
    ]
    comparison = comparisons.AbsoluteDifference("type", "Desk")
    se_method = standard_errors.Jackknife("cookie")
    output = (core.Analyze(self.data_complex).split_by("grp").relative_to(
        comparison).calculate(metric).with_standard_errors(se_method)).run(1)
    sum_res = core.Analyze(
        self.data_complex).split_by("grp").relative_to(comparison).calculate(
            metric[0]).with_standard_errors(se_method).run(1)
    sum_res["foo"] = ""
    sum_res = sum_res.reset_index("type").set_index(["foo", "type"],
                                                    append=True)
    const_res = core.Analyze(
        self.data_complex).split_by("grp").relative_to(comparison).calculate(
            metric[1]).with_standard_errors(se_method).run(1)
    correct = pd.concat([sum_res, const_res])
    pd.util.testing.assert_frame_equal(output, correct)

  def testFlatIndexSeriesPercentageDifferenceJackknifeSplitByNotMelted(self):
    # Output is screenshot/h47VVTRNfVW
    flat_idx = pd.Series(
        list(range(3)), index=pd.Index(list("abc"), name="foo"))
    metric = [
        metrics.Sum("X"),
        metrics.Metric("Constant", fn=lambda df: flat_idx)
    ]
    comparison = comparisons.AbsoluteDifference("type", "Desk")
    se_method = standard_errors.Jackknife("cookie")
    output = (core.Analyze(self.data_complex).split_by("grp").relative_to(
        comparison).calculate(metric).with_standard_errors(se_method)).run()
    corrects = []
    for grp in self.n_grp:
      df = self.data_complex[self.data_complex["grp"] == grp]
      res = core.Analyze(df).relative_to(comparison).calculate(
          metric).with_standard_errors(se_method).run()
      corrects.append(res)
    correct = pd.concat(corrects, keys=self.n_grp, names=["grp"])
    pd.util.testing.assert_frame_equal(output, correct)

  def testSumInCustomFunction(self):
    correct_sum = metrics.Sum("X", "foo_name")
    lambda_sum = metrics.Metric("foo_name", fn=lambda df: sum(df.X))
    comparison = comparisons.AbsoluteDifference("Y", 0, include_base=True)
    se_method = standard_errors.Jackknife(unit="U")
    correct = core.Analyze(
        self.data).relative_to(comparison).with_standard_errors(
            se_method).calculate(correct_sum).run()
    output = core.Analyze(self.data).relative_to(
        comparison).with_standard_errors(se_method).calculate(lambda_sum).run()

    pd.util.testing.assert_frame_equal(correct, output)

  def testSumInCustomFunctionMelted(self):
    correct_sum = metrics.Sum("X", "foo_name")
    lambda_sum = metrics.Metric("foo_name", fn=lambda df: sum(df.X))
    comparison = comparisons.PercentageDifference("Y", 0, include_base=True)
    se_method = standard_errors.Jackknife(unit="U")
    correct = core.Analyze(
        self.data).relative_to(comparison).with_standard_errors(
            se_method).calculate(correct_sum).run(1)
    output = core.Analyze(self.data).relative_to(
        comparison).with_standard_errors(se_method).calculate(lambda_sum).run(1)

    pd.util.testing.assert_frame_equal(correct, output)

  def testDistributionWithOverlappingOverColumnsMelted(self):
    # The output is screenshot/Yj3jHqDpXdA
    metric = [
        metrics.Distribution("X", "Y"),
        metrics.CumulativeDistribution("U", ["X", "Y"]),
        metrics.Sum("X"),
        metrics.Metric("Doubled", fn=lambda df: pd.Series(2 * np.arange(3)))
    ]
    output = core.Analyze(self.data).calculate(metric).run(1)
    single_dist = core.Analyze(self.data).calculate(metric[0]).run(1)
    single_dist["X"], single_dist["MetricIndex0"] = "", ""
    single_dist.set_index(["X", "MetricIndex0"], append=True, inplace=True)
    # We need to switch the order of over columns to gropu by Y first because Y
    # is already used as grouping column in single_dist.
    multi_cum_dist = core.Analyze(self.data).calculate(
        metrics.CumulativeDistribution("U", ["Y", "X"]),).run(1)
    multi_cum_dist["MetricIndex0"] = ""
    multi_cum_dist.set_index("MetricIndex0", append=True, inplace=True)
    sum_x = pd.DataFrame({("sum(X)", "", "", ""): {"Value": 45.0}}).T
    doubled = core.Analyze(self.data).calculate(metric[-1]).run(1)
    doubled["X"], doubled["Y"] = "", ""
    doubled.set_index(["X", "Y"], append=True, inplace=True)
    order = ["Metric", "MetricIndex0", "Y", "X"]
    single_dist = single_dist.reorder_levels(order)
    multi_cum_dist = multi_cum_dist.reorder_levels(order)
    sum_x.index.names = order
    doubled = doubled.reorder_levels(order)
    correct = pd.concat([single_dist, multi_cum_dist, sum_x, doubled])
    pd.util.testing.assert_frame_equal(correct, output)

  _normalize_x = lambda d: pd.DataFrame({  # pylint: disable=g-long-lambda
      "L1": d["X"] / sum(d["X"]),
      "L2": d["X"]**2 / sum(d["X"]**2),
      "X": d["X"]
  }).set_index("X")
  over_metric = metrics.Metric("Normalized", fn=_normalize_x)
  simple_data = pd.DataFrame({
      "X": [1, 2],
  })
  split_by_data = pd.DataFrame({
      "X": [1, 2, 3, 4],
      "Y": [0, 0, 1, 1],
  })

  def testOverMetricMelted(self):
    # The output is screenshot/84Z6tZfeOkM
    correct = pd.DataFrame({
        "Metric": ["Normalized"] * 4,
        "MetricIndex0": ["L1", "L2", "L1", "L2"],
        "X": [1, 1, 2, 2],
        "Value": [1. / 3, 1. / 5, 2. / 3, 4. / 5]
    })
    correct.set_index(["Metric", "MetricIndex0", "X"], inplace=True)
    output = core.Analyze(self.simple_data).calculate(self.over_metric).run(1)
    pd.util.testing.assert_frame_equal(correct, output)

  def testOverMetricNotMelted(self):
    # The output is screenshot/oWV0YEVqbui
    correct = pd.DataFrame({
        "Metric": ["Normalized"] * 4,
        "MetricIndex0": ["L1", "L2", "L1", "L2"],
        "X": [1, 1, 2, 2],
        "Value": [1. / 3, 1. / 5, 2. / 3, 4. / 5]
    })
    correct.set_index(["Metric", "MetricIndex0", "X"], inplace=True)
    correct = correct.unstack(["Metric", "MetricIndex0"])
    correct.columns = correct.columns.droplevel(0)
    correct.columns.names = [None, "MetricIndex0"]
    output = core.Analyze(self.simple_data).calculate(self.over_metric).run()
    pd.util.testing.assert_frame_equal(correct, output)

  def testOverMetricWithSplitByMelted(self):
    # The output is screenshot/DhSEgejC7X0
    data = self.split_by_data
    correct = pd.DataFrame({
        "Metric": ["Normalized"] * 4,
        "MetricIndex0": ["L1", "L2", "L1", "L2"],
        "X": [1, 1, 2, 2],
        "Value": [1. / 3, 1. / 5, 2. / 3, 4. / 5]
    })
    groups = data.groupby(["Y"])
    results = []
    for _, grp in groups:
      results.append(core.Analyze(grp).calculate(self.over_metric).run(1))
    correct = pd.concat(results, keys=data["Y"].unique())
    correct.index.names = ["Y"] + correct.index.names[1:]
    output = core.Analyze(data).split_by("Y").calculate(self.over_metric).run(1)
    pd.util.testing.assert_frame_equal(correct, output)

  def testOverMetricWithSplitByNotMelted(self):
    # The output is screenshot/Vr10XLGOV3q
    data = self.split_by_data
    correct = pd.DataFrame({
        "Metric": ["Normalized"] * 4,
        "MetricIndex0": ["L1", "L2", "L1", "L2"],
        "X": [1, 1, 2, 2],
        "Value": [1. / 3, 1. / 5, 2. / 3, 4. / 5]
    })
    groups = data.groupby(["Y"])
    results = []
    for _, grp in groups:
      results.append(core.Analyze(grp).calculate(self.over_metric).run())
    correct = pd.concat(results, keys=data["Y"].unique())
    correct.index.names = ["Y"] + correct.index.names[1:]
    output = core.Analyze(data).split_by("Y").calculate(self.over_metric).run()
    pd.util.testing.assert_frame_equal(correct, output)

  over_and_index_metrics = [over_metric, metrics.Sum("X")]

  def testOverMetricWithRegularMetricMelted(self):
    # The output is screenshot/6kLoYtE3e9w
    data = self.simple_data
    correct = pd.DataFrame({
        "Metric": ["Normalized"] * 4 + ["sum(X)"],
        "MetricIndex0": ["L1", "L2", "L1", "L2", ""],
        "X": [1, 1, 2, 2, ""],
        "Value": [1. / 3, 1. / 5, 2. / 3, 4. / 5, 3]
    })
    correct.set_index(["Metric", "MetricIndex0", "X"], inplace=True)
    output = core.Analyze(data).calculate(self.over_and_index_metrics).run(1)
    pd.util.testing.assert_frame_equal(correct, output)

  def testOverMetricWithRegularMetricNotMelted(self):
    # The output is screenshot/KB38FYM7Ctv
    data = self.simple_data
    correct = pd.DataFrame({
        "Metric": ["Normalized"] * 4 + ["sum(X)"],
        "MetricIndex0": ["L1", "L2", "L1", "L2", ""],
        "X": [1, 1, 2, 2, ""],
        "Value": [1. / 3, 1. / 5, 2. / 3, 4. / 5, 3]
    })
    correct.set_index(["Metric", "MetricIndex0", "X"], inplace=True)
    correct = correct.unstack(["Metric", "MetricIndex0"])
    correct.columns = correct.columns.droplevel(0)
    correct.columns.names = [None, "MetricIndex0"]
    output = core.Analyze(data).calculate(self.over_and_index_metrics).run()
    pd.util.testing.assert_frame_equal(correct, output)

  def testOverMetricWithRegularMetricWithSplitByMelted(self):
    # The output is screenshot/idx8Q6S9AHG
    data = self.split_by_data
    normalized_x = core.Analyze(data).split_by("Y").calculate(
        self.over_and_index_metrics[0]).run(1)
    sum_x = core.Analyze(data).split_by("Y").calculate(
        self.over_and_index_metrics[1]).run(1)
    sum_x["MetricIndex0"] = ""
    sum_x["X"] = ""
    sum_x.set_index(["MetricIndex0", "X"], append=True, inplace=True)
    correct = pd.concat([normalized_x, sum_x])
    output = core.Analyze(data).split_by("Y").calculate(
        self.over_and_index_metrics).run(1)
    pd.util.testing.assert_frame_equal(correct, output)

  def testOverMetricWithRegularMetricWithSplitByNotMelted(self):
    # The output is screenshot/VrbDUTEVsM4
    data = self.split_by_data
    normalized_x = core.Analyze(data).split_by("Y").calculate(
        self.over_and_index_metrics[0]).run()
    sum_x = core.Analyze(data).split_by("Y").calculate(
        self.over_and_index_metrics[1]).run()
    sum_x["X"] = ""
    sum_x.set_index(["X"], append=True, inplace=True)
    sum_x.columns = pd.MultiIndex.from_product([sum_x.columns, [""]])
    correct = pd.concat([normalized_x, sum_x], axis=1)
    correct.columns.names = [None, "MetricIndex0"]
    output = core.Analyze(data).split_by("Y").calculate(
        self.over_and_index_metrics).run()
    pd.util.testing.assert_frame_equal(correct, output)


if __name__ == "__main__":
  googletest.main()
