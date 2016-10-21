# MeterStick Documentation

The meterstick package provides a concise syntax to describe and execute
routine data analysis tasks.

[TOC]

## Building up an analysis

The `Analyze` function creates an analysis object. Everything else is
implemented as a method of this analysis object and (except for `calculate`)
returns an analysis object with the appropriate changes. Thus, an analysis can
be specified by chained method calls starting with `Analyze` and ending with
`calculate`..

Here is an example of a full analysis:

```python
(Analyze(data).
  calculate([Ratio('Conversion', 'Clicks'), Ratio('Clicks', 'Impressions')]).
  relative_to(PercentageDifference('Experiment', 'Control')).
  split_by('Country').
  with_standard_errors(Jackknife()).
  run())
```

This calculates the percent change over the control condition for CvR and CTR
for desktop split by country with jackknife standard errors.

Repeating an operation group will raise an error.

```python
(Analyze(data).
  calculate(Ratio('Conversion', 'Clicks'))).
  split_by('Country').
  split_by('Device'). # this will raise an error
  run())
```

## Building Blocks of an Analysis Object

### Data

Every analysis needs a source of data. Thus, every analysis object
begins with `Analyze(data)` which creates an analysis object for the
data `data`. Currently only pandas dataframes are supported.

To make the best use of the library functions endeavor to transform the data
into a [tidy](http://vita.had.co.nz/papers/tidy-data.pdata) format:

+   Each variable forms a column
+   Each observational unit forms a row

As an example, consider a data set with the following columns: (AdvertiserId,
Clicks-Control, Impressions-Control, Clicks-Experiment,
Impressions-Experiment). This violates the rule that each variable forms a
column; as a consequence it's difficult to calculate the percent difference in
CTR split by experimental condition. At best you can run the following analysis
and manually compare the results.

```python
(Analyze(data).
  calculate([Ratio('Clicks-Control', 'Impressions-Control'),
             Ratio('Clicks-Experiment', 'Impressions-Experiment')]).
  run())
```

In contrast, this data could be equivalently expressed by the following columns:
(AdvertiserId, Condition, Clicks, Impressions). Now calculating we can calculate
this difference using the following analysis.

```python
(Analyze(data).
  relative_to(PercentageDifference('Condition', 'Control')).
  calculate(Ratio('Clicks', 'Impressions')).
  run())
```

For guidance in transforming your data into a tidy format, consult panda's
documentation on
[reshaping](http://pandas.pydata.org/pandas-docs/stable/reshaping.html).

### Subsetting and splitting

The library implements some basic data manipulation operations. If
more complex data manipulation steps are required, it's suggested to
preprocess the dataset using more specialized tools before starting to
analyze the data.

The `where(query)` method applies the analysis only to the subset of the data
where the query is true. The query should be a string which, in the context of
the dataframe, evaluates to an array of booleans. Thus, `where("Device in
['google', 'google_table']")` will restrist the analysis only to those two
devices. Where methods can be chained and are combined by an implicit "and".

The `split_by(variables)` method applies the analysis separately to
every subset of the data defined by a unique combination of the values
of `variables`. For instance, `split_by('Device')` will run the
analysis on the dataframe will return results for `Device1` and
`Device2` separately.

### Comparison functions

A function which takes a metric and compares them across subgroups
is a comparison function.

These are added by the `relative_to(comparison)` method. Each
comparison method takes a `condition` and `baseline` argument. The
`condition` argument is a string for the variable which represents the
conditions that are to be compared.  The `baseline` argument is the
value of the `condition` variable which will be designated as the
baseline against which to compare.

Currently supported comparisons include:

+   `PercentageDifference(condition, baseline)` : Computes the percent
    change (other - baseline) / baseline
+   `AbsoluteDifference(condition, baseline)` : Computes the absolute
    change other - baseline
+   `MH(condition, baseline, index)` : Computes the
    [Mantel-Haenszel estimator]
    (http://www.statsdirect.com/help/meta_analysis/mh.htm). The
    metric being computed must be a `Ratio`. The `index` variable
    specifies the group ids for the MH estimator.

A common use case would be
`relative_to(PercentageDifference('Experimental Condition',
'Control'))`

### Standard error functions

A function which takes a metric or comparison and calculates
standard errors is a standard error function

Standard error functions are added to an analysis using the
`with_standard_errors(standard_error_function)` method.

Currently implemented standard error functions include:

+   `Jackknife(unit, leave_out)` : Computes a leave-one-out jackknife
    estimate of the standard error.

    `unit` is a string for the variable whose unique values will be
    resampled; if `unit` is not supplied the rows will be the unit.

    `leave_out` is an integer for the number of buckets to be
    left-out. The default value is 1 for leave-one-out jackknife.

+   `Bootstrap(num_replicates, unit)` : Computes a bootstrap estimate
    of the standard error.

    `num_replicates` is the number of bootstrap replicates.

    `unit` is a string for the the variable whose unique values will
    be resampled; if `unit` is not supplied the rows will be the unit.

A sample usage is `with_standard_errors(Jackknife())` which will add
the leave-one-out jackknife on rows.

### Metrics

The previous methods merely set up the analysis. To actually run the analysis,
use the `calculate()` method on the analysis object created using the above
methods.  The results will be in the form of a pandas dataframe.

A function which takes a dataframe and returns a result (usually a single
number) is a metric.  These are added by the `calculate(metric)`
method.

Currently supported metrics include:

+   `Sum(variable)` : calculates the sum of `variable`
+   `Mean(variable)`: calculates the mean of `variable`
+   `Weighted Mean(variable, weight_variable)` : calculates the
    weighted mean of `variable` with the weights from
    `weight_variable`.
+   `Ratio(numerator, denominator)` : calculates `Sum(numerator) /
    Sum(denominator)`.
+   `Quantile(variable, quantile)`: calculates the `quantile` quantile
    for `variable`.
+   `Variance(variable, unbiased=True)`: calculates the variance of `variable`;
    `unbiased` determines whether the unbiased or MLE estimate is used.
+   `StandardDeviation(variable, unbiased=True)`: calculates the standard
    deviations of `variable`; `unbiased` determines whether the unbiased or MLE
    estimate is used.
+   `Correlation(variable1, variable2)`: calculates the Pearson correction
    between `variable1` and `variable2`.
+   `CV(variable, unbiased=True)`: calculates the coefficient of variation of
    `variable`; `unbiased` determines whether the unbiased or MLE estimate of
    the standard deviation is used.

All metrics have an optional `name` argument which determines the column name
for results. If not specified, a default value will be provided. For instance,
the metric `Sum("Clicks")` will have default name `sum(Clicks)`.

Common metrics can be implemented (modulo variable names) as follows:

+   Click-through rate (CTR): `Ratio('Clicks', 'Impressions', name = 'CTR')`
+   Conversion rate (CvR): `Ratio('Conversions', 'Clicks', name = 'CvR')`
+   Cost per click (CPC): `Ratio('Cost', 'Clicks', name = 'CPC')`
