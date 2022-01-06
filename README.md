# Meterstick Documentation

The meterstick package provides a concise syntax to describe and execute
routine data analysis tasks. Please see meterstick_demo.ipynb for examples.

## Disclaimer

This is not an officially supported Google product.


## tl;dr

Modify the demo colab [notebook](https://colab.research.google.com/github/google/meterstick/blob/master/meterstick_demo.ipynb) and adapt it to your needs.

## Building up an analysis

Every analysis starts with a `Metric` or a `MetricList`. A full list of Metrics
can be found below.

A `Metric` may be modified by one or more `Operation`s. For example, we might
want to calculate a confidence interval for the metric, a treatment-control
comparison, or both.

Once we have specified the analysis, we pass in the data to compute the
analysis on, as well as variables to slice by.

Here is an example of a full analysis:

```python
# define metrics
cvr = Ratio("Conversions", "Visits")
bounce_rate = Ratio("Bounces", "Visits")

(MetricList((cvr, bounce_rate))
 | PercentChange("Experiment", "Control")
 | Jackknife("Cookie", confidence=.95)
 | compute_on(data, ["Country", "Device"]))
```

This calculates the percent change in conversion rate and bounce rate,
relative to the control arm, for each country and device, together with
95% confidence intervals based on jackknife standard errors.


## Building Blocks of an Analysis Object

### Metrics

A Meterstick analysis begins with one or more metrics.

Currently built-in metrics include:

+   `Count(variable)`: calculates the number of (non-null) entries of `variable`
+   `Sum(variable)` : calculates the sum of `variable`
+   `Mean(variable)`: calculates the mean of `variable`
+   `Max(variable)`: calculates the max of `variable`
+   `Min(variable)`: calculates the min of `variable`
+   `Ratio(numerator, denominator)` : calculates `Sum(numerator) /
    Sum(denominator)`.
+   `Quantile(variable, quantile(s))`: calculates the `quantile(s)` quantile for
    `variable`.
+   `Variance(variable, unbiased=True)`: calculates the variance of `variable`;
    `unbiased` determines whether the unbiased (sample) or population estimate
    is used.
+   `StandardDeviation(variable, unbiased=True)`: calculates the standard
    deviations of `variable`; `unbiased` determines whether the unbiased or MLE
    estimate is used.
+   `CV(variable, unbiased=True)`: calculates the coefficient of variation of
    `variable`; `unbiased` determines whether the unbiased or MLE estimate of
    the standard deviation is used.
+   `Correlation(variable1, variable2)`: calculates the Pearson correlation
    between `variable1` and `variable2`.
+   `Cov(variable1, variable2)`: calculates the covariance between `variable1`
    and `variable2`.

All metrics have an optional `name` argument which determines the column name
in the output. If not specified, a default name will be provided. For instance,
the metric `Sum("Clicks")` will have the default name `sum(Clicks)`.

Metrics such as `Mean` and `Quantile` have an optional `weight` argument that
specifies a weighting column. The resulting metric is a weighted mean or
weighted quantile.

To calculate multiple metrics at once, create a `MetricList` of the individual
`Metric`s. For example, to calculate both total visits and conversion rate,
we would write:

```python
sum_visits = Sum("Visits")
MetricList([sum_visits, Sum("Conversions") / sum_visits])
```

When computing analyses involving multiple metrics, Meterstick will try to
cache redundant computations. For example, both metrics above require
calculating `Sum("Visits")`; Meterstick will only calculate this once.

You can also define custom metrics. See section `Custom Metric` below for
instructions.

#### Composite Metrics

Metrics are also **composable**. For example, you can:

+ Add metrics: `Sum("X") + Sum("Y")` or `Sum("X") + 1`.
+ Subtract metrics: `Sum("X") - Sum("Y")` or `Sum("X") - 1`.
+ Multiply metrics: `Sum("X") * Sum("Y")` or `100 * Sum("X")`.
+ Divide metrics: `Sum("X") / Sum("Y")` or `Sum("X") / 2`.
  (Note that the first is equivalent to `Ratio("X", "Y")`.)
+ Raise metrics to a power: `Sum("X") ** 2` or `2 ** Sum("X")` or
  `Sum("X") ** Sum("Y")`.
+ ...or any combination of these: `100 * (Sum("X") / Sum("Y") - 1)`.

Common metrics can be implemented as follows:

+   Click-through rate: `Ratio('Clicks', 'Impressions', 'CTR')`
+   Conversion rate: `Ratio('Conversions', 'Visits', 'CvR')`
+   Bounce rate: `Ratio('Bounce', 'Visits', 'BounceRate')`
+   Cost per click (CPC): `Ratio('Cost', 'Clicks', 'CPC')`

### Operations

Operations are defined on top of metrics. Operations include comparisons,
standard errors, and distributions.

#### Comparisons

A **comparison** operation calculates the change in a metric between various
conditions and a baseline. In A/B testing, the "condition" is
typically a treatment and the "baseline" a control.

Built-in comparisons include:

+   `PercentChange(condition_column, baseline)` : Computes the percent change
    (other - baseline) / baseline.
+   `AbsoluteChange(condition_column, baseline)` : Computes the absolute change
    (other - baseline).
+   `MH(condition_column, baseline, stratified_by)` : Computes the
    [Mantel-Haenszel estimator](https://en.wikipedia.org/wiki/Cochran%E2%80%93Mantel%E2%80%93Haenszel_statistics).
    The metric being computed must be a `Ratio` or a `MetricList` of `Ratio`s.
    The `stratified_by` argument specifies the strata over which the MH
    estimator is computed.
+   `CUPED(condition_column, baseline, covariates, stratified_by)` : Computes
    the absolute change that has been adjusted using the
    [CUPED](http://bit.ly/expCUPED) approach. See the
    [demo](https://colab.research.google.com/github/google/meterstick/blob/master/meterstick_demo.ipynb#scrollTo=_jqCd-ZHwv8i) for details.
+   `PrePostChange(condition_column, baseline, covariates, stratified_by)` :
    Computes the percent change that has been adjusted using the
    [PrePost](https://arxiv.org/pdf/1711.00562.pdf) approach. See the
    [demo](https://colab.research.google.com/github/google/meterstick/blob/master/meterstick_demo.ipynb#scrollTo=x1wSwf3MT5Yf) for details.

Example Usage: `... | PercentChange("Experiment", "Control")`

Note that `condition_column` can be a list of columns, in which case `baseline`
should be a tuple of baselines, one for each condition variable.

#### Standard Errors

A **standard error** operation adds the standard error of the metric
(or confidence interval) to the point estimate.

Built-in standard errors include:

+   `Jackknife(unit, confidence)` : Computes a leave-one-out jackknife estimate
    of the standard error of the child Metric.

    `unit` is a string for the variable whose unique values will be resampled.

    `confidence` in (0,1) represents the level of the conidence interval;
    optional

+   `Bootstrap(unit, num_replicates, confidence)` : Computes a bootstrap
    estimate of the standard error.

    `num_replicates` is the number of bootstrap replicates, default is 10000.

    `unit` is a string for the variable whose unique values will be resampled;
    if `unit` is not supplied the rows will be the unit.

    `confidence` in (0,1) represents the level of the conidence interval;
    optional

Example Usage: `... | Jackknife('CookieBucket', confidence=.95)`

#### Distributions

A **distribution** operation produces the distribution of the metric over
a variable.

+   `Distribution(over)`: calculates the distribution of the metric over the
    variables in `over`; the values are normalized so that they sum to 1. It has
    an alias `Normalize`.
+   `CumulativeDistribution(over, order=None, ascending=True)`: calculates the
    cumulative distribution of the metric over the variables in `over`. The
    `over` column will be sorted. You can pass in a list of values as a custom
    `order`. `ascending` determines whether the variables in `over` should be
    sorted in ascending or descending order.

Example Usage: `Sum("Queries") | Distribution("Device")` calculates the
proportion of queries that come from each device.

### Models

A Meterstick **Model** fits a model on data computed by children Metrics.

`Model(y, x, groupby).compute_on(data)` is equivalent to

1.  Computes `y.compute_on(data, groupby)` and `x.compute_on(data, groupby)`.
2.  Fits the underlying model on the results from #1.

We have built-in support for `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`
and `LogisticRegression`. Example Usage: `LinearRegression(Sum('Y'), Sum('X'),
'country')` calculates the sum of Y and X by country respectively, then fits a
linear regression between them.

Note that `x`, the 2nd arg, can be a Metric, a MetricList, or a list of Metrics.

### Filtering

We can restrict our metrics to subsets of the data. For instance to calculate
metrics for non-spam clicks you can add a `where` clause to the Metric or
MetricList. This clause is a boolean expression which can be passed to pandas'
[query() method](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html).


```python
sum_non_spam_clicks = Sum("Clicks", where="~IsSpam")
MetricList([Sum("Clicks"), Sum("Conversions")], where="~IsSpam")
```

### Data and Slicing

Once we have specified the metric(s) and operation(s), it is time to
compute the analysis on some data. The final step is to pass in the data,
along with any variables we want to slice by. The analysis will be carried out
for each slice separately.

The data can be supplied in two forms:

+  a pandas `DataFrame`
+  a string representing a SQL table or subquery.

Example Usage: `compute_on(df, ["Country", "Device"])`

Example Usage:

`compute_on_sql("SELECT * FROM table WHERE date = '20200101'", "Country")`

#### Customizing the Output Format

When calculating multiple metrics, Meterstick will store each metric as a
separate column by default. However, it is sometimes more convenient to store
the data in a different shape: with one column storing the metric values and
another column storing the metric names. This makes it easier to facet by metric
in packages like `ggplot2` and `altair`. This is known as the "melted"
representation of the data. To return the output in melted form, simply add the
argument `melted=True` in compute_on() or compute_on_sql().

#### Visualization

If the last operation applied to the metric is [Jackknife](https://colab.research.google.com/github/google/meterstick/blob/master/meterstick_demo.ipynb#scrollTo=53NI01DoqyDe) or [Bootstrap](https://colab.research.google.com/github/google/meterstick/blob/master/meterstick_demo.ipynb#scrollTo=uKBRJlBBqskw) with
confidence, the output can be displayed in a way that highlights significant changes by calling
`.display()`.

![Rasta-style display of Meterstick result](http://services.google.com/fh/files/misc/confidence_interval_display.png)

You can customize the `display`. It takes the same arguments as the underlying
visualization
[library](https://colab.research.google.com/github/google/meterstick/blob/master/confidence_interval_display_demo.ipynb).

## SQL

You can get the SQL query for all built-in Metrics and Operations (except
weighted Quantile/CV/Correlation/Cov) by calling `to_sql(sql_data_source,
split_by)` on the Metric. `sql_data_source` could be a table or a subquery. The
dialect it uses is the [standard SQL](https://cloud.google.com/bigquery/docs/reference/standard-sql)
in Google Cloud's BigQuery. For example,

```python
MetricList((Sum('X', where='Y > 0'), Sum('X'))).to_sql('table', 'grp')
```

gives

```sql
SELECT
  grp,
  SUM(IF(Y > 0, X, NULL)) AS `sum(X)`,
  SUM(X) AS `sum(X)_1`
FROM table
GROUP BY grp
```

Very often what you need is the execution of the SQL query, then you can call

```
compute_on_sql(sql_data_source, split_by=None, execute=None, melted=False)
```

directly, which will give you a output similar to `compute_on()`. `execute` is a
function that can execute SQL query.

## Custom Metric

You can write your own Metric and Operartion. Below is a Metric taken from the demo [colab](https://colab.sandbox.google.com/github/google/meterstick/blob/master/meterstick_demo.ipynb#scrollTo=QFjhj96EdK-r).
The Metric fits a LOWESS model.

```python
import statsmodels.api as sm
lowess = sm.nonparametric.lowess

class Lowess(Metric):
 def __init__(self, x, y, name=None, where=None):
   self.x = x
   self.y = y
   name = name or 'LOWESS(%s ~ %s)' % (y, x)
   super(Lowess, self).__init__(name, where=where)

 def compute(self, data):
   lowess_fit = pd.DataFrame(
       lowess(data[self.y], data[self.x]), columns=[self.x, self.y])
   return lowess_fit.drop_duplicates().reset_index(drop=True)
```

As long as the Metric obeys some [rules](https://colab.research.google.com/github/google/meterstick/blob/master/meterstick_demo.ipynb#scrollTo=AQjJAr3YcQB2), it
will work with all built-in Metrics and Operations. For example, we can pass it
to `Jackknife` to get a confidence interval.

```python
jk = Lowess('x', 'y') | Jackknife('cookie', confidence=0.9) | compute_on(df)
point_est = jk[('y', 'Value')]
ci_lower = jk[('y', 'Jackknife CI-lower')]
ci_upper = jk[('y', 'Jackknife CI-upper')]

plt.scatter(df.x, df.y)
plt.plot(x, point_est, c='g')
plt.fill_between(x, ci_lower, ci_upper, color='g', alpha=0.5)
plt.show()
```
![LOWESS with jackknife](http://services.google.com/fh/files/misc/lowess.png)
