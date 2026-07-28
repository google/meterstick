"""Meterstick example implementations and usage."""
# pytype: skip-file

import copy
from typing import Optional

import attrs
import numpy as np
import pandas as pd

try:
  # pylint: disable=g-import-not-at-top
  import pandas_gbq
  HAS_GBQ = True
except ImportError:
  HAS_GBQ = False

# ==============================================================================
# Data Setup
# ==============================================================================

np.random.seed(42)
n = 100
df = pd.DataFrame({
    'lost': np.random.choice([0, 1.], n),
    'region': np.random.choice(('US', 'non-US'), n),
    'experiment': np.random.choice(
        ('control', 'experiment1', 'experiment2', 'experiment3'), n
    ),
})

# To run the BigQuery/Cloud parts, you need a GCP project.
# 1. Set your GCP project ID here (replace 'meterstick' with your
# project ID):
project_id = 'meterstick'

# 2. Uncomment the following lines to upload the demo data to BigQuery:
# if not HAS_GBQ:
#   raise ImportError("pandas_gbq is required to upload data to BigQuery.")
# pandas_gbq.to_gbq(df, 'demo.data', project_id=project_id, if_exists='replace')


# ==============================================================================
# Metric implementations
# ==============================================================================


class Metric:
  """Base class for all metrics."""

  def compute_on(self, data, split_by=None):
    if split_by:
      res = self.compute(data.groupby(split_by))
    else:
      res = [self.compute(data)]
    res = pd.DataFrame(res)
    res.columns = self.names
    return res

  def set_names(self, names):
    self._names = names
    return self

  @property
  def names(self):
    return getattr(self, '_names', self.default_names)

  def sql_aggregate(self, data, dimensions):
    # Helper function for constructing aggregation queries.
    dim_sql = ','.join(dimensions) + ',' if dimensions else ''
    groupby = 'GROUP BY ' + ','.join(dimensions) if dim_sql else ''
    val_cols = ','.join([f'{s} AS {n}' for s, n in zip(self.sql, self.names)])
    return f'SELECT {dim_sql} {val_cols} FROM {data} {groupby}'

  def to_sql(self, data, split_by=None):
    return self.sql_aggregate(data, split_by)

  def compute_on_sql(self, data, split_by=None):
    """Computes the metric using SQL in BigQuery."""
    # Note: This requires project_id to be set and data to be uploaded to
    # BigQuery.
    if not HAS_GBQ:
      raise ImportError(
          'pandas_gbq is not installed. Please install it to run SQL'
          ' computations.'
      )
    res = pandas_gbq.read_gbq(
        self.to_sql(data, split_by), project_id=project_id
    )
    dims = split_by + self.extra_dims
    return res.set_index(dims).sort_index() if dims else res

  def __truediv__(self, other):
    return Div(self, other)

  def __or__(self, fn):
    """Overwrites the '|' operator to enable pipeline chaining."""
    return fn(self)


class Operation(Metric):
  """Base class for operations that take one or more metrics as input."""

  def compute_on(self, data, split_by=None):
    data_preprocessed = self.preprocess(data, split_by)
    child_res = self.compute_children(data_preprocessed, split_by)
    return self.process_results(child_res, split_by)

  def __call__(self, child: Metric):
    op = copy.deepcopy(self) if self.child else self
    op.child = child
    return op

  def sql_select(self, data, dimensions):
    # Helper function for constructing select queries.
    dim_sql = ','.join(dimensions) + ',' if dimensions else ''
    val_cols = ','.join([f'{s} AS {n}' for s, n in zip(self.sql, self.names)])
    return f'SELECT {dim_sql} {val_cols} FROM {data}'

  def to_sql(self, data, split_by=None):
    data_preprocessed = self.preprocess_sql(data, split_by)
    children_query = self.children_to_sql(data_preprocessed, split_by)
    return self.assemble_query(children_query, split_by)

  @property
  def extra_dims(self):
    return []


@attrs.define
class Sum(Metric):
  """A metric that computes the sum of a variable."""
  var: str

  def compute(self, data):
    return data[self.var].sum()

  @property
  def default_names(self):
    return [f'sum_{self.var}']

  @property
  def sql(self):
    return [f'SUM({self.var})']


@attrs.define
class Count(Metric):
  """A metric that computes the count of a variable."""
  var: str

  def compute(self, data):
    return data[self.var].count()

  @property
  def default_names(self):
    return [f'count_{self.var}']

  @property
  def sql(self):
    return [f'COUNT({self.var})']


@attrs.define
class Div(Operation):
  """An operation that divides one metric by another."""

  child1: Metric
  child2: Metric

  def preprocess(self, data, split_by):  # pylint: disable=unused-argument
    return data

  def compute_children(self, data, split_by):
    return (
        self.child1.compute_on(data, split_by),
        self.child2.compute_on(data, split_by),
    )

  def process_results(self, child_res, split_by):  # pylint: disable=unused-argument
    num, denom = child_res
    num.columns = self.names
    denom.columns = self.names
    return num / denom

  @property
  def default_names(self):
    return map('_div_'.join, zip(self.child1.names, self.child2.names))

  def preprocess_sql(self, data, split_by):  # pylint: disable=unused-argument
    return data

  def children_to_sql(self, data, split_by):  # pylint: disable=unused-argument
    return data

  def assemble_query(self, child_res, split_by):
    return self.sql_aggregate(child_res, split_by)

  @property
  def sql(self):
    return map(' / '.join, zip(self.child1.sql, self.child2.sql))

  @property
  def extra_dims(self):
    return []


@attrs.define
class PercentChange(Operation):
  """An operation that computes the percent change against a baseline."""
  condition: str
  baseline: str
  child: Optional[Metric] = None

  def preprocess(self, data, split_by):  # pylint: disable=unused-argument
    return data

  def compute_children(self, data, split_by):
    return self.child.compute_on(data, split_by + [self.condition])

  def process_results(self, child_res, split_by):
    if split_by:
      base = child_res.xs(self.baseline, level=self.condition)
    else:
      base = child_res.loc[self.baseline]
    res = child_res / base - 1
    res.columns = self.names
    return res * 100

  @property
  def default_names(self):
    return [f'pct_change_of_{n}' for n in self.child.names]

  def preprocess_sql(self, data, split_by):  # pylint: disable=unused-argument
    return data

  def children_to_sql(self, data, split_by):
    return self.child.to_sql(data, split_by + [self.condition])

  def assemble_query(self, child_res, split_by):
    dims = self.extra_dims + split_by
    u = ','.join(dims[1:])
    join = f'T JOIN Base USING ({u})'
    if not u:
      join = 'T CROSS JOIN Base'
    return f"""
    WITH T AS ({child_res}),
    Base AS (SELECT *
    EXCEPT ({self.condition}) FROM T
    WHERE {self.condition}
      = '{self.baseline}')
    {self.sql_select(join, dims)}"""

  @property
  def sql(self):
    return [f'(T.{c} / Base.{c} - 1) * 100' for c in self.child.names]

  @property
  def extra_dims(self):
    return [self.condition] + self.child.extra_dims


@attrs.define
class Bootstrap(Operation):
  """An operation that computes bootstrap standard errors for a metric."""

  n_rep: int = attrs.field(default=50)
  child: Optional[Metric] = None

  def preprocess(self, data, split_by):  # pylint: disable=unused-argument
    for _ in range(self.n_rep):
      yield data.sample(frac=1, replace=True)

  def compute_children(self, data, split_by):
    sample_res = [self.child.compute_on(sample, split_by) for sample in data]
    return pd.concat(sample_res, axis=1)

  def process_results(self, child_res, split_by):  # pylint: disable=unused-argument
    std = child_res.T.groupby(level=0).std().T
    std.columns = self.names
    return std

  @property
  def default_names(self):
    return [f'se_{n}' for n in self.child.names]

  def preprocess_sql(self, data, split_by):
    return resample_n_times(data, split_by, self.n_rep)

  def children_to_sql(self, data, split_by):
    return (*data, self.child.to_sql('Samples', split_by + ['sample_idx']))

  def assemble_query(self, child_res, split_by):
    (input_data, samples, sample_res) = child_res
    sql = self.sql_aggregate('SampleRes', split_by + self.extra_dims)
    return f"""
      CREATE TEMP TABLE Data
        AS ({input_data});
      WITH Samples AS ({samples}),
      SampleRes AS ({sample_res})
      {sql}"""

  @property
  def sql(self):
    return [f'STDDEV({n})' for n in self.child.names]

  @property
  def extra_dims(self):
    return self.child.extra_dims


def resample_n_times(data, split_by, n_rep):
  """Generates SQL queries for resampling data N times for bootstrap.

  Args:
    data: The name of the input table or query.
    split_by: The dimensions to split the data by.
    n_rep: The number of bootstrap replications.

  Returns:
    A tuple of two SQL strings:
      1. A query to prepare the input data with random row numbers.
      2. A query to generate the resampled data by joining the prepared data
         with itself.
  """
  by_sql = ','.join(split_by) + ',' if split_by else ''
  input_data = f"""
    SELECT
      *,
      ROW_NUMBER() OVER (PARTITION BY sample_idx) AS row_number,
      CEIL(RAND() * COUNT(*) OVER (PARTITION BY sample_idx))
        AS random_row_number,
    FROM {data},
    UNNEST(GENERATE_ARRAY(1, {n_rep})) AS sample_idx"""
  samples = f"""
    SELECT b.*
    FROM (
      SELECT
        {by_sql}
        sample_idx,
        random_row_number AS row_number
      FROM Data) AS a
    JOIN Data AS b
    USING ({by_sql} sample_idx, row_number)"""
  return (input_data, samples)


# ==============================================================================
# Execution
# ==============================================================================

if __name__ == '__main__':
  split_bys = [[],]
  churn = (Sum('lost') / Count('lost')).set_names(['churn'])
  pct = churn | PercentChange('experiment', 'control')
  bst = pct | Bootstrap()

  for s_by in split_bys:
    print(f'split_by is {s_by}\n\n')
    print(f'Churn rate is {churn.compute_on(df, s_by)}\n')
    print(f'Percent change is {pct.compute_on(df, s_by)}\n')
    print(f'Bootstrap is {bst.compute_on(df, s_by)}\n')

  # SQL computation (requires GCP project and data upload)
  # for s_by in split_bys:
  #   print(f'split_by is {s_by}\n\n')
  #   print(f"Churn rate is {churn.compute_on_sql('demo.data', s_by)}\n")
  #   print( f"Percent change is {pct.compute_on_sql('demo.data', s_by)}\n")
  #   print(f"Bootstrap is {bst.compute_on_sql('demo.data', s_by)}\n")
