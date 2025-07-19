# Copyright 2023 Google LLC
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
"""Module to generate SQL scripts for Metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from collections import abc
import copy
import functools
import re
from typing import Any, Iterable, List, Optional, Text, Union


DEFAULT_DIALECT = 'GoogleSQL'
DIALECT = None
# If need to use CREATE TEMP TABLE. It's only needed when the engine doesn't
# evaluate RAND() only once in the WITH clause. Namely,
# run_only_once_in_with_clause() returns False.
NEED_TEMP_TABLE = None
CREATE_TEMP_TABLE_FN = None
# ORDER BY is required for ROW_NUMBER() in some dialects.
ROW_NUMBER_REQUIRE_ORDER_BY = None
GROUP_BY_FN = None
RAND_FN = None
CEIL_FN = None
SAFE_DIVIDE_FN = None
QUANTILE_FN = None
ARRAY_AGG_FN = None
ARRAY_INDEX_FN = None
NTH_VALUE_FN = None
COUNTIF_FN = None
FLOAT_CAST_FN = None
STRING_CAST_FN = None
UNIFORM_MAPPING_FN = None
UNNEST_ARRAY_FN = None
UNNEST_ARRAY_LITERAL_FN = None
GENERATE_ARRAY_FN = None
DUPLICATE_DATA_N_TIMES_FN = None


def drop_table_if_exists(alias: str):
  return f'DROP TABLE IF EXISTS {alias};'


def drop_temp_table_if_exists(alias: str):
  return f'DROP TEMPORARY TABLE IF EXISTS {alias};'


def drop_table_if_exists_then_create_temp_table(alias: str, query: str):
  """Drops a table if it exists then creates a temporary table."""
  return (
      drop_table_if_exists(alias)
      + f'\nCREATE TEMPORARY TABLE {alias} AS {query}'
  )


def drop_temp_table_if_exists_then_create_temp_table(alias: str, query: str):
  """Drops a table if it exists then creates a temporary table."""
  return (
      drop_temp_table_if_exists(alias)
      + f'\nCREATE TEMPORARY TABLE {alias} AS {query}'
  )


def create_temp_table_fn_not_implemented(alias: str, query: str):
  del alias, query  # Unused
  raise NotImplementedError('CREATE TEMP TABLE is not implemented.')


def sql_server_rand_fn_not_implemented():
  raise NotImplementedError(
      "SQL Server's RAND() without a seed parameter will return the same value"
      " for every row within the same SELECT statement, which doesn't work"
      ' for us.'
  )


def safe_divide_fn_default(numer: str, denom: str):
  return (
      f'CASE WHEN {{denom}} = 0 THEN NULL ELSE {FLOAT_CAST_FN("{numer}")} /'
      f' {FLOAT_CAST_FN("{denom}")} END'.format(numer=numer, denom=denom)
  )


def approx_quantiles_fn(percentile):
  p = int(100 * percentile)
  return f'APPROX_QUANTILES({{}}, 100)[SAFE_OFFSET({p})]'


def percentile_cont_fn(percentile):
  return f'PERCENTILE_CONT({percentile}) WITHIN GROUP (ORDER BY {{}})'


def approx_percentile_fn(percentile):
  return f'APPROX_PERCENTILE({{}}, {percentile})'


def quantile_fn_not_implemented(percentile):
  del percentile  # Unused
  raise NotImplementedError('Quantile is not implemented.')


def array_agg_fn_googlesql(
    sort_by: Optional[str],
    ascending: Optional[bool],
    dropna: Optional[bool],
    limit: Optional[int],
):
  """Uses GoogleSQL's ARRAY_AGG to aggregate arrays."""
  dropna = ' IGNORE NULLS' if dropna else ''
  order_by = f' ORDER BY {sort_by}' if sort_by else ''
  if order_by is not None:
    order_by += '' if ascending else ' DESC'
  limit = f' LIMIT {limit}' if limit else ''
  return f'ARRAY_AGG({{}}{dropna}{order_by}{limit})'


def array_agg_fn_no_use_filter_no_limit(
    sort_by: Optional[str],
    ascending: Optional[bool],
    dropna: Optional[bool],
    limit: Optional[int],
):
  """Uses ARRAY_AGG to aggregate arrays. Use FILTER to filter out NULLs."""
  del limit  # LIMIT is not supported in PostgreSQL so just skip.
  dropna = ' FILTER (WHERE {} IS NOT NULL)' if dropna else ''
  order_by = f' ORDER BY {sort_by}' if sort_by else ''
  if order_by is not None:
    order_by += '' if ascending else ' DESC'
  return f'ARRAY_AGG({{}}{order_by}){dropna}'


def json_array_agg_fn(
    sort_by: Optional[str],
    ascending: Optional[bool],
    dropna: Optional[bool],
    limit: Optional[int],
):
  """Uses JSON_ARRAYAGG to aggregate arrays."""
  del limit  # LIMIT is not supported in PostgreSQL so just skip.
  if not dropna:
    raise NotImplementedError('Respecting NULLS is not supported.')
  order_by = f' ORDER BY {sort_by}' if sort_by else ''
  if order_by is not None:
    order_by += '' if ascending else ' DESC'
  return f'JSON_ARRAYAGG({{}}{order_by})'


def array_agg_fn_not_implemented(
    sort_by: Optional[str],
    ascending: Optional[bool],
    dropna: Optional[bool],
    limit: Optional[int],
):
  del sort_by, ascending, dropna, limit  # Unused
  raise NotImplementedError('ARRAY_AGG is not implemented.')


def array_index_fn_googlesql(array: str, zero_based_idx: int):
  return f'{array}[SAFE_OFFSET({zero_based_idx})]'


def array_subscript_fn(array: str, zero_based_idx: int):
  return f'({array})[{zero_based_idx + 1}]'


def element_at_index_fn(array: str, zero_based_idx: int):
  return f'element_at({array}, {zero_based_idx + 1})'


def json_extract_fn(array: str, zero_based_idx: int):
  return f"JSON_EXTRACT({array}, '$[{zero_based_idx}]')"


def json_value_fn(array: str, zero_based_idx: int):
  return f"JSON_VALUE({array}, '$[{zero_based_idx}]')"


def array_index_fn_not_implemented(array: str, zero_based_idx: int):
  del array, zero_based_idx  # Unused
  raise NotImplementedError('ARRAY_INDEX is not implemented.')


def nth_fn_default(
    zero_based_idx: int,
    sort_by: Optional[str],
    ascending: Optional[bool],
    dropna: Optional[bool],
    limit: Optional[int],
):
  try:
    array = ARRAY_AGG_FN(sort_by, ascending, dropna, limit)
    return ARRAY_INDEX_FN(array, zero_based_idx)
  except NotImplementedError as e:
    raise NotImplementedError('Nth value is not implemented.') from e


def uniform_mapping_fn_not_implemented(c):
  raise NotImplementedError('Uniform mapping is not implemented.')


def unnest_array_with_offset_fn(
    array: str,
    alias: Optional[str] = None,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
):
  """Unnests an array in GoogleSQL."""
  if alias is None:
    return f'UNNEST({array})'
  if not offset:
    return f'UNNEST({array}) AS {alias}'
  where = f' WHERE {offset} < {limit}' if limit else ''
  return f'UNNEST({array}) {alias} WITH OFFSET AS {offset}{where}'


def unnest_array_with_ordinality_fn(
    array: str,
    alias: Optional[str] = None,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
):
  """Unnests an array in PostgreSQL."""
  if alias is None:
    return f'UNNEST({array})'
  if not offset:
    return f'UNNEST({array}) unnested({alias})'
  where = f' WHERE {offset} < {limit + 1}' if limit else ''
  return (
      f'UNNEST({array}) WITH ORDINALITY AS unnested({alias}, {offset}){where}'
  )


def unnest_json_array_fn(
    array: str,
    alias: Optional[str] = None,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
):
  """Unnests a JSON_ARRAY in Oracle SQL."""
  where = f' WHERE {offset} < {limit + 1}' if limit else ''
  return f'''JSON_TABLE({array}, '$[*]'
        COLUMNS (
            {alias} FLOAT PATH '$',
            {offset} FOR ORDINALITY
        )
    ) AS foobar{where}'''


def unnest_array_fn_not_implemented(
    array: str,
    alias: Optional[str] = None,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
):
  del array, alias, offset, limit  # Unused
  raise NotImplementedError('UNNEST is not implemented.')


def unnest_array_literal_fn_googlesql(array: List[Any], alias: str = ''):
  return f'UNNEST({array}) {alias}'.strip()


def unnest_array_literal_fn_postgresql(array: List[Any], alias: str = ''):
  return f'UNNEST(ARRAY{array}) {alias}'.strip()


def unnest_array_literal_fn_not_implemented(array, alias=''):
  del array, alias  # Unused
  raise NotImplementedError('UNNEST with literal array is not implemented.')


def generate_array_fn(n):
  """Generates an array of n elements using GENERATE_ARRAY."""
  return f'GENERATE_ARRAY(1, {n})'


def generate_series_fn(n):
  """Generates an array of n elements using GENERATE_SERIES."""
  return f'GENERATE_SERIES(1, {n})'


def generate_sequence_fn_mariadb(n):
  """Generates an array of n elements using sequence in MariaDB."""
  try:
    n = int(n)
    if not 1 < n < 9223372036854775807:
      raise ValueError(
          'Only support generating sequence for an integer between 1 and'
          f' 2^63 - 1. Got: {n}'
      )
    return f'seq_0_to_{int(n) - 1}'
  except ValueError as e:
    raise NotImplementedError(
        f'Only support generating sequence for an integer. Got: {n}'
    ) from e


def generate_array_fn_oracle(n, alias: str = '_'):
  """Generates an array of n elements using sequence in Oracle."""
  try:
    return f'SELECT LEVEL AS {alias} FROM DUAL CONNECT BY LEVEL <= {int(n)}'
  except ValueError as e:
    raise NotImplementedError(
        f'Only support generating sequence for an integer. Got: {n}'
    ) from e


def generate_sequence_fn_trino(n):
  """Generates an array of n elements using sequence in Trino."""
  return f'SEQUENCE(1, {n})'


def generate_array_fn_not_implemented(n):
  del n  # Unused
  raise NotImplementedError(
      'GENERATE_ARRAY/GENERATE_SERIES is not implemented.'
  )


def unnest_generated_array(n, alias: Optional[str] = None):
  """Unnest a generated array, used to duplicate data."""
  return UNNEST_ARRAY_FN(GENERATE_ARRAY_FN(n), alias)


def implicitly_unnest_generated_array(n, alias: Optional[str] = None):
  """Unnest a generated series, used to duplicate data."""
  if not alias:
    return GENERATE_ARRAY_FN(n)
  return f'{GENERATE_ARRAY_FN(n)} {alias}'


def implicitly_unnest_generated_sequence(n, alias: Optional[str] = None):
  """Unnest a generated series, used to duplicate data."""
  if not alias:
    return GENERATE_ARRAY_FN(n)
  return f'(SELECT seq AS {alias} FROM {GENERATE_ARRAY_FN(n)}) unnested'


def duplicate_data_n_times_oracle(n, alias: Optional[str] = None):
  if not alias:
    return generate_array_fn_oracle(n)
  return generate_array_fn_oracle(n, alias)


def duplicate_data_n_times_not_implemented(n, alias: Optional[str] = None):
  del n, alias  # Unused
  raise NotImplementedError(
      'Duplicate data n times is not implemented.'
  )


CREATE_TEMP_TABLE_OPTIONS = {
    'Default': drop_table_if_exists_then_create_temp_table,
    'GoogleSQL': 'CREATE OR REPLACE TEMP TABLE {alias} AS {query};'.format,
    'MariaDB': drop_temp_table_if_exists_then_create_temp_table,
    'Oracle': create_temp_table_fn_not_implemented,
    'SQL Server': 'SELECT * INTO #{alias} FROM ({query});'.format,
}
ROW_NUMBER_REQUIRE_ORDER_BY_OPTIONS = {
    'Default': False,
    'Oracle': True,
    'SQL Server': True,
}
# In 'SELECT x + 1 AS foo, COUNT(*) FROM T GROUP BY ...', we can
# GROUP BY foo, GROUP BY 1, or GROUP BY x + 1. Most dialects support all three.
# But Oracle doesn't support GROUP BY 1 and SQL Server only supports
# GROUP BY x + 1. We prefer to use GROUP BY foo to GROUP BY 1 to GROUP BY x + 1
# from the readability perspective.
GROUP_BY_OPTIONS = {
    'Default': lambda columns: ', '.join(columns.aliases),
    'SQL Server': lambda columns: ', '.join(columns.expressions),
    'Trino': lambda columns: ', '.join(map(str, range(1, len(columns) + 1))),
}
SAFE_DIVIDE_OPTIONS = {
    'Default': safe_divide_fn_default,
    'GoogleSQL': 'SAFE_DIVIDE({numer}, {denom})'.format,
}
# When make changes, manually evaluate the run_only_once_in_with_clause and
# update the NEED_TEMP_TABLE_OPTIONS.
RAND_OPTIONS = {
    'Default': 'RANDOM()'.format,
    'GoogleSQL': 'RAND()'.format,
    'MariaDB': 'RAND()'.format,
    'Oracle': 'DBMS_RANDOM.VALUE'.format,
    'SQL Server': sql_server_rand_fn_not_implemented,
    'SQLite': '0.5 - RANDOM() / CAST(-9223372036854775808 AS REAL) / 2'.format,
}
# Manually evalueated run_only_once_in_with_clause for each dialect.
NEED_TEMP_TABLE_OPTIONS = {
    'Default': True,
    'PostgreSQL': False,
    'MariaDB': False,
    'Oracle': True,
    'SQL Server': True,
    'Trino': True,
    'SQLite': False,
}
CEIL_OPTIONS = {
    'Default': 'CEIL({})'.format,
    'SQL Server': 'CEILING({})'.format,
}
QUANTILE_OPTIONS = {
    'Default': quantile_fn_not_implemented,
    'GoogleSQL': approx_quantiles_fn,
    'PostgreSQL': percentile_cont_fn,
    'Oracle': percentile_cont_fn,
    'Trino': approx_percentile_fn,
}
ARRAY_AGG_OPTIONS = {
    'Default': array_agg_fn_not_implemented,
    'GoogleSQL': array_agg_fn_googlesql,
    'PostgreSQL': array_agg_fn_no_use_filter_no_limit,
    'MariaDB': json_array_agg_fn,
    'Oracle': json_array_agg_fn,
    # JSON_ARRAYAGG has been added in SQL Server 2025. Will update later.
    'SQL Server': array_agg_fn_not_implemented,
    'Trino': array_agg_fn_no_use_filter_no_limit,
}
ARRAY_INDEX_OPTIONS = {
    'Default': array_index_fn_not_implemented,
    'GoogleSQL': array_index_fn_googlesql,
    'PostgreSQL': array_subscript_fn,
    'MariaDB': json_extract_fn,
    'Oracle': json_value_fn,
    'Trino': element_at_index_fn,
}
NTH_OPTIONS = {
    'Default': nth_fn_default,
}
COUNTIF_OPTIONS = {
    'Default': 'COUNT(CASE WHEN {} THEN 1 END)'.format,
    'GoogleSQL': 'COUNTIF({})'.format,
}
FLOAT_CAST_OPTIONS = {
    'Default': 'CAST({} AS FLOAT)'.format,
    'Trino': 'CAST({} AS DOUBLE)'.format,
}
STRING_CAST_OPTIONS = {
    'Default': 'CAST({} AS TEXT)'.format,
    'GoogleSQL': 'CAST({} AS STRING)'.format,
    'MariaDB': 'CAST({} AS NCHAR)'.format,
    'Oracle': 'TO_CHAR({})'.format,
    'SQL Server': 'CAST({} AS VARCHAR(MAX))'.format,
    'Trino': 'CAST({} AS VARCHAR)'.format,
}
UNIFORM_MAPPING_OPTIONS = {
    'Default': uniform_mapping_fn_not_implemented,
    'GoogleSQL': lambda c: f'FARM_FINGERPRINT({c}) / 0xFFFFFFFFFFFFFFFF + 0.5',
    # These queries are verified in
    # https://colab.research.google.com/drive/1C1klaXsus0fWnOAT_vWzNHOT3Q21LZi7#scrollTo=O4--SViiuAv9&line=4&uniqifier=1.
    'PostgreSQL': lambda c: f'ABS(HASHTEXT({c})::BIGINT) / 2147483647.',
    'MariaDB': lambda c: (
        f'CAST(CONV(SUBSTRING(MD5({c}), 1, 16), 16, 10) AS DECIMAL(38, 0)) /'
        ' POW(2, 64)'
    ),
    'Trino': lambda c: (
        f'CAST(from_big_endian_64(xxhash64(CAST({c} AS varbinary))) AS DOUBLE)'
        ' / POWER(2, 64) + 0.5'
    ),
}
UNNEST_ARRAY_OPTIONS = {
    'Default': unnest_array_fn_not_implemented,
    'GoogleSQL': unnest_array_with_offset_fn,
    'PostgreSQL': unnest_array_with_ordinality_fn,
    'MariaDB': unnest_json_array_fn,
    'Oracle': unnest_json_array_fn,
    'Trino': unnest_array_with_ordinality_fn,
}
UNNEST_ARRAY_LITERAL_OPTIONS = {
    'Default': unnest_array_literal_fn_not_implemented,
    'GoogleSQL': unnest_array_literal_fn_googlesql,
    'PostgreSQL': unnest_array_literal_fn_postgresql,
    'Trino': unnest_array_literal_fn_postgresql,
}
GENERATE_ARRAY_OPTIONS = {
    'Default': generate_array_fn_not_implemented,
    'GoogleSQL': generate_array_fn,
    'PostgreSQL': generate_series_fn,
    'MariaDB': generate_sequence_fn_mariadb,
    'Oracle': generate_array_fn_oracle,
    'SQL Server': generate_series_fn,
    'Trino': generate_sequence_fn_trino,
}
DUPLICATE_DATA_N_TIMES_OPTIONS = {
    'Default': duplicate_data_n_times_not_implemented,
    'GoogleSQL': unnest_generated_array,
    'PostgreSQL': implicitly_unnest_generated_array,
    'MariaDB': implicitly_unnest_generated_sequence,
    'Oracle': duplicate_data_n_times_oracle,
    'SQL Server': implicitly_unnest_generated_array,
    'Trino': unnest_generated_array,
}


def set_dialect(dialect: str):
  """Sets the dialect of the SQL query."""
  # You can manually override the options below. You can manually test it in
  # https://colab.research.google.com/drive/1y3UigzEby1anMM3-vXocBx7V8LVblIAp?usp=sharing.
  global DIALECT, NEED_TEMP_TABLE, CREATE_TEMP_TABLE_FN, ROW_NUMBER_REQUIRE_ORDER_BY, GROUP_BY_FN, RAND_FN, CEIL_FN, SAFE_DIVIDE_FN, QUANTILE_FN, ARRAY_AGG_FN, ARRAY_INDEX_FN, NTH_VALUE_FN, COUNTIF_FN, STRING_CAST_FN, FLOAT_CAST_FN, UNIFORM_MAPPING_FN, UNNEST_ARRAY_FN, UNNEST_ARRAY_LITERAL_FN, GENERATE_ARRAY_FN, DUPLICATE_DATA_N_TIMES_FN
  DIALECT = dialect
  NEED_TEMP_TABLE = _get_dialect_option(NEED_TEMP_TABLE_OPTIONS)
  CREATE_TEMP_TABLE_FN = _get_dialect_option(CREATE_TEMP_TABLE_OPTIONS)
  ROW_NUMBER_REQUIRE_ORDER_BY = _get_dialect_option(
      ROW_NUMBER_REQUIRE_ORDER_BY_OPTIONS
  )
  GROUP_BY_FN = _get_dialect_option(GROUP_BY_OPTIONS)
  RAND_FN = _get_dialect_option(RAND_OPTIONS)
  CEIL_FN = _get_dialect_option(CEIL_OPTIONS)
  SAFE_DIVIDE_FN = _get_dialect_option(SAFE_DIVIDE_OPTIONS)
  QUANTILE_FN = _get_dialect_option(QUANTILE_OPTIONS)
  ARRAY_AGG_FN = _get_dialect_option(ARRAY_AGG_OPTIONS)
  ARRAY_INDEX_FN = _get_dialect_option(ARRAY_INDEX_OPTIONS)
  NTH_VALUE_FN = _get_dialect_option(NTH_OPTIONS)
  COUNTIF_FN = _get_dialect_option(COUNTIF_OPTIONS)
  STRING_CAST_FN = _get_dialect_option(STRING_CAST_OPTIONS)
  FLOAT_CAST_FN = _get_dialect_option(FLOAT_CAST_OPTIONS)
  UNIFORM_MAPPING_FN = _get_dialect_option(UNIFORM_MAPPING_OPTIONS)
  UNNEST_ARRAY_FN = _get_dialect_option(UNNEST_ARRAY_OPTIONS)
  UNNEST_ARRAY_LITERAL_FN = _get_dialect_option(UNNEST_ARRAY_LITERAL_OPTIONS)
  GENERATE_ARRAY_FN = _get_dialect_option(GENERATE_ARRAY_OPTIONS)
  DUPLICATE_DATA_N_TIMES_FN = _get_dialect_option(
      DUPLICATE_DATA_N_TIMES_OPTIONS
  )


def _get_dialect_option(options: dict[str, Any]):
  return options.get(DIALECT, options['Default'])


set_dialect(DEFAULT_DIALECT)


def is_compatible(sql0, sql1):
  """Checks if two datasources are compatible so their columns can be merged.

  Being compatible means datasources
  1. SELECT FROM the same data source
  2. have same GROUP BY clauses
  3. have the same WITH clause (usually None)
  4. do not SELECT DISTINCT.

  Args:
    sql0: A Sql instance.
    sql1: A Sql instance.

  Returns:
    If sql0 and sql1 are compatible.
  """
  if not isinstance(sql0, Sql) or not isinstance(sql1, Sql):
    raise ValueError('Both inputs must be a Sql instance!')
  return (
      sql0.from_data == sql1.from_data
      and sql0.where == sql1.where
      and sql0.groupby == sql1.groupby
      and sql0.with_data == sql1.with_data
      and not sql0.columns.distinct
      and not sql1.columns.distinct
  )


def add_suffix(alias):
  """Adds an int suffix to alias."""
  alias = alias.strip('`')
  m = re.search(r'([0-9]+)$', alias)
  if m:
    suffix = m.group(1)
    alias = alias[:-len(suffix)] + str(int(suffix) + 1)
    return alias
  else:
    return alias + '_1'


def rand_run_only_once_in_with_clause(execute):
  """Check if the RAND() is only evaluated once in the WITH clause."""
  d = execute(
      f'''WITH T AS (SELECT {RAND_FN()} AS r)
      SELECT t1.r - t2.r AS d
      FROM T t1 CROSS JOIN T t2'''
  )
  return bool(d.iloc[0, 0] == 0)


def dep_on_rand_table(query, rand_tables):
  """Returns if a SQL query depends on any stochastic table in rand_tables."""
  for rand_table in rand_tables:
    if re.search(r'\b%s\b' % rand_table, str(query)):
      return True
  return False


def get_temp_tables(with_data: 'Datasources'):
  """Gets all the subquery tables that need to be materialized.

  When generating the query, we assume that volatile functions like RAND() in
  the WITH clause behave as if they are evaluated only once. Unfortunately, not
  all engines behave like that. In those cases, we need to CREATE TEMP TABLE to
  materialize the subqueries that have volatile functions, so that the same
  result is used in all places. An example is
    WITH T AS (SELECT RAND() AS r)
    SELECT t1.r - t2.r AS d
    FROM T t1 CROSS JOIN T t2.
  If it doesn't always evaluates to 0, we need to create a temp table for T.
  A subquery needs to be materialized if
    1. it depends on any stochastic table
    (e.g. RAND()) and
    2. the random column is referenced in the same query multiple times.
  #2 is hard to check so we check if the stochastic table is referenced in the
  same query multiple times instead.
  An exception is the BootstrapRandomChoices table, which refers to a stochastic
  table twice but only one refers to the stochasic column, so we don't need to
  materialize it.
  This function finds all the subquery tables in the WITH clause that need to be
  materialized by
  1. finding all the stochastic tables,
  2. finding all the tables that depend, even indirectly, on a stochastic table,
  3. finding all the tables in #2 that are referenced in the same query multiple
    times.

  Args:
    with_data: The with clause.

  Returns:
    A set of table names that need to be materialized.
  """
  tmp_tables = set()
  for rand_table in with_data:
    query = with_data[rand_table]
    if RAND_FN() not in str(query):
      continue
    dep_on_rand = set([rand_table])
    for alias in with_data:
      if dep_on_rand_table(with_data[alias].from_data, dep_on_rand):
        dep_on_rand.add(alias)
    for t in dep_on_rand:
      from_data = with_data[t].from_data
      if isinstance(from_data, Join) and not t.startswith(
          'BootstrapRandomChoices'
      ):
        if dep_on_rand_table(from_data.ds1, dep_on_rand) and dep_on_rand_table(
            from_data.ds2, dep_on_rand
        ):
          tmp_tables.add(rand_table)
          break
  return tmp_tables


def get_alias(c):
  return getattr(c, 'alias_raw', c)


def escape_alias(alias):
  """Replaces special characters in SQL column name alias."""
  special = set(r""" `~!@#$%^&*()-=+[]{}\|;:'",.<>/?""")
  if not alias or not special.intersection(alias):
    return alias
  escape = {c: '_' for c in special}
  escape.update({
      '!': '_not_',
      '$': '_macro_',
      '@': '_at_',
      '%': '_pct_',
      '^': '_to_the_power_',
      '*': '_times_',
      ')': '',
      '-': '_minus_',
      '=': '_equals_',
      '+': '_plus_',
      '.': '_point_',
      '/': '_divides_',
      '>': '_greater_than_',
      '<': '_smaller_than_',
  })
  res = (
      ''.join(escape.get(c, c) for c in alias)
      .strip('_')
      .strip(' ')
      .replace('__', '_')
  )
  return 'col_' + res if res[0].isdigit() else res


@functools.total_ordering
class SqlComponent:
  """Base class for a SQL component like column, tabel and filter."""

  def __eq__(self, other):
    return str(self) == str(other)

  def __lt__(self, other):
    return str(self) < other

  def __repr__(self):
    return str(self)

  def __hash__(self):
    return hash(str(self))

  def __bool__(self):
    return bool(str(self))

  def __nonzero__(self):
    return bool(str(self))

  def __add__(self, other):
    return str.__add__(str(self), other)

  def __mul__(self, other):
    return str.__mul__(str(self), other)

  def __rmul__(self, other):
    return str.__rmul__(str(self), other)

  def __getitem__(self, idx):
    return str(self)[idx]


class SqlComponents(SqlComponent):
  """Base class for a bunch of SQL components like columns and filters."""

  def __init__(self, children=None):
    super(SqlComponents, self).__init__()
    self.children = []
    self.add(children)

  def add(self, children):
    if not isinstance(children, str) and isinstance(children, abc.Iterable):
      for c in list(children):
        self.add(c)
    else:
      if children and children not in self.children:
        self.children.append(children)
    return self

  def __iter__(self):
    for c in self.children:
      yield c

  def __len__(self):
    return len(self.children)

  def __getitem__(self, key):
    return self.children[key]

  def __setitem__(self, key, value):
    self.children[key] = value


class Filter(SqlComponent):
  """Represents single condition in SQL WHERE clause."""

  def __init__(self, cond: Optional[Text]):
    super(Filter, self).__init__()
    self.cond = ''
    if isinstance(cond, Filter):
      self.cond = cond.cond
    elif cond:
      self.cond = cond.replace('==', '=') or ''

  def __str__(self):
    if not self.cond:
      return ''
    return '(%s)' % self.cond if ' OR ' in self.cond.upper() else self.cond


class Filters(SqlComponents):
  """Represents a bunch of SQL conditions."""

  @property
  def where(self):
    return sorted((str(Filter(f)) for f in self.children))

  def remove(self, filters):
    if not filters:
      return self
    self.children = [c for c in self.children if c not in Filters(filters)]
    return self

  def __str__(self):
    return ' AND '.join(self.where)


class Column(SqlComponent):
  """Represents a SQL column.

  Generates a single row in the SELECT clause in SQL. Here are some examples of
  the input and representation.

  Input => Representation
  Column('click', 'SUM({})') => SUM(click) AS `sum(click)`
  Column('click * weight', 'SUM({})', 'foo') => SUM(click * weight) AS foo
  Column('click', 'SUM({})', auto_alias=False) => SUM(click)
  Column('click', 'SUM({})', filters='region = "US"') =>
    SUM(IF(region = "US", click, NULL)) AS `sum(click)`
  Column('region') => region  # No alias because it's same as the column.
  Column('* EXCEPT (click)', auto_alias=False) => * EXCEPT (click)
  Column(('click', 'impression'), 'SAFE_DIVIDE({}, {})', 'ctr') =>
    SAFE_DIVIDE(click, impression) AS ctr.
  Column(('click', 'impr'), 'SAFE_DIVIDE({}, {})', 'ctr', 'click > 5') =>
    SAFE_DIVIDE(IF(click > 5, click, NULL), IF(click > 5, impr, NULL)) AS ctr.
  Column('click', 'SUM({})', partition='region', auto_alias=False) =>
    SUM(click) OVER (PARTITION BY region)

  The representation is generated by applying the self.fn to self.column, then
  adding optional OVER clause and renaming. The advantange of using Column
  instead of raw string is
  1. It handles filters nicely.
  2. Even you don't need filters you can still pass the raw string, for exmaple,
    '* EXCEPT (click)', in and it'd equivalent to a string, but can be used
    with other Columns.
  3. It supports arithmetic operations.
    Column('click') * 2 is same as Column('click * 2') and
    Column('click')  Column('impression') is same as
    Column(('click', 'impression'), 'SAFE_DIVIDE({}, {})') except for the
    auto-generated aliases. This makes constructing complex SQL column easy.
  4. Alias will be sanitized and auto-added if necessary.
  """

  def __init__(
      self,
      column,
      fn: Text = '{}',
      alias: Optional[Text] = None,
      filters=None,
      partition=None,
      order=None,
      window_frame=None,
      auto_alias=True,
  ):
    super(Column, self).__init__()
    self.column = [column] if isinstance(column, str) else column or []
    self.fn = fn
    # For a single column, we apply the function to the column repeatedly.
    if len(self.column) == 1 and fn.count('{}') > 1:
      self.column *= fn.count('{}')
    self.filters = Filters(filters)
    self.alias_raw = alias.strip('`') if alias else None
    if not alias and auto_alias:
      self.alias_raw = fn.lower().format(*self.column)
    self.partition = partition
    self.order = order
    self.window_frame = window_frame
    self.auto_alias = auto_alias
    self.suffix = 0

  @property
  def alias(self):
    a = self.alias_raw
    if self.suffix:
      a = '%s_%s' % (a, self.suffix)
    return escape_alias(a)

  @alias.setter
  def alias(self, alias):
    self.alias_raw = alias.strip('`')

  def set_alias(self, alias):
    self.alias = alias
    return self

  def add_suffix(self):
    self.suffix += 1
    return self.alias

  @property
  def expression(self):
    """Genereates the representation without the 'AS ...' part."""
    over = None
    if not (self.partition is None and self.order is None and
            self.window_frame is None):
      partition_cols_str = [
          STRING_CAST_FN(c) for c in Columns(self.partition).expressions
      ]
      partition = 'PARTITION BY %s' % ', '.join(
          partition_cols_str) if self.partition else ''
      order = 'ORDER BY %s' % ', '.join(Columns(
          self.order).expressions) if self.order else ''
      frame = self.window_frame
      window_clause = ' '.join(c for c in (partition, order, frame) if c)
      over = ' OVER (%s)' % window_clause
    # Some Beam engines don't support SUM(IF(cond, var, NULL)) well so we use 0
    # as the base to make it work.
    base = '0' if self.fn.upper() == 'SUM({})' else 'NULL'
    # CASE WHEN has better compatibility with other engines than
    # IF(filter, c, NULL). For example, PostgreSQL only supports CASE WHEN.
    column = (
        f'CASE WHEN {self.filters} THEN {c} ELSE {base} END'
        if self.filters
        else c
        for c in self.column
    )
    res = self.fn.format(*column)
    return res + over if over else res

  def __str__(self):
    if not self.expression:
      return ''
    res = self.expression
    if (not self.alias_raw and not self.auto_alias) or res == self.alias:
      return res
    return '%s AS %s' % (res, self.alias)

  def __add__(self, other):
    return Column(
        '{} + {}'.format(*add_parenthesis_if_needed(self, other)),
        alias='%s + %s' % (self.alias_raw, get_alias(other)))

  def __radd__(self, other):
    alias = '%s + %s' % (get_alias(other), self.alias_raw)
    return Column(
        '{} + {}'.format(*add_parenthesis_if_needed(other, self)), alias=alias)

  def __sub__(self, other):
    return Column(
        '{} - {}'.format(*add_parenthesis_if_needed(self, other)),
        alias='%s - %s' % (self.alias_raw, get_alias(other)))

  def __rsub__(self, other):
    alias = '%s - %s' % (get_alias(other), self.alias_raw)
    return Column(
        '{} - {}'.format(*add_parenthesis_if_needed(other, self)), alias=alias)

  def __mul__(self, other):
    return Column(
        '{} * {}'.format(*add_parenthesis_if_needed(self, other)),
        alias='%s * %s' % (self.alias_raw, get_alias(other)))

  def __rmul__(self, other):
    alias = '%s * %s' % (get_alias(other), self.alias_raw)
    return Column(
        '{} * {}'.format(*add_parenthesis_if_needed(other, self)), alias=alias)

  def __neg__(self):
    return Column(
        '-{}'.format(*add_parenthesis_if_needed(self)),
        alias='-%s' % self.alias_raw)

  def __div__(self, other):
    return Column(
        SAFE_DIVIDE_FN(
            numer=self.expression, denom=getattr(other, 'expression', other)
        ),
        alias='%s / %s' % (self.alias_raw, get_alias(other)),
    )

  def __truediv__(self, other):
    return self.__div__(other)

  def __rdiv__(self, other):
    alias = '%s / %s' % (get_alias(other), self.alias_raw)
    return Column(
        SAFE_DIVIDE_FN(
            numer=getattr(other, 'expression', other), denom=self.expression
        ),
        alias=alias,
    )

  def __rtruediv__(self, other):
    return self.__rdiv__(other)

  def __pow__(self, other):
    if isinstance(other, float) and other == 0.5:
      return Column(
          'SAFE.SQRT({})'.format(self.expression),
          alias='sqrt(%s)' % self.alias_raw)
    return Column(
        'SAFE.POWER({}, {})'.format(self.expression,
                                    getattr(other, 'expression', other)),
        alias='%s ^ %s' % (self.alias_raw, get_alias(other)))

  def __rpow__(self, other):
    alias = '%s ^ %s' % (get_alias(other), self.alias_raw)
    return Column(
        'SAFE.POWER({}, {})'.format(
            getattr(other, 'expression', other), self.expression),
        alias=alias)


def add_parenthesis_if_needed(*columns):
  for column in columns:
    if not isinstance(column, Column):
      yield column
      continue
    expression = column.expression
    if '+' in expression or '-' in expression:
      yield '(%s)' % expression
      continue
    yield expression


class Columns(SqlComponents):
  """Represents a bunch of SQL columns."""

  def __init__(self, columns=None, distinct=None):  # pylint: disable=super-init-not-called
    super(Columns, self).__init__()
    self.add(columns)
    self.distinct = distinct
    if distinct is None and isinstance(columns, Columns):
      self.distinct = columns.distinct

  @property
  def aliases(self):
    return [c.alias for c in self]

  @property
  def original_columns(self):
    # Returns the original Column instances added.
    return [c.column[0] for c in self]

  def get_alias(self, expression):
    res = [c for c in self if c.expression == expression]
    if res:
      return res[0].alias_raw
    return None

  def get_column(self, alias):
    res = [c for c in self if c.alias == alias]
    if res:
      return res[0]
    return None

  def add(self, children):
    """Adds a Column if not existing.

    Renames it when necessary.

    If the Column already exists with the same alias. Do nothing.
    If neither the Column nor the alias exist. Add it.
    If the Column exists but with a different alias. Don't add the Column but
    set its alias to the existing one's.
    If the Column doesn't exists but the alias exists. Give the Column
    a new alias by adding a unique suffix. Then add it under the new alias.

    Args:
      children: A string or a Column or an iterable of them.

    Returns:
      self.
    """
    if not isinstance(children, str) and isinstance(children, abc.Iterable):
      for c in children:
        self.add(c)
      return self
    if not children:
      return self
    if isinstance(children, str):
      return self.add(Column(children))
    alias, expr = children.alias, children.expression
    found = self.get_alias(expr)
    if found:
      children.set_alias(found)
      return self
    if alias not in self.aliases:
      self.children.append(children)
      return self
    else:
      children.add_suffix()
      return self.add(children)

  def difference(self, columns):
    return Columns((c for c in self if c not in Columns(columns)))

  @property
  def expression(self):
    return list(map(str, self))

  @property
  def expressions(self):
    return [c.expression for c in self]

  def get_columns(self, break_line=False, indent=True):
    delimiter = ',\n' if break_line else ', '
    if indent:
      res = delimiter.join(('  %s' % e for e in self.expression))
      return '  DISTINCT\n' + res if self.distinct else res
    res = delimiter.join(self.expression)
    return 'DISTINCT ' + res if self.distinct else res

  def as_groupby(self):
    return GROUP_BY_FN(self)

  def __str__(self):
    return self.get_columns(True)


class Datasource(SqlComponent):
  """Represents a SQL datasource, could be a table name or a SQL query."""

  def __init__(self, table, alias=None):
    super(Datasource, self).__init__()
    self.table = table
    self.alias = alias
    if isinstance(table, Datasource):
      self.table = table.table
      self.alias = alias or table.alias
    self.alias = escape_alias(self.alias)
    self.is_table = (
        not str(self.table).strip().upper().startswith('SELECT')
        and 'WITH ' not in str(self.table).upper()
        and 'WITH\n' not in str(self.table).upper()
    )

  def get_expression(self, form='FROM'):
    """Gets the expression that can be used in a FROM or WITH clause."""
    if form.upper() not in ('FROM', 'WITH'):
      raise ValueError('Unrecognized form for datasource!')
    if form.upper() == 'WITH':
      if not self.alias:
        raise ValueError('Datasource in a WITH clause must have an alias!')
      if self.is_table:
        raise ValueError('Datasource in a WITH clause must be a SQL query!')
      return '%s AS (%s)' % (self.alias, self.table)
    else:
      return str(self)

  def join(self, other, on=None, using=None, join='', alias=None):
    return Join(self, other, on, using, join, alias)

  def __str__(self):
    table = self.table if self.is_table else '(%s)' % self.table
    # No "AS" between a table and its alias is supported by more dialects.
    return '%s %s' % (table, self.alias) if self.alias else str(table)


class Join(Datasource):
  """Represents a JOIN of two Datasources."""

  def __init__(self,
               datasource1,
               datasource2,
               on=None,
               using=None,
               join='',
               alias=None):
    if on and using:
      raise ValueError('A JOIN cannot have both ON and USING condition!')
    if join.upper() not in ('', 'INNER', 'FULL', 'FULL OUTER', 'LEFT',
                            'LEFT OUTER', 'RIGHT', 'RIGHT OUTER', 'CROSS'):
      raise ValueError('Unrecognized JOIN type!')
    if 'FULL' in join.upper() and DIALECT == 'MariaDB':
      raise NotImplementedError('FULL JOIN is not supported in MariaDB.')
    self.ds1 = Datasource(datasource1)
    self.ds2 = Datasource(datasource2)
    self.join_type = join.upper()
    self.on = Filters(on)
    self.using = Columns(using)
    super(Join, self).__init__(self, alias)

  def __str__(self):
    if self.ds1 == self.ds2:
      return str(self.ds1)
    join_type = self.join_type
    if not join_type and not self.on and not self.using:
      join_type = 'CROSS'  # Being explicit is compatible with more dialects.
    join = '%s JOIN' % join_type if join_type else 'JOIN'
    sql = '\n'.join(map(str, (self.ds1, join, self.ds2)))
    if self.on:
      return '%s\nON %s' % (sql, self.on)
    if self.using:
      return '%s\nUSING (%s)' % (sql, ', '.join(self.using.aliases))
    return sql


class Datasources(SqlComponents):
  """Represents a bunch of SQL datasources in a WITH clause."""

  def __init__(self, datasources=None):
    super(Datasources, self).__init__()
    self.children = collections.OrderedDict()
    self.temp_tables = set()
    self.add(datasources)

  @property
  def datasources(self):
    return (Datasource(v, k) for k, v in self.children.items())

  def merge(self, new_child: Union[Datasource, 'Datasources', 'Sql']):
    """Merges a datasource if possible.

    The difference between merge() and add() is that in add() we skip only when
    there is a table in self that is exactly same as the table, except for the
    alias, being added. In merge(), we try to find a table that is compatible,
    but not necessarily the same, and merge the columns in the new table to it.
    In one word, merge() returns more compact query.

    If the Datasource already exists with the same alias, do nothing.
    If a compatible Datasource already exists, don't add the new Datasource.
    Instead expand the compatible Datasource with new columns. Return the alias
    of the compatible Datasource.
    If no compatible Datasource nor the alias exist, add new Datasource.
    If a compatible Datasource doesn't exists but the alias exists, give the
    new Datasource a new alias by adding a unique suffix. Then add it under the
    new alias.
    Note that this function might have side effects. When the columns in
    new_child are merged with existing columns with different aliases, the
    columns' aliases will be set to the existing ones in-place.

    Args:
      new_child: A Datasource instance or an iterable of Datasource(s).

    Returns:
      The alias of the Datasource we eventually add.
    """
    if isinstance(new_child, Sql):
      new_child = Datasource(new_child)
    if isinstance(new_child, Datasources):
      for d in new_child.datasources:
        self.merge(d)
      return
    if not isinstance(new_child, Datasource):
      raise ValueError(
          '%s is a %s, not a Datasource! You can try .add() instead.' %
          (new_child, type(new_child)))
    alias, table = new_child.alias, new_child.table
    # If there is a compatible data, most likely it has the same alias.
    if alias in self.children:
      target = self.children[alias]
      if isinstance(target, Sql):
        merged = target.merge(table)
        if merged:
          return alias
    for a, t in self.children.items():
      if a == alias or not isinstance(t, Sql):
        continue
      merged = t.merge(table)
      if merged:
        return a
    while new_child.alias in self.children:
      new_child.alias = add_suffix(new_child.alias)
    self.children[new_child.alias] = table
    return new_child.alias

  def add(self, children: Union[Datasource, Iterable[Datasource]]):
    """Adds a datasource if not existing.

    Renames it when necessary.

    If the Datasource already exists with the same alias. Do nothing.
    If neither the Datasource nor the alias exist. Add it.
    If the Datasource exists but with a different alias. Don't add it, and
    change its alias to the one already exists in-place.
    If the Datasource doesn't exists but the alias exists. Give the Datasource
    a new alias by adding a unique suffix. Then add it under the new alias.

    Args:
      children: A Datasource instance or an iterable of Datasource(s).

    Returns:
      The alias of the Datasource we eventually add.
    """
    if isinstance(children, Datasources):
      for c in children.datasources:
        self.add(c)
      return
    if not isinstance(children, str) and isinstance(children, abc.Iterable):
      for c in children:
        self.add(c)
      return
    if not children:
      return
    if not isinstance(children, Datasource):
      raise ValueError('Not a Datasource!')
    alias, table = children.alias, children.table
    if alias not in self.children:
      if table not in self.children.values():
        self.children[alias] = table
        return alias
      children.alias = [k for k, v in self.children.items() if v == table][0]
      return children.alias
    else:
      if table == self.children[alias]:
        return alias
      children.alias = add_suffix(alias)
      return self.add(children)

  def add_temp_table(self, table: Union[str, 'Sql', Join, Datasource]):
    """Marks alias and all its data dependencies as temp tables."""
    if isinstance(table, str):
      self.temp_tables.add(table)
      if table in self.children:
        self.add_temp_table(self.children[table])
      return
    if isinstance(table, Join):
      self.add_temp_table(table.ds1)
      self.add_temp_table(table.ds2)
      return
    if isinstance(table, Datasource):
      return self.add_temp_table(table.table)
    if isinstance(table, Sql):
      return self.add_temp_table(table.from_data)
    return self

  def extend(self, other: 'Datasources'):
    """Merge other to self. Adjust the query if a new alias is needed."""
    datasources = list(other.datasources)
    while datasources:
      d = datasources.pop(0)
      original_alias = d.alias
      new_alias = self.add(d)
      if original_alias != new_alias:
        for d2 in datasources:
          if original_alias in str(d2):
            d2.table = re.sub(r'\b%s\b' % original_alias, new_alias,
                              str(d2.table))
    return self

  def __str__(self):
    temp_tables = []
    with_tables = []
    for d in self.datasources:
      if d.alias in self.temp_tables:
        cp = copy.copy(d)
        cp.alias = None
        temp_tables.append(CREATE_TEMP_TABLE_FN(alias=d.alias, query=str(cp)))
      else:
        with_tables.append(d.get_expression('WITH'))
    res = '\n'.join(temp_tables)
    if with_tables:
      res += '\nWITH\n' + ',\n'.join(with_tables)
    return res.strip()


class Sql(SqlComponent):
  """Represents a SQL query."""

  def __init__(
      self,
      columns,
      from_data: Union[str, 'Sql', Datasource],
      where=None,
      groupby=None,
      with_data=None,
      orderby=None,
  ):
    super(Sql, self).__init__()
    self.columns = Columns(columns)
    self.where = Filters(where)
    self.groupby = Columns(groupby)
    self.orderby = Columns(orderby)
    self.with_data = Datasources(with_data)
    if not isinstance(from_data, Datasource):
      from_data = Datasource(from_data)
    self.from_data = from_data
    from_data_table = from_data.table
    if isinstance(from_data_table, Sql) and not self.from_data.is_table:
      if from_data_table.with_data:
        with_data_to_merge = from_data_table.with_data
        from_data_table.with_data = None
        self.from_data = Datasource(
            with_data_to_merge.add(Datasource(from_data, 'NoNameTable'))
        )
        self.with_data.extend(with_data_to_merge)
      if not self.columns:
        # Consolidate outer and inner Sql if the outer Sql doesn't have columns.
        self.columns = from_data_table.columns
        self.where = Filters(from_data_table.where).add(self.where)
        self.groupby = from_data_table.groupby
        self.orderby = from_data_table.orderby
        self.from_data = from_data_table.from_data
      elif not from_data_table.columns:
        # Consolidate outer and inner Sql if the inner Sql doesn't have columns.
        self.where = Filters(from_data_table.where).add(self.where)
        self.from_data = from_data_table.from_data

  @property
  def all_columns(self):
    return Columns(self.groupby).add(self.columns)

  def add(self, attr, values):
    getattr(self, attr).add(values)
    return self

  def merge(self, other: 'Sql'):
    """Merges columns from other to self if possible.

    If self and other are compatible, we can merge their columns. The columns
    from other that have conflicting names will be renamed in-place.

    Args:
      other: Another Sql instance.

    Returns:
      If two Sql instances are mergeable and a dict to look up new column names.
    """
    if not isinstance(other, Sql):
      raise ValueError('Expected a Sql instance but got %s!' % type(other))
    if self == other:
      return True
    if not is_compatible(self, other):
      return False
    self.columns.add(other.columns)
    return True

  def __str__(self):
    with_clause = str(self.with_data) if self.with_data else None
    all_columns = self.all_columns or '*'
    select_clause = f'SELECT\n{all_columns}'
    from_clause = ('FROM %s'
                   if self.from_data.is_table else 'FROM\n%s') % self.from_data
    where_clause = 'WHERE\n%s' % self.where if self.where else None
    if self.columns.distinct:
      if self.groupby:
        raise ValueError('GROUP BY cannot exist with DISTINCT')
      groupby_clause = 'GROUP BY %s' % self.columns.as_groupby()
    else:
      groupby_clause = (
          'GROUP BY %s' % self.groupby.as_groupby() if self.groupby else None
      )
    orderby_clause = 'ORDER BY %s' % self.orderby.as_groupby(
    ) if self.orderby else None
    clauses = [
        c for c in (with_clause, select_clause, from_clause, where_clause,
                    groupby_clause, orderby_clause) if c is not None
    ]
    return '\n'.join(clauses)
