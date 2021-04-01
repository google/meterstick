# Copyright 2020 Google LLC
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
from typing import Iterable, Optional, Text, Union


def is_compatible(sql0, sql1):
  """Checks if two datasources are compatible so their columns can be merged.

  Being compatible means datasources have same FROM and GROUP BY clauses. The
  FROM clause is more complex. If a FROM clause is a substing of another one,
  which means the latter is a JOIN of the former and some other datasources, we
  still consider them compatible. You might wonder why because the JOIN add or
  drop rows from the original data. The reason is all the JOINs we generate are
  1. CROSS JOIN with a single value, which is just adding a constant column.
  2. INNER JOIN The JOIN is done on all GROUP BY columns on both tables and they
    both have all slices, even with NULLs, so again the effect is same as adding
    columns.
  3. The LEFT JOIN. The left side is the original table. This might add rows
    with NULL values as some children SQL might miss some slices. But all the
    SimpleMetrics ignore NULL so it won't affect the result.
  As the result, as long as one datasource contains another one, they are
  considered compatible. If in the future we have a rule that generates JOIN
  that isn't compatible with original data, we need to change this function.

  Args:
    sql0: A Sql instance.
    sql1: A Sql instance.

  Returns:
    If sql0 and sql1 are compatible.
    The larger FROM clause if compatible.
  """
  if not isinstance(sql0, Sql) or not isinstance(sql1, Sql):
    raise ValueError('Both inputs must be a Sql instance!')
  if sql0.where != sql1.where or sql0.groupby != sql1.groupby:
    return False, None
  if sql0.from_data == sql1.from_data:
    return True, sql1.from_data
  # Exclude cases where two differ on suffix.
  if (str(sql0.from_data) + '\n' in str(sql1.from_data) or
      str(sql0.from_data) + ' ' in str(sql1.from_data)):
    return True, sql1.from_data
  if (str(sql1.from_data) + '\n' in str(sql0.from_data) or
      str(sql1.from_data) + ' ' in str(sql0.from_data)):
    return True, sql0.from_data
  return False, None


def add_suffix(alias):
  """Adds an int suffix to alias."""
  m = re.search(r'([0-9]+)$', alias)
  if m:
    suffix = m.group(1)
    alias = alias[:-len(suffix)] + str(int(suffix) + 1)
    return alias
  else:
    return alias + '_1'


def get_alias(c):
  return getattr(c, 'alias_raw', c)


def escape_alias(alias):
  # Macro still gets parsed inside backquotes.
  if alias and '$' in alias:
    alias = alias.replace('$', 'macro_')
  # Don't escape if alias is already escaped.
  if alias and set(r""" `~!@#$%^&*()-=+[]{}\|;:'",.<>/?""").intersection(
      alias) and not (alias.startswith('`') and alias.endswith('`')):
    return '`%s`' % alias.replace('\\', '\\\\')
  return alias


@functools.total_ordering
class SqlComponent():
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
      for c in children:
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

  def __init__(self,
               column=None,
               fn: Text = '{}',
               alias: Optional[Text] = None,
               filters=None,
               partition=None,
               order=None,
               window_frame=None,
               auto_alias=True):
    super(Column, self).__init__()
    self.column = [column] if isinstance(column, str) else column or []
    self.fn = fn
    self.filters = Filters(filters)
    self.alias_raw = alias
    if not alias and auto_alias:
      self.alias_raw = fn.lower().format(*self.column)
    self.partition = partition
    self.order = order
    self.window_frame = window_frame
    self.auto_alias = auto_alias

  @property
  def alias(self):
    return escape_alias(self.alias_raw)

  @alias.setter
  def alias(self, alias):
    self.alias_raw = alias

  def set_alias(self, alias):
    self.alias = alias
    return self

  @property
  def expression(self):
    """Genereates the representation without the 'AS ...' part."""
    over = None
    if not (self.partition is None and self.order is None and
            self.window_frame is None):
      partition = 'PARTITION BY %s' % ', '.join(
          Columns(self.partition).expressions) if self.partition else ''
      order = 'ORDER BY %s' % ', '.join(Columns(
          self.order).expressions) if self.order else ''
      frame = self.window_frame
      window_clause = ' '.join(c for c in (partition, order, frame) if c)
      over = ' OVER (%s)' % window_clause
    column = ('IF(%s, %s, NULL)' % (self.filters, c) if self.filters else c
              for c in self.column)
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
        'SAFE_DIVIDE({}, {})'.format(self.expression,
                                     getattr(other, 'expression', other)),
        alias='%s / %s' % (self.alias_raw, get_alias(other)))

  def __truediv__(self, other):
    return self.__div__(other)

  def __rdiv__(self, other):
    alias = '%s / %s' % (get_alias(other), self.alias_raw)
    return Column(
        'SAFE_DIVIDE({}, {})'.format(
            getattr(other, 'expression', other), self.expression),
        alias=alias)

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
    self.columns = collections.OrderedDict()
    self.add(columns)
    self.distinct = distinct
    if distinct is None and isinstance(columns, Columns):
      self.distinct = columns.distinct

  @property
  def children(self):
    return tuple(self.columns.values())

  @property
  def aliases(self):
    return [escape_alias(c.alias) for c in self]

  def add(self, children):
    """Adds a Column if not existing.

    Renames it when necessary.

    If the Column already exists with the same alias. Do nothing.
    If neither the Column nor the alias exist. Add it.
    If the Column exists but with a different alias. Add it.
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
    alias = children.alias_raw
    if alias not in self.columns:
      self.columns[alias] = children
      return self
    else:
      if children == self.columns[alias]:
        return self
      children.alias = add_suffix(alias)
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
    return ', '.join(self.aliases)

  def __str__(self):
    return self.get_columns(True)


class Datasource(SqlComponent):
  """Represents a SQL datasource, could be a table name or a SQL query."""

  def __init__(self, table, alias=None):
    super(Datasource, self).__init__()
    if isinstance(table, Datasource):
      self.table = table.table
      self.alias = table.alias
    else:
      self.table = table
      self.alias = alias
    self.alias = escape_alias(self.alias)
    self.is_table = not str(self.table).startswith('SELECT')

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
    return '%s AS %s' % (table, self.alias) if self.alias else str(table)


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
    self.ds1 = Datasource(datasource1)
    self.ds2 = Datasource(datasource2)
    self.join_type = join.upper()
    self.on = Filters(on)
    self.using = Columns(using)
    super(Join, self).__init__(str(self), alias)

  def __str__(self):
    join = '%s JOIN' % self.join_type if self.join_type else 'JOIN'
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
    self.add(datasources)

  @property
  def datasources(self):
    return (Datasource(v, k) for k, v in self.children.items())

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

  def merge(self, other: 'Datasources'):
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
    return ',\n'.join((d.get_expression('WITH') for d in self.datasources if d))


class Sql(SqlComponent):
  """Represents a SQL query."""

  def __init__(self,
               columns,
               from_data,
               where=None,
               groupby=None,
               with_data=None):
    super(Sql, self).__init__()
    self.columns = Columns(columns)
    self.where = Filters(where)
    self.groupby = Columns(groupby)
    self.with_data = Datasources(with_data)
    if isinstance(from_data, Sql) and from_data.with_data:
      from_data = copy.deepcopy(from_data)
      with_data_to_merge = from_data.with_data
      from_data.with_data = None
      from_data = with_data_to_merge.add(Datasource(from_data, 'NoNameTable'))
      self.with_data.merge(with_data_to_merge)
    self.from_data = Datasource(from_data)

  def add(self, attr, values):
    getattr(self, attr).add(values)
    return self

  def __str__(self):
    with_clause = 'WITH\n%s' % self.with_data if self.with_data else None
    tmpl = 'SELECT DISTINCT\n%s' if self.columns.distinct else 'SELECT\n%s'
    select_clause = tmpl % Columns(self.groupby).add(self.columns)
    from_clause = ('FROM %s'
                   if self.from_data.is_table else 'FROM\n%s') % self.from_data
    where_clause = 'WHERE\n%s' % self.where if self.where else None
    groupby_clause = 'GROUP BY %s' % self.groupby.as_groupby(
    ) if self.groupby else None
    clauses = [
        c for c in (with_clause, select_clause, from_clause, where_clause,
                    groupby_clause) if c is not None
    ]
    return '\n'.join(clauses)
