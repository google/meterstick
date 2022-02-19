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


def is_compatible(sql0, sql1, exact_from=False):
  """Checks if two datasources are compatible so their columns can be merged.

  Being compatible means datasources have same GROUP BY clauses and compatible
  FROM clause. Compatible means a FROM clause, though may be not exactly same as
  the other, doesn't change the results. For example, a table selecting from t1
  could have the same result if it was selecting from t1 LEFT JOIN t2 as long as
  all values added to t1 are NULLs and our computation ignores NULLs.

  We have several kinds of JOINs in the codes.
  1. CROSS JOIN with a single value, which is just adding a constant column, so
    should be considered compatible.
  2. FULL JOIN. This only happens once in MetricList and it's on all indexing
    columns. Both tables got expanded but the added values are NULLs and
    are effectively dropped in the computation so the result doesn't change. So
    this JOIN should be considered compatible.
  3. LEFT JOIN. If the left side is the main table (both sql queries select FROM
    it) and the join is done on all indexing columns, then the join expand the
    left table with NULLs and don't change the result. There is only one LEFT
    JOIN which isn't such kind. It's in Bootstrap where we have
    BootstrapRandomChoices LEFT JOIN original_table.
  4. (INNER) JOIN. This happens in Comparison. The right table is a subset of
    the left so it's equivalent to a LEFT JOIN and we don't use the right table
    alone. So if one query uses t1 JOIN t2 while the other uses t1 it's OK to
    merge them. Again Bootstrap is an exception, it has JOINs with UNNEST(ARRAY)
    which replicates the table and will affect the results.
  As the result, even two queries select from different sources, as long as one
  contains the other and it's not about Bootstrap, we can still merge them.
  IMPORTANT: We need to check the logic of every new Metics added in the future.

  Args:
    sql0: A Sql instance.
    sql1: A Sql instance.
    exact_from: If the FROM clauses need to be exactly the same to be considered
      comaptible.

  Returns:
    If sql0 and sql1 are compatible.
    The larger FROM clause if compatible.
  """
  if not isinstance(sql0, Sql) or not isinstance(sql1, Sql):
    raise ValueError('Both inputs must be a Sql instance!')
  if (sql0.where != sql1.where or sql0.groupby != sql1.groupby or
      sql0.with_data != sql1.with_data or sql0.columns.distinct or
      sql1.columns.distinct):
    return False, None
  if sql0.from_data == sql1.from_data:
    return True, sql1.from_data
  if exact_from:
    return False, None
  mergeable = False
  # Exclude cases where two differ on suffix.
  if (str(sql0.from_data) + '\n' in str(sql1.from_data) or
      str(sql0.from_data) + ' ' in str(sql1.from_data)):
    mergeable, larger = True, sql1.from_data
  if (str(sql1.from_data) + '\n' in str(sql0.from_data) or
      str(sql1.from_data) + ' ' in str(sql0.from_data)):
    mergeable, larger = True, sql0.from_data
  if mergeable:
    if 'UNNEST' in str(larger) or 'BootstrapRandomChoices' in str(larger):
      return False, None
    return mergeable, larger
  return False, None


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
        self.children.append(copy.deepcopy(children))
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
    super(Columns, self).__init__()
    self.add(columns)
    self.distinct = distinct
    if distinct is None and isinstance(columns, Columns):
      self.distinct = columns.distinct

  @property
  def aliases(self):
    return [c.alias for c in self]

  def get_alias(self, expression):
    res = [c for c in self if c.expression == expression]
    if res:
      return res[0].alias
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
    return ', '.join(self.aliases)

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
    self.is_table = not str(self.table).strip().upper().startswith('SELECT')

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
    super(Join, self).__init__(self, alias)

  def __str__(self):
    if self.ds1 == self.ds2:
      return str(self.ds1)
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

  def merge(self, new_child: Datasource):
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
    if not isinstance(new_child, Datasource):
      raise ValueError(
          '%s is a %s, not a Datasource! You can try .add() instead.' %
          (new_child, type(new_child)))
    alias, table = new_child.alias, new_child.table
    # If there is a compatible data, most likely it has the same alias.
    if alias in self.children:
      target = self.children[alias]
      if isinstance(target, Sql):
        merged, rename = target.merge(table)
        if merged:
          return alias, rename
    for a, t in self.children.items():
      if a == alias or not isinstance(t, Sql):
        continue
      merged, rename = t.merge(table)
      if merged:
        return a, rename
    while new_child.alias in self.children:
      new_child.alias = add_suffix(new_child.alias)
    self.children[new_child.alias] = table
    return new_child.alias, {}

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
    return ',\n'.join((d.get_expression('WITH') for d in self.datasources if d))


class Sql(SqlComponent):
  """Represents a SQL query."""

  def __init__(self,
               columns,
               from_data,
               where=None,
               groupby=None,
               with_data=None,
               orderby=None):
    super(Sql, self).__init__()
    self.columns = Columns(columns)
    self.where = Filters(where)
    self.groupby = Columns(groupby)
    self.orderby = Columns(orderby)
    self.with_data = Datasources(with_data)
    if isinstance(from_data, Sql) and from_data.with_data:
      from_data = copy.deepcopy(from_data)
      with_data_to_merge = from_data.with_data
      from_data.with_data = None
      from_data = with_data_to_merge.add(Datasource(from_data, 'NoNameTable'))
      self.with_data.extend(with_data_to_merge)
    if not isinstance(from_data, Datasource):
      from_data = Datasource(from_data)
    self.from_data = from_data

  def add(self, attr, values):
    getattr(self, attr).add(values)
    return self

  def merge(self, other: 'Sql', expand_from=False):
    """Merges columns from other to self if possible.

    If self and other are compatible, we can merge their columns. The columns
    from other that have conflicting names will be renamed in-place.

    Args:
      other: Another Sql instance.
      expand_from: Determines if to merge when self and other are compatible but
        other has a larger FROM clause.

    Returns:
      If two Sql instances are mergeable and a dict to look up new column names.
    """
    if not isinstance(other, Sql):
      raise ValueError('Expected a Sql instance but got %s!' % type(other))
    if self == other:
      return True, {}
    compatible, larger = is_compatible(self, other, expand_from)
    if not compatible:
      return False, {}
    if self.from_data != larger and not expand_from:
      return False, {}
    self.from_data = copy.deepcopy(larger)
    curr_aliases = other.columns.aliases
    self.columns.add(other.columns)
    return True, dict(
        [i for i in zip(curr_aliases, other.columns.aliases) if i[0] != i[1]])

  def __str__(self):
    with_clause = 'WITH\n%s' % self.with_data if self.with_data else None
    tmpl = 'SELECT DISTINCT\n%s' if self.columns.distinct else 'SELECT\n%s'
    select_clause = tmpl % Columns(self.groupby).add(self.columns)
    from_clause = ('FROM %s'
                   if self.from_data.is_table else 'FROM\n%s') % self.from_data
    where_clause = 'WHERE\n%s' % self.where if self.where else None
    groupby_clause = 'GROUP BY %s' % self.groupby.as_groupby(
    ) if self.groupby else None
    orderby_clause = 'ORDER BY %s' % self.orderby.as_groupby(
    ) if self.orderby else None
    clauses = [
        c for c in (with_clause, select_clause, from_clause, where_clause,
                    groupby_clause, orderby_clause) if c is not None
    ]
    return '\n'.join(clauses)
