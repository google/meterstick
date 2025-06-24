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
"""SQL Dialect system for cross-platform SQL generation."""

from abc import ABC, abstractmethod

from typing import Optional, Dict, Union, List


class SqlDialect(ABC):
    """Abstract base class for SQL dialect implementations.
    
    This class defines the interface that all SQL dialects must implement
    to support cross-platform SQL generation in meterstick.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this SQL dialect (e.g., 'bigquery', 'trino')."""
        pass
    
    # Division operations
    @abstractmethod
    def safe_divide(self, numerator: str, denominator: str) -> str:
        """Generate safe division SQL that handles division by zero.
        
        Args:
            numerator: SQL expression for the numerator
            denominator: SQL expression for the denominator
            
        Returns:
            SQL expression that safely divides numerator by denominator,
            returning NULL when denominator is zero.
        """
        pass
    
    # Quantile functions
    @abstractmethod
    def approx_quantile(self, column: str, quantile: float) -> str:
        """Generate approximate quantile calculation SQL.
        
        Args:
            column: Column expression to calculate quantile for
            quantile: Quantile value between 0 and 1
            
        Returns:
            SQL expression that calculates the approximate quantile.
        """
        pass
    
    # Array operations
    @abstractmethod
    def array_agg_with_limit(self, column: str, order_by: str, limit: int, offset: int) -> str:
        """Generate array aggregation with limit and offset.
        
        Args:
            column: Column to aggregate
            order_by: ORDER BY clause expression
            limit: Maximum number of elements to include
            offset: Zero-based offset for element selection
            
        Returns:
            SQL expression for array aggregation with indexing.
        """
        pass
    
    @abstractmethod
    def array_agg_ignore_nulls(self, column: str, order_by: str) -> str:
        """Generate array aggregation ignoring NULL values.
        
        Args:
            column: Column to aggregate
            order_by: ORDER BY clause expression
            
        Returns:
            SQL expression for array aggregation without NULLs.
        """
        pass
    
    # Conditional counting
    @abstractmethod
    def count_if(self, condition: str) -> str:
        """Generate conditional counting SQL.
        
        Args:
            condition: Boolean condition expression
            
        Returns:
            SQL expression that counts rows where condition is true.
        """
        pass
    
    # Data type casting
    @abstractmethod
    def cast_to_string(self, column: str) -> str:
        """Cast column to string/text type.
        
        Args:
            column: Column expression to cast
            
        Returns:
            SQL expression that casts column to string type.
        """
        pass
    
    # Mathematical functions
    @abstractmethod
    def safe_power(self, base: str, exponent: str) -> str:
        """Generate safe power function SQL.
        
        Args:
            base: Base expression
            exponent: Exponent expression
            
        Returns:
            SQL expression for power calculation.
        """
        pass
    
    @abstractmethod
    def safe_sqrt(self, value: str) -> str:
        """Generate safe square root function SQL.
        
        Args:
            value: Value expression
            
        Returns:
            SQL expression for square root calculation.
        """
        pass
    
    # Array generation and unnest
    @abstractmethod
    def generate_array(self, start: int, end: int) -> str:
        """Generate array sequence from start to end.
        
        Args:
            start: Start value (inclusive)
            end: End value (inclusive)
            
        Returns:
            SQL expression that generates an array sequence.
        """
        pass
    
    @abstractmethod
    def unnest_with_alias(self, array_expr: str, alias: str) -> str:
        """Generate UNNEST with proper alias syntax.
        
        Args:
            array_expr: Array expression to unnest
            alias: Alias for the unnested values
            
        Returns:
            SQL expression for UNNEST with alias.
        """
        pass
    
    @abstractmethod
    def nth_value_with_ignore_nulls(self, column: str, sort_by: str, n: int, ascending: bool = True) -> str:
        """Generate nth value with ignore nulls SQL.
        
        Args:
            column: Column expression to get nth value from
            sort_by: Column to sort by
            n: Zero-based index (0 = first, 1 = second, etc.)
            ascending: Whether to sort in ascending order
            
        Returns:
            SQL expression that gets the nth value ignoring nulls.
        """
        pass

    # PoissonBootstrap-specific operations
    @abstractmethod
    def poisson_bootstrap_uniform_hash(self, columns: List[str]) -> str:
        """Generate uniform random variable for PoissonBootstrap using consistent hashing.
        
        Args:
            columns: List of column expressions to include in hash
            
        Returns:
            SQL expression that generates a uniform random variable [0,1) 
            using consistent hashing of the columns.
        """
        pass
    
    @abstractmethod
    def select_all_except(self, excluded_columns: List[str]) -> str:
        """Generate SQL to select all columns except specified ones.
        
        Args:
            excluded_columns: List of column names to exclude
            
        Returns:
            SQL expression that selects all columns except the excluded ones.
        """
        pass


class BigQueryDialect(SqlDialect):
    """BigQuery SQL dialect implementation.
    
    This maintains the original BigQuery-specific SQL generation behavior
    to ensure backwards compatibility.
    """
    
    @property
    def name(self) -> str:
        return "bigquery"
    
    def safe_divide(self, numerator: str, denominator: str) -> str:
        return f'IF(({denominator}) = 0, NULL, ({numerator}) / ({denominator}))'
    
    def approx_quantile(self, column: str, quantile: float) -> str:
        offset = int(100 * quantile)
        return f'APPROX_QUANTILES({column}, 100)[OFFSET({offset})]'
    
    def array_agg_with_limit(self, column: str, order_by: str, limit: int, offset: int) -> str:
        return f'ARRAY_AGG({column} ORDER BY {order_by} LIMIT {limit})[SAFE_OFFSET({offset})]'
    
    def array_agg_ignore_nulls(self, column: str, order_by: str) -> str:
        return f'ARRAY_AGG({column} IGNORE NULLS ORDER BY {order_by})'
    
    def count_if(self, condition: str) -> str:
        return f'COUNTIF({condition})'
    
    def cast_to_string(self, column: str) -> str:
        return f'CAST({column} AS STRING)'
    
    def safe_power(self, base: str, exponent: str) -> str:
        return f'SAFE.POWER({base}, {exponent})'
    
    def safe_sqrt(self, value: str) -> str:
        return f'SAFE.SQRT({value})'
    
    def generate_array(self, start: int, end: int) -> str:
        return f'GENERATE_ARRAY({start}, {end})'
    
    def unnest_with_alias(self, array_expr: str, alias: str) -> str:
        return f'UNNEST({array_expr}) AS {alias}'
    
    def nth_value_with_ignore_nulls(self, column: str, sort_by: str, n: int, ascending: bool = True) -> str:
        order = '' if ascending else ' DESC'
        # For dropna case: ARRAY_AGG(column IGNORE NULLS ORDER BY sort_by)[SAFE_OFFSET(n)]
        return f'ARRAY_AGG({column} IGNORE NULLS ORDER BY {sort_by}{order})[SAFE_OFFSET({n})]'

    def poisson_bootstrap_uniform_hash(self, columns: List[str]) -> str:
        """Generate FARM_FINGERPRINT-based uniform random variable for BigQuery."""
        cols_str = ', '.join(columns)
        return f'FARM_FINGERPRINT(CONCAT({cols_str})) / 0xFFFFFFFFFFFFFFFF + 0.5'
    
    def select_all_except(self, excluded_columns: List[str]) -> str:
        """Generate BigQuery EXCEPT syntax."""
        excluded_str = ', '.join(excluded_columns)
        return f'* EXCEPT({excluded_str})'


class TrinoDialect(SqlDialect):
    """Trino SQL dialect implementation.
    
    This implements Trino-specific SQL syntax to provide compatibility
    with Trino/Presto SQL engines.
    """
    
    @property
    def name(self) -> str:
        return "trino"
    
    def safe_divide(self, numerator: str, denominator: str) -> str:
        return (f'CASE WHEN ({denominator}) = 0 THEN NULL '
                f'ELSE CAST(({numerator}) AS DOUBLE) / CAST(({denominator}) AS DOUBLE) END')
    
    def approx_quantile(self, column: str, quantile: float) -> str:
        return f'approx_percentile({column}, {quantile})'
    
    def array_agg_with_limit(self, column: str, order_by: str, limit: int, offset: int) -> str:
        # Trino uses 1-based indexing for arrays
        trino_index = offset + 1
        return f'element_at(array_agg({column} ORDER BY {order_by}), {trino_index})'
    
    def array_agg_ignore_nulls(self, column: str, order_by: str) -> str:
        return f'array_agg({column} ORDER BY {order_by}) FILTER (WHERE {column} IS NOT NULL)'
    
    def count_if(self, condition: str) -> str:
        return f'COUNT(CASE WHEN {condition} THEN 1 END)'
    
    def cast_to_string(self, column: str) -> str:
        return f'CAST({column} AS VARCHAR)'
    
    def safe_power(self, base: str, exponent: str) -> str:
        # Trino doesn't have SAFE functions, but POWER handles most cases gracefully
        return f'POWER({base}, {exponent})'
    
    def safe_sqrt(self, value: str) -> str:
        # Use CASE to handle negative values gracefully
        return f'CASE WHEN ({value}) >= 0 THEN SQRT({value}) ELSE NULL END'
    
    def generate_array(self, start: int, end: int) -> str:
        return f'sequence({start}, {end})'
    
    def unnest_with_alias(self, array_expr: str, alias: str) -> str:
        # Trino requires table alias format for UNNEST
        return f'UNNEST({array_expr}) AS t({alias})'
    
    def nth_value_with_ignore_nulls(self, column: str, sort_by: str, n: int, ascending: bool = True) -> str:
        order = '' if ascending else ' DESC'
        # For dropna case: array_agg(column ORDER BY sort_by) FILTER (WHERE column IS NOT NULL) then element_at(..., n+1)
        array_agg = f'array_agg({column} ORDER BY {sort_by}{order}) FILTER (WHERE {column} IS NOT NULL)'
        return f'element_at({array_agg}, {n + 1})'

    def poisson_bootstrap_uniform_hash(self, columns: List[str]) -> str:
        """PoissonBootstrap with unit grouping is not supported in Trino."""
        raise NotImplementedError(
            'PoissonBootstrap with unit grouping requires FARM_FINGERPRINT function '
            'which is not supported in Trino.'
        )
    
    def select_all_except(self, excluded_columns: List[str]) -> str:
        """PoissonBootstrap without pre-aggregation is not supported in Trino."""
        raise NotImplementedError(
            'PoissonBootstrap without pre-aggregation requires "* EXCEPT(...)" syntax '
            'which is not supported in Trino.'
        )


# Global dialect configuration
_available_dialects = {
    'bigquery': BigQueryDialect(),
    'trino': TrinoDialect()
}
_current_dialect = _available_dialects['bigquery']  # BigQuery as default


# Public API functions
def set_sql_dialect(dialect_name: str):
    """Set the global SQL dialect.
    
    Args:
        dialect_name: Name of the dialect to set ('bigquery', 'trino', etc.)
        
    Raises:
        ValueError: If the dialect name is not available
        
    Example:
        import meterstick as ms
        ms.set_sql_dialect('trino')
    """
    global _current_dialect
    if dialect_name not in _available_dialects:
        available = list(_available_dialects.keys())
        raise ValueError(
            f"Unknown dialect: '{dialect_name}'. "
            f"Available dialects: {available}"
        )
    _current_dialect = _available_dialects[dialect_name]


def get_sql_dialect() -> SqlDialect:
    """Get the current SQL dialect.
    
    Returns:
        The currently active SqlDialect instance
        
    Example:
        dialect = ms.get_sql_dialect()
        sql = dialect.safe_divide('clicks', 'impressions')
    """
    return _current_dialect


def get_available_sql_dialects() -> List[str]:
    """Get list of available SQL dialect names.
    
    Returns:
        List of available dialect names
    """
    return list(_available_dialects.keys())


def register_sql_dialect(dialect: SqlDialect):
    """Register a custom SQL dialect.
    
    Args:
        dialect: Custom SqlDialect instance to register
        
    Raises:
        ValueError: If a dialect with the same name is already registered
        
    Example:
        class MyCustomDialect(ms.SqlDialect):
            # ... implementation
            pass
            
        ms.register_sql_dialect(MyCustomDialect())
    """
    if dialect.name in _available_dialects:
        raise ValueError(f"Dialect '{dialect.name}' is already registered")
    _available_dialects[dialect.name] = dialect 