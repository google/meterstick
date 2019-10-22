# Lint as: python2, python3
"""Pandas utilities for //ads/metrics/lib/meterstick."""

from __future__ import division

import numpy as np
import pandas as pd


def select_by_position(data, idx):
  """Gets row(s) of data corresponding to position idx.

  Args:
    data: a DataFrame
    idx: a position (integer from 0 to len(data)-1)

  Returns:
    A DataFrame containing the rows corresponding to
    position idx.
  """
  try:
    # Double brackets force Pandas to return a DataFrame.
    return data.iloc[[idx]]
  except KeyError:
    # Return a correctly shaped DataFrame of NaNs.
    if isinstance(data.index, pd.MultiIndex):
      return np.nan * data.reset_index(level=0, drop=True)
    else:
      return np.nan * data.reset_index(drop=True)


def select_by_label(data, idx):
  """Gets row(s) of data corresponding to index label idx.

  Args:
    data: a DataFrame
    idx: an index label

  Returns:
    A DataFrame containing the rows corresponding to
    index label idx.
  """
  if isinstance(data.index, pd.MultiIndex):
    try:
      # Pandas always returns DataFrame for a MultiIndex.
      return data.loc[idx]
    except KeyError:
      # If idx not found, return a data frame of all NaNs
      # with the first level dropped.
      return np.nan * data.reset_index(level=0, drop=True)
  else:
    # Force Pandas to return a DataFrame.
    inds = data.index.isin([idx])
    if inds.any():
      # This was the fastest solution in speed testing.
      return data[inds].reset_index(drop=True)
    else:
      return np.nan * data.reset_index(drop=True)


def concat(objects, axis=0, keys=None, name=None):
  """Concatenates Pandas objects.

  Works similarly to pd.concat(), except it is able to
  concatenate scalars as well.

  Args:
    objects: a list of scalars, Series, or DataFrames
    axis: 0 or 1, indicating which axis to concatenate along
    keys: the label to add for each element that gets concatenated
    name: the name for the Series or DataFrame

  Returns:
    A Series or a DataFrame

  Raises:
    ValueError: results could not be concatenated
  """
  if all(isinstance(obj, (pd.Series, pd.DataFrame)) for obj in objects):
    output = pd.concat(objects, axis=axis, keys=keys, names=[name])
  elif all(np.isscalar(obj) for obj in objects):
    if axis == 0:
      output = pd.Series(objects, index=pd.Index(keys, name=name))
    elif axis == 1:
      output = pd.DataFrame([objects], columns=keys)
  else:
    raise ValueError("Could not concatenate objects because the types "
                     "of the objects did not match.")

  return output


def index_product(index1, index2):
  """Produces an index containing the product of the levels of two indexes.

  Functions similarly to pd.MultiIndex.from_product(), but is able to
  take products of MultiIndexes.

  Args:
    index1: A pandas Index (possibly a MultiIndex)
    index2: A pandas Index (possibly a MultiIndex)

  Returns:
    A Pandas MultiIndex whose levels are the product of all levels from
    index1 with all levels from index2.

  Raises:
    ValueError: Both indexes are None, or the indexes have non-unique
      levels.
  """
  # return index2 if index1 is None
  if index1 is None:
    if index2 is None:
      raise ValueError("Both indexes are None.")
    else:
      return index2
  # return index1 if index2 is None
  elif index2 is None:
    return index1

  # check that the values in the indexes are unique
  if not index1.is_unique or not index2.is_unique:
    raise ValueError("Can only take the product of two indexes with "
                     "unique levels.")

  # Get a list of the possible index values.
  # (Each index value is itself a list.)
  index_values = []
  for i in index1:
    for j in index2:
      index_value = []
      if isinstance(index1, pd.MultiIndex):
        index_value.extend(i)
      else:
        index_value.append(i)
      if isinstance(index2, pd.MultiIndex):
        index_value.extend(j)
      else:
        index_value.append(j)
      index_values.append(index_value)

  # Get the names for the index levels.
  index_names = index1.names + index2.names

  # Construct MultiIndex from the values and the names
  return pd.MultiIndex.from_tuples(index_values, names=index_names)


def index_product_from_vars(data, variables, expand):
  """Constructs an index consisting of combinations of levels in vars.

  Args:
    data: A Pandas DataFrame.
    variables: A list of strings representing variables of the DataFrame.
    expand: A boolean; if True, expand index to include all possible
      combinations of levels, whether or not they appear in the data or not.

  Returns:
    A Pandas Index consisting of all combinations of levels in the variables.
  """
  if expand:
    return pd.MultiIndex.from_product(
        [data[var].drop_duplicates() for var in variables],
        names=variables)
  else:
    return (data[variables].drop_duplicates().
            set_index(variables).index)


def any_null(obj):
  """Checks if there are any null values in obj.

  Args:
    obj: A scalar, Series, or DataFrame.

  Returns:
    A boolean. True if there are any NaN values in obj.

  Raises:
    ValueError: if obj is not a scalar, Series, or DataFrame
  """
  if np.isscalar(obj):
    return pd.isnull(obj)
  elif isinstance(obj, pd.Series):
    return obj.isnull().any()
  elif isinstance(obj, pd.DataFrame):
    return obj.isnull().values.any()
  else:
    raise ValueError("obj is not a scalar, Series, or DataFrame.")
