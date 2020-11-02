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
"""Generates HTML to display confidence interval nicely for DataFrames."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from IPython.display import display
from IPython.display import HTML
import pandas as pd
import six
from six.moves import range


CSS = '''
  /* Make the table scrollable */
  #meterstick-container {
    max-height: 700px;
    overflow-y: auto;
  }

  #meterstick td {
    border-color: lightgrey;
    border-style: solid;
    border-width: thin;
    text-align: center;
    padding: 0;
  }

  /* When scrolling, the header sticks and keeps its white background */
  #meterstick th {
    background: white;
    border-color: lightgrey;
    border-bottom-color: black;
    border-style: solid;
    border-width: thin;
    border-top-width: 0;
    position: sticky;
    top: 0;
  }

  /* Wrap the long column name. */
  th {
    max-width: 120px;
    white-space: normal !important;
    word-wrap: break-word;
  }

  .ci-display-cell {
    height: 100%;
    padding: 5px;
  }

  /* Formats Dimension column if there is one. */
  .ci-display-dimension {
    color: green;
    padding: 2px;
  }

  /* Renders the experiment id in blue. */
  .ci-display-experiment-id {
    color: #15c;
  }

  /* Break line in a flex display. */
  .ci-display-flex-line-break {
    width: 100%;
  }

  /* Renders the cells with positive confidence interval to green. */
  .ci-display-good-change {
    background-color: rgb(221,255,187);
    color: green;
  }

  /* Renders the cells with negative confidence interval to red. */
  .ci-display-bad-change {
    background-color: rgb(255,221,187);
    color: red;
  }

  .ci-display-ratio {
    font-size: 120%; /* Renders the ratio value in larger font. */
  }

  .ci-display-ci-range {
    white-space: nowrap;  /* Don't break line in the middle of CI range. */
  }
'''  # You can overwrite this from outside with custom CSS.
# Line breaker in a flex display.
LINE_BREAK = '<div class="ci-display-flex-line-break"></div>'


def _sorted_long_to_wide(df, dims, sort_by):
  """Returns a df in wide format for metrics.

  The input df is in so-called long, or tidy, format, i.e., each
  row is one metric for one slice. This function transforms the df to a wide
  format, where all metrics for one slice are collected into one row. It also
  does extra some things:
  1. Sorts the df,
  2. Collects metric information (value, ratio, ci_lower, ci_upper) to a tuple,
  3. Drops columns no more needed.

  Args:
    df: A dataframe in long format, could just be metrics_types.as_dataframe().
    dims: The column name of slicing dimesions, can be a list or a string.
    sort_by: In the form of [{'column': 'CI-lower', 'ascending': False}},
      {'column': 'Dim_2': 'order': ['Logged-in', 'Logged-out']}]. The 'column'
      is the column to sort by, 'order' is optional and for categorical column,
      and 'ascending' is optional and default True. The result will be displayed
      in the order specified by sort_by from top to bottom.

  Returns:
    A df containing enough information in wide format.
  """
  default_dims = dims + [
      'Control_Id', 'Is_Control', 'Experiment_Id', 'Description'
  ]
  existing_index_cols = [x for x in default_dims if x in df]
  if 'Control_Id' in df and all(pd.isnull(df.Control_Id)):
    # All None column makes pivot_table fail.
    df.drop(columns=['Control_Id'], inplace=True)
    existing_index_cols = [c for c in existing_index_cols if c != 'Control_Id']
  if not existing_index_cols:
    df['_placeholder'] = 42
    existing_index_cols = ['_placeholder']
  val_cols = ['Value', 'Ratio', 'CI_Lower', 'CI_Upper']
  df = df[existing_index_cols + val_cols + ['Metric']]
  # Object columns will get dropped during the unstack().
  df = df.astype({c: 'float64' for c in val_cols})
  for col in existing_index_cols:
    # Missing category will still appear after groupby(). We need to drop them.
    if isinstance(df[col].dtypes, pd.CategoricalDtype):
      df[col] = pd.Categorical(
          df[col], [c for c in df[col].cat.categories if c in df[col].unique()],
          ordered=df[col].cat.ordered)
  # Spread metrics in Metric column to individual columns, i.e., long to wide.
  # pivot_table() doesn't work if there's NA.
  # https://github.com/pandas-dev/pandas/issues/18030#issuecomment-340442023
  df = df.groupby(existing_index_cols + ['Metric']).agg('mean').unstack(-1)
  if not sort_by:
    sorting_cols = existing_index_cols
    ascending = [s != 'Is_Control' for s in sorting_cols]
  else:
    sorting_cols = []
    ascending = []
    for s in sort_by:
      col = s['column']
      sorting_cols.append(col)
      ascending.append(s.get('ascending', True))
      if 'order' in s:
        if col in df:
          df[col] = pd.Categorical(df[col], s['order'], ordered=True)
        else:
          df.reset_index(col, inplace=True)
          df[col] = pd.Categorical(df[col], s['order'], ordered=True)
          df.set_index(col, append=True, inplace=True)
  if sorting_cols:
    df = df.sort_values(sorting_cols, ascending=ascending)
  # Collects [Value, Ratio, CI_Lower, CI_Upper] for each Metric * slice. val_col
  # might be dropped during pivot b/c of na, so we make a dict first.
  df = df.groupby(level=1, axis=1).apply(
      lambda x: x.droplevel(1, 1).apply(lambda row: row.to_dict(), 1))
  df = df.applymap(lambda x: [x.get(k) for k in val_cols]).reset_index()
  if '_placeholder' == existing_index_cols[0]:
    df.drop(columns='_placeholder', inplace=True)

  return df


def _merge_dimensions(df, dims):
  """Merge dimension info columns to a 'Dimensions' column."""
  agg_cols = dims + ['Experiment_Id', 'Is_Control', 'Description']
  agg_cols = [c for c in agg_cols if c in df]
  if agg_cols:
    df['Dimensions'] = df[agg_cols].apply(lambda row: row.to_dict(), axis=1)
    df.drop(columns=agg_cols + ['Control_Id'], inplace=True, errors='ignore')
    # Make 'Dimensions' the first column.
    # df[['Dimensions'] + df.columns[:-1].tolist()] will fail when metric aren't
    # in the same type.
    # https://stackoverflow.com/questions/45175041/reorder-pandas-dataframe-columns-with-mixed-tuple-and-string-columns
    dim_vals = df['Dimensions']
    del df['Dimensions']
    df.insert(0, 'Dimensions', dim_vals)
  return df


def _pre_aggregate_df(df,
                      dims,
                      aggregate_dimensions,
                      show_control,
                      ctrl_id,
                      sort_by=None,
                      auto_decide_control_vals=False,
                      auto_add_description=True):
  """Process a long-format df to an appropriate format for display.

  Args:
    df: A dataframe similar to the one returned by metrics_types.as_dataframe().
    dims: The column name of slicing dimesions, can be a list or a string.
    aggregate_dimensions: If True, all dimension columns are collected into a
      'Dimensions' column, and original dimension columns are dropped.
    show_control: If False, only ratio values in non-control rows are shown.
    ctrl_id: The control experiment id(s). For single control case, it can be
      basically any type that can be used as an experiment key except dict. For
      multiple controls, it should be a dict, with keys being control ids,
      values being list of corresponding experiment id(s).
    sort_by: In the form of [{'column': 'CI-lower', 'ascending': False}},
      {'column': 'Dim_2': 'order': ['Logged-in', 'Logged-out']}]. The 'column'
      is the column to sort by, 'order' is optional and for categorical column,
      and 'ascending' is optional and default True. The result will be displayed
      in the order specified by sort_by from top to bottom.
    auto_decide_control_vals: By default, if users want to see control
      experiments, df needs to have rows for control, but the 'Value' there
      is supposed to be equal to the 'Control_Value' column in experiment rows.
      So if control rows are missing, we can use 'Control_Value' column to fill
      them. The problem is when there are multiple experiments their
      Control_Values might be different (though they shouldn't be). In that case
      we raise a warning and skip. Also if user arleady provide control rows for
      certain slices, we won't fill those slices.
    auto_add_description: If add Control/Not Control as descriptions.

  Returns:
    A pandas dataframe with stylized content for display. The display effect is
      similar to tge_estimation.display().

  Raises:
    ValueError: If metrics is not an instance of MetricsTablesByExperiments,
      MetricsTable, or MetricsPerSlice.
  """
  if 'CI_Upper' not in df or 'CI_Lower' not in df:
    df['CI_Upper'] = df['Ratio'] + (df['CI_Range'] / 2)
    df['CI_Lower'] = df['Ratio'] - (df['CI_Range'] / 2)
  df = _add_is_control_and_control_id(df, ctrl_id)

  if auto_add_description:
    is_ctrl = df['Is_Control'] if 'Is_Control' in df else None
    if is_ctrl is not None:  # ctrl id could be 0 or ''.
      is_ctrl = ['Control' if x else 'Not Control' for x in is_ctrl]
      if 'Description' not in df:
        df['Description'] = is_ctrl
      else:
        df['Description'] = df['Description'].where(
            df['Description'].astype(bool), is_ctrl)  # Only fills empty cells

  if show_control:
    if 'Is_Control' in df:
      # When Ratio is None, CI won't be displayed. This is intended for control.
      df.loc[df['Is_Control'], 'Ratio'] = None
  else:
    df['Value'] = None
    if 'Is_Control' in df:
      # Make a copy to avoid "A value is trying to be set on a copy of a slice
      # from a DataFrame." warning.
      df = df[~df['Is_Control']].copy()
  if auto_decide_control_vals:
    df = add_control_rows(df, dims)
  pre_agg_df = _sorted_long_to_wide(df, dims, sort_by)
  if aggregate_dimensions:
    pre_agg_df = _merge_dimensions(pre_agg_df, dims)
  return pre_agg_df


def _div(s, class_name=None):
  if class_name:
    return '<div class="%s">%s</div>' % (class_name, s)
  return '<div>%s</div>' % s


def _span(s, class_name=None):
  if class_name:
    return '<span class="%s">%s</span>' % (class_name, s)
  return '<span>%s</span>' % s


class MetricFormatter(object):
  """A formatter to highlight significant metric change.

  Concatenates 'Value', 'Ratio', 'CI_Lower', 'CI_Upper' columns in df to a
  stylized form which can be rendered to HTML directly later. Cells with
  positive CI change are rendered green, with negative CI change are rendered
  red.

  Attributes:
    metric_formats: A dict specifying how to display metric values. Keys can be
      'Value' and 'Ratio'. Values can be 'absolute', 'percent', 'pp' or a
      formatting string. For example, '{:.2%}' would have the same effect as
      'percent'. By default, Value is in absolute form and Ratio in percent.
    if_flip_color: A boolean indicating if to flip green/red coloring scheme.
    hide_null_ctrl: If to hide control value or use '-' to represent it when it
      is null,
    form_lookup: A dict to look up formatting str for the display.
    unit_lookup: A dict to look up the unit to append to numbers in display.

  Returns:
    A string specifying a named <div> containing concatenated values. Div may
      include html classes used to style with CSS.
  """

  def __init__(self,
               metric_formats=None,
               if_flip_color=None,
               hide_null_ctrl=False):
    metric_formats = metric_formats or {}
    metric_formats.setdefault('Value', 'absolute')
    metric_formats.setdefault('Ratio', 'absolute')
    self.if_flip_color = if_flip_color
    self.hide_null_ctrl = hide_null_ctrl
    self.metric_formats = metric_formats
    self.form_lookup = {
        'percent': '{:.2f}',
        'absolute': '{:.4f}',
        'pp': '{:.2f}'
    }
    self.unit_lookup = {'percent': '%', 'pp': 'pp'}

  def _format_value(self, val, form, is_ci=False):
    """Formats val in the required form.

    Args:
      val: A single value or a list of [ci_lower, ci_upper].
      form: 'Absolute', 'percent', 'pp' or a formatting string.
      is_ci: If val is a list for CI values.

    Returns:
      A formatted string for display.
    """
    val_format = self.form_lookup.get(form, form)
    unit = self.unit_lookup.get(form, '')
    if isinstance(val, str):
      return val + ' ' + unit

    if not is_ci:
      if pd.isnull(val):
        return 'N/A'
      res = val_format.format(val) + unit
    else:
      ci_lower = 'N/A' if pd.isnull(val[0]) else val_format.format(val[0])
      ci_upper = 'N/A' if pd.isnull(val[1]) else val_format.format(val[1])
      res = '[{}, {}]'.format(ci_lower, ci_upper)
      if unit:
        res = res + ' ' + unit
    return res

  def __call__(self, x, div=_div, span=_span, line_break_join=LINE_BREAK.join):
    """Format elements in x - the metric column, for meaningful display.

    Args:
      x: The metric column of result df. Each cell is a a tuple of (value,
        ratio, ci_lower, ci_upper).
      div: A function to wrap thing into a div.
      span:  A function to wrap thing into a span.
      line_break_join: A function to insert line break.

    Returns:
      A string specifying a named <div> containing dimension. The returned div
      has html class 'ci-display-dimension' for styling with CSS.
    """
    if not isinstance(x, (tuple, list)):
      return div('N/A')

    value, ratio, ci_lower, ci_upper = x
    if self.metric_formats['Value'] == 'percent':
      # This makes more sense for the default case as the units of ratio and
      # value differ by 100.
      value *= 100.0

    value_formatted = self._format_value(value, self.metric_formats['Value'])
    ratio_formatted = self._format_value(ratio, self.metric_formats['Ratio'])
    ratio_formatted = span(ratio_formatted, 'ci-display-ratio')
    ci_formatted = self._format_value([ci_lower, ci_upper],
                                      self.metric_formats['Ratio'], True)
    ci_formatted = span(ci_formatted, 'ci-display-ci-range')
    if pd.isnull(ratio):
      return div(value_formatted, 'ci-display-cell')
    elif pd.isnull(value):
      if self.hide_null_ctrl:
        res = line_break_join([ratio_formatted, ci_formatted])
      else:
        res = line_break_join(['<div>-</div>', ratio_formatted, ci_formatted])
    else:
      res = line_break_join([value_formatted, ratio_formatted, ci_formatted])
    res = div(res)

    ci_lower = ci_lower if ci_lower is not None else 0
    ci_upper = ci_upper if ci_upper is not None else 0
    if ((ci_lower > 0 and not self.if_flip_color) or
        (ci_upper < 0 and self.if_flip_color)):
      return div(res, 'ci-display-good-change ci-display-cell')
    if ((ci_upper < 0 and not self.if_flip_color) or
        (ci_lower > 0 and self.if_flip_color)):
      return div(res, 'ci-display-bad-change ci-display-cell')
    return div(res, 'ci-display-cell')


def dimension_formatter(x,
                        div=_div,
                        span=_span,
                        line_break_join=LINE_BREAK.join):
  """Format elements in x - the Dimension column, for meaningful display.

  Args:
    x: The dimension column of result df. Each cell is a dict. Possible keys are
      'Experiment_Id', 'Description' and 'Dim_[0-9]+'.
    div: A function to wrap thing into a div.
    span:  A function to wrap thing into a span.
    line_break_join: A function to insert line break.

  Returns:
    A string specifying a named <div> containing dimension. The returned div
    has html class 'ci-display-dimension' for styling with CSS.
  """
  slice_cols = [s for s in x if s.startswith('Dim_')]  # Could be empty.
  slice_cols.sort(key=lambda x: int(x.replace('Dim_', '')))
  slice_info = ' * '.join([str(x[s]) for s in slice_cols])
  description = x.get('Description')
  expr_id = x.get('Experiment_Id')
  d = []
  if description:
    d.append(span(description, 'ci-display-description-text'))
  if expr_id:
    d.append(span(expr_id, 'ci-display-experiment-id'))
  if slice_info:
    d.append(span(slice_info, 'ci-display-dimension'))
  return div(div(line_break_join(d)))


def _get_formatter(df,
                   dims,
                   if_flip_colors,
                   hide_null_ctrl=False,
                   metric_formats=None):
  """Returns a custom formatter for df.

  Args:
    df: A dataframe. The first column might be 'Dimensions', the rest should all
      be metric coulmns.
    dims: The column names of slicing dimesions, can be a list or a string.
    if_flip_colors: An array of boolean indicating if to flip green/red coloring
      scheme for that column.
    hide_null_ctrl: If to hide control value or use '-' to represent it when it
      is null,
    metric_formats: A dict specifying how to display metric values. Keys can be
      'Value' and 'Ratio'. Values can be 'absolute', 'percent', 'pp' or a
      formatting string. For example, '{:.2%}' would have the same effect as
      'percent'. By default, Value is in absolute form and Ratio in percent.

  Returns:
    A dict which can be used as a custom formatter for
      colabtools.iteractive_table().
  """
  custom_formatter = {}
  for i, col in enumerate(df.columns):
    if col in dims + ['Experiment_Id', 'Is_Control', 'Description']:
      custom_formatter[i] = '<div>{}</div>'.format
    elif col == 'Dimensions':
      custom_formatter[i] = dimension_formatter
    else:
      custom_formatter[i] = MetricFormatter(metric_formats, if_flip_colors[i],
                                            hide_null_ctrl)
  return custom_formatter


def add_control_rows(df, dims):
  """Adds control rows to df. See auto_decide_control_vals below for context."""
  if 'Is_Control' not in df:
    return df

  curr_ctrl_rows = df[df.Is_Control]
  expr_rows = df[~df.Is_Control]
  control_vals = expr_rows.loc[:, [
      'Experiment_Id', 'Control_Id', 'Control_Value', 'Metric'
  ] + dims]
  grp = control_vals.groupby(['Metric', 'Control_Id'] + dims)['Control_Value']
  control_vals = grp.agg([max, min])
  thresh = 1**-5
  ambiguous_vals = control_vals[control_vals['max'] -
                                control_vals['min'] > thresh]
  msg = ('Warning: Metric "%s" of control %s in slice %s has inconsistent '
         'control values. Skip.')
  for k in ambiguous_vals.iterrows():
    print(msg % (k[0][0], k[0][1], k[0][2:]))
  valid_ctrl = control_vals[control_vals['max'] -
                            control_vals['min'] < thresh][['min']]
  valid_ctrl.reset_index(inplace=True)
  valid_ctrl['Experiment_Id'] = valid_ctrl['Control_Id']
  if valid_ctrl.shape[0]:
    new_ctrl_rows = curr_ctrl_rows.merge(
        valid_ctrl, 'outer', ['Experiment_Id', 'Control_Id', 'Metric'] + dims)
  new_ctrl_rows['Value'] = new_ctrl_rows['Value'].fillna(new_ctrl_rows['min'])
  new_ctrl_rows.drop(columns='min', inplace=True)
  new_ctrl_rows['Is_Control'] = True
  if 'Description' in df:
    new_ctrl_rows['Description'] = 'Control'
  return pd.concat([expr_rows, new_ctrl_rows], sort=False)


def _add_is_control_and_control_id(df, ctrl_id):
  """Adds Is_Control and Control_Id columns to df.

  Args:
    df: A dataframe similar to the one returned by metrics_types.as_dataframe().
    ctrl_id: The control experiment id(s). For single control case, it can be
      basically any type that can be used as an experiment key except dict. For
      multiple controls, it should be a dict, with keys being control ids,
      values being list of corresponding experiment id(s).

  Returns:
    A DataFrame. Is_Control and Control_Id are either both presented or neither.
  """
  if 'Experiment_Id' not in df:
    return df
  if ctrl_id is None:
    if 'Is_Control' not in df and 'Control_Id' in df and 'Experiment_Id' in df:
      df['Is_Control'] = df['Control_Id'] == df['Experiment_Id']
    return df

  if not isinstance(ctrl_id, dict):
    df['Is_Control'] = df['Experiment_Id'] == ctrl_id
    df['Control_Id'] = ctrl_id
    return df

  ctrl_id_lookup = {}
  for ctrl, exp_ids in six.iteritems(ctrl_id):
    if not isinstance(exp_ids, list):
      raise ValueError('The experiment id(s) {} is not a list.'.format(exp_ids))
    for e in exp_ids:
      ctrl_id_lookup[e] = ctrl
    ctrl_id_lookup[ctrl] = ctrl  # The control of a control is itself.
  df['Is_Control'] = df['Experiment_Id'].apply(lambda k: k in ctrl_id)
  df['Control_Id'] = df['Experiment_Id'].apply(ctrl_id_lookup.get)
  return df


def get_formatted_df(df,
                     dims=None,
                     aggregate_dimensions=True,
                     show_control=True,
                     metric_formats=None,
                     metric='Metric',
                     ratio='Ratio',
                     value='Value',
                     ci_upper='CI_Upper',
                     ci_lower='CI_Lower',
                     ci_range='CI_Range',
                     control_value='Control_Value',
                     expr_id='Experiment_Id',
                     ctrl_id=None,
                     description='Description',
                     sort_by=None,
                     metric_order=None,
                     flip_color=None,
                     hide_null_ctrl=False,
                     display_expr_info=False,
                     auto_decide_control_vals=False,
                     auto_add_description=True,
                     return_pre_agg_df=False):
  """Gets the formatted df with raw HTML as values in every cell.

  When rendered, if aggregate_dimensions=True, the first column is for dimension
  info. Each cell looks like
      description (if provided)
      experiment id
      dim1 * dim2 *...
  If not aggregate_dimensions, the dimension info will be spreaded into multple
  columns at left of the display.
  All the remaining columns are for metrics. Each cell, if in control rows, will
  display a single value, for the experiment rows, will show three rows,

                        metric value of experiment
                    ratio (metric value / contraol value)
      [ratio's confidence interval lower, ratio's confidence interval higher]

  If the confidence interval is significant, the cell will be rendered in green
  (for positive change) or red (for negative change).

  Args:
    df: A long-format dataframe with columns for (all optional unless specified)
      dimensions - if dims is None, we look for columns 'Dim_1', 'Dim_2'... in
        df.
      experiment id (Required). We check ctrl_id against this column to decide
        control rows.
      metric - A column with all metric names, hence df is in long format.
      value (Required) - The value of the metric, displayed in the first row in
        an experiment metric cell.
      control value - The value of the metric in the control experiment,
        displayed in cells for control rows.
      ratio (Required) - value / control valu, displayed in the 2nd row in an
        experiment metric cell.
        NOTE We don't check if ratio really equals value / control value and we
        don't try to calculate it if it's missing. We just render it in the 2nd
        row, so it's possible you provide a wrong ratio column and the numbers
        displayed don't make sense.
      ci lower - The lower bound of the confidence interval of ratio.
      ci upper - The upper bound of the confidence interval of ratio.
      ci range - ci upper - ci lower. You need to provide either ci range or
        (ci lower and upper).
      description - The description of the experiment, when provided, will be
        rendered as the first row in a dimension cell.
    dims: The column names of slicing dimesions, can be a list or a string.
    aggregate_dimensions: Whether to aggregate all dimensions in to one column.
    show_control: If False, only ratio values in non-control rows are shown.
    metric_formats: A dict specifying how to display metric values. Keys can be
      'Value' and 'Ratio'. Values can be 'absolute', 'percent', 'pp' or a
      formatting string. For example, '{:.2%}' would have the same effect as
      'percent'. By default, Value is in absolute form and Ratio in percent.
    metric: The column name of metrics.
    ratio: The column name for ratio.
    value: The column name for value.
    ci_upper: The column name for ci_upper.
    ci_lower: The column name for ci_lower.
    ci_range: The column name for ci_range.
    control_value: The column name for control_value.
    expr_id: The column name for experiment ids.
    ctrl_id: The control experiment id(s). For single control case, it can be
      basically any type that can be used as an experiment key except dict. For
      multiple controls, it should be a dict, with keys being control ids,
      values being list of corresponding experiment id(s).
    description: The column name for the optional description of each row. It'll
      be rendered together with experiment id for a more meaningful display.
    sort_by: In the form of
      [{'column': ('CI_Lower', 'Metric Foo'), 'ascending': False}},
       {'column': 'Dim Bar': 'order': ['Logged-in', 'Logged-out']}]. 'column'
      is the column to sort by. If you want to sort by a metric, use
      (field, metric name) where field could be 'Ratio', 'Value', 'CI_Lower' and
      'CI_Upper'. 'order' is optional and for categorical column. 'ascending' is
      optional and default True. The result will be displayed in the order
      specified by sort_by from top to bottom.
    metric_order: An iterable. The metric will be displayed by the order from
      left to right.
    flip_color: A iterable of metric names that positive changes will be
      displayed in red and negative changes in green.
    hide_null_ctrl: If to hide control value or use '-' to represent it when it
      is null,
    display_expr_info: If to display 'Control_id', 'Is_Control' and
      'Description' columns. Only has effect when aggregate_dimensions = False.
    auto_decide_control_vals: By default, if users want to see control
      experiments, df needs to have rows for control, but the 'Value' there
      is supposed to be equal to the 'Control_Value' column in experiment rows.
      So if control rows are missing, we can use 'Control_Value' column to fill
      them. The problem is when there are multiple experiments their
      Control_Values might be different (though they shouldn't be). In that case
      we raise a warning and skip. Also if user arleady provide control rows for
      certain slices, we won't fill those slices.
    auto_add_description: If add Control/Not Control as descriptions.
    return_pre_agg_df: If to return the pre-aggregated df.

  Returns:
    A DataFrame with raw HTML in each cell ready to be rendered. If
    return_pre_agg_df, it returns a not yet stringnified version where you can
    play with numbers.
  """
  if not dims:
    dims = df.columns[df.columns.str.startswith('Dim_')].tolist()
    dims.sort()
  dims = [dims] if isinstance(dims, str) else dims
  # Use fixed column names to make data manipulation easier.
  col_rename = {
      metric: 'Metric',
      ratio: 'Ratio',
      value: 'Value',
      ci_upper: 'CI_Upper',
      ci_lower: 'CI_Lower',
      ci_range: 'CI_Range',
      control_value: 'Control_Value',
      expr_id: 'Experiment_Id',
      description: 'Description',
  }
  if aggregate_dimensions:
    for i, d in enumerate(dims):
      col_rename[d] = 'Dim_%s' % (i + 1)
    dims = ['Dim_%s' % i for i in range(1, len(dims) + 1)]
  sort_by = sort_by or []
  for s in sort_by:
    s['column'] = col_rename.get(s['column'], s['column'])
  df_renamed = df.rename(columns=col_rename)
  formatted_df = _pre_aggregate_df(df_renamed, dims, aggregate_dimensions,
                                   show_control, ctrl_id, sort_by,
                                   auto_decide_control_vals,
                                   auto_add_description)

  if metric_order:
    # Sort metric order based on the given argument or default metric order.
    columns = ['Dimensions'] if aggregate_dimensions else dims[:]
    columns += ['Experiment_Id', 'Is_Control', 'Description']
    columns += list(metric_order)
    column_order = [c for c in columns if c in formatted_df]
    formatted_df = formatted_df[column_order]

  if not aggregate_dimensions and not display_expr_info:
    formatted_df.drop(['Control_Id', 'Is_Control', 'Description'],
                      axis=1,
                      inplace=True,
                      errors='ignore')

  flip_color = flip_color or []
  if_flip_colors = [c in flip_color for c in formatted_df.columns]

  custom_formatters = _get_formatter(formatted_df, dims, if_flip_colors,
                                     hide_null_ctrl, metric_formats)
  formatted_df = formatted_df.rename(columns={
      'Experiment_Id': expr_id,
      'Description': description
  })
  if return_pre_agg_df:
    return formatted_df

  for i, col in enumerate(formatted_df.columns):
    formatted_df[col] = formatted_df[col].apply(custom_formatters[i])
  return formatted_df


def display_formatted_df(formatted_df, extra_css=''):
  """Renders formatted_df returned by get_formatted_df() in notebook.

  Args:
    formatted_df: A DataFrame whose cell values are raw HTML to be rendered.
    extra_css: Additional CSS to apply. You can use it to overwrite the default
      one.
  """
  curr_max_colwidth = pd.get_option('display.max_colwidth')
  pd.set_option('display.max_colwidth', None)  # So HTML won't get truncated.
  df_html = formatted_df.to_html(
      escape=False,
      index=False,
      index_names=False,
      table_id='meterstick',
      justify='center')
  raw_html = '<style>%s</style><div id="meterstick-container">%s</div>' % (
      CSS + extra_css, df_html)
  display(HTML(raw_html))
  pd.set_option('display.max_colwidth', curr_max_colwidth)


def render(df,
           dims=None,
           aggregate_dimensions=True,
           show_control=True,
           metric_formats=None,
           metric='Metric',
           ratio='Ratio',
           value='Value',
           ci_upper='CI_Upper',
           ci_lower='CI_Lower',
           ci_range='CI_Range',
           control_value='Control_Value',
           expr_id='Experiment_Id',
           ctrl_id=None,
           description='Description',
           sort_by=None,
           metric_order=None,
           flip_color=None,
           hide_null_ctrl=False,
           display_expr_info=False,
           auto_decide_control_vals=False,
           auto_add_description=True,
           return_pre_agg_df=False,
           return_formatted_df=False):
  """Gets the formatted df with raw HTML as values in every cell.

  When rendered, if aggregate_dimensions=True, the first column is for dimension
  info. Each cell looks like
      description (if provided)
      experiment id
      dim1 * dim2 *...
  If not aggregate_dimensions, the dimension info will be spreaded into multple
  columns at left of the display.
  All the remaining columns are for metrics. Each cell, if in control rows, will
  display a single value, for the experiment rows, will show three rows,

                        metric value of experiment
                    ratio (metric value / contraol value)
      [ratio's confidence interval lower, ratio's confidence interval higher]

  If the confidence interval is significant, the cell will be rendered in green
  (for positive change) or red (for negative change).

  Args:
    df: A long-format dataframe with columns for (all optional unless specified)
      dimensions - if dims is None, we look for columns 'Dim_1', 'Dim_2'... in
        df.
      experiment id (Required). We check ctrl_id against this column to decide
        control rows.
      metric - A column with all metric names, hence df is in long format.
      value (Required) - The value of the metric, displayed in the first row in
        an experiment metric cell.
      control value - The value of the metric in the control experiment,
        displayed in cells for control rows.
      ratio (Required) - value / control valu, displayed in the 2nd row in an
        experiment metric cell.
        NOTE We don't check if ratio really equals value / control value and we
        don't try to calculate it if it's missing. We just render it in the 2nd
        row, so it's possible you provide a wrong ratio column and the numbers
        displayed don't make sense.
      ci lower - The lower bound of the confidence interval of ratio.
      ci upper - The upper bound of the confidence interval of ratio.
      ci range - ci upper - ci lower. You need to provide either ci range or
        (ci lower and upper).
      description - The description of the experiment, when provided, will be
        rendered as the first row in a dimension cell.
    dims: The column names of slicing dimesions, can be a list or a string.
    aggregate_dimensions: Whether to aggregate all dimensions in to one column.
    show_control: If False, only ratio values in non-control rows are shown.
    metric_formats: A dict specifying how to display metric values. Keys can be
      'Value' and 'Ratio'. Values can be 'absolute', 'percent', 'pp' or a
      formatting string. For example, '{:.2%}' would have the same effect as
      'percent'. By default, Value is in absolute form and Ratio in percent.
    metric: The column name of metrics.
    ratio: The column name for ratio.
    value: The column name for value.
    ci_upper: The column name for ci_upper.
    ci_lower: The column name for ci_lower.
    ci_range: The column name for ci_range.
    control_value: The column name for control_value.
    expr_id: The column name for experiment ids.
    ctrl_id: The control experiment id(s). For single control case, it can be
      basically any type that can be used as an experiment key except dict. For
      multiple controls, it should be a dict, with keys being control ids,
      values being list of corresponding experiment id(s).
    description: The column name for the optional description of each row. It'll
      be rendered together with experiment id for a more meaningful display.
    sort_by: In the form of
      [{'column': ('CI_Lower', 'Metric Foo'), 'ascending': False}},
       {'column': 'Dim Bar': 'order': ['Logged-in', 'Logged-out']}]. 'column'
      is the column to sort by. If you want to sort by a metric, use
      (field, metric name) where field could be 'Ratio', 'Value', 'CI_Lower' and
      'CI_Upper'. 'order' is optional and for categorical column. 'ascending' is
      optional and default True. The result will be displayed in the order
      specified by sort_by from top to bottom.
    metric_order: An iterable. The metric will be displayed by the order from
      left to right.
    flip_color: A iterable of metric names that positive changes will be
      displayed in red and negative changes in green.
    hide_null_ctrl: If to hide control value or use '-' to represent it when it
      is null,
    display_expr_info: If to display 'Control_id', 'Is_Control' and
      'Description' columns. Only has effect when aggregate_dimensions = False.
    auto_decide_control_vals: By default, if users want to see control
      experiments, df needs to have rows for control, but the 'Value' there
      is supposed to be equal to the 'Control_Value' column in experiment rows.
      So if control rows are missing, we can use 'Control_Value' column to fill
      them. The problem is when there are multiple experiments their
      Control_Values might be different (though they shouldn't be). In that case
      we raise a warning and skip. Also if user arleady provide control rows for
      certain slices, we won't fill those slices.
    auto_add_description: If add Control/Not Control as descriptions.
    return_pre_agg_df: If to return the pre-aggregated df.
    return_formatted_df: If to return raw HTML df to be rendered.

  Returns:
    Displays confidence interval nicely for df, or aggregated/formatted if
    return_pre_agg_df/return_formatted_df is True
  """
  formatted_df = get_formatted_df(df, dims, aggregate_dimensions, show_control,
                                  metric_formats, metric, ratio, value,
                                  ci_upper, ci_lower, ci_range, control_value,
                                  expr_id, ctrl_id, description, sort_by,
                                  metric_order, flip_color, hide_null_ctrl,
                                  display_expr_info, auto_decide_control_vals,
                                  auto_add_description, return_pre_agg_df)
  if return_pre_agg_df or return_formatted_df:
    return formatted_df
  return display_formatted_df(formatted_df)
