from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from meterstick import confidence_interval_display
import pandas as pd
from pandas.util.testing import assert_frame_equal
from google3.testing.pybase import googletest

DF_WITH_DIMENSIONS = pd.DataFrame({
    'CI_Lower': [None, None, -5.035, 0.73],
    'CI_Upper': [None, None, 2.235, 4.85],
    'Control_Id': ['expr_foo', 'expr_foo', 'expr_foo', 'expr_foo'],
    'Control_Value': [None, None, 0.780933, 0.21599],
    'Country': ['GB', 'US', 'GB', 'US'],
    'Type': ['WEB', 'WEB', 'WEB', 'WEB'],
    'Experiment_Id': ['expr_foo', 'expr_foo', 42, 42],
    'Is_Control': [True, True, False, False],
    'Metric': ['PLA_CONV_CVR', 'PLA_CONV_CVR', 'PLA_CONV_CVR', 'PLA_CONV_CVR'],
    'Ratio': [None, None, -1.4, 2.79],
    'Value': [0.787, 0.216, 0.77, 0.222026]
})
DF_NO_DIMENSION = pd.DataFrame({
    'CI_Lower': [None, 5],
    'CI_Upper': [None, 15.000],
    'Control_Id': ['expr_foo', 'expr_foo'],
    'Control_Value': [None, 1],
    'Experiment_Id': [2, 'expr_bar'],
    'Is_Control': [True, False],
    'Metric': ['metric_foo', 'metric_foo'],
    'Ratio': [None, 10],
    'Value': [1, 1.10]
})
LINE_BREAK = '<div class="ci-display-flex-line-break"></div>'


class DisplayMetricsTest(googletest.TestCase):

  def testNormal(self):
    expected = pd.DataFrame(
        {
            'Country': [
                '<div>GB</div>',
                '<div>GB</div>',
                '<div>US</div>',
                '<div>US</div>',
            ],
            'Experiment_Id': [
                '<div>expr_foo</div>',
                '<div>42</div>',
                '<div>expr_foo</div>',
                '<div>42</div>',
            ],
            'PLA_CONV_CVR': [
                '<div class="ci-display-cell">0.7870</div>',
                LINE_BREAK.join(
                    ('<div class="ci-display-cell"><div>0.7700',
                     '<span class="ci-display-ratio">-1.40%</span>',
                     '<span class="ci-display-ci-range">[-5.04, 2.23] %</span>'
                     '</div></div>')),
                '<div class="ci-display-cell">0.2160</div>',
                LINE_BREAK.join((
                    '<div class="ci-display-good-ci-change ci-display-cell">'
                    '<div>0.2220',
                    '<span class="ci-display-ratio">2.79%</span>',
                    '<span class="ci-display-ci-range">[0.73, 4.85] %</span>'
                    '</div></div>')),
            ],
        },
        columns=['Country', 'Experiment_Id', 'PLA_CONV_CVR'])
    actual = confidence_interval_display.get_formatted_df(
        DF_WITH_DIMENSIONS,
        dims=['Country'],
        aggregate_dimensions=False,
        show_control=True,
        ctrl_id='expr_foo',
        auto_add_description=False)
    assert_frame_equal(expected, actual, check_names=False)

  def testFlipColor(self):
    expected = pd.DataFrame(
        {
            'Country': [
                '<div>GB</div>',
                '<div>GB</div>',
                '<div>US</div>',
                '<div>US</div>',
            ],
            'Experiment_Id': [
                '<div>expr_foo</div>',
                '<div>42</div>',
                '<div>expr_foo</div>',
                '<div>42</div>',
            ],
            'PLA_CONV_CVR': [
                '<div class="ci-display-cell">0.7870</div>',
                LINE_BREAK.join(
                    ('<div class="ci-display-cell"><div>0.7700',
                     '<span class="ci-display-ratio">-1.40%</span>',
                     '<span class="ci-display-ci-range">[-5.04, 2.23] %</span>'
                     '</div></div>')),
                '<div class="ci-display-cell">0.2160</div>',
                LINE_BREAK.join((
                    '<div class="ci-display-bad-ci-change ci-display-cell"><div>'
                    '0.2220', '<span class="ci-display-ratio">2.79%</span>',
                    '<span class="ci-display-ci-range">[0.73, 4.85] %</span>'
                    '</div></div>')),
            ],
        },
        columns=['Country', 'Experiment_Id', 'PLA_CONV_CVR'])
    actual = confidence_interval_display.get_formatted_df(
        DF_WITH_DIMENSIONS,
        dims=['Country'],
        aggregate_dimensions=False,
        flip_color=['PLA_CONV_CVR'],
        show_control=True,
        ctrl_id='expr_foo',
        auto_add_description=False)
    assert_frame_equal(expected, actual, check_names=False)

  def testDisplayDfWithDimensions(self):
    expected = pd.DataFrame(
        {
            'Dimensions': [{
                'Dim_1': 'GB',
                'Experiment_Id': 'expr_foo',
                'Is_Control': True,
                'Dim_2': 'WEB'
            }, {
                'Dim_1': 'GB',
                'Experiment_Id': 42,
                'Is_Control': False,
                'Dim_2': 'WEB'
            }, {
                'Dim_1': 'US',
                'Experiment_Id': 'expr_foo',
                'Is_Control': True,
                'Dim_2': 'WEB'
            }, {
                'Dim_1': 'US',
                'Experiment_Id': 42,
                'Is_Control': False,
                'Dim_2': 'WEB'
            }],
            'PLA_CONV_CVR': [(0.787, None, None, None),
                             (0.77, -1.4, -5.035, 2.235),
                             (0.216, None, None, None),
                             (0.222026, 2.79, 0.73, 4.85)]
        },
        columns=['Dimensions', 'PLA_CONV_CVR'])
    actual = confidence_interval_display.get_formatted_df(
        DF_WITH_DIMENSIONS,
        dims=['Country', 'Type'],
        aggregate_dimensions=True,
        show_control=True,
        ctrl_id='expr_foo',
        auto_add_description=False,
        return_pre_agg_df=True)
    assert_frame_equal(expected, actual, check_names=False)

  def testDisplayExprInfo(self):
    expected = pd.DataFrame(
        {
            'Country': ['GB', 'GB', 'US', 'US'],
            'Control_Id': ['expr_foo'] * 4,
            'Experiment_Id': ['expr_foo', 42, 'expr_foo', 42],
            'Is_Control': [True, False, True, False],
            'Type': ['WEB', 'WEB', 'WEB', 'WEB'],
            'PLA_CONV_CVR': [(0.787, None, None, None),
                             (0.77, -1.4, -5.035, 2.235),
                             (0.216, None, None, None),
                             (0.222026, 2.79, 0.73, 4.85)]
        },
        columns=[
            'Country', 'Type', 'Control_Id', 'Is_Control', 'Experiment_Id',
            'PLA_CONV_CVR'
        ])
    actual = confidence_interval_display.get_formatted_df(
        DF_WITH_DIMENSIONS,
        dims=['Country', 'Type'],
        aggregate_dimensions=False,
        show_control=True,
        ctrl_id={'expr_foo': [42, 'not existed']},
        display_expr_info=True,
        auto_add_description=False,
        return_pre_agg_df=True)
    assert_frame_equal(expected, actual, check_names=False)

  def testAutoAddDescription(self):
    expected = pd.DataFrame(
        {
            'Country': ['GB', 'GB', 'US', 'US'],
            'Control_Id': ['expr_foo'] * 4,
            'Experiment_Id': ['expr_foo', 42, 'expr_foo', 42],
            'Is_Control': [True, False, True, False],
            'Type': ['WEB', 'WEB', 'WEB', 'WEB'],
            'Description': ['Control', 'Not Control'] * 2,
            'PLA_CONV_CVR': [(0.787, None, None, None),
                             (0.77, -1.4, -5.035, 2.235),
                             (0.216, None, None, None),
                             (0.222026, 2.79, 0.73, 4.85)]
        },
        columns=[
            'Country', 'Type', 'Control_Id', 'Is_Control', 'Experiment_Id',
            'Description', 'PLA_CONV_CVR'
        ])
    actual = confidence_interval_display.get_formatted_df(
        DF_WITH_DIMENSIONS,
        dims=['Country', 'Type'],
        aggregate_dimensions=False,
        show_control=True,
        ctrl_id={'expr_foo': [42, 'not existed']},
        display_expr_info=True,
        return_pre_agg_df=True)
    assert_frame_equal(expected, actual, check_names=False)

  def testDisplayDfWithDimensionsAggregateDimensionsFalse(self):
    expected = pd.DataFrame(
        {
            'Country': ['GB', 'GB', 'US', 'US'],
            'Experiment_Id': ['expr_foo', 42, 'expr_foo', 42],
            'Type': ['WEB', 'WEB', 'WEB', 'WEB'],
            'PLA_CONV_CVR': [(0.787, None, None, None),
                             (0.77, -1.4, -5.035, 2.235),
                             (0.216, None, None, None),
                             (0.222026, 2.79, 0.73, 4.85)]
        },
        columns=['Country', 'Type', 'Experiment_Id', 'PLA_CONV_CVR'])
    actual = confidence_interval_display.get_formatted_df(
        DF_WITH_DIMENSIONS,
        dims=['Country', 'Type'],
        aggregate_dimensions=False,
        show_control=True,
        ctrl_id={'expr_foo': [42, 'not existed']},
        auto_add_description=False,
        return_pre_agg_df=True)
    assert_frame_equal(expected, actual, check_names=False)

  def testDisplayDfWithDimensionsShowControlFalse(self):
    expected = pd.DataFrame(
        {
            'PLA_CONV_CVR': [(None, -1.4, -5.035, 2.235),
                             (None, 2.79, 0.73, 4.85)],
            'Country': ['GB', 'US'],
            'Experiment_Id': [42, 42],
            'Type': ['WEB', 'WEB']
        },
        columns=['Country', 'Type', 'Experiment_Id', 'PLA_CONV_CVR'])
    actual = confidence_interval_display.get_formatted_df(
        DF_WITH_DIMENSIONS,
        dims=['Country', 'Type'],
        aggregate_dimensions=False,
        show_control=False,
        ctrl_id='expr_foo',
        auto_add_description=False,
        return_pre_agg_df=True)
    assert_frame_equal(expected, actual, check_names=False)

  def testDisplayDfNoDimension(self):
    expected = pd.DataFrame(
        {
            'Dimensions': [{
                'Experiment_Id': 2,
                'Is_Control': True
            }, {
                'Experiment_Id': 'expr_bar',
                'Is_Control': False
            }],
            'metric_foo': [(1.0, None, None, None), (1.1, 10.0, 5.0, 15.0)]
        },
        columns=['Dimensions', 'metric_foo'])
    actual = confidence_interval_display.get_formatted_df(
        DF_NO_DIMENSION,
        dims=['Country', 'Type'],
        aggregate_dimensions=True,
        show_control=True,
        ctrl_id=2,
        auto_add_description=False,
        return_pre_agg_df=True)
    assert_frame_equal(expected, actual, check_names=False)

  def testDisplayUsingCIRange(self):
    expected = confidence_interval_display.get_formatted_df(
        DF_NO_DIMENSION,
        dims=['Country', 'Type'],
        aggregate_dimensions=True,
        show_control=True,
        ctrl_id=2,
        auto_add_description=False,
        return_pre_agg_df=True)
    df_no_ci = DF_NO_DIMENSION.copy()
    df_no_ci['CI_Range'] = df_no_ci['CI_Upper'] - df_no_ci['CI_Lower']
    del df_no_ci['CI_Upper'], df_no_ci['CI_Lower']
    actual = confidence_interval_display.get_formatted_df(
        df_no_ci,
        dims=['Country', 'Type'],
        aggregate_dimensions=True,
        show_control=True,
        ctrl_id=2,
        auto_add_description=False,
        return_pre_agg_df=True)
    assert_frame_equal(expected, actual)

  def testDisplayDfNoDimensionAggregateDimensionsFalse(self):
    expected = pd.DataFrame(
        {
            'Experiment_Id': [2, 'expr_bar'],
            'metric_foo': [(1.0, None, None, None), (1.1, 10.0, 5.0, 15.0)]
        },
        columns=['Experiment_Id', 'metric_foo'])
    actual = confidence_interval_display.get_formatted_df(
        DF_NO_DIMENSION,
        dims=['Country', 'Type'],
        aggregate_dimensions=False,
        show_control=True,
        ctrl_id=2,
        auto_add_description=False,
        return_pre_agg_df=True)
    assert_frame_equal(expected, actual, check_names=False)

  def testDisplayDfNoDimensionShowControlFalse(self):
    expected = pd.DataFrame(
        {
            'Experiment_Id': ['expr_bar'],
            'metric_foo': [(None, 10.0, 5.0, 15.0)]
        },
        columns=['Experiment_Id', 'metric_foo'])
    actual = confidence_interval_display.get_formatted_df(
        DF_NO_DIMENSION,
        dims=['Country', 'Type'],
        aggregate_dimensions=False,
        show_control=False,
        ctrl_id=2,
        auto_add_description=False,
        return_pre_agg_df=True)
    assert_frame_equal(expected, actual, check_names=False)

  def testMetricFormatterWithNoRatio(self):
    expected = '<div class="ci-display-cell">1.2000</div>'
    actual = confidence_interval_display.MetricFormatter()(
        (1.2, None, None, None))
    self.assertEqual(expected, actual)

  def testMetricFormatterWithNoValue(self):
    expected = LINE_BREAK.join(
        ('<div class="ci-display-good-ci-change ci-display-cell"><div>'
         '<div>-</div>', '<span class="ci-display-ratio">1.20%</span>',
         '<span class="ci-display-ci-range">[1.00, 1.40] %</span></div></div>'))
    actual = confidence_interval_display.MetricFormatter()((None, 1.2, 1, 1.4))
    self.assertEqual(expected, actual)

  def testMetricFormatterWithPercentageValueAndAbsoluteRatio(self):
    expected = ('<div class="ci-display-good-ci-change ci-display-cell"><div>'
                '24.44%' + LINE_BREAK +
                '<span class="ci-display-ratio">1.2000</span>' + LINE_BREAK +
                '<span class="ci-display-ci-range">[1.0000, 1.4000]</span>' +
                '</div></div>')
    formats = {'Ratio': 'absolute', 'Value': 'percent'}
    actual = confidence_interval_display.MetricFormatter(formats)(
        (0.24444, 1.2, 1, 1.4))
    self.assertEqual(expected, actual)

  def testMetricFormatterWithPercentageValueAndPpRatio(self):
    expected = LINE_BREAK.join((
        '<div class="ci-display-good-ci-change ci-display-cell"><div>24.44%',
        '<span class="ci-display-ratio">1.20pp</span>',
        '<span class="ci-display-ci-range">[1.00, 1.40] pp</span></div></div>'))
    formats = {'Ratio': 'pp', 'Value': 'percent'}
    actual = confidence_interval_display.MetricFormatter(formats)(
        (0.24444, 1.2, 1, 1.4))
    self.assertEqual(expected, actual)

  def testMetricFormatterWithCustomFormatting(self):
    expected = LINE_BREAK.join(
        ('<div class="ci-display-good-ci-change ci-display-cell"><div>constant',
         '<span class="ci-display-ratio">1</span>',
         '<span class="ci-display-ci-range">[1, 1]</span></div></div>'))
    formats = {'Ratio': '{:.0f}', 'Value': 'constant'}
    actual = confidence_interval_display.MetricFormatter(formats)(
        (0.24444, 1.2, 1, 1.4))
    self.assertEqual(expected, actual)

  def testMetricFormatterWithPositiveCi(self):
    expected = LINE_BREAK.join(
        ('<div class="ci-display-good-ci-change ci-display-cell"><div>1.2000',
         '<span class="ci-display-ratio">1.10%</span>',
         '<span class="ci-display-ci-range">[1.02, 1.18] %</span></div></div>'))
    actual = confidence_interval_display.MetricFormatter()(
        (1.2, 1.1, 1.02, 1.18))
    self.assertEqual(expected, actual)

  def testMetricFormatterWithNegativeCi(self):
    expected = LINE_BREAK.join(
        ('<div class="ci-display-bad-ci-change ci-display-cell"><div>1.2000',
         '<span class="ci-display-ratio">-1.10%</span>',
         '<span class="ci-display-ci-range">[-1.02, -1.18] %</span></div></div>'
        ))
    actual = confidence_interval_display.MetricFormatter()(
        (1.2, -1.1, -1.02, -1.18))
    self.assertEqual(expected, actual)

  def testMetricFormatterWithNACis(self):
    expected = LINE_BREAK.join(
        ('<div class="ci-display-cell"><div>24.44%',
         '<span class="ci-display-ratio">1.2000</span>',
         '<span class="ci-display-ci-range">[N/A, N/A]</span></div></div>'))
    formats = {'Ratio': 'absolute', 'Value': 'percent'}
    actual = confidence_interval_display.MetricFormatter(formats)(
        (0.24444, 1.2, None, None))
    self.assertEqual(expected, actual)

  def testMetricFormatterWithPositiveCiFlipColor(self):
    expected = LINE_BREAK.join(
        ('<div class="ci-display-bad-ci-change ci-display-cell"><div>1.2000',
         '<span class="ci-display-ratio">1.10%</span>',
         '<span class="ci-display-ci-range">[1.02, 1.18] %</span></div></div>'))
    actual = confidence_interval_display.MetricFormatter(if_flip_color=True)(
        (1.2, 1.1, 1.02, 1.18))
    self.assertEqual(expected, actual)

  def testMetricFormatterWithNegativeCiFlipColor(self):
    expected = LINE_BREAK.join(
        ('<div class="ci-display-good-ci-change ci-display-cell"><div>1.2000',
         '<span class="ci-display-ratio">-1.10%</span>',
         '<span class="ci-display-ci-range">[-1.02, -1.18] %</span></div></div>'
        ))
    actual = confidence_interval_display.MetricFormatter(if_flip_color=True)(
        (1.2, -1.1, -1.02, -1.18))
    self.assertEqual(expected, actual)

  def testDimensionFormatter(self):
    x = {
        'Dim_1': 'Mobile',
        'Dim_2': 'WEB',
        'Description': 'foo',
        'Experiment_Id': 42
    }
    expected = LINE_BREAK.join(
        ('<div><div><span class="ci-display-description-text">foo</span>',
         '<span class="ci-display-experiment-id">42</span>',
         '<span class="ci-display-dimension">Mobile * WEB</span></div></div>'))
    actual = confidence_interval_display.dimension_formatter(x)
    self.assertEqual(expected, actual)

  def testDimensionFormatterWithMissingField(self):
    x = {
        'Dim_1': 'Mobile',
        'Dim_2': 'WEB',
        'Experiment_Id': 42
    }  # No description.
    expected = LINE_BREAK.join(
        ('<div><div><span class="ci-display-experiment-id">42</span>',
         '<span class="ci-display-dimension">Mobile * WEB</span></div></div>'))
    actual = confidence_interval_display.dimension_formatter(x)
    self.assertEqual(expected, actual)


if __name__ == '__main__':
  googletest.main()
