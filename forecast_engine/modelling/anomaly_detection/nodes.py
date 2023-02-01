
# Anomaly Detection

def detect_ts_local(df, max_anoms=0.10, direction='pos',
              alpha=0.05, only_last=None, threshold=None,
              e_value=False, longterm=False,
              piecewise_median_period_weeks=2, plot=False,
              y_log=False, xlabel = '', ylabel = 'count',
              title=None, verbose=False):
  
    from collections import namedtuple
    from pyculiarity.date_utils import format_timestamp, get_gran, date_format, datetimes_from_ts
    from math import ceil
    from pandas import DataFrame, Timestamp
    import datetime
    import numpy as np
    from six import string_types
    
    Direction = namedtuple('Direction', ['one_tail', 'upper_tail'])
    
    if not isinstance(df, DataFrame):
        raise ValueError("data must be a single data frame.")
    else:
        if len(df.columns) != 2 or not df.iloc[:,1].map(np.isreal).all():
            raise ValueError(("data must be a 2 column data.frame, with the"
                              "first column being a set of timestamps, and "
                              "the second coloumn being numeric values."))

        if (not (df.dtypes[0].type is np.datetime64)
            and not (df.dtypes[0].type is np.int64)):
            df = format_timestamp(df)

    if list(df.columns.values) != ["timestamp", "value"]:
        df.columns = ["timestamp", "value"]

    # Sanity check all input parameters
    if max_anoms > 0.49:
        length = len(df.value)
        raise ValueError(
            ("max_anoms must be less than 50% of "
             "the data points (max_anoms =%f data_points =%s).")
                         % (round(max_anoms * length, 0), length))

    if not direction in ['pos', 'neg', 'both']:
        raise ValueError("direction options are: pos | neg | both.")

    if not (0.01 <= alpha or alpha <= 0.1):
        if verbose:
            import warnings
            warnings.warn(("alpha is the statistical signifigance, "
                           "and is usually between 0.01 and 0.1"))

    if only_last and not only_last in ['day', 'hr']:
        raise ValueError("only_last must be either 'day' or 'hr'")

    if not threshold in [None,'med_max','p95','p99']:
        raise ValueError("threshold options are: None | med_max | p95 | p99")

    if not isinstance(e_value, bool):
        raise ValueError("e_value must be a boolean")

    if not isinstance(longterm, bool):
        raise ValueError("longterm must be a boolean")

    if piecewise_median_period_weeks < 2:
        raise ValueError(
            "piecewise_median_period_weeks must be at greater than 2 weeks")

    if not isinstance(plot, bool):
        raise ValueError("plot must be a boolean")

    if not isinstance(y_log, bool):
        raise ValueError("y_log must be a boolean")

    if not isinstance(xlabel, string_types):
        raise ValueError("xlabel must be a string")

    if not isinstance(ylabel, string_types):
        raise ValueError("ylabel must be a string")

    if title and not isinstance(title, string_types):
        raise ValueError("title must be a string")

    if not title:
        title = ''
    else:
        title = title + " : "

    gran = get_gran(df)

    if gran == "day":
        num_days_per_line = 7
        if isinstance(only_last, string_types) and only_last == 'hr':
            only_last = 'day'
    else:
        num_days_per_line = 1

    if gran == 'sec':
        df.timestamp = date_format(df.timestamp, "%Y-%m-%d %H:%M:00")
        df = format_timestamp(df.groupby('timestamp').aggregate(np.sum))

    # if the data is daily, then we need to bump
    # the period to weekly to get multiple examples
    gran_period = {
        'min': 1440,
        'hr': 24,
        'day': 7
    }
    period = gran_period.get(gran)
    period = 12
    if not period:
        raise ValueError('%s granularity detected. This is currently not supported.' % gran)
    num_obs = len(df.value)

    clamp = (1 / float(num_obs))
    if max_anoms < clamp:
        max_anoms = clamp

    if longterm:
        if gran == "day":
            num_obs_in_period = period * piecewise_median_period_weeks + 1
            num_days_in_period = 7 * piecewise_median_period_weeks + 1
        else:
            num_obs_in_period = period * 7 * piecewise_median_period_weeks
            num_days_in_period = 7 * piecewise_median_period_weeks

        last_date = df.timestamp.iloc[-1]

        all_data = []

        for j in range(0, len(df.timestamp), num_obs_in_period):
            start_date = df.timestamp.iloc[j]
            end_date = min(start_date
                           + datetime.timedelta(days=num_days_in_period),
                           df.timestamp.iloc[-1])

            # if there is at least 14 days left, subset it,
            # otherwise subset last_date - 14days
            if (end_date - start_date).days == num_days_in_period:
                sub_df = df[(df.timestamp >= start_date)
                            & (df.timestamp < end_date)]
            else:
                sub_df = df[(df.timestamp >
                     (last_date - datetime.timedelta(days=num_days_in_period)))
                    & (df.timestamp <= last_date)]
            all_data.append(sub_df)
    else:
        all_data = [df]

    all_anoms = DataFrame(columns=['timestamp', 'value'])
    seasonal_plus_trend = DataFrame(columns=['timestamp', 'value'])

    # Detect anomalies on all data (either entire data in one-pass,
    # or in 2 week blocks if longterm=TRUE)
    for i in range(len(all_data)):
        directions = {
            'pos': Direction(True, True),
            'neg': Direction(True, False),
            'both': Direction(False, True)
        }
        anomaly_direction = directions[direction]

        # detect_anoms actually performs the anomaly detection and
        # returns the results in a list containing the anomalies
        # as well as the decomposed components of the time series
        # for further analysis.

        s_h_esd_timestamps = detect_anoms_local(all_data[i], k=max_anoms, alpha=alpha,
                                          num_obs_per_period=period,
                                          use_decomp=True,
                                          one_tail=anomaly_direction.one_tail,
                                          upper_tail=anomaly_direction.upper_tail,
                                          verbose=verbose)

        # store decomposed components in local variable and overwrite
        # s_h_esd_timestamps to contain only the anom timestamps
        data_decomp = s_h_esd_timestamps['stl']
        s_h_esd_timestamps = s_h_esd_timestamps['anoms']

        # -- Step 3: Use detected anomaly timestamps to extract the actual
        # anomalies (timestamp and value) from the data
        if s_h_esd_timestamps:
            anoms = all_data[i][all_data[i].timestamp.isin(s_h_esd_timestamps)]
            sorter = s_h_esd_timestamps
            sorterIndex = dict(zip(sorter, range(len(sorter))))
            anoms['anoms_rank'] = anoms['timestamp'].map(sorterIndex)
            anoms.sort_values(['anoms_rank'], ascending = [True], inplace = True)
            anoms.drop('anoms_rank', 1, inplace = True)
        else:
            anoms = DataFrame(columns=['timestamp', 'value'])

        # Filter the anomalies using one of the thresholding functions if applicable
        if threshold:
            # Calculate daily max values
            periodic_maxes = df.groupby(
                df.timestamp.map(Timestamp.date)).aggregate(np.max).value

            # Calculate the threshold set by the user
            if threshold == 'med_max':
                thresh = periodic_maxes.median()
            elif threshold == 'p95':
                thresh = periodic_maxes.quantile(.95)
            elif threshold == 'p99':
                thresh = periodic_maxes.quantile(.99)

            # Remove any anoms below the threshold
            anoms = anoms[anoms.value >= thresh]

        all_anoms = all_anoms.append(anoms)
        seasonal_plus_trend = seasonal_plus_trend.append(data_decomp)

    # Cleanup potential duplicates
    try:
        all_anoms.drop_duplicates(subset=['timestamp'], inplace=True)
        seasonal_plus_trend.drop_duplicates(subset=['timestamp'], inplace=True)
    except TypeError:
        all_anoms.drop_duplicates(cols=['timestamp'], inplace=True)
        seasonal_plus_trend.drop_duplicates(cols=['timestamp'], inplace=True)

    # -- If only_last was set by the user,
    # create subset of the data that represent the most recent day
    if only_last:
        start_date = df.timestamp.iloc[-1] - datetime.timedelta(days=7)
        start_anoms = df.timestamp.iloc[-1] - datetime.timedelta(days=1)
        if gran is "day":
            breaks = 3 * 12
            num_days_per_line = 7
        else:
            if only_last == 'day':
                breaks = 12
            else:
                start_date = df.timestamp.iloc[-1] - datetime.timedelta(days=2)
                # truncate to days
                start_date = datetime.date(start_date.year,
                                           start_date.month, start_date.day)
                start_anoms = (df.timestamp.iloc[-1]
                               - datetime.timedelta(hours=1))
                breaks = 3

        # subset the last days worth of data
        x_subset_single_day = df[df.timestamp > start_anoms]
        # When plotting anoms for the last day only
        # we only show the previous weeks data
        x_subset_week = df[(df.timestamp <= start_anoms)
                           & (df.timestamp > start_date)]
        if len(all_anoms) > 0:
            all_anoms = all_anoms[all_anoms.timestamp >=
                                  x_subset_single_day.timestamp.iloc[0]]
        num_obs = len(x_subset_single_day.value)

    # Calculate number of anomalies as a percentage
    anom_pct = (len(df.value) / float(num_obs)) * 100

    if anom_pct == 0:
        return {
            "anoms": None,
            "plot": None
        }

    # The original R implementation handles plotting here.
    # Plotting is currently not implemented in this version.
    # if plot:
    #     plot_something()

    all_anoms.index = all_anoms.timestamp

    if e_value:
        d = {
            'timestamp': all_anoms.timestamp,
            'anoms': all_anoms.value,
            'expected_value': seasonal_plus_trend[
                seasonal_plus_trend.timestamp.isin(
                    all_anoms.timestamp)].value.values
        }
    else:
        d = {
            'timestamp': all_anoms.timestamp,
            'anoms': all_anoms.value
        }
    anoms = DataFrame(d, index=d['timestamp'].index)

    return {
        'anoms': anoms,
        'plot': None
    }
  








def detect_anoms_local(data, k=0.49, alpha=0.05, num_obs_per_period=None,
                 use_decomp=True, one_tail=True,
                 upper_tail=True, verbose=False):
    
    from pyculiarity.date_utils import format_timestamp
    from rstl import STL
    from itertools import groupby
    from math import trunc, sqrt
    from scipy.stats import t as student_t
    from statsmodels.robust.scale import mad
    import numpy as np
    import pandas as ps
    import statsmodels.api as sm
    import sys

    
    if num_obs_per_period is None:
        raise ValueError("must supply period length for time series decomposition")

    if list(data.columns.values) != ["timestamp", "value"]:
        data.columns = ["timestamp", "value"]

    num_obs = len(data)

    # Check to make sure we have at least two periods worth of data for anomaly context
    if num_obs < num_obs_per_period * 2:
        raise ValueError("Anom detection needs at least 2 periods worth of data")

    # Check if our timestamps are posix
    posix_timestamp = data.dtypes[0].type is np.datetime64

    # run length encode result of isnull, check for internal nulls
    if (len(list(map(lambda x: x[0], list(groupby(ps.isnull(
            ps.concat([ps.Series([np.nan]),
                       data.value,
                       ps.Series([np.nan])]))))))) > 3):
        raise ValueError("Data contains non-leading NAs. We suggest replacing NAs with interpolated values (see na.approx in Zoo package).")
    else:
        data = data.dropna()

    # -- Step 1: Decompose data. This returns a univarite remainder which will be used for anomaly detection. Optionally, we might NOT decompose.

    data = data.set_index('timestamp')

    # if not isinstance(data.index, ps.Int64Index):
    #     resample_period = {
    #         1440: 'T',
    #         24: 'H',
    #         7: 'D'
    #     }
    #     resample_period = resample_period.get(num_obs_per_period)
    #     if not resample_period:
    #         raise ValueError('Unsupported resample period: %d' % resample_period)
    #     data = data.resample(resample_period)


    decomp = STL(data.value, num_obs_per_period, "periodic", robust=True)
    
#     print('decomposition is', decomp)

    # Remove the seasonal component, and the median of the data to create the univariate remainder
    d = {
        'timestamp': data.index,
        'value': data.value - decomp.seasonal - decomp.trend   # data.value.median()
    }
    data = ps.DataFrame(d)

    p = {
        'timestamp': data.index,
        'value': ps.to_numeric(ps.Series(decomp.trend + decomp.seasonal).truncate())
    }
    data_decomp = ps.DataFrame(p)

    # Maximum number of outliers that S-H-ESD can detect (e.g. 49% of data)
    max_outliers = int(num_obs * k)

    if max_outliers == 0:
        raise ValueError("With longterm=TRUE, AnomalyDetection splits the data into 2 week periods by default. You have %d observations in a period, which is too few. Set a higher piecewise_median_period_weeks." % num_obs)

    ## Define values and vectors.
    n = len(data.timestamp)
    R_idx = list(range(max_outliers))

    num_anoms = 0

    # Compute test statistic until r=max_outliers values have been
    # removed from the sample.
    for i in range(1, max_outliers + 1):
        if one_tail:
            if upper_tail:
                ares = data.value - data.value.median()
            else:
                ares = data.value.median() - data.value
        else:
            ares = (data.value - data.value.median()).abs()

        # protect against constant time series
        data_sigma = mad(data.value)
        if data_sigma == 0:
            break

            
#         print('data is', data)    
            
        ares = ares / float(data_sigma)
        
#         print('ares is', ares)    
        

        R = ares.max()

        temp_max_idx = ares[ares == R].index.tolist()[0]

        R_idx[i - 1] = temp_max_idx

        data = data[data.index != R_idx[i - 1]]

        if one_tail:
            p = 1 - alpha / float(n - i + 1)
        else:
            p = 1 - alpha / float(2 * (n - i + 1))

        t = student_t.ppf(p, (n - i - 1))
        lam = t * (n - i) / float(sqrt((n - i - 1 + t**2) * (n - i + 1)))

        if R > lam:
            num_anoms = i

    if num_anoms > 0:
        R_idx = R_idx[:num_anoms]
    else:
        R_idx = None

    return {
        'anoms': R_idx,
        'stl': data_decomp
    }  












