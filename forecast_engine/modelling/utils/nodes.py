
import pandas as pd
import numpy as np


#Splitting time series


class TimeSeriesSplit:
  
  """
  Class for rolling-window cross validation to pick the best model to forecast a time series. Used inside every model to divide indexes in a time series into train and validation in every cross-validationfold.
  """
  
  def __init__(self, n_splits = 2, validation_window=3):
    
    """
    init function for the class.
    
    Parameters
    ----------
    n_splits : number of cross-validation folds
    
    validation_window : length of each cross-validation fold
    
    """
    
    self.n_splits = n_splits
    self.validation_window = validation_window
 
  def printx(self):
    
    """
    Function to print n_splits and validation_window
    """
    
    print(self.n_splits, self.validation_window)
    
  def split(self, series=None):
    
    """
    Function to split the time series into train and validation, given the size of the validation_window and n_splits.
    
    Parameters
    ----------
    series : time series to be split into folds for cross-validation
    
    Returns
    -------
    train : array of indexes for training period in every fold
    val : array of indexes for validation period in every fold
    
    """
    
    length = len(series)
    n_splits = self.n_splits
    validation_window = self.validation_window
    
    for i in range(self.n_splits,0,-1):
      if (length<i*validation_window):
        raise Exception('Length of time series is less than validation window: In class TimeSeriesSplit, Length={}, validation_range={}'.format(length, i*validation_window))
      train = range(length-i*validation_window)
      val = range(length-i*validation_window, length-(i-1)*validation_window)
      yield train, val



# Detect Seasonality


def get_seasonal_periods(tsdf):
    from heapq import nlargest
    tsdf = tsdf.reset_index()
    col = tsdf.iloc[:,0]
    n = len(col)
    
    if col.dtypes=='O':
      col=col.str.replace(" ","")
    try:
      col=pd.to_datetime(col,format="%Y-%m")
    except:
      col=pd.to_datetime(col,format="%Y-%m-%d")

    largest, second_largest = nlargest(2, col.drop_duplicates())
    gran = int(np.round(np.timedelta64(largest - second_largest) / np.timedelta64(1, 'D')))
    if gran >= 84:
        return 4
    elif gran >= 28:
        return 12
    elif gran >= 7:
        return 52
    elif gran >= 1:
        return 365
    else:
      raise Exception("No seasonality detected.")



# Data Transform

def get_ti_in_shape(ti):
  
  
  
  ti_series = ti[ti.index.str.contains(' - ') & ~(ti.index.str.contains('_'))]

  series_flag = ti_series[(ti_series.size-1)]
  ti_key = ti[~(ti.index.str.contains(' ')) & ~(ti.index.str.contains(' - '))]


  ti_vars =ti[(ti.index.str.contains(' ')) & (ti.index.str.contains('_'))]
  ti_vars = ti[(ti.index.str.contains(' ')) & (ti.index.str.contains('_'))]
  ti = pd.concat([ti_key,ti_series])


  ti_vars = pd.DataFrame(ti_vars).reset_index()
  ti_vars['month_id'] = pd.Series()
  ti_vars['driver_desc'] = pd.Series()

  if ti_vars.shape[0]>0:
      ti_vars['month_id'] = ti_vars['index'].str.split("_", expand=True)[0]
      ti_vars['driver_desc'] = ti_vars['index'].str.split("_", expand=True)[1]

  ti_vars.columns = ['index','value','month_id','driver_desc']
  drivers_local = ti_vars.drop(['index'],axis=1)
  drivers_local = drivers_local[['month_id','driver_desc','value']]

  #   print('drivers_local is', drivers_local)

  drivers_local = pd.DataFrame(drivers_local.pivot(index='month_id',columns='driver_desc'))
  drivers_local.columns = drivers_local.columns.droplevel()
  drivers_local = drivers_local.reset_index()
  drivers_local['month_id'] = drivers_local['month_id'].astype(str)
  drivers_local = drivers_local.dropna(axis=1, how = 'all')
  variable_list = pd.Series(drivers_local.columns[1:])


  ti=pd.DataFrame([ti]).T
  ti['Value']=ti.index
  ti=ti.reset_index(drop=True)
  ti=ti[0:(ti.shape[0]-1)]
  ti.columns=["Value","Header"]
  ti = ti[ti['Header'].str.contains(' - ')]
  ti.index=ti['Header']
  ti = ti.iloc[:,0]
  ti = pd.DataFrame(ti)
  series_index = ti.index

  if ti_vars.shape[0]>0:
    ti = pd.merge(ti,drivers_local,left_on=series_index,right_on=drivers_local['month_id'],how="outer")
    ti['Header'] = ti['key_0']
    ti = ti.drop(['month_id','key_0'],axis=1)
    cols = list(ti.columns)
    cols = [cols[-1]] + cols[:-1]
    ti = ti[cols]
    ti.sort_values('Header', inplace=True)
  else: 
      ti = ti.reset_index()
  
  return ti, ti_key, variable_list, series_flag




# Impute drive if any

def driver_imputation(ti):
  for cols in ti.columns[2:]:
    if (ti[cols]).isna().sum()>0:
      ti[cols]=ti[cols].astype('float64').interpolate(method='linear')
    first_ind=ti[cols].first_valid_index()   
    if first_ind is None:
      first_ind = 0
    if ti[cols].index[0]<first_ind:
      ti[cols].iloc[:first_ind]= ti[cols].iloc[first_ind]  
  ti = ti.sort_values('Header')
  return ti


def actual_imputation(ti):
  for cols in ti.columns[1:2]:
    if (ti[cols]).isna().sum()>0:
      ti[cols]=ti[cols].astype('float64').interpolate(method='linear')
    first_ind=ti[cols].first_valid_index()   
    if first_ind is None:
      first_ind = 0
    if ti[cols].index[0]<first_ind:
      ti[cols].iloc[:first_ind]= ti[cols].iloc[first_ind]  
  ti = ti.sort_values('Header')
  return ti