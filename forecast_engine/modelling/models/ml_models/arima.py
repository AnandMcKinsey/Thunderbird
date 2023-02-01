import numpy as np
import pandas as pd
import forecast_engine.modelling.utils.nodes as utils



def arima(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
    from pmdarima import auto_arima
    from sklearn.metrics import mean_squared_error as mse
    from datetime import datetime
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    # import pickle 
    
#     if debug_flag==1:
#       with open(ti_pickle_path, 'wb') as f:  # Python 3: open(..., 'wb')
#         pickle.dump([ti, h, row_counter, n_splits, validation_window], f)    
    
    oos=ti[row_counter-1:row_counter-1+h]
    ti=ti[:(row_counter-1)]
    ti=ti.truncate(before=ti.index[np.min(np.where(ti.notnull())[0])])
    seasonal_periods = utils.get_seasonal_periods(ti)
    #print(ti)
    min_ = np.min(ti.values)
    max_ = np.max(ti.values)
    scale_factor = np.abs(max_-min_)
    ti = ti/scale_factor
    min_series =2*np.abs(np.min(ti))+1e-5
    ti=ti+min_series
    
    y_actual=[]; y_pred=[]; rmse = []; map_error=[] 
    forecast1=pd.DataFrame(columns=['date_','forecast','method'])
    
    arima_noseason = auto_arima(ti.values.astype('float64'), start_p=2, d=None, start_q=2, max_p=5, max_d=1, max_q=5, start_P=0, D=0, start_Q=0, max_P=1, max_D=1, max_Q=1, max_order=10, m=seasonal_periods, stepwise=True, seasonal_test='ch', n_jobs=1)
    aic_noseason = arima_noseason.aic()
    
    try:
      arima_season = auto_arima(ti.values.astype('float64'), start_p=2, d=None, start_q=2, max_p=5, max_d=1, max_q=5, start_P=0, D=1, start_Q=0, max_P=1, max_D=1, max_Q=1, max_order=10, m=seasonal_periods, stepwise=True, seasonal_test='ch', n_jobs=1)
      aic_season = arima_season.aic()
    except:
      aic_season = 1e50;
      if 'arima' in debug_models: print('ARIMA: Exception in seasonal differencing')
      
#     print(aic_noseason, aic_season)
    
    if (aic_noseason <= aic_season):
      arima_model = arima_noseason
    else:
      arima_model = arima_season
      
    tscv = utils.TimeSeriesSplit(n_splits = n_splits, validation_window=validation_window)
    fitted_train=[]
    for train_index, test_index in tscv.split(ti):
      cv_train, cv_test = ti.iloc[train_index], ti.iloc[test_index]
      d = SARIMAX(cv_train.values.astype('float64'), order=arima_model.order, seasonal_order=arima_model.seasonal_order, trend='c')
      res = d.fit()
      #print(aic_noseason,aic_season)

      predictions = res.predict(start=len(cv_train), end=len(cv_train)+len(cv_test)-1)
      true_values = cv_test.values
      y_actual=y_actual+list((true_values-min_series)*scale_factor)
      y_pred=y_pred+list((predictions-min_series)*scale_factor)
      forecast=pd.concat([pd.Series(cv_test.index),pd.Series(predictions-min_series), pd.Series(['arima']*len(cv_test))],axis=1)
      forecast.columns=['date_','forecast','method']
      forecast1=pd.concat([forecast1,forecast],axis=0,ignore_index=True)
      
      if aic_noseason > aic_season: 
        train_fit = list((res.predict(start=train_index[seasonal_periods], end = train_index[-1]) - min_series)*scale_factor)
        fitted_train1=pd.concat([pd.Series(ti.index[train_index[seasonal_periods]:test_index[0]]) ,pd.Series(train_fit) ,pd.Series(['arima']*len(train_fit) ,name='method')],axis=1)
      else: 
        train_fit = list((res.predict(start=train_index[0], end = train_index[-1]) - min_series)*scale_factor)
        fitted_train1=pd.concat([pd.Series(ti.index[train_index[0]:test_index[0]]) ,pd.Series(train_fit) ,pd.Series(['arima']*len(train_fit) ,name='method')],axis=1)
      fitted_train1.columns=['date_','forecast','method']
      fitted_train.append(fitted_train1)
      #print('fitted_train every iter:\n',fitted_train)
      #prev_index = train_index[-1]+1
    
    d = SARIMAX(ti.values.astype('float64'), order=arima_model.order, seasonal_order=arima_model.seasonal_order, trend='c')
    res = d.fit()
    predictions=res.predict(start = len(ti), end=len(ti)+h-1)
    forecast2=pd.concat([pd.Series(oos.index),pd.Series(predictions-min_series),pd.Series(['arima']*h)],axis=1)
    forecast2.columns=['date_','forecast','method']

    forecast1['forecast'] = forecast1['forecast']*scale_factor
    forecast2['forecast'] = forecast2['forecast']*scale_factor
    forecast1['forecast']=forecast1['forecast'].astype('float64')  
    forecast2['forecast']=forecast2['forecast'].astype('float64')
#     print('forecast1:\n',forecast1['forecast'])
#     print('forecast2:\n',forecast2['forecast'])

    #Impact snippet
    rest_var = variable_list.tolist()
    rest_len = len(rest_var)
    var_coeff = pd.DataFrame()
    for month_ in range(0,len(oos)):
      rest_coeff = pd.concat([pd.Series(['arima']*rest_len), pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len), pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff=pd.concat([var_coeff,rest_coeff],axis=0)
    
    if (np.sum(forecast1['forecast'].isnull())) or (np.sum(forecast2['forecast'].isnull())) or np.sum(np.isinf(forecast1['forecast'])) or np.sum(np.isinf(forecast2['forecast'])):
      raise Exception('NaN in output')

    if aic_noseason > aic_season: 
      total_fit = res.predict(start=seasonal_periods, end=len(ti)-1)
      total_fit=pd.concat([pd.Series(ti.index[seasonal_periods:]),pd.Series(total_fit-min_series), pd.Series(['arima']*len(total_fit))],axis=1)
    else: 
      total_fit = res.predict(start=0, end=len(ti)-1)
      total_fit=pd.concat([pd.Series(ti.index),pd.Series(total_fit-min_series),pd.Series(['arima']*len(ti))],axis=1)
    total_fit.columns = ['date_', 'forecast','method']
    total_fit['forecast'] = total_fit['forecast']*scale_factor 
    
#     print('total_fit:\n',total_fit)
#     print('forecast1:\n',forecast1)
#     print('forecast2:\n',forecast2)
    
    if 'arima' in debug_models: 
      print(forecast1)
      print(forecast2)
  
    return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}