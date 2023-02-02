import forecast_engine.modelling.utils.nodes as utils


def rw_drift(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
    from pmdarima.arima import auto_arima
    from sklearn.metrics import mean_squared_error as mse
    from datetime import datetime
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import numpy as np
    import pandas as pd

    oos=ti[row_counter-1:row_counter-1+h]
    ti=ti[:(row_counter-1)]
    ti=ti.truncate(before=ti.index[np.min(np.where(ti.notnull())[0])])
    min_ = np.min(ti.values)
    max_ = np.max(ti.values)
    scale_factor = np.abs(max_-min_)
    ti = ti/scale_factor
    min_series =2*np.abs(np.min(ti))+1e-5
    ti=ti+min_series
    
    y_actual=[]; y_pred=[]; rmse = []; map_error=[]; fitted_train=[]
    forecast1=pd.DataFrame(columns=['date_','forecast','method'])
      
    tscv = utils.TimeSeriesSplit(n_splits = n_splits, validation_window=validation_window)
    for train_index, test_index in tscv.split(ti):
      cv_train, cv_test = ti.iloc[train_index], ti.iloc[test_index]
      np.random.seed(1234)
      fit = auto_arima(cv_train, start_p=0, d=1, start_q=0, max_p=0, max_d=1, max_q=0, max_order=0, seasonal=False, stationary=False, stepwise=True, n_jobs=-1)
      predictions = fit.predict(n_periods=len(cv_test.index))
      true_values = cv_test.values
      y_actual=y_actual+list((true_values-min_series)*scale_factor)
      y_pred=y_pred+list((predictions-min_series)*scale_factor)
      forecast=pd.concat([pd.Series(cv_test.index),pd.Series(predictions-min_series), pd.Series(['rw_drift']*len(cv_test))],axis=1)
      forecast.columns=['date_','forecast','method']
      forecast1=pd.concat([forecast1,forecast],axis=0,ignore_index=True)
      
      train_fit = list((fit.predict_in_sample(start=train_index[1],end=train_index[-1]) - min_series)*scale_factor)
      fitted_train1=pd.concat([pd.Series(ti.index[train_index[1]:test_index[0]]) ,pd.Series(train_fit) ,pd.Series(['rw_drift']*len(train_fit) ,name='method')],axis=1)
      fitted_train1.columns=['date_','forecast','method']
      fitted_train.append(fitted_train1)
   # print(fitted_train)  
    np.random.seed(1234)
    fit2 = auto_arima(ti.values, start_p=0, d=1, start_q=0, max_p=0, max_d=1, max_q=0, max_order=0, seasonal=False, stationary=False, stepwise=True, n_jobs=-1)
    predictions=fit2.predict(n_periods=len(oos))
    forecast2=pd.concat([pd.Series(oos.index),pd.Series(predictions-min_series),pd.Series(['rw_drift']*h)],axis=1)
    forecast2.columns=['date_','forecast','method']

    forecast1['forecast'] = forecast1['forecast']*scale_factor
    forecast2['forecast'] = forecast2['forecast']*scale_factor
    forecast1['forecast']=forecast1['forecast'].astype('float64')  
    forecast2['forecast']=forecast2['forecast'].astype('float64')

    #Impact snippet
    rest_var = variable_list.tolist()
    rest_len = len(rest_var)
    var_coeff = pd.DataFrame()
    for month_ in range(0,len(oos)):
      rest_coeff = pd.concat([pd.Series(['rw_drift']*rest_len), pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len), pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff=pd.concat([var_coeff,rest_coeff],axis=0)
    
    if (np.sum(forecast1['forecast'].isnull())) or (np.sum(forecast2['forecast'].isnull())) or np.sum(np.isinf(forecast1['forecast'])) or np.sum(np.isinf(forecast2['forecast'])):
      raise Exception('NaN in output')
    
    if 'rw_drift' in debug_models: 
      print(forecast1)
      print(forecast2)
    
    total_fit = list(fit.predict_in_sample(start=1,end=len(ti)-1))
    total_fit=pd.concat([pd.Series(ti.index[1:]),pd.Series(total_fit-min_series),pd.Series(['rw_drift']*(len(ti)-1))],axis=1)
    total_fit.columns = ['date_', 'forecast','method']
    total_fit['forecast'] = total_fit['forecast']*scale_factor 
    
    return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}