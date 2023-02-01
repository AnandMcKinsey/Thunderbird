
import forecast_engine.modelling.utils.nodes as utils


def garch(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
    from arch import arch_model
    from sklearn.metrics import mean_squared_error as mse
    from datetime import datetime
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
      am = arch_model(cv_train.values.astype(float), mean = 'ARX', lags=[1,2,3,4,5,6,12], vol='GARCH',p=1, q=1, dist='normal')
      fit = am.fit(update_freq=0, disp='off')
      if (fit.convergence_flag!=0):
        raise Exception('optimization has not converged')
      predictions = fit.forecast(start=15, horizon=validation_window)
      predictions = predictions.mean.iloc[-1,:].values
      true_values = cv_test.values
      y_actual=y_actual+list((true_values-min_series)*scale_factor)
      y_pred=y_pred+list((predictions-min_series)*scale_factor)
      forecast=pd.concat([pd.Series(cv_test.index),pd.Series(predictions-min_series), pd.Series(['garch']*len(cv_test))],axis=1)
      forecast.columns=['date_','forecast','method']
      forecast1=pd.concat([forecast1,forecast],axis=0,ignore_index=True)

      train_fit = np.array(cv_train - fit.resid - min_series)
      first_non_na = pd.Series(train_fit).first_valid_index()
      fitted_train1=pd.concat([pd.Series(ti.index[train_index[first_non_na]:test_index[0]]) ,pd.Series(train_fit[first_non_na:]*scale_factor) ,pd.Series(['garch']*len(train_fit[first_non_na:]) ,name='method')],axis=1)
      fitted_train1.columns=['date_','forecast','method']
      fitted_train.append(fitted_train1)
#     print(np.array(cv_train))
#     print(np.array(cv_test))
    #print(fit.dep_var)
   # print(fitted_train)
    np.random.seed(1234)
    am = arch_model(ti.values.astype(float), mean = 'ARX', lags=[1,2,3,4,5,6,12], vol='GARCH',p=1, q=1, dist='normal')
    fit2 = am.fit(update_freq=0, disp='off')
    if (fit2.convergence_flag!=0):
      raise Exception('optimization has not converged')    
    predictions = fit2.forecast(start=15, horizon=h)
    predictions = predictions.mean.iloc[-1,:].values
    forecast2=pd.concat([pd.Series(oos.index),pd.Series(predictions-min_series),pd.Series(['garch']*h)],axis=1)
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
      rest_coeff = pd.concat([pd.Series(['garch']*rest_len), pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len), pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff=pd.concat([var_coeff,rest_coeff],axis=0)
    
    if (np.sum(forecast1['forecast'].isnull())) or (np.sum(forecast2['forecast'].isnull())) or np.sum(np.isinf(forecast1['forecast'])) or np.sum(np.isinf(forecast2['forecast'])):
      raise Exception('NaN in output')
    
    total_fit = np.array(ti - fit2.resid - min_series)
    first_non_na = pd.Series(total_fit).first_valid_index()
    total_fit=pd.concat([pd.Series(ti.index[first_non_na:]),pd.Series(total_fit[first_non_na:]),pd.Series(['garch']*(len(ti[first_non_na:])))],axis=1)
    total_fit.columns = ['date_', 'forecast','method']
    total_fit['forecast'] = total_fit['forecast']*scale_factor 
    
    return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}