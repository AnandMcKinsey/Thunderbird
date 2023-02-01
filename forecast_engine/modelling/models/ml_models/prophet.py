import forecast_engine.modelling.utils.nodes as utils


def prophet(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
    from fbprophet import Prophet
    from sklearn.metrics import mean_squared_error as mse
    from datetime import date
    import numpy as np
    import pandas as pd
    import logging
    logging.getLogger('fbprophet').setLevel(logging.WARNING)
    changepoint_prior_scale = 0.1
    
    oos=ti[row_counter-1:row_counter-1+h]
    ti=ti[:(row_counter-1)]     
    ti=ti.truncate(before=ti.index[np.min(np.where(ti.notnull())[0])])
    min_series =2*np.abs(np.min(ti))+100
    ti=ti+min_series
    ti=ti.reset_index()
    ti.columns=['ds','y']
    ti['ds'] = ti['ds'].map(lambda x: date(year=int(x.split('-')[0]), month = int(x.split('-')[1]), day = 1))
    
    y_actual=[]; y_pred=[]; rmse = []; map_error=[]; fitted_train=[]
    forecast1=pd.DataFrame(columns=['date_','forecast','method'])

    tscv = utils.TimeSeriesSplit(n_splits = n_splits, validation_window=validation_window)
    for train_index, test_index in tscv.split(ti):
      cv_train, cv_test = ti.iloc[train_index], ti.iloc[test_index]
      m = Prophet(changepoint_prior_scale=changepoint_prior_scale,yearly_seasonality=False,weekly_seasonality=False,daily_seasonality=False)
      m.add_seasonality(name='yearly', period=365.25, fourier_order=5, prior_scale=0.1)
      
      m.fit(cv_train)
      future = m.make_future_dataframe(periods=len(cv_test['y']),freq='MS')
      predictions=m.predict(future)['yhat'].iloc[-len(cv_test['y']):].values
      true_values = cv_test['y'].values
      y_actual=y_actual+list(true_values-min_series)
      y_pred=y_pred+list(predictions-min_series)
      cv_train.index = cv_train['ds'].map(lambda x: x.strftime("%Y - %m"))
      cv_test.index = cv_test['ds'].map(lambda x: x.strftime("%Y - %m"))
      forecast=pd.concat([pd.Series(cv_test.index),pd.Series(predictions-min_series), pd.Series(['prophet']*len(cv_test))],axis=1)
      forecast.columns=['date_','forecast','method']
      forecast1=pd.concat([forecast1,forecast],axis=0,ignore_index=True)
      
      train_fit = m.predict(future)['yhat'].iloc[:-len(cv_test)].values - min_series
      fitted_train1=pd.concat([pd.Series(cv_train.index) ,pd.Series(train_fit) ,pd.Series(['prophet']*len(train_fit) ,name='method')],axis=1)
      fitted_train1.columns=['date_','forecast','method']
      fitted_train.append(fitted_train1)
      
    m = Prophet(changepoint_prior_scale=changepoint_prior_scale,yearly_seasonality=False,weekly_seasonality=False,daily_seasonality=False)
    m.add_seasonality(name='yearly', period=365.25, fourier_order=5, prior_scale=0.1)
    m.fit(ti)
    future=m.make_future_dataframe(periods=len(oos),freq='MS')
    predictions=m.predict(future)['yhat'].iloc[-len(oos):].values
    forecast2=pd.concat([pd.Series(oos.index),pd.Series(predictions-min_series),pd.Series(['prophet']*h)],axis=1)
    forecast2.columns=['date_','forecast','method']

    forecast1['forecast']=forecast1['forecast'].astype('float64')  
    forecast2['forecast']=forecast2['forecast'].astype('float64')

    #Impact snippet
    rest_var = variable_list.tolist()
    rest_len = len(rest_var)
    var_coeff = pd.DataFrame()
    for month_ in range(0,len(oos)):
      rest_coeff = pd.concat([pd.Series(['prophet']*rest_len), pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len), pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff=pd.concat([var_coeff,rest_coeff],axis=0)
    
    if (np.sum(forecast1['forecast'].isnull())) or (np.sum(forecast2['forecast'].isnull())) or np.sum(np.isinf(forecast1['forecast'])) or np.sum(np.isinf(forecast2['forecast'])):
      raise Exception('NaN in output')
    
    total_fit=m.predict(future)['yhat'].iloc[:-len(oos)].values
    total_fit=pd.concat([ti['ds'].map(lambda x: x.strftime("%Y - %m")),pd.Series(total_fit-min_series),pd.Series(['prophet']*len(ti))],axis=1)
    total_fit.columns = ['date_', 'forecast','method']
    
    return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}