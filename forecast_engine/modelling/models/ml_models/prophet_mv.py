import forecast_engine.modelling.utils.nodes as utils



def prophet_mv(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
    from fbprophet import Prophet
    from sklearn.metrics import mean_squared_error as mse
    from datetime import date
    import numpy as np
    import logging
    import pandas as pd
    logging.getLogger('fbprophet').setLevel(logging.WARNING)
    changepoint_prior_scale = 0.1
    ti=ti.reset_index()
    ti.index=ti.iloc[:,0]

    min_series =0#2*np.abs(np.min(ti['Value']))+100
    ti['Value']=ti['Value']+min_series

    ti.columns=[*['ds','y'],*ti.columns[2:]]
    ti['ds'] = ti['ds'].map(lambda x: date(year=int(x.split('-')[0]), month = int(x.split('-')[1]), day = 1))
    oos=ti[row_counter-1:row_counter-1+h]
    ti=ti[:(row_counter-1)]
    ti_orig=pd.concat([ti,oos],axis=0)
    ti_orig=ti_orig.fillna(0)
    ti=ti.truncate(before=ti.index[np.min(np.where(ti['y'].notnull())[0])])
    # print('ti in prophet:\n',ti)

    y_actual=[]; y_pred=[]; rmse = []; map_error=[]
    forecast1=pd.DataFrame(columns=['date_','forecast','method'])

    tscv = utils.TimeSeriesSplit(n_splits = n_splits, validation_window=validation_window)
    for train_index, test_index in tscv.split(ti):
      cv_train, cv_test = ti.iloc[train_index], ti.iloc[test_index]
      m = Prophet(changepoint_prior_scale=changepoint_prior_scale,yearly_seasonality=False,weekly_seasonality=False,daily_seasonality=False)
      m.add_seasonality(name='yearly', period=365.25, fourier_order=5, prior_scale=0.1)
      for col in cv_train.columns[2:]:
        m.add_regressor(col)
      m.fit(cv_train)
      future = m.make_future_dataframe(periods=len(cv_test['y']),freq='MS')
      future.set_index(['ds'], drop=False, inplace=True)
      future.index.name = 'Header'
      future['ds'] = future.index.map(lambda x: x.strftime("%Y - %m"))
      future.set_index(['ds'], inplace=True)

      future = pd.concat([future,ti.iloc[:len(cv_train)+len(cv_test),2:]],axis=1)
      future.index.name = 'ds'
      future.reset_index(inplace=True)
      predictions=m.predict(future)['yhat'].iloc[-len(cv_test['y']):].values
      # print(m.predict(future)[[x for x in ti.columns[2:]]].head())
      true_values = cv_test['y'].values
      y_actual=y_actual+list(true_values-min_series)
      y_pred=y_pred+list(predictions-min_series)
      cv_test.index = cv_test['ds'].map(lambda x: x.strftime("%Y - %m"))
      forecast=pd.concat([pd.Series(cv_test.index),pd.Series(predictions-min_series), pd.Series(['prophet_mv']*len(cv_test))],axis=1)
      forecast.columns=['date_','forecast','method']
      forecast1=pd.concat([forecast1,forecast],axis=0,ignore_index=True)

    m = Prophet(changepoint_prior_scale=changepoint_prior_scale,yearly_seasonality=False,weekly_seasonality=False,daily_seasonality=False)
    m.add_seasonality(name='yearly', period=365.25, fourier_order=5, prior_scale=0.1)
    for col in ti.columns[2:]:
      m.add_regressor(col)
    m.fit(ti)
    future=m.make_future_dataframe(periods=len(oos),freq='MS')
    future.set_index(['ds'], drop=True, inplace=True)
    future.index.name = 'Header'
    future['ds'] = future.index 
    future=pd.concat([future,ti.iloc[:len(ti)+len(oos),2:]],axis=1)
    future = future.iloc[-len(oos):,1:]=ti_orig.iloc[-len(oos):,2:]
    future.index.name = 'ds'
    future.reset_index(inplace=True)

    predictions=m.predict(future)['yhat'].iloc[-len(oos):].values
    oos.index = oos['ds'].map(lambda x: x.strftime("%Y - %m"))
    forecast2=pd.concat([pd.Series(oos.index),pd.Series(predictions-min_series),pd.Series(['prophet_mv']*h)],axis=1)
    forecast2.columns=['date_','forecast','method']
    forecast1['forecast']=forecast1['forecast'].astype('float64')  
    forecast2['forecast']=forecast2['forecast'].astype('float64')
    if (np.sum(forecast1['forecast'].isnull())) or (np.sum(forecast2['forecast'].isnull())) or np.sum(np.isinf(forecast1['forecast'])) or np.sum(np.isinf(forecast2['forecast'])):
      raise Exception('NaN in output')
      
    #Impact snippet
    var_len =  ti.iloc[:,2:].shape[1]
    impact_df = pd.DataFrame()
    for month_ in range(0,len(oos)):
      reg_cols=[]
      for i in ti.columns[2:]:
        reg_cols.append(np.where(m.train_component_cols[i]==1)[0][0])
      var_coeff = pd.concat([pd.Series(['prophet_mv']*var_len),pd.Series([oos.index[month_]]*var_len), pd.Series(ti.iloc[:,2:].columns), pd.DataFrame(((m.params['beta'][0][reg_cols]/np.array([m.extra_regressors[i]['std'] for i in ti.iloc[:,2:].columns])*ti['y'].abs().max())*oos.iloc[month_,2:]).values)],axis=1)
      if 'prophet_mv' in debug_models: print(var_coeff)
      var_coeff.columns=['Model','Date','Driver','Impact']
      rest_var = variable_list[~variable_list.isin(list(ti.iloc[:,1:].columns))].tolist()
      rest_len = len(rest_var)
      rest_coeff = pd.concat([pd.Series(['prophet_mv']*rest_len),pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len),pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff = pd.concat([var_coeff,rest_coeff],axis=0,ignore_index=True)
      impact_df=pd.concat([impact_df,var_coeff],axis=0)
    #     print(impact_df)  
    var_coeff=impact_df.copy()
    #     print(var_coeff)
    total_fit=m.predict(future)['yhat'].iloc[:-len(oos)].values
    date_ = ti['ds'].map(lambda x: x.strftime("%Y - %m")).reset_index(drop=True)
    total_fit=pd.concat([date_, pd.Series(total_fit-min_series), pd.Series(['prophet_mv']*len(ti))], axis=1)
    total_fit.columns = ['date_', 'forecast','method']
    feature_scores = {'prophet_mv':(pd.Series(index=ti.iloc[:,1:].columns).sort_values(ascending=False))}
    return {'forecast':forecast,'oos forecast':forecast2,'var_coeff':var_coeff,'fitted_values_all':total_fit,"feature_scores":feature_scores}

