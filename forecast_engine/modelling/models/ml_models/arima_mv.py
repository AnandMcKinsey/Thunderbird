import pandas as pd
import forecast_engine.modelling.utils.nodes as utils





def arima_mv(ti_mv,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
    from pmdarima.arima import auto_arima
    from sklearn.metrics import mean_squared_error as mse
    from datetime import datetime
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import numpy as np
    
    #ti_mv=ti_mv.drop(['Market_Value'],axis=1)
    oos=ti_mv[row_counter-1:row_counter-1+h]
    ti_mv=ti_mv[:(row_counter-1)]
    ti_mv_orig=pd.concat([ti_mv,oos],axis=0)
    ti_mv_orig=ti_mv_orig.fillna(0)
    
    ti_mv=ti_mv.truncate(before=ti_mv.index[np.min(np.where(ti_mv['Value'].notnull())[0])])    
    min_series = 0#2*np.abs(np.min(ti_mv['Value']))+100
    
    ti_mv.loc[:,'Value']=ti_mv.loc[:,'Value']+min_series
    ti_mv_orig.loc[:,'Value']=ti_mv_orig.loc[:,'Value']+min_series
    y_actual=[]; y_pred=[]; rmse = []; map_error=[]
    forecast1=pd.DataFrame(columns=['date_','forecast','method'])
#     print(ti_mv)
  
    arima_noseason = auto_arima(ti_mv['Value'].values, exogenous=np.asarray(ti_mv.iloc[:,1:].fillna(0)), start_p=1, d=None, start_q=1, max_p=5, max_d=1, max_q=5, start_P=0, D=0, start_Q=0, max_P=1, max_D=1, max_Q=1, max_order=10, m=12, stepwise=True, seasonal_test='ch', n_jobs=1, enforce_invertibility=False, enforce_stationarity=False)
    aic_noseason = arima_noseason.aic()
#     if 'arima_mv' in debug_models:
#       print(arima_noseason.summary())
    
    if (ti_mv.shape[0]-n_splits*validation_window > 12):
      try:
        arima_season = auto_arima(ti_mv['Value'].values, exogenous=np.asarray(ti_mv.iloc[:,1:].fillna(0)),start_p=1, d=None, start_q=1, max_p=5, max_d=1, max_q=5, start_P=0, D=1, start_Q=0, max_P=1, max_D=1, max_Q=1, max_order=10, m=12, stepwise=True, seasonal_test='ch', n_jobs=1, enforce_invertibility=False, enforce_stationarity=False)
        aic_season = arima_season.aic()
      except:
        aic_season = 1e50;
        if 'arima_mv' in debug_models: print('ARIMA: Exception in seasonal differencing')
    
    if (aic_noseason <= aic_season):
      arima_model = arima_noseason
    else:
      arima_model = arima_season
    
    tscv = utils.TimeSeriesSplit(n_splits = n_splits, validation_window=validation_window)

    for train_index, test_index in tscv.split(ti_mv):
      cv_train, cv_test = ti_mv.iloc[train_index], ti_mv.iloc[test_index]
      #print('order=',arima_model.order,'\nseasonal_order=',arima_model.seasonal_order)
      d = SARIMAX(cv_train['Value'].values.astype('float64'), exog=np.asarray(cv_train.iloc[:,1:].fillna(0)), order=arima_model.order, seasonal_order=arima_model.seasonal_order, trend='c', enforce_invertibility=False, enforce_stationarity=False)
      res = d.fit()
      predictions = res.predict(start=len(cv_train), end=len(cv_train)+len(cv_test)-1, exog=np.asarray(ti_mv.iloc[len(cv_train):len(cv_train)+len(cv_test),1:].fillna(0)))
      
      true_values = cv_test['Value'].values
      y_actual=y_actual+list(true_values-min_series)
      y_pred=y_pred+list(predictions-min_series)
      forecast=pd.concat([pd.Series(cv_test.index),pd.Series(predictions-min_series), pd.Series(['arima_mv']*len(cv_test))],axis=1)
      forecast.columns=['date_','forecast','method']
      forecast1=pd.concat([forecast1,forecast],axis=0,ignore_index=True)
    
    d = SARIMAX(ti_mv['Value'].values.astype('float64'), exog=np.asarray(ti_mv.iloc[:,1:].fillna(0)), order=arima_model.order, seasonal_order=arima_model.seasonal_order, trend='c', enforce_invertibility=False, enforce_stationarity=False)
    res = d.fit()
    predictions=res.predict(start = len(ti_mv), end=len(ti_mv)+h-1, exog=np.asarray(ti_mv_orig.iloc[-len(oos):,1:].fillna(0)))
    forecast2=pd.concat([pd.Series(oos.index),pd.Series(predictions-min_series),pd.Series(['arima_mv']*h)],axis=1)
    forecast2.columns=['date_','forecast','method']
    
    forecast1['forecast']=forecast1['forecast'].astype('float64')  
    forecast2['forecast']=forecast2['forecast'].astype('float64')
    if (np.sum(forecast1['forecast'].isnull())) or (np.sum(forecast2['forecast'].isnull())) or np.sum(np.isinf(forecast1['forecast'])) or np.sum(np.isinf(forecast2['forecast'])):
      raise Exception('NaN in output')
    
    #Impact snippet
    var_len =  ti_mv.iloc[:,1:].shape[1]
    impact_df = pd.DataFrame()
    for month_ in range(0,len(oos)):
      var_coeff = pd.concat([pd.Series(['arima_mv']*var_len),pd.Series([oos.index[month_]]*var_len),pd.Series(ti_mv.iloc[:,1:].columns),pd.DataFrame((oos.iloc[month_,1:]*res.params[1:(1+var_len)]).values)],axis=1)
      var_coeff.columns=['Model','Date','Driver','Impact']
      rest_var = variable_list[~variable_list.isin(list(ti_mv.iloc[:,1:].columns))].tolist()
      rest_len = len(rest_var)
      rest_coeff = pd.concat([pd.Series(['arima_mv']*rest_len),pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len),pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff = pd.concat([var_coeff,rest_coeff],axis=0,ignore_index=True)
      impact_df=pd.concat([impact_df,var_coeff],axis=0)
#     print(impact_df)  
    var_coeff=impact_df.copy()
#     print(var_coeff)
    if aic_noseason > aic_season: 
      total_fit = res.predict(start=12, end=len(ti_mv)-1,exog=np.asarray(ti_mv_orig.iloc[0:len(ti_mv)-1,13:].fillna(0)))
      total_fit=pd.concat([pd.Series(ti_mv.index[12:]),pd.Series(total_fit-min_series),pd.Series(['arima_mv']*len(total_fit))],axis=1)
    else: 
      total_fit = res.predict(start=0, end=len(ti_mv)-1,exog=np.asarray(ti_mv_orig.iloc[0:len(ti_mv)-1,1:].fillna(0)))
      total_fit=pd.concat([pd.Series(ti_mv.index),pd.Series(total_fit-min_series),pd.Series(['arima_mv']*len(ti_mv))],axis=1)
    total_fit.columns = ['date_', 'forecast','method']
#     total_fit['forecast'] = total_fit['forecast']*scale_factor 
    
    return {'forecast':forecast1,'oos forecast':forecast2,'var_coeff':var_coeff,'fitted_values_all':total_fit}
