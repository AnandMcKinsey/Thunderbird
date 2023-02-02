import forecast_engine.modelling.error_metrics.nodes as error_metrics



def linearregression_mv(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
    import sklearn
    from datetime import datetime
    from sklearn.linear_model import Ridge 
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error as mse
    import numpy as np
    import pandas as pd

    ti=pd.DataFrame(ti)
    months_to_forecast = h
    oos = ti[row_counter-1:row_counter-1+h]
    y=pd.Series(ti['Value'])[:-12]
    y_train=y[:len(y)-12]
    y_val=y[len(y)-12:]

    if months_to_forecast == 1: intervals=[3,4,5,6]
    if months_to_forecast > 1: intervals=[months_to_forecast,months_to_forecast+1,months_to_forecast+2,months_to_forecast+3]
    if months_to_forecast == 1: lags = 3
    if months_to_forecast > 1: lags = months_to_forecast

    lag_params=pd.DataFrame()
    for k in intervals:
        lagged_ti=pd.Series(ti.Value.shift(k),name='lag_'+str(k))
        lag_params=pd.concat([lag_params,lagged_ti],axis=1,sort=False)

    rolling_params=pd.DataFrame()
    for k in intervals:
          rolling_mean=pd.Series(ti.Value.fillna(0).rolling(k).mean(),name='rolling_mean_'+str(k))
          rolling_std=pd.Series(ti.Value.fillna(0).rolling(k).std(),name='rolling_std_'+str(k))
          rolling_quantile_75=pd.Series(ti.Value.fillna(0).rolling(k).quantile(.75,interpolation='midpoint'),name='rolling_quantile_75_'+str(k))
          rolling_quantile_25=pd.Series(ti.Value.fillna(0).rolling(k).quantile(.25,interpolation='midpoint'),name='rolling_quantile_25_'+str(k))
          rolling_params=pd.concat([rolling_params,rolling_mean,rolling_std,rolling_quantile_75,rolling_quantile_25],axis=1,sort=False)
        
    other_params=pd.DataFrame()
    month=pd.Series(pd.to_datetime(ti.index),name='month').dt.month
    year=pd.Series(pd.to_datetime(ti.index),name='year').dt.year
    yearmonth=pd.Series(year*100+month,name='yearmonth')
    quarter=pd.Series(pd.to_datetime(ti.index),name='quarter').dt.quarter
    other_params=pd.concat([other_params,month,quarter],axis=1,sort=False)
    other_params.index=ti.index

    other_Xs=ti[[x for x in ti.columns if x!='Value']].shift(lags)

    regressors=pd.concat([lag_params,other_params,other_Xs,rolling_params],axis=1).interpolate(method='linear', axis=0).ffill().bfill()
    X_train=regressors[regressors.index.isin(y_train.index)]
    X_val=regressors[regressors.index.isin(y_val.index)]
    select_oos = regressors[row_counter-1:row_counter-1+h]
    select_oos.replace(0, np.nan, inplace=True)
    select_oos = select_oos.interpolate(method='linear', axis=0).ffill().bfill()

    scalerx = StandardScaler()
    scalery = StandardScaler()
    X_train = scalerx.fit_transform(X_train.values.astype('float64'))
    y_train = scalery.fit_transform(pd.DataFrame(y_train).values.astype('float64'))
    X_test = scalerx.transform(X_val)



    ridgeparamlist = [0,0.001,0.01,0.1,1,2,5,8,10,20,50,80,100,1000,10000]
    rmselist = []
    rmselist_fullwindow = []

    y_predict_list = []



    for ridgeparam in ridgeparamlist:
      kf = KFold(n_splits=10)
      y_predict = []
      for train_index, valid_index in kf.split(X_train):
        X_train1, X_valid1 = X_train[train_index,:], X_train[valid_index,:]
        Y_train1 = y_train[train_index,]
        reg = Ridge(alpha=ridgeparam)
        reg.fit(X_train1, Y_train1)
        y_predict = y_predict + list(reg.predict(X_valid1))

      rmselist.append(error_metrics.wrmse(y_train[-n_splits*validation_window:],y_predict[-n_splits*validation_window:]))  
      rmselist_fullwindow.append(error_metrics.wrmse(y_train,y_predict))  
      y_predict_list.append(y_predict)
    best_ridge_param = ridgeparamlist[np.array(rmselist_fullwindow).argmin()]
    y_predict = y_predict_list[np.array(rmselist_fullwindow).argmin()]

    predictions = y_predict[-n_splits*validation_window:]
    forecast1=pd.concat([pd.Series(y.index[-n_splits*validation_window:]),pd.Series(scalery.inverse_transform(predictions).ravel()), pd.Series(['linearregression_mv']*n_splits*validation_window)], axis=1)
    forecast1.columns=['date_','forecast','method']      
    reg1 = Ridge(alpha=best_ridge_param)
    reg1.fit(X_train, y_train)

    forecast2=pd.concat([pd.Series(oos.index),pd.Series(reg1.predict(select_oos.values).ravel()),pd.Series(['linearregression_mv']*h)],axis=1)
    forecast2.columns=['date_','forecast','method']   


    var_coeff = pd.DataFrame()
    total_fit = reg1.predict(regressors[regressors.index.isin(y[:len(y)-12].index)].values) 
    total_fit = pd.concat([pd.Series(regressors[regressors.index.isin(y[:len(y)-12].index)].index), pd.Series(reg1.predict(regressors[regressors.index.isin(y[:len(y)-12].index)].values).ravel()) ,pd.Series(['linearregression_mv']*len(X_train))],axis=1) 
    total_fit.columns = ['date_','forecast','method']
    total_fit = pd.concat([total_fit,forecast2])
    feature_scores = {'linearregression_mv':(pd.Series(index=regressors.columns).sort_values(ascending=False))}




    return {'forecast':forecast1,'oos forecast':forecast2,'var_coeff':var_coeff,'fitted_values_all':total_fit,"feature_scores":feature_scores}

