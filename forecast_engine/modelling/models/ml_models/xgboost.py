
import forecast_engine.modelling.utils.nodes as utils
import forecast_engine.modelling.error_metrics.nodes as error_metrics


def xgboost(ti,h,row_counter,debug_models,variable_list, n_splits, validation_window):
    import numpy as np
    from xgboost import XGBRegressor
    #from sklearn.model_selection import TimeSeriesSplit
    from sklearn.model_selection import GridSearchCV
    from sklearn.feature_selection import SelectFromModel
    from sklearn.metrics import mean_squared_error as mse
    from datetime import datetime
    import pandas as pd

  
    


    oos=ti[row_counter-1:row_counter-1+h]
    y=ti.copy()
    y_train=y[:len(ti)]
    y_val=y[len(ti):]
    intervals=[2,3,4,5,6,12]
    rolling_params=pd.DataFrame()
    for k in intervals:
      rolling_mean=pd.Series(y.rolling(k).mean(),name='rolling_mean_'+str(k))
      rolling_std=pd.Series(y.rolling(k).std(),name='rolling_std_'+str(k))
      rolling_quantile_75=pd.Series(y.rolling(k).quantile(.75,interpolation='midpoint'),name='rolling_quantile_75_'+str(k))
      rolling_quantile_25=pd.Series(y.rolling(k).quantile(.25,interpolation='midpoint'),name='rolling_quantile_25_'+str(k))
      rolling_params=pd.concat([rolling_params,rolling_mean,rolling_std,rolling_quantile_75,rolling_quantile_25],axis=1,sort=False)

    lag_params=pd.DataFrame()
    for k in intervals:
      lagged_ti=pd.Series(y.shift(k),name='lag_'+str(k))
      lag_params=pd.concat([lag_params,lagged_ti],axis=1,sort=False)
      
    other_params=pd.DataFrame()
    month=pd.Series(pd.to_datetime(ti.index),name='month').dt.month
    year=pd.Series(pd.to_datetime(ti.index),name='year').dt.year
    yearmonth=pd.Series(year*100+month,name='yearmonth')
    quarter=pd.Series(pd.to_datetime(ti.index),name='quarter').dt.quarter
    other_params=pd.concat([other_params,month,year,yearmonth,quarter],axis=1,sort=False)
    other_params.index=ti.index


    regressors=pd.concat([rolling_params,lag_params,other_params],axis=1)
    regressors=regressors.fillna(0)
    X_train=regressors[:len(regressors)]
    X_val=regressors[len(regressors):]

    model=XGBRegressor(objective='reg:squarederror')
    model.fit(X_train[:-12], y_train[:-12])


    thresholds=np.sort(model.feature_importances_)[::-1]

    error_matrix=[]
    for thresh in thresholds[0:5]:
      selection = SelectFromModel(model, threshold=thresh, prefit=True)
      select_X=selection.transform(regressors)
      #print(select_X)
      select_X_train=select_X[:len(select_X)]
      select_X_val=select_X[len(select_X):]

      model1=XGBRegressor(objective='reg:squarederror')
      model1.fit(select_X_train[:-12], y_train[:-12])
      y_pred = model1.predict(select_X_val)
      error = error_metrics.mape(y_val, y_pred)
      if error_metrics.mape==np.inf:  error=np.sqrt(mse(y_val, y_pred))
      error_matrix.append(error)
      
    min_err=error_matrix.index(min(error_matrix))
    selection = SelectFromModel(model, threshold=thresholds[min_err], prefit=True)
    select_X=selection.transform(regressors)
    #print(select_X)
    # select_X_train=select_X[:len(select_X)-h]
    #select_X_val=select_X[len(select_X)-h:]
    select_oos=select_X[row_counter-1:row_counter-1+h]
    select_X=select_X[:(row_counter-1)]
    ti=ti[:(row_counter-1)]

    y_actual=[]
    y_pred=[]
    rmse = []
    trmse=[]
    tmap_error=[]
    map_error=[]
    tscv = utils.TimeSeriesSplit(n_splits = 3, validation_window=3)
    forecast=pd.DataFrame(columns=['date_','forecast','method'])
    for train_index, test_index in tscv.split(select_X):
      cv_train, cv_test = select_X[train_index], select_X[test_index]
      gsearch = GridSearchCV(estimator=XGBRegressor(objective='reg:squarederror',subsample=0.05,n_jobs=-1,random_state=123),
                param_grid= {'max_depth' : [1,3,5,7,9],'learning_rate' : [.01,.03,0.1,.3,1],'min_child_weight' : [1,3,5,7,9]},
                cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
      gsearch.fit(cv_train.astype('float'), y_train[:len(cv_train)].astype('float'))
      predictions = gsearch.predict(cv_test)
      train_pred=gsearch.predict(cv_train)
      train_true_values=ti[:len(cv_train)]
      true_values=ti[len(cv_train):len(cv_train)+len(cv_test)]
      y_actual=y_actual+list(true_values)
      y_pred=y_pred+list(predictions)
      trmse.append(error_metrics.wrmse(train_true_values, train_pred))
      tmap_error.append(error_metrics.wmape(train_true_values,train_pred))
      # rmse.append(wrmse(true_values, y_pred))
      # map_error.append(wmape(true_values,y_pred))
      forecast1=pd.concat([pd.Series(ti.index[len(cv_train):len(cv_train)+len(cv_test)]),pd.Series(gsearch.predict(cv_test)),pd.Series(['xgboost']*len(cv_test))],axis=1)
      forecast1.columns=['date_','forecast','method']
      forecast=pd.concat([forecast,forecast1],axis=0,ignore_index=True)
    # model1=XGBRegressor(objective='reg:squarederror',booster='dart',subsample=0.1,n_jobs=-1,random_state=123).fit(select_X, ti)
    model1=(gsearch.best_estimator_).fit(select_X,ti)
    #print(model1.predict(select_oos))
    forecast2=pd.concat([pd.Series(oos.index),pd.Series(model1.predict(select_oos).ravel()),pd.Series(['xgboost']*h)],axis=1)
    forecast2.columns=['date_','forecast','method']
    # forecast=pd.concat([forecast,forecast2],axis=0,ignore_index=True)

    train_accuracy={'RMSE':np.mean(trmse),'MAPE':np.mean(tmap_error)}
    forecast.columns=['date_','forecast','method']
    var_coeff = pd.DataFrame()
    total_fit = y_pred 
    return {'forecast':forecast,'oos forecast':forecast2,'var_coeff':var_coeff,'fitted_values_all':total_fit}