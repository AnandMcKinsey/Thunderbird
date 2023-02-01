import forecast_engine.modelling.utils.nodes as utils
import forecast_engine.modelling.error_metrics.nodes as error_metrics


def lgbm(ti,oos_length,row_counter,debug_models,variable_list,n_splits,validation_window):
  import numpy as np
  from lightgbm import LGBMRegressor
  #from sklearn.model_selection import TimeSeriesSplit
  from sklearn.model_selection import GridSearchCV
  from sklearn.feature_selection import SelectFromModel
  from sklearn.metrics import mean_squared_error as mse
  from datetime import datetime
  import pandas as pd
  
  oos=ti[row_counter-1:row_counter-1+h]
  ti=ti[:(row_counter-1)]
  ti=pd.concat([ti,oos],axis=0)
  ti=ti.truncate(before=ti.index[np.min(np.where(ti.notnull())[0])])
#   print('ti initially:\n',ti)
  ti=ti.astype('float64') 
  y_train=pd.Series(ti[:len(ti)-len(oos)-6],name='Value')
  y_val=pd.Series(ti[(len(ti)-len(oos)-6):(-len(oos))],name='Value')
#   print('y_train initially:\n',y_train)
#   print('y_val initially:\n',y_val)
  intervals=[4,12]
  rolling_params=pd.DataFrame()
  for k in intervals:
    rolling_mean=pd.Series(ti.rolling(k).mean(),name='rolling_mean_'+str(k))
    rolling_std=pd.Series(ti.rolling(k).std(),name='rolling_std_'+str(k))
    rolling_quantile_75=pd.Series(ti.rolling(k).quantile(.75,interpolation='midpoint'),name='rolling_quantile_75_'+str(k))
    rolling_quantile_25=pd.Series(ti.rolling(k).quantile(.25,interpolation='midpoint'),name='rolling_quantile_25_'+str(k))
    rolling_params=pd.concat([rolling_params,rolling_mean,rolling_std,rolling_quantile_75,rolling_quantile_25],axis=1,sort=False)

  lag_params=pd.DataFrame()
  for k in intervals:
    lagged_ti=pd.Series(ti.shift(k),name='lag_'+str(k))
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
  regressors_oos=regressors[-len(oos):]
  regressors=regressors[:-len(oos)]
  X_train=regressors[:len(regressors)-6]
  X_val=regressors[len(regressors)-6:]
  
  model1 = LGBMRegressor(n_jobs=-1,seed=123)
  model1.fit(X_train, y_train)
  
  thresholds=np.sort(model1.booster_.feature_importance())[::-1]
  
  error_matrix=[]
  for thresh in thresholds[0:5]:
    selection = SelectFromModel(model1, threshold=thresh, prefit=True)
    select_X=selection.transform(regressors)
    select_X_train=select_X[:len(select_X)-6]
    select_X_val=select_X[len(select_X)-6:]

    model = LGBMRegressor(n_jobs=-1,seed=123)
    model.fit(select_X_train, y_train)
    y_pred = model.predict(select_X_val)
#     print(y_val)
#     print('y_pred:\n',y_pred)
    error = error_metrics.wrmse(y_val, y_pred)
   # if mape==np.inf:  error=np.sqrt(mse(y_val, y_pred))
    error_matrix.append(error)
  y_train=pd.concat([y_train,y_val],axis=0)
  min_err=error_matrix.index(np.min(error_matrix))
  selection = SelectFromModel(model1, threshold=thresholds[min_err], prefit=True)
  select_X=selection.transform(regressors)
  #select_X_train=select_X[:len(select_X)-h]
  #select_X_val=select_X[len(select_X)-h:]
  select_X_oos=selection.transform(regressors_oos)
  
  #select_oos=select_X[row_counter-1:row_counter-1+h]
  select_X=select_X[:(row_counter-1)]
  ti=ti[:(row_counter-1)]
   # print(select_X)
   # print(select_oos)
  
  y_actual=[]
  y_pred=[]
  rmse = []
  map_error=[]
  trmse = []
  tmap_error=[]; fitted_train=[]
  tscv = utils.TimeSeriesSplit(n_splits = n_splits, validation_window=validation_window)
  forecast1=pd.DataFrame(columns=['date_','forecast','method'])
  select_X=pd.DataFrame(select_X)
  for train_index, test_index in tscv.split(select_X):
    cv_train, cv_test = select_X.iloc[train_index], select_X.iloc[test_index]
   # model1=LGBMRegressor(n_jobs=4,seed=123,learning_rate=.1,num_iterations=200,early_stopping_round=10)
   # param_search = {'max_depth' : [1,3,5,7,9],'min_child_weight' : [1,5,10,20],'num_leaves':[2,4,8,16,32,64]}
    #my_cv = TimeSeriesSplit(n_splits=2).split(cv_train)
    #gsearch = GridSearchCV(estimator=model1, cv=tscv.split(select_X),scoring = 'neg_mean_squared_error',param_grid=param_search).fit(cv_train.astype('float64'), y_train[:len(cv_train)].astype('float64').values)
    model1=LGBMRegressor(n_jobs=4,seed=123,learning_rate=.1,num_iterations=20).fit(X=cv_train,y=y_train[y_train.index[train_index]])
    predictions = model1.predict(cv_test)
    train_pred=model1.predict(cv_train)
    train_true_values=ti[:len(cv_train)]
    true_values=ti[len(cv_train):len(cv_train)+len(cv_test)]
    y_actual=y_actual+list(true_values)
    y_pred=y_pred+list(predictions)
    trmse.append(error_metrics.wrmse(train_true_values, train_pred))
    tmap_error.append(error_metrics.wmape(train_true_values,train_pred))
   # rmse.append(wrmse(true_values, y_pred))
   # map_error.append(wmape(true_values,y_pred))
    forecast= pd.concat([pd.Series(ti.index[len(cv_train):len(cv_train)+len(cv_test)]),pd.Series(model1.predict(cv_test)),pd.Series(['lgbm']*len(cv_test))],axis=1)
    forecast.columns=['date_','forecast','method']
    forecast1=pd.concat([forecast1,forecast],axis=0,ignore_index=True)
 # train_accuracy={'RMSE':np.mean(trmse),'MAPE':np.mean(tmap_error)}
 # model=model1.best_estimator_.fit(select_X,ti)
  model1=LGBMRegressor(n_jobs=4,seed=123,learning_rate=.1,num_iterations=20).fit(select_X,ti)
  forecast2=pd.concat([pd.Series(oos.index),pd.Series(model1.predict(select_X_oos)),pd.Series(['lgbm']*h)],axis=1)
  forecast2.columns=['date_','forecast','method']
  # forecast=pd.concat([forecast,forecast2],axis=0,ignore_index=True)
  return {'forecast':forecast1,'oos forecast':forecast2} 
