import forecast_engine.modelling.error_metrics.nodes as error_metrics



# Ensemble

import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import collections
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingRegressor
warnings.simplefilter('ignore')

def simple_mean_ensemble(oos_forecast,val_forecast,val_forecast_list,oos_forecast_list,list_of_models,smape_weights, anomalous_months):
  ensemble=val_forecast.copy()
  X=[val_forecast_list[your_key] for your_key in list_of_models.keys()]
  all_Xs=[x['forecast'].iloc[:,1] for x in X]
  X_future=[oos_forecast_list[your_key] for your_key in list_of_models.keys()]
  all_X_futures=[x['forecast'].iloc[:,1] for x in X_future]
  ensemble.loc[:,'forecast']=(pd.DataFrame(all_Xs)[~pd.DataFrame(all_Xs).T.isin([np.inf, -np.inf]).any(0)].T).mean(axis=1)
  ensemble.loc[:,'type']=' '.join([x for x in list_of_models])
  
  if len(anomalous_months)>0:
#     print('excluded dates:\n',ensemble['date_'][ensemble['date_'].str.contains('|'.join(anomalous_months),regex=True)])
    ensemble_anomaly_removed = ensemble[~ensemble['date_'].str.contains('|'.join(anomalous_months),regex=True)]
  else: 
    ensemble_anomaly_removed = ensemble.copy()
  ensemble.loc[:,'Forecast_accuracy']=error_metrics.wsmape(ensemble_anomaly_removed['Actual'],ensemble_anomaly_removed['forecast'], smape_weights)
#   print(ensemble)
  ensemble_forecast=oos_forecast.copy()
  ensemble_forecast.loc[:,'forecast']=(pd.DataFrame(all_X_futures)[~pd.DataFrame(all_X_futures).T.isin([np.inf, -np.inf]).any(0)].T).mean(axis=1)
  ensemble_forecast.loc[:,'type']=' '.join([x for x in list_of_models])
  ensemble=pd.concat([ensemble,ensemble_forecast],axis=0,ignore_index=True)
  ensemble.loc[:,'type']=ensemble.loc[:,'type']+' 01'
  
  return ensemble

def weighted_array(forecasts,y):
  value=[]
  for i in range(0,len(forecasts[0])):
    dtype = [('forecasts', 'double'), ('accuracy', 'double')]
    values = list(zip([x[i] for x in forecasts],y))
    a = np.asarray(values, dtype=dtype)       
    a=list(np.sort(a, order='accuracy')[::-1])
    weights=[np.power(0.7,len(a)-a.index(x)-1) for x in a]
    value.append(np.average([x[0] for x in a],weights=weights))   
  return value,weights

def weighted_mean_ensemble(oos_forecast,val_forecast,val_forecast_list,oos_forecast_list,list_of_models,dataf,smape_weights,anomalous_months):
  ensemble=val_forecast.copy()
  X=[val_forecast_list[your_key] for your_key in list_of_models.keys()]
  
  weights = dataf.loc[dataf['models'].isin(list(list_of_models.keys())),'smape'].tolist()
  all_Xs=[x['forecast'].iloc[:,1] for x in X]
  X_future=[oos_forecast_list[your_key] for your_key in list_of_models.keys()]
  all_X_futures=[x['forecast'].iloc[:,1] for x in X_future]
  results=weighted_array(list(filter(lambda x: all(map(np.isfinite, x)), all_Xs)),weights)
  ensemble.loc[:,'forecast']=results[0]
  ensemble.loc[:,'type']=' '.join([x for x in list_of_models])

  if len(anomalous_months)>0:
#     print('excluded dates:\n',ensemble['date_'][ensemble['date_'].str.contains('|'.join(anomalous_months),regex=True)])
    ensemble_anomaly_removed = ensemble[~ensemble['date_'].str.contains('|'.join(anomalous_months),regex=True)]
  else: 
    ensemble_anomaly_removed = ensemble.copy()
  ensemble.loc[:,'Forecast_accuracy']=error_metrics.wsmape(ensemble_anomaly_removed['Actual'],ensemble_anomaly_removed['forecast'], smape_weights)
#   print(ensemble)
  
  ensemble_forecast=oos_forecast.copy()
  ensemble_forecast.loc[:,'forecast']=weighted_array(list(filter(lambda x: all(map(np.isfinite, x)), all_X_futures)),results[1])[0]
  ensemble_forecast.loc[:,'type']=' '.join([x for x in list_of_models])
  ensemble=pd.concat([ensemble,ensemble_forecast],axis=0,ignore_index=True)
  ensemble.loc[:,'type']=ensemble.loc[:,'type']+' 02'
  return ensemble

def linear_reg(X,y,X_future):
  lr=LinearRegression(fit_intercept=False)
  return lr.fit(X,y).predict(X),lr.fit(X,y).predict(X_future)
def linear_reg_bounds_error(params, *args):
    X = args[0]; y = args[1];
    error = np.sum((y - X.dot(params))**2)
    return error
def linear_reg_bounds_output(params, x):
    output = x.dot(params)
    return output
def linear_reg_bounds(X,y,X_future):
  numvars = X.shape[1]
  init_params = np.array([1/numvars]*numvars)
  bounds = [(0,1) for i in range(len(init_params))]
  optim_output = minimize(linear_reg_bounds_error, init_params, args = (X,y), bounds=bounds)  
  return linear_reg_bounds_output(optim_output.x, X), linear_reg_bounds_output(optim_output.x, X_future)

def linear_regressor_ensemble(oos_forecast,val_forecast,val_forecast_list,oos_forecast_list,list_of_models,smape_weights, anomalous_months):
  my_dict=collections.OrderedDict(list(zip([your_key for your_key in list_of_models.keys()],[val_forecast_list[your_key] for your_key in list_of_models.keys()])))
 
  X=pd.DataFrame([x['forecast'].iloc[:,1] for x in my_dict.values()]).T
  X.columns=[x for x in list(my_dict.keys())]
  y=val_forecast['Actual']
  fut_dict=collections.OrderedDict(list(zip([your_key for your_key in list_of_models.keys()],[oos_forecast_list[your_key] for your_key in list_of_models.keys()])))
  X_future=pd.DataFrame([x['forecast'].iloc[:,1] for x in fut_dict.values()]).T
  X_future.columns=[x for x in list(fut_dict.keys())]
  X_future=X_future[X.columns]
  
  X = X[X.columns[~X.isin([np.inf, -np.inf,np.nan]).any(0)]]
  X_future = X_future[X_future.columns[~X_future.isin([np.inf, -np.inf,np.nan]).any(0)]]
  cols=list(np.intersect1d(X.columns,X_future.columns))
  X_future=X_future[cols]
  X = X[cols]
  
  ensemble=val_forecast.copy()
  
  kfold = KFold(n_splits = 3)
  for train, test in kfold.split(range(len(X))):
    X_train = X.iloc[train,:]
    y_train = y.iloc[train]
    X_test = X.iloc[test,:] 
    ensemble['forecast'].iloc[test]=linear_reg_bounds(X_train,y_train,X_test)[1]
    
  ensemble.loc[:,'type']=' '.join([x for x in X.columns])
  
  if len(anomalous_months)>0:
#     print('excluded dates:\n',ensemble['date_'][ensemble['date_'].str.contains('|'.join(anomalous_months),regex=True)])
    ensemble_anomaly_removed = ensemble[~ensemble['date_'].str.contains('|'.join(anomalous_months),regex=True)]
  else: 
    ensemble_anomaly_removed = ensemble.copy()
  ensemble.loc[:,'Forecast_accuracy']=error_metrics.wsmape(ensemble_anomaly_removed['Actual'],ensemble_anomaly_removed['forecast'], smape_weights)
#   print(ensemble)
  
#   ensemble.loc[:,'Forecast_accuracy']=smape(ensemble['Actual'],ensemble['forecast'])
#   ensemble.loc[:,'Forecast_accuracy']=wsmape(ensemble['Actual'],ensemble['forecast'], smape_weights)
  
  ensemble_forecast=oos_forecast.copy()
  
  ensemble_forecast.loc[:,'forecast']=linear_reg_bounds(X,y,X_future)[1]
  ensemble_forecast.loc[:,'type']=' '.join([x for x in X.columns])
  ensemble=pd.concat([ensemble,ensemble_forecast],axis=0,ignore_index=True)
  ensemble.loc[:,'type']=ensemble.loc[:,'type']+' 03'
  return ensemble

def bagging_reg(X,y,X_future):
  br=BaggingRegressor(max_samples=X.shape[0], max_features=X.shape[1],n_estimators=5,random_state=100)
  return br.fit(X,y).predict(X),br.fit(X,y).predict(X_future)

def bagging_regressor_ensemble(oos_forecast,val_forecast,val_forecast_list,oos_forecast_list,list_of_models,smape_weights, anomalous_months):
  my_dict=collections.OrderedDict(list(zip([your_key for your_key in list_of_models.keys()],[val_forecast_list[your_key] for your_key in list_of_models.keys()])))
  X=pd.DataFrame([x['forecast'].iloc[:,1] for x in my_dict.values()]).T
  for a in range(0,len(X.iloc[-1,:])):
    if (X.iloc[:,a]>=np.power(10,20)).any(): X.iloc[-1,a]=np.nan
  X.columns=[x for x in list(my_dict.keys())]
  X = X[X.columns[~X.isin([np.inf, -np.inf,np.nan]).any(0)]]
  y=val_forecast['Actual']
  fut_dict=collections.OrderedDict(list(zip([your_key for your_key in list_of_models.keys()],[oos_forecast_list[your_key] for your_key in list_of_models.keys()])))
  X_future=pd.DataFrame([x['forecast'].iloc[:,1] for x in fut_dict.values()]).T
  for a in range(0,len(X_future.iloc[-1,:])):
    if (X_future.iloc[:,a]>=np.power(10,20)).any(): X_future.iloc[-1,a]=np.nan
  
  X_future.columns=[x for x in list(fut_dict.keys())]
  X_future = X_future[X_future.columns[~X_future.isin([np.inf, -np.inf,np.nan]).any(0)]]
  cols=list(np.intersect1d(X.columns,X_future.columns))
  X_future=X_future[cols]
  X = X[cols]
  ensemble=val_forecast.copy()

  kfold = KFold(n_splits = 3)
  for train, test in kfold.split(range(len(X))):
    X_train = X.iloc[train,:]
    y_train = y.iloc[train]
    X_test = X.iloc[test,:] 
    # X_test.fillna(X_test.mean(), inplace=True)
    # X_train.fillna(X_train.mean(), inplace=True)
    # X_train.fillna(X_train.mean(), inplace=True)
    ensemble['forecast'].iloc[test]=bagging_reg(X_train,y_train,X_test)[1]
  
  ensemble.loc[:,'type']=' '.join([x for x in X.columns])
  
  if len(anomalous_months)>0:
#     print('excluded dates:\n',ensemble['date_'][ensemble['date_'].str.contains('|'.join(anomalous_months),regex=True)])
    ensemble_anomaly_removed = ensemble[~ensemble['date_'].str.contains('|'.join(anomalous_months),regex=True)]
  else: 
    ensemble_anomaly_removed = ensemble.copy()
  ensemble.loc[:,'Forecast_accuracy']=error_metrics.wsmape(ensemble_anomaly_removed['Actual'],ensemble_anomaly_removed['forecast'], smape_weights)
#   print(ensemble)
  
#   ensemble.loc[:,'Forecast_accuracy']=smape(ensemble['Actual'],ensemble['forecast'])
#   ensemble.loc[:,'Forecast_accuracy']=wsmape(ensemble['Actual'],ensemble['forecast'], smape_weights)
  ensemble_forecast=oos_forecast.copy()
  ensemble_forecast.loc[:,'forecast']=bagging_reg(X,y,X_future)[1]
  ensemble_forecast.loc[:,'type']=' '.join([x for x in X.columns])
  ensemble=pd.concat([ensemble,ensemble_forecast],axis=0,ignore_index=True)
  ensemble.loc[:,'type']=ensemble.loc[:,'type']+' 04'
  return ensemble

