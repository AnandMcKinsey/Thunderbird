import forecast_engine.modelling.utils.nodes as utils
import forecast_engine.modelling.error_metrics.nodes as error_metrics
from statsmodels.tsa.holtwinters import SimpleExpSmoothing as SES


import pandas as pd
import numpy as np


def zero(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
    
    # if 'zero' in debug_models:
    #   with open(ti_pickle_path, 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([ti, h, row_counter, variable_list, n_splits, validation_window], f)    
    
    oos=ti[row_counter-1:row_counter-1+h]
    ti=ti[:(row_counter-1)]
    ti=ti.truncate(before=ti.index[np.min(np.where(ti.notnull())[0])])
      
    y_actual=[]; y_pred=[]; rmse = []; map_error=[]; fitted_train=[]
    forecast1=pd.DataFrame(columns=['date_','forecast','method'])
    if n_splits==0:
      forecast1=pd.concat([pd.Series(ti.index),pd.Series([0]*len(ti)),pd.Series(['zero']*len(ti))],axis=1)
      forecast1.columns=['date_','forecast','method']
      forecast2=pd.concat([pd.Series(oos.index),pd.Series([0]*len(oos)),pd.Series(['zero']*len(oos))],axis=1)
      forecast2.columns=['date_','forecast','method']
      val_accuracy = 0
    else:
      tscv = utils.TimeSeriesSplit(n_splits = n_splits, validation_window=validation_window)
      for train_index, test_index in tscv.split(ti):
        cv_train, cv_test = ti.iloc[train_index], ti.iloc[test_index]
        predictions = [0]*len(cv_test)
        true_values = cv_test.values
        y_actual=y_actual+list(true_values)
        y_pred=y_pred+list(predictions)
        forecast=pd.concat([pd.Series(cv_test.index),pd.Series(predictions), pd.Series(['zero']*len(cv_test))],axis=1)
        forecast.columns=['date_','forecast','method']
        forecast1=pd.concat([forecast1,forecast],axis=0,ignore_index=True)
        
        train_fit = [0]*len(cv_train) 
        fitted_train1=pd.concat([pd.Series(ti.index[train_index[0]:test_index[0]]),pd.Series(train_fit),pd.Series(['zero']*len(train_fit),name='method')],axis=1)
        fitted_train1.columns=['date_','forecast','method']
        fitted_train.append(fitted_train1)

      predictions = [0]*h
      forecast2=pd.concat([pd.Series(oos.index),pd.Series(predictions),pd.Series(['zero']*h)],axis=1)
      forecast2.columns=['date_','forecast','method']
    
    forecast1['forecast']=forecast1['forecast'].astype('float64')  
    forecast2['forecast']=forecast2['forecast'].astype('float64')    
    
    #Impact snippet
    rest_var = variable_list.tolist()
    rest_len = len(rest_var)
    var_coeff = pd.DataFrame()
    for month_ in range(0,len(oos)):
      rest_coeff = pd.concat([pd.Series(['zero']*rest_len), pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len), pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff=pd.concat([var_coeff,rest_coeff],axis=0)
      
    if (np.sum(forecast1['forecast'].isnull())) or (np.sum(forecast2['forecast'].isnull())) or np.sum(np.isinf(forecast1['forecast'])) or np.sum(np.isinf(forecast2['forecast'])):
      raise Exception('NaN in output')
    
    total_fit = [0]*len(ti)
    total_fit=pd.concat([pd.Series(ti.index),pd.Series(total_fit),pd.Series(['zero']*len(ti))],axis=1)
    total_fit.columns = ['date_', 'forecast','method']
    total_fit['forecast'] = total_fit['forecast']
    
    return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def poor_naive(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
    
    oos=ti[row_counter-1:row_counter-1+h]
    ti=ti[:(row_counter-1)]
    ti=ti.truncate(before=ti.index[np.min(np.where(ti.notnull())[0])])
    
    y_actual=[]; y_pred=[]; rmse = []; map_error=[]; fitted_train=[]
    forecast1=pd.DataFrame(columns=['date_','forecast','method'])

    tscv = utils.TimeSeriesSplit(n_splits = n_splits, validation_window=validation_window)
    for train_index, test_index in tscv.split(ti):
      cv_train, cv_test = ti.iloc[train_index], ti.iloc[test_index]
      predictions = [cv_train[len(cv_train)-1]]*len(cv_test)
      true_values = cv_test.values
      y_actual=y_actual+list(true_values)
      y_pred=y_pred+list(predictions)
      forecast=pd.concat([pd.Series(cv_test.index),pd.Series(predictions), pd.Series(['poor_naive']*len(cv_test))],axis=1)
      forecast.columns=['date_','forecast','method']
      forecast1=pd.concat([forecast1,forecast],axis=0,ignore_index=True)

      train_fit = list(cv_train.shift(1).values[1:])
      fitted_train1=pd.concat([pd.Series(ti.index[train_index[1]:test_index[0]]),pd.Series(train_fit),pd.Series(['poor_naive']*len(train_fit),name='method')],axis=1)
      fitted_train1.columns=['date_','forecast','method']
      fitted_train.append(fitted_train1)      

    predictions = [ti[len(ti)-1]]*h
    forecast2=pd.concat([pd.Series(oos.index),pd.Series(predictions),pd.Series(['poor_naive']*h)],axis=1)
    forecast2.columns=['date_','forecast','method']

    forecast1['forecast']=forecast1['forecast'].astype('float64')  
    forecast2['forecast']=forecast2['forecast'].astype('float64')
    
    #Impact snippet
    rest_var = variable_list.tolist()
    rest_len = len(rest_var)
    var_coeff = pd.DataFrame()
    for month_ in range(0,len(oos)):
      rest_coeff = pd.concat([pd.Series(['poor_naive']*rest_len), pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len), pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff=pd.concat([var_coeff,rest_coeff],axis=0)
    
    if (np.sum(forecast1['forecast'].isnull())) or (np.sum(forecast2['forecast'].isnull())) or np.sum(np.isinf(forecast1['forecast'])) or np.sum(np.isinf(forecast2['forecast'])):
      raise Exception('NaN in output')
      
    total_fit = list(ti.shift(1).values[1:])
    total_fit=pd.concat([pd.Series(ti.index[1:]),pd.Series(total_fit),pd.Series(['poor_naive']*len(total_fit))],axis=1)
    total_fit.columns = ['date_', 'forecast','method']
    total_fit['forecast'] = total_fit['forecast']   

    return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}



def naive(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
    
    oos=ti[row_counter-1:row_counter-1+h]
    ti=ti[:(row_counter-1)]
    ti=ti.truncate(before=ti.index[np.min(np.where(ti.notnull())[0])])
    
    y_actual=[]; y_pred=[]; rmse = []; map_error=[]; fitted_train=[]
    forecast1=pd.DataFrame(columns=['date_','forecast','method'])

    tscv = utils.TimeSeriesSplit(n_splits = n_splits, validation_window=validation_window)
    for train_index, test_index in tscv.split(ti):
      cv_train, cv_test = ti.iloc[train_index], ti.iloc[test_index]
      predictions = [cv_train[len(cv_train)-1]]*len(cv_test)
      true_values = cv_test.values
      y_actual=y_actual+list(true_values)
      y_pred=y_pred+list(predictions)
      forecast=pd.concat([pd.Series(cv_test.index),pd.Series(predictions), pd.Series(['naive']*len(cv_test))],axis=1)
      forecast.columns=['date_','forecast','method']
      forecast1=pd.concat([forecast1,forecast],axis=0,ignore_index=True)
      
      train_fit = list(cv_train.shift(1).fillna(0).values[1:])
      fitted_train1=pd.concat([pd.Series(ti.index[train_index[0]:test_index[0]]),pd.Series(train_fit),pd.Series(['naive']*len(train_fit),name='method')],axis=1)
      fitted_train1.columns=['date_','forecast','method']
      fitted_train.append(fitted_train1)         

    predictions = [ti[len(ti)-1]]*h
    forecast2=pd.concat([pd.Series(oos.index),pd.Series(predictions),pd.Series(['naive']*h)],axis=1)
    forecast2.columns=['date_','forecast','method']

    forecast1['forecast']=forecast1['forecast'].astype('float64')  
    forecast2['forecast']=forecast2['forecast'].astype('float64')
    
    #Impact snippet
    rest_var = variable_list.tolist()
    rest_len = len(rest_var)
    var_coeff = pd.DataFrame()
    for month_ in range(0,len(oos)):
      rest_coeff = pd.concat([pd.Series(['naive']*rest_len), pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len), pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff=pd.concat([var_coeff,rest_coeff],axis=0)
    
    if (np.sum(forecast1['forecast'].isnull())) or (np.sum(forecast2['forecast'].isnull())) or np.sum(np.isinf(forecast1['forecast'])) or np.sum(np.isinf(forecast2['forecast'])):
      raise Exception('NaN in output')
      
    total_fit = list(ti.shift(1).values[1:])
    total_fit=pd.concat([pd.Series(ti.index[1:]),pd.Series(total_fit),pd.Series(['naive']*len(total_fit))],axis=1)
    total_fit.columns = ['date_', 'forecast','method']
    total_fit['forecast'] = total_fit['forecast']      
    
    return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def mean_model(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):

#     if debug_flag==1:
#       with open(ti_pickle_path, 'wb') as f:  # Python 3: open(..., 'wb')
#         pickle.dump([ti, h, row_counter, n_splits, validation_window], f)    
    
    oos=ti[row_counter-1:row_counter-1+h]
    
    ti=ti[:(row_counter-1)]
    ti=ti.truncate(before=ti.index[np.min(np.where(ti.notnull())[0])])
    y_actual=[]; y_pred=[]; rmse = []; map_error=[]; fitted_train=[]
    forecast1=pd.DataFrame(columns=['date_','forecast','method'])
    
    if n_splits==0:
      mean_ = np.mean(ti[(ti!=0)&~(pd.isna(ti))])
      forecast1=pd.concat([pd.Series(ti.index),pd.Series([mean_]*len(ti)),pd.Series(['mean_model']*len(ti))],axis=1)
      forecast1.columns=['date_','forecast','method']
      forecast2=pd.concat([pd.Series(oos.index),pd.Series([mean_]*len(oos)),pd.Series(['mean_model']*len(oos))],axis=1)
      forecast2.columns=['date_','forecast','method']
      val_accuracy = 0
    else:
      tscv = utils.TimeSeriesSplit(n_splits = n_splits, validation_window=validation_window)
      for train_index, test_index in tscv.split(ti):
        cv_train, cv_test = ti.iloc[train_index], ti.iloc[test_index]
        predictions = [np.mean(cv_train.values)]*len(cv_test)
        true_values = cv_test.values
        y_actual=y_actual+list(true_values)
        y_pred=y_pred+list(predictions)
        forecast=pd.concat([pd.Series(cv_test.index),pd.Series(predictions), pd.Series(['mean_model']*len(cv_test))],axis=1)
        forecast.columns=['date_','forecast','method']
        forecast1=pd.concat([forecast1,forecast],axis=0,ignore_index=True)
        
        train_fit = list(cv_train.expanding().mean())    
        fitted_train1=pd.concat([pd.Series(ti.index[train_index[0]:test_index[0]]),pd.Series(train_fit),pd.Series(['mean_model']*len(train_fit),name='method')],axis=1)
        fitted_train1.columns=['date_','forecast','method']
        fitted_train.append(fitted_train1)           
        
    predictions = [np.mean(ti.values)]*h
    forecast2=pd.concat([pd.Series(oos.index),pd.Series(predictions),pd.Series(['mean_model']*h)],axis=1)
    forecast2.columns=['date_','forecast','method']
    
    forecast1['forecast']=forecast1['forecast'].astype('float64')  
    forecast2['forecast']=forecast2['forecast'].astype('float64') 
    
    #Impact snippet
    rest_var = variable_list.tolist()
    rest_len = len(rest_var)
    var_coeff = pd.DataFrame()
    for month_ in range(0,len(oos)):
      rest_coeff = pd.concat([pd.Series(['mean_model']*rest_len), pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len), pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff=pd.concat([var_coeff,rest_coeff],axis=0)
    
    if (np.sum(forecast1['forecast'].isnull())) or (np.sum(forecast2['forecast'].isnull())) or np.sum(np.isinf(forecast1['forecast'])) or np.sum(np.isinf(forecast2['forecast'])):
      raise Exception('NaN in output')
      
    total_fit = list(ti.expanding().mean()) 
    total_fit=pd.concat([pd.Series(ti.index[0:]),pd.Series(total_fit),pd.Series(['mean_model']*len(total_fit))],axis=1)
    total_fit.columns = ['date_', 'forecast','method']
    total_fit['forecast'] = total_fit['forecast']       
      
    return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def median_model(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
    
    oos=ti[row_counter-1:row_counter-1+h]
    ti=ti[:(row_counter-1)]
    ti=ti.truncate(before=ti.index[np.min(np.where(ti.notnull())[0])])
    
    y_actual=[]; y_pred=[]; rmse = []; map_error=[]; fitted_train=[]
    forecast1=pd.DataFrame(columns=['date_','forecast','method'])

    tscv = utils.TimeSeriesSplit(n_splits = n_splits, validation_window=validation_window)
    for train_index, test_index in tscv.split(ti):
      cv_train, cv_test = ti.iloc[train_index], ti.iloc[test_index]
      predictions = [np.median(cv_train.values)]*len(cv_test)
      true_values = cv_test.values
      y_actual=y_actual+list(true_values)
      y_pred=y_pred+list(predictions)
      forecast=pd.concat([pd.Series(cv_test.index),pd.Series(predictions), pd.Series(['median_model']*len(cv_test))],axis=1)
      forecast.columns=['date_','forecast','method']
      forecast1=pd.concat([forecast1,forecast],axis=0,ignore_index=True)
      
      train_fit = list(cv_train.expanding().median())    
      fitted_train1=pd.concat([pd.Series(ti.index[train_index[0]:test_index[0]]),pd.Series(train_fit),pd.Series(['median_model']*len(train_fit),name='method')],axis=1)
      fitted_train1.columns=['date_','forecast','method']
      fitted_train.append(fitted_train1)        

    predictions = [np.median(ti.values)]*h
    forecast2=pd.concat([pd.Series(oos.index),pd.Series(predictions),pd.Series(['median_model']*h)],axis=1)
    forecast2.columns=['date_','forecast','method']
    
    forecast1['forecast']=forecast1['forecast'].astype('float64')  
    forecast2['forecast']=forecast2['forecast'].astype('float64')

    #Impact snippet
    rest_var = variable_list.tolist()
    rest_len = len(rest_var)
    var_coeff = pd.DataFrame()
    for month_ in range(0,len(oos)):
      rest_coeff = pd.concat([pd.Series(['median_model']*rest_len), pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len), pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff=pd.concat([var_coeff,rest_coeff],axis=0)
    
    if (np.sum(forecast1['forecast'].isnull())) or (np.sum(forecast2['forecast'].isnull())) or np.sum(np.isinf(forecast1['forecast'])) or np.sum(np.isinf(forecast2['forecast'])):
      raise Exception('NaN in output')
    
    total_fit = list(ti.expanding().median()) 
    total_fit=pd.concat([pd.Series(ti.index[0:]),pd.Series(total_fit),pd.Series(['median_model']*len(total_fit))],axis=1)
    total_fit.columns = ['date_', 'forecast','method']
    total_fit['forecast'] = total_fit['forecast']     
    
    return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def snaive(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
    
    oos=ti[row_counter-1:row_counter-1+h]
    ti=ti[:(row_counter-1)]
    ti=ti.truncate(before=ti.index[np.min(np.where(ti.notnull())[0])])
#     print('ti values after truncation in snaive:\n',ti.values)
    seasonal_periods = utils.get_seasonal_periods(ti)
    y_actual=[]; y_pred=[]; rmse = []; map_error=[]; fitted_train=[]
    forecast1=pd.DataFrame(columns=['date_','forecast','method'])

    tscv = utils.TimeSeriesSplit(n_splits = n_splits, validation_window=validation_window)
    for train_index, test_index in tscv.split(ti):
      cv_train, cv_test = ti.iloc[train_index], ti.iloc[test_index]
      if len(cv_train)<seasonal_periods:
        raise Exception('In snaive: Length of dataframe not sufficient for snaive')
      
      predictions = ti[len(cv_train)-seasonal_periods:len(cv_train)-seasonal_periods+len(cv_test)].values
      true_values = cv_test.values
      y_actual=y_actual+list(true_values)
      y_pred=y_pred+list(predictions)
      forecast=pd.concat([pd.Series(cv_test.index),pd.Series(predictions), pd.Series(['snaive']*len(cv_test))],axis=1)
      forecast.columns=['date_','forecast','method']
      forecast1=pd.concat([forecast1,forecast],axis=0,ignore_index=True)
      
      try:
        train_fit = list(cv_train.shift(seasonal_periods).values[seasonal_periods:])
        fitted_train1=pd.concat([pd.Series(ti.index[train_index[seasonal_periods]:test_index[0]]),pd.Series(train_fit),pd.Series(['snaive']*len(train_fit),name='method')],axis=1)
        fitted_train1.columns=['date_','forecast','method']
        fitted_train.append(fitted_train1)   
      except:
        train_fit = [0]*len(cv_train) 
        fitted_train1=pd.concat([pd.Series(ti.index[train_index[0]:test_index[0]]),pd.Series(train_fit),pd.Series(['snaive']*len(train_fit),name='method')],axis=1)
        fitted_train1.columns=['date_','forecast','method']
        fitted_train.append(fitted_train1)        
      
    predictions = ti[len(ti)-seasonal_periods:len(ti)-seasonal_periods+h].values
    forecast2=pd.concat([pd.Series(oos.index),pd.Series(predictions),pd.Series(['snaive']*h)],axis=1)
    forecast2.columns=['date_','forecast','method']
    
    forecast1['forecast']=forecast1['forecast'].astype('float64')  
    forecast2['forecast']=forecast2['forecast'].astype('float64')
    
    #Impact snippet
    rest_var = variable_list.tolist()
    rest_len = len(rest_var)
    var_coeff = pd.DataFrame()
    for month_ in range(0,len(oos)):
      rest_coeff = pd.concat([pd.Series(['snaive']*rest_len), pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len), pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff=pd.concat([var_coeff,rest_coeff],axis=0)
    
    if (np.sum(forecast1['forecast'].isnull())) or (np.sum(forecast2['forecast'].isnull())) or np.sum(np.isinf(forecast1['forecast'])) or np.sum(np.isinf(forecast2['forecast'])):
      raise Exception('NaN in output')

    try:
      total_fit = list(ti.shift(seasonal_periods).values[seasonal_periods:])
      total_fit=pd.concat([pd.Series(ti.index[seasonal_periods:]),pd.Series(total_fit),pd.Series(['snaive']*len(total_fit))],axis=1)
      total_fit.columns = ['date_', 'forecast','method']
      total_fit['forecast'] = total_fit['forecast']  
    except:
      total_fit = [0]*len(ti)
      total_fit=pd.concat([pd.Series(ti.index),pd.Series(total_fit),pd.Series(['snaive']*len(ti))],axis=1)
      total_fit.columns = ['date_', 'forecast','method']
      total_fit['forecast'] = total_fit['forecast']      
    
    return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}


def snaive_twoseasons(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
    
    oos=ti[row_counter-1:row_counter-1+h]
    ti=ti[:(row_counter-1)]
    ti=ti.truncate(before=ti.index[np.min(np.where(ti.notnull())[0])])
#     print('ti values after truncation in snaive_twoseasons:\n',ti.values)
    seasonal_periods = utils.get_seasonal_periods(ti)
    y_actual=[]; y_pred=[]; rmse = []; map_error=[]; fitted_train=[]
    forecast1=pd.DataFrame(columns=['date_','forecast','method'])

    tscv = utils.TimeSeriesSplit(n_splits = n_splits, validation_window=validation_window)
    for train_index, test_index in tscv.split(ti):
      cv_train, cv_test = ti.iloc[train_index], ti.iloc[test_index]
      if len(cv_train)<24:
        raise Exception('In snaive: Length of dataframe not sufficient for snaive_twoseasons')
        
      predictions = ti[len(cv_train)-2*seasonal_periods:len(cv_train)-2*seasonal_periods+len(cv_test)].values
      true_values = cv_test.values
      y_actual=y_actual+list(true_values)
      y_pred=y_pred+list(predictions)
      forecast=pd.concat([pd.Series(cv_test.index),pd.Series(predictions), pd.Series(['snaive_twoseasons']*len(cv_test))],axis=1)
      forecast.columns=['date_','forecast','method']
      forecast1=pd.concat([forecast1,forecast],axis=0,ignore_index=True)

      try:
        train_fit = list(cv_train.shift(2*seasonal_periods).values[2*seasonal_periods:])
        fitted_train1=pd.concat([pd.Series(ti.index[train_index[2*seasonal_periods]:test_index[0]]),pd.Series(train_fit),pd.Series(['snaive_twoseasons']*len(train_fit),name='method')],axis=1)
        fitted_train1.columns=['date_','forecast','method']
        fitted_train.append(fitted_train1)   
      except:
        train_fit = [0]*len(cv_train) 
      
    predictions = ti[len(ti)-2*seasonal_periods:len(ti)-2*seasonal_periods+h].values
    if 'snaive_twoseasons' in debug_models: print('sn-2s len ti',len(ti))
    forecast2=pd.concat([pd.Series(oos.index),pd.Series(predictions),pd.Series(['snaive_twoseasons']*h)],axis=1)
    forecast2.columns=['date_','forecast','method']
#     print('y_actual in snaive_twoseasons:\n',y_actual)
#     print('y_pred in snaive_twoseasons:\n',y_pred)
#     if len(ti)<33:
#       val_accuracy=1e50
#       forecast1.loc[:,'forecast']=1e50
#     else:  val_accuracy={'RMSE':wrmse(y_actual,y_pred)}
    forecast1['forecast']=forecast1['forecast'].astype('float64')  
    forecast2['forecast']=forecast2['forecast'].astype('float64')

    #Impact snippet
    rest_var = variable_list.tolist()
    rest_len = len(rest_var)
    var_coeff = pd.DataFrame()
    for month_ in range(0,len(oos)):
      rest_coeff = pd.concat([pd.Series(['snaive_twoseasons']*rest_len), pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len), pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff=pd.concat([var_coeff,rest_coeff],axis=0)
    
    if (np.sum(forecast1['forecast'].isnull())) or (np.sum(forecast2['forecast'].isnull())) or np.sum(np.isinf(forecast1['forecast'])) or np.sum(np.isinf(forecast2['forecast'])):
      raise Exception('NaN in output')
    
    try:
      total_fit = list(ti.shift(2*seasonal_periods).values[2*seasonal_periods:])
      total_fit=pd.concat([pd.Series(ti.index[2*seasonal_periods:]),pd.Series(total_fit), pd.Series(['snaive_twoseasons']*len(total_fit))] ,axis=1)
      total_fit.columns = ['date_', 'forecast','method']
      total_fit['forecast'] = total_fit['forecast']  
    except:
      total_fit = [0]*len(ti)
      total_fit=pd.concat([pd.Series(ti.index),pd.Series(total_fit),pd.Series(['snaive_twoseasons']*len(ti))],axis=1)
      total_fit.columns = ['date_', 'forecast','method']
      total_fit['forecast'] = total_fit['forecast']      
    
#     total_fit=[0]
#     fitted_train=[0]
# #     print('val_accuracy in snaive_twoseasons:',val_accuracy)
    return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}


def naive_ets(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):

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
      fit = SES(cv_train.values).fit(optimized=True)
      predictions = fit.predict(start=len(cv_train), end=len(cv_train)+len(cv_test)-1)
#       print(fit.params)
#       print(predictions-min_series)
      true_values = cv_test.values
      y_actual=y_actual+list((true_values-min_series)*scale_factor)
      y_pred=y_pred+list((predictions-min_series)*scale_factor)
      forecast=pd.concat([pd.Series(cv_test.index),pd.Series(predictions-min_series), pd.Series(['naive_ets']*len(cv_test))],axis=1)
      forecast.columns=['date_','forecast','method']
      forecast1=pd.concat([forecast1,forecast],axis=0,ignore_index=True)
      
      train_fit = list((fit.predict(start=train_index[0], end = train_index[-1]) - min_series)*scale_factor)
      fitted_train1=pd.concat([pd.Series(ti.index[train_index[0]:test_index[0]]) ,pd.Series(train_fit) ,pd.Series(['naive_ets']*len(train_fit) ,name='method')],axis=1)
      fitted_train1.columns=['date_','forecast','method']
      #print('fitted_train:\n',fitted_train1)
      fitted_train.append(fitted_train1)
      
    
    np.random.seed(1234)
    fit2 = SES(ti.values).fit(optimized=True)
    predictions=fit2.predict(start=len(ti), end=len(ti)+h-1)
    forecast2=pd.concat([pd.Series(oos.index),pd.Series(predictions-min_series),pd.Series(['naive_ets']*h)],axis=1)
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
      rest_coeff = pd.concat([pd.Series(['naive_ets']*rest_len), pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len), pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff=pd.concat([var_coeff,rest_coeff],axis=0)
    
    if (np.sum(forecast1['forecast'].isnull())) or (np.sum(forecast2['forecast'].isnull())) or np.sum(np.isinf(forecast1['forecast'])) or np.sum(np.isinf(forecast2['forecast'])):
      raise Exception('NaN in output')    
    
    total_fit = fit2.predict(start=0, end=len(ti)-1)
    total_fit=pd.concat([pd.Series(ti.index),pd.Series(total_fit-min_series),pd.Series(['naive_ets']*len(total_fit))],axis=1)
    total_fit.columns = ['date_', 'forecast','method']
    total_fit['forecast'] = total_fit['forecast']*scale_factor 
#     print('total_fit:\n',total_fit)
    
    return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}