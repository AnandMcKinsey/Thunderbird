import pandas as pd
import numpy as np
import forecast_engine.modelling.anomaly_detection.pipeline as anomaly_pipeline
import forecast_engine.modelling.forecast_engine.variable_selection as variable_selection
import forecast_engine.modelling.forecast_engine.model_list as model_list_pipeline
import forecast_engine.modelling.error_metrics.nodes as error_metrics
import forecast_engine.modelling.models.ensembles.ensemble_engine as ensemble_engine
import forecast_engine.modelling.forecast_engine.analysis_dataframe as analysis_dataframe
import forecast_engine.modelling.forecast_engine.model_selection as model_selection




# Model Comaprison function

def model_comparison_function(ti, series_flag,row_counter,list_of_models,debug_flag,debug_models,validation_window, max_validation, variable_list,single_series,oos_length,outlier_flag,select_variable_flag):
  import collections
  import random
  import itertools
  from collections import OrderedDict
  from copy import deepcopy
  
  n_splits = int(np.floor(np.minimum(max_validation,(len(ti['Value'][:(row_counter-1)].dropna())-1))/validation_window))
  index = ti['Value'].first_valid_index()

  if n_splits==0: series_flag = 5
  if debug_flag==1: print(n_splits)
  if debug_flag==1: print(series_flag)

  ti,index,list_of_models = model_selection.model_selection_function(ti,series_flag,list_of_models,debug_flag)

  ti = ti.dropna(axis=1, how = 'all')
  for i in ti.iloc[:,2:].columns:
      col_length = pd.Series(ti[i].unique()).dropna().shape[0]
      if col_length == 1:
          ti = ti.drop([i],axis=1)

  ti1 = ti.copy()  

  anomalous_months = []

  if outlier_flag == 1:
      if series_flag==1:
              tad = anomaly_pipeline.twitteranomalydetection(ti1,plot=False)
              anomalous_months = tad['anomalies'].index
              ti1['Value'][ti1['Header'].isin(anomalous_months)] = tad['anomalies']['expected_value'].values
      

  ######### VARIABLE SELECTION #########
  if select_variable_flag == 1:
      ti_select= variable_selection.variable_selection(ti1.copy(),row_counter)
      ti1=ti_select.copy()  
  ######### VARIABLE SELECTION #########
  
  model_list = model_list_pipeline.model_list_function(ti1,series_flag,oos_length,list_of_models,index,row_counter,debug_flag,debug_models,n_splits,validation_window,variable_list,single_series)
  fitted_values_list = deepcopy(model_list['fitted_values_list'])
  val_forecast_list = deepcopy(model_list['val_forecast_list'])
  oos_forecast_list = deepcopy(model_list['oos_forecast_list'])
  
  if (len(val_forecast_list)==0) or (len(oos_forecast_list)==0):
    index = ti['Header'].values[row_counter-n_splits*3-1:row_counter-1]
    predictions = np.zeros(len(index))
    val_forecast=pd.concat([pd.Series(index),pd.Series(predictions),pd.Series(['no_models']*len(index))],axis=1)
    val_forecast.columns = ['date_','forecast','method']

    index = ti['Header'].values[row_counter-1:row_counter]
    predictions = np.zeros(len(index))
    oos_forecast=pd.concat([pd.Series(index),pd.Series(predictions),pd.Series(['no_models']*len(index))],axis=1)
    oos_forecast.columns = ['date_','forecast','method']
    oos_forecast['Forecast_accuracy'] = np.nan
    oos_forecast['Actual'] = np.nan
    oos_forecast.rename(columns = {'method':'type'}, inplace=True)
    val_forecast.rename(columns = {'method':'type'}, inplace=True) 
    classifier_train_set = pd.DataFrame(list(list_of_models.keys())).T
    classifier_train_set.columns = classifier_train_set.iloc[0]
    for i in classifier_train_set.columns:
      classifier_train_set[i] = 0
    all_forecasts = pd.concat([val_forecast, oos_forecast],axis=0)
    all_forecasts = all_forecasts.reset_index(drop=True)
    var_coeff_no_model=pd.DataFrame(0,index=np.arange(0,1),columns=['Model','Date','Driver','Impact'])
    return {'all_forecasts': all_forecasts, 'oos_forecast': oos_forecast, 'classifier_train_set':classifier_train_set, 'var_coeff': var_coeff_no_model}  
  
  var_coeff = model_list['var_coeff']
  model_list = model_list['model_list']
  
  oos_month=ti['Header'][row_counter-1]
  
  if outlier_flag == 1:
    if series_flag==1:
        ti1 = ti1[~ti1.index.isin(anomalous_months)]
    
  
  models_smape = OrderedDict()
  
  
  validation_len = min([y for y in [len(val_forecast_list[x]['forecast']) for x in list(val_forecast_list.keys())]])
  for i in val_forecast_list.keys():
      val_forecast_list[i]['forecast'] = val_forecast_list[i]['forecast'][-validation_len:].reset_index(drop=True)

  smape_weights = pd.Series([1]*validation_len,name='weights')
  for i in model_list:
      y_pred = val_forecast_list[i]['forecast'].copy()
      y_pred = y_pred[~y_pred['date_'].isin(anomalous_months)]
      y_act = ti1[ti1['Header'].isin(list(y_pred['date_']))]
      
      if debug_flag==1:
          print(i)
          print(y_pred)    


      models_smape[i] = error_metrics.wsmape(y_act['Value'],y_pred['forecast'], smape_weights[0:len(y_act)])


  dataf = pd.DataFrame.from_dict(models_smape, orient='index').reset_index()
  dataf.columns = ['models','smape']
  dataf = dataf.sort_values(by = 'smape').reset_index(drop=True)
  print(dataf)
    
  ti_backup=ti.copy()
  ti = ti[index:(row_counter-1)]
  
  models = dataf[~dataf['models'].isin(['xgboost_mv','xgboost'])].iloc[0:5,0].reset_index(drop=True).tolist()
  classifier_train_set = pd.DataFrame(list(list_of_models.keys())).T
  classifier_train_set.columns = classifier_train_set.iloc[0]
  for i in classifier_train_set.columns:
    classifier_train_set[i] = int(pd.Series(i).isin(models))
  
  list_of_models = dict((k, list_of_models[k]) for k in models)
  best_model = dataf.iloc[0:1,0].reset_index(drop=True)[0]
  best_model_smape = dataf.iloc[0:1,1].reset_index(drop=True)[0]
  
  oos_forecast = oos_forecast_list[best_model]['forecast'].copy()
  oos_forecast['Actual'] = np.nan
  print('oos_forecast before:\n',oos_forecast)
      
  val_length = val_forecast_list[best_model]['forecast'].shape[0]
  x = ti[(ti.shape[0]-val_length):(ti.shape[0])]['Value']
  val_forecast = val_forecast_list[best_model]['forecast'].copy()
  val_forecast['Forecast_accuracy'] = best_model_smape
  val_forecast.loc[:,'Actual'] = x.values
  oos_forecast.rename(columns = {'method':'type'}, inplace=True)
  val_forecast.rename(columns = {'method':'type'}, inplace=True)
  
  best_ensemble=pd.DataFrame();ensemble_1=pd.DataFrame();ensemble_2=pd.DataFrame();ensemble_3=pd.DataFrame();ensemble_4=pd.DataFrame()
  
  if series_flag==1:
    ensemble_1=ensemble_engine.simple_mean_ensemble(oos_forecast, val_forecast, val_forecast_list, oos_forecast_list, list_of_models, smape_weights, anomalous_months)
    ensemble_2=ensemble_engine.weighted_mean_ensemble(oos_forecast,val_forecast,val_forecast_list,oos_forecast_list,list_of_models,dataf,smape_weights, anomalous_months)
    ensemble_3=ensemble_engine.linear_regressor_ensemble(oos_forecast,val_forecast,val_forecast_list,oos_forecast_list,list_of_models,smape_weights, anomalous_months)
    ensemble_4=ensemble_engine.bagging_regressor_ensemble(oos_forecast,val_forecast,val_forecast_list,oos_forecast_list,list_of_models,smape_weights, anomalous_months)
    best_ensemble= pd.concat([val_forecast.iloc[0].T,ensemble_1.iloc[0].T,ensemble_2.iloc[0].T,ensemble_3.iloc[0].T],axis=1).T [['Forecast_accuracy','type']].reset_index(drop=True)
    ensemble_type=pd.Series(['val_forecast','ensemble_1','ensemble_2','ensemble_3'],name='ensemble_type')
    
    
    
    best_ensemble=pd.concat([best_ensemble,ensemble_type],axis=1)
    best_ensemble_name=best_ensemble.iloc[(np.where(best_ensemble.iloc[:,0]==(best_ensemble.iloc[:,0]).min()))[0].min(),2]  
    if best_ensemble_name!='val_forecast':
      val_forecast = eval(best_ensemble_name).iloc[0:eval(best_ensemble_name).shape[0]-oos_length,:]
      print('oos_forecast before:\n',oos_forecast)
      oos_forecast = pd.DataFrame(eval(best_ensemble_name).iloc[eval(best_ensemble_name).shape[0]-oos_length:,:])
      print('oos_forecast after:\n',oos_forecast)
      multivariate_models = [x for x in model_list if '_mv' in x]
      impact_models=list(set(list(list_of_models.keys())) & (set(multivariate_models)))
      if len(impact_models)>0:
        best_model_index=np.min([np.where(dataf['models']==x) for x in impact_models])
        best_mv_model=dataf['models'].iloc[best_model_index]
        var_coeff=var_coeff[best_mv_model]['forecast']
      else:
        var_coeff=var_coeff[models[0]]['forecast']
        var_coeff['Model']=', '.join(list_of_models)
    else: 
      var_coeff=var_coeff[best_model]['forecast']
  
  else: 
    var_coeff=var_coeff[best_model]['forecast'] 
        
  df_univariate, df_multivariate, df_ensemble = analysis_dataframe.get_analysis_dfs(dataf,oos_forecast_list,fitted_values_list,best_ensemble,val_forecast,ensemble_1,ensemble_2,ensemble_3,ensemble_4)
  val_length = 3
  if val_forecast.shape[0]>val_length:
    val_forecast = val_forecast[(val_forecast.shape[0]-val_length):val_forecast.shape[0]]
  else:
    index=[ti_backup['Header'].values[row_counter-1-val_length+i] for i in range(0,val_length-val_forecast.shape[0])]
    for i in range(0,len(index)):
      extra=pd.DataFrame.from_dict({'date_':[index[i]],'forecast':[0],'type':['no_model'],'Forecast_accuracy':[0],'Actual':[0]})
      val_forecast=pd.concat([extra,val_forecast],axis=0)
  
  val_forecast = val_forecast[(val_forecast.shape[0]-val_length):val_forecast.shape[0]]
  all_forecasts = pd.concat([val_forecast,oos_forecast], axis=0)
  all_forecasts = all_forecasts.reset_index(drop=True)
  print('oos_forecast:\n',oos_forecast)
  return {'dataf': dataf,'all_forecasts': all_forecasts, 'oos_forecast': oos_forecast, 'classifier_train_set':classifier_train_set,'var_coeff':var_coeff,'df_univariate':df_univariate ,'df_multivariate':df_multivariate,'df_ensemble':df_ensemble,'fitted_values_list':fitted_values_list}
