import pandas as pd
import numpy as np
import forecast_engine.modelling.utils.nodes as utils
import forecast_engine.modelling.forecast_engine.forecast_loop as forecast_loop


# Forecast engine

def forecast_engine(ti, months_to_forecast,debug_flag,debug_models,list_of_models,single_series,  all_at_once_flag, validation_window, max_validation, outlier_flag, select_variable_flag):
  
  from pandas import MultiIndex
  import datetime 
  
  ts1 = datetime.datetime.now().timestamp()
  if (single_series == 0):
    ti = pd.Series(ti)

  
  
  series_key = str(ti['index'])
  print(series_key)
  
#   print('ti at 1 is ', ti.index[50:55])
  
  ti1, ti_key, variable_list, series_flag = utils.get_ti_in_shape(ti.copy())
  
  ti = ti[2:]
  ti = ti1.copy()
  ts2 = datetime.datetime.now().timestamp()
    
  ti1 = utils.actual_imputation(ti.copy())
  ti=ti1.copy()
  
  ti2 = utils.driver_imputation(ti1.copy())
  ti=ti2.copy()

  

  
  
  try:
    forecast_out = forecast_loop.forecast_loop_function(ti, series_flag, months_to_forecast, debug_flag, debug_models, list_of_models, variable_list, single_series, all_at_once_flag, validation_window, max_validation, outlier_flag, select_variable_flag)
  except:
    raise Exception('In Forecast Engine: Series '+series_key+' errored out')
  ts3 = datetime.datetime.now().timestamp()
  
  fitted_values_list = forecast_out['fitted_values_list'].copy()  
  var_coeff = forecast_out['var_coeff'].reset_index(drop=True)
  classifier_train_set = forecast_out['classifier_train_set'].reset_index(drop=True)
  dataf = forecast_out['dataf'].reset_index(drop=True)
  ################################################################# FOR ANALYSIS MODULE IN ANY ENV ######################################################################
  df_univariate=forecast_out['df_univariate'].copy()
  df_multivariate=forecast_out['df_multivariate'].copy()
  df_ensemble=forecast_out['df_ensemble'].copy()
  ################################################################# FOR ANALYSIS MODULE IN ANY ENV ######################################################################
  
  forecast_out = forecast_out['forecast_df'].copy()
  
  forecast_out.columns=['DATE', 'Forecast', 'Method', 'Forecast_accuracy', 'Actual', 'timing']
  forecast_out = forecast_out.reset_index(drop=True)


  ti_key = pd.DataFrame([ti_key])
  key_classifier_train_set = pd.concat([ti_key]*classifier_train_set.shape[0],axis=0, ignore_index=True)
  classifier_train_set = pd.concat([key_classifier_train_set,classifier_train_set],axis=1)
  
  ################################################################# FOR ANALYSIS MODULE IN NON-PROD ENV ######################################################################
  key_df_univariate = pd.concat([ti_key]*df_univariate.shape[0],axis=0, ignore_index=True)
  df_univariate = pd.concat([key_df_univariate,df_univariate],axis=1)
  key_df_multivariate = pd.concat([ti_key]*df_multivariate.shape[0],axis=0, ignore_index=True)
  df_multivariate = pd.concat([key_df_multivariate,df_multivariate],axis=1)
  key_df_ensemble = pd.concat([ti_key]*df_ensemble.shape[0],axis=0, ignore_index=True)
  df_ensemble = pd.concat([key_df_ensemble,df_ensemble],axis=1)
  ################################################################# FOR ANALYSIS MODULE IN NON-PROD ENV ######################################################################
  
  try:
    key_var = pd.concat([ti_key]*var_coeff.shape[0],axis=0, ignore_index=True)
    var_coeff = pd.concat([key_var,var_coeff],axis=1)
  except: pass
  
  ti_key = pd.concat([ti_key]*forecast_out.shape[0], ignore_index=True)
  output = pd.concat([ti_key,forecast_out],axis=1)
   
  try:
    var_coeff['key'] = var_coeff['d1_id'].astype(str)+var_coeff['d2_id'].astype(str)+var_coeff['d3_id'].astype(str)+var_coeff['d4_id'].astype(str)+var_coeff['d5_id'].astype(str)+var_coeff['d6_id'].astype(str)+var_coeff['kpi_id'].astype(str)
  except: pass
  
  ti_imputed = ti[ti['Header'].isin(var_coeff['Date'].unique().tolist())]
  ti_imputed = pd.melt(ti_imputed, id_vars = 'Header')
  ti_imputed.columns = ['Date','Driver','Imputed_Value']
  try:
    var_coeff = pd.merge(var_coeff,ti_imputed,on = ['Date','Driver'])
  except:
    var_coeff = pd.DataFrame(columns=['index','d1_id','d2_id','d3_id','d4_id','d5_id','d6_id','kpi_id','currency_code','randNumCol','Model','Date','Driver','Impact','key','Imputed_Value'])
  
  ts4 = datetime.datetime.now().timestamp()
  output['pre_timing'] = ts2 - ts1
  output['mid_timing'] = ts3 - ts2
  output['post_timing'] = ts4 - ts3
  output['total'] = ts4 - ts1
    
  return {'dataf':dataf,'output':output,'classifier_train_set':classifier_train_set,'var_coeff':var_coeff,'df_univariate':df_univariate ,'df_multivariate':df_multivariate, 'df_ensemble':df_ensemble,'fitted_values_list':fitted_values_list}
