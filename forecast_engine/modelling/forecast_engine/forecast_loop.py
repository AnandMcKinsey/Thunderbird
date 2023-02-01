
import pandas as pd
import numpy as np
import forecast_engine.modelling.forecast_engine.model_comparison as model_comparison


def forecast_loop_function(ti, series_flag, months_to_forecast, debug_flag, debug_models, list_of_models, variable_list, single_series, all_at_once_flag, validation_window, max_validation, outlier_flag,select_variable_flag):
  
  import datetime
  ts1 = datetime.datetime.now().timestamp()
  

  
  row_counter = ti.shape[0] +1
  fitted_values_list=list()
  ################################################################# FOR ANALYSIS MODULE IN ANY ENV ######################################################################
  df_multivariate = pd.DataFrame()
  df_univariate = pd.DataFrame()
  df_ensemble = pd.DataFrame()
  ################################################################# FOR ANALYSIS MODULE IN ANY ENV ######################################################################
  
  oos_length = 1
  if all_at_once_flag == 1:
    oos_length = months_to_forecast
    months_to_forecast = 1
  cols_proper = list(ti.columns)
  for i in range(0,months_to_forecast):
    if debug_flag==1: print('Run Number '+str(i))
    ti = ti.append(pd.Series(np.nan),ignore_index=True)    
    ti = ti.append(pd.Series(np.nan),ignore_index=True)    
    ti = ti.append(pd.Series(np.nan),ignore_index=True)    
    ti = ti.append(pd.Series(np.nan),ignore_index=True)    
    ti = ti.append(pd.Series(np.nan),ignore_index=True)    
    ti = ti.append(pd.Series(np.nan),ignore_index=True)    
    ti = ti.append(pd.Series(np.nan),ignore_index=True)    
    ti = ti.append(pd.Series(np.nan),ignore_index=True)    
    ti = ti.append(pd.Series(np.nan),ignore_index=True)    
    ti = ti.append(pd.Series(np.nan),ignore_index=True)    
    ti = ti.append(pd.Series(np.nan),ignore_index=True)    
    ti = ti.append(pd.Series(np.nan),ignore_index=True)    
    
    ti = ti[cols_proper]
    ti['Header'] = pd.to_datetime(ti['Header'],format="%Y - %m")
    ti['Header'] = pd.date_range(start=ti['Header'][0],periods=(ti.shape[0]),freq='MS').strftime('%Y - %m')
    print(ti.tail(20))
    op = model_comparison.model_comparison_function(ti.copy(), series_flag, row_counter,  list_of_models, debug_flag, debug_models, validation_window, max_validation, variable_list, single_series, oos_length, outlier_flag, select_variable_flag)
    
    ################################################################# FOR ANALYSIS MODULE IN NON-PROD ENV ######################################################################
    df_univariate=pd.concat([df_univariate,pd.concat([pd.Series([i]*len(op['df_univariate'])),op['df_univariate']],axis=1)],axis=0)
    df_univariate=df_univariate.reset_index(drop=True)
    df_multivariate=pd.concat([df_multivariate,pd.concat([pd.Series([i]*len(op['df_multivariate'])),op['df_multivariate']],axis=1)],axis=0)
    df_multivariate=df_multivariate.reset_index(drop=True)
  
    df_ensemble=pd.concat([df_ensemble,pd.concat([pd.Series([i]*len(op['df_ensemble'])),op['df_ensemble']],axis=1)],axis=0)
    df_ensemble=df_ensemble.reset_index(drop=True)
    
    ################################################################# FOR ANALYSIS MODULE IN NON-PROD ENV ######################################################################
    
    var_coeff = op['var_coeff'].copy()
    var_coeff = var_coeff.reset_index(drop=True)
    dataf = op['dataf'].copy()
    dataf = dataf.reset_index(drop=True)
    all_forecasts = op['all_forecasts'].copy()
    all_forecasts = all_forecasts.reset_index(drop=True)
    oos_forecast = op['oos_forecast'].copy()
    oos_forecast = oos_forecast.reset_index(drop=True)
    op['classifier_train_set']['date_'] = np.nan
    op['classifier_train_set']['date_'].iloc[0] = oos_forecast['date_'].iloc[0]
    
    if(i == 0):
      forecast_df = all_forecasts.copy()
      classifier_train_set = op['classifier_train_set'].copy()
      coeffs = var_coeff.copy()
      
    else:
      forecast_df = pd.concat([forecast_df,pd.DataFrame(all_forecasts.iloc[(all_forecasts.shape[0]-1),:]).T])
      classifier_train_set = pd.concat([classifier_train_set,op['classifier_train_set']],axis=0)
      coeffs = pd.concat([coeffs,op['var_coeff']],axis=0)
      
    fitted_values_list.append(op['fitted_values_list'])
    ti['Value'][(row_counter)-1] = oos_forecast['forecast'][0]
    row_counter = row_counter+1
    
    ts2 = datetime.datetime.now().timestamp()
    forecast_df['timing'] = ts2 - ts1

  return {'dataf': dataf,'forecast_df':forecast_df,'classifier_train_set':classifier_train_set,'var_coeff':coeffs,'df_univariate':df_univariate ,'df_multivariate':df_multivariate,'df_ensemble':df_ensemble,'fitted_values_list':fitted_values_list}
