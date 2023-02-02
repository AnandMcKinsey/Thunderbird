
import forecast_engine.modelling.forecast_engine.individual_model_run as individual_model_run


# Model list function

def model_list_function(ti,series_flag,oos_length,list_of_models,index,row_counter,debug_flag,debug_models, n_splits, validation_window, variable_list,single_series):
  
  import datetime
  from copy import deepcopy
  import collections

  ti_mv = ti.copy()
  ti_mv.index=ti_mv['Header']
  ti_mv = ti_mv.drop(['Header'],axis=1)
  ti.index = ti['Header']
  ti = ti['Value'].copy()
  multivariate_models = ['arima_mv','xgboost_mv','random_forest_mv','lightgbm_mv','causalimpact_mv','prophet_mv','linearregression_mv','olsregression_mv']
  model_list=list()
  val_forecast_list=collections.OrderedDict()
  oos_forecast_list=collections.OrderedDict()
  fitted_values_list=collections.OrderedDict()
  var_coeff=collections.OrderedDict()
  
  if (debug_flag == 0) and (single_series==1):
    
    other_models = list(filter(lambda x: '_b' not in x, list_of_models.keys())) 
    
    ts1 = datetime.datetime.now().timestamp()
    from joblib import Parallel, delayed
    temp_var = Parallel(n_jobs=-1)(delayed(individual_model_run.individual_model_run_function)(key,multivariate_models,list_of_models,ti_mv,ti,oos_length,row_counter,debug_models,n_splits,validation_window,variable_list,model_list,val_forecast_list,oos_forecast_list,fitted_values_list,var_coeff) for key in other_models)
    
    for i in range(0,len(temp_var)):
      if (len(temp_var[i][1])>0) and (len(temp_var[i][2])>0):
        model_list.append(temp_var[i][0])
        val_forecast_list[temp_var[i][0]] = list(temp_var[i][1].values())[0]
        oos_forecast_list[temp_var[i][0]] = list(temp_var[i][2].values())[0]
        var_coeff[temp_var[i][0]] = list(temp_var[i][3].values())[0]
        fitted_values_list[temp_var[i][0]] = list(temp_var[i][4].values())[0]    
        fitted_values_list[temp_var[i][0]] = list(temp_var[i][4].values())[0]    
        


    ts2 = datetime.datetime.now().timestamp()
    print(ts2 - ts1)
    
  else:
    for model_name in list_of_models:
      if debug_flag==1: print(model_name)
      ts1 = datetime.datetime.now().timestamp()
      model_name,val_forecast_list,oos_forecast_list,var_coeff,fitted_values_list = individual_model_run.individual_model_run_function(model_name,multivariate_models,list_of_models,ti_mv,ti,oos_length,row_counter,debug_models,n_splits,validation_window,variable_list,model_list,val_forecast_list,oos_forecast_list,fitted_values_list,var_coeff)
      ts2 = datetime.datetime.now().timestamp()
      if debug_flag==1: print(ts2 - ts1)
        
  return {'model_list': model_list, 'val_forecast_list': val_forecast_list, 'oos_forecast_list': oos_forecast_list, 'fitted_values_list':fitted_values_list, 'var_coeff':var_coeff}
