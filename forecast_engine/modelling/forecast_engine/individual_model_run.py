
# Individual model run

def individual_model_run_function(model_name,multivariate_models,list_of_models,ti_mv,ti,oos_length,row_counter,debug_models,n_splits,validation_window,variable_list,model_list,original_val_forecast_list,original_oos_forecast_list,original_fitted_values_list,var_coeff):

  from copy import deepcopy
  import collections
  
  val_forecast_list = deepcopy(original_val_forecast_list)
  oos_forecast_list = deepcopy(original_oos_forecast_list)
  fitted_values_list = deepcopy(original_fitted_values_list)
  feature_scores= collections.OrderedDict()
  try:
    if model_name in multivariate_models:
      val_forecast_list[model_name] = list_of_models[model_name](ti_mv,oos_length,row_counter,debug_models,variable_list, n_splits, validation_window)
    else:
      val_forecast_list[model_name] = list_of_models[model_name](ti,oos_length,row_counter,debug_models,variable_list,n_splits,validation_window)

      
    oos_forecast_list[model_name] = deepcopy({'forecast':val_forecast_list[model_name]['oos forecast']})
    var_coeff[model_name] = deepcopy({'forecast':val_forecast_list[model_name]['var_coeff']})
    fitted_values_list[model_name] = deepcopy({'forecast':val_forecast_list[model_name]['fitted_values_all']})
    # feature_scores['random_forest_mv'] = deepcopy({'forecast':val_forecast_list['model_name']['feature_scores']})

    del val_forecast_list[model_name]['oos forecast']
    del val_forecast_list[model_name]['var_coeff']
    model_list.append(model_name)
    
  except:
    print(f"{model_name} : errored_out")

  
  return model_name,val_forecast_list,oos_forecast_list,var_coeff,fitted_values_list
