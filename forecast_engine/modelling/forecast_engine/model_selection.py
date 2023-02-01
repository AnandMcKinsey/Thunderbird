import forecast_engine.modelling.models.ml_models.naive_models as naive_models
import forecast_engine.modelling.models.ml_models.croston as croston


#Model Selection

def model_selection_function(ti,series_flag,list_of_models,debug_flag):
  import collections
  #List of Models for Flag 1 & 4
  if ((series_flag==1) | (series_flag==4)):  
    index = ti['Value'].first_valid_index()
  
  #List of Models for Flag 2
  elif series_flag==2:
    list_of_models = {'zero':naive_models.zero}
    index = ti['Value'].first_valid_index()
    ti['Value'] = ti['Value'][index:].fillna(0)
    
  #List of Models for Flag 3
  elif series_flag==3:
    list_of_models = {'mean_model':naive_models.mean_model,'zero':naive_models.zero,'median_model':naive_models.median_model,'poor_naive':naive_models.poor_naive,'snaive':naive_models.snaive,'snaive_twoseasons':naive_models.snaive_twoseasons,'croston':croston.croston}
    index = ti['Value'].first_valid_index()
    ti['Value'] = ti['Value'][index:].fillna(0)
  
  #List of Models for Flag 5
  elif series_flag==5:
    list_of_models = {'mean_model':naive_models.mean_model,'zero':naive_models.zero}
    index = ti['Value'].first_valid_index()
    ti['Value'] = ti['Value'][index:].fillna(0)
  
  return ti, index, list_of_models