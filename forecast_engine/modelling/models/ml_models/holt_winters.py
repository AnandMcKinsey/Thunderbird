
import forecast_engine.modelling.utils.nodes as utils

import numpy as np
import pandas as pd

class holt_winters_model:
  def __init__(self,oos_length=1,row_counter=12,n_splits=3,validation_window=3,model_name='hw_ets_NFN'):
    self.oos_length=oos_length
    self.row_counter=row_counter
    self.n_splits=n_splits
    self.validation_window=validation_window
    self.model_name=model_name
    components = model_name.split('_')[-1]
    components_dict = {'N':None,'A':'add','M':'mul','F':False,'T':True}
    self.error = 'add'
    self.trend = components_dict[components[0]]
    self.damped = components_dict[components[1]]
    self.seasonality = components_dict[components[2]]
    self.start_index = 0
    self.scale_factor = 1
    self.min_series = 1e-6
    
  def scaling(self,ti,add_min_series=True):
    min_ = np.min(ti.values)
    max_ = np.max(ti.values)
    self.scale_factor = np.abs(max_-min_)
    ti = ti/self.scale_factor
    if add_min_series:
      self.min_series =2*np.abs(np.min(ti))+1e-5
      ti=ti+self.min_series
    return ti
  
  def descaling(self,ti,subtract_min_series=True):
    if subtract_min_series:
      ti=ti-self.min_series
    ti=ti*self.scale_factor
    return ti  
        
  def cross_validation(self,ti):
    seasonal_periods=utils.get_seasonal_periods(ti)
    if self.seasonality is not None:
        self.start_index=seasonal_periods
    else: 
        self.start_index=0
    ti = ti.astype('float64')
    ti=ti[:(self.row_counter-1)]
    ti=ti.truncate(before=ti.index[np.min(np.where(ti.notnull())[0])])
    y_pred=[]; map_error=[]; count=0; fitted_train=[]
    ti=self.scaling(ti,True)
    forecast1=pd.DataFrame(columns=['date_','forecast','method'])
    tscv = utils.TimeSeriesSplit(n_splits = self.n_splits, validation_window=self.validation_window)
    
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as ets
    for train_index, test_index in tscv.split(ti):
      cv_train, cv_test = ti.iloc[train_index], ti.iloc[test_index]
      np.random.seed(1234)
      if count==0:
        fit = ets(cv_train.values,trend=self.trend,damped=self.damped,seasonal=self.seasonality,seasonal_periods=seasonal_periods).fit(optimized=True, start_params=None)
        count = count + 1
        start_params = fit.mle_retvals.x
      else:
        fit = ets(cv_train.values,trend=self.trend,damped=self.damped,seasonal=self.seasonality,seasonal_periods=seasonal_periods).fit(optimized=True, start_params=start_params)
      if 'success' in dir(fit.mle_retvals):
        success = fit.mle_retvals.success
      else:
        success = fit.mle_retvals.lowest_optimization_result.success
      if not success:
        print('Optimization not converged')#Exception('Optimization not converged')  
      start_params = fit.mle_retvals.x  
        
      predictions = fit.predict(start=len(cv_train), end=len(cv_train)+len(cv_test)-1)
      forecast=pd.concat([pd.Series(ti.index[len(cv_train):len(cv_train)+len(cv_test)]), pd.Series(self.descaling(predictions,True)), pd.Series([self.model_name]*len(cv_test))],axis=1)
      forecast.columns=['date_','forecast','method']
      forecast1=pd.concat([forecast1,forecast],axis=0,ignore_index=True)
      
      train_fit = list(self.descaling(fit.predict(start=train_index[self.start_index], end = train_index[-1]),True))
      fitted_train1=pd.concat([pd.Series(ti.index[train_index[self.start_index]:test_index[0]]) ,pd.Series(train_fit) ,pd.Series([self.model_name]*len(train_fit) ,name='method')],axis=1)
      fitted_train1.columns=['date_','forecast','method']
      fitted_train.append(fitted_train1)
      
    forecast1['forecast']=forecast1['forecast'].astype('float64')  
    if (np.sum(forecast1['forecast'].isnull())) or np.sum(np.isinf(forecast1['forecast'])):
      raise Exception('NaN in output')
    return fitted_train, forecast1, start_params

  def oos_forecast(self,ti, start_params):
    np.random.seed(1234)
    ti=ti.astype('float64')
    oos=ti[self.row_counter-1:self.row_counter-1+self.oos_length]
    ti=ti[:(self.row_counter-1)]
    ti=ti.truncate(before=ti.index[np.min(np.where(ti.notnull())[0])])
    seasonal_periods=utils.get_seasonal_periods(ti)
    ti=self.scaling(ti,True)
    
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as ets
    fit2=ets(ti.values,trend=self.trend,damped=self.damped,seasonal=self.seasonality,seasonal_periods=seasonal_periods).fit(optimized=True, start_params = start_params)
    if 'success' in dir(fit2.mle_retvals):
      success = fit2.mle_retvals.success
    else:
      success = fit2.mle_retvals.lowest_optimization_result.success
    if not success:
      raise Exception('Optimization not converged')  
    predictions=fit2.predict(start=len(ti), end=len(ti)+self.oos_length-1)
    total_fit = fit2.predict(start=self.start_index, end=len(ti)-1)
      
    forecast2=pd.concat([pd.Series(oos.index),pd.Series(self.descaling(predictions,True)),pd.Series([self.model_name]*self.oos_length)],axis=1)
    forecast2.columns=['date_','forecast','method']
    forecast2['forecast']=forecast2['forecast'].astype('float64')
    if (np.sum(forecast2['forecast'].isnull())) or np.sum(np.isinf(forecast2['forecast'])):
      raise Exception('NaN in output')        
    total_fit=pd.concat([pd.Series(ti.index[self.start_index:]), pd.Series(self.descaling(total_fit,True)), pd.Series([self.model_name]*len(total_fit))], axis=1)
    total_fit.columns = ['date_', 'forecast','method']
    return total_fit, forecast2
    
  def impact_calculation(self,ti,variable_list):
    oos=ti[self.row_counter-1:self.row_counter-1+self.oos_length]
    rest_var = variable_list.tolist()
    rest_len = len(rest_var)
    var_coeff = pd.DataFrame()
    for month_ in range(0,len(oos)):
      rest_coeff = pd.concat([pd.Series([self.model_name]*rest_len), pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len), pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff=pd.concat([var_coeff,rest_coeff],axis=0)
    return var_coeff

def hw_ets_NFN(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_NFN')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list) 
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data') 
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_NFA(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_NFA')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_NFM(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_NFM')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_AFA(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_AFA')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_AFM(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_AFM')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_AFN(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_AFN')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_ATA(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_ATA')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_ATM(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_ATM')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_ATN(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_ATN')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_MFA(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_MFA')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_MFM(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_MFM')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_MFN(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_MFN')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_MTA(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_MTA')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_MTM(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_MTM')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_MTN(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_MTN')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_r_MNN(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_r_MNN')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_r_MNA(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_r_MNA')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_r_MNM(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_r_MNM')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_r_MAN(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_r_MAN')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_r_MAA(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_r_MAA')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_r_MAM(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_r_MAM')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_r_MMN(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_r_MMN')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_r_MMA(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_r_MMA')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}

def hw_ets_r_MMM(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
  hw=holt_winters_model(oos_length=h,row_counter=row_counter,n_splits=n_splits,validation_window=validation_window,model_name='hw_ets_r_MMM')
  fitted_train, forecast1, start_params=hw.cross_validation(ti)
  total_fit, forecast2=hw.oos_forecast(ti, start_params)
  var_coeff = hw.impact_calculation(ti,variable_list)
  if np.mean(np.abs(forecast2['forecast'].values)) > 100*np.max(np.abs(forecast1['forecast'].values)):
      raise Exception('Forecast is more than 100 times max data')  
  return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}