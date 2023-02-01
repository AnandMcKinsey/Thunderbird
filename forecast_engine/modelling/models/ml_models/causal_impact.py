import forecast_engine.modelling.utils.nodes as utils




def causalimpact(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
    from causalimpact import CausalImpact
    from sklearn.metrics import mean_squared_error as mse
    from datetime import datetime
    import numpy as np
    import pandas as pd
    #ti=pd.Series(ti).astype(str).astype(float)    
    oos=ti[row_counter-1:row_counter-1+h]
    ti=ti[:(row_counter-1)]
    ti=ti.truncate(before=ti.index[np.min(np.where(ti.notnull())[0])])
    seasonal_periods=utils.get_seasonal_periods(ti)
    min_series =2*np.abs(np.min(ti))+100
    ti=ti+min_series
  
    y_actual=[]; y_pred=[]; rmse = []; map_error=[]; fitted_train=[]
    forecast1=pd.DataFrame(columns=['date_','forecast','method'])

    tscv = utils.TimeSeriesSplit(n_splits = n_splits, validation_window=validation_window)
    for train_index, test_index in tscv.split(ti):
      cv_train, cv_test = ti.iloc[:,1:].iloc[train_index], ti.iloc[:,1:].iloc[test_index]
      np.random.seed(1234)
      predictions=[]
      train2 = ti.iloc[:,1:].iloc[np.concatenate([train_index,test_index])]
      pre_period, post_period = [0,len(cv_train)-1],[len(cv_train),len(cv_train)+len(cv_test)-1]
      impact=CausalImpact(np.asarray(train2).astype('float'), pre_period, post_period)
      predictions = predictions + list(impact.inferences['preds'].iloc[len(cv_train):len(cv_train)+len(cv_test)].values)
      true_values = cv_test.values
      y_actual=y_actual+list(true_values-min_series)
      y_pred=y_pred+[x-min_series for x in predictions]
      forecast=pd.concat([pd.Series(cv_test.index),pd.Series([x-min_series for x in predictions]), pd.Series(['causalimpact']*len(cv_test))],axis=1)
      forecast.columns=['date_','forecast','method']
      forecast1=pd.concat([forecast1,forecast],axis=0,ignore_index=True)
      
      train_fit = list(impact.inferences['preds'].iloc[:len(cv_train)].values[12:]) 
      fitted_train1=pd.concat([pd.Series(ti.index[train_index[12]:test_index[0]]) ,pd.Series(train_fit-min_series) ,pd.Series(['causalimpact']*len(train_fit) ,name='method')],axis=1)
      fitted_train1.columns=['date_','forecast','method']
      fitted_train.append(fitted_train1)

    np.random.seed(1234)
    pre_period = [0,len(ti)-1]
    post_period = [len(ti),len(ti)+len(oos)-1]  
    impact = CausalImpact(np.asarray(pd.concat([ti,oos],axis=0)).astype('float'),pre_period,post_period,nseasons=[{'period': 12}])
    predictions = impact.inferences['preds'][-h:].reset_index(drop=True).values
    forecast2=pd.concat([pd.Series(oos.index),pd.Series([x-min_series for x in predictions]),pd.Series(['causalimpact']*h)],axis=1)
    forecast2.columns=['date_','forecast','method']
    forecast1['forecast']=forecast1['forecast'].astype('float64')  
    forecast2['forecast']=forecast2['forecast'].astype('float64')
 
    #Impact snippet
    rest_var = variable_list.tolist()
    rest_len = len(rest_var)
    var_coeff = pd.DataFrame()
    for month_ in range(0,len(oos)):
      rest_coeff = pd.concat([pd.Series(['causalimpact']*rest_len), pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len), pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff=pd.concat([var_coeff,rest_coeff],axis=0)
    
    if (np.sum(forecast1['forecast'].isnull())) or (np.sum(forecast2['forecast'].isnull())) or np.sum(np.isinf(forecast1['forecast'])) or np.sum(np.isinf(forecast2['forecast'])):
      raise Exception('NaN in output')
      
    total_fit = list(impact.inferences['preds'].iloc[12:len(ti)].values) 
    total_fit=pd.concat([pd.Series(ti.index[12:]),pd.Series(total_fit-min_series),pd.Series(['causalimpact']*(len(ti)-12))],axis=1)
    total_fit.columns = ['date_', 'forecast','method']
    
    return {'forecast':forecast1,'oos forecast':forecast2,'fitted_values_all':total_fit,'fitted_values_train':fitted_train,'var_coeff':var_coeff}
