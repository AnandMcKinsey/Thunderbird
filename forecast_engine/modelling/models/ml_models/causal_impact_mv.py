
import forecast_engine.modelling.utils.nodes as utils



def causalimpact_mv(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
    from causalimpact import CausalImpact
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.preprocessing import StandardScaler
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import pickle
    
    scaler = StandardScaler()
    impact_scale = scaler.fit(ti.iloc[:row_counter-1,1:].values).scale_
    
    oos=ti[row_counter-1:row_counter-1+h]
    ti=ti[:(row_counter-1)]
    ti_orig=pd.concat([ti,oos],axis=0)
    ti_orig=ti_orig.fillna(0)
    
    ti=ti.truncate(before=ti.index[np.min(np.where(ti.notnull())[0])])
    min_series =0#2*np.abs(np.min(ti['Value']))+100
    ti['Value']=ti['Value']+min_series
    y_actual=[]; y_pred=[]; rmse = []; map_error=[]
    forecast1=pd.DataFrame(columns=['date_','forecast','method'])

    tscv = utils.TimeSeriesSplit(n_splits = n_splits, validation_window=validation_window)
    for train_index, test_index in tscv.split(ti):
      cv_train, cv_test = ti.iloc[:,1:].iloc[train_index], ti.iloc[:,1:].iloc[test_index]
      np.random.seed(1234)
      predictions=[]
      train2 = ti.iloc[:,1:].iloc[np.concatenate([train_index,test_index])]
      pre_period, post_period = [0,len(cv_train)-1],[len(cv_train),len(cv_train)+len(cv_test)-1]
     # if 'causalimpact_mv' in debug_models: print(train2)
      impact=CausalImpact(np.asarray(train2).astype('float'), pre_period, post_period)
      
      predictions = predictions + list(impact.inferences['preds'].iloc[len(cv_train):len(cv_train)+len(cv_test)].values)
      true_values = cv_test['Value'].values
      y_actual=y_actual+list(true_values-min_series)
      y_pred=y_pred+[x-min_series for x in predictions]
      forecast=pd.concat([pd.Series(cv_test.index),pd.Series([x-min_series for x in predictions]), pd.Series(['causalimpact_mv']*len(cv_test))],axis=1)
      forecast.columns=['date_','forecast','method']
      forecast1=pd.concat([forecast1,forecast],axis=0,ignore_index=True)
   # print(impact.summary())
    np.random.seed(1234)
    pre_period = [0,len(ti)-1]
    post_period = [len(ti),len(ti)+len(oos)-1]  
  #  print(pd.concat([ti,oos],axis=0))
    impact = CausalImpact(np.asarray(ti_orig).astype('float'),pre_period,post_period,nseasons=[{'period': 12}])

    predictions = impact.inferences['preds'][-h:].reset_index(drop=True).values
    forecast2=pd.concat([pd.Series(oos.index),pd.Series([x-min_series for x in predictions]),pd.Series(['causalimpact_mv']*h)],axis=1)
    forecast2.columns=['date_','forecast','method']

    #Impact snippet
    var_len =  ti.iloc[:,1:].shape[1]
    impact_df = pd.DataFrame()
    for month_ in range(0,len(oos)):
      var_coeff = pd.concat([pd.Series(['causalimpact_mv']*var_len), pd.Series([oos.index[month_]]*var_len), pd.Series(ti.iloc[:,1:].columns), pd.DataFrame(((impact.trained_model.params[[x for x in impact.trained_model.params.keys() if 'beta' in x]].values/impact_scale[1:]*impact_scale[0])*oos.iloc[month_,1:]).values)],axis=1)

      var_coeff.columns=['Model','Date','Driver','Impact']
      rest_var = variable_list[~variable_list.isin(list(ti.iloc[:,1:].columns))].tolist()
      rest_len = len(rest_var)
      rest_coeff = pd.concat([pd.Series(['causalimpact_mv']*rest_len),pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len),pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff = pd.concat([var_coeff,rest_coeff],axis=0,ignore_index=True)
      impact_df=pd.concat([impact_df,var_coeff],axis=0)
#     print(impact_df)  
    var_coeff=impact_df.copy()
#     print(var_coeff)
    total_fit = pd.Series(impact.inferences['preds'].iloc[12:len(ti)].values) 
    total_fit=pd.concat([pd.Series(ti.index[12:]),pd.Series(total_fit-min_series),pd.Series(['causalimpact_mv']*(len(ti)-12))],axis=1)
    total_fit.columns = ['date_', 'forecast','method']
    
    return {'forecast':forecast1,'oos forecast':forecast2,'var_coeff':var_coeff,'fitted_values_all':total_fit}
