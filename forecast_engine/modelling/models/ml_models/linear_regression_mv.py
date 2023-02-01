import forecast_engine.modelling.error_metrics.nodes as error_metrics



def linearregression_mv(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
    import sklearn
    from datetime import datetime
    from sklearn.linear_model import Ridge 
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error as mse
    import numpy as np
    import pandas as pd
  
    ti=pd.DataFrame(ti)
    ti_original_cols=ti.columns
    ti['previous_val'] = ti['Value'].shift(3)
    ti['previous_year_val'] = ti['Value'].shift(12)
    ti['previous_quarter_val'] = ti['Value'].shift(4)
    oos=ti[row_counter-1:row_counter-1+h]
#     print(ti.tail(20))
    prev_val_ind = np.where(pd.isna(oos['previous_val']))[0] + row_counter - 1  
    oos.loc[pd.isna(oos['previous_val']),'previous_val']=ti.loc[:,'previous_val'].values[prev_val_ind-12]
    prev_qtr_ind = np.where(pd.isna(oos['previous_quarter_val']))[0] + row_counter - 1  
    oos.loc[pd.isna(oos['previous_quarter_val']),'previous_quarter_val']=ti.loc[:,'previous_quarter_val'].values[prev_qtr_ind-12]
#     print(oos)
    ti=ti[:(row_counter-1)]
    print(ti.tail(20))
    scalerx = StandardScaler()
    scalery = StandardScaler()
    scalerx.fit(ti.iloc[:row_counter-1,1:].values.astype('float64'))
    scalery.fit(ti.iloc[:row_counter-1,[0]].values.astype('float64'))
    impact_scale = np.concatenate((scalery.scale_, scalerx.scale_), axis=0)
    ti=ti.truncate(before=ti.index[np.min(np.where(ti['previous_year_val'].notnull())[0])])
    y_actual=[]; y_pred=[]; rmse = []; map_error=[]
    Y_train = ti['Value'].values.astype('float64')
    X_train = ti.iloc[:,1:].astype('float64') 
    
    X_train = scalerx.transform(X_train)
    X_test = scalerx.transform(oos.iloc[:,1:])
    ridgeparamlist = [0,0.001,0.01,0.1,1,2,5,8,10,20,50,80,100,1000,10000]
    rmselist = []
    rmselist_fullwindow = []

    y_predict_list = []
    
    
    
    for ridgeparam in ridgeparamlist:
      kf = KFold(n_splits=10)
      y_predict = []
      for train_index, valid_index in kf.split(X_train):
        X_train1, X_valid1 = X_train[train_index,:], X_train[valid_index,:]
        Y_train1 = Y_train[train_index,]
        reg = Ridge(alpha=ridgeparam)
        reg.fit(X_train1, Y_train1)
        y_predict = y_predict + list(reg.predict(X_valid1))

      rmselist.append(error_metrics.wrmse(Y_train[-n_splits*validation_window:],y_predict[-n_splits*validation_window:]))  
      rmselist_fullwindow.append(error_metrics.wrmse(Y_train,y_predict))  
      y_predict_list.append(y_predict)
    best_ridge_param = ridgeparamlist[np.array(rmselist_fullwindow).argmin()]
    y_predict = y_predict_list[np.array(rmselist_fullwindow).argmin()]
    
    predictions = y_predict[-n_splits*validation_window:]
    forecast1=pd.concat([pd.Series(ti.index[-n_splits*validation_window:]),pd.Series(predictions), pd.Series(['linearregression_mv']*n_splits*validation_window)], axis=1)
    forecast1.columns=['date_','forecast','method']      
    reg1 = Ridge(alpha=best_ridge_param)
    reg1.fit(X_train, Y_train)
    
    forecast2=pd.concat([pd.Series(oos.index),pd.Series(reg1.predict(X_test)),pd.Series(['linearregression_mv']*h)],axis=1)
    forecast2.columns=['date_','forecast','method']    
    
    forecast1['forecast']=forecast1['forecast'].astype('float64')  
    forecast2['forecast']=forecast2['forecast'].astype('float64')
    if (np.sum(forecast1['forecast'].isnull())) or (np.sum(forecast2['forecast'].isnull())) or np.sum(np.isinf(forecast1['forecast'])) or np.sum(np.isinf(forecast2['forecast'])):
      raise Exception('NaN in output')

#     print('rmse is', val_accuracy)
#     print('rmselist is', rmselist)
#     print('best is', best_ridge_param)
    #Impact snippet
    var_len =  ti.iloc[:,1:].shape[1]
    
    impact_df = pd.DataFrame()
    for month_ in range(0,len(oos)):
      var_coeff = pd.concat([pd.Series(['linearregression_mv']*var_len),pd.Series([oos.index[month_]]*var_len), pd.Series(ti.iloc[:,1:].columns), pd.DataFrame(((reg1.coef_/impact_scale[1:])*oos.iloc[month_,1:]).values)],axis=1)
#       print(pd.DataFrame(((reg1.coef_/impact_scale[1:])*oos.iloc[i,1:]).values))
      var_coeff.columns=['Model','Date','Driver','Impact']
      rest_var = variable_list[~variable_list.isin(list(ti.iloc[:,1:].columns))].tolist()
      rest_len = len(rest_var)
      rest_coeff = pd.concat([pd.Series(['linearregression_mv']*rest_len),pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len), pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff = pd.concat([var_coeff,rest_coeff],axis=0,ignore_index=True)
      var_coeff=var_coeff[~var_coeff['Driver'].isin(['previous_val','previous_quarter_val','previous_year_val'])]
      
      impact_df=pd.concat([impact_df,var_coeff],axis=0)
#     print(impact_df)  
    var_coeff=impact_df.copy()
#     print(var_coeff)
    total_fit = pd.Series(reg1.predict(X_train)) 
    total_fit=pd.concat([pd.Series(ti.index),pd.Series(total_fit),pd.Series(['linearregression_mv']*(len(ti)))],axis=1)
    total_fit.columns = ['date_', 'forecast','method']
    return {'forecast':forecast1,'oos forecast':forecast2,'var_coeff':var_coeff, 'fitted_values_all':total_fit}

