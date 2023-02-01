



def olsregression_mv(ti,h,row_counter,debug_models,variable_list,n_splits=3,validation_window=3):
    import sklearn
    import statsmodels.api as sm
    from datetime import datetime
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error as mse
    import pandas as pd
    import numpy as np
    
#     import pickle
#     if 'olsregression_mv' in debug_models:
#       #print(ti)
#       with open(ti_mv_pickle_path, 'wb') as f:  # Python 3: open(..., 'wb')
#         pickle.dump([ti, h, row_counter, n_splits, variable_list, validation_window], f)
# #   
   # ti=ti.drop(['Market_Value'],axis=1)
#     print("ti in ols:\n",ti)
    ti_original_cols=ti.columns
    ti=ti.copy()
    ti['previous_val'] = ti['Value'].shift(1)
    ti['previous_year_val'] = ti['Value'].shift(12)
    ti['previous_quarter_val'] = ti['Value'].shift(3)
    oos=ti[row_counter-1:row_counter-1+h]
    ti=ti[:(row_counter-1)]
    
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
  #  X_test = oos.iloc[:,1:]
    X_train = np.concatenate((np.ones((X_train.shape[0],1)),X_train), axis=1)
    X_test = np.concatenate((np.ones((X_test.shape[0],1)),X_test), axis=1)

    
    kf = KFold(n_splits=10)
    y_predict = []
    for train_index, valid_index in kf.split(X_train):
      X_train1, X_valid1 = X_train[train_index,:], X_train[valid_index,:]
      Y_train1 = Y_train[train_index,]
      reg = sm.OLS(Y_train1, X_train1)
      results = reg.fit()
      y_predict = y_predict + list(results.predict(X_valid1))
    
    predictions = y_predict[-n_splits*validation_window:]
    forecast1=pd.concat([pd.Series(ti.index[-n_splits*validation_window:]),pd.Series(predictions), pd.Series(['olsregression_mv']*n_splits*validation_window)],axis=1)
    forecast1.columns=['date_','forecast','method']      
      
    
    reg1 = sm.OLS(Y_train, X_train)
    results1 = reg1.fit()
    forecast2=pd.concat([pd.Series(oos.index),pd.Series(results1.predict(X_test)),pd.Series(['olsregression_mv']*h)],axis=1)
    forecast2.columns=['date_','forecast','method']    
    forecast1['forecast']=forecast1['forecast'].astype('float64')  
    forecast2['forecast']=forecast2['forecast'].astype('float64')
    
    if 'olsregression_mv' in debug_models:
      print(results1.summary())
    #Impact snippet
    var_len =  ti.iloc[:,1:].shape[1]
    
    impact_df = pd.DataFrame()
    for month_ in range(0,len(oos)):
      var_coeff = pd.concat([pd.Series(['olsregression_mv']*var_len),pd.Series([oos.index[month_]]*var_len),pd.Series(ti.iloc[:,1:].columns),pd.DataFrame(((results1.params[1:]/impact_scale[1:])*oos.iloc[month_,1:]).values)],axis=1)
      var_coeff.columns=['Model','Date','Driver','Impact']
      rest_var = variable_list[~variable_list.isin(list(ti.iloc[:,1:].columns))].tolist()
      rest_len = len(rest_var)
      rest_coeff = pd.concat([pd.Series(['olsregression_mv']*rest_len),pd.Series([pd.Series(oos.index).iloc[month_]]*rest_len),pd.Series(rest_var),pd.Series([0]*rest_len)],axis=1)
      rest_coeff.columns=['Model','Date','Driver','Impact']
      var_coeff = pd.concat([var_coeff,rest_coeff],axis=0,ignore_index=True)
      var_coeff=var_coeff[~var_coeff['Driver'].isin(['previous_val','previous_quarter_val','previous_year_val'])]
      impact_df=pd.concat([impact_df,var_coeff],axis=0)
#     print(impact_df)  
    var_coeff=impact_df.copy()
      
    total_fit = pd.Series(results1.predict(X_train)) 
    total_fit=pd.concat([pd.Series(ti.index),pd.Series(total_fit),pd.Series(['olsregression_mv']*(len(ti)))],axis=1)
    total_fit.columns = ['date_', 'forecast','method']
    
    if (np.sum(forecast1['forecast'].isnull())) or (np.sum(forecast2['forecast'].isnull())) or np.sum(np.isinf(forecast1['forecast'])) or np.sum(np.isinf(forecast2['forecast'])):
      raise Exception('NaN in output')
    
    return {'forecast':forecast1,'oos forecast':forecast2,'var_coeff':var_coeff, 'fitted_values_all':total_fit}