import forecast_engine.modelling.utils.nodes as utils
import forecast_engine.modelling.error_metrics.nodes as error_metrics



def random_forest_mv(ti,h,row_counter,debug_models,variable_list, n_splits, validation_window): 
    import numpy as np 
    import pandas as pd 
    import re
    from scipy.stats import chi2_contingency

    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    import forecast_engine.modelling.error_metrics.nodes as error_metrics
    from sklearn.feature_selection import SelectFromModel
    from sklearn.metrics import mean_squared_error as mse   

    months_to_forecast = h
    oos = ti[row_counter-1:row_counter-1+h]
    y=pd.Series(ti['Value'])[:-12]
    y_train=y[:len(y)-12]
    y_val=y[len(y)-12:]

    if months_to_forecast == 1: intervals=[3,4,5,6]
    if months_to_forecast > 1: intervals=[months_to_forecast,months_to_forecast+1,months_to_forecast+2,months_to_forecast+3]
    if months_to_forecast == 1: lags = 3
    if months_to_forecast > 1: lags = months_to_forecast

    lag_params=pd.DataFrame()
    for k in intervals:
        lagged_ti=pd.Series(ti.Value.shift(k),name='lag_'+str(k))
        lag_params=pd.concat([lag_params,lagged_ti],axis=1,sort=False)

    rolling_params=pd.DataFrame()
    for k in intervals:
        rolling_mean=pd.Series(ti.Value.fillna(0).rolling(k).mean(),name='rolling_mean_'+str(k))
        rolling_std=pd.Series(ti.Value.fillna(0).rolling(k).std(),name='rolling_std_'+str(k))
        rolling_quantile_75=pd.Series(ti.Value.fillna(0).rolling(k).quantile(.75,interpolation='midpoint'),name='rolling_quantile_75_'+str(k))
        rolling_quantile_25=pd.Series(ti.Value.fillna(0).rolling(k).quantile(.25,interpolation='midpoint'),name='rolling_quantile_25_'+str(k))
        rolling_params=pd.concat([rolling_params,rolling_mean,rolling_std,rolling_quantile_75,rolling_quantile_25],axis=1,sort=False)
        
    other_params=pd.DataFrame()
    month=pd.Series(pd.to_datetime(ti.index),name='month').dt.month
    year=pd.Series(pd.to_datetime(ti.index),name='year').dt.year
    yearmonth=pd.Series(year*100+month,name='yearmonth')
    quarter=pd.Series(pd.to_datetime(ti.index),name='quarter').dt.quarter
    other_params=pd.concat([other_params,month,quarter],axis=1,sort=False)
    other_params.index=ti.index

    other_Xs=ti[[x for x in ti.columns if x!='Value']].shift(lags)

    regressors=pd.concat([lag_params,other_params,other_Xs,rolling_params],axis=1).interpolate(method='linear', axis=0).ffill().bfill()
    X_train=regressors[regressors.index.isin(y_train.index)]
    X_val=regressors[regressors.index.isin(y_val.index)]
    select_oos = regressors[row_counter-1:row_counter-1+h]
    select_oos.replace(0, np.nan, inplace=True)
    select_oos = select_oos.interpolate(method='linear', axis=0).ffill().bfill()






    y_actual=[]
    y_pred=[]
    rmse = []
    trmse=[]
    tmap_error=[]
    map_error=[]

    tscv = utils.TimeSeriesSplit(n_splits = 3, validation_window=3)
    forecast=pd.DataFrame(columns=['date_','forecast','method'])
    for train_index, test_index in tscv.split(X_train):
        cv_train, cv_test = X_train.reset_index(drop=True).loc[train_index], X_train.reset_index(drop=True).loc[test_index]
        gsearch = GridSearchCV(estimator=RandomForestRegressor(random_state=123),
                param_grid= {'max_depth' : [1,3,5,7,9],'n_estimators': [100, 150, 200, 250, 300],'max_leaf_nodes' : [1,3,5,7,9],'bootstrap': [True, False]},
                cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
        gsearch.fit(cv_train.astype('float'), y_train[:len(cv_train)].astype('float'))
        predictions = gsearch.predict(cv_test)
        true_values=  y_train.reset_index(drop = True).loc[test_index]
        y_actual=y_actual+list(true_values)
        y_pred=y_pred+list(predictions)
        forecast1=pd.concat([pd.Series(ti.index[len(cv_train):len(cv_train)+len(cv_test)]),pd.Series(gsearch.predict(cv_test)),pd.Series(['random_forest']*len(cv_test))],axis=1)
        forecast1.columns=['date_','forecast','method']
        forecast=pd.concat([forecast,forecast1],axis=0,ignore_index=True)

    model1=(gsearch.best_estimator_).fit(pd.concat([X_train,X_val]),pd.concat([y_train,y_val]))
    forecast2=pd.concat([pd.Series(oos.index),pd.Series(model1.predict(select_oos)),pd.Series(['random_forest']*h)],axis=1)
    forecast2.columns=['date_','forecast','method']
    forecast.columns=['date_','forecast','method']
    var_coeff = pd.DataFrame()
    total_fit = model1.predict(X_train) 
    total_fit = pd.concat([pd.Series(X_train.index),pd.Series(model1.predict(X_train)),pd.Series(['random_forest']*len(X_train))],axis=1) 
    total_fit.columns = ['date_','forecast','method']
    total_fit = pd.concat([total_fit,forecast2])
    feature_scores = {'random_forest_mv':(pd.Series(model1.feature_importances_, index=regressors.columns).sort_values(ascending=False))}

    return {'forecast':forecast,'oos forecast':forecast2,'var_coeff':var_coeff,'fitted_values_all':total_fit,"feature_scores":feature_scores}
    