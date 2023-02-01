import pandas as pd
import numpy as np

# Analysis df

def get_analysis_dfs(dataf,oos_forecast_list,fitted_values_list,best_ensemble,val_forecast,ensemble_1,ensemble_2,ensemble_3,ensemble_4):
  df_univariate = pd.DataFrame()
  df_multivariate = pd.DataFrame()
  df_ensemble = pd.DataFrame()
  
  best_uv_model = dataf[~dataf['models'].str.contains(r'_mv|_b',regex=True)].iloc[0][0]
  uv_oos_forecast = oos_forecast_list[best_uv_model]['forecast'].copy()
  uv_fitted_values = fitted_values_list[best_uv_model]['forecast'].copy()
  df_univariate = pd.concat([uv_fitted_values,uv_oos_forecast],axis = 0)
  df_univariate = df_univariate.reset_index(drop = True)

  if (len(dataf.index[dataf['models'].str.contains('_mv')]) > 0):
    best_mv_model = dataf[(dataf['models'].str.contains(r'_mv')) & (dataf['models']!= 'xgboost_mv')].iloc[0][0]
    mv_oos_forecast = oos_forecast_list[best_mv_model]['forecast'].copy()
    mv_fitted_values = fitted_values_list[best_mv_model]['forecast'].copy()
    df_multivariate = pd.concat([mv_fitted_values,mv_oos_forecast],axis = 0)
    df_multivariate = df_multivariate.reset_index(drop = True)
  else:
    df_multivariate = df_univariate.copy()
    df_multivariate.iloc[:,1:] = 0
    df_multivariate.loc[:,'method']='no_multivariate'
  
  
    
  try:
    best_ensemble=best_ensemble[~best_ensemble['ensemble_type'].isin(['val_forecast'])]
    best_ensemble_name = best_ensemble.iloc[(np.where(best_ensemble.iloc[:,0]==(best_ensemble.iloc[:,0]).min()))[0].min(),2]     

    df_ensemble = eval(best_ensemble_name)
    df_ensemble=df_ensemble[val_forecast.columns]
    df_ensemble.drop(['Forecast_accuracy','Actual'],axis=1,inplace=True)
    df_ensemble.rename(columns={'type':'method'},inplace=True)
    
  except:
    df_ensemble = df_univariate.copy()
    try: df_ensemble = df_ensemble[df_ensemble['date_']>=np.min(val_forecast['date_'].values)].reset_index(drop=True)
    except: pass
    df_ensemble.iloc[:,1:] = 0
    df_ensemble.loc[:,'method']='no_ensemble'
    
  return df_univariate, df_multivariate, df_ensemble