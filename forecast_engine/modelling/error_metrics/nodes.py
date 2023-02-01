import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse

#Error metrics
def mape(y_true, y_pred):
  if pd.Series(pd.Series(y_true)==0).any(): return np.inf
  else:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def wmape(y_true, y_pred):
  if pd.Series(pd.Series(y_true)==0).any(): return np.inf
  else:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    wt=np.array([np.float64(0)]*len(y_true))
    for i in range(0,len(y_true)):
      try:wt[len(y_true)-1-i]=np.exp(-1/(int((len(y_true)-i)/3)))
      except: wt[i]=np.exp(0)
    return np.mean(wt*np.abs((y_true - y_pred) / y_true)) * 100

def rmse(y_true, y_pred):
  return np.sqrt(mse(np.asarray(y_true),np.asarray(y_pred)))


def wrmse(y_true, y_pred):
  wt=np.array([np.float64(0)]*len(y_true))
  for i in range(0,len(y_true)):
      try:wt[i]=np.exp(-1/(int((len(y_true)-i)/3)))
      except: wt[i]=np.exp(0)
  return np.sqrt(mse(np.asarray(y_true),np.asarray(y_pred),sample_weight=wt))

def smape(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  y_true[y_true==0] = 1e-3
  y_pred[y_pred==0] = 1e-3
  
  return np.mean(np.abs(y_true - y_pred)/(np.abs(y_true) + np.abs(y_pred))) *100

def smape_weight_function(ti,oos_month,anomalous_months):
  last_index=np.where(ti.index==oos_month)[0][0]
  
  z = ti[:last_index]
  z = z[~z.index.isin(anomalous_months)]
  
  z['Value'] = z['Value'].astype('float64')
  z['Month'] = z.index.map(lambda x: x[7:])
  z['Year'] = z.index.map(lambda x: x[:4])

  year_mean = z.groupby('Year').agg({'Value': lambda x: np.abs(np.mean(x))}).reset_index()
  year_mean.rename({'Value': 'Year_mean_value'}, axis=1, inplace=True)

  z = pd.merge(z, year_mean, left_on='Year', right_on = 'Year', how='left')
  z['SI'] = np.abs(z['Value']/z['Year_mean_value'])

  SI = z.groupby('Month').agg({'SI': np.mean})
  
  SI['SI'] = np.roll(SI.values, shift=-len(SI.index)+int(ti.index[last_index][-2:])-1, axis=0)
  SI.index = np.roll(SI.index.values, shift=-len(SI.index)+int(ti.index[last_index][-2:])-1, axis=0)

  distance = np.abs(SI['SI'] - SI['SI'][0])
  weights = pd.Series(1./(1 + distance),index=SI.index,name='weights')
  return weights

def wsmape(y_true,y_pred, smape_weights):
  y_true=np.asarray(y_true,dtype='float64')
  y_pred=np.asarray(y_pred,dtype='float64')
  y_true[y_true==0] = 1e-3
  y_pred[y_pred==0] = 1e-3
  
  if (len(smape_weights)==0) or (len(y_true)==0) or (len(y_pred)==0):
    return 100.0
  else: return np.average(np.abs(y_true - y_pred)/(np.abs(y_true) + np.abs(y_pred)),weights=smape_weights)*100

