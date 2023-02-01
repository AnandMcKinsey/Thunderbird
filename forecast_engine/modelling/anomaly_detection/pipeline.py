import forecast_engine.modelling.anomaly_detection.nodes as anomaly
import pandas as pd

def twitteranomalydetection(ti,plot):
  
  import matplotlib.pyplot as plt
  from pyculiarity import detect_ts
  
  ti = ti[['Header','Value']].dropna(subset = ['Value'])
  ti.index  = ti['Header']
  ti = ti['Value']
  
  plt.clf()
  data_check = pd.DataFrame(ti).reset_index()
  data_check.iloc[:,0] = pd.to_datetime(data_check.iloc[:,0])
  data_check.iloc[:,1] = data_check.iloc[:,1].astype(float)
  #print("Shape of the series going into detection - ")
  #print(data_check.shape)
  anomalies = anomaly.detect_ts_local(data_check, max_anoms=0.05, direction='both',e_value=True, alpha=0.001)
  #anomalies['anoms'] = anomalies['anoms']['anoms'].copy()   
  
  data_check.index = data_check['timestamp']
  data_check = pd.Series(data_check['value'])
  anomalies['anoms'].index = pd.to_datetime(anomalies['anoms'].index).strftime('%Y - %m')
  ti_updated = ti.copy()
  
  #if remove==True:
  #  ti_updated = ti[~ti.index.isin(anomalies['anoms'].index)]
  
  if plot==True:
    plt.plot(ti)
    if len(anomalies['anoms']['anoms']):
      plt.plot(anomalies['anoms']['anoms'], 'r*')
    plt.rcParams.update({'font.size': 10})
    return {'ti':ti_updated,'anomalies' : anomalies['anoms'], 'plt': plt}
  else:
    return {'ti':ti_updated,'anomalies' : anomalies['anoms']}