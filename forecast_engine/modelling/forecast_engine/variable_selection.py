# Variable selection

def variable_selection(ti1,row_counter):
  from sklearn.feature_selection import mutual_info_regression
  from sklearn.feature_selection import GenericUnivariateSelect
  try:
    if ti1.shape[1]>2:
      y = ti1['Value'].iloc[:row_counter-1].astype('float64')
      X = ti1.iloc[:row_counter-1,2:].astype('float64')
      mutual_information = mutual_info_regression(X, y)
      select=GenericUnivariateSelect(score_func=mutual_info_regression, mode='k_best', param=3).fit(X,y)
      ti1=ti1[[*['Header','Value'],*X.columns[select.get_support()]]]
  except:pass
  return ti1