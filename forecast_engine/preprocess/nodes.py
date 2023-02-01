from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import os
import datetime
import statsmodels.api as sm
import os
import ruptures as rpt


month_map = {'1':'01', '2':'02', '3':'03', '4':'04', '5':'05', '6':'06', '7':'07', '8':'08', '9':'09'}
month_map_2 = {'JAN':'01', 'FEB':'02', 'MAR':'03', 'APR':'04', 'MAY':'05', 'JUN':'06', 'JUL':'07', 'AUG':'08', 'SEP':'09','OCT':'10','NOV':'11','DEC':'12'}

def read_files(filenames, folder_path ,header, skip_rows, sheet = 'Sheet1' ):
    # TODO: add docstring
    def read_file(filename,header, skip_rows,sheet = sheet, folder=None ):
        # TODO: add docstring
        if folder is not None:
            filename = os.path.join(folder, filename)
        df = pd.read_excel(filename,sheet_name = sheet,header = header,skiprows = skip_rows )
        df.reset_index(drop=True, inplace=True)
        return df

    merged_df = pd.concat([read_file(f,header,skip_rows,sheet,folder=folder_path ) for f in filenames])
    return merged_df

def read_prep_sales_order(path, year_list = ['2019','2020','2021','2022']):
   # TODO: add docstring
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    sales_data = read_files(onlyfiles,path,header=0,skip_rows=0)
    sales_data = sales_data[sales_data['Requested deliv.date'] != "#"]
    sales_data = sales_data[sales_data['Requested deliv.date'].notna()]
    sales_data['month'] = sales_data['Requested deliv.date'].apply(lambda x : x.split('.')[1])
    sales_data['month'] = sales_data['month'].map(lambda x: month_map.get(x, x))

    sales_data['year'] = sales_data['Requested deliv.date'].apply(lambda x : x.split('.')[2])

    sales_data['version'] = sales_data['year'] + " - " + sales_data['month'] 
    # sales_data['version'] = sales_data['version'].apply(lambda x: datetime.datetime.strptime(x, "%Y - %m").date() )

    sales_data = sales_data[sales_data['Order Qty. (SU)']!="*"]
    sales_data = sales_data[sales_data['Material']!="#"]
    sales_data['Order Qty. (SU)'] = sales_data['Order Qty. (SU)'].astype(float)
    sales_data = sales_data[sales_data['Order Qty. (SU)']>0]

    sales_data['order_value'] = sales_data['Net Price based on SU'] * sales_data['Order Qty. (SU)']
    sales = sales_data.groupby(['Material','version','year','month','Bill-to party', 'Sales document']).agg({'Order Qty. (SU)':sum}).reset_index()
    
    sales_data = sales_data.groupby(['Material','version','year','month']).agg({'Order Qty. (SU)':sum}).reset_index()
    lst_filt = [i for e in year_list for i in list(sales_data['version'].unique()) if e in str(i)]
    sales_data = sales_data[sales_data['version'].isin(lst_filt)]
    sales = sales[sales['version'].isin(lst_filt)]

    df = sales_data.pivot(index="Material",columns="version", values='Order Qty. (SU)') \
       .reset_index().rename_axis(None, axis=1)

    return sales, df


def read_prep_weather(path, country = ['Mexico'], weather_cols = ['Avg Temp (C)','Max Temp (C)','Min Temp (C)','Relative Humidity (%)','Precipitation (mm)']):
    # TODO: add docstring
    wthr_data = pd.read_excel(path,sheet_name = 'Sheet1')
    wthr_data = wthr_data[wthr_data['Country'].isin(country)]
    wthr_data['month'] = wthr_data['Date'].astype(str).apply(lambda x : x.split('-')[1])
    wthr_data['year'] = wthr_data['Date'].astype(str).apply(lambda x : x.split('-')[0])
    wthr_data['version'] = wthr_data['year'] + " - " + wthr_data['month'] 
    # wthr_data['version'] = wthr_data['version'].apply(lambda x: datetime.datetime.strptime(x, "%Y - %m").date() )

    wthr_data = wthr_data.groupby(['version','Country'])[weather_cols].apply(lambda x : x.mean()).reset_index()

    wthr_data = wthr_data.melt(id_vars=['version','Country'], var_name='weather', value_name='value')
    wthr_data['key'] = wthr_data['version'].astype(str) + "_" + wthr_data['weather']
    col_order = ['Country']
    wthr_data = wthr_data.sort_values(['weather','version'],ascending=True)
    col_order.extend(list(wthr_data['key']))
 
    wthr_data =  wthr_data.pivot(index="Country",columns="key", values='value') \
            .reset_index().rename_axis(None, axis=1)
    wthr_data =  wthr_data.reindex(col_order, axis=1)

    return wthr_data


def read_macro_files(path,  country = ['Mexico']):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    # TODO: add docstring
    def read_file(path,country,filename):
        df = pd.read_csv(os.path.join(path,filename))
        df['Month'] = df['Month'].astype(str)
        df['Month'] = df['Month'].map(lambda x: month_map.get(x, x))
        df['version'] = df['year'].astype(str) + " - " + df['Month'].astype(str)
        # df['version'] = df['version'].apply(lambda x: datetime.datetime.strptime(x, "%Y - %m").date() )
        df['key'] = df['version'].astype(str) + "_" + filename.split('.')[0]
        col_order = ['Country']
        df['Country'] = country[0]
        df = df.sort_values(['version'],ascending=True)
        col_order.extend(list(df['key']))
        df =  df.pivot(index="Country",columns="key", values=df.columns[0]) \
            .reset_index().rename_axis(None, axis=1)
        df =  df.reindex(col_order, axis=1)
        return df

    merged_df = pd.concat([read_file(path,country,f) for f in onlyfiles],axis=1)
    merged_df = merged_df.drop('Country', axis = 1)
    merged_df['Country'] = country[0]
    return merged_df

def process_inventories(df):
    df = df.melt(id_vars=['Profit Center (Mat_Plant)', 'Unnamed: 1', 'Plant', 'Unnamed: 3','Material', 'Cal. Year/Month'], 
                    var_name='Date_inventory_value', value_name='inventory_value')
    df = df[['Material','Date_inventory_value', 'inventory_value']]
    df['date'] = df['Date_inventory_value'].apply(lambda x: x.split('_')[0])
    df['Date_inventory_value'] = df['Date_inventory_value'].apply(lambda x: x.split('_')[1])
    df = pd.pivot_table(df, values = 'inventory_value', index=['Material','date'], columns = 'Date_inventory_value').reset_index().rename_axis(None, axis=1)
    df['year'] = df["date"].apply(lambda x : x.split('.')[1])
    df['month'] = df["date"].apply(lambda x : x.split('.')[0])
    return df
    
def read_process_inventories(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    cols = ['Profit Center (Mat_Plant)', 'Unnamed: 1', 'Plant', 'Unnamed: 3','Cal. Year/Month']
    df = pd.DataFrame()
    lstq = []
    for i in onlyfiles:
        if len(df) == 0:
            df_t = pd.read_excel(join(path, i), sheet_name = 'Sheet1')
            df = pd.concat([df,process_inventories(df_t)])
            print(f"{i},{df_t.shape}")
            lstq.extend(list(df_t['Material'].unique()))
        else:
            
            df_t = pd.read_excel(join(path, i), sheet_name = 'Sheet1')
            lst = [x for x in list(df_t.columns) if x not in cols]
            df_t = process_inventories(df_t)
            df = pd.concat([df,df_t])
            print(f"{i},{df_t.shape}")
            lstq.extend(list(df_t['Material'].unique()))
    df['version'] = df['year'] + " - " + df['month'] 
    df['key'] = df['Material'] +"_"+ df['version']
    return df

def process_trans(df):
    df = df[['Sales Order Number','Goods Issue Date','sales_value','net_sales','Sales quantity','Product_number_cleaned','Product number']]
    df['Sales quantity'] = df['Sales quantity'].astype(float)
    df = df[df['Sales quantity']>=0]
    df = df[df['sales_value']>=0]
    df['Goods Issue Date'] = pd.to_datetime(df['Goods Issue Date'])
    df['year'] = df["Goods Issue Date"].dt.year
    df['month'] = df["Goods Issue Date"].dt.month
    df = df.groupby(['Sales Order Number','year','month','Product number']).agg({'net_sales':'sum','Sales quantity':'sum','sales_value':'sum'}).reset_index()
    

    return df

def read_process_trans(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    df1_trans = process_trans(pd.read_csv(os.path.join(path,onlyfiles[0]), encoding = 'unicode_escape'))
    df2_trans = process_trans(pd.read_csv(os.path.join(path,onlyfiles[2]), encoding = 'unicode_escape'))
    df3_trans = process_trans(pd.read_csv(os.path.join(path,onlyfiles[3]), encoding = 'unicode_escape'))
    pdList = [df1_trans, df2_trans, df3_trans]  # List of your dataframes
    transaction_data_merge = pd.concat(pdList)
    df = transaction_data_merge.copy()
    df['Product number'] = df['Product number'].astype(str)
    df["Product_ID_Len"]  =  df['Product number'].astype(str).str.replace(" " , "").str.len()
    df["Product_ID_new" ] =  df.apply(lambda x : "U" + "0"*(17-x["Product_ID_Len"]) + x['Product number'] if  x['Product_ID_Len'] <18  else "U" +  x['Product number'][1:] , axis = 1)
    df['month'] = df['month'].astype(str).apply(lambda x : x[:-2]).astype(str).map(lambda x: month_map.get(x, x))
    df['year'] = df['year'].astype(str).apply(lambda x : x[:-2])
    df['year'] = np.where(df['year'] == '202','2020',df['year'])
    df['version'] = df['year'] + " - " + df['month'] 
    df ['key'] = df['Product_ID_new'] +"_"+ df['version']
    
    return df



def prep_dates_list(df):
  # TODO: add docstring
  dates_index=[]
  for index, column_header in enumerate(df.columns):
      try:
        pd.to_datetime(column_header)
        p = index
        dates_index.append(p)
      except:
        continue
  dates = df.columns[dates_index]
  act_dates = dates.insert(len(dates),'Material')
  return dates, act_dates


# Intermittence
def intermittence(time_series):
  # TODO: add docstring
  i=time_series.transpose().reset_index(drop=True).first_valid_index()
  time_series=time_series.iloc[i:]
  interm=time_series.isna().sum()/len(time_series)

  
  return interm

# COV
def cov(time_series):
  # TODO: add docstring
  c = np.nanstd(time_series.astype(float))/np.nanmean(time_series.astype(float))
  if c == np.inf:
    c=0
  elif c == np.nan:
    c=0
  return c



# Time series flags
def seq_flag(df,dates,columns_save):
    """{'1':'Regular Series','2':'Discontinued Business','3':'Sparsely Populated Series','4':'Recently Started Business'}
    """

    df = df.reset_index(drop=True)
    df.index = list(columns_save)
    y = df[df.index.isin(columns_save)]
    y = pd.DataFrame(y)
    y = pd.DataFrame(y.transpose().reset_index(drop=True))

    if y[dates[-6:]].isnull().all(1)[0]:
        return 2
    elif y[dates[-4:]].count(axis=1).eq(4)[0] & y[dates[:-12]].count(axis=1).eq(0)[0]:
        return 4
    elif y.intermittence[0] > .3:
        return 3
    else:
        return 1


# Time series decompose
def decomposition(df,dates,columns_save):
  # TODO: add docstring
  # df = pd.Series(df)
  df = df.reset_index(drop=True)
  df.index = list(columns_save)
  
  y = df[df.index.isin(dates)]
  y = y.fillna(0)
  
  decomp = sm.tsa.seasonal_decompose(y, period=12, extrapolate_trend='freq')
  trend = pd.DataFrame(decomp.trend)
  seasonal = pd.DataFrame(decomp.seasonal)
  error = pd.DataFrame(decomp.resid)
  
  list_1 = trend.values.ravel().tolist()
  list_2 = seasonal.values.ravel().tolist()
  list_3 = error.values.ravel().tolist()
  
  return list_2