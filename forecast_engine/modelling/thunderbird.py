import re
import os
import pandas as pd
import numpy as np


import forecast_engine.modelling.utils.nodes as utils
import forecast_engine.modelling.forecast_engine.forecast_loop as forecast_loop
import forecast_engine.modelling.forecast_engine.forecast_engine_pipeline as forecast_engine_pipeline





import forecast_engine.modelling.models.ml_models.naive_models as naive_models
import forecast_engine.modelling.models.ml_models.holt_winters as holt_winters

import forecast_engine.modelling.models.ml_models.garch as garch
import forecast_engine.modelling.models.ml_models.rw_drift as rw_drift
import forecast_engine.modelling.models.ml_models.arima as arima
import forecast_engine.modelling.models.ml_models.croston as croston

import forecast_engine.modelling.models.ml_models.causal_impact as causal_impact
import forecast_engine.modelling.models.ml_models.prophet as prophet

import forecast_engine.modelling.models.ml_models.light_gbm as light_gbm
import forecast_engine.modelling.models.ml_models.xgboost as xgboost




import forecast_engine.modelling.models.ml_models.linear_regression_mv as linear_regression_mv
import forecast_engine.modelling.models.ml_models.arima_mv as arima_mv
import forecast_engine.modelling.models.ml_models.causal_impact_mv as causal_impact_mv
import forecast_engine.modelling.models.ml_models.xgboost_mv as xgboost_mv
import forecast_engine.modelling.models.ml_models.light_gbm_mv as light_gbm_mv
import forecast_engine.modelling.models.ml_models.prophet_mv as prophet_mv
import forecast_engine.modelling.models.ml_models.olsregression_mv as olsregression_mv
import forecast_engine.modelling.models.ml_models.random_forest_mv as random_forest_mv


import forecast_engine.modelling.utils.config as config
import forecast_engine.modelling.utils.logger as logger
import forecast_engine.preprocess.nodes as preprocess






class thunderbird():
    
    all = {}


    # Load congif
    config_catalog = config.load_catalog_params(os.getcwd(), 'conf/catalogs')
    config_parameters = config.load_catalog_params(os.getcwd(), 'conf/parameters')
    
    

    # Inputs
    country = config_catalog['data_preprocessing']['country']
    sales_order = config_catalog['data_preprocessing']['sales_order_file_path']
    weather = config_catalog['data_preprocessing']['drivers']['weather']
    macro = config_catalog['data_preprocessing']['drivers']['macro']
        
    # Output Paths
    processed_data = config_catalog['output_file_path']['processed_data']
    classifier_train = config_catalog['output_file_path']['classifier_train']
    univariate = config_catalog['output_file_path']['univariate']
    multivariate = config_catalog['output_file_path']['multivariate']
    ensemble = config_catalog['output_file_path']['ensemble']
    fitted_values = config_catalog['output_file_path']['fitted_values']
    model_list = config_catalog['output_file_path']['model_list']
    forecast_output = config_catalog['output_file_path']['forecast_output']
    
    # Forecast Engine Parameters
    debug_flag = config_parameters['forecast_engineSettings']['debug_flag']
    debug_models = config_parameters['forecast_engineSettings']['debug_models']
    outlier_flag = config_parameters['forecast_engineSettings']['outlier_flag']
    select_variable_flag = config_parameters['forecast_engineSettings']['select_variable_flag']
    all_at_once_flag = config_parameters['forecast_engineSettings']['all_at_once_flag']
    months_to_forecast = config_parameters['forecast_engineSettings']['months_to_forecast']
    max_validation = config_parameters['forecast_engineSettings']['max_validation']
    validation_window = config_parameters['forecast_engineSettings']['validation_window']
    single_series = config_parameters['forecast_engineSettings']['single_series']
    multivariate_flag = config_parameters['forecast_engineSettings']['multivariate_flag']
    logger.logger.info(f"loaded config and parameters for preprocess")
    logger.logger.info(f"loaded config and parameters for forecast engine")
    

    
    

    def __init__(self,name,df):

        self.name = name
        self.df = df

        thunderbird.all[self.name] = self.df 
        
        
                
    @classmethod
    def instantiate_inputs(cls):
        
        # Load input files
        sales_data, df = preprocess.read_prep_sales_order(thunderbird.sales_order)
        wthr_data = preprocess.read_prep_weather(thunderbird.weather)
        macro = preprocess.read_macro_files(thunderbird.macro, country = thunderbird.country)

        logger.logger.info(f" name : raw_data loaded, data shape: {df.shape}")
        logger.logger.info(f" name : wthr_data loaded, data shape: {wthr_data.shape}")
        logger.logger.info(f" name : macro loaded, data shape: {macro.shape}")
        thunderbird('raw_data',df), thunderbird('wthr_data',wthr_data), thunderbird('macro',macro)

    
    @classmethod
    def preprocess_inputs(cls):
        dates, act_dates = preprocess.prep_dates_list(thunderbird.all['raw_data'])
        sub_season = [str(x)+'_Season' for x in dates]
        thunderbird.all['raw_data'][sub_season] = thunderbird.all['raw_data'].apply( lambda x : preprocess.decomposition(x,dates,thunderbird.all['raw_data'].columns), axis = 1, result_type="expand")
        thunderbird.all['raw_data']['intermittence'] = thunderbird.all['raw_data'].apply(lambda x : preprocess.intermittence(x), axis =1)
        thunderbird.all['raw_data']['cov'] = thunderbird.all['raw_data'][dates].apply(lambda x : preprocess.cov(x), axis =1)
        thunderbird.all['raw_data']['seq - flag'] = thunderbird.all['raw_data'].apply(lambda x : preprocess.seq_flag(x,dates,thunderbird.all['raw_data'].columns) , axis = 1)
        thunderbird.all['raw_data'].drop(['intermittence','cov'],axis =1,inplace=True)
        thunderbird.all['raw_data']['Country'] = 'Mexico'
        data = pd.merge(thunderbird.all['raw_data'], thunderbird.all['macro'], on = 'Country', how = 'left')
        data = pd.merge(data, thunderbird.all['wthr_data'], on = 'Country', how = 'left')
        data.drop('Country',axis=1,inplace=True)
        data.rename(columns={'Material':'index'},inplace=True)
        data.columns = [str(i) for i in list(data.columns) ]
        logger.logger.info(f" name : model_input loaded, data shape: {data.shape}")
        thunderbird('data',data)
            
        

    @classmethod
    def thunder_fuel(cls):
        univariate_models = {}
        multivariate_models = {}
        if globals()['holt_winters'] : univariate_models = {func.__name__:func for func in filter(callable, globals()['holt_winters'].__dict__.values()) if func.__name__ in thunderbird.config_parameters['forecast_engineSettings']['univariate']}
        if globals()['naive_models'] : univariate_models.update({func.__name__:func for func in filter(callable, globals()['naive_models'].__dict__.values()) if func.__name__ in thunderbird.config_parameters['forecast_engineSettings']['univariate']})
        thunderbird.config_parameters['forecast_engineSettings']['univariate'] = [x for x in thunderbird.config_parameters['forecast_engineSettings']['univariate'] if x not in list(univariate_models.keys())] 

        for i in thunderbird.config_parameters['forecast_engineSettings']['univariate']:
            univariate_models.update({func.__name__:func  for func in filter(callable, globals()[i].__dict__.values())})

        for i in thunderbird.config_parameters['forecast_engineSettings']['multivariate']:
            multivariate_models.update({func.__name__:func  for func in filter(callable, globals()[i].__dict__.values())})
        logger.logger.info(f"Thunderbird fueled : loaded models")

        list_of_models = univariate_models.copy()
        if thunderbird.multivariate_flag == 1:  list_of_models.update(multivariate_models)
        thunderbird('thunder_fuel',list_of_models)



    def __repr__(self):
        return f"{self.name}, {self.df}"

    @classmethod
    def switch_on(cls,key=None):
        try:
            if key:
                df_output = thunderbird.all['data'][thunderbird.all['data']['index']==key].apply(lambda x: forecast_engine_pipeline.forecast_engine(x, thunderbird.months_to_forecast,
                                thunderbird.debug_flag,thunderbird.debug_models,thunderbird.all['thunder_fuel'], thunderbird.single_series,  thunderbird.all_at_once_flag, thunderbird.validation_window, 
                                thunderbird.max_validation, thunderbird.outlier_flag, thunderbird.select_variable_flag),axis = 1)    
                logger.logger.info(f"Thunderbird ready to fly to {thunderbird.country} : Test run completed")
            else:
                df_output = thunderbird.all['data'].apply(lambda x: forecast_engine_pipeline.forecast_engine(x, thunderbird.months_to_forecast,
                                thunderbird.debug_flag,thunderbird.debug_models,thunderbird.all['thunder_fuel'], thunderbird.single_series,  thunderbird.all_at_once_flag, thunderbird.validation_window, 
                                thunderbird.max_validation, thunderbird.outlier_flag, thunderbird.select_variable_flag),axis = 1)    
                logger.logger.info(f"Thunderbird flew to {thunderbird.country} : engine run completed")
        except:
            logger.logger.info(f"Thunderbird crashed & burned : enigne errored out")

        df_output = df_output.ravel()
        output_temp = df_output[0]['output']
        df_classifier_train_set = df_output[0]['classifier_train_set']
        impact_temp = df_output[0]['var_coeff']
        df_univariate_temp = df_output[0]['df_univariate']
        mp = df_output[0]['df_multivariate']
        df_ensemble_temp = df_output[0]['df_ensemble']
        df_fitted_values_list = df_output[0]['fitted_values_list']
        model_list = df_output[0]['dataf']
        
        thunderbird('oos',output_temp),  thunderbird('univariate',df_univariate_temp), thunderbird('multivariate',mp), thunderbird('ensemble',df_ensemble_temp), thunderbird('model_list',model_list)
        
    