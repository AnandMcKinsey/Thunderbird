import logging
import warnings
logging.getLogger('fbprophet').setLevel(logging.ERROR)
logging.getLogger('statsmodels').setLevel(logging.WARNING)
logging.getLogger('arch').setLevel(logging.WARNING)
logging.getLogger('croston').setLevel(logging.WARNING)
logging.getLogger('pandas').setLevel(logging.WARNING)
logging.getLogger('pmdarima').setLevel(logging.WARNING)
logging.getLogger('causalimpact').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import forecast_engine.modelling.utils.logger as logger
import forecast_engine.modelling.thunderbird as thunderbird




#TODO post processing block(dependent on the data pipeline requirements )




def run_pipeline():
    """
    Main pipeline for Running Thunderbird
    """
    
    
    logger.logger.info("loading data preprocessing pipeline...")
    thunderbird.thunderbird.instantiate_inputs()
    thunderbird.thunderbird.preprocess_inputs()
    logger.logger.info("data preprocessing pipeline complete...")
    
    logger.logger.info("loading model parameters...")
    thunderbird.thunderbird.thunder_fuel()
    logger.logger.info("model parameters loaded...")

    logger.logger.info("running forecast engine...")
    thunderbird.thunderbird.switch_on('U00000000006032021')
    logger.logger.info("running forecast engine...")
    


    logger.logger.info("running post processing pipeline...")
    
    logger.logger.info("post processing pipeline complete...")
    logger.logger.info("Saving model outputs...")
    logger.logger.info("Thunder bird forecast engine run complete.")
    
 
       
if __name__ == "__main__":
    logger.logger.info("Running main pipeline...")
    run_pipeline()