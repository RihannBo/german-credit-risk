import os
import sys
from exception import CustomException
from logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from components.data_transformation import DataTransformation



@dataclass 
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # 1. Read data
            df = pd.read_csv('data/german_credit_data.csv')
            logging.info('Read the dataset as dataframe')

            # 2. Column renaming
            df.columns = (
                df.columns
                .str.lower()
                .str.strip()
                .str.replace(' ','_')
            )
            logging.info("Column name standardized")

            # 3. Remove unwanted index column if exists
            df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

            # 4. Duplicate check
            num_dupliactes = df.duplicated().sum()
            if num_dupliactes > 0:
                logging.info(f"Found {num_dupliactes} duplicate rows")
                df = df.drop_duplicates()
                logging.info(f"Dropped {num_dupliactes} duplicate rows")
            else:
                logging.info("No duplicate rows found")

            # 5. Create artifacts folder
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # 6. Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
 
            logging.info("Train test split initiated")
            # 7. Train test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


# if __name__ == "__main__":
#     obj = DataIngestion()
#     train_path, test_path, _ = obj.initiate_data_ingestion()

#     data_transformation = DataTransformation()
#     data_transformation.initiate_data_transformation(
#         train_path=train_path,
#         test_path=test_path
#     )