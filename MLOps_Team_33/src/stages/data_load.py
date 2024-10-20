from scipy.io import arff
from typing import Text
import pandas as pd
import argparse
import yaml

def data_load(config_path: Text) -> pd.DataFrame:
    with open('params.yaml') as config_file:
        config = yaml.safe_load(config_file)
    
        data, meta = arff.loadarff(config['data']['dataset_arff'])

    return pd.DataFrame(data)

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest = 'config', required=True)

    args = args_parser.parse_args()

    data_load(config_path=args.config)