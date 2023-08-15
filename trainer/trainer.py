import json
import csv
import sys
import argparse
import logging

# read dataset, save in a struct
# main loop: if -v is activated print into terminal
#   if -i is activated open graph and actualize (plot data and linear model)    
# save theta0 and theta1 into configuration file

def arg_handler() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='Training Model',
        description='Linear Regression model trainer'
    )
    parser.add_argument('DATASET', 
                        default='./csv_file',
                        nargs='?')
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    parser.add_argument('-i', '--visual',
                        action='store_true')
    return parser.parse_args()

def training_model():
    with open('../configuration.json') as c_file:
        c = json.loads(c_file)
        with open('csv_file') as d_file:
            dataset = csv.
            pass
        pass
    pass
            
            
if __name__ == "__main__":
    args = arg_handler()
    try:
        training_model(args)
    except Exception as e:
        logging.error(str(e))
        sys.exit(1)
    sys.exit(0)