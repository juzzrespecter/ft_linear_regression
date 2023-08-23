#!/usr/bin/env python3

import json
import csv
import sys
import argparse
import logging

# main loop: if -v is activated print into terminal
#   if -i is activated open graph and actualize (plot data and linear model)    
# save theta0 and theta1 into configuration file

            
class TrainingModel(dict):
    def __set_value(self, name, value):
        self.__dict__[name] = value

    def __get_value(self, name):
        return self.__dict__[name]
    
    def __init__(self, dataset='../data.csv'):
        with open('../configuration.json') as c_file:
            c = json.loads(c_file.read())
        with open(dataset) as ds_file:
            ds = csv.DictReader(ds_file)
            self.__set_value('dataset', csv.DictReader(ds))
        self.__set_value('learning_rate', c['hyperparameters']['learning_rate'])
        self.__set_value('n_iterations', c['hyperparameters']['n_iterations'])
        self.__set_value('tolerance', c['hyperparameters']['tolerance'])
        self.__set_value('theta0', c['theta0'])
        self.__set_value('theta1', c['theta1'])

    def __theta0_sum(self, theta0, theta1):
        ds = self.__get_value('dataset')
        n = len(list(ds))
        sum = 0
        for i in range(n):
            sum = sum + theta0 + theta1 * ds[i]['mileage'] - ds[i]['price']
        return sum * (1 / n)
            
    def __theta1_sum(self, theta0, theta1):
        ds = self.__get_value('dataset')
        n = len(list(ds))
        for i in range(n):
            sum = sum + (theta0 + theta1 * ds[i]['mileage'] - ds[i]['price']) * ds[i]['mileage']
        return sum * (1 / n)

    def train(self):
        lr = self.__get_value('learning_rate')
        n = self.__get_value('n_iterations')
        e = self.__get_value('tolerance')
        theta0 = self.__get_value('theta0')
        theta1 = self.__get_value('theta1')
        for _ in range(n):
            tmpTheta0 = lr * self.__theta0_sum(theta0, theta1)
            tmpTheta1 = lr * self.__theta1_sum(theta0, theta1)
            if abs(theta1 - tmpTheta1) < e:
                break
            theta0 = tmpTheta0
            theta1 = tmpTheta1
            print(f'y = {theta0} + {theta1} * x')
            
        with open('../configuration.json') as c_file:
            c = json.loads(c_file.read())
            c.theta0 = theta0
            c.theta1 = theta1
            json.dumps(c, c_file)
        

def arg_handler() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='Training Model',
        description='Linear Regression model trainer'
    )
    parser.add_argument('DATASET', 
                        default='../data.csv',
                        nargs='?')
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    parser.add_argument('-i', '--visual',
                        action='store_true')
    return parser.parse_args()        
            
if __name__ == "__main__":
    args = arg_handler()
    try:
        t = TrainingModel(dataset=args.DATASET)
        t.train()
    except Exception as e:
        logging.error(str(e))
        sys.exit(1)
    sys.exit(0)