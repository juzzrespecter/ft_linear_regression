#!/usr/bin/env python3

import json
from pandas import read_csv
import sys
import argparse
import logging
import matplotlib.pyplot as plt

            
class TrainingModel(dict):
    def __set_value(self, name, value):
        self.__dict__[name] = value

    def __get_value(self, name):
        return self.__dict__[name]

    def __normalize(self, a):
        min, max = a.min(), a.max()
        return a.map(lambda n: (n - min) / (max - min))
    
    def __get_std_coeff(self, x, y):
        return (y.max() - y.min()) / (x.max() - x.min())
    
    def __init__(self, dataset='../data.csv', verbose=False, visual=False):
        with open('../configuration.json') as c_file:
            c = json.loads(c_file.read())
        plt.plot(xlabel='mileage', ylabel='price', title='ft_linear_regression')
        with open(dataset) as ds_file:
            tr_set = read_csv(ds_file)
            n_tr_set = (tr_set - tr_set.min()) / (tr_set.max() - tr_set.min())
#            plt.scatter(x=tr_set['km'], y=tr_set['price'])
            self.__set_value('x', self.__normalize(n_tr_set['km']))
            self.__set_value('y', self.__normalize(n_tr_set['price']))

            self.__set_value('training_set', tr_set)
            self.__set_value('normalized_training_set', n_tr_set)
            plt.scatter(x=self.__get_value('x'), y=self.__get_value('y'))            
        self.__set_value('learning_rate', c['hyperparameters']['learning_rate'])
        self.__set_value('n_iterations', c['hyperparameters']['n_iterations'])
        self.__set_value('tolerance', c['hyperparameters']['tolerance'])
        self.__set_value('theta0', c['theta0'])
        self.__set_value('theta1', c['theta1'])
        self.__set_value('verbose', verbose)
        self.__set_value('visual', visual)

    def __cost_func(self, theta, m):
        x, y = self.__get_value('x'), self.__get_value('y')
        err_sum = 0
        for i in range(m):
            err_sum += (self.__y_predict(theta, x[i]) - y[i]) ** 2
        return err_sum / (2 * m)

    def __y_predict(self, theta, x):
        theta0, theta1 = theta
        return theta0 + theta1 * x

    def __theta0_sum(self, theta0, theta1):
        x = self.__get_value('x')
        y = self.__get_value('y')
        n = len(x)
        sum = 0
        for i in range(n):
            sum += self.__y_predict([theta0, theta1], x[i]) - y[i]
        return sum * (1 / n)
            
    def __theta1_sum(self, theta0, theta1):
        x = self.__get_value('x')
        y = self.__get_value('y')
        n = len(x)
        sum = 0
        for i in range(n):
            sum += (self.__y_predict([theta0, theta1], x[i]) - y[i]) * x[i]      
        return sum * (1 / n)
    
    def __debug(self):
        pass

    def train(self):
        lr = self.__get_value('learning_rate')
        n = self.__get_value('n_iterations')
        e = self.__get_value('tolerance')
        theta0 = self.__get_value('theta0')
        theta1 = self.__get_value('theta1')        

        # set up figure
#        _, ax = plt.subplots()
#        ax.scatter(x=self.__get_value('x'), y=self.__get_value('y'))
#        ax.set(xlabel='mileage', ylabel='price', title='ft_linear_regression')

        for i in range(n):
            tmpTheta0 = lr * self.__theta0_sum(theta0, theta1)
            tmpTheta1 = lr * self.__theta1_sum(theta0, theta1)
            theta0 = theta0 - tmpTheta0   
            theta1 = theta1 - tmpTheta1
            if self.
            print(f'[ ITERATION {i} ] y = {theta0} + {theta1} * x')

#        line = ax.axline(xy1=(0, theta0), slope=theta1)
        #print(f'before conv: {theta0}, {theta1}')
        #theta1 = theta1 * self.__get_value('std_coeff')
        #theta0 = self.__get_value('ymin') - theta1 * self.__get_value('xmin')
        #print(f'after conv: {theta0}, {theta1}')        
        plt.axline(xy1=(0, theta0), slope=theta1)
        with open('../configuration.json', 'r+') as c_file:
            c = json.loads(c_file.read())
            print(f'{theta0}, {theta1}')
            c['theta0'] = theta0
            c['theta1'] = theta1
            c_file.seek(0)
            json.dump(c, c_file, indent=2)        
        plt.show()


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
#    try:
    t = TrainingModel(dataset=args.DATASET,
                      verbose=args.verbose,
                      visual=args.visual)
    t.train()
#    except Exception as e:
#        logging.error(str(e))
#        sys.exit(1)
    sys.exit(0)


# TODO
# sklearn for testing