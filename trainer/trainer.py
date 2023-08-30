#!/usr/bin/env python3

import json
from pandas import read_csv
import sys
import argparse
import logging
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from math import isnan


            
class TrainingModel(dict):
    def __set_value(self, name, value):
        self.__dict__[name] = value

    def __get_value(self, name):
        return self.__dict__[name]

    def __normalize(self, a):
        min, max = a.min(), a.max()
        return a.map(lambda n: (n - min) / (max - min))
    
    def __denormalize_theta(self, theta):
        theta0, theta1 = theta
        tr_set = self.__get_value('training_set')
        xmax, xmin = tr_set['km'].max(), tr_set['km'].min()
        ymax, ymin = tr_set['price'].max(), tr_set['price'].min()
        theta1_d = theta1 * (ymax - ymin) / (xmax - xmin)
        theta0_d = theta0 * (ymax - ymin) + ymin - (theta1_d * xmin)
        return theta0_d, theta1_d
    
    def __init__(self, dataset='../data.csv', visual=False, debug=False):
        self.__set_value('visual', visual)
        self.__set_value('debug', debug)        
        with open('../configuration.json') as c_file:
            c = json.loads(c_file.read())
        with open(dataset) as ds_file:
            tr_set = read_csv(ds_file)
            n_tr_set = (tr_set - tr_set.min()) / (tr_set.max() - tr_set.min())
            self.__set_value('x', self.__normalize(n_tr_set['km']))
            self.__set_value('y', self.__normalize(n_tr_set['price']))
            self.__set_value('training_set', tr_set)
            self.__set_value('normalized_training_set', n_tr_set)
        self.__set_value('learning_rate', c['hyperparameters']['learning_rate'])
        self.__set_value('n_iterations', c['hyperparameters']['n_iterations'])
        self.__set_value('tolerance', c['hyperparameters']['tolerance'])
        self.__set_value('theta0', c['n_theta0'])
        self.__set_value('theta1', c['n_theta1'])

    def __cost_func(self, theta):
        x, y = self.__get_value('x'), self.__get_value('y')
        n = len(x)
        err_sum = 0
        for i in range(n):
            err_sum += (self.__y_predict(theta, x[i]) - y[i]) ** 2
        return err_sum / (2 * n)

    def __y_predict(self, theta, x):
        theta0, theta1 = theta
        return theta0 + theta1 * x

    def __plot_values(self, h_cost, theta):
        theta0, theta1 = theta
        _, (lr, cf, eh) = plt.subplots(nrows=1, ncols=3)
        self.training_set = self.training_set.sort_values(by='km')
        x, y = self.training_set['km'], self.training_set['price']
        y_pred = [self.__y_predict(theta, xi) for xi in x]
        error = [y_predi - yi for y_predi, yi in zip(y_pred, y)]

        lr.scatter(self.training_set['km'], self.training_set['price'])
        lr.set(xlabel='mileage', ylabel='price', title='Linear Regression')
        lr.axline(xy1=(0,theta0), slope=theta1, color='red')
        cost_n = list(range(len(h_cost)))
        err_n = list(range(len(error)))
        cf.plot(cost_n, h_cost)
        cf.set(xlabel='n iterations', ylabel='error', title='Cost Evolution')
        eh.bar(err_n, error, width=0.5, color='red')
        eh.set_ylabel(ylabel='error')
        eh.set_title('Error Histogram')
        plt.show()

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
    
    def __calc_r_squared(self, theta):
        x, y = self.training_set['km'], self.training_set['price']
        y_pred = [self.__y_predict(theta=theta, x=xi) for xi in x]
        y_mean = sum(y) / len(y)
        ss_res = sum([(yi - y_predi) ** 2 for yi, y_predi in zip(y, y_pred)])
        ss_tot = sum([(yi - y_mean) ** 2 for yi in y])
        return 1 - (ss_res / ss_tot)

    def train(self):
        lr = self.__get_value('learning_rate')
        n = self.__get_value('n_iterations')
        e = self.__get_value('tolerance')
        n_theta0 = self.__get_value('theta0')
        n_theta1 = self.__get_value('theta1')
        h_cost = []
        for _ in range(n):
            tmpTheta0 = lr * self.__theta0_sum(n_theta0, n_theta1)
            tmpTheta1 = lr * self.__theta1_sum(n_theta0, n_theta1)
            n_theta0 = n_theta0 - tmpTheta0   
            n_theta1 = n_theta1 - tmpTheta1
            cost = self.__cost_func([n_theta0, n_theta1])
            h_cost.append(cost)
            if cost < e:
                break
        theta0, theta1 = self.__denormalize_theta([n_theta0, n_theta1])
        print('[ TRAINER ] Finished model training, with (theta0, theta1): (%.3f, %.3f)' % (theta0, theta1))
        print('[         ] Regression precision: %.4f' % self.__calc_r_squared([theta0, theta1]))
        if self.visual:
            self.__plot_values(h_cost=h_cost, theta=[theta0, theta1])        
        with open('../configuration.json', 'r') as c_file:
            c = json.loads(c_file.read())
            c['n_theta0'] = n_theta0
            c['n_theta1'] = n_theta1
            c['theta0'] = theta0
            c['theta1'] = theta1
        with open('../configuration.json', 'w') as c_file:
            json.dump(c, c_file, indent=2)


def arg_handler():
    parser = argparse.ArgumentParser(
        prog='trainer.py',
        description='Linear Regression model trainer'
    )
    parser.add_argument('DATASET', 
                        default='data.csv',
                        nargs='?')
    parser.add_argument('-v', '--visual',
                        action='store_true')
    parser.add_argument('-d', '--debug',
                        action='store_true')
    return parser.parse_args()        
            
if __name__ == "__main__":
    args = arg_handler()
    try:
        t = TrainingModel(dataset=args.DATASET,
                          visual=args.visual,
                          debug=args.debug)
        t.train()
    except Exception as e:
        logging.error(str(e))
        sys.exit(1)
    sys.exit(0)
