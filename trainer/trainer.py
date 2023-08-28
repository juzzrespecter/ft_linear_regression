#!/usr/bin/env python3

import json
from pandas import read_csv
import sys
import argparse
import logging
#import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots


            
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
            if self.visual:
                #_, (pred, cost, hist) = plt.subplots(nrows=1, ncols=3)
                # fig = make_subplots(rows=1, cols=3)
                # pred.scatter(x=tr_set['km'], y=tr_set['price'], color='red')
                # pred.set(xlabel='mileage', ylabel='price', title='Linear Regression')
                # cost.set(xlabel='n iterations', ylabel='cost', title='Cost funciton')
                # hist.set(xlabel='mileage', ylabel='error')
                # self.__set_value('pred_ax', pred)
                # self.__set_value('cost_ax', cost)
                # self.__set_value('hist_ax', hist)
            self.__set_value('x', self.__normalize(n_tr_set['km']))
            self.__set_value('y', self.__normalize(n_tr_set['price']))
            self.__set_value('training_set', tr_set)
            self.__set_value('normalized_training_set', n_tr_set)
        self.__set_value('learning_rate', c['hyperparameters']['learning_rate'])
        self.__set_value('n_iterations', c['hyperparameters']['n_iterations'])
        self.__set_value('tolerance', c['hyperparameters']['tolerance'])
        self.__set_value('theta0', c['theta0'])
        self.__set_value('theta1', c['theta1'])

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

    def __plot_hist(self, hist_cost, theta):
        x, y = self.training_set['km'], self.training_set['price']
        err = [self.__y_predict(xi, theta) - yi for (xi, yi) in (x, y)]
        self.__get_value('hist_ax').bar(x=len(err), height=err)

    def __plot_cost(self, hist_cost):
        i_cost = enumerate(hist_cost)
        i = [i[0] for i in i_cost]
        cost = [cost[1] for cost in i_cost]
        self.__get_value('cost_ax').plot(i, cost, color='cyan')

    def __plot_values(self, hist_cost, theta):
        x, y = self.training_set['km'], self.training_set['price']
        lr = px.scatter(self.training_set, x='mileage', y='price')
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=("Linear Regresssion", "Cost Function", "Error Histogram"))

        pass

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

    def train(self):
        lr = self.__get_value('learning_rate')
        n = self.__get_value('n_iterations')
        e = self.__get_value('tolerance')
        theta0 = self.__get_value('theta0')
        theta1 = self.__get_value('theta1')
        hist_cost = []
        for _ in range(n):
            tmpTheta0 = lr * self.__theta0_sum(theta0, theta1)
            tmpTheta1 = lr * self.__theta1_sum(theta0, theta1)
            theta0 = theta0 - tmpTheta0   
            theta1 = theta1 - tmpTheta1
            cost = self.__cost_func([theta0, theta1])
            hist_cost.append(cost)
            if cost < e:
                break
        theta0, theta1 = self.__denormalize_theta([theta0, theta1])
        if self.visual:
            self.__get_value('pred_ax').axline(xy1=(0, theta0), slope=theta1)
            self.__plot_cost(hist_cost=hist_cost)
            self.__plot_hist(theta=theta)
            #self.__get_value('hist_ax').hist([])
        with open('../configuration.json', 'r+') as c_file:
            c = json.loads(c_file.read())
            c['theta0'] = theta0
            c['theta1'] = theta1
            c_file.seek(0)
            json.dump(c, c_file, indent=2)
        if self.visual:   
            plt.show()


def arg_handler() -> argparse.Namespace:
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
#    try:
    t = TrainingModel(dataset=args.DATASET,
                      visual=args.visual,
                      debug=args.debug)
    t.train()
#    except Exception as e:
#        logging.error(str(e))
#        sys.exit(1)
    sys.exit(0)
