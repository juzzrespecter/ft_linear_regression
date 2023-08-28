#!/usr/bin/env python3

import argparse
import json
from sklearn.linear_model import LinearRegression
from pandas import read_csv
import sys
#import matplotlib.pyplot as plt
import logging

def argument_parser():
    parser = argparse.ArgumentParser(
        prog='predict.py',
        description='mileage predictor'
    )
    parser.add_argument(name='VALUE')
    parser.add_argument('-d', '--debug',
                        action='store_true')
    return parser.parse_args()


class Predictor():
    def __init__(self, value, debug=False):
        self.value = value
        self.debug = debug
        if self.debug:
            with open('trainer/data.csv', 'r') as csv:
                self.training_data = read_csv(csv)
            plt.scatter(x=self.training_data['km'], 
                        y=self.training_data['price'], 
                        color='green')

    def __debug(self, pred):
        lr = LinearRegression()
        lr.fit(X=self.training_set[['km']],
               y=self.training_set['price'])
        sk_pred = lr.predict(self.value)
        plt.plot(self.value, sk_pred, color='blue')
        plt.plot(self.value, pred, oolor='yellow')

    def predict(self, value):
        with open('../configuration.json') as c_file:
            c = json.load(c_file)
        prediction = c.theta0 + value * c.theta1
        print(f'For {value} -> {prediction}')
        if self.debug:
            self.__debug()


if __name__ == "__main__":
    args = argument_parser()
    try:
        p = Predictor(debug=args.debug)
        p.predict(args.VALUE,
                  args.debug)
    except Exception as e:
        print('oops')
        sys.exit(1)
    sys.exit(0)
