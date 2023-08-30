#!/usr/bin/env python3

import argparse
import json
from sklearn.linear_model import LinearRegression
from pandas import read_csv
import sys
import matplotlib.pyplot as plt
import logging

def argument_parser():
    parser = argparse.ArgumentParser(
        prog='predict.py',
        description='Mileage predictor'
    )
    parser.add_argument('VALUE', type=int)
    parser.add_argument('-d', '--debug',
                        action='store_true')
    return parser.parse_args()


class Predictor():
    def __init__(self, value, debug=False):
        self.value = value
        self.debug = debug
        if self.debug:
            with open('../trainer/data.csv', 'r') as csv:
                self.training_set = read_csv(csv)
            plt.scatter(x=self.training_set['km'], 
                        y=self.training_set['price'], 
                        color='green')

    def __debug(self, pred):
        lr = LinearRegression()
        lr.fit(X=self.training_set['km'].values.reshape(-1, 1),
               y=self.training_set['price'])
        sk_pred = lr.predict([[self.value]])[0]
        plt.scatter(x=[self.value, self.value],
                    y=[sk_pred, pred],
                    c=['red', 'yellow'])
        plt.xlabel('km') 
        plt.ylabel('price')
        plt.title=('Prediction')
        print(f'[ OURS       ] {pred}')
        print(f'[ THEIRS     ] {sk_pred}')
        plt.show()

    def predict(self):
        with open('../configuration.json') as c_file:
            c = json.load(c_file)
        prediction = c['theta0'] + self.value * c['theta1']
        print(f'[ PREDICTION ] For value of {self.value} -> {prediction}')
        if self.debug:
            self.__debug(pred=prediction)


if __name__ == "__main__":
    args = argument_parser()
    try:
        p = Predictor(value=args.VALUE,
                      debug=args.debug)
        p.predict()
    except Exception as e:
        logging.error(str(e))
        sys.exit(1)
    sys.exit(0)
