import argparse
import json

def linear_model_prediction(value: int) -> None:
    try:
        with open('../configuration.json') as c_file:
            c = json.load(c_file)
        prediction = c.theta0 + value * c.theta1
        print(f'For {value} -> {prediction}')
    except:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser
    parser.add_argument(name='value')
    pass
