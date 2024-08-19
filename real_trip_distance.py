import pickle
import numpy as np
from config import get_config

def distance(pred, car_name):
    PATH = './model_regression/' + car_name + '/scaler_y.pkl'
    with open(PATH, 'rb') as file:
        scaler = pickle.load(file)

        return scaler.inverse_transform(np.array([[pred]]))
    
def main():
    config = get_config()

    print(distance(config.pred, config.car_name))

if __name__ == '__main__':
    main()