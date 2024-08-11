import argparse

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='none')
    parser.add_argument('--threshold', type=float, default=30)
    parser.add_argument('--car_name', type=str, default='ev_golf')
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--num_folds', type=int, default=5)

    args = parser.parse_args()

    return args
