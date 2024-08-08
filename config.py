import argparse

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='none')
    parser.add_argument('--threshold', type=float, default=30)

    args = parser.parse_args()

    return args
