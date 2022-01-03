import math
import csv
import seaborn as sns
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jsonlines
import os
import argparse
from glob import glob
import pickle5 as pickle

from scripts.eval import parse_jsonlines_log


def get_args():
    """
    A function to process script args.

    Returns
    -------
    argparse.Namespace
        Collection of command line arguments and their values
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('log', type=str, help="The full path to the trajectory log file")
    
    return parser.parse_args()

class animator_docking_oriented_2d:

    def __init__(self, log_data):
        self.log_data = log_data

        self.fig, self.ax = plt.subplots()

        deputy_xs = [d['info']['deputy']['x'] for d in self.log_data]
        deputy_ys = [d['info']['deputy']['y'] for d in self.log_data]

        xlim = (max(deputy_xs), min(deputy_xs))

        self.ax.set_xlim()

    def frame_init(self):
        pass

    def frame_change(self, frame):
        return self._frame_change(self.log_data[frame])

    def _frame_change(self, data):
        ...


def main():
    # process args
    args = get_args()

    log_data = parse_jsonlines_log(args.log)



if __name__ == "__main__":
    main()
