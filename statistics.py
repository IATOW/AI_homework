import numpy as np
import matplotlib.pyplot as plt


def run():
    y = np.load("data/statistics/correct_rate_2_16.npy")
    plt.plot(y, label="d = 2, w = 16")
    y = np.load("data/statistics/correct_rate_3_16.npy")
    plt.plot(y, label="d = 3, w = 16")
    y = np.load("data/statistics/correct_rate_4_16.npy")
    plt.plot(y, label="d = 4, w = 16")
    y = np.load("data/statistics/correct_rate_5_16.npy")
    plt.plot(y, label="d = 5, w = 16")
    y = np.load("data/statistics/correct_rate_6_16.npy")
    plt.plot(y, label="d = 6, w = 16")
    y = np.load("data/statistics/correct_rate_7_16.npy")
    plt.plot(y, label="d = 7, w = 16")
    plt.xlabel("batch count")
    plt.ylabel("correct rate on test set")
    plt.legend()
    plt.show()


def run_ec():
    y = np.load("data/statistics/ec/ec_error__pop20_gen480_sig1.npy")
    plt.plot(-y, label="sigma = 0.1, pop = 20")
    plt.xlabel("generation")
    plt.ylabel("correct rate")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_ec()