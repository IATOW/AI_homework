from collections import namedtuple
import random
import numpy as np

from Utils import Utils


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


class Memory:
    def __init__(self, size):
        self.mem = []
        self.size = size
        self.position = 0

    def push(self, *args):
        if len(self.mem) < self.size:
            self.mem.append(None)
        self.mem[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size):
        return random.sample(self.mem, batch_size)


class GameDataset:
    def __init__(self):
        self.x = np.load("data/npy/rl_data_x.npy")
        self.to_play = np.load("data/npy/rl_data_to_play.npy")
        self.size = self.x.size(0)


    def sample(self):
        rnd = random.randint(0, self.size - 1)
        return self.x[rnd], self.to_play[rnd]



mem = Memory(10000)
dataset = GameDataset()
episode = 10

for i in range(episode):
    state, to_play = dataset.sample()
