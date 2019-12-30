import numpy as np

import torch
import torch.nn.functional as F

from Net import Net

from data_loader import DataLoader

from Utils import Utils


class Board:
    def __init__(self):
        self.data = np.zeros((13, 13), dtype=np.int)

    def set(self, x, y, val):
        self.data[x][y] = val

    def print(self):
        print("            6")
        for i in range(13):
            for j in range(i):
                if i == 6 and j == 4:
                    print("6 ", end="")
                    break
                print(" ", end="")
            for j in range(13):
                if self.data[i][j] == 0:
                    print(". ", end="")
                elif self.data[i][j] == 1:
                    print("* ", end="")
                elif self.data[i][j] == 2:
                    print("o ", end="")
            print("")

    def get_data(self):
        return self.data


def run():
    board = Board()

    # move = [int(m) for m in input("black move: ").split(" ")]
    # board.set(move[0], move[1], 1)
    # board.print()
    #
    # move = [int(m) for m in input("white move: ").split(" ")]
    # board.set(move[0], move[1], 2)
    # board.print()

    model_black = Net(8, 128, 9)
    model_black.load_state_dict(torch.load("model/model_8_128_black.pt"))
    model_white = Net(8, 128, 9)
    model_white.load_state_dict(torch.load("model/model_8_128_white.pt"))

    while True:
        pad_board = DataLoader.pad_board(board.get_data(), 13, 2)
        features = DataLoader.retrieve_all_features(pad_board, 17)
        y = model_black(torch.from_numpy(features).float().view(1, 9, 17, 17))
        y = F.softmax(y, dim=1).reshape((13, 13)) * 100
        Utils.print_result(y)

        move = [int(m) for m in input("black move: ").split(" ")]
        board.set(move[0], move[1], 1)
        board.print()

        pad_board = DataLoader.pad_board(board.get_data(), 13, 2)
        features = DataLoader.retrieve_all_features(pad_board, 17)
        y = model_white(torch.from_numpy(features).float().view(1, 9, 17, 17))
        y = F.softmax(y, dim=1).reshape((13, 13)) * 100
        Utils.print_result(y)

        move = [int(m) for m in input("white move: ").split(" ")]
        board.set(move[0], move[1], 2)
        board.print()
        # break

if __name__ == "__main__":
    run()