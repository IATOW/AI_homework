import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader


adjacent_point = [
    [0, -1],
    [-1, 0],
    [-1, 1],
    [0, 1],
    [1, 0],
    [1, -1]
]


class Utils:
    @staticmethod
    def print_plane(data):
        x, y = data.shape
        for idx, i in enumerate(range(x)):
            for j in range(idx):
                print(" ", end="")
            for j in range(y):
                print(data[i][j], "", end="")
            print("")

    @staticmethod
    def load_npy_to_tensor(prefix):
        x = np.load("data/npy/" + prefix + "_x.npy")
        y = np.load("data/npy/" + prefix + "_y.npy")

        x = torch.from_numpy(x).type(torch.float)
        y = torch.from_numpy(y).type(torch.long)

        return x, y

    # @staticmethod
    # def print_result(data):
    #     for i in range(13):
    #         for j in range(i):
    #             print(" ", end="")
    #         for j in range(13):
    #             print("%.1f " % data[i][j], end="")
    #         print("")

    @staticmethod
    def game_over(board):
        size = -1
        if board is np.ndarray:
            size = board.shape[0]
        elif board is torch.Tensor:
            size = board.size(0)

        vis = np.zeros((size, size), dtype=np.int)
        st = []

        for i in range(size):
            if board[0][i] == 1:
                st.append((0, i))
                vis[0][i] = 1

        while len(st) != 0:
            x, y = st.pop()
            for i in range(6):
                nx = x + adjacent_point[i][0]
                ny = y + adjacent_point[i][1]
                if 0 <= nx < size and 0 <= ny < size and board[nx][ny] == 1 and vis[nx][ny] == 0:
                    if nx == size - 1:
                        return 1
                    vis[nx][ny] = 1
                    st.append((nx, ny))
