import numpy as np
from Utils import Utils
import random
import os

bridge_end_point = [
    [-1, -1],
    [-2, 1],
    [-1, 2],
    [1, 1],
    [2, -1],
    [1, -2]
]
adjacent_point = [
    [0, -1],
    [-1, 0],
    [-1, 1],
    [0, 1],
    [1, 0],
    [1, -1]
]


def pad_board(board, pad):
    in_size = board.shape[0]
    out_size = in_size + pad * 2
    ans = np.zeros((out_size, out_size), dtype=np.int)
    ans[:pad] = 1
    ans[out_size - pad:] = 1
    ans[pad:out_size - pad, :pad] = 2
    ans[pad:out_size - pad, out_size - pad:] = 2
    ans[pad:out_size - pad, pad:out_size - pad] = board
    return ans


def retrieve_black_point(board):
    size = board.shape[0]
    ans = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if board[i][j] == 1:
                ans[i][j] = 1
    return ans


def retrieve_white_point(board):
    size = board.shape[0]
    ans = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if board[i][j] == 2:
                ans[i][j] = 1
    return ans


def retrieve_empty_point(board):
    size = board.shape[0]
    ans = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if board[i][j] == 0:
                ans[i][j] = 1
    return ans


def retrieve_black_bridge(board):
    size = board.shape[0]
    ans = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if board[i][j] != 1:
                continue
            for k in range(6):
                nx = i + bridge_end_point[k][0]
                ny = j + bridge_end_point[k][1]
                a1 = (i + adjacent_point[k][0], j + adjacent_point[k][1])
                a2 = (i + adjacent_point[(k + 1) % 6][0], j + adjacent_point[(k + 1) % 6][1])
                if 0 <= nx < size and 0 <= ny < size:
                    if board[nx][ny] != 1:
                        continue
                    if board[a1[0]][a1[1]] == 0 and board[a2[0]][a2[1]] == 0:
                        ans[i][j] = 1
                        ans[nx][ny] = 1
    return ans


def retrieve_white_bridge(board):
    size = board.shape[0]
    ans = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if board[i][j] != 2:
                continue
            for k in range(6):
                nx = i + bridge_end_point[k][0]
                ny = j + bridge_end_point[k][1]
                a1 = (i + adjacent_point[k][0], j + adjacent_point[k][1])
                a2 = (i + adjacent_point[(k + 1) % 6][0], j + adjacent_point[(k + 1) % 6][1])
                if 0 <= nx < size and 0 <= ny < size:
                    if board[nx][ny] != 2:
                        continue
                    if board[a1[0]][a1[1]] == 0 and board[a2[0]][a2[1]] == 0:
                        ans[i][j] = 1
                        ans[nx][ny] = 1
    return ans


def retrieve_black_save_bridge(board):
    size = board.shape[0]
    ans = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if board[i][j] != 1:
                continue
            for k in range(6):
                nx = i + bridge_end_point[k][0]
                ny = j + bridge_end_point[k][1]
                if not (0 <= nx < size and 0 <= ny < size):
                    continue
                if board[nx][ny] != 1:
                    continue
                a1 = (i + adjacent_point[k][0], j + adjacent_point[k][1])
                a2 = (i + adjacent_point[(k + 1) % 6][0], j + adjacent_point[(k + 1) % 6][1])
                if board[a1[0]][a1[1]] == 0 and board[a2[0]][a2[1]] == 2:
                    ans[a1[0]][a1[1]] = 1
                if board[a1[0]][a1[1]] == 2 and board[a2[0]][a2[1]] == 0:
                    ans[a2[0]][a2[1]] = 1
    return ans


def retrieve_white_save_bridge(board):
    size = board.shape[0]
    ans = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if board[i][j] != 2:
                continue
            for k in range(6):
                nx = i + bridge_end_point[k][0]
                ny = j + bridge_end_point[k][1]
                if not (0 <= nx < size and 0 <= ny < size):
                    continue
                if board[nx][ny] != 2:
                    continue
                a1 = (i + adjacent_point[k][0], j + adjacent_point[k][1])
                a2 = (i + adjacent_point[(k + 1) % 6][0], j + adjacent_point[(k + 1) % 6][1])
                if board[a1[0]][a1[1]] == 0 and board[a2[0]][a2[1]] == 1:
                    ans[a1[0]][a1[1]] = 1
                if board[a1[0]][a1[1]] == 1 and board[a2[0]][a2[1]] == 0:
                    ans[a2[0]][a2[1]] = 1
    return ans


def retrieve_black_form_bridge(board):
    size = board.shape[0]
    ans = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if board[i][j] != 1:
                continue
            for k in range(6):
                nx = i + bridge_end_point[k][0]
                ny = j + bridge_end_point[k][1]
                if not (0 <= nx < size and 0 <= ny < size):
                    continue
                if board[nx][ny] != 0:
                    continue
                a1 = (i + adjacent_point[k][0], j + adjacent_point[k][1])
                a2 = (i + adjacent_point[(k + 1) % 6][0], j + adjacent_point[(k + 1) % 6][1])
                if board[a1[0]][a1[1]] == 0 and board[a2[0]][a2[1]] == 0:
                    ans[nx][ny] = 1
    return ans


def retrieve_white_form_bridge(board):
    size = board.shape[0]
    ans = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if board[i][j] != 2:
                continue
            for k in range(6):
                nx = i + bridge_end_point[k][0]
                ny = j + bridge_end_point[k][1]
                if not (0 <= nx < size and 0 <= ny < size):
                    continue
                if board[nx][ny] != 0:
                    continue
                a1 = (i + adjacent_point[k][0], j + adjacent_point[k][1])
                a2 = (i + adjacent_point[(k + 1) % 6][0], j + adjacent_point[(k + 1) % 6][1])
                if board[a1[0]][a1[1]] == 0 and board[a2[0]][a2[1]] == 0:
                    ans[nx][ny] = 1
    return ans


def retrieve_all_features(board):
    size = board.shape[0]
    ans = np.zeros((9, size, size))

    ans[0] = retrieve_black_point(board)
    ans[1] = retrieve_white_point(board)
    ans[2] = retrieve_empty_point(board)
    ans[3] = retrieve_black_bridge(board)
    ans[4] = retrieve_white_bridge(board)
    ans[5] = retrieve_black_save_bridge(board)
    ans[6] = retrieve_white_save_bridge(board)
    ans[7] = retrieve_black_form_bridge(board)
    ans[8] = retrieve_white_form_bridge(board)

    return ans


def retrieve_features_from_original_board(board):
    padded_board = pad_board(board, 2)
    return retrieve_all_features(padded_board)


def load_raw_to_npy():
    x = []
    y = []
    file_count = 0
    while True:
        if file_count % 10 == 0:
            print(file_count)
        filename = "data/raw/" + str(file_count) + ".txt"
        file_count += 1
        if not os.path.exists(filename):
            break
        f = open(filename)
        line = f.readline()
        f.close()
        if line == "":
            continue
        actions = [int(s) for s in line.strip(" ").split(" ")]

        board = np.zeros((11, 11))
        to_play = 1
        for i in actions:
            padded_board = pad_board(board, 2)
            x.append(retrieve_all_features(padded_board))
            y.append(i)
            board[i // 11][i % 11] = to_play
            to_play = 3 - to_play

    x = np.array(x)
    y = np.array(y)
    print(x.shape, y.shape)
    train_ratio = 0.9
    bound = int(x.shape[0] * train_ratio)
    print("train size:", bound)
    np.save("data/npy/train_x.npy", x[:bound])
    np.save("data/npy/train_y.npy", y[:bound])
    np.save("data/npy/test_x.npy", x[bound:])
    np.save("data/npy/test_y.npy", y[bound:])


def load_raw_to_rl_data():
    file_count = 0
    x = []
    y = []
    while True:
        if file_count % 10 == 0:
            print(file_count)
        filename = "data/raw/" + str(file_count) + ".txt"
        file_count += 1
        if not os.path.exists(filename):
            break
        f = open(filename)
        line = f.readline()
        f.close()
        if line == "":
            continue
        actions = [int(s) for s in line.strip(" ").split(" ")]

        board = np.zeros((11, 11))
        to_play = 1
        for i in actions:
            x.append(board)
            y.append(to_play)
            board[i // 11][i % 11] = to_play
            to_play = 3 - to_play

    x = np.array(x)
    y = np.array(y)
    print(x.shape, y.shape)
    np.save("data/npy/rl_data_x.npy", x)
    np.save("data/npy/rl_data_to_play.npy", y)


if __name__ == "__main__":
    # load_raw_to_npy()
    load_raw_to_rl_data()