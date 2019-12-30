import torch
import random
from deap import base, creator, tools
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np

from Utils import Utils
from Net import Net
from DataLoader import retrieve_features_from_original_board

net_depth = 2
net_filters = 8
in_channels = 9

net = Net(net_depth, net_filters, in_channels)
net.load_state_dict(torch.load("model/model_ec_sig1.pt"))
net.eval()

board = np.zeros((11, 11))
to_play = 1
while True:
    features = retrieve_features_from_original_board(board)
    features = torch.from_numpy(features).type(torch.float)
    features = features.unsqueeze(0)
    # print(features.size())
    with torch.no_grad():
        outputs = net(features)
        outputs = outputs.squeeze(0)
        outputs = F.softmax(outputs, dim=0).view(11, 11) * 100
    Utils.print_plane(outputs)
    x, y = (int(i) for i in input("enter move:").strip(" ").split(" "))
    board[x][y] = to_play
    to_play = 3 - to_play