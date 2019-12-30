import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Utils import Utils


class Net(nn.Module):
    def __init__(self, depth, filters_count, in_channel):
        super(Net, self).__init__()

        self.depth = depth
        self.filters_count = filters_count
        self.in_channel = in_channel

        self.conv1 = nn.Conv2d(in_channel, filters_count, 5)
        self.conv2 = nn.ModuleList([nn.Conv2d(filters_count, filters_count, 3, padding=1) for _ in range(depth - 1)])
        # self.conv2 = nn.ModuleList([nn.Conv2d(w, w, 5, padding=2) for _ in range(d - 1)])
        self.conv3 = nn.Conv2d(filters_count, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        for i in range(self.depth - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
        x = self.conv3(x)
        x = x.view(-1, 121)
        return x

    def param_count(self):
        ans = 0
        for i in self.parameters():
            temp = 1
            for j in i.data.size():
                temp *= j
            ans += temp
        return ans

    def set_param(self, params):
        it = 0
        for param in self.parameters():
            length = len(param.data.size())
            if length == 4:
                for p1 in range(param.size(0)):
                    for p2 in range(param.size(1)):
                        for p3 in range(param.size(2)):
                            for p4 in range(param.size(3)):
                                param.data[p1][p2][p3][p4] = params[it]
                                it += 1
            elif length == 1:
                for p1 in range(param.size(0)):
                    param.data[p1] = params[it]
                    it += 1

    def param_to_list(self):
        ans = []
        for param in self.parameters():
            length = len(param.data.size())
            if length == 4:
                for p1 in range(param.size(0)):
                    for p2 in range(param.size(1)):
                        for p3 in range(param.size(2)):
                            for p4 in range(param.size(3)):
                                ans.append(param.data[p1][p2][p3][p4])
            elif length == 1:
                for p1 in range(param.size(0)):
                    ans.append(param.data[p1])
        return ans
