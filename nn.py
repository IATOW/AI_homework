import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from Net import Net
from Utils import Utils


def run():
    net_depth = 2
    net_filters = 8
    in_channels = 9
    epochs = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    net = Net(net_depth, net_filters, in_channels)
    # net.load_state_dict(torch.load("model/model_8_128_black.pt"))
    # net.eval()
    net.to(device)
    print(net)

    train_loader = DataLoader(TensorDataset(*Utils.load_npy_to_tensor("train")),
                              batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(TensorDataset(*Utils.load_npy_to_tensor("test")),
                             batch_size=128, shuffle=True, num_workers=2)

    print("load data finished")

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    batch_index = 0
    correct_rate = []
    for e in range(epochs):
        print("epoch:", e)
        for idx, data in enumerate(train_loader):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            # loss = criterion(outputs, labels)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            optimizer.step()

            batch_index += 1

            if batch_index % 50 == 0:
                correct = 0
                total = 0
                for test_data in test_loader:
                    test_input, test_label = test_data[0].to(device), test_data[1].to(device)
                    test_output = net(test_input)

                    _, predict = torch.max(test_output, 1)
                    total += test_label.size(0)
                    correct += (predict == test_label).sum().item()
                correct_rate.append(correct / total)
                print(correct / total)
                print("loss:", loss.item())

    torch.save(net.state_dict(), "model/model_" + str(net_depth) + "_" + str(net_filters) + ".pt")
    np.save("data/statistics/correct_rate_" + str(net_depth) + "_" + str(net_filters), correct_rate)


if __name__ == "__main__":
    run()