import torch
import random
from deap import base, creator, tools
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np

from Utils import Utils
from Net import Net

print("start")
net_depth = 2
net_filters = 8
in_channels = 9

net = Net(net_depth, net_filters, in_channels)
net.load_state_dict(torch.load("model/model_2_8.pt"))
net.eval()

psize = net.param_count()
print(psize)

train_loader = DataLoader(TensorDataset(*Utils.load_npy_to_tensor("train")),
                          batch_size=128, shuffle=True)
train_iter = iter(train_loader)

test_loader = DataLoader(TensorDataset(*Utils.load_npy_to_tensor("test")),
                         batch_size=100000, shuffle=True)

print("load data finished", len(train_loader))


def evaluate_all(individual):
    global test_loader
    net.set_param(individual)
    correct = 0
    total = 0
    with torch.no_grad():
        for test_data in test_loader:
            test_input, test_label = test_data
            test_output = net(test_input)

            _, predict = torch.max(test_output, 1)
            total += test_label.size(0)
            correct += (predict == test_label).sum().item()
    return -correct / total,


def evaluate_batch(individual):
    net.set_param(individual)
    global train_iter, train_loader
    correct = 0
    total = 0

    try:
        test_data = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        test_data = next(train_iter)

    test_input, test_label = test_data

    with torch.no_grad():
        test_output = net(test_input)

        _, predict = torch.max(test_output, 1)
        total += test_label.size(0)
        correct += (predict == test_label).sum().item()
    return correct / total,


def evaluate_batch_error(individual):
    net.set_param(individual)
    for i in test_loader:
        test_data = i
        break
    # global train_iter, train_loader
    # try:
    #     test_data = next(train_iter)
    # except StopIteration:
    #     train_iter = iter(train_loader)
    #     test_data = next(train_iter)

    with torch.no_grad():
        test_inputs, test_labels = test_data
        test_outputs = net(test_inputs)
        return F.cross_entropy(test_outputs, test_labels).item(),


def evaluate_all_error(individual):
    net.set_param(individual)
    err = 0
    total = 0
    with torch.no_grad():
        for test_data in test_loader:
            test_inputs, test_labels = test_data
            print(test_inputs.size())
            print(test_labels.size())
            test_outputs = net(test_inputs)
            err += F.cross_entropy(test_outputs, test_labels).item()
            return err,
            total += test_inputs.size(0)
    return err,


creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=psize)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

sigma = 1
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=sigma, indpb=0.5)
# toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("select", tools.selBest)
toolbox.register("evaluate", evaluate_all)


initial_param = net.param_to_list()
initial_individual = toolbox.individual()
for i in range(len(initial_individual)):
    initial_individual[i] = initial_param[i]
print("initial rate", evaluate_all(initial_individual)[0])
print("initial error", evaluate_all_error(initial_individual)[0])


def evolve():
    pop_size = 20
    pop = toolbox.population(n=pop_size)
    # pop.append(toolbox.clone(initial_individual))
    CXPB, MUTPB, NGEN = 0.5, 0.5, 500

    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
        print(ind.fitness.values)

    error = []
    correct_rate = []

    for g in range(NGEN):
        # offspring = toolbox.select(pop, len(pop))

        # offspring = list(map(toolbox.clone, offspring))

        # for child1, child2 in zip(offspring[::2], offspring[1::2]):
        #     if random.random() < CXPB:
        #         toolbox.mate(child1, child2)
        #         child1.fitness.values = toolbox.evaluate(child1)
        #         child2.fitness.values = toolbox.evaluate(child2)

        for mutant in pop:
            if random.random() < MUTPB:
                temp = toolbox.clone(mutant)
                toolbox.mutate(temp)
                temp.fitness.values = toolbox.evaluate(temp)
                pop.append(temp)
                # mutant.fitness.values = toolbox.evaluate(mutant)

        offspring = toolbox.select(pop, pop_size)
        offspring = list(map(toolbox.clone, offspring))

        m = 10000
        m_ind = offspring[0]
        for ind in offspring:
            if ind.fitness.values[0] < m:
                m = ind.fitness.values[0]
                m_ind = ind
        error.append(m)
        rate = evaluate_all(m_ind)[0]
        correct_rate.append(rate)
        print("error:", m, "rate:", rate)
        pop[:] = offspring

        if g % 20 == 0:
            np.save("data/statistics/ec/ec_error_" + "_pop" + str(
                pop_size) + "_gen" + str(g) + "_sig" + str(sigma), error)
    torch.save(net.state_dict(), "model/model_ec_sig" + str(sigma) + ".pt")

    # np.save("data/statistics/ec/ec_error_" + "_pop" + str(
    #     pop_size) + "_gen" + str(NGEN) + "_sig" + str(sigma), error)
    # np.save("data/statistics/ec/ec_rate_" + "_pop" + str(
    #     pop_size) + "_gen" + str(NGEN) + "_sig" + str(sigma), error)
    # np.save("data/statistics/ec/individual" + "_pop" + str(
    #     pop_size) + "_gen" + str(NGEN) + "_sig" + str(sigma), [i for i in m_ind])



evolve()