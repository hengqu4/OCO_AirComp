#!/usr/bin/env python
import math
import h5py
import matplotlib.pyplot as plt
import numpy as np
from utils.plot_utils import *
from oco_model import node
import torch

torch.manual_seed(0)
# TODO:
num_samples = 224
num_features = 14
num_nodes = 30
time_horizon = 300
ER_prob = 0.2
weight = 1 / (1 + num_nodes)
radius = 5


def data_generation(num_samples, num_features):
    coeff = np.random.rand(num_features, 1)
    X = np.random.rand(num_samples, num_features)
    noise = np.random.rand(num_samples, 1)
    y = X @ coeff + noise
    return X, y


def index_scheduler():
    return np.random.randint(0, num_samples, (time_horizon, num_nodes))


def dataset_iter(dataset):
    rand_index = np.random.randint(0, num_samples, (1, 1))[0]
    return dataset[0][:][rand_index][0], dataset[1][rand_index][0]


def find_optimal_solution(dataset, index):
    A = dataset[0][:][index]
    b = dataset[1][index]
    opt_sol = np.linalg.inv(A.T @ A) @ A.T @ b
    return opt_sol


def projection(lin_comb, radius):
    return (radius * lin_comb) / np.linalg.norm(lin_comb, ord=2)


def get_regret(decision, optimal_decision, schedule):
    all_regret = []
    for t in range(time_horizon):
        reg = 0
        for i in range(num_nodes):
            index = schedule[t][i]
            a = np.zeros((num_features, 1))
            for i in range(num_features):
                a[i] = dataset[0][index][i]
            b = dataset[1][index]
            reg += np.linalg.norm(a.T @ decision[t][i] - b) ** 2 - np.linalg.norm(a.T @ optimal_decision[i] - b) ** 2
        all_regret.append(reg)
            # print(np.linalg.norm(a.T @ decision[t][i] - b) ** 2 - np.linalg.norm(a.T @ optimal_decision[i] - b) ** 2)
    return all_regret

if __name__ == "__main__":
    dataset = data_generation(224, 14)
    schedule = index_scheduler()
    optimal_solution = []
    for i in range(num_nodes):
        optimal_solution.append(find_optimal_solution(dataset, schedule[:][i]))
    # print(len(optimal_solution), optimal_solution[0].shape)

    graph = node.Graph(num_nodes)
    decision = [[np.zeros((num_features, 1))] * num_nodes] * time_horizon
    # decision = np.zeros((time_horizon, num_nodes, num_features))
    half_step = [[np.zeros((num_features, 1))] * num_nodes] * time_horizon
    # print(decision)
    print(len(decision))
    # t=T-1
    for t in range(time_horizon - 1):
        graph.generate(ER_prob)
        # max_neighbour = max([graph.neighbour(index_agent)[0] for index_agent in range(num_nodes)])
        for i in range(num_nodes):
            num_neighbour, neighbours = graph.neighbour(i)
            index = schedule[t][i]
            # print(dataset)
            # print("dataset[0]：", type(dataset), type(dataset[0]), dataset[0].shape)
            a = np.zeros((num_features, 1))
            for i in range(num_features):
                a[i] = dataset[0][index][i]
            # a = dataset[0][index, :]
            b = dataset[1][index]
            # print("a:", type(a), a.shape)
            # print("b:", b)
            # print("decision [t][i]: ", decision[t][i])
            # print(decision[t][i] - 2 * a @ (a.T @ decision[t][i] - b))
            # print(decision[t][i])
            half_step[t][i] = decision[t][i] - 2 * (1 / math.sqrt(t + 1)) * (a @ (a.T @ decision[t][i] - b))
            lin_comb = (1 - weight * num_neighbour) * half_step[t][i]
            for neighbour in neighbours:
                lin_comb += weight * half_step[t][neighbour]
            decision[t + 1][i] = projection(lin_comb, radius)

    regret = get_regret(decision, optimal_solution, schedule)
    avg_regret = []
    for i in range(len(regret)):
    #     if i % 20
        avg_regret.append(regret[i] / (i + 1))

    print("avg_regret:",avg_regret)
    plt.plot(avg_regret)
    plt.xlabel("time_horizon(T)")
    plt.ylabel("S_Regret")
    plt.yscale("log")
    plt.show()

    # print("half_step[t][i]:", half_step[t][i])
