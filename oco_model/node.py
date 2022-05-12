# author:KangKi
# contact: panbk@shanghaitech.edu.cn
# datetime:2022/5/12 16:40
# software: PyCharm


"""
File description：
    init graph
    generate graph

"""
import numpy as np
import random
import math

class Graph:
    graph = []
    node_num = 0

    def __init__(self,node_num=30):
        self.node_num = node_num
        self.now_graph = None
        self.graph = np.array([[1]*node_num for i in range(node_num)])
        for i in range(node_num):
            self.graph[i][i] = 0

    def generate(self,prob=0.2):
        # pro_list = np.random.choice([0, 1], size=self.node_num, p=[prob, 1-prob])
        zero_size = math.ceil(self.node_num * prob)
        pro_list = [0] * zero_size + [1] * (self.node_num - zero_size)
        graph_now = np.zeros((self.node_num, self.node_num))
        for i in range(self.node_num):
            random.shuffle(pro_list)
            graph_now[i, ::] = pro_list
            graph_now[i, i] = 0
        self.now_graph = graph_now
        # return graph_now

    def neighbour(self, agent):
        count = 0
        neighbour = []
        for i in range(self.node_num):
            if self.now_graph[i][agent] == 1:
                count = 1
                neighbour.append(i)
        return count, neighbour

if __name__ == "__main__":
    g = Graph(4)
    print(g.graph)
    g.generate(0.5)
    print(g.graph)