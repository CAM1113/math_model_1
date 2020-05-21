
import numpy as np
import math
import re
import matplotlib.pyplot as plt

NODE_TYPE_A = "A"
NODE_TYPE_V = "V"
NODE_TYPE_P = "P"
PIP_TYPE_I = "I"
PIP_TYPE_II = "II"
MAX_LENGTH = float("inf")


class Node:
    def __init__(self):
        self.index = 0
        self.node_type = ""
        self.node_x = 0.0
        self.node_y = 0.0


class Edge:
    def __init__(self, start_node, end_node):
        self.start_node = start_node
        self.end_node = end_node
        # 供水中心向1级供水站供水
        if NODE_TYPE_A in start_node.node_type and NODE_TYPE_V in end_node.node_type:
            self.pip_type = PIP_TYPE_I
        # 一级供水站之间供水
        elif NODE_TYPE_V in start_node.node_type and NODE_TYPE_V in end_node.node_type:
            self.pip_type = PIP_TYPE_I

        # 一级供水站向二级供水站供水
        elif NODE_TYPE_V in start_node.node_type and NODE_TYPE_P in end_node.node_type:
            self.pip_type = PIP_TYPE_II
        #  二级供水站之间供水
        elif NODE_TYPE_P in start_node.node_type and NODE_TYPE_P in end_node.node_type:
            self.pip_type = PIP_TYPE_II
        else:
            assert "供水方向错误"
        self.distance = compute_distance(start_node, end_node)


# 读取数据文件，创建节点列表
def read_data(path="./data.txt"):
    f = open(path, "r")
    lines = f.readlines()  # 读取全部内容 ，并以列表方式返回
    nodes = []
    for line in lines:
        data = re.split(r"	", line)
        node = Node()
        node.index = int(data[0])
        node.node_type = data[1]
        node.node_x = float(data[2])
        node.node_y = float(data[3])
        nodes.append(node)
    f.close()
    return nodes


def compute_distance(start_node, end_node):
    distance = MAX_LENGTH
    NODE_TYPE_A = "A"
    NODE_TYPE_V = "V"
    NODE_TYPE_P = "P"

    # 供水中心向1级供水站供水
    if NODE_TYPE_A in start_node.node_type and NODE_TYPE_V in end_node.node_type:
        distance = math.sqrt((start_node.node_x - end_node.node_x) ** 2 + (start_node.node_y - end_node.node_y) ** 2)
    # 一级供水站之间供水
    elif NODE_TYPE_V in start_node.node_type and NODE_TYPE_V in end_node.node_type:
        distance = math.sqrt((start_node.node_x - end_node.node_x) ** 2 + (start_node.node_y - end_node.node_y) ** 2)

    # 一级供水站向二级供水站供水
    elif NODE_TYPE_V in start_node.node_type and NODE_TYPE_P in end_node.node_type:
        distance = math.sqrt((start_node.node_x - end_node.node_x) ** 2 + (start_node.node_y - end_node.node_y) ** 2)
    #  二级供水站之间供水
    elif NODE_TYPE_P in start_node.node_type and NODE_TYPE_P in end_node.node_type:
        distance = math.sqrt((start_node.node_x - end_node.node_x) ** 2 + (start_node.node_y - end_node.node_y) ** 2)
    return distance


def build_distance_matrix(nodes):
    length = len(nodes)
    matrix = np.zeros(shape=(length, length))
    for i in np.arange(length):
        for j in np.arange(length):
            if i == j:
                dis = MAX_LENGTH
                matrix[i, j] = dis
                continue
            dis = compute_distance(nodes[i], nodes[j])
            matrix[i, j] = dis
    return matrix


# 获取最短边,和被选出的node的下标
def get_shortest_edge(nodes, distance_matrix, is_chosen):
    min_start = 0
    min_end = 0
    min_length = MAX_LENGTH

    # 找出已被选择的所有节点的下标
    chosen_index = np.argwhere(is_chosen == 1)
    for index in chosen_index:
        row = distance_matrix[index][0]
        # print("row = {}".format(row))
        min_distance_in_row = np.min(row)
        if min_distance_in_row < min_length:
            min_end = int(np.argwhere(row == min_distance_in_row)[0])
            min_start = index[0]
            min_length = min_distance_in_row
    # print("min_start = {}".format(min_start))
    # print("min_end = {}".format(min_end))
    edge = Edge(nodes[min_start], nodes[min_end])
    return edge, min_end


if __name__ == '__main__':
    nodes = read_data()
    distance_matrix = build_distance_matrix(nodes)
    print("max = {}".format(distance_matrix))
    for node in nodes:
        print(node.index, node.node_type, node.node_x, node.node_y)

    # Prim算法构建生成树
    is_chosen = np.zeros(shape=(len(nodes)))
    # 选择供水中心作为开始节点
    edges = []
    is_chosen[0] = 1
    # 并不是所有node都被选中
    times = 0
    print("distance_matrix{}".format(distance_matrix))
    while not is_chosen.all():
        print("is_chosen = {}".format(is_chosen.sum()))
        edge, end = get_shortest_edge(nodes, distance_matrix, is_chosen)
        edges.append(edge)
        # print("eg.start_node = {},{},{}".format(edge.start_node.index, edge.end_node.index, edge.distance))
        is_chosen[end] = 1
        # 集合内部的距离设置为max
        chosen_index = np.argwhere(is_chosen == 1)
        for index in chosen_index:
            distance_matrix[index, end] = MAX_LENGTH
            distance_matrix[end, index] = MAX_LENGTH
        # print("matrix after = {}".format(distance_matrix))

    for eg in edges:
        x = [eg.start_node.node_x, eg.end_node.node_x]
        y = [eg.start_node.node_y, eg.end_node.node_y]
        color = 'r'
        if eg.pip_type == PIP_TYPE_II:
            color = 'b'
        plt.plot(np.array(x), np.array(y), color)
    center_nodes = []
    I_nodes = []
    II_nodes = []
    for node in nodes:
        # NODE_TYPE_A = "A"
        # NODE_TYPE_V = "V"
        # NODE_TYPE_P = "P"
        if NODE_TYPE_A in node.node_type:
            center_nodes.append(node)
        if NODE_TYPE_V in node.node_type:
            I_nodes.append(node)
        if NODE_TYPE_P in node.node_type:
            II_nodes.append(node)

    # 画中心供水站
    x = []
    y = []
    for node in center_nodes:
        x.append(node.node_x)
        y.append(node.node_y)
    x = np.array(x)
    y = np.array(y)
    plt.scatter(x, y, c='r', s=5, linewidth=5)

    # 画中心一级供水站
    x = []
    y = []
    for node in I_nodes:
        x.append(node.node_x)
        y.append(node.node_y)
    x = np.array(x)
    y = np.array(y)
    plt.scatter(x, y)

    # 画中心二级供水站
    x = []
    y = []
    for node in II_nodes:
        x.append(node.node_x)
        y.append(node.node_y)
    x = np.array(x)
    y = np.array(y)
    plt.scatter(x, y)

    plt.show()
