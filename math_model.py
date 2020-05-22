from time import time

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
np.set_printoptions(linewidth=400)


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
            print("供水方向错误,start_node = {},end_node = {}".format(start_node.index, end_node.index))
        self.distance = compute_distance(start_node, end_node)


def draw_line(nodes, edges):
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

    # 画中心二级供水站
    x = []
    y = []
    for node in II_nodes:
        x.append(node.node_x)
        y.append(node.node_y)
    x = np.array(x)
    y = np.array(y)
    plt.scatter(x, y, c='g', s=2, linewidth=2)
    plt.scatter(x, y)

    # 画中心一级供水站
    x = []
    y = []
    for node in I_nodes:
        x.append(node.node_x)
        y.append(node.node_y)
    x = np.array(x)
    y = np.array(y)
    plt.scatter(x, y, c='y', s=4, linewidth=4)
    plt.scatter(x, y)
    tims = time()
    # plt.savefig('{}.jpg'.format("row"))
    plt.show()


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
    min_start = -1
    min_end = -1
    min_length = MAX_LENGTH

    for index in range(len(nodes)):
        # 边的起点在已生成的树中早
        if is_chosen[index] == 0:
            continue
        row = distance_matrix[index]
        # print("row = {}".format(row))
        for index_row, num_row in enumerate(row):
            # index_row对应的节点已经被选择过，不能作为新找的边
            if is_chosen[index_row] == 1:
                continue
            if num_row < min_length:
                min_end = index_row
                min_start = index
                min_length = num_row

    edge = Edge(nodes[min_start], nodes[min_end])
    return edge, min_end


# 计算管道长度
def compute_length(edges):
    len_pip_type_I = 0
    len_pip_type_II = 0
    for eg in edges:
        if eg.pip_type == PIP_TYPE_I:
            len_pip_type_I += eg.distance
        else:
            len_pip_type_II += eg.distance
    return len_pip_type_I, len_pip_type_II, len_pip_type_I + len_pip_type_II


def prime(nodes, distance_matrix, is_chosen):
    # 选择供水中心作为开始节点
    edges = []
    # 并不是所有node都被选中
    # print("distance_matrix{}".format(distance_matrix))
    while not is_chosen.all():
        # print("is_chosen = {}".format(is_chosen.sum()))
        edge, end = get_shortest_edge(nodes, distance_matrix, is_chosen)
        edges.append(edge)
        # print("eg.start_node = {},{},{}".format(edge.start_node.index, edge.end_node.index, edge.distance))
        is_chosen[end] = 1
        # 集合内部的距离设置为max

    return edges


def get_edges(nodes, is_draw):
    pre_node = []
    for node in nodes:
        if NODE_TYPE_P in node.node_type:
            continue
        else:
            pre_node.append(node)
    is_chosen = np.zeros(shape=(len(pre_node)))
    is_chosen[0] = 1
    distance_matrix = build_distance_matrix(pre_node)
    edge1 = prime(pre_node, distance_matrix, is_chosen)

    # 已选出的边更新权重
    distance_matrix = build_distance_matrix(nodes)
    # Prim算法构建生成树
    is_chosen = np.zeros(shape=(len(nodes)))
    for node in pre_node:
        is_chosen[node.index] = 1
    edges2 = prime(nodes, distance_matrix, is_chosen)
    for eg in edges2:
        edge1.append(eg)
    if is_draw:
        draw_line(nodes, edge1)
    return edge1


def compute_tree(nodes, is_draw):
    pre_node = []
    for node in nodes:
        if NODE_TYPE_P in node.node_type:
            continue
        else:
            pre_node.append(node)
    is_chosen = np.zeros(shape=(len(pre_node)))
    is_chosen[0] = 1
    distance_matrix = build_distance_matrix(pre_node)
    edge1 = prime(pre_node, distance_matrix, is_chosen)

    # 已选出的边更新权重
    distance_matrix = build_distance_matrix(nodes)
    # Prim算法构建生成树
    is_chosen = np.zeros(shape=(len(nodes)))
    for node in pre_node:
        is_chosen[node.index] = 1
    edges2 = prime(nodes, distance_matrix, is_chosen)
    for eg in edges2:
        edge1.append(eg)
    if is_draw:
        draw_line(nodes, edge1)
    i, ii, sum_ = compute_length(edge1)
    return i, ii, sum_


def main_1():
    nodes = read_data()
    i, ii, sum_ = compute_tree(nodes, True)
    print(i, ii, sum_)


# 暴力法，弃用
def main_2():
    nodes = read_data()
    i, ii, sum_ = compute_tree(nodes, False)
    min_ii_distance_1 = ii
    min_ii_distance_2 = min_ii_distance_1
    index_1 = -1
    index_2 = -1

    for i1 in np.arange(len(nodes)):
        for i2 in np.arange(start=i1 + 1, stop=len(nodes), step=1):
            #
            # ran = np.random.random(size=(1, 2))
            # if ran[0][1] > 0.005:
            #     continue
            if NODE_TYPE_P in nodes[i1].node_type and NODE_TYPE_P in nodes[i2].node_type:
                nodes[i1].node_type = NODE_TYPE_V
                nodes[i2].node_type = NODE_TYPE_V
                i_2, ii_2, sum_2 = compute_tree(nodes, False)
                if ii_2 < min_ii_distance_2:
                    min_ii_distance_2 = ii_2
                    index_1 = i1
                    index_2 = i2
                    print("current min i1 = {},i2 = {},dis = {}".format(i1, i2, min_ii_distance_2))
                nodes[i1].node_type = NODE_TYPE_P
                nodes[i2].node_type = NODE_TYPE_P
    print("index_1 = {},index_2 ={},min_ii_distance_2={}".format(index_1, index_2, min_ii_distance_2))


def main_2_2():
    nodes = read_data()
    pre_node = []
    for node in nodes:
        if NODE_TYPE_P in node.node_type:
            continue
        else:
            pre_node.append(node)
    is_chosen = np.zeros(shape=(len(pre_node)))
    is_chosen[0] = 1
    distance_matrix = build_distance_matrix(pre_node)
    edge1 = prime(pre_node, distance_matrix, is_chosen)

    # 已选出的边更新权重
    distance_matrix = build_distance_matrix(nodes)
    # Prim算法构建生成树
    is_chosen = np.zeros(shape=(len(nodes)))
    for node in pre_node:
        is_chosen[node.index] = 1
    edges2 = prime(nodes, distance_matrix, is_chosen)
    for eg in edges2:
        edge1.append(eg)
    # 计算管道整体情况
    i, ii, sum_ = compute_length(edge1)
    # 查找二型管道中最大和次大的管道
    max_length = 0
    max_eg = None
    second_max_eg = None
    for eg in edge1:
        if eg.pip_type == PIP_TYPE_I:
            # 一型管道不用管
            continue
        if eg.distance > max_length:
            second_max_eg = max_eg
            max_length = eg.distance
            max_eg = eg
    print("II型管道：{}".format(ii))
    print("最长二型管道：{}->{} = {}".format(max_eg.start_node.node_type, max_eg.end_node.node_type, max_eg.distance))
    print("次大二型管道：{}->{}={}".format(second_max_eg.start_node.node_type, second_max_eg.end_node.node_type,
                                    second_max_eg.distance))
    print("修改点1：{}".format(max_eg.end_node.node_type))
    print("修改点2：{}".format(second_max_eg.end_node.node_type))
    print("减少距离：{}".format(ii - max_eg.distance - second_max_eg.distance))


# 获取某条边在边集合中的下标
# def get_index_in_edges(eg, edges):
#     for index in range(len(edges)):
#         if edges[index].start_node.index == eg.index:
#             return index
#     return None


if __name__ == '__main__':
    main_1()
