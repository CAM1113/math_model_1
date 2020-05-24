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


def prime(nodes, distance_matrix, is_chosen, edges):
    # 选择供水中心作为开始节点
    # 并不是所有node都被选中
    # print("distance_matrix{}".format(distance_matrix))
    while not is_chosen.all():
        # print("is_chosen = {}".format(is_chosen.sum()))
        edge, end = get_shortest_edge(nodes, distance_matrix, is_chosen, edges)
        edges.append(edge)
        # print("eg.start_node = {},{},{}".format(edge.start_node.index, edge.end_node.index, edge.distance))
        is_chosen[end] = 1
        # 集合内部的距离设置为max
        print("get one edge")

    return edges


# 获取最短边,和被选出的node的下标
def get_shortest_edge(nodes, distance_matrix, is_chosen, edges):
    min_start = MAX_LENGTH
    min_end = MAX_LENGTH
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
                # 验证添加这条边后，是否满足要求
                edge = Edge(nodes[index], nodes[index_row])
                edges.append(edge)

                print("get_i_node = {},nodes[index] = {},nodes[index_row] = {}".format(len(edges), nodes[index].index,
                                                                                       nodes[index_row].index))
                i_node = get_i_node(nodes[index_row], edges)
                dis = compute_length_from_node(i_node, edges)
                edges.remove(edge)
                if dis > 40:
                    continue
                min_end = index_row
                min_start = index
                min_length = num_row

    if min_start == MAX_LENGTH:
        print("min_start == -1")
        print(np.where(is_chosen == 0))
        draw_line(nodes, edges)

    edge = Edge(nodes[min_start], nodes[min_end])
    print("get_edge")
    return edge, min_end


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


def get_edges(nodes, pre_node, is_draw):
    # is_chosen = np.zeros(shape=(len(pre_node)))
    # is_chosen[0] = 1
    # distance_matrix = build_distance_matrix(pre_node)
    edge1 = []
    # edge1 = prime(pre_node, distance_matrix, is_chosen, edge1)
    distance_matrix = build_distance_matrix(nodes)
    # Prim算法构建生成树
    is_chosen = np.zeros(shape=(len(nodes)))
    for node in pre_node:
        is_chosen[node.index] = 1
    print(np.where(is_chosen == 1))
    edge1.clear()
    print("二次建树")
    edges2 = prime(nodes, distance_matrix, is_chosen, edge1)
    print("edges2 = {}".format(len(edges2)))

    if is_draw:
        draw_line(nodes, edge1)
    return edge1


def get_next_edge(edges, start_node):
    next_edge = []
    for eg in edges:
        if eg.start_node.index == start_node.index:
            next_edge.append(eg)
    return next_edge


# 递归计算从某个节点开始的子树的长度
def compute_length_from_node(start_node, edges):
    next_edges = get_next_edge(edges, start_node)
    if len(next_edges) == 0:
        return 0
    distance = 0
    for eg in next_edges:
        distance = distance + eg.distance
        # 一型管道，代表其下一个节点是一级节点
        if eg.pip_type == PIP_TYPE_I or NODE_TYPE_V in eg.end_node.node_type:
            continue
        distance = distance + compute_length_from_node(eg.end_node, edges)
    return distance


# 递归查找树的父节点，返回父节点中的i级供水站
def get_i_node(node, edges):
    # print(" {},edge.len = {}".format(node.index, len(edges)))
    if NODE_TYPE_V in node.node_type or NODE_TYPE_A in node.node_type:
        print("return node")
        return node
    for eg in edges:
        if eg.end_node.index == node.index:
            return get_i_node(eg.start_node, edges)
    print("error")
    return None


def get_father_edge(pre_node, distance_matrix, is_chosen, edges):
    min_start = MAX_LENGTH
    min_end = MAX_LENGTH
    min_length = MAX_LENGTH

    for index in range(len(pre_node)):
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
                # 验证添加这条边后，是否满足要求
                # 从中心节点通向一级节点，可以
                if NODE_TYPE_A in pre_node[index].node_type:
                    min_end = index_row
                    min_start = index
                    min_length = num_row

                # 从一级节点通向一级节点
                if compute_length_from_node(pre_node[index], edges) + num_row > 40:
                    # 添加这一节点后，连接大于了40，放弃这一节点
                    continue
                min_end = index_row
                min_start = index
                min_length = num_row

    if min_start == MAX_LENGTH:
        print("min_start == -1")
        print(np.where(is_chosen == 0))

    edge = Edge(pre_node[min_start], pre_node[min_end])
    print("get_edge")
    return edge, min_end


def get_father_edges(pre_node, edges):
    distance_matrix = build_distance_matrix(pre_node)
    is_chosen = np.zeros(len(pre_node))
    is_chosen[0] = 1
    while not is_chosen.all():
        edge, end = get_father_edge(pre_node, distance_matrix, is_chosen, edges)
        edges.append(edge)
        # print("eg.start_node = {},{},{}".format(edge.start_node.index, edge.end_node.index, edge.distance))
        is_chosen[end] = 1
        # 集合内部的距离设置为max
        print("get one edge")
    return edges




def is_over(category_list):
    if (category_list == 0).any():
        return False
    head = category_list[0]
    for item in category_list:
        if item != head:
            return False
    return True


def get_short_edge_not_same_category(node, category_list, distance_matrix, edges):
    min_row = MAX_LENGTH
    min_col = MAX_LENGTH
    min_distance = MAX_LENGTH
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if distance_matrix[i, j] < min_distance:
                if category_list[i] == 0 or category_list[j] == 0 or category_list[i] != category_list[j]:
                    if NODE_TYPE_A in node[i].node_type:
                        min_distance = distance_matrix[i, j]
                        min_row = i
                        min_col = j
                        continue

                    # 从一级节点通向一级节点
                    if compute_length_from_node(node[i], edges) + distance_matrix[i, j] > 40:
                        # 添加这一节点后，连接大于了40，放弃这一节点
                        continue

                    min_distance = distance_matrix[i, j]
                    min_row = i
                    min_col = j

    return min_row, min_col, min_distance


def get_father_edges_2(pre_nodes, edges, distance_matrix):
    category = 1
    category_list = np.zeros(len(pre_nodes))
    # 选择供水中心作为开始节点
    while not is_over(category_list):
        # 查找符合要求的最短边
        min_row, min_col, min_distance = get_short_edge_not_same_category(pre_nodes, category_list, distance_matrix,
                                                                          edges)
        print(category_list)
        if (category_list[min_row] == category_list[min_col] and category_list[min_col] != 0):
            print("error in get edge")
        edge = Edge(pre_nodes[min_row], pre_nodes[min_col])
        edges.append(edge)
        # print("eg.start_node = {},{},{}".format(edge.start_node.index, edge.end_node.index, edge.distance))

        # 修改类别列表
        if category_list[min_row] == 0 and category_list[min_col] == 0:
            category_list[min_row] = category
            category_list[min_col] = category
            category += 1

        elif category_list[min_row] == 0 and category_list[min_col] != 0:
            category_list[min_row] = category_list[min_col]

        elif category_list[min_row] != 0 and category_list[min_col] == 0:
            category_list[min_col] = category_list[min_row]

        elif category_list[min_row] != 0 and category_list[min_col] != 0:
            ca = category_list[min_col]
            for ii in range(category_list.shape[0]):
                if category_list[ii] == ca:
                    category_list[ii] = category_list[min_row]
    return edges


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


def main_3():
    nodes = read_data()
    # 89,  94,  95, 109 4个点是离群点
    # 94：633.2633195813704
    # 89：633.2075
    # 95：635.6650738323617
    # 109：632.5095308326057
    nodes[109].node_type = "V109"
    pre_nodes = []
    for node in nodes:
        if NODE_TYPE_P in node.node_type:
            continue
        else:
            print(node.node_type)
            pre_nodes.append(node)


    for i in range(len(nodes)):
        nodes[i].index = i
    # 创建以一级节点为根节点的子树
    edges = get_edges(nodes, pre_nodes, True)
    # 将中心节点和一级节点创建子树
    distance_matrix = build_distance_matrix(pre_nodes)
    edges2 = get_father_edges_2(pre_nodes, edges, distance_matrix)
    print("edges = {}".format(len(edges)))

    draw_line(nodes, edges2)
    i, ii, sum_ = compute_length(edges2)
    print("一型管道：{}".format(i))
    print("二型管道：{}".format(ii))
    print("总管道：{}".format(sum_))
    for node in pre_nodes:
        if NODE_TYPE_A in node.node_type:
            continue
        print("{}出发的管道长度={}".format(node.node_type, compute_length_from_node(node, edges2)))


if __name__ == '__main__':
    main_3()
