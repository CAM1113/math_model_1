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
            assert "供水方向错误"
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
    plt.scatter(x, y)

    # 画中心一级供水站
    x = []
    y = []
    for node in I_nodes:
        x.append(node.node_x)
        y.append(node.node_y)
    x = np.array(x)
    y = np.array(y)
    plt.scatter(x, y)
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
    min_start = 0
    min_end = 0
    min_length = MAX_LENGTH

    # 找出已被选择的所有节点的下标
    for index in range(len(nodes)):
        if is_chosen[index] == 0:
            continue
        row = distance_matrix[index]
        # print("row = {}".format(row))
        for index_row, num_row in enumerate(row):
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


def is_over(category_list):
    if (category_list == 0).any():
        return False
    head = category_list[0]
    for item in category_list:
        if item != head:
            return False
    return True


def get_short_edge_not_same_category(category_list, distance_matrix):
    min_row = -1
    min_col = -1
    min_distance = MAX_LENGTH
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if distance_matrix[i, j] < min_distance:
                if category_list[i] == 0 or category_list[j] == 0 or category_list[i] != category_list[j]:
                    min_distance = distance_matrix[i, j]
                    min_row = i
                    min_col = j
    return min_row, min_col, min_distance


def ke_lu_si_ka_er(category_list, pre_nodes, distance_matrix):
    edges = []
    category = 1
    # 选择供水中心作为开始节点
    while not is_over(category_list):
        # 查找符合要求的最短边
        min_row, min_col, min_distance = get_short_edge_not_same_category(category_list, distance_matrix)
        print(category_list)
        if (category_list[min_row] == category_list[min_col] and category_list[min_col] != 0):
            assert "error in get edge"
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


def main():
    nodes = read_data()

    pre_nodes = nodes[:13]
    category_list = np.zeros(shape=(len(pre_nodes)))
    distance_matrix = build_distance_matrix(pre_nodes)
    edges = ke_lu_si_ka_er(category_list, pre_nodes, distance_matrix)
    for eg in edges:
        print("{}->{} = {}".format(eg.start_node.index, eg.end_node.index, eg.distance))

    category_list = np.zeros(shape=(len(nodes)))
    for iii in range(13):
        category_list[iii] = -1
    distance_matrix = build_distance_matrix(nodes)
    edges2 = ke_lu_si_ka_er(category_list, nodes, distance_matrix)
    for e in edges2:
        edges.append(e)

    draw_line(nodes, edges)
    i, ii, sum_ = compute_length(edges)
    print(i, ii, sum_)


#     找出能源不足的二级供水站
#     for


if __name__ == '__main__':
    main()
