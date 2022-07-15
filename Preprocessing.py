import csv
import pandas as pd
import click
import dgl
import jieba
import torch
from pathlib import Path
import numpy as np
from scapy.compat import raw
from scapy.all import *
from scapy.error import Scapy_Exception
from scapy.layers.inet import IP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scipy import sparse
from utils.utilsformalware import should_omit_packet, read_pcap, PREFIX_TO_Malware_ID
from utils.utilsforapps import PREFIX_TO_APP_ID
from utils.utilsforentropy import PREFIX_TO_ENTROPY_ID, ID_TO_ENTROPY
from dgl.data.utils import save_graphs


def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet


def mask_ip(packet):
    if IP in packet:
        packet[IP].src = '0.0.0.0'
        packet[IP].dst = '0.0.0.0'

    return packet


def pad_udp(packet):
    if UDP in packet:
        # get layers after udp
        layer_after = packet[UDP].payload.copy()

        # build a padding layer
        pad = Padding()
        pad.load = '\x00' * 12

        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

        return packet

    return packet


# def packet_to_sparse_array(packet, max_length=1500):
#     arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length]
#     # arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length] / 255
#     if len(arr) < max_length:
#         pad_width = max_length - len(arr)
#         arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
#     # print(arr)
#     arr = sparse.csr_matrix(arr)
#     return arr
def packet_to_sparse_array(packet, max_length=1500):
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length]
    # arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length] / 255
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
    arr = sparse.csr_matrix(arr)
    return arr


def transform_packet(packet):
    # if should_omit_packet(packet):
    #     return None

    packet = remove_ether_header(packet)
    packet = pad_udp(packet)
    packet = mask_ip(packet)
    # wrpcap("/home/pcl/PangBo/pro/GNNTrafficClassification/Dataset/ProcessedData/" + name, packet, append=True)
    arr = packet_to_sparse_array(packet)

    return arr


# def transform_pcap_malware(path, listofnodenum_test, listofkind_test, feature_matrix_test):
#     # output_path = 'processdata/' + path.name
#     # if Path(output_path + '_SUCCESS').exists():
#     #     return
#     # with Path(output_path + '_SUCCESS').open('w') as f:
#     #     f.write('')
#     # print(path,"Done")
#     # print(path)
#     # try:
#     #     read_pcap(path)
#     # except Scapy_Exception:
#     #     os.remove(path)
#     #     return
#     rows = []
#     j = 0
#     for i, packet in enumerate(read_pcap(path)):
#         arr = transform_packet(packet)
#         if arr is not None:
#             prefix = path.name.split('.')[0]
#             app_label = PREFIX_TO_Malware_ID.get(prefix)
#             j = j + 1
#             rows.append(arr.todense().tolist()[0])
#     if j > 1:
#         listofnodenum_test.append(j)
#         listofkind_test.append(app_label)
#         feature_matrix_test.append(rows)
#         print(path, 'Done')
#
#
# def transform_pcap_entropy(path, listofnodenum, listofkind, feature_matrix):
#     rows = []
#     j = 0
#     for i, packet in enumerate(read_pcap(path)):
#         arr = transform_packet(packet)
#         if arr is not None:
#             prefix = path.name.split('.')[0]
#             app_label = PREFIX_TO_ENTROPY_ID.get(prefix)
#             j = j + 1
#             rows.append(arr.todense().tolist()[0])
#     if j > 1:
#         listofnodenum.append(j)
#         listofkind.append(app_label)
#         feature_matrix.append(rows)
#         print(path, 'Done')


def transform_pcap(pcap_path, graphlist, listofkind):
    feature_matrix = []
    text_feature = []
    direcs = []
    print(pcap_path)
    with open(pcap_path) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        for row in csv_reader:
            feature_matrix.append(list(map(float,row[0].split(','))))
            text_feature.append(np.array(list(map(float,row[1].split(',')))))
            # print(np.array(list(map(float,row[1].split(',')))))
            # print(len(np.array(list(map(float,row[1].split(','))))))
            direcs.append(int(row[2]))
    if len(direcs) > 0:
        node_num = int(pcap_path.name.split('_')[0])
        label = int(pcap_path.name.split('_')[1])
        g = single_graph(node_num, feature_matrix, text_feature, direcs)
        listofkind.append(label)
        graphlist.append(g)


class graphdataset(object):
    def __init__(self, graph_list, listofkind, num_classes):
        super(graphdataset, self).__init__()
        self.graphs = graph_list
        self.num_classes = num_classes
        self.labels = listofkind
        # self.feature_matrix=feature_matrix

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, int)
            The graph and its label.
        """
        return self.graphs[idx], self.labels[idx]


def single_graph(node_num, node_feature, text_feature, direcs):
    list_point = []
    list_point_2 = []
    for i in range(node_num - 1):
        list_point.append(i)
        list_point_2.append(i + 1)
    for i in range(node_num - 1):
        list_point.append(i + 1)
        list_point_2.append(i)
    # 加上TCP的边
    for i in range(len(direcs) - 2):
        if direcs[i] == direcs[i + 1] and direcs[i] != direcs[i + 2]:
            temp = i + 2
            while (direcs[i] != direcs[i + 2]):
                list_point.append(temp)
                list_point_2.append(i)
                i = i - 1

    # 构图
    g = dgl.graph((list_point, list_point_2))
    g = dgl.add_self_loop(g)
    g.ndata['h'] = torch.tensor(node_feature)
    g.ndata['x'] = torch.tensor(text_feature)
    return g


def make_graph(data_dir_path, valid_data_dir_path, testdata_dir_path, kind):
    testdata_dir_path = Path(testdata_dir_path)
    valid_data_dir_path = Path(valid_data_dir_path)
    data_dir_path = Path(data_dir_path)
    if kind == 'app':
        label_dic = PREFIX_TO_APP_ID
    elif kind == 'malware':
        label_dic = PREFIX_TO_Malware_ID
    else:
        label_dic = PREFIX_TO_ENTROPY_ID
    num_classes = len(label_dic)

    # 训练集
    graphlist = []
    listofkind = []
    for pcap_path in sorted(data_dir_path.iterdir()):
        transform_pcap(pcap_path, graphlist, listofkind)

    # trainset = graphdataset(graphlist, listofkind, num_classes)
    #
    # # 验证集
    # graphlist = []
    # listofkind = []
    # for pcap_path in sorted(valid_data_dir_path.iterdir()):
    #     transform_pcap(pcap_path, graphlist, listofkind)
    # validset = graphdataset(graphlist, listofkind, num_classes)
    #
    # # 测试集
    # graphlist = []
    # listofkind = []
    # for pcap_path in sorted(testdata_dir_path.iterdir()):
    #     transform_pcap(pcap_path, graphlist, listofkind)
    # testset = graphdataset(graphlist, listofkind, num_classes)
    train_g = []
    train_l = []
    valid_g = []
    valid_l = []
    test_g = []
    test_l = []
    temp_g = []
    temp_l = []
    for i in range(len(graphlist)):
        if i % 10 == 0:
            valid_g.append(graphlist[i])
            valid_l.append(listofkind[i])
        else:
            temp_g.append(graphlist[i])
            temp_l.append(listofkind[i])
    for i in range(len(temp_g)):
        if i % 9 == 0:
            test_g.append(temp_g[i])
            test_l.append(temp_l[i])
        else:
            train_g.append(temp_g[i])
            train_l.append(temp_l[i])
    trainset = graphdataset(train_g, train_l, num_classes)
    # 验证集
    # graphlist = []
    # listofkind = []
    # for pcap_path in sorted(valid_data_dir_path.iterdir()):
    #     transform_pcap(pcap_path, graphlist, listofkind,kind)
    validset = graphdataset(valid_g, valid_l, num_classes)

    # 测试集
    # graphlist = []
    # listofkind = []
    # for pcap_path in sorted(testdata_dir_path.iterdir()):
    #     transform_pcap(pcap_path, graphlist, listofkind,kind)
    testset = graphdataset(test_g, test_l, num_classes)
    save_graphs("Graphs/app/trainset.bin", trainset.graphs,
                {'labels': torch.tensor(trainset.labels)})
    save_graphs("Graphs/app/validset.bin", validset.graphs,
                {'labels': torch.tensor(validset.labels)})
    save_graphs("Graphs/app/testset.bin", testset.graphs,
                {'labels': torch.tensor(testset.labels)})


# def make_entropy_graph(data_dir_path, valid_data_dir_path, testdata_dir_path):
#     testdata_dir_path = Path(testdata_dir_path)
#     valid_data_dir_path = Path(valid_data_dir_path)
#     data_dir_path = Path(data_dir_path)
#     listofkind = []
#     listofnodenum = []
#     feature_matrix = []
#     listofkind_valid = []
#     listofnodenum_valid = []
#     feature_matrix_valid = []
#     listofkind_test = []
#     listofnodenum_test = []
#     feature_matrix_test = []
#     num_classes = len(PREFIX_TO_ENTROPY_ID)
#     for pcap_path in sorted(data_dir_path.iterdir()):
#         transform_pcap_entropy(pcap_path, listofnodenum, listofkind, feature_matrix)
#     for pcap_path in sorted(valid_data_dir_path.iterdir()):
#         transform_pcap_entropy(pcap_path, listofnodenum_valid, listofkind_valid, feature_matrix_valid)
#     for pcap_path in sorted(testdata_dir_path.iterdir()):
#         transform_pcap_entropy(pcap_path, listofnodenum_test, listofkind_test, feature_matrix_test)
#     trainset = graphdataset(listofnodenum, listofkind, feature_matrix, num_classes)
#     validset = graphdataset(listofnodenum_valid, listofkind_valid, feature_matrix_valid, num_classes)
#     testset = graphdataset(listofnodenum_test, listofkind_test, feature_matrix_test, num_classes)
#     # graph_labels = {"graph_sizes": torch.tensor()}
#     save_graphs("Graphs/Entropy/trainset.bin", trainset.graphs,
#                 {'labels': torch.tensor(trainset.labels)})
#     save_graphs("Graphs/Entropy/validset.bin", validset.graphs,
#                 {'labels': torch.tensor(validset.labels)})
#     save_graphs("Graphs/Entropy/testset.bin", testset.graphs,
#                 {'labels': torch.tensor(testset.labels)})
#
#
# def make_malware_graph(data_dir_path, valid_data_dir_path, testdata_dir_path):
#     testdata_dir_path = Path(testdata_dir_path)
#     valid_data_dir_path = Path(valid_data_dir_path)
#     data_dir_path = Path(data_dir_path)
#     listofkind = []
#     listofnodenum = []
#     feature_matrix = []
#     listofkind_valid = []
#     listofnodenum_valid = []
#     feature_matrix_valid = []
#     listofkind_test = []
#     listofnodenum_test = []
#     feature_matrix_test = []
#     num_classes = len(PREFIX_TO_Malware_ID)
#     for pcap_path in sorted(data_dir_path.iterdir()):
#         transform_pcap_malware(pcap_path, listofnodenum, listofkind, feature_matrix)
#     for pcap_path in sorted(valid_data_dir_path.iterdir()):
#         transform_pcap_malware(pcap_path, listofnodenum_valid, listofkind_valid, feature_matrix_valid)
#     for pcap_path in sorted(testdata_dir_path.iterdir()):
#         transform_pcap_malware(pcap_path, listofnodenum_test, listofkind_test, feature_matrix_test)
#     print(len(listofnodenum))
#     print(len(feature_matrix))
#     trainset = graphdataset(listofnodenum, listofkind, feature_matrix, num_classes)
#     validset = graphdataset(listofnodenum_valid, listofkind_valid, feature_matrix_valid, num_classes)
#     testset = graphdataset(listofnodenum_test, listofkind_test, feature_matrix_test, num_classes)
#     # graph_labels = {"graph_sizes": torch.tensor()}
#     save_graphs("Graphs/Malware/trainset.bin", trainset.graphs,
#                 {'labels': torch.tensor(trainset.labels)})
#     save_graphs("Graphs/Malware/validset.bin", validset.graphs,
#                 {'labels': torch.tensor(validset.labels)})
#     save_graphs("Graphs/Malware/testset.bin", testset.graphs,
#                 {'labels': torch.tensor(testset.labels)})


@click.command()
@click.option('-k', '--kind', help='object to be classificated', required=True)
def main(kind):
    if kind == 'app':
        make_graph("Dataset/Processed_data/app/train", "Dataset/Processed_data/app/valid", "Dataset/Processed_data/app/test", kind)
    elif kind == 'malware':
        make_graph("Dataset/Processed_data/malware/malware/train", "Dataset/Processed_data/malware/malware/valid",
                   "Dataset/Processed_data/malware/malware/test", 'malware')
    else:
        make_graph("Dataset/Splite_Session/Entropy/train", "Dataset/Splite_Session/Entropy/valid",
                   "Dataset/Splite_Session/Entropy/test", 'entropy')


if __name__ == '__main__':
    main()
