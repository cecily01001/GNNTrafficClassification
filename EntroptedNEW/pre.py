import csv

import click
import dgl
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
from utilsforentropy import PREFIX_TO_ENTROPY_ID,ID_TO_ENTROPY
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
    # print(arr)
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

def transform_pcap_entropy(path, listofnodenum, listofkind, feature_matrix):
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
    prefix = path.name.split('.')[0]
    app_label = PREFIX_TO_ENTROPY_ID.get(prefix)
    j = 0
    for i, pcap in enumerate(result):
        if i!=0:
            length=len(pcap)
            arr = list(map(eval,(pcap[1:length - 1])))
            # print(arr)
            if arr is not None:
                j = int(arr[1]+arr[2])
            print(j)
            if j > 1:
                listofnodenum.append(j)
                listofkind.append(app_label)
                feature_matrix.append(arr)
                # print(i)
                print("done")



class graphdataset(object):

    def __init__(self, num_of_graphnode, listofkind, feature_matrix,num_classes):
        super(graphdataset, self).__init__()
        graphlist = []
        k = 0
        for i in num_of_graphnode:
            list_point = []
            list_point_2 = []
            for j in range(i - 1):
                list_point.append(j)
                list_point_2.append(j + 1)
            for j in range(i - 1):
                list_point.append(j + 1)
                list_point_2.append(j)
            leng=int(i)
            # print(leng)
            # print('len:'+str(leng))
            fix_list_point=list_point[0:leng-1]
            fix_list_point2 = list_point_2[0:leng-1]
            g = dgl.graph((list_point, list_point_2))
            g = dgl.add_self_loop(g)
            # print(i)
            # print(feature_matrix[k])
            fix_feature=np.ones((i,67))
            for i in range(leng):
                fix_feature[i]=feature_matrix[k]
                # print(len(fix_feature[i]))
            # fix_feature=feature_matrix[k]
            # fix_feature=fix_feature[0:leng]
            # print(fix_feature)
            g.ndata['h'] = torch.tensor(fix_feature)
            # print(g)
            k = k + 1
            graphlist.append(g)
        # print(list_point)

        self.graphs = graphlist
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



def make_entropy_graph(data_dir_path):
    data_dir_path = Path(data_dir_path)
    listofkind = []
    listofnodenum = []
    feature_matrix = []
    listofkind_valid = []
    listofnodenum_valid = []
    feature_matrix_valid = []
    listofkind_test = []
    listofnodenum_test = []
    feature_matrix_test = []
    num_classes = len(PREFIX_TO_ENTROPY_ID)

    for pcap_path in sorted(data_dir_path.iterdir()):
        transform_pcap_entropy(pcap_path, listofnodenum, listofkind, feature_matrix)
    for i in range(len(listofkind)):
        if i%10==0:
            listofkind_valid.append(listofkind[i])
            listofnodenum_valid.append(listofnodenum[i])
            feature_matrix_valid.append(feature_matrix[i])
        else:
            listofkind_test.append(listofkind[i])
            listofnodenum_test.append(listofnodenum[i])
            feature_matrix_test.append(feature_matrix[i])
    trainset = graphdataset(listofnodenum_test, listofkind_test, feature_matrix_test, num_classes)
    validset = graphdataset(listofnodenum_valid, listofkind_valid, feature_matrix_valid, num_classes)
    # testset = graphdataset(listofnodenum_test, listofkind_test, feature_matrix_test, num_classes)
    # graph_labels = {"graph_sizes": torch.tensor()}
    save_graphs("Graphs/Entropy/trainset.bin", trainset.graphs,
                {'labels': torch.tensor(trainset.labels)})
    save_graphs("Graphs/Entropy/validset.bin", validset.graphs,
                {'labels': torch.tensor(validset.labels)})
    # save_graphs("Graphs/Entropy/testset.bin", testset.graphs,
    #             {'labels': torch.tensor(testset.labels)})

def main():
    make_entropy_graph("/home/pcl/PangBo/pro/GNNTrafficClassification/Dataset/newentropy")
if __name__ == '__main__':
    main()
