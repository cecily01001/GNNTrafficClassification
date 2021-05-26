import click
import dgl
import numpy
import torch
from pathlib import Path
import numpy as np
from scapy.compat import raw
from scapy.layers.inet import IP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scipy import sparse
from utils.utilsformalware import should_omit_packet, read_pcap, PREFIX_TO_Malware_ID
from utils.utilsforapps import PREFIX_TO_APP_ID
from dgl.data.utils import save_graphs
import os


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
    if should_omit_packet(packet):
        return None

    packet = remove_ether_header(packet)
    packet = pad_udp(packet)
    packet = mask_ip(packet)

    arr = packet_to_sparse_array(packet)

    return arr


def transform_pcap_malware(path, listofnodenum_test, listofkind_test, feature_matrix_test):
    # output_path = 'processdata/' + path.name
    # if Path(output_path + '_SUCCESS').exists():
    #     return
    # with Path(output_path + '_SUCCESS').open('w') as f:
    #     f.write('')
    # print(path,"Done")
    # print(path)
    # try:
    #     read_pcap(path)
    # except Scapy_Exception:
    #     os.remove(path)
    #     return
    rows = []
    j = 0
    for i, packet in enumerate(read_pcap(path)):
        arr = transform_packet(packet)
        if arr is not None:
            # get labels for app identification
            prefix = path.name.split('.')[0]
            app_label = PREFIX_TO_Malware_ID.get(prefix)
            j = j + 1
            rows.append(arr.todense().tolist()[0])

    listofnodenum_test.append(j)
    listofkind_test.append(app_label)
    feature_matrix_test.append(rows)
    print(path, 'Done')

def transform_pcap(path, listofnodenum, listofkind, feature_matrix):
    # try:
    #     read_pcap(path)
    # except Scapy_Exception:
    #     os.remove(path)
    #     print(path)
    #     return
    # else:
    #     with Path(output_path + '_SUCCESS').open('w') as f:
    #         f.write('')
    rows = []
    j = 0
    for i, packet in enumerate(read_pcap(path)):
        arr = transform_packet(packet)
        if arr is not None:
            # get labels for app identification
            prefix = path.name.split('.')[4]
            app_label = PREFIX_TO_APP_ID.get(prefix)
            j = j + 1
            rows.append(arr.todense().tolist()[0])
    rows=np.array(rows)
    mean_row=rows.mean(axis=0)  # 列
    # mean_row = np.array(mean_row)
    listofnodenum.append(j)
    listofkind.append(app_label)
    feature_matrix.append(mean_row)
    print(path, 'Done')


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
            g = dgl.graph((list_point, list_point_2))
            g = dgl.add_self_loop(g)
            # print(i)
            g.ndata['h'] = torch.tensor(feature_matrix[k])
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


def make_app_graph(data_dir_path, valid_data_dir_path, testdata_dir_path):
    testdata_dir_path = Path(testdata_dir_path)
    valid_data_dir_path = Path(valid_data_dir_path)
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
    num_classes=len(PREFIX_TO_APP_ID)
    for pcap_path in sorted(data_dir_path.iterdir()):
        transform_pcap(pcap_path, listofnodenum, listofkind, feature_matrix)
    result=[0]*41
    num=[0]*41
    length=len(listofnodenum)
    dif_num=0
    dif_sum=0
    dif_num0 = 0
    dif_sum0 = 0
    for i in range(length):
        label1=listofkind[i]
        num[label1]=num[label1]+1
        vec1 = feature_matrix[i]
        for j in range(length):
            label2=listofkind[j]
            vec2=feature_matrix[j]
            dist = numpy.linalg.norm(vec1 - vec2)
            if label1==label2:
                if dist!=0:
                    result[label1]=result[label1]+dist
                    dif_num0 = dif_num0 + 1
                    dif_sum0 = dif_sum0 + dist

            else:
                if dist != 0:
                    dif_num = dif_num + 1
                    dif_sum = dif_sum + dist
    print(result)
    print(num)
    for i in range(41):
        result[i]=result[i]/num[i]
    print(result)
    print(dif_sum0)
    print(dif_num0)
    print(dif_sum0 / dif_num0)
    print(dif_sum)
    print(dif_num)
    print(dif_sum/dif_num)

    # for pcap_path in sorted(valid_data_dir_path.iterdir()):
    #     transform_pcap(pcap_path, listofnodenum_valid, listofkind_valid, feature_matrix_valid)
    # for pcap_path in sorted(testdata_dir_path.iterdir()):
    #     transform_pcap(pcap_path, listofnodenum_test, listofkind_test, feature_matrix_test)

    # trainset = graphdataset(listofnodenum, listofkind, feature_matrix,num_classes)
    # validset = graphdataset(listofnodenum_valid, listofkind_valid, feature_matrix_valid,num_classes)
    # testset = graphdataset(listofnodenum_test, listofkind_test, feature_matrix_test,num_classes)
    # graph_labels = {"graph_sizes": torch.tensor()}
    # save_graphs("Graphs/APP/trainset.bin", trainset.graphs,
    #             {'labels': torch.tensor(trainset.labels)})
    # save_graphs("Graphs/APP/validset.bin", validset.graphs,
    #             {'labels': torch.tensor(validset.labels)})
    # save_graphs("Graphs/APP/testset.bin", testset.graphs,
    #             {'labels': torch.tensor(testset.labels)})

def make_malware_graph(data_dir_path, valid_data_dir_path, testdata_dir_path):
    testdata_dir_path = Path(testdata_dir_path)
    valid_data_dir_path = Path(valid_data_dir_path)
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
    num_classes = len(PREFIX_TO_Malware_ID)
    for pcap_path in sorted(data_dir_path.iterdir()):
        transform_pcap_malware(pcap_path, listofnodenum, listofkind, feature_matrix)
    for pcap_path in sorted(valid_data_dir_path.iterdir()):
        transform_pcap_malware(pcap_path, listofnodenum_valid, listofkind_valid, feature_matrix_valid)
    for pcap_path in sorted(testdata_dir_path.iterdir()):
        transform_pcap_malware(pcap_path, listofnodenum_test, listofkind_test, feature_matrix_test)
    trainset = graphdataset(listofnodenum, listofkind, feature_matrix,num_classes)
    validset = graphdataset(listofnodenum_valid, listofkind_valid, feature_matrix_valid,num_classes)
    testset = graphdataset(listofnodenum_test, listofkind_test, feature_matrix_test,num_classes)
    # graph_labels = {"graph_sizes": torch.tensor()}
    save_graphs("Graphs/Malware/trainset.bin", trainset.graphs,
                {'labels': torch.tensor(trainset.labels)})
    save_graphs("Graphs/Malware/validset.bin", validset.graphs,
                {'labels': torch.tensor(validset.labels)})
    save_graphs("Graphs/Malware/testset.bin", testset.graphs,
                {'labels': torch.tensor(testset.labels)})


@click.command()
@click.option('-k', '--kind', help='object to be classificated', required=True)
def main(kind):
    if kind == 'app':
        make_app_graph("Dataset/APP/trainset", "Dataset/APP/validset", "Dataset/APP/testset")
    elif kind == 'malware':
        make_malware_graph("Dataset/Malware/trainset", "Dataset/Malware/validset", "Dataset/Malware/testset")


if __name__ == '__main__':
    main()
