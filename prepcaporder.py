import click
import dgl
import jieba
import torch
from pathlib import Path
import numpy as np
from scapy.compat import raw
from scapy.all import *
from scapy.error import Scapy_Exception
from scapy.layers.inet import IP, UDP, TCP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scipy import sparse
from utils.utilsformalware import should_omit_packet, read_pcap, PREFIX_TO_Malware_ID
from utils.utilsforapps import PREFIX_TO_APP_ID
from utils.utilsforentropy import PREFIX_TO_ENTROPY_ID, ID_TO_ENTROPY
from dgl.data.utils import save_graphs

ack = 2
seq = 2
pcaplen = 0
port1 = 0
port2 = 0
turn = True
next_num = pcaplen + seq
last_syn = 'S'
time_last = 0


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
    print([word for word in jieba.cut(raw(packet)) if word.strip()])
    # arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length] / 255
    leng = len(arr)
    if leng < max_length:
        pad_width = max_length - leng
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
    # print(arr)
    arr = sparse.csr_matrix(arr)
    return arr


def transform_packet(packet, port1):
    global flag, next_num, turn, last_syn, time_last
    flag = True
    edge_f = []
    e_flag_1 = 00
    gap = 0
    time_now = packet.time
    if (time_now != 0):
        gap = float(time_now - time_last)
    if TCP in packet:
        seq_temp = packet[TCP].seq
        ack_temp = packet[TCP].ack
        sport = packet[TCP].sport
        syn_temp = packet.sprintf('%TCP.flags%')
        if (last_syn == 'S'):
            e_flag_1 = 10
        elif (last_syn == 'SA'):
            if sport == port1:
                turn = True
            else:
                turn = False
            next_num = seq_temp
            e_flag_1 = 11
        elif last_syn != 'R':
            if sport == int(port1):
                e_flag_1 = 12
            else:
                e_flag_1 = 13
        else:
            e_flag_1 = 14

        if syn_temp != 'SA' and syn_temp != 'S' and syn_temp != 'R':
            pcaplen = len(packet[TCP].payload)
            if sport == int(port1):
                if turn:  # 与之前的报文方向相同
                    if seq_temp == next_num:
                        flag = True
                    else:
                        flag = False
                else:  # 与之前的报文方向相反
                    if ack_temp == next_num:
                        flag = True
                    else:
                        flag = False
                turn = True
            else:
                if not turn:  # 与之前的报文方向相同
                    if seq_temp == next_num:
                        flag = True
                    else:
                        flag = False
                else:  # 与之前的报文方向相反
                    if ack_temp == next_num:
                        flag = True
                    else:
                        flag = False
                turn = False
            next_num = seq_temp + pcaplen
        last_syn = syn_temp
        time_last = time_now
        if flag == False:
            print('not in order')

    packet = remove_ether_header(packet)
    packet = pad_udp(packet)
    packet = mask_ip(packet)
    arr = packet_to_sparse_array(packet)

    edge_f.append(e_flag_1)
    edge_f.append(gap)
    if gap < 0:
        flag = False
    return arr, edge_f


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
    if j > 1:
        listofnodenum_test.append(j)
        listofkind_test.append(app_label)
        feature_matrix_test.append(rows)
        print(path, 'Done')


def transform_pcap_entropy(path, listofnodenum, listofkind, feature_matrix):
    rows = []
    j = 0
    IPPORT = path.name.split('.')[2]
    port1 = IPPORT.split('_')[2]
    for i, packet in enumerate(read_pcap(path)):
        arr = transform_packet(packet, port1)
        if flag == False:
            print(path)
        if arr is not None:
            prefix = path.name.split('.')[0]
            app_label = PREFIX_TO_ENTROPY_ID.get(prefix)
            j = j + 1
            rows.append(arr.todense().tolist()[0])
    if j > 1:
        # if app_label:
        listofnodenum.append(j)
        listofkind.append(app_label)
        feature_matrix.append(rows)
        # print(path, 'Done')
        # else:
        #     print(prefix)
        #     print(app_label)
        #     listofnodenum.append(j)
        #     listofkind.append(app_label)
        #     feature_matrix.append(rows)
        # print(path, 'Done')


def transform_pcap(path, listofnodenum, listofkind, n_feature_matrix, e_feature_matrix):
    # try:
    #     read_pcap(path)
    # except Scapy_Exception:
    #     os.remove(path)
    #     print(path)
    #     return
    # else:
    #     with Path(output_path + '_SUCCESS').open('w') as f:
    #         f.write('')
    global time_last
    rows = []
    j = 0
    IPPORT = path.name.split('.')[2]
    port1 = IPPORT.split('_')[2]
    edge_rows = []
    looped = [0, 0]
    time_last = 0
    for i, packet in enumerate(read_pcap(path)):
        arr, e_feature = transform_packet(packet, port1)
        if flag == False:
            print(path)
            print(e_feature)
            print(i)
        if arr is not None:
            # get labels for app identification
            # prefix = path.name.split('.')[4]
            prefix = path.name.split('.')[0]
            # if prefix=='yahoo_messenger':
            #     prefix='yahoo!_messenger'
            # else:
            #     prefix=prefix[0:len(prefix)-1]
            # print(prefix)
            app_label = PREFIX_TO_APP_ID.get(prefix)
            j = j + 1
            rows.append(arr.todense().tolist()[0])
            edge_rows.append(e_feature)
    if j >= 1:
        listofnodenum.append(j)
        listofkind.append(app_label)
        n_feature_matrix.append(rows)
        edge_rows = edge_rows[1:len(edge_rows)]
        leng = len(edge_rows)
        for i in range(leng):
            edge_rows.append(edge_rows[i])
        for i in range(j):
            edge_rows.append(looped)
        e_feature_matrix.append(edge_rows)
        # print(path, 'Done')
        # print(app_label)
        # print(prefix)
        # print(app_label)
        # listofnodenum.append(j)
        # listofkind.append(app_label)
        # feature_matrix.append(rows)
        # print(path, 'Done')


class graphdataset(object):

    def __init__(self, num_of_graphnode, listofkind, n_feature_matrix, e_feature_matrix, num_classes):
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
            print(g.edges())
            g.ndata['h'] = torch.tensor(n_feature_matrix[k])
            g.edata['h'] = torch.tensor(e_feature_matrix[k])
            print(g.edata['h'])
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


def make_graph(data_dir_path, valid_data_dir_path, testdata_dir_path):
    testdata_dir_path = Path(testdata_dir_path)
    valid_data_dir_path = Path(valid_data_dir_path)
    data_dir_path = Path(data_dir_path)
    listofkind = []
    listofnodenum = []
    n_feature_matrix = []
    e_feature_matrix = []
    listofkind_valid = []
    listofnodenum_valid = []
    n_feature_matrix_valid = []
    e_feature_matrix_valid = []
    listofkind_test = []
    listofnodenum_test = []
    n_feature_matrix_test = []
    e_feature_matrix_test = []
    num_classes = len(PREFIX_TO_APP_ID)
    for pcap_path in sorted(data_dir_path.iterdir()):
        transform_pcap(pcap_path, listofnodenum, listofkind, n_feature_matrix, e_feature_matrix)
    for pcap_path in sorted(valid_data_dir_path.iterdir()):
        transform_pcap(pcap_path, listofnodenum_valid, listofkind_valid, n_feature_matrix_valid, e_feature_matrix_valid)
    for pcap_path in sorted(testdata_dir_path.iterdir()):
        transform_pcap(pcap_path, listofnodenum_test, listofkind_test, n_feature_matrix_test, e_feature_matrix_test)

    trainset = graphdataset(listofnodenum, listofkind, n_feature_matrix, e_feature_matrix, num_classes)
    validset = graphdataset(listofnodenum_valid, listofkind_valid, n_feature_matrix_valid, e_feature_matrix_valid,
                            num_classes)
    testset = graphdataset(listofnodenum_test, listofkind_test, n_feature_matrix_test, e_feature_matrix_test,
                           num_classes)
    save_graphs("Graphs/APP/trainset.bin", trainset.graphs,
                {'labels': torch.tensor(trainset.labels)})
    save_graphs("Graphs/APP/validset.bin", validset.graphs,
                {'labels': torch.tensor(validset.labels)})
    save_graphs("Graphs/APP/testset.bin", testset.graphs,
                {'labels': torch.tensor(testset.labels)})


def make_entropy_graph(data_dir_path, valid_data_dir_path, testdata_dir_path):
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
    num_classes = len(PREFIX_TO_ENTROPY_ID)
    for pcap_path in sorted(data_dir_path.iterdir()):
        transform_pcap_entropy(pcap_path, listofnodenum, listofkind, feature_matrix)
    for pcap_path in sorted(valid_data_dir_path.iterdir()):
        transform_pcap_entropy(pcap_path, listofnodenum_valid, listofkind_valid, feature_matrix_valid)
    for pcap_path in sorted(testdata_dir_path.iterdir()):
        transform_pcap_entropy(pcap_path, listofnodenum_test, listofkind_test, feature_matrix_test)
    trainset = graphdataset(listofnodenum, listofkind, feature_matrix, num_classes)
    validset = graphdataset(listofnodenum_valid, listofkind_valid, feature_matrix_valid, num_classes)
    testset = graphdataset(listofnodenum_test, listofkind_test, feature_matrix_test, num_classes)
    # graph_labels = {"graph_sizes": torch.tensor()}
    save_graphs("Graphs/Entropy/trainset.bin", trainset.graphs,
                {'labels': torch.tensor(trainset.labels)})
    save_graphs("Graphs/Entropy/validset.bin", validset.graphs,
                {'labels': torch.tensor(validset.labels)})
    save_graphs("Graphs/Entropy/testset.bin", testset.graphs,
                {'labels': torch.tensor(testset.labels)})


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
    print(len(listofnodenum))
    print(len(feature_matrix))
    trainset = graphdataset(listofnodenum, listofkind, feature_matrix, num_classes)
    validset = graphdataset(listofnodenum_valid, listofkind_valid, feature_matrix_valid, num_classes)
    testset = graphdataset(listofnodenum_test, listofkind_test, feature_matrix_test, num_classes)
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
        make_graph("Dataset/Processed_data/train", "Dataset/Processed_data/valid", "Dataset/Processed_data/test", 'app')
    elif kind == 'malware':
        make_graph("Dataset/Splite_Session/Malware/trainset", "Dataset/Splite_Session/Malware/validset",
                   "Dataset/Splite_Session/Malware/testset", 'malware')
    else:
        make_graph("Dataset/Splite_Session/Entropy/train", "Dataset/Splite_Session/Entropy/valid",
                   "Dataset/Splite_Session/Entropy/test", 'entropy')


if __name__ == '__main__':
    main()
# make_app_graph("C:/Users/cecil/Desktop/pcapfiles", "Dataset/Splite_Session/app/valid", "C:/Users/cecil/Desktop/pcapfiles")
