import csv
from pathlib import Path
import numpy as np
from scapy.all import *
from scapy.layers.inet import IP, UDP,TCP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scipy import sparse
from utilsforapps import PREFIX_TO_APP_ID


f_train = open(r'sampletrain.csv', 'w', encoding='utf-8', newline="")
train_csv = csv.writer(f_train)
train_csv.writerow(['label', 'text'])

f_valid = open(r'samplevalid.csv', 'w', encoding='utf-8', newline="")
valid_csv = csv.writer(f_valid)
valid_csv.writerow(['label', 'text'])

f_test = open(r'sampletest.csv', 'w', encoding='utf-8', newline="")
test_csv = csv.writer(f_test)
test_csv.writerow(['label', 'text'])

def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet
def mask_ip(packet):
    if IP in packet:
        packet[IP].src = '0.0.0.0'
        packet[IP].dst = '0.0.0.0'

    return packet
def read_pcap(path):
    print(path)
    f = open(str(path),"r",encoding="utf-8")
    try:
        l = f.read()
        l.replace("\t", "")
        packets = l.split(' Encapsulation type:')
        packets=packets[1:]
        for i in range(len(packets)):
            packets[i]=packets[i][0:10000]
    except UnicodeDecodeError:
        packets=None
    return packets
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

def packet_to_sparse_array(packet,max_length=1500):
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0:max_length]
    leng=len(arr)
    if leng < max_length:
        pad_width = max_length - leng
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
    # print(arr)
    arr = sparse.csr_matrix(arr)
    return arr

def transform_packet(packet):
    packet = remove_ether_header(packet)
    packet = pad_udp(packet)
    packet = mask_ip(packet)
    arr = packet_to_sparse_array(packet)
    return arr

def transform_pcap(path,kind):

    packets=read_pcap(path)
    if(packets):
        if kind=='train':
            j=0
            for i, packet in enumerate(packets):
                j = j + 1
                if j%10==0:
                    prefix = path.name.split('.')[0]
                    app_label = PREFIX_TO_APP_ID.get(prefix)
                    train_csv.writerow([app_label, packet])
        elif kind=='valid':
            for i, packet in enumerate(packets):
                prefix = path.name.split('.')[0]
                app_label = PREFIX_TO_APP_ID.get(prefix)
                valid_csv.writerow([app_label, packet])
        else:
            for i, packet in enumerate(packets):
                prefix = path.name.split('.')[0]
                app_label = PREFIX_TO_APP_ID.get(prefix)
                test_csv.writerow([app_label, packet])

def make_app_graph(data_dir_path, valid_data_dir_path, testdata_dir_path):
    testdata_dir_path = Path(testdata_dir_path)
    valid_data_dir_path = Path(valid_data_dir_path)
    data_dir_path = Path(data_dir_path)
    for pcap_path in sorted(data_dir_path.iterdir()):
        transform_pcap(pcap_path,'train')
    for pcap_path in sorted(testdata_dir_path.iterdir()):
        transform_pcap(pcap_path,'valid')
    for pcap_path in sorted(valid_data_dir_path.iterdir()):
        transform_pcap(pcap_path,'test')

make_app_graph("../Dataset/Text/app/Train", "../Dataset/Text/app/valid", "../Dataset/Text/app/Test")
# make_app_graph("C:/Users/cecil/Desktop/pcapfiles", "C:/Users/cecil/Desktop/pcapfiles", "C:/Users/cecil/Desktop/pcapfiles")