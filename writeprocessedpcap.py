from scapy.all import *
from pathlib import Path
from scapy.layers.inet import IP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
import os
import shutil
def read_pcap(path: Path):
    packets = rdpcap(str(path))

    return packets

def remove_ether_header(packet):
    if Ether in packet:
        packet[Ether].src = "00:00:00:00:00:00"
        packet[Ether].dst = "00:00:00:00:00:00"
        # return packet[Ether].payload

    return packet


def mask_ip(packet):
    if IP in packet:
        packet[IP].src = '0.0.0.0'
        packet[IP].dst = '0.0.0.0'
    if UDP in packet:
        packet[UDP].sport = 0000
        packet[UDP].dport = 0000
    packet.sport = 0000
    packet.dport = 0000
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


def transform_pcap(path,output_dir_path):
    prefix = path.name.split('/')
    for i in prefix:
        name=i
    for i, packet in enumerate(read_pcap(path)):
        # packet = remove_ether_header(packet)
        # packet = pad_udp(packet)
        packet = mask_ip(packet)
        wrpcap(output_dir_path+'/'+name, packet, append=True)
    shutil.copy(path, output_dir_path)
    print(path, 'Done')

paths = [['/home/pcl/PangBo/pro/GNNTrafficClassification/Splite/3_ProcessedSession/FilteredSession/Train', '/home/pcl/PangBo/pro/GNNTrafficClassification/Dataset/Splite_Session/Entropy/train']]
data_dir_path = paths[0][0]
output_dir_path=paths[0][1]
data_dir_path = Path(data_dir_path)
for pcap_path_dir in sorted(data_dir_path.iterdir()):
    prefix0 = pcap_path_dir.name.split('/')
    for i in prefix0:
        name = i
    # out_pcap_dir=output_dir_path+'/'+name
    # os.mkdir(out_pcap_dir)
    for pcap_path in sorted(pcap_path_dir.iterdir()):
        transform_pcap(pcap_path,output_dir_path)
