import csv
from pathlib import Path

from numpy import *

from scapy.layers.inet import IP

from scapy.all import *
import os
# f = open("Mtrain_pcap.csv","a")
pcaplengthfile = open('Apcaplengthfile.csv', 'w', encoding='utf-8', newline="")
paths = ['/home/pcl/PangBo/pro/GNNTrafficClassification/Dataset/Splite_Session/app/Train', '/home/pcl/PangBo/pro/GNNTrafficClassification/Dataset/Splite_Session/Entropy/train']
data_dir_path = paths[0]
output_dir_path=paths[1]
data_dir_path = Path(data_dir_path)
pcaplengthfile_writer = csv.writer(pcaplengthfile)
pcaplengthfile_writer.writerow(["len"])

for pcap_file in sorted(data_dir_path.iterdir()):
    list1 = pcap_file.name.split('/')
    for i in list1:
        pcapname=i
    packets = rdpcap(str(pcap_file))
    # len = []
    # len.append(pcapname)
    for pkt in packets:
        if IP in pkt:
            # len.append(pkt.sprintf("%IP.len%"))
            pcaplengthfile_writer.writerow([pkt.sprintf("%IP.len%")])

    # line=' '.join(len)
    # f.write(line+'\n')
# f.close()
# packets = rdpcap(str(path))
# len=[]
# for pkt in packets:
#     if IP in pkt:
#         len.append(int(pkt.sprintf("%IP.len%")))
#         print(pkt.sprintf("%IP.len%"))
