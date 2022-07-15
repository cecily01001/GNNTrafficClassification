import dgl
from numpy import *
import rule as rule
import torch as th
from dgl.nn import SGConv, AvgPooling
from scapy.layers.inet import IP
from torch.nn import functional as F, Linear
from scapy.utils import rdpcap
from scapy.all import *
# g = dgl.graph(([0,1,2,3,4,5,4,3,2,1], [1,2,3,4,5,4,3,2,1,0]))
# g = dgl.add_self_loop(g)
# # feat = th.randint(0,255,(6,6))
# feat=[[199,231,230,6,47,150],[84,135,235,13,205,173],[201,122,31,73,69,55],[1, 163,73, 180, 158, 155],[ 65, 199, 104,  75, 204, 158],[  4,  76, 139, 122, 155, 232]]
# feat=th.tensor(feat)
# print(feat)
# sgc1 = SGConv(6, 5)
# sgc2 = SGConv(5, 4)
# pooling = AvgPooling()
# classify = Linear(4, 2)
# res = F.relu(sgc1(g, feat))
# print(res)
# res=F.relu(sgc2(g,res))
# print(res)
# hg = pooling(g, res)
# print(hg)
# y = classify(hg)
# print(y)
# y= th.softmax(y, 1)
# print(y)
path="hangouts_audio4.pcap"
packets = rdpcap(str(path))
len=[]
for pkt in packets:
    if IP in pkt:
        len.append(int(pkt.sprintf("%IP.len%")))
        print(pkt.sprintf("%IP.len%"))
print(mean(len))