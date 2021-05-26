import time
import dgl
import click
import torch
from numpy.ma import mean
from torch import nn as nn
from torch.nn import functional as F
from utils.utilsformalware import PREFIX_TO_Malware_ID
from utils.utilsforapps import PREFIX_TO_APP_ID
from torch.utils.data import DataLoader
from dgl.nn import SGConv, GATConv, TAGConv, GlobalAttentionPooling, AvgPooling
from dgl.data.utils import load_graphs


class graphdataset(object):
    def __init__(self, graphs, labels, num_classes):
        super(graphdataset, self).__init__()
        self.graphs = graphs
        self.num_classes = num_classes
        self.labels = labels

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


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        # self.conv1 = SGConv(in_dim, 256)
        # self.conv2 = SGConv(hidden_dim, 256)
        self.conv1 = TAGConv(in_dim, hidden_dim)
        self.conv2 = TAGConv(hidden_dim, 256)
        # self.gat = GATConv(516, 200, num_heads=1)
        # gate_nn = nn.Linear(516, 1)
        self.pooling = AvgPooling()
        self.classify = nn.Linear(256, n_classes)

    def forward(self, g):
        # 使用节点的入度作为初始特征
        h = g.ndata['h'].float()
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        # h=self.fc1(h)
        # h=self.fc2(h)
        g.ndata['h'] = h  ## 节点特征经过两层卷积的输出
        # hg = dgl.mean_nodes(g, 'h')  # 图的特征是所有节点特征的均值
        hg = self.pooling(g, h)
        y = self.classify(hg)
        # y=self.pooling
        return y


test_graphs, test_labels = load_graphs("Graphs/Malware/validset.bin")
print("load Mtest graphs")
testset = graphdataset(test_graphs, test_labels['labels'].numpy().tolist(), len(PREFIX_TO_Malware_ID))
print("testset done")
model = torch.load('mmodel.pkl')
test_X, test_Y = map(list, zip(*testset))
test_bg = dgl.batch(test_X)
test_Y = torch.tensor(test_Y).float().view(-1, 1)
dif_list = []
probs_Y = torch.softmax(model(test_bg), 1)
# for i in range(100):
#     begin = time.clock()
#     probs_Y = torch.softmax(model(test_bg), 1)
#     end = time.clock()
#     dif = end - begin
#     dif_list.append(dif)
#     print(dif)
#     print('推理时长：{:.4f}'.format(dif))
#     mean_dif = mean(dif_list)
#     print('平均推理时长：{:.4f}'.format(mean_dif))
probs_Y = torch.softmax(model(test_bg), 1)
sampled_Y = torch.multinomial(probs_Y, 1)
argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
sampled_Y_done = sampled_Y.float()
argmax_Y_done = argmax_Y.float()
print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
    (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
    (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))
sum = [0] * 6
index = [0] * 6
pred_sum = [0] * 6
for i in range(len(test_Y)):
    sum[int(test_Y[i, 0])] = sum[int(test_Y[i, 0])] + 1
    pred_sum[int(argmax_Y_done[i, 0])] = pred_sum[int(argmax_Y_done[i, 0])] + 1
    if test_Y[i] == argmax_Y_done.float()[i]:
        index[int(test_Y[i, 0])] = index[int(test_Y[i, 0])] + 1
    print(sum)
    print(pred_sum)
    print(index)

# feature_matrix.clear()
j = 0
print("Accuracy of argmax predictions")
for i in range(len(sum)):
    if sum[j] is not 0 and pred_sum[j] is not 0:
        # print(str(i) + ' kind recall is: ' + str(index[j] / sum[j]))
        print(str(i) + ' kind precision is: ' + str(index[j] / pred_sum[j]))
    else:
        print(str(i) + ' kind precision is: -1')
    j = j + 1

sum = [0] * 6
index = [0] *6
pred_sum = [0] * 6
print("Accuracy of sampled predictions")
for i in range(len(test_Y)):
    sum[int(test_Y[i, 0])] = sum[int(test_Y[i, 0])] + 1
    pred_sum[int(sampled_Y_done[i, 0])] = pred_sum[int(sampled_Y_done[i, 0])] + 1
    if test_Y[i] == sampled_Y_done.float()[i]:
        index[int(test_Y[i, 0])] = index[int(test_Y[i, 0])] + 1
    print(sum)
    print(pred_sum)
    print(index)

# feature_matrix.clear()
j = 0
for i in range(len(sum)):
    if sum[j] is not 0 and pred_sum[j] is not 0:
        # print(str(i) + ' kind recall is: ' + str(index[j] / sum[j]))
        print(str(i) + ' kind precision is: ' + str(index[j] / pred_sum[j]))
    else:
        print(str(i) + ' kind precision is: -1')
    j = j + 1
