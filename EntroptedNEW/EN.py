import csv
import time
from pathlib import Path

import seaborn as sns
import matplotlib as mpl
import matplotlib as mpl
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
mpl.use('Agg')
import matplotlib.pyplot as plt
import dgl
import click
import torch
from torch import nn as nn
from numpy import *
from torch.nn import functional as F

from torch.utils.data import DataLoader
from dgl.nn import SGConv, GATConv, TAGConv, GlobalAttentionPooling, AvgPooling, GraphConv,MaxPooling,SumPooling
from dgl.data.utils import load_graphs
from utilsforentropy import PREFIX_TO_ENTROPY_ID,ID_TO_ENTROPY

def normalise_cm(cm):
    with errstate(all='ignore'):
        normalised_cm = cm / cm.sum(axis=1, keepdims=True)
        normalised_cm = nan_to_num(normalised_cm)
        return normalised_cm


def plot_confusion_matrix(cm, labels,num):
    normalised_cm = normalise_cm(cm)
    fig, ax = plt.subplots(figsize=(num, num))
    sns.heatmap(
        data=normalised_cm, cmap='YlGnBu',
        xticklabels=labels, yticklabels=labels,
        annot=True, ax=ax, fmt='.2f'
    )
    ax.set_xlabel('Predict labels')
    ax.set_ylabel('True labels')
    return fig


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


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    # print(batched_graph)
    # print(labels)
    return batched_graph, torch.tensor(labels)


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = SGConv(in_dim, hidden_dim)
        self.conv2 = SGConv(hidden_dim, 256)
        # self.conv3 = SGConv(256, 126)
        # self.conv1 = TAGConv(in_dim, hidden_dim)
        # self.conv2 = TAGConv(hidden_dim, 256)
        # self.conv1 = GraphConv(in_dim, hidden_dim)
        # self.conv2 = GraphConv(hidden_dim, 256)
        # self.gat = GATConv(hidden_dim, 256, num_heads=1)
        # gate_nn = nn.Linear(516, 1)
        # self.pooling = SumPooling()
        # self.pooling = MaxPooling()
        self.pooling = AvgPooling()
        self.classify = nn.Linear(256, n_classes)

    def forward(self, g):
        # 使用节点的入度作为初始特征
        h = g.ndata['h'].float()
        # print(h)
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        # h = F.relu(self.conv3(g, h))
        # h=self.fc1(h)
        # h=self.fc2(h)
        g.ndata['h'] = h  ## 节点特征经过两层卷积的输出
        # hg = dgl.mean_nodes(g, 'h')  # 图的特征是所有节点特征的均值
        hg = self.pooling(g, h)
        y = self.classify(hg)
        # y=self.pooling
        return y
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
            if arr is not None:
                j = int(arr[2]+arr[3])

            if j > 1:
                listofnodenum.append(j)
                listofkind.append(app_label)
                feature_matrix.append(arr)
                print(i)
                print("done")
def train_entropy_graphs(train,valid):
    train_graphs, train_labels = load_graphs(train)
    print("load train graphs")
    valid_graphs, valid_labels = load_graphs(valid)
    print("load valid graphs")
    test_graphs, test_labels = load_graphs(valid)
    print("load test graphs")
    trainset = graphdataset(train_graphs, train_labels['labels'].numpy().tolist(), len(PREFIX_TO_ENTROPY_ID))
    print("trainset done")
    validset = graphdataset(valid_graphs, valid_labels['labels'].numpy().tolist(), len(PREFIX_TO_ENTROPY_ID))
    print("validset done")
    testset = graphdataset(test_graphs, test_labels['labels'].numpy().tolist(), len(PREFIX_TO_ENTROPY_ID))
    print("testset done")
    data_loader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate)
    model = Classifier(67, 516, trainset.num_classes)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # model.train()
    min_loss = 1000
    max_acc=0
    # epoch_losses = []
    avoid_over = 0
    epoch_file = open('entropyresult/entropy_epoch_process.csv', 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(epoch_file)
    csv_writer.writerow(["num", "epoch"])
    time_file = open('entropyresult/train.csv', 'w', encoding='utf-8', newline="")
    time_writer = csv.writer(time_file)
    time_writer.writerow(["epoch", "time"])
    for epoch in range(400):
        since = time.time()
        model.train()
        epoch_loss = 0
        for iter, (bg, label) in enumerate(data_loader):
            prediction = model(bg)
            # print(prediction)
            # print(label)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        # epoch_losses.append(epoch_loss)
        csv_writer.writerow([epoch, epoch_loss])
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        if epoch_loss < min_loss and epoch_loss < 100 :
            avoid_over = 0
            model.eval()
            valid_x, valid_y = map(list, zip(*validset))
            print(valid_y)
            valid_bg = dgl.batch(valid_x)
            valid_y = torch.tensor(valid_y).float().view(-1, 1)
            valid_probs_Y = torch.softmax(model(valid_bg), 1)
            valid_sampled_Y = torch.multinomial(valid_probs_Y, 1)
            valid_argmax_Y = torch.max(valid_probs_Y, 1)[1].view(-1, 1)
            accuracy1 = (valid_y == valid_sampled_Y.float()).sum().item() / len(valid_y) * 100
            accuracy2 = (valid_y == valid_argmax_Y.float()).sum().item() / len(valid_y) * 100
            valid_sampled_Y_done = valid_sampled_Y.float()
            print(valid_argmax_Y)
            sum = [0] *11
            index = [0] * 11
            pred_sum = [0] * 11
            for i in range(len(valid_x)):
                sum[int(valid_y[i, 0])] = sum[int(valid_y[i, 0])] + 1
                pred_sum[int(valid_sampled_Y_done[i, 0])] = pred_sum[int(valid_sampled_Y_done[i, 0])] + 1
                if valid_y[i] == valid_sampled_Y.float()[i]:
                    index[int(valid_y[i, 0])] = index[int(valid_y[i, 0])] + 1
            print('Epoch {},Accuracy of sampled predictions on the valid set: {:.4f}%'.format(epoch, accuracy1))
            print('Epoch {},Accuracy of argmax predictions on the valid set: {:.4f}%'.format(epoch, accuracy2))
            if accuracy1 > 90 or accuracy2 > 90:
                max_acc_temp=[accuracy2,accuracy1][accuracy1>accuracy2]
                min_loss = epoch_loss
                torch.save(model, 'entropyresult/entropymodel.pkl')
                print("model saved")
                # avoid_over = avoid_over + 1
                # if avoid_over > 20:
                #     break
            # else
        elif epoch_loss <= 0.02:
            avoid_over = avoid_over + 1
            if avoid_over > 20:
                break
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        time_writer.writerow([epoch, time_elapsed])
    model = torch.load('entropyresult/entropymodel.pkl')
    test_X, test_Y = map(list, zip(*testset))
    test_bg = dgl.batch(test_X)
    test_Y = torch.tensor(test_Y).float().view(-1, 1)
    probs_Y = torch.softmax(model(test_bg), 1)
    sampled_Y = torch.multinomial(probs_Y, 1)
    argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
    app_test_accuracy1 = (test_Y == sampled_Y.float()).sum().item() / len(test_Y)
    app_test_accuracy2 = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)
    print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
        (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
    print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
        (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))
    accuracyfile = open('entropyresult/accuracyfile.csv', 'w', encoding='utf-8', newline="")
    accuracyfile_writer = csv.writer(accuracyfile)
    accuracyfile_writer.writerow(["num", "accuracy"])
    accuracyfile_writer.writerow(["sampled", app_test_accuracy1])
    accuracyfile_writer.writerow(["argmax", app_test_accuracy2])
    cm = zeros((trainset.num_classes, trainset.num_classes), dtype=float)
    if app_test_accuracy1 >= app_test_accuracy2:
        sampled_Y_done = sampled_Y.float()
        for i in range(len(test_Y)):
            cm[int(test_Y[i, 0]), int(sampled_Y_done[i, 0])] += 1
        sum = [0] * 11
        index = [0] * 11
        pred_sum = [0] * 11
        for i in range(len(test_Y)):
            sum[int(test_Y[i, 0])] = sum[int(test_Y[i, 0])] + 1
            pred_sum[int(sampled_Y_done[i, 0])] = pred_sum[int(sampled_Y_done[i, 0])] + 1
            if test_Y[i] == sampled_Y.float()[i]:
                index[int(test_Y[i, 0])] = index[int(test_Y[i, 0])] + 1
            # print(sum)
            # print(pred_sum)
            # print(index)
        j = 0
        for i in range(len(sum)):
            if sum[j] is not 0 and pred_sum[j] is not 0:
                recall = index[j] / sum[j]
                precision = index[j] / pred_sum[j]
                print(str(i) + ' kind recall is: ' + str(recall))
                print(str(i) + ' kind precision is: ' + str(precision))
            else:
                recall = 0
                precision = -1
            app_name = ID_TO_ENTROPY.get(i)
            csv_writer.writerow([app_name, recall, precision])
            j = j + 1
    else:
        argmax_Y_done = argmax_Y.float()
        for i in range(len(test_Y)):
            cm[int(test_Y[i, 0]), int(argmax_Y_done[i, 0])] += 1
        sum = [0] * 11
        index = [0] * 11
        pred_sum = [0] * 11
        for i in range(len(test_Y)):
            sum[int(test_Y[i, 0])] = sum[int(test_Y[i, 0])] + 1
            pred_sum[int(argmax_Y_done[i, 0])] = pred_sum[int(argmax_Y_done[i, 0])] + 1
            if test_Y[i] == argmax_Y_done.float()[i]:
                index[int(test_Y[i, 0])] = index[int(test_Y[i, 0])] + 1
            # print(sum)
            # print(pred_sum)
            # print(index)
        j = 0
        print("Accuracy of argmax predictions")
        f = open('entropyresult/recision_and_recall.csv', 'w', encoding='utf-8', newline="")
        csv_writer = csv.writer(f)
        csv_writer.writerow(["app", "recall", "precision"])
        for i in range(len(sum)):
            if sum[j] is not 0 and pred_sum[j] is not 0:
                recall = index[j] / sum[j]
                precision = index[j] / pred_sum[j]
                print(str(i) + ' kind recall is: ' + str(recall))
                print(str(i) + ' kind precision is: ' + str(precision))
            else:
                recall = 0
                precision = -1
            app_name = ID_TO_ENTROPY.get(i)
            csv_writer.writerow([app_name, recall, precision])
            j = j + 1
    app_labels = []
    for i in sorted(list(ID_TO_ENTROPY.keys())):
        app_labels.append(ID_TO_ENTROPY[i])
    fig = plot_confusion_matrix(cm, app_labels, 11)
    fig.savefig('entropyresult/entropy_heatmap.png')

def main():
    train_entropy_graphs("/home/pcl/PangBo/pro/GNNTrafficClassification/EntroptedNEW/Graphs/Entropy/trainset.bin","/home/pcl/PangBo/pro/GNNTrafficClassification/EntroptedNEW/Graphs/Entropy/validset.bin")
if __name__ == '__main__':
    main()
