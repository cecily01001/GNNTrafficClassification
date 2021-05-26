import dgl
import click
import torch
from torch import nn as nn
import time
from numpy import *
from torch.nn import functional as F
from utils.utilsformalware import PREFIX_TO_Malware_ID
from utils.utilsforapps import PREFIX_TO_APP_ID
from torch.utils.data import DataLoader
from dgl.nn import SGConv, GATConv, TAGConv,GlobalAttentionPooling,AvgPooling,GraphConv
from dgl.data.utils import load_graphs
class graphdataset(object):
    def __init__(self, graphs, labels,num_classes):
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
        hg=self.pooling(g,h)
        y = self.classify(hg)
        # y=self.pooling
        return y


def train_app_graphs(train, valid, test):
    train_graphs, train_labels = load_graphs(train)
    print("load train graphs")
    valid_graphs, valid_labels = load_graphs(valid)
    print("load valid graphs")
    test_graphs, test_labels = load_graphs(test)
    print("load test graphs")
    trainset = graphdataset(train_graphs, train_labels['labels'].numpy().tolist(),len(PREFIX_TO_APP_ID))
    print("trainset done")
    validset = graphdataset(valid_graphs, valid_labels['labels'].numpy().tolist(),len(PREFIX_TO_APP_ID))
    print("validset done")
    testset = graphdataset(test_graphs, test_labels['labels'].numpy().tolist(),len(PREFIX_TO_APP_ID))
    print("testset done")
    data_loader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate)
    model = Classifier(1500, 516, trainset.num_classes)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # model.train()
    min_loss = 1000
    # epoch_losses = []
    avoid_over = 0

    for epoch in range(1000):
        model.train()
        epoch_loss = 0
        for iter, (bg, label) in enumerate(data_loader):
            prediction = model(bg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()

        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        if epoch_loss < min_loss and epoch_loss < 0.04 and epoch_loss > 0.0004:
            avoid_over = 0
            model.eval()
            valid_x, valid_y = map(list, zip(*validset))
            valid_bg = dgl.batch(valid_x)
            valid_y = torch.tensor(valid_y).float().view(-1, 1)
            valid_probs_Y = torch.softmax(model(valid_bg), 1)
            valid_sampled_Y = torch.multinomial(valid_probs_Y, 1)
            valid_argmax_Y = torch.max(valid_probs_Y, 1)[1].view(-1, 1)
            result1 = (valid_y == valid_sampled_Y.float()).sum().item() / len(valid_y) * 100
            result2 = (valid_y == valid_argmax_Y.float()).sum().item() / len(valid_y) * 100
            valid_sampled_Y_done=valid_sampled_Y.float()
            sum = [0] * 41
            index = [0] * 41
            pred_sum=[0]*41
            for i in range(len(valid_x)):
                sum[int(valid_y[i, 0])] = sum[int(valid_y[i, 0])] + 1
                pred_sum[int(valid_sampled_Y_done[i,0])]=pred_sum[int(valid_sampled_Y_done[i,0])]+1
                if valid_y[i] == valid_sampled_Y.float()[i]:
                    index[int(valid_y[i, 0])] = index[int(valid_y[i, 0])] + 1
                # print(sum)
                print(pred_sum)
                print(index)

            print('Epoch {},Accuracy of sampled predictions on the valid set: {:.4f}%'.format(epoch, result1))
            print('Epoch {},Accuracy of argmax predictions on the valid set: {:.4f}%'.format(epoch, result2))
            if result1 > 78 or result2 > 78:
                min_loss = epoch_loss
                torch.save(model, 'ifmodel.pkl')
                print("model saved")
        elif epoch_loss <= 0.0004:
            avoid_over = avoid_over + 1
            if avoid_over > 10:
                break
    model = torch.load('2model.pkl')
    test_X, test_Y = map(list, zip(*testset))
    test_bg = dgl.batch(test_X)
    test_Y = torch.tensor(test_Y).float().view(-1, 1)
    probs_Y = torch.softmax(model(test_bg), 1)
    sampled_Y = torch.multinomial(probs_Y, 1)
    argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
    print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
        (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
    print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
        (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))
    sampled_Y_done = sampled_Y.float()
    sum = [0] * 41
    index = [0] * 41
    pred_sum = [0] * 41
    for i in range(len(test_Y)):
        sum[int(test_Y[i, 0])] = sum[int(test_Y[i, 0])] + 1
        pred_sum[int(sampled_Y_done[i, 0])] = pred_sum[int(sampled_Y_done[i, 0])] + 1
        if test_Y[i] == sampled_Y.float()[i]:
            index[int(test_Y[i, 0])] = index[int(test_Y[i, 0])] + 1
        # print(sum)
        print(pred_sum)
        print(index)
    j = 0
    for i in range(len(sum)):
        if sum[j] is not 0 and pred_sum[j] is not 0:
            print(str(i) + ' kind recall is: ' + str(index[j] / sum[j]))
            print(str(i) + ' kind precision is: ' + str(index[j] / pred_sum[j]))
        j = j + 1
    argmax_Y_done = argmax_Y.float()
    sum = [0] * 41
    index = [0] * 41
    pred_sum = [0] * 41
    for i in range(len(test_Y)):
        sum[int(test_Y[i, 0])] = sum[int(test_Y[i, 0])] + 1
        pred_sum[int(argmax_Y_done[i, 0])] = pred_sum[int(argmax_Y_done[i, 0])] + 1
        if test_Y[i] == argmax_Y_done.float()[i]:
            index[int(test_Y[i, 0])] = index[int(test_Y[i, 0])] + 1
        # print(sum)
        print(pred_sum)
        print(index)
    j = 0
    for i in range(len(sum)):
        if sum[j] is not 0 and pred_sum[j] is not 0:
            print(str(i) + ' kind recall is: ' + str(index[j] / sum[j]))
            print(str(i) + ' kind precision is: ' + str(index[j] / pred_sum[j]))
        j = j + 1

def train_malware_graphs(train, valid, test):
    train_graphs, train_labels = load_graphs(train)
    print("load Mtrain graphs")
    valid_graphs, valid_labels = load_graphs(test)
    print("load Mvalid graphs")
    test_graphs, test_labels = load_graphs(valid)
    print("load Mtest graphs")
    trainset = graphdataset(train_graphs, train_labels['labels'].numpy().tolist(),len(PREFIX_TO_Malware_ID))
    print("trainset done")
    validset = graphdataset(valid_graphs, valid_labels['labels'].numpy().tolist(),len(PREFIX_TO_Malware_ID))
    print("validset done")
    testset = graphdataset(test_graphs, test_labels['labels'].numpy().tolist(),len(PREFIX_TO_Malware_ID))
    print("testset done")
    data_loader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate)
    model = Classifier(1000, 300, trainset.num_classes)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # model.train()
    min_loss = 1000
    # epoch_losses = []
    avoid_over = 0
    dif_list=[]
    for epoch in range(1000):
        model.train()
        epoch_loss = 0
        # begin = time.clock()
        for iter, (bg, label) in enumerate(data_loader):
            prediction = model(bg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        # end = time.clock()
        # dif = end - begin
        # if len(dif_list) < 100:
        #     dif_list.append(dif)
        #     print('训练时长：{：.4f}'.format(dif))
        # elif len(dif_list) == 100:
        #     mean_dif=mean(dif_list)
        #     print('平均训练时长：{：.4f}'.format(mean_dif))
        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        if epoch_loss < min_loss and epoch_loss < 0.1 and epoch_loss > 0.0003:
            avoid_over = 0
            model.eval()
            valid_x, valid_y = map(list, zip(*validset))
            valid_bg = dgl.batch(valid_x)
            valid_y = torch.tensor(valid_y).float().view(-1, 1)
            valid_probs_Y = torch.softmax(model(valid_bg), 1)
            valid_sampled_Y = torch.multinomial(valid_probs_Y, 1)
            valid_argmax_Y = torch.max(valid_probs_Y, 1)[1].view(-1, 1)
            result1 = (valid_y == valid_sampled_Y.float()).sum().item() / len(valid_y) * 100
            result2 = (valid_y == valid_argmax_Y.float()).sum().item() / len(valid_y) * 100

            sum = [0] * 6
            index = [0] * 6
            for i in range(len(valid_x)):
                sum[int(valid_y[i, 0])] = sum[int(valid_y[i, 0])] + 1
                if valid_y[i] == valid_sampled_Y.float()[i]:
                    index[int(valid_y[i, 0])] = index[int(valid_y[i, 0])] + 1
                print(sum)
                print(index)

            print('Epoch {},Accuracy of sampled predictions on the valid set: {:.4f}%'.format(epoch, result1))
            print('Epoch {},Accuracy of argmax predictions on the valid set: {:.4f}%'.format(epoch, result2))
            if result1 > 80 or result2 > 80:
                min_loss = epoch_loss
                torch.save(model, 'mmodel.pkl')
                print("model saved")
        elif epoch_loss <= 0.001:
            avoid_over = avoid_over + 1
            if avoid_over > 10:
                break
    model = torch.load('mmodel.pkl')
    test_X, test_Y = map(list, zip(*testset))
    test_bg = dgl.batch(test_X)
    test_Y = torch.tensor(test_Y).float().view(-1, 1)
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
        pred_sum[int(sampled_Y_done[i, 0])] = pred_sum[int(sampled_Y_done[i, 0])] + 1
        if test_Y[i] == sampled_Y.float()[i]:
            index[int(test_Y[i, 0])] = index[int(test_Y[i, 0])] + 1
        print(pred_sum)
        print(index)
    j = 0
    print("Accuracy of sampled predictions")
    for i in range(len(sum)):
        if sum[j] is not 0 :
            print(str(i) + ' kind recall is: ' + str(index[j] / sum[j]))
        else:
            print(str(i) + ' kind recall is: -1')
        if pred_sum[j] is not 0:
            print(str(i) + ' kind precision is: ' + str(index[j] / pred_sum[j]))
        else:
            print(str(i) + ' kind precision is: -1')
        j = j + 1

    sum = [0] * 6
    index = [0] * 6
    pred_sum = [0] * 6
    for i in range(len(test_Y)):
        sum[int(test_Y[i, 0])] = sum[int(test_Y[i, 0])] + 1
        pred_sum[int(argmax_Y_done[i, 0])] = pred_sum[int(argmax_Y_done[i, 0])] + 1
        if test_Y[i] == argmax_Y.float()[i]:
            index[int(test_Y[i, 0])] = index[int(test_Y[i, 0])] + 1
        print(pred_sum)
        print(index)
    j = 0
    print("Accuracy of argmax predictions")
    for i in range(len(sum)):
        if sum[j] is not 0:
            print(str(i) + ' kind recall is: ' + str(index[j] / sum[j]))
        else:
            print(str(i) + ' kind recall is: -1')
        if pred_sum[j] is not 0:
            print(str(i) + ' kind precision is: ' + str(index[j] / pred_sum[j]))
        else:
            print(str(i) + ' kind precision is: -1')
        j = j + 1

@click.command()
@click.option('-k', '--kind', help='object to be classificated', required=True)
def main(kind):
    if kind=='app':
        train_app_graphs("Graphs/APP/trainset.bin","Graphs/APP/testset.bin","Graphs/APP/validset.bin")
    elif kind=='malware':
        train_malware_graphs("Graphs/Malware/trainset.bin","Graphs/Malware/validset.bin","Graphs/Malware/testset.bin")


if __name__ == '__main__':
    main()
