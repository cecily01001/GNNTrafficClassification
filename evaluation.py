import csv
import seaborn as sns
import matplotlib as mpl
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import dgl
import click
import torch
from torch import nn as nn
import time
import numpy as np
from torch.nn import functional as F
from utils.utilsformalware import PREFIX_TO_Malware_ID
from utils.utilsforapps import PREFIX_TO_APP_ID, ID_TO_APP
from torch.utils.data import DataLoader
from dgl.nn import SGConv, GATConv, TAGConv, GlobalAttentionPooling, AvgPooling, GraphConv
from dgl.data.utils import load_graphs
from utils.utilsforentropy import PREFIX_TO_ENTROPY_ID,ID_TO_ENTROPY

from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
cnames = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}

def plot_embedding_2D(data, label, title):
    # x_min, x_max = np.min(data, 0), np.max(data, 0)
    vis_x = data[:, 0]  # 0维
    vis_y = data[:, 1]  # 1维
    father_list = []
    color_list=[]
    fig = plt.figure()
    # for i in range(41):
    #     temp_list = []
    #     father_list.append(temp_list)
    # for j in range(41):
    #     father_list[j]= [i for i in range(len(label)) if label == j]
    # for i in range(41):
    #     color = plt.cm.Set1(i)
    #     color_list.append(color)
    # print(color_list)
    # for i in range(41):
    #     plt.scatter(vis_x[father_list[i]], vis_y[father_list[i]], c=color_list[i], cmap='brg', label=i)

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    color_list=[]
    for i in cnames:
        color_list.append(i)
    print(label)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 # color=color_list[label[i]],
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig
def get_sne_data(label, feature_array):  # Input_path为你自己原始数据存储路径，我的路径就是上面的'./Images'

    data = np.zeros((len(label), 5))  # 初始化一个np.array数组用于存数据
    label_np = np.zeros((len(label),))  # 初始化一个np.array数组用于存数据

    for k in range(len(label)):
        label_np[k] = int(label[k])
    # print('label_np')
    # print(label_np)
    for i in range(len(label)):
        data[i] = feature_array[i].detach()
        n_samples, n_features = data.shape
    print(data)
    return data, label, n_samples, n_features

def normalise_cm(cm):
    with np.errstate(all='ignore'):
        normalised_cm = cm / cm.sum(axis=1, keepdims=True)
        normalised_cm = np.nan_to_num(normalised_cm)
        return normalised_cm


def plot_confusion_matrix(cm, labels,num):
    normalised_cm = normalise_cm(cm)
    fig, ax = plt.subplots(figsize=(num, num))
    sns.heatmap(
        data=normalised_cm, cmap='YlGnBu',
        xticklabels=labels, yticklabels=labels,
        annot=True, ax=ax, fmt='.1f'
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
        self.pooling = AvgPooling()
        self.classify = nn.Linear(256, n_classes)

    def forward(self, g):
        # 使用节点的入度作为初始特征
        h = g.ndata['h'].float()
        # print(h)
        # h1 = F.relu(self.conv1(g, h))
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


def train_app_graphs(train, valid, test):
    train_graphs, train_labels = load_graphs(train)
    print("load train graphs")
    valid_graphs, valid_labels = load_graphs(valid)
    print("load valid graphs")
    test_graphs, test_labels = load_graphs(test)
    print("load test graphs")
    trainset = graphdataset(train_graphs, train_labels['labels'].numpy().tolist(), len(PREFIX_TO_APP_ID))
    print("trainset done")
    validset = graphdataset(valid_graphs, valid_labels['labels'].numpy().tolist(), len(PREFIX_TO_APP_ID))
    print("validset done")
    testset = graphdataset(test_graphs, test_labels['labels'].numpy().tolist(), len(PREFIX_TO_APP_ID))
    print("testset done")
    data_loader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate)
    model = Classifier(1500, 516, trainset.num_classes)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # model.train()
    min_loss = 1000
    # epoch_losses = []
    avoid_over = 0
    epoch_file = open('appresult/app_epoch_process.csv', 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(epoch_file)
    csv_writer.writerow(["num", "epoch"])
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
        # epoch_losses.append(epoch_loss)
        csv_writer.writerow([epoch, epoch_loss])
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        if epoch_loss < min_loss and epoch_loss < 1 and epoch_loss > 0.006:
            avoid_over = 0
            model.eval()
            valid_x, valid_y = map(list, zip(*validset))
            valid_bg = dgl.batch(valid_x)
            valid_y = torch.tensor(valid_y).float().view(-1, 1)
            valid_probs_Y = torch.softmax(model(valid_bg), 1)
            valid_sampled_Y = torch.multinomial(valid_probs_Y, 1)
            valid_argmax_Y = torch.max(valid_probs_Y, 1)[1].view(-1, 1)
            accuracy1 = (valid_y == valid_sampled_Y.float()).sum().item() / len(valid_y) * 100
            accuracy2 = (valid_y == valid_argmax_Y.float()).sum().item() / len(valid_y) * 100
            valid_sampled_Y_done = valid_sampled_Y.float()
            sum = [0] * 41
            index = [0] * 41
            pred_sum = [0] * 41
            for i in range(len(valid_x)):
                sum[int(valid_y[i, 0])] = sum[int(valid_y[i, 0])] + 1
                pred_sum[int(valid_sampled_Y_done[i, 0])] = pred_sum[int(valid_sampled_Y_done[i, 0])] + 1
                if valid_y[i] == valid_sampled_Y.float()[i]:
                    index[int(valid_y[i, 0])] = index[int(valid_y[i, 0])] + 1
                # print(sum)
                # print(pred_sum)
                # print(index)
            print('Epoch {},Accuracy of sampled predictions on the valid set: {:.4f}%'.format(epoch, accuracy1))
            print('Epoch {},Accuracy of argmax predictions on the valid set: {:.4f}%'.format(epoch, accuracy2))
            if accuracy1 > 85 or accuracy2 > 85:
                min_loss = epoch_loss
                torch.save(model, 'appresult/appmodel.pkl')
                print("model saved")
        elif epoch_loss <= 0.006:
            avoid_over = avoid_over + 1
            if avoid_over > 20:
                break
    model = torch.load('appresult/appmodel.pkl')
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
    accuracyfile = open('appresult/app_accuracyfile.csv', 'w', encoding='utf-8', newline="")
    accuracyfile_writer = csv.writer(accuracyfile)
    accuracyfile_writer.writerow(["num", "accuracy"])
    accuracyfile_writer.writerow(["sampled",app_test_accuracy1])
    accuracyfile_writer.writerow(["argmax", app_test_accuracy2])
    cm = np.zeros((trainset.num_classes, trainset.num_classes), dtype=float)
    if app_test_accuracy1>=app_test_accuracy2:
        sampled_Y_done = sampled_Y.float()
        for i in range(len(test_Y)):
            cm[int(test_Y[i, 0]), int(sampled_Y_done[i, 0])] += 1
        sum = [0] * 41
        index = [0] * 41
        pred_sum = [0] * 41
        for i in range(len(test_Y)):
            sum[int(test_Y[i, 0])] = sum[int(test_Y[i, 0])] + 1
            pred_sum[int(sampled_Y_done[i, 0])] = pred_sum[int(sampled_Y_done[i, 0])] + 1
            if test_Y[i] == sampled_Y.float()[i]:
                index[int(test_Y[i, 0])] = index[int(test_Y[i, 0])] + 1
        j = 0
        f = open('appresult/app_recision_and_recall.csv', 'w', encoding='utf-8', newline="")
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
            app_name = ID_TO_APP.get(i)
            csv_writer.writerow([app_name, recall, precision])
            j = j + 1
    else:
        argmax_Y_done = argmax_Y.float()
        for i in range(len(test_Y)):
            cm[int(test_Y[i, 0]), int(argmax_Y_done[i, 0])] += 1
        sum = [0] * 41
        index = [0] * 41
        pred_sum = [0] * 41
        for i in range(len(test_Y)):
            sum[int(test_Y[i, 0])] = sum[int(test_Y[i, 0])] + 1
            pred_sum[int(argmax_Y_done[i, 0])] = pred_sum[int(argmax_Y_done[i, 0])] + 1
            if test_Y[i] == argmax_Y_done.float()[i]:
                index[int(test_Y[i, 0])] = index[int(test_Y[i, 0])] + 1
        j = 0
        print("Accuracy of argmax predictions")
        f = open('appresult/app_recision_and_recall.csv', 'w', encoding='utf-8', newline="")
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
            app_name = ID_TO_APP.get(i)
            csv_writer.writerow([app_name, recall, precision])
            j = j + 1
    app_labels = []
    for i in sorted(list(ID_TO_APP.keys())):
        app_labels.append(ID_TO_APP[i])
    fig=plot_confusion_matrix(cm, app_labels)
    fig.savefig('appresult/heatmap.png')

def train_entropy_graphs(train, valid, test):
    train_graphs, train_labels = load_graphs(train)
    print("load train graphs")
    valid_graphs, valid_labels = load_graphs(valid)
    print("load valid graphs")
    test_graphs, test_labels = load_graphs(test)
    print("load test graphs")
    trainset = graphdataset(train_graphs, train_labels['labels'].numpy().tolist(), len(PREFIX_TO_ENTROPY_ID))
    print("trainset done")
    validset = graphdataset(valid_graphs, valid_labels['labels'].numpy().tolist(), len(PREFIX_TO_ENTROPY_ID))
    print("validset done")
    testset = graphdataset(test_graphs, test_labels['labels'].numpy().tolist(), len(PREFIX_TO_ENTROPY_ID))
    print("testset done")
    data_loader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate)
    print(trainset.num_classes)
    print(trainset.labels)
    model = Classifier(1500, 516, trainset.num_classes)
    # loss_func = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # model.train()
    # min_loss = 1000
    # # epoch_losses = []
    # avoid_over = 0
    # epoch_file = open('entropyresult/entropy_epoch_process.csv', 'w', encoding='utf-8', newline="")
    # csv_writer = csv.writer(epoch_file)
    # csv_writer.writerow(["num", "epoch"])
    # for epoch in range(1000):
    #     model.train()
    #     epoch_loss = 0
    #     for iter, (bg, label) in enumerate(data_loader):
    #         prediction = model(bg)
    #         loss = loss_func(prediction, label)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.detach().item()
    #     epoch_loss /= (iter + 1)
    #     # epoch_losses.append(epoch_loss)
    #     csv_writer.writerow([epoch, epoch_loss])
    #     print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    #     if epoch_loss < min_loss and epoch_loss < 0.6 and epoch_loss > 0.53:
    #         avoid_over = 0
    #         model.eval()
    #         valid_x, valid_y = map(list, zip(*validset))
    #         valid_bg = dgl.batch(valid_x)
    #         valid_y = torch.tensor(valid_y).float().view(-1, 1)
    #         valid_probs_Y = torch.softmax(model(valid_bg), 1)
    #         valid_sampled_Y = torch.multinomial(valid_probs_Y, 1)
    #         valid_argmax_Y = torch.max(valid_probs_Y, 1)[1].view(-1, 1)
    #         accuracy1 = (valid_y == valid_sampled_Y.float()).sum().item() / len(valid_y) * 100
    #         accuracy2 = (valid_y == valid_argmax_Y.float()).sum().item() / len(valid_y) * 100
    #         valid_sampled_Y_done = valid_sampled_Y.float()
    #         sum = [0] * 41
    #         index = [0] * 41
    #         pred_sum = [0] * 41
    #         for i in range(len(valid_x)):
    #             sum[int(valid_y[i, 0])] = sum[int(valid_y[i, 0])] + 1
    #             pred_sum[int(valid_sampled_Y_done[i, 0])] = pred_sum[int(valid_sampled_Y_done[i, 0])] + 1
    #             if valid_y[i] == valid_sampled_Y.float()[i]:
    #                 index[int(valid_y[i, 0])] = index[int(valid_y[i, 0])] + 1
    #         print('Epoch {},Accuracy of sampled predictions on the valid set: {:.4f}%'.format(epoch, accuracy1))
    #         print('Epoch {},Accuracy of argmax predictions on the valid set: {:.4f}%'.format(epoch, accuracy2))
    #         if accuracy1 > 70 or accuracy2 > 70:
    #             min_loss = epoch_loss
    #             torch.save(model, 'entropyresult/entropymodel.pkl')
    #             print("model saved")
    #     elif epoch_loss <= 0.53:
    #         avoid_over = avoid_over + 1
    #         if avoid_over > 10:
    #             break
    model = torch.load('entropyresult/entropymodel.pkl')
    # model = torch.load('appresult/appmodel.pkl')
    since = time.time()
    test_X, test_Y = map(list, zip(*testset))
    test_bg = dgl.batch(test_X)
    test_Y = torch.tensor(test_Y).float().view(-1, 1)
    probs_Y = torch.softmax(model(test_bg), 1)
    sampled_Y = torch.multinomial(probs_Y, 1)
    argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
    app_test_accuracy1 = (test_Y == sampled_Y.float()).sum().item() / len(test_Y)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    app_test_accuracy2 = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)
    print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
        (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
    print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
        (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))
    accuracyfile = open('entropyresult/accuracyfile.csv', 'w', encoding='utf-8', newline="")
    accuracyfile_writer = csv.writer(accuracyfile)
    accuracyfile_writer.writerow(["num", "accuracy"])
    accuracyfile_writer.writerow(["sampled",app_test_accuracy1])
    accuracyfile_writer.writerow(["argmax", app_test_accuracy2])
    if app_test_accuracy1>=app_test_accuracy2:
        sampled_Y_done = sampled_Y.float()
        sum = [0] * 11
        index = [0] * 11
        pred_sum = [0] * 11
        for i in range(len(test_Y)):
            sum[int(test_Y[i, 0])] = sum[int(test_Y[i, 0])] + 1
            pred_sum[int(sampled_Y_done[i, 0])] = pred_sum[int(sampled_Y_done[i, 0])] + 1
            if test_Y[i] == sampled_Y.float()[i]:
                index[int(test_Y[i, 0])] = index[int(test_Y[i, 0])] + 1
        j = 0
        f = open('entropyresult/app_recision_and_recall.csv', 'w', encoding='utf-8', newline="")
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
            app_name = ID_TO_APP.get(i)
            csv_writer.writerow([app_name, recall, precision])
            j = j + 1
    else:
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


def train_malware_graphs(train, valid, test):
    # train_graphs, train_labels = load_graphs(train)
    # print("load Mtrain graphs")
    # valid_graphs, valid_labels = load_graphs(test)
    # print("load Mvalid graphs")
    test_graphs, test_labels = load_graphs(valid)
    print("load Mtest graphs")
    # trainset = graphdataset(train_graphs, train_labels['labels'].numpy().tolist(), len(PREFIX_TO_Malware_ID))
    # print("trainset done")
    # validset = graphdataset(valid_graphs, valid_labels['labels'].numpy().tolist(), len(PREFIX_TO_Malware_ID))
    # print("validset done")
    testset = graphdataset(test_graphs, test_labels['labels'].numpy().tolist(), len(PREFIX_TO_Malware_ID))
    print("testset done")
    # data_loader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate)
    # model = Classifier(1000, 516, trainset.num_classes)
    # loss_func = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # model.train()
    min_loss = 1000
    # epoch_losses = []
    avoid_over = 0
    dif_list = []
    # for epoch in range(1000):
    #     model.train()
    #     epoch_loss = 0
    #     # begin = time.clock()
    #     for iter, (bg, label) in enumerate(data_loader):
    #         prediction = model(bg)
    #         loss = loss_func(prediction, label)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.detach().item()
    #     # end = time.clock()
    #     # dif = end - begin
    #     # if len(dif_list) < 100:
    #     #     dif_list.append(dif)
    #     #     print('训练时长：{：.4f}'.format(dif))
    #     # elif len(dif_list) == 100:
    #     #     mean_dif=mean(dif_list)
    #     #     print('平均训练时长：{：.4f}'.format(mean_dif))
    #     epoch_loss /= (iter + 1)
    #     print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    #     if epoch_loss < min_loss and epoch_loss < 0.1 and epoch_loss > 0.0003:
    #         avoid_over = 0
    #         model.eval()
    #         valid_x, valid_y = map(list, zip(*validset))
    #         valid_bg = dgl.batch(valid_x)
    #         valid_y = torch.tensor(valid_y).float().view(-1, 1)
    #         valid_probs_Y = torch.softmax(model(valid_bg), 1)
    #         valid_sampled_Y = torch.multinomial(valid_probs_Y, 1)
    #         valid_argmax_Y = torch.max(valid_probs_Y, 1)[1].view(-1, 1)
    #         result1 = (valid_y == valid_sampled_Y.float()).sum().item() / len(valid_y) * 100
    #         result2 = (valid_y == valid_argmax_Y.float()).sum().item() / len(valid_y) * 100
    #
    #         sum = [0] * 6
    #         index = [0] * 6
    #         for i in range(len(valid_x)):
    #             sum[int(valid_y[i, 0])] = sum[int(valid_y[i, 0])] + 1
    #             if valid_y[i] == valid_sampled_Y.float()[i]:
    #                 index[int(valid_y[i, 0])] = index[int(valid_y[i, 0])] + 1
    #             print(sum)
    #             print(index)
    #
    #         print('Epoch {},Accuracy of sampled predictions on the valid set: {:.4f}%'.format(epoch, result1))
    #         print('Epoch {},Accuracy of argmax predictions on the valid set: {:.4f}%'.format(epoch, result2))
    #         if result1 > 80 or result2 > 80:
    #             min_loss = epoch_loss
    #             torch.save(model, 'mmodel.pkl')
    #             print("model saved")
    #     elif epoch_loss <= 0.001:
    #         avoid_over = avoid_over + 1
    #         if avoid_over > 10:
    #             break

    # model = torch.load('/home/pcl/PangBo/pro/GNNTrafficClassification/malwareresult/mmodel.pkl')
    model = torch.load('appresult/appmodel.pkl')
    test_X, test_Y = map(list, zip(*testset))
    test_bg = dgl.batch(test_X)
    test_Y = torch.tensor(test_Y).float().view(-1, 1)
    # y=model(test_bg)
    # data, label, n_samples, n_features = get_sne_data(test_labels['labels'].numpy().tolist(), y)
    # tsne_2D = TSNE(n_components=2, init='pca', random_state=0)  # 调用TSNE
    # result_2D = tsne_2D.fit_transform(data)
    # fig1 = plot_embedding_2D(result_2D, label, 'm_1_t-SNE')
    # # plt.show(fig1)
    # fig1.savefig('m_1_sne.png')
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
        if sum[j] is not 0:
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
    f = open('app_recision_and_recall.csv', 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["app", "recall", "precision"])
    # for item in acc_list:
    #     csv_writer.writerow([item[1], item[2], item[3]])
    for i in range(len(sum)):
        if sum[j] is not 0:
            recall = str(index[j] / sum[j])
            print(str(i) + ' kind recall is: ' + str(index[j] / sum[j]))
        else:
            recall = -1
            print(str(i) + ' kind recall is: -1')
        if pred_sum[j] is not 0:
            precision = str(index[j] / pred_sum[j])
            print(str(i) + ' kind precision is: ' + str(index[j] / pred_sum[j]))
        else:
            precision = -1
            print(str(i) + ' kind precision is: -1')
        app_name = ID_TO_APP.get(i)
        csv_writer.writerow([app_name, recall, precision])
        j = j + 1


@click.command()
@click.option('-k', '--kind', help='object to be classificated', required=True)
def main(kind):
    if kind == 'app':
        train_app_graphs("Graphs/APP/trainset.bin", "Graphs/APP/testset.bin", "Graphs/APP/validset.bin")
    elif kind == 'malware':
        train_malware_graphs("Graphs/Malware/trainset.bin", "Graphs/Malware/validset.bin", "Graphs/Malware/testset.bin")
    else:
        train_entropy_graphs("Graphs/Entropy/trainset.bin", "Graphs/Entropy/validset.bin", "Graphs/Entropy/testset.bin")

if __name__ == '__main__':
    main()
