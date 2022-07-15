import csv
import time
import seaborn as sns
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
from utils.utilsformalware import PREFIX_TO_Malware_ID,ID_TO_Malware
from utils.utilsforapps import PREFIX_TO_APP_ID, ID_TO_APP
from torch.utils.data import DataLoader
from dgl.nn import SGConv, GATConv, TAGConv, GATConv, AvgPooling, GraphConv,MaxPooling,SumPooling,EdgeConv
from dgl.data.utils import load_graphs
from utils.utilsforentropy import PREFIX_TO_ENTROPY_ID, ID_TO_ENTROPY

# def plot_embedding_2D(data, label, title):
#     x_min, x_max = np.min(data, 0), np.max(data, 0)
#     data = (data - x_min) / (x_max - x_min)
#     fig = plt.figure()
#     for i in range(data.shape[0]):
#         plt.text(data[i, 0], data[i, 1], str(label[i]),
#                  color=plt.cm.Set1(label[i]),
#                  fontdict={'weight': 'bold', 'size': 9})
#     plt.xticks([])
#     plt.yticks([])
#     plt.title(title)
#     return fig
# def get_sne_data(label,feature_array,node_numlist): #Input_path为你自己原始数据存储路径，我的路径就是上面的'./Images'
#
#     data=np.zeros((len(label),40000)) #初始化一个np.array数组用于存数据
#     label=np.zeros((len(label),)) #初始化一个np.array数组用于存数据
#
#     #读取并存储图片数据，原图为rgb三通道，而且大小不一，先灰度化，再resize成200x200固定大小
#     for i in range(len(node_numlist)):
#         image_path=os.path.join(Input_path,Image_names[i])
#         img=cv2.imread(image_path)
#         img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#         img=cv2.resize(img_gray,(200,200))
#         img=feature_array.reshape(1,node_num*1500)
#         data[i]=img
#         n_samples, n_features = data.shape
#     return data, label, n_samples, n_features
use_gpu = torch.cuda.is_available()
print('use gpu:',use_gpu)
# use_gpu=False
class Attention_Layer(nn.Module):

    # 用来实现mask-attention layer
    def __init__(self, hidden_dim, is_bi_rnn):
        super(Attention_Layer, self).__init__()

        self.hidden_dim = hidden_dim
        self.is_bi_rnn = is_bi_rnn

        # 下面使用nn的Linear层来定义Q，K，V矩阵
        if is_bi_rnn:
            # 是双向的RNN
            self.Q_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
            self.K_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
            self.V_linear = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        else:
            # 单向的RNN
            self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, inputs, lens):

        size = inputs.size()
        # print(size)
        # 计算生成QKV矩阵
        Q = self.Q_linear(inputs)
        # K = self.K_linear(inputs)
        K = self.K_linear(inputs).permute(0, 2, 1)  # 先进行一次转置
        V = self.V_linear(inputs)

        # 还要计算生成mask矩阵
        max_len = max(lens)  # 最大的句子长度，生成mask矩阵
        sentence_lengths = torch.Tensor(lens)  # 代表每个句子的长度
        mask = torch.arange(sentence_lengths.max().item())[None, :] < sentence_lengths[:, None]
        mask = mask.unsqueeze(dim=1)  # [batch_size, 1, max_len]
        mask = mask.expand(size[0], max_len, max_len)  # [batch_size, max_len, max_len]

        # print('\nmask is :', mask.size())

        # 下面生成用来填充的矩阵
        padding_num = torch.ones_like(mask)
        padding_num = -2 ** 31 * padding_num.float()

        # print('\npadding num is :', padding_num.size())

        # 下面开始计算啦
        alpha = torch.matmul(Q, K)
        # 下面开始mask
        alpha = torch.where(mask.cuda(), alpha.cuda(), padding_num.cuda())
        # 下面开始softmax
        alpha = F.softmax(alpha)
        # alpha = F.softmax(alpha)
        out = torch.matmul(alpha, V)

        return out
class attMLP(nn.Module):
    def __init__(self, input_size, common_size):
        super(attMLP, self).__init__()
        # self.transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        # self.gat = GATConv(input_size, 128, num_heads=1)
        # print(input_size)
        self.gat = Attention_Layer(input_size, False)
        self.linear = nn.Sequential(
            nn.Linear(input_size, common_size),
            nn.ReLU(inplace=True),
            # nn.Linear(256, 180),
            # nn.ReLU(inplace=True),
            # nn.Linear(256,128),
            # nn.ReLU(inplace=True),
            # nn.Linear(64, 50),
            # nn.ReLU(inplace=True),
            # nn.Linear(64, 54),
            # nn.ReLU(inplace=True),
            # nn.Linear(common_size, common_size)
        )
        self.w_omega = nn.Parameter(torch.Tensor(input_size , input_size ))
        self.u_omega = nn.Parameter(torch.Tensor(input_size , input_size))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def attention_net(self, x):  # x:[batch, seq_len, hidden_dim*2]
        u = torch.tanh(torch.matmul(x, self.w_omega))  # [batch, seq_len, hidden_dim*2]
        att = torch.matmul(u, self.u_omega)  # [batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)
        context = x * att_score  # [batch, seq_len, hidden_dim*2]
        return context
    def forward(self, x):
        # x = self.gat(g,x)
        x=x.unsqueeze(2)
        x=x.expand(len(x),64,64)
        print(x.shape)
        x = self.gat(x,[64]*len(x))
        # output = x.permute(1, 0)
        # attn_output = self.attention_net(x)

        print(x[0:2])
        out = self.linear(x)
        print(out.shape)
        return out
class MLP(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, 64),
            # nn.ReLU(inplace=True),
            # nn.Linear(256, 180),
            # nn.ReLU(inplace=True),
            # nn.Linear(256,128),
            # nn.ReLU(inplace=True),
            # nn.Linear(64, 50),
            # nn.ReLU(inplace=True),
            # nn.Linear(64, 54),
            # nn.ReLU(inplace=True),
            # nn.Linear(50, common_size)
        )
    def forward(self, x):
        out = self.linear(x)
        return out

class MLP_try1(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, 82),
            nn.ReLU(inplace=True),
            # nn.Linear(256, 180),
            # nn.ReLU(inplace=True),
            nn.Linear(82,64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            # nn.Linear(64, 54),
            # nn.ReLU(inplace=True),
            nn.Linear(64, common_size)
        )
    def forward(self, x):
        out = self.linear(x)
        return out

class MLP1(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP1, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, (256)),
            nn.ReLU(inplace=True),
            # nn.Linear(256, 180),
            # nn.ReLU(inplace=True),
            nn.Linear(256,128),
            # nn.ReLU(inplace=True),
            # nn.Linear(128, 64),
            # nn.ReLU(inplace=True),
            # nn.Linear(64, 54),
            # nn.ReLU(inplace=True),
            # nn.Linear(64, common_size)
        )
    def forward(self, x):
        out = self.linear(x)
        return out
class MLP2(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP2, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            # nn.Linear(64, 54),
            # nn.ReLU(inplace=True),
            nn.Linear(64, common_size)
        )
    def forward(self, x):
        out = self.linear(x)
        return out
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
        self.sgc0 = SGConv(in_dim, 256)
        self.sgc1 = SGConv(in_dim, hidden_dim)
        self.sgc2 = SGConv(hidden_dim, 64)
        self.gcn0 = GraphConv(in_dim, 256, norm='both', weight=True, bias=True)
        self.gcn1 = GraphConv(1024, hidden_dim, norm='both', weight=True, bias=True)
        self.gcn2 = GraphConv(hidden_dim, 256, norm='both', weight=True, bias=True)
        # self.conv3 = SGConv(256, 126)
        # self.conv1 = TAGConv(in_dim, hidden_dim)
        # self.conv2 = TAGConv(hidden_dim, 256)
        # self.conv1 = GraphConv(in_dim, hidden_dim)
        # self.conv2 = GraphConv(hidden_dim, 256)
        self.gat = GATConv(hidden_dim, 256, num_heads=1)
        # gate_nn = nn.Linear(516, 1)
        # self.pooling = SumPooling()
        # self.pooling = MaxPooling()
        self.mlp = MLP(64 + n_classes + 1, n_classes)
        self.mlp1 = MLP1(256 + n_classes + 1, n_classes)
        self.pooling = AvgPooling()
        self.mlp2=MLP2(256+n_classes+2,n_classes)
        self.attmlp=attMLP(64,41)
        # self.classify = nn.Linear(256+n_classes+1, n_classes)

    def forward(self, g):
        # 使用节点的入度作为初始特征
        h = g.ndata['h'].float()
        h = F.relu(self.sgc1(g, h))
        h = F.relu(self.sgc2(g, h))
        # h = F.relu(self.gcn2(g, h))
        # h = F.relu(self.conv1(g, h))
        # h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h  ## 节点特征经过两层卷积的输出
        text = g.ndata['x'].float()
        # print(h.shape)
        # print(text.shape)
        # print(torch.max(text, 1)[1])
        h = torch.cat((h, text), 1)
        g.ndata['h'] = h
        # print(h.shape)
        h = self.mlp(torch.tensor(h))
        y = self.pooling(g, h)
        y=self.attmlp(y)
        # y = self.mlp2(hg)
        return y


def train_app_graphs(train, valid, test):
    train_graphs, train_labels = load_graphs(train)
    print("load train graphs")
    valid_graphs, valid_labels = load_graphs(valid)
    print("load valid graphs")
    test_graphs, test_labels = load_graphs(test)
    print("load test graphs")
    device = torch.device("cuda:0")
    trainset = graphdataset(train_graphs, train_labels['labels'].numpy().tolist(), len(PREFIX_TO_APP_ID))
    print("trainset done")
    validset = graphdataset(valid_graphs, valid_labels['labels'].numpy().tolist(), len(PREFIX_TO_APP_ID))
    print("validset done")
    testset = graphdataset(test_graphs, test_labels['labels'].numpy().tolist(), len(PREFIX_TO_APP_ID))
    print("testset done")
    data_loader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate)
    model = Classifier(1500, 512, trainset.num_classes)
    loss_func = nn.CrossEntropyLoss()

    if (use_gpu):
        model = model.to(device)
        loss_func = loss_func.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # model.train()
    min_loss = 1000
    # epoch_losses = []
    avoid_over = 0
    epoch_file = open('appresult/app_epoch_process.csv', 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(epoch_file)
    csv_writer.writerow(["num", "epoch"])
    time_file = open('appresult/train.csv', 'w', encoding='utf-8', newline="")
    time_writer = csv.writer(time_file)
    time_writer.writerow(["epoch", "time"])
    for epoch in range(400):
        # since = time.time()
        model.train()
        epoch_loss = 0
        for iter, (bg, label) in enumerate(data_loader):
            if (use_gpu):
                bg, label = bg.to(device), label.to(device)
            prediction = model(bg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        # epoch_losses.append(epoch_loss)
        # time_elapsed = time.time() - since
        # print('Training complete in {:.0f}m {:.0f}s'.format(
        #     time_elapsed // 60, time_elapsed % 60))
        csv_writer.writerow([epoch, epoch_loss])
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        if epoch_loss < min_loss and epoch_loss < 0.1 and epoch_loss > 0.006:
            avoid_over = 0
            model.eval()
            valid_x, valid_y = map(list, zip(*validset))
            valid_bg = dgl.batch(valid_x).to(device)
            valid_y = torch.tensor(valid_y).float().view(-1, 1).to(device)
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

        # time_writer.writerow([epoch, time_elapsed])
    model = torch.load('appresult/appmodel.pkl')
    test_X, test_Y = map(list, zip(*testset))
    test_bg = dgl.batch(test_X).to(device)
    test_Y = torch.tensor(test_Y).float().view(-1, 1).to(device)
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
    accuracyfile_writer.writerow(["sampled", app_test_accuracy1])
    accuracyfile_writer.writerow(["argmax", app_test_accuracy2])
    cm = zeros((trainset.num_classes, trainset.num_classes), dtype=float)
    if app_test_accuracy1 >= app_test_accuracy2:
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
            # print(sum)
            # print(pred_sum)
            # print(index)
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
    fig = plot_confusion_matrix(cm, app_labels,41)
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
    print(trainset)
    validset = graphdataset(valid_graphs, valid_labels['labels'].numpy().tolist(), len(PREFIX_TO_ENTROPY_ID))
    print("validset done")
    testset = graphdataset(test_graphs, test_labels['labels'].numpy().tolist(), len(PREFIX_TO_ENTROPY_ID))
    print("testset done")
    data_loader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate)
    model = Classifier(1500, 516, trainset.num_classes)
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
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        # epoch_losses.append(epoch_loss)
        csv_writer.writerow([epoch, epoch_loss])
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        if epoch_loss < min_loss and epoch_loss < 0.6 :
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
            app_name = ID_TO_APP.get(i)
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



def train_malware_graphs(train, valid, test):
    train_graphs, train_labels = load_graphs(train)
    print("load Mtrain graphs")
    valid_graphs, valid_labels = load_graphs(test)
    print("load Mvalid graphs")
    test_graphs, test_labels = load_graphs(valid)
    print("load Mtest graphs")
    device = torch.device("cuda:0")
    trainset = graphdataset(train_graphs, train_labels['labels'].numpy().tolist(), len(PREFIX_TO_Malware_ID))
    print("trainset done")
    validset = graphdataset(valid_graphs, valid_labels['labels'].numpy().tolist(), len(PREFIX_TO_Malware_ID))
    print("validset done")
    testset = graphdataset(test_graphs, test_labels['labels'].numpy().tolist(), len(PREFIX_TO_Malware_ID))
    print("testset done")
    data_loader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate)
    print(trainset.num_classes)
    # model = Classifier(1500, 516, trainset.num_classes)
    model = Classifier(1500, 516, 41)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # model.train()
    min_loss = 1000
    # epoch_losses = []
    avoid_over = 0
    epoch_file = open('malwareresult/malware_epoch_process.csv', 'w', encoding='utf-8', newline="")
    csv_writer = csv.writer(epoch_file)
    csv_writer.writerow(["num", "epoch"])
    for epoch in range(400):
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
        csv_writer.writerow([epoch, epoch_loss])
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

            sum = [0] * 5
            index = [0] * 5
            for i in range(len(valid_x)):
                sum[int(valid_y[i, 0])] = sum[int(valid_y[i, 0])] + 1
                if valid_y[i] == valid_sampled_Y.float()[i]:
                    index[int(valid_y[i, 0])] = index[int(valid_y[i, 0])] + 1
                print(sum)
                print(index)

            print('Epoch {},Accuracy of sampled predictions on the valid set: {:.4f}%'.format(epoch, result1))
            print('Epoch {},Accuracy of argmax predictions on the valid set: {:.4f}%'.format(epoch, result2))
            if result1 > 85 or result2 > 85:
                min_loss = epoch_loss
                torch.save(model, 'malwareresult/mmodel.pkl')
                print("model saved")
        elif epoch_loss <= 0.05:
            avoid_over = avoid_over + 1
            if avoid_over > 10:
                break

    model = torch.load('malwareresult/mmodel.pkl')
    test_X, test_Y = map(list, zip(*testset))
    test_bg = dgl.batch(test_X)
    test_Y = torch.tensor(test_Y).float().view(-1, 1)
    probs_Y = torch.softmax(model(test_bg), 1)
    sampled_Y = torch.multinomial(probs_Y, 1)
    argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
    test_accuracy1 = (test_Y == sampled_Y.float()).sum().item() / len(test_Y)
    test_accuracy2 = (test_Y == argmax_Y.float()).sum().item() / len(test_Y)
    print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
        (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
    print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
        (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))
    accuracyfile = open('malwareresult/malware_accuracyfile.csv', 'w', encoding='utf-8', newline="")
    accuracyfile_writer = csv.writer(accuracyfile)
    accuracyfile_writer.writerow(["num", "accuracy"])
    accuracyfile_writer.writerow(["sampled", test_accuracy1])
    accuracyfile_writer.writerow(["argmax", test_accuracy2])
    cm = zeros((trainset.num_classes, trainset.num_classes), dtype=float)
    if test_accuracy1 >= test_accuracy2:
        sampled_Y_done = sampled_Y.float()
        for i in range(len(test_Y)):
            cm[int(test_Y[i, 0]), int(sampled_Y_done[i, 0])] += 1
        sum = [0] * 5
        index = [0] * 5
        pred_sum = [0] * 5
        for i in range(len(test_Y)):
            sum[int(test_Y[i, 0])] = sum[int(test_Y[i, 0])] + 1
            pred_sum[int(sampled_Y_done[i, 0])] = pred_sum[int(sampled_Y_done[i, 0])] + 1
            if test_Y[i] == sampled_Y.float()[i]:
                index[int(test_Y[i, 0])] = index[int(test_Y[i, 0])] + 1
        j = 0
        f = open('malwareresult/malware_recision_and_recall.csv', 'w', encoding='utf-8', newline="")
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
            app_name = ID_TO_Malware.get(i)
            csv_writer.writerow([app_name, recall, precision])
            j = j + 1
    else:
        argmax_Y_done = argmax_Y.float()
        for i in range(len(test_Y)):
            cm[int(test_Y[i, 0]), int(argmax_Y_done[i, 0])] += 1
        sum = [0] * 5
        index = [0] * 5
        pred_sum = [0] * 5
        for i in range(len(test_Y)):
            sum[int(test_Y[i, 0])] = sum[int(test_Y[i, 0])] + 1
            pred_sum[int(argmax_Y_done[i, 0])] = pred_sum[int(argmax_Y_done[i, 0])] + 1
            if test_Y[i] == argmax_Y.float()[i]:
                index[int(test_Y[i, 0])] = index[int(test_Y[i, 0])] + 1
        j = 0
        print("Accuracy of argmax predictions")
        f = open('malwareresult/malware_recision_and_recall.csv', 'w', encoding='utf-8', newline="")
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
            app_name = ID_TO_Malware.get(i)
            csv_writer.writerow([app_name, recall, precision])
            j = j + 1
    app_labels = []
    for i in sorted(list(ID_TO_Malware.keys())):
        app_labels.append(ID_TO_Malware[i])
    fig = plot_confusion_matrix(cm, app_labels,5)
    fig.savefig('malwareresult/malware_heatmap.png')


@click.command()
@click.option('-k', '--kind', help='object to be classificated', required=True)
def main(kind):
    if kind == 'app':
        train_app_graphs("Graphs/app/trainset.bin", "Graphs/app/testset.bin", "Graphs/app/validset.bin")
    elif kind == 'malware':
        train_malware_graphs("Graphs/malware/trainset.bin", "Graphs/malware/validset.bin", "Graphs/malware/testset.bin")
    else:
        train_entropy_graphs("Graphs/Entropy/trainset.bin", "Graphs/Entropy/validset.bin", "Graphs/Entropy/testset.bin")


if __name__ == '__main__':
    main()
