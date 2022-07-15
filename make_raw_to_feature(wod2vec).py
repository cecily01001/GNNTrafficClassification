import csv
import shutil
import click
import dill
import torch
from pathlib import Path
import numpy as np
from torchtext.legacy import data
from scapy.all import *
from scapy.error import Scapy_Exception
from scapy.layers.inet import IP, UDP,TCP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scipy import sparse
from utils.utilsformalware import should_omit_packet, read_pcap, PREFIX_TO_Malware_ID
from utils.utilsforapps import PREFIX_TO_APP_ID
from utils.utilsforentropy import PREFIX_TO_ENTROPY_ID, ID_TO_ENTROPY
from dgl.data.utils import save_graphs
from CNNModel.cnnmodel import TextCNN
import argparse as arg
from spacy.lang.en import English
from BertModel.Utils.Classifier_utils import InputExample, convert_examples_to_features, convert_features_to_tensors

spacy_en = English()
def tokenizer(text): # create a tokenizer function
    # regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
    # text = regex.sub(' ', text)
    # return [word for word in jieba.cut(text) if word.strip()]
    return [tok.text for tok in spacy_en.tokenizer(text)]
with open('CNNModel/Dataset/textSplitDataset/train_examples', 'rb')as f:
    train_examples = dill.load(f)
origin_text = data.Field(sequential=True, lower=True, tokenize=tokenizer)
origin_label = data.Field(sequential=False)
origin_train = data.Dataset(examples=train_examples, fields=[('text', origin_text),('label', origin_label)])
origin_text.build_vocab(origin_train)
origin_label.build_vocab(origin_train)
print(origin_label.vocab.stoi)

# def get_dataset(text_field, label_field, test=False):
#     fields = [('text', text_field), ('label', label_field)]
#     examples = []
#
#     if test:
#         with open(corpus_path, "r", encoding="utf-8") as f:
#             for line in tqdm.tqdm(f):
#                 text, label=get_text_and_label(line)
#                 examples.append(data.Example.fromlist([ text, None], fields))
#     else:
#         with open(corpus_path, "r", encoding="utf-8") as f:
#             for line in tqdm.tqdm(f):
#                 text, label = get_text_and_label(line)
#                 examples.append(data.Example.fromlist([text, label], fields))
#     return examples, fields

def load_data(train,text,label):

    # text = data.Field(sequential=True, lower=True, tokenize=tokenizer)
    # label = data.Field(sequential=False)
    # #
    # text.tokenize = tokenizer
    # train = data.TabularDataset(
    #         path='CNNModel/singletrain.csv',
    #         # skip_header=True,
    #         # train='sampletrain.tsv',
    #         format='csv',
    #         fields=[('label', label), ('text', text)],
    #     )
    # for i in train.Tweet:
    #     print(i)
    # for i in train.Affect_Dimension:
    #     print(i)
    # f1 = open("CNNModel/data/pcap.vector", 'rb')
    # f2 = open("CNNModel/data/label.vector", 'rb')
    # pcap = dill.load(f1)
    # pcap_label = dill.load(f2)
    # text=origin_text # 此处改为你自己的词向量
    # label=origin_label
    text.vocab=origin_text.vocab
    label.vocab=origin_label.vocab
    # label.vocab=pcap_label
    # print(text.vocab.stoi)
    # for i in label.vocab:
    #     print(i)
    # args.embedding_dim = text.vocab.vectors.size()[-1]
    # args.embedding_dim = 128
    # args.vectors = text.vocab.vectors
    #
    # else:
    #     text.build_vocab(train, val)
    #     label.build_vocab(train, val)
    #     with open("data/pcap.vector", 'wb')as f:
    #         dill.dump(text.vocab, f)
    #     with open("data/label.vector", 'wb')as f:
    #         dill.dump(label.vocab, f)

    # print(text.vocab.itos)
    train_iter = data.Iterator(
            dataset=train,
            sort=False,
            batch_size=len(train), # 训练集设置batch_size,验证集整个集合用于测试
            train=False,
            # device=-1,
            device=torch.device('cuda')
    )
    # for batch in train_iter:
    #     print('输出text嵌入结果：')
    #     print(batch.text[1])
    #     print(len(batch.text[0]))
    #     print(batch.text.shape)

    # print(len(text.vocab))
    return train_iter

def load_model(args):

    model = TextCNN(args)
    model.load_state_dict(torch.load('CNNModel/model_dir/best_steps_1100.pt'))
    model.eval()
    return model
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
    # arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length] / 255
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
    arr = sparse.csr_matrix(arr)
    return arr


def tokenizer(text):  # create a tokenizer function
    # regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
    # text = regex.sub(' ', text)
    # return [word for word in jieba.cut(text) if word.strip()]
    return [tok.text for tok in spacy_en.tokenizer(text)]

def load_args():
    parser = arg.ArgumentParser(description='TextCNN text classifier')
    parser.add_argument('-lr', type=float, default=0.001, help='学习率')
    parser.add_argument('-k', type=str, default='app', help='种类')
    parser.add_argument('-batch-size', type=int, default=128)
    parser.add_argument('-epoch', type=int, default=20)
    parser.add_argument('-filter-num', type=int, default=100, help='卷积核的个数')
    parser.add_argument('-filter-sizes', type=str, default='3,4,5', help='不同卷积核大小')
    parser.add_argument('-embedding-dim', type=int, default=128, help='词向量的维度')
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-label-num', type=int, default=41, help='标签个数')
    parser.add_argument('-static', type=bool, default=False, help='是否使用预训练词向量')
    parser.add_argument('-fine-tune', type=bool, default=True, help='预训练词向量是否要微调')
    parser.add_argument('-cuda', type=bool, default=True)
    parser.add_argument('-log-interval', type=int, default=1, help='经过多少iteration记录一次训练状态')
    parser.add_argument('-test-interval', type=int, default=100, help='经过多少iteration对验证集进行测试')
    parser.add_argument('-early-stopping', type=int, default=1000, help='早停时迭代的次数')
    parser.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
    parser.add_argument('-save-dir', type=str, default='model_dir', help='存储训练模型位置')

    args = parser.parse_args()
    return args

args=load_args()
args.vocab_size = len(origin_text.vocab)
args.label_num = len(origin_label.vocab)
model = load_model(args)
if args.cuda: model.cuda()
def transform_packet(packet):
    # if should_omit_packet(packet):
    #     return None

    packet = remove_ether_header(packet)
    packet = pad_udp(packet)
    packet = mask_ip(packet)
    # wrpcap("/home/pcl/PangBo/pro/GNNTrafficClassification/Dataset/ProcessedData/" + name, packet, append=True)
    arr = packet_to_sparse_array(packet)
    return arr
# def transform_pcap_malware(path, listofnodenum_test, listofkind_test, feature_matrix_test):
#     # output_path = 'processdata/' + path.name
#     # if Path(output_path + '_SUCCESS').exists():
#     #     return
#     # with Path(output_path + '_SUCCESS').open('w') as f:
#     #     f.write('')
#     # print(path,"Done")
#     # print(path)
#     # try:
#     #     read_pcap(path)
#     # except Scapy_Exception:
#     #     os.remove(path)
#     #     return
#     rows = []
#     j = 0
#     for i, packet in enumerate(read_pcap(path)):
#         arr = transform_packet(packet)
#         if arr is not None:
#             prefix = path.name.split('.')[0]
#             app_label = PREFIX_TO_Malware_ID.get(prefix)
#             j = j + 1
#             rows.append(arr.todense().tolist()[0])
#     if j > 1:
#         listofnodenum_test.append(j)
#         listofkind_test.append(app_label)
#         feature_matrix_test.append(rows)
#         print(path, 'Done')
#
#
# def transform_pcap_entropy(path, listofnodenum, listofkind, feature_matrix):
#     rows = []
#     j = 0
#     for i, packet in enumerate(read_pcap(path)):
#         arr = transform_packet(packet)
#         if arr is not None:
#             prefix = path.name.split('.')[0]
#             app_label = PREFIX_TO_ENTROPY_ID.get(prefix)
#             j = j + 1
#             rows.append(arr.todense().tolist()[0])
#     if j > 1:
#         listofnodenum.append(j)
#         listofkind.append(app_label)
#         feature_matrix.append(rows)
#         print(path, 'Done')
def get_text_feature(train,text,label):
    text_list=[]
    train_iter = load_data(train,text,label)
    for batch in train_iter:
        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        # print(torch.max(logits, 1)[1])
        # print(target)
        # print('feature num:'+str(len(feature)))
        # print('logits num:' + str(len(logits)))
        for i in logits:
            text_list.append(i.detach().cpu().numpy().tolist())
    # print(text_list)
    return text_list
def transform_pcap(path,label_dic,kind,datakind):
    rows=[]
    direcs=[]
    # 数据集定义
    examples = []
    text = data.Field(sequential=True, lower=True, tokenize=tokenizer)
    label = data.Field(sequential=False)
    fields = [('text', text), ('label', label)]
    prefix = path.name.split('.')[0]
    first_w=prefix[0:1]
    second=prefix[1:2]
    flag=True
    # if first_w=='e':
    #     if second>='s':
    #         flag=True
    # elif first_w>'e':
    #         flag=True
    app_label = label_dic.get(prefix)
    # 查看是否能够成功读取原始pcap文件
    try:
        packets_list=read_pcap(path)
    except:
        return

    save_dir=os.path.join('Dataset/Processed_data/'+kind+'/',datakind)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    graph_path = os.path.join(save_dir+'/'+str(len(packets_list))+ '_'+ str(app_label) + '_'+path.name+'.csv')
    graph_file_temp = open(graph_path, 'w', encoding='utf-8', newline="")
    graph_file = csv.writer(graph_file_temp)
    # graph_file.writerow(['node_feature', 'text_feature','direc'])
    p1=int(path.name.split('.')[2].split('_')[2])
    text_path=str(path).split('Splite_Session')[0]+'Text'+str(path).split('Splite_Session')[1]+'.txt'
    try:
        f = open(text_path, "r", encoding="utf-8")
        l = f.read()
        text_packets = l.split(' Encapsulation type:')
        text_packets = text_packets[1:]
        for i in range(len(text_packets)):
            text_packets[i] = text_packets[i][0:10000]
    except:
        return
    for i, packet in enumerate(packets_list):
        direc=-1
        if TCP in packet:
            sport = packet[TCP].sport
            if sport==p1:
                direc=0
            else: direc=1
        direcs.append(direc)
        arr = transform_packet(packet)
        # print(arr)
        # if arr is not None:
        arr_feature=arr.todense().tolist()[0]
        rows.append(arr_feature)
        packet_text=text_packets[i]
        examples.append(data.Example.fromlist([packet_text, app_label], fields))

    train = data.Dataset(examples, fields)
    texts=get_text_feature(train,text,label)
    # print('rows length:'+str(len(rows)))
    if len(texts)==len(rows):
        if len(texts) >= 1:
            for i in range(len(texts)):
                graph_file.writerow([','.join(map(str,rows[i])), ','.join(map(str,texts[i])),direcs[i]])
        if len(texts) != len(packets_list):
            graph_path_fix = os.path.join(
                save_dir + '/' + str(len(texts)) + '_' + str(app_label) + '_' + path.name + '.csv')
            os.rename(graph_path, graph_path_fix)
        print(path, 'Done')
    else:
        print(path, 'Error')


# class graphdataset(object):
#     def __init__(self, num_of_graphnode, listofkind, feature_matrix, num_classes,text_list):
#         super(graphdataset, self).__init__()
#         graphlist = []
#         k = 0
#         for i in num_of_graphnode:
#             list_point = []
#             list_point_2 = []
#             for j in range(i - 1):
#                 list_point.append(j)
#                 list_point_2.append(j + 1)
#             for j in range(i - 1):
#                 list_point.append(j + 1)
#                 list_point_2.append(j)
#             leng = int(i)
#             # print('len:'+str(leng))
#             fix_list_point = list_point[0:leng - 1]
#             fix_list_point2 = list_point_2[0:leng - 1]
#             g = dgl.graph((list_point, list_point_2))
#             g = dgl.add_self_loop(g)
#             # print(i)
#             # print(feature_matrix[k])
#             # fix_feature = feature_matrix[k]
#             # fix_feature = fix_feature[0:leng]
#             g.ndata['h'] = torch.tensor(feature_matrix[k])
#             g.ndata['x']=torch.tensor(text_list[k])
#             # print(g)
#             k = k + 1
#             graphlist.append(g)
#         # print(list_point)
#
#         self.graphs = graphlist
#         self.num_classes = num_classes
#         self.labels = listofkind
#         # self.feature_matrix=feature_matrix
#
#     def __len__(self):
#         """Return the number of graphs in the dataset."""
#         return len(self.graphs)
#
#     def __getitem__(self, idx):
#         """Get the i^th sample.
#
#         Paramters
#         ---------
#         idx : int
#             The sample index.
#
#         Returns
#         -------
#         (dgl.DGLGraph, int)
#             The graph and its label.
#         """
#         return self.graphs[idx], self.labels[idx]


def make_feature(data_dir_path, valid_data_dir_path, testdata_dir_path,kind):
    testdata_dir_path = Path(testdata_dir_path)
    valid_data_dir_path = Path(valid_data_dir_path)
    data_dir_path = Path(data_dir_path)
    if kind=='app':
        label_dic=PREFIX_TO_APP_ID
    elif kind == 'malware':
        label_dic = PREFIX_TO_Malware_ID
    else:
        label_dic = PREFIX_TO_ENTROPY_ID
    # shutil.rmtree('Dataset/Processed_data/'+kind)
    os.makedirs('Dataset/Processed_data/'+kind)
    for pcap_path in sorted(data_dir_path.iterdir()):
        transform_pcap(pcap_path, label_dic,kind,'train')
    for pcap_path in sorted(valid_data_dir_path.iterdir()):
        transform_pcap(pcap_path, label_dic,kind,'valid')
    for pcap_path in sorted(testdata_dir_path.iterdir()):
        transform_pcap(pcap_path, label_dic,kind,'test')

@click.command()
@click.option('-k', '--kind', help='object to be classificated', required=True)
def main(kind):
    if kind == 'app':
        make_feature("Dataset/Splite_Session/app/Train", "Dataset/Splite_Session/app/valid",
                       "Dataset/Splite_Session/app/Test",kind)
        # make_feature("/home/user1/PangBo/GNNTrafficClassification/test",
        #              "/home/user1/PangBo/GNNTrafficClassification/test",
        #              "/home/user1/PangBo/GNNTrafficClassification/test",kind)
    elif kind == 'malware':
        make_feature("Dataset/Splite_Session/Malware/trainset", "Dataset/Splite_Session/Malware/validset",
                           "Dataset/Splite_Session/Malware/testset",kind)
    else:
        make_feature("Dataset/Splite_Session/ENTROPY2/train", "Dataset/Splite_Session/ENTROPY2/valid",
                           "Dataset/Splite_Session/ENTROPY2/test",kind)

if __name__ == '__main__':
    main()
